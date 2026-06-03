"""
语言引导注意力模块 — 词汇驱动的自上而下注意调制

核心理念：已掌握的词汇引导注意力到相关的感知通道。
实现 Waxman & Gelman (2009) 的标签-注意耦合效应。

例如：知道 "red" → 注意力偏向视觉颜色通道；
      知道 "loud" → 注意力偏向听觉通道。

纯 PyTorch 实现，零 numpy 依赖。
"""

from typing import Dict, List, Optional, Set

import torch

from src.core.device import get_device


# =====================================================================
# 模态-符号关联规则
# =====================================================================

# 预设的符号-模态映射（可通过 learn_symbol_modality 扩展）
_SYMBOL_MODALITY_HINTS: Dict[str, str] = {
    # 颜色 → 视觉
    'red': 'visual', 'blue': 'visual', 'green': 'visual',
    'yellow': 'visual', 'white': 'visual', 'black': 'visual',
    # 形状 → 视觉
    'circle': 'visual', 'square': 'visual', 'triangle': 'visual',
    'round': 'visual', 'flat': 'visual',
    # 声音 → 听觉
    'loud': 'audio', 'quiet': 'audio', 'sharp': 'audio',
    'soft_sound': 'audio', 'humming': 'audio',
    # 触感 → 触觉
    'rough': 'tactile', 'smooth': 'tactile', 'hard': 'tactile',
    'soft': 'tactile', 'sharp_edge': 'tactile',
    # 大小 → 视觉（空间维度）
    'big': 'visual', 'small': 'visual',
}


# =====================================================================
# AttentionModulator — 语言引导注意力调制器
# =====================================================================

class AttentionModulator:
    """
    自上而下注意力调制器。

    基于已知词汇，调整感知通道权重：
    - 颜色相关符号 → 增强视觉通道
    - 声音相关符号 → 增强听觉通道
    - 位置相关符号 → 增强位置通道

    实现了标签-注意耦合（Waxman & Gelman, 2009）：
    学习标签后，注意力更快聚焦到标签相关的特征维度。
    """

    def __init__(self, visual_dim: int = 256,
                 audio_dim: int = 13,
                 position_dim: int = 2,
                 device: str = 'auto'):
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.position_dim = position_dim
        self.device = get_device(device)

        # 基线通道权重（未经调制时的默认值）
        self._base_weights = torch.tensor(
            [0.6, 0.2, 0.2],  # [visual, audio, position]
            dtype=torch.float32,
            device=self.device,
        )

        # 当前调制后的权重
        self._modulated_weights = self._base_weights.clone()

        # 符号 → 模态映射（可学习）
        self._symbol_modality: Dict[str, str] = dict(_SYMBOL_MODALITY_HINTS)

        # 已知符号集合
        self._known_symbols: Set[str] = set()

        # 调制历史（用于统计）
        self._modulation_count: int = 0
        self._boost_history: Dict[str, int] = {
            'visual': 0, 'audio': 0, 'position': 0,
        }

        # 调制强度系数 — 控制标签对注意力的增强幅度
        self._boost_strength: float = 0.3

    # ------------------------------------------------------------------
    # 核心方法：注意力调制
    # ------------------------------------------------------------------

    def modulate(self, observation: Dict[str, torch.Tensor],
                 known_symbols: List[str]) -> Dict[str, torch.Tensor]:
        """
        根据已知符号调整感知通道权重。

        如果符号与颜色相关 → 增强视觉通道。
        如果符号与声音相关 → 增强听觉通道。

        Args:
            observation: 观测字典，包含:
                - 'visual': (visual_dim,) 视觉特征
                - 'audio': (audio_dim,) 听觉特征
                - 'position': (position_dim,) 位置特征
            known_symbols: 当前已知（已掌握）的符号列表

        Returns:
            调制后的观测字典（通道值按权重加权）
        """
        self._known_symbols = set(known_symbols)

        # 计算每个通道的调制权重
        visual_boost = 0.0
        audio_boost = 0.0
        position_boost = 0.0

        for sym in known_symbols:
            modality = self._symbol_modality.get(sym)
            if modality == 'visual':
                visual_boost += self._boost_strength
            elif modality == 'audio':
                audio_boost += self._boost_strength
            elif modality == 'position':
                position_boost += self._boost_strength
            elif modality == 'tactile':
                # 触觉映射到视觉通道（当前架构中触觉无独立通道）
                visual_boost += self._boost_strength * 0.5

        # 归一化增强量，防止过度放大
        total_boost = visual_boost + audio_boost + position_boost
        if total_boost > 0:
            scale = min(total_boost, 1.0) / max(total_boost, 1e-6)
            visual_boost *= scale
            audio_boost *= scale
            position_boost *= scale

        # 更新调制权重
        self._modulated_weights = self._base_weights.clone()
        self._modulated_weights[0] += visual_boost
        self._modulated_weights[1] += audio_boost
        self._modulated_weights[2] += position_boost

        # softmax 归一化
        self._modulated_weights = torch.softmax(self._modulated_weights, dim=0)

        # 记录增强历史
        self._modulation_count += 1
        if visual_boost > 0:
            self._boost_history['visual'] += 1
        if audio_boost > 0:
            self._boost_history['audio'] += 1
        if position_boost > 0:
            self._boost_history['position'] += 1

        # 应用调制权重到观测
        result: Dict[str, torch.Tensor] = {}
        w = self._modulated_weights

        if 'visual' in observation:
            vis = observation['visual'].to(self.device).float()
            if vis.dim() == 1:
                vis = vis.unsqueeze(0)
            result['visual'] = vis * w[0]
        if 'audio' in observation:
            aud = observation['audio'].to(self.device).float()
            if aud.dim() == 1:
                aud = aud.unsqueeze(0)
            result['audio'] = aud * w[1]
        if 'position' in observation:
            pos = observation['position'].to(self.device).float()
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            result['position'] = pos * w[2]

        return result

    # ------------------------------------------------------------------
    # 权重查询
    # ------------------------------------------------------------------

    def get_weights(self) -> Dict[str, float]:
        """
        返回当前的通道权重。

        Returns:
            {'visual': float, 'audio': float, 'position': float}
        """
        w = self._modulated_weights
        return {
            'visual': w[0].item(),
            'audio': w[1].item(),
            'position': w[2].item(),
        }

    def get_boost_distribution(self) -> Dict[str, float]:
        """
        返回各通道被增强的比例。

        Returns:
            {'visual': float, 'audio': float, 'position': float}
        """
        total = max(self._modulation_count, 1)
        return {
            ch: count / total
            for ch, count in self._boost_history.items()
        }

    # ------------------------------------------------------------------
    # 学习接口
    # ------------------------------------------------------------------

    def learn_symbol_modality(self, symbol: str, modality: str) -> None:
        """
        学习符号与模态的关联。

        当 Agent 通过经验发现某个符号属于特定模态时调用。

        Args:
            symbol: 符号字符串
            modality: 模态名称（'visual', 'audio', 'position', 'tactile'）
        """
        self._symbol_modality[symbol] = modality

    def add_known_symbol(self, symbol: str) -> None:
        """添加已知符号"""
        self._known_symbols.add(symbol)

    def get_known_symbols(self) -> List[str]:
        """返回所有已知符号"""
        return sorted(self._known_symbols)

    def get_symbol_modality_map(self) -> Dict[str, str]:
        """返回完整的符号-模态映射"""
        return dict(self._symbol_modality)

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def reset_modulation(self) -> None:
        """重置调制权重回基线"""
        self._modulated_weights = self._base_weights.clone()

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        """序列化模块状态"""
        return {
            'base_weights': self._base_weights.detach().cpu(),
            'modulated_weights': self._modulated_weights.detach().cpu(),
            'symbol_modality': dict(self._symbol_modality),
            'known_symbols': list(self._known_symbols),
            'modulation_count': self._modulation_count,
            'boost_history': dict(self._boost_history),
            'boost_strength': self._boost_strength,
        }

    def load_state(self, state: dict) -> None:
        """从字典恢复模块状态"""
        if 'base_weights' in state:
            self._base_weights = state['base_weights'].to(self.device)
        if 'modulated_weights' in state:
            self._modulated_weights = state['modulated_weights'].to(self.device)
        self._symbol_modality = state.get('symbol_modality', dict(_SYMBOL_MODALITY_HINTS))
        self._known_symbols = set(state.get('known_symbols', []))
        self._modulation_count = state.get('modulation_count', 0)
        self._boost_history = state.get('boost_history', {
            'visual': 0, 'audio': 0, 'position': 0,
        })
        self._boost_strength = state.get('boost_strength', 0.3)
