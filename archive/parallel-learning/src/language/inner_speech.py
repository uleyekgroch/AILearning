"""
内部言语模块 — 语言作为内部规划工具（Vygotsky 内部言语）

核心理念：在产出话语之前，Agent 先构建场景的「内部描述」。
这种内部语言使用提升了描述效率，尤其对复杂场景效果显著。

纯 PyTorch 实现，零 numpy 依赖。
"""

from typing import Dict, List, Optional, Tuple

import torch

from src.core.device import get_device


class InnerSpeechModule:
    """
    内部言语模块（Vygotsky 内部言语）

    职责：
    1. 将场景特征编码为内部表示张量
    2. 识别目标物体的区分性特征
    3. 在产出话语前规划最优描述策略
    4. 追踪内部规划对描述准确度的提升
    """

    def __init__(self, obs_dim: int = 40, device: str = 'auto'):
        self.obs_dim = obs_dim
        self.device = get_device(device)

        # 场景编码器权重 — 将特征映射到内部表示空间
        self._encoder = torch.randn(obs_dim, obs_dim, device=self.device) * 0.1
        # 注意力权重 — 决定哪些特征维度最重要
        self._attention_weights = torch.ones(obs_dim, device=self.device)

        # 内部规划历史：记录规划前后准确度对比
        self._planning_log: List[Dict] = []
        # 统计：有/无内部规划时的成功次数
        self._with_planning: Dict[str, int] = {'success': 0, 'total': 0}
        self._without_planning: Dict[str, int] = {'success': 0, 'total': 0}

        # 已学到的特征-符号映射（用于快速检索区分性特征）
        self._feature_symbol_map: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # 核心方法：内部规划
    # ------------------------------------------------------------------

    def plan_description(self, scene_features: List[Dict],
                         target_idx: int) -> List[str]:
        """
        在产出话语前构建内部描述。

        步骤：
        1. 编码整个场景为内部表示
        2. 识别目标物体的区分性特征
        3. 选择最优符号序列

        Args:
            scene_features: 场景中每个物体的特征字典列表
            target_idx: 目标物体的索引

        Returns:
            规划后的符号列表（描述策略）
        """
        if not scene_features or target_idx >= len(scene_features):
            return []

        # 1. 编码场景
        scene_repr = self.encode_scene(scene_features)

        # 2. 提取目标特征并找到区分性特征
        target = scene_features[target_idx]
        distinctive = self._find_distinctive(target, scene_features, target_idx)

        # 3. 选择最佳符号
        planned_symbols = self._select_symbols(distinctive, target)

        # 记录到规划日志
        self._planning_log.append({
            'target_idx': target_idx,
            'scene_size': len(scene_features),
            'distinctive_features': distinctive,
            'planned_symbols': planned_symbols,
        })

        return planned_symbols

    def encode_scene(self, scene_features: List[Dict]) -> torch.Tensor:
        """
        将场景特征转换为内部表示张量。

        把每个物体的特征字典映射为一个固定维度的向量，
        然后通过编码器矩阵投射到内部表示空间。

        Args:
            scene_features: 物体特征字典列表

        Returns:
            shape (num_objects, obs_dim) 的内部表示张量
        """
        n = len(scene_features)
        if n == 0:
            return torch.zeros(0, self.obs_dim, device=self.device)

        # 将特征字典转为原始向量（one-hot 式哈希编码）
        raw = torch.zeros(n, self.obs_dim, device=self.device)
        for i, obj in enumerate(scene_features):
            for key, val in obj.items():
                # 简单确定性哈希 → 分配到 obs_dim 维空间
                h = hash(f'{key}:{val}') % self.obs_dim
                raw[i, h] += 1.0

        # 通过编码器投射
        encoded = torch.matmul(raw, self._encoder)
        return encoded

    # ------------------------------------------------------------------
    # 区分性特征识别
    # ------------------------------------------------------------------

    def _find_distinctive(self, target: Dict,
                          scene: List[Dict],
                          target_idx: int) -> Dict[str, str]:
        """
        找到目标物体相对于场景中其他物体的区分性特征。

        只返回能唯一区分目标的特征键值对。
        """
        distinctive: Dict[str, str] = {}

        for key, val in target.items():
            # 统计场景中有多少物体共享该特征值
            count = sum(
                1 for i, obj in enumerate(scene)
                if i != target_idx and obj.get(key) == val
            )
            # 如果只有目标拥有该特征值，则它区分度最高
            if count == 0:
                distinctive[key] = val

        # 如果没有完全唯一的特征，选取共享最少的
        if not distinctive:
            candidates = []
            for key, val in target.items():
                count = sum(
                    1 for i, obj in enumerate(scene)
                    if i != target_idx and obj.get(key) == val
                )
                candidates.append((key, val, count))
            candidates.sort(key=lambda x: x[2])
            # 取共享最少的若干特征，直到能唯一标识
            for key, val, _ in candidates:
                distinctive[key] = val
                # 检查组合是否已能唯一标识
                if self._is_unique_with(distinctive, scene, target_idx):
                    break

        return distinctive

    def _is_unique_with(self, features: Dict[str, str],
                        scene: List[Dict], target_idx: int) -> bool:
        """检查给定特征组合是否能在场景中唯一标识目标"""
        count = 0
        for i, obj in enumerate(scene):
            if all(obj.get(k) == v for k, v in features.items()):
                count += 1
        return count == 1

    def _select_symbols(self, distinctive: Dict[str, str],
                        target: Dict) -> List[str]:
        """
        根据区分性特征选择最佳符号序列。

        利用已学到的特征-符号映射，优先选择区分度高的符号。
        """
        symbols = []
        # 区分性特征对应的符号优先
        for key, val in distinctive.items():
            sym = self._feature_symbol_map.get(f'{key}:{val}', [val])
            symbols.extend(sym if isinstance(sym, list) else [sym])

        # 补充目标的其他特征（顺序靠后）
        for key, val in target.items():
            if key not in distinctive:
                sym = self._feature_symbol_map.get(f'{key}:{val}', [val])
                symbols.extend(sym if isinstance(sym, list) else [sym])

        return symbols

    # ------------------------------------------------------------------
    # 效果度量
    # ------------------------------------------------------------------

    def record_outcome(self, success: bool, used_planning: bool) -> None:
        """记录一次描述结果，用于统计规划收益"""
        if used_planning:
            self._with_planning['total'] += 1
            if success:
                self._with_planning['success'] += 1
        else:
            self._without_planning['total'] += 1
            if success:
                self._without_planning['success'] += 1

    def get_planning_benefit(self) -> float:
        """
        内部规划带来的准确度提升。

        Returns:
            规划组成功率 - 无规划组成功率。正值表示规划有帮助。
        """
        rate_with = (
            self._with_planning['success'] / max(self._with_planning['total'], 1)
        )
        rate_without = (
            self._without_planning['success'] / max(self._without_planning['total'], 1)
        )
        return rate_with - rate_without

    # ------------------------------------------------------------------
    # 学习特征-符号映射
    # ------------------------------------------------------------------

    def learn_mapping(self, feature_key: str, symbol: str) -> None:
        """学习特征键值与符号的映射关系"""
        key = feature_key
        if key not in self._feature_symbol_map:
            self._feature_symbol_map[key] = []
        if symbol not in self._feature_symbol_map[key]:
            self._feature_symbol_map[key].append(symbol)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        """序列化模块状态"""
        return {
            'encoder': self._encoder.detach().cpu(),
            'attention_weights': self._attention_weights.detach().cpu(),
            'with_planning': dict(self._with_planning),
            'without_planning': dict(self._without_planning),
            'feature_symbol_map': dict(self._feature_symbol_map),
            'planning_log': list(self._planning_log[-100:]),
        }

    def load_state(self, state: dict) -> None:
        """从字典恢复模块状态"""
        if 'encoder' in state:
            self._encoder = state['encoder'].to(self.device)
        if 'attention_weights' in state:
            self._attention_weights = state['attention_weights'].to(self.device)
        self._with_planning = state.get('with_planning', {'success': 0, 'total': 0})
        self._without_planning = state.get('without_planning', {'success': 0, 'total': 0})
        self._feature_symbol_map = state.get('feature_symbol_map', {})
        self._planning_log = state.get('planning_log', [])
