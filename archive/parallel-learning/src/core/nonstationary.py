"""非平稳环境适应 — 语言随环境体制变化而调整

当环境发生体制变化 (regime shift)（如四季更替）时，
词汇必须适应: 遗忘过时词、学习新词、重新激活休眠知识。

核心概念:
  EnvironmentRegime — 环境体制定义（特征分布 + 持续步数）
  NonstationaryAdapter — 语言适应控制器

机制:
  1. forget_obsolete   — 按衰减率降低过时词汇的使用频率
  2. reactivate_dormant — 当旧体制重新出现时，激活休眠词汇
  3. 适应度评分 — 追踪当前词汇与环境的匹配程度

所有数值计算使用 torch.Tensor，零 numpy 依赖。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from src.core.device import get_device


@dataclass
class EnvironmentRegime:
    """环境体制定义

    Attributes:
        name: 体制名称（如 'spring', 'summer'）
        feature_distribution: 特征概率分布 {feature: probability}
        duration_steps: 该体制持续的步数
    """
    name: str
    feature_distribution: Dict[str, float]
    duration_steps: int = 200


# 预定义体制序列
REGIME_SEQUENCES = {
    'seasons': [
        EnvironmentRegime('spring', {'green': 0.3, 'warm': 0.2, 'wet': 0.3}, 200),
        EnvironmentRegime('summer', {'bright': 0.3, 'hot': 0.4, 'dry': 0.2}, 200),
        EnvironmentRegime('autumn', {'orange': 0.3, 'cool': 0.2, 'windy': 0.2}, 200),
        EnvironmentRegime('winter', {'white': 0.3, 'cold': 0.4, 'dry': 0.1}, 200),
    ],
    'day_night': [
        EnvironmentRegime('day', {'bright': 0.4, 'warm': 0.3, 'active': 0.3}, 100),
        EnvironmentRegime('dusk', {'orange': 0.3, 'cool': 0.2, 'calm': 0.2}, 50),
        EnvironmentRegime('night', {'dark': 0.4, 'cold': 0.3, 'quiet': 0.3}, 100),
        EnvironmentRegime('dawn', {'pink': 0.3, 'cool': 0.2, 'fresh': 0.2}, 50),
    ],
    'terrain': [
        EnvironmentRegime('plains', {'flat': 0.3, 'green': 0.3, 'open': 0.2}, 150),
        EnvironmentRegime('forest', {'dense': 0.3, 'green': 0.2, 'shaded': 0.3}, 150),
        EnvironmentRegime('mountain', {'steep': 0.3, 'rocky': 0.3, 'cold': 0.2}, 150),
        EnvironmentRegime('coast', {'wet': 0.3, 'sandy': 0.2, 'windy': 0.3}, 150),
    ],
}


class NonstationaryAdapter:
    """非平稳环境语言适应器

    追踪当前体制，在体制切换时触发词汇适应:
    - 遗忘: 过时词汇的使用频率按指数衰减
    - 激活: 休眠词汇在旧体制回归时被重新激活
    - 学习: 新体制需要新词汇来描述新特征

    Args:
        regime_sequence: 体制序列名称，需在 REGIME_SEQUENCES 中注册
        decay_rate: 遗忘衰减率 (0, 1)，越大遗忘越快，默认 0.1
        reactivation_threshold: 休眠词汇重新激活的频率阈值，默认 0.1
    """

    def __init__(self, regime_sequence: str = 'seasons',
                 decay_rate: float = 0.1,
                 reactivation_threshold: float = 0.1):
        self._device = get_device()
        self.decay_rate = decay_rate
        self.reactivation_threshold = reactivation_threshold

        # 加载体制序列
        self._regimes = REGIME_SEQUENCES.get(
            regime_sequence, REGIME_SEQUENCES['seasons']
        )
        self._regime_index = 0
        self._step_in_regime = 0

        # 历史体制记录（用于追踪循环模式）
        self._regime_history: List[str] = []

        # 词汇休眠记录: {word: (regime_name, last_frequency)}
        self._dormant_vocabulary: Dict[str, tuple] = {}

        # 适应度追踪
        self._adaptation_scores: List[float] = []

    def step(self) -> Optional[str]:
        """推进一个时间步

        Returns:
            如果体制发生变化，返回新体制名称；否则返回 None
        """
        self._step_in_regime += 1

        current_regime = self._regimes[self._regime_index]

        if self._step_in_regime >= current_regime.duration_steps:
            # 切换到下一个体制（循环）
            self._regime_history.append(current_regime.name)
            self._regime_index = (self._regime_index + 1) % len(self._regimes)
            self._step_in_regime = 0

            new_regime = self._regimes[self._regime_index]
            return new_regime.name

        return None

    def get_current_regime(self) -> EnvironmentRegime:
        """获取当前环境体制"""
        return self._regimes[self._regime_index]

    def get_adaptation_score(self) -> float:
        """计算当前词汇对环境的适应度

        适应度 = 当前词汇特征与体制特征分布的匹配度。
        高分表示词汇与环境高度匹配。

        Returns:
            适应度 [0, 1]
        """
        regime = self.get_current_regime()
        if not regime.feature_distribution:
            return 1.0

        # 将体制特征分布转为张量
        features = sorted(regime.feature_distribution.keys())
        regime_probs = torch.tensor(
            [regime.feature_distribution[f] for f in features],
            dtype=torch.float32, device=self._device
        )

        # 归一化
        regime_probs = regime_probs / (regime_probs.sum() + 1e-8)

        # 简化评估: 基于 regime 历史中该体制出现频率
        # 以及词汇库中相关特征词的比例
        regime_name = regime.name
        regime_appearances = self._regime_history.count(regime_name)
        total_regimes = len(self._regime_history) + 1

        # 经验匹配度: 出现次数越多，理论上适应越好
        experience = min(regime_appearances / max(total_regimes * 0.25, 1), 1.0)

        # 休眠词汇中有多少可以被激活
        reactivatable = sum(
            1 for word, (rname, _) in self._dormant_vocabulary.items()
            if rname == regime_name
        )
        dormant_bonus = min(reactivatable / 10.0, 0.2)

        score = experience + dormant_bonus
        score = min(max(score, 0.0), 1.0)

        self._adaptation_scores.append(score)
        if len(self._adaptation_scores) > 200:
            self._adaptation_scores = self._adaptation_scores[-200:]

        return score

    def forget_obsolete(self, vocabulary: Dict[str, float],
                        rate: float = 0.1) -> Dict:
        """衰减与当前体制不匹配的词汇

        对每个词，检查其是否出现在当前体制的特征分布中。
        不匹配的词按 rate 指数衰减。

        Args:
            vocabulary: {word: frequency} 词汇频率字典
            rate: 衰减率，默认使用 self.decay_rate

        Returns:
            衰减后的词汇频率字典
        """
        if not vocabulary:
            return vocabulary

        regime = self.get_current_regime()
        active_features = set(regime.feature_distribution.keys())
        decay = rate if rate > 0 else self.decay_rate

        # 衰减因子张量
        decay_tensor = torch.tensor(1.0 - decay, dtype=torch.float32,
                                    device=self._device)

        updated = {}
        for word, freq in vocabulary.items():
            # 检查词是否与当前体制相关
            word_features = set(word.replace('_', ' ').replace('-', ' ').split())

            if word_features & active_features:
                # 相关词: 保持或略微提升
                updated[word] = freq
            else:
                # 不相关词: 衰减
                freq_tensor = torch.tensor(freq, dtype=torch.float32,
                                           device=self._device)
                new_freq = freq_tensor * decay_tensor

                if new_freq.item() < 0.01:
                    # 移入休眠
                    self._dormant_vocabulary[word] = (regime.name, freq)
                else:
                    updated[word] = new_freq.item()

        return updated

    def reactivate_dormant(self, vocabulary: Dict[str, float],
                           threshold: float = 0.1) -> Dict:
        """重新激活与当前体制匹配的休眠词汇

        当旧体制回归时，检查休眠词汇是否与当前体制相关，
        如果相关则重新插入词汇表。

        Args:
            vocabulary: 当前词汇频率字典
            threshold: 激活阈值，休眠词的原始频率需超过此值

        Returns:
            添加了激活词汇的字典
        """
        if not self._dormant_vocabulary:
            return vocabulary

        regime = self.get_current_regime()
        active_features = set(regime.feature_distribution.keys())

        reactivated = dict(vocabulary)
        to_remove = []

        for word, (regime_name, last_freq) in self._dormant_vocabulary.items():
            # 检查休眠词是否与当前体制匹配
            word_features = set(word.replace('_', ' ').replace('-', ' ').split())

            matches_current = bool(word_features & active_features)
            same_regime = (regime_name == regime.name)

            if matches_current or same_regime:
                if last_freq >= threshold:
                    # 激活: 以原始频率的一半恢复
                    reactivated[word] = last_freq * 0.5
                    to_remove.append(word)

        # 从休眠中移除已激活的词
        for word in to_remove:
            del self._dormant_vocabulary[word]

        return reactivated

    def get_regime_features_tensor(self) -> torch.Tensor:
        """获取当前体制的特征概率分布张量

        Returns:
            归一化的特征概率张量 (n_features,)
        """
        regime = self.get_current_regime()
        if not regime.feature_distribution:
            return torch.zeros(1, dtype=torch.float32, device=self._device)

        features = sorted(regime.feature_distribution.keys())
        probs = torch.tensor(
            [regime.feature_distribution[f] for f in features],
            dtype=torch.float32, device=self._device
        )
        return probs / (probs.sum() + 1e-8)

    def get_cycle_count(self) -> int:
        """获取已完成的体制循环次数"""
        if not self._regime_history:
            return 0
        cycle_length = len(self._regimes)
        return len(self._regime_history) // cycle_length

    # ── 持久化 ────────────────────────────────────────────────

    def save_state(self) -> dict:
        return {
            'regime_index': self._regime_index,
            'step_in_regime': self._step_in_regime,
            'decay_rate': self.decay_rate,
            'reactivation_threshold': self.reactivation_threshold,
            'regime_history': list(self._regime_history),
            'dormant_vocabulary': {
                k: list(v) for k, v in self._dormant_vocabulary.items()
            },
            'adaptation_scores': list(self._adaptation_scores),
        }

    def load_state(self, state: dict) -> None:
        self._regime_index = state.get('regime_index', 0)
        self._step_in_regime = state.get('step_in_regime', 0)
        self.decay_rate = state.get('decay_rate', 0.1)
        self.reactivation_threshold = state.get('reactivation_threshold', 0.1)
        self._regime_history = state.get('regime_history', [])
        self._dormant_vocabulary = {
            k: tuple(v) for k, v in state.get('dormant_vocabulary', {}).items()
        }
        self._adaptation_scores = state.get('adaptation_scores', [])
