"""
噪声语言系统：概率化反馈信号

核心思想：
真实的语言交流充满噪声——听错、误解、注意力分散。
当前系统的二值反馈（成功/失败）太强，导致所有 agent 快速趋同。

噪声机制：
- NoisyListener：以 noise_rate 概率选择随机物体（而非最佳匹配）
- NoisyCommunicationGame：在标准游戏基础上添加噪声
- 不同 agent 可以有不同的噪声水平（模拟个体差异）

效果：
- 15% 噪声 → 每个符号的成功率上限约 85%
- 学习信号变弱 → 收敛速度降低 3-5x
- 不同 agent 的噪声样本不同 → 方言漂移
"""

import random
import numpy as np
from typing import List, Dict, Optional

from language_emergence import (
    EmergingLanguage, Speaker, Listener, CommunicationGame,
    LanguageAgent, cross_language_round, generate_rich_scene
)


class NoisyListener(Listener):
    """
    带噪声的 Listener

    以 noise_rate 概率选择随机物体，而非最佳匹配。
    模拟真实交流中的听错、误解、注意力分散。
    """

    def __init__(self, language: EmergingLanguage, noise_rate: float = 0.15):
        super().__init__(language)
        self.noise_rate = noise_rate
        self.noise_events = 0
        self.total_interpretations = 0

    def interpret(self, utterance: List[str],
                  scene_features: List[Dict[str, str]]) -> Optional[int]:
        """
        带噪声的解释

        以 noise_rate 概率：
        - 选择随机物体（模拟听错）
        - 返回 None（模拟注意力分散）
        """
        self.total_interpretations += 1

        # 标准匹配
        best_idx = super().interpret(utterance, scene_features)

        # 噪声注入
        if random.random() < self.noise_rate:
            self.noise_events += 1
            noise_type = random.random()
            if noise_type < 0.7:
                # 70% 噪声：选随机物体（听错）
                return random.randint(0, len(scene_features) - 1)
            else:
                # 30% 噪声：返回 None（注意力分散）
                return None

        return best_idx

    def get_noise_stats(self) -> Dict:
        """获取噪声统计"""
        return {
            'noise_rate': self.noise_rate,
            'total_interpretations': self.total_interpretations,
            'noise_events': self.noise_events,
            'actual_noise_rate': (self.noise_events / max(1, self.total_interpretations)),
        }


class NoisyLanguageAgent(LanguageAgent):
    """
    带噪声的语言 Agent

    使用 NoisyListener 替代标准 Listener。
    不同 agent 可以有不同的噪声水平。
    """

    def __init__(self, agent_id: str, noise_rate: float = 0.15):
        super().__init__(agent_id)
        # 用 NoisyListener 替换标准 Listener
        self.listener = NoisyListener(self.language, noise_rate)
        self.noise_rate = noise_rate


class NoisyCommunicationGame(CommunicationGame):
    """
    带噪声的交流游戏

    Speaker 和 Listener 都可以有噪声：
    - Speaker 噪声：描述时偶尔遗漏属性（模拟表达不清）
    - Listener 噪声：解释时偶尔选错物体（模拟听错）
    """

    def __init__(self, speaker_noise_rate: float = 0.0,
                 listener_noise_rate: float = 0.15):
        self.speaker_noise_rate = speaker_noise_rate
        self.listener_noise_rate = listener_noise_rate

    def play_round_noisy(self, speaker: LanguageAgent,
                         listener: NoisyLanguageAgent,
                         scene_features: List[Dict[str, str]],
                         target_idx: int) -> bool:
        """
        带噪声的一轮交流

        Args:
            speaker: 说话者
            listener: 带噪声的听者
            scene_features: 场景特征
            target_idx: 目标物体索引

        Returns:
            是否成功
        """
        if target_idx >= len(scene_features) or not scene_features:
            return False

        target = scene_features[target_idx]

        # Speaker 描述（可能有表达噪声）
        utterance = speaker.speak(target, scene_features)

        # Speaker 噪声：以概率遗漏一个符号
        if self.speaker_noise_rate > 0 and len(utterance) > 1:
            if random.random() < self.speaker_noise_rate:
                # 随机移除一个符号
                drop_idx = random.randint(0, len(utterance) - 1)
                utterance = utterance[:drop_idx] + utterance[drop_idx + 1:]

        if not utterance:
            return False

        # Listener 解释（带噪声）
        chosen_idx = listener.listen(utterance, scene_features)
        success = (chosen_idx == target_idx)

        # 更新两个 agent 的语言
        speaker.update_from_communication(utterance, success)
        listener.update_from_communication(utterance, success)

        return success


def noisy_cross_language_round(speaker_agent: LanguageAgent,
                               listener_agent: NoisyLanguageAgent,
                               scene_features: List[Dict[str, str]],
                               target_idx: int) -> bool:
    """
    带噪声的跨语言交流

    与 cross_language_round 相同，但 listener 使用 NoisyListener。
    """
    if target_idx >= len(scene_features) or not scene_features:
        return False

    target = scene_features[target_idx]

    # speaker 描述
    utterance = speaker_agent.speak(target, scene_features)
    if not utterance:
        return False

    # listener 解释（带噪声）
    chosen_idx = listener_agent.listen(utterance, scene_features)
    success = (chosen_idx == target_idx)

    # 分别更新两个 agent 的语言
    speaker_agent.update_from_communication(utterance, success)
    listener_agent.update_from_communication(utterance, success)

    return success
