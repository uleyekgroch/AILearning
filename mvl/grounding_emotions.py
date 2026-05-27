"""
情感接地模块：情感符号从 agent 内部状态中涌现

核心思想：
情感不是外在标签，而是 agent 对自身内部状态的符号化表达。
"happy" = 低预测误差+高学习进度的可靠信号，
"curious" = 高预测误差+高学习进度的状态。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from language_emergence import (
    EmergingLanguage, Speaker, Listener,
    EMOTIONS, _symbol_category, generate_rich_scene
)


@dataclass
class EmotionState:
    """agent 的内部情感状态"""
    prediction_error: float    # 当前预测误差 (0-1)
    learning_progress: float   # 学习进度 (0-1)
    reward_signal: float       # 最近的奖励信号 (0-1)
    confidence: float          # 对自身能力的信心 (0-1)

    def primary_emotion(self) -> str:
        """映射到主要情感符号"""
        return EmotionMapper.from_signals(
            self.prediction_error,
            self.learning_progress,
            self.reward_signal,
            self.confidence,
        )


class EmotionMapper:
    """将 agent 内部信号映射到情感符号"""

    @staticmethod
    def from_signals(prediction_error: float,
                     learning_progress: float,
                     reward: float,
                     confidence: float) -> str:
        """
        从内部信号推断情感

        规则（基于心理学研究）：
        - happy:      低误差 + 高奖励
        - calm:       低误差 + 低奖励（稳定状态）
        - curious:    高误差 + 高进度（探索动力）
        - frustrated: 高误差 + 低进度（卡住了）
        - surprised:  奖励突变（意外结果）
        - bored:      低误差 + 低进度（无新事物）
        - confident:  高信心 + 低误差
        - scared:     高误差 + 低信心（不确定）
        """
        # 置信度综合指标
        competence = confidence * (1 - prediction_error)

        if reward > 0.7 and prediction_error < 0.3:
            return 'happy'
        if competence > 0.7:
            return 'confident'
        if prediction_error > 0.6 and learning_progress > 0.3:
            return 'curious'
        if prediction_error > 0.6 and learning_progress < 0.3:
            return 'frustrated'
        if prediction_error < 0.3 and reward < 0.3:
            if learning_progress < 0.2:
                return 'bored'
            return 'calm'
        if confidence < 0.3 and prediction_error > 0.5:
            return 'scared'
        if reward > 0.5:
            return 'happy'
        return 'calm'

    @staticmethod
    def inject_emotion_features(scene: List[Dict[str, str]],
                                 emotion: str) -> List[Dict[str, str]]:
        """将情感状态注入场景特征（每个物体都携带情感标签）"""
        enriched = []
        for obj in scene:
            enriched.append({**obj, 'emotion': emotion})
        return enriched

    @staticmethod
    def generate_emotion_signal() -> EmotionState:
        """生成随机的内部状态（用于实验）"""
        return EmotionState(
            prediction_error=np.random.random(),
            learning_progress=np.random.random(),
            reward_signal=np.random.random(),
            confidence=np.random.random(),
        )


def generate_emotion_scene(num_objects: int = 6,
                            complexity: str = 'medium') -> Tuple[List[Dict[str, str]], List[str]]:
    """
    生成带情感标注的场景

    每个物体有不同的情感标签（代表 agent 对该物体的情感反应）。
    这使得情感符号可以成为区分特征。
    """
    base_scene = generate_rich_scene(complexity)[:num_objects]
    emotions = list(EMOTIONS)
    assigned_emotions = []
    for i, obj in enumerate(base_scene):
        emotion = np.random.choice(emotions)
        obj['emotion'] = emotion
        assigned_emotions.append(emotion)
    return base_scene, assigned_emotions


class EmotionCommunicationGame:
    """情感交流游戏：描述"感觉如何+物体特征"，listener 识别目标"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)
        self.game_log = []

    def play_round(self, scene_features: List[Dict[str, str]],
                   target_idx: int,
                   emotion: str = None) -> bool:
        if target_idx >= len(scene_features):
            return False

        # 如果场景中已有情感标签（per-object），直接使用
        # 否则注入统一情感（向后兼容）
        if emotion is not None and 'emotion' not in scene_features[0]:
            enriched = EmotionMapper.inject_emotion_features(scene_features, emotion)
        else:
            enriched = scene_features

        # speaker 描述目标（包含情感）
        target = enriched[target_idx]
        utterance = self.speaker.describe(target, enriched)
        if not utterance:
            return False

        # listener 解释
        chosen_idx = self.listener.interpret(utterance, enriched)
        success = (chosen_idx == target_idx)

        # 更新语言统计
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        if len(utterance) > 1:
            self.language.multi_symbol_games += 1

        self.language.record_usage(utterance, success)
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                self.language.record_collocation(utterance[i], utterance[i + 1], success)
            self.language.record_ngram(utterance, success)

        self.game_log.append({
            'target_idx': target_idx,
            'emotion': emotion,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
        })
        return success


class EmotionLanguageAgent:
    """拥有独立情感语言的 agent"""

    def __init__(self, agent_id: str):
        self.id = agent_id
        self.language = EmergingLanguage()
        self.speaker = Speaker(self.language)
        self.listener = Listener(self.language)
        self.internal_state = EmotionState(
            prediction_error=0.5,
            learning_progress=0.0,
            reward_signal=0.5,
            confidence=0.5,
        )

    def update_internal_state(self, prediction_error: float,
                               learning_progress: float,
                               reward: float):
        """更新内部状态（平滑过渡）"""
        alpha = 0.3  # 平滑系数
        self.internal_state.prediction_error = (
            alpha * prediction_error + (1 - alpha) * self.internal_state.prediction_error
        )
        self.internal_state.learning_progress = (
            alpha * learning_progress + (1 - alpha) * self.internal_state.learning_progress
        )
        self.internal_state.reward_signal = (
            alpha * reward + (1 - alpha) * self.internal_state.reward_signal
        )
        # 信心随成功积累
        if reward > 0.5:
            self.internal_state.confidence = min(1.0, self.internal_state.confidence + 0.05)
        else:
            self.internal_state.confidence = max(0.0, self.internal_state.confidence - 0.02)

    def speak(self, scene: List[Dict[str, str]], target_idx: int) -> List[str]:
        """描述目标（包含当前情感状态）"""
        emotion = self.internal_state.primary_emotion()
        enriched = EmotionMapper.inject_emotion_features(scene, emotion)
        if target_idx < len(enriched):
            return self.speaker.describe(enriched[target_idx], enriched)
        return []

    def listen(self, utterance: List[str], scene: List[Dict[str, str]]) -> Optional[int]:
        """解释描述（场景已包含情感标签）"""
        return self.listener.interpret(utterance, scene)

    def update(self, utterance: List[str], success: bool):
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                self.language.record_collocation(utterance[i], utterance[i + 1], success)
            self.language.record_ngram(utterance, success)


def cross_emotion_round(speaker_agent: EmotionLanguageAgent,
                        listener_agent: EmotionLanguageAgent,
                        scene: List[Dict[str, str]],
                        target_idx: int) -> bool:
    """跨语言情感交流一轮"""
    utterance = speaker_agent.speak(scene, target_idx)
    if not utterance:
        return False

    # listener 需要用 speaker 的情感来解释
    speaker_emotion = speaker_agent.internal_state.primary_emotion()
    enriched = EmotionMapper.inject_emotion_features(scene, speaker_emotion)
    chosen_idx = listener_agent.listen(utterance, enriched)
    success = (chosen_idx == target_idx)

    speaker_agent.update(utterance, success)
    listener_agent.update(utterance, success)
    return success
