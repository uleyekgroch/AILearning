"""
Phase 19: 因果推理模块 —— 让 "because" 涌现

核心思想：
"because" 未涌现的根本原因是系统无法区分因果关系和时序关系。
解决方案：创建混杂场景——事件既有时序相关又有因果相关，
Speaker 必须用 "because" 标记真正的因果关系，用 "then" 标记纯时序关系。

涌现条件：
1. 场景中同时存在因果对和虚假相关对
2. Listener 需要根据标记预测结果（因果对预测可靠，虚假相关不可靠）
3. "because" 提供预测优势，"then" 不提供
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, _symbol_category,
    COLORS, SHAPES, SIZES, MATERIALS, ACTIONS,
)

# 因果推理相关标记
CAUSAL_REASONING_MARKERS = {'because', 'so', 'therefore', 'thus', 'hence'}
TEMPORAL_MARKERS = {'then', 'after', 'before', 'when'}
ALL_CONNECTORS = CAUSAL_REASONING_MARKERS | TEMPORAL_MARKERS


class CausalWorld:
    """
    因果世界：包含因果规则和虚假相关

    因果规则：P(effect | cause) = 0.9（可靠）
    虚假相关：P(effect | cause) = 0.5（巧合）
    """

    def __init__(self):
        self.causal_rules: Dict[Tuple[str, str], float] = {}  # (action, effect) -> probability
        self.spurious_correlations: Dict[Tuple[str, str], float] = {}
        self.event_history: List[Tuple[str, str, bool]] = []  # (action, effect, is_causal)

    def add_causal_rule(self, action: str, effect: str, probability: float = 0.9):
        """添加因果规则"""
        self.causal_rules[(action, effect)] = probability

    def add_spurious_correlation(self, action: str, effect: str, probability: float = 0.5):
        """添加虚假相关"""
        self.spurious_correlations[(action, effect)] = probability

    def execute_action(self, action: str) -> Optional[str]:
        """
        执行动作，返回效果

        根据因果规则或虚假相关决定是否产生效果
        """
        # 检查因果规则
        for (cause, effect), prob in self.causal_rules.items():
            if cause == action:
                if np.random.random() < prob:
                    self.event_history.append((action, effect, True))
                    return effect

        # 检查虚假相关
        for (cause, effect), prob in self.spurious_correlations.items():
            if cause == action:
                if np.random.random() < prob:
                    self.event_history.append((action, effect, False))
                    return effect

        # 无效果
        self.event_history.append((action, 'nothing', False))
        return 'nothing'

    def is_causal(self, action: str, effect: str) -> bool:
        """检查 (action, effect) 是否是因果关系"""
        return (action, effect) in self.causal_rules

    def get_probability(self, action: str, effect: str) -> float:
        """获取 (action, effect) 的概率"""
        if (action, effect) in self.causal_rules:
            return self.causal_rules[(action, effect)]
        if (action, effect) in self.spurious_correlations:
            return self.spurious_correlations[(action, effect)]
        return 0.1  # 默认低概率


class CausalModel:
    """
    内部因果模型：从观察中学习因果关系

    使用观察频率估计 P(effect | action)
    区分因果和虚假相关需要多次观察
    """

    def __init__(self):
        self.observations: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {'success': 0, 'total': 0})
        self.causal_confidence: Dict[Tuple[str, str], float] = {}

    def observe(self, action: str, effect: str, success: bool):
        """观察一次事件"""
        key = (action, effect)
        self.observations[key]['total'] += 1
        if success:
            self.observations[key]['success'] += 1

        # 更新置信度
        stats = self.observations[key]
        if stats['total'] >= 3:  # 至少 3 次观察才更新
            self.causal_confidence[key] = stats['success'] / stats['total']

    def get_confidence(self, action: str, effect: str) -> float:
        """获取因果置信度"""
        return self.causal_confidence.get((action, effect), 0.5)

    def is_likely_causal(self, action: str, effect: str, threshold: float = 0.7) -> bool:
        """判断是否可能是因果关系"""
        return self.get_confidence(action, effect) >= threshold

    def is_likely_spurious(self, action: str, effect: str, threshold: float = 0.6) -> bool:
        """判断是否可能是虚假相关"""
        conf = self.get_confidence(action, effect)
        return 0.4 < conf < threshold


class CausalReasoningAgent:
    """
    具有因果推理能力的 Agent

    能力：
    1. 观察事件并更新因果模型
    2. 根据因果置信度选择标记（"because" vs "then"）
    3. 根据标记预测结果的可靠性
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.causal_model = CausalModel()
        self.because_usage = 0
        self.then_usage = 0

    def observe_event(self, action: str, effect: str, is_causal: bool):
        """观察事件并更新模型"""
        self.causal_model.observe(action, effect, is_causal)

    def choose_connector(self, action: str, effect: str) -> str:
        """
        根据因果置信度选择连接词

        高置信度（>0.7）→ "because"（因果）
        中置信度（0.4-0.7）→ "then"（时序）
        低置信度（<0.4）→ "then"（时序）
        """
        confidence = self.causal_model.get_confidence(action, effect)

        # 检查词汇表中是否有 "because"
        has_because = 'because' in self.language.vocabulary

        if confidence > 0.7:
            # 高置信度：使用 "because"
            self.because_usage += 1
            return 'because'
        else:
            # 低置信度：使用 "then"
            self.then_usage += 1
            return 'then'

    def interpret_connector(self, connector: str, action: str, effect: str) -> float:
        """
        根据连接词预测结果的可靠性

        "because" → 高可靠性（0.9）
        "then" → 中可靠性（0.5）
        """
        if connector == 'because':
            return 0.9  # 因果关系，预测可靠
        elif connector == 'then':
            return 0.5  # 时序关系，预测不可靠
        else:
            return 0.5

    def get_stats(self) -> Dict:
        total = self.because_usage + self.then_usage
        return {
            'because_usage': self.because_usage,
            'then_usage': self.then_usage,
            'because_rate': self.because_usage / max(1, total),
            'causal_model_size': len(self.causal_model.causal_confidence),
        }


class CausalCommunicationGame:
    """
    因果交流游戏

    游戏流程：
    1. 生成场景：包含因果对和虚假相关对
    2. Speaker 观察事件，选择连接词描述
    3. Listener 根据描述预测结果
    4. 验证预测：因果对预测可靠，虚假相关不可靠

    涌现压力：
    - 如果 Speaker 对所有事件都用 "then"，Listener 无法区分因果和虚假
    - 如果 Speaker 用 "because" 标记因果，Listener 可以做出更好的预测
    - "because" 提供预测优势 → 涌现
    """

    def __init__(self):
        self.language = EmergingLanguage()
        self.speaker = CausalReasoningAgent(self.language)
        self.listener = CausalReasoningAgent(self.language)
        self.game_log = []
        self.because_used = 0
        self.because_success = 0
        self.then_used = 0
        self.then_success = 0

    def play_round(self, events: List[Tuple[str, str, bool]]) -> bool:
        """
        进行一轮因果交流游戏

        参数：
            events: [(action, effect, is_causal), ...] 事件列表

        返回：
            预测是否成功
        """
        if not events:
            return False

        # 选择一个目标事件
        target_idx = np.random.randint(0, len(events))
        action, effect, is_causal = events[target_idx]

        # Speaker 选择连接词
        connector = self.speaker.choose_connector(action, effect)

        # 构造描述
        utterance = [action, connector, effect]

        # Listener 解释
        predicted_reliability = self.listener.interpret_connector(connector, action, effect)

        # 验证预测
        actual_reliability = 0.9 if is_causal else 0.5
        prediction_correct = abs(predicted_reliability - actual_reliability) < 0.3

        # 更新统计
        self.language.total_games += 1
        if prediction_correct:
            self.language.total_successes += 1

        if connector == 'because':
            self.because_used += 1
            if prediction_correct:
                self.because_success += 1
        else:
            self.then_used += 1
            if prediction_correct:
                self.then_success += 1

        self.language.record_usage(utterance, prediction_correct)

        # 更新双方的因果模型
        self.speaker.observe_event(action, effect, is_causal)
        self.listener.observe_event(action, effect, is_causal)

        self.game_log.append({
            'action': action,
            'effect': effect,
            'is_causal': is_causal,
            'connector': connector,
            'predicted_reliability': predicted_reliability,
            'actual_reliability': actual_reliability,
            'success': prediction_correct,
        })
        return prediction_correct

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['because_used'] = self.because_used
        stats['because_success'] = self.because_success / max(1, self.because_used)
        stats['then_used'] = self.then_used
        stats['then_success'] = self.then_success / max(1, self.then_used)
        stats['speaker_stats'] = self.speaker.get_stats()
        stats['listener_stats'] = self.listener.get_stats()
        return stats


def generate_causal_scenario(mode: str = 'spurious',
                              num_pairs: int = 3) -> List[Tuple[str, str, bool]]:
    """
    生成因果场景

    参数：
        mode: 场景模式
            - 'pure_causal': 只有因果对（"then" 足够）
            - 'spurious': 混合因果和虚假相关（需要 "because"）
            - 'counterfactual': 因果对 + 反事实（需要 "because"）
            - 'chain': 因果链（需要 "because" 区分直接和间接）
        num_pairs: 事件对数量

    返回：
        [(action, effect, is_causal), ...] 事件列表
    """
    action_pool = list(ACTIONS)
    effect_pool = ['displacement', 'attraction', 'attached', 'detached',
                   'translation', 'directional', 'halt', 'rotation',
                   'broken', 'heated', 'cooled', 'colored']

    np.random.shuffle(action_pool)
    np.random.shuffle(effect_pool)

    events = []

    if mode == 'pure_causal':
        # 所有事件都是因果关系
        for i in range(num_pairs):
            action = action_pool[i % len(action_pool)]
            effect = effect_pool[i % len(effect_pool)]
            events.append((action, effect, True))

    elif mode == 'spurious':
        # 混合因果和虚假相关
        num_causal = max(1, num_pairs // 2)
        num_spurious = num_pairs - num_causal

        for i in range(num_causal):
            action = action_pool[i % len(action_pool)]
            effect = effect_pool[i % len(effect_pool)]
            events.append((action, effect, True))

        for i in range(num_spurious):
            action = action_pool[(num_causal + i) % len(action_pool)]
            effect = effect_pool[(num_causal + i) % len(effect_pool)]
            events.append((action, effect, False))

    elif mode == 'counterfactual':
        # 因果对 + 反事实（相同动作，不同效果）
        for i in range(num_pairs):
            action = action_pool[i % len(action_pool)]
            if i % 2 == 0:
                effect = effect_pool[i % len(effect_pool)]
                events.append((action, effect, True))
            else:
                # 反事实：相同动作，随机效果
                effect = np.random.choice(effect_pool)
                events.append((action, effect, False))

    elif mode == 'chain':
        # 因果链：A->B->C
        for i in range(num_pairs):
            action = action_pool[i % len(action_pool)]
            effect = effect_pool[(i + 1) % len(effect_pool)]
            events.append((action, effect, True))

    np.random.shuffle(events)
    return events


class BaselineCausalGame:
    """无因果推理的基线游戏（只用 "then"）"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.game_log = []

    def play_round(self, events: List[Tuple[str, str, bool]]) -> bool:
        if not events:
            return False

        target_idx = np.random.randint(0, len(events))
        action, effect, is_causal = events[target_idx]

        # 基线：总是用 "then"
        utterance = [action, 'then', effect]

        # 随机预测可靠性
        predicted_reliability = 0.5
        actual_reliability = 0.9 if is_causal else 0.5
        prediction_correct = abs(predicted_reliability - actual_reliability) < 0.3

        self.language.total_games += 1
        if prediction_correct:
            self.language.total_successes += 1
        self.language.record_usage(utterance, prediction_correct)

        self.game_log.append({
            'action': action,
            'effect': effect,
            'is_causal': is_causal,
            'connector': 'then',
            'success': prediction_correct,
        })
        return prediction_correct

    def get_stats(self) -> Dict:
        return self.language.get_stats()


def test_causal_reasoning():
    """测试因果推理机制"""
    print("=== 因果推理机制测试 ===")

    # 创建因果世界
    world = CausalWorld()
    world.add_causal_rule('push', 'displacement', 0.9)
    world.add_spurious_correlation('pull', 'attraction', 0.5)

    print(f"\n因果规则: {world.causal_rules}")
    print(f"虚假相关: {world.spurious_correlations}")

    # 测试因果模型
    model = CausalModel()
    for _ in range(10):
        model.observe('push', 'displacement', True)
        model.observe('pull', 'attraction', np.random.random() < 0.5)

    print(f"\n因果置信度:")
    for key, conf in model.causal_confidence.items():
        print(f"  {key}: {conf:.2f}")

    # 测试 Agent
    language = EmergingLanguage()
    agent = CausalReasoningAgent(language)

    for _ in range(10):
        agent.observe_event('push', 'displacement', True)
        agent.observe_event('pull', 'attraction', np.random.random() < 0.5)

    print(f"\n连接词选择:")
    print(f"  push/displacement: {agent.choose_connector('push', 'displacement')}")
    print(f"  pull/attraction: {agent.choose_connector('pull', 'attraction')}")

    print(f"\nAgent 统计: {agent.get_stats()}")


if __name__ == '__main__':
    test_causal_reasoning()
