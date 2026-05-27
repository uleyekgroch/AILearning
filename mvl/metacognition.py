"""
元认知与自我反思模块

核心思想：
元认知 = 对自身认知过程的觉察
- Agent 能评估自己的不确定性（"我不确定"）
- Agent 能反思自己的学习策略（"我应该换种方法"）
- Agent 能请求帮助（"请再教我一次"）

元认知信号：
- "uncertain": 预测误差高时发出
- "help": 无法理解时发出
- "confident": 对自己的理解有信心

涌现条件：
1. Agent 遇到交流困难（成功率低）
2. 元认知信号能改善学习效率
3. Teacher 根据信号调整教学策略
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, _symbol_category,
    COLORS, SHAPES, SIZES, MATERIALS,
)

# 元认知信号
METACOGNITIVE_SIGNALS = {'uncertain', 'help', 'confident', 'confused', 'understand'}


class MetacognitiveAgent:
    """具有元认知能力的 Agent"""

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.uncertainty_history = []  # 不确定性历史
        self.strategy_history = []  # 策略历史
        self.help_requests = 0  # 请求帮助次数
        self.confidence = 0.5  # 初始置信度

    def assess_uncertainty(self, recent_successes: int,
                           recent_games: int) -> float:
        """
        评估当前不确定性

        基于最近的成功率计算不确定性
        """
        if recent_games == 0:
            return 0.5

        success_rate = recent_successes / recent_games
        uncertainty = 1.0 - success_rate
        self.uncertainty_history.append(uncertainty)
        return uncertainty

    def should_request_help(self, uncertainty: float,
                           threshold: float = 0.7) -> bool:
        """
        判断是否应该请求帮助

        当不确定性超过阈值时，请求帮助
        """
        if uncertainty > threshold:
            self.help_requests += 1
            return True
        return False

    def get_metacognitive_signal(self, uncertainty: float) -> str:
        """
        根据不确定性返回元认知信号

        - uncertainty > 0.7: "uncertain" 或 "help"
        - uncertainty < 0.3: "confident"
        - 其他: "understand"
        """
        if uncertainty > 0.7:
            if np.random.random() < 0.5:
                return 'uncertain'
            else:
                return 'help'
        elif uncertainty < 0.3:
            return 'confident'
        else:
            return 'understand'

    def update_confidence(self, success: bool):
        """更新置信度"""
        if success:
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.confidence = max(0.0, self.confidence - 0.1)


class MetacognitiveTeacher:
    """具有元认知感知的教师"""

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.adaptations = 0  # 教学调整次数

    def adapt_teaching(self, student_signal: str,
                       current_strategy: str) -> str:
        """
        根据学生的元认知信号调整教学策略

        参数：
            student_signal: 学生的元认知信号
            current_strategy: 当前教学策略

        返回：
            新的教学策略
        """
        if student_signal in ('uncertain', 'help', 'confused'):
            self.adaptations += 1
            # 切换到更简单的策略
            if current_strategy == 'complex':
                return 'simple'
            elif current_strategy == 'simple':
                return 'basic'
            else:
                return 'basic'
        elif student_signal == 'confident':
            # 可以增加难度
            if current_strategy == 'basic':
                return 'simple'
            elif current_strategy == 'simple':
                return 'complex'
        return current_strategy


class MetacognitiveGame:
    """元认知交流游戏"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.agent = MetacognitiveAgent(self.language)
        self.teacher = MetacognitiveTeacher(self.language)
        self.game_log = []
        self.signal_used = 0
        self.signal_success = 0
        self.current_strategy = 'simple'

    def play_round(self, scene_features: List[Dict[str, str]],
                   target_idx: int) -> bool:
        """
        进行一轮元认知游戏

        流程：
        1. Agent 尝试描述目标
        2. Agent 评估自己的不确定性
        3. 如果不确定性高，发出元认知信号
        4. Teacher 根据信号调整教学
        5. Agent 根据调整后的策略重新尝试
        """
        if target_idx >= len(scene_features):
            return False

        target = scene_features[target_idx]

        # 第一次尝试
        utterance = self._describe_with_strategy(target, scene_features,
                                                  self.current_strategy)
        if not utterance:
            return False

        # 评估不确定性
        recent_successes = self.language.total_successes
        recent_games = self.language.total_games
        uncertainty = self.agent.assess_uncertainty(recent_successes, recent_games)

        # 获取元认知信号
        signal = self.agent.get_metacognitive_signal(uncertainty)
        used_signal = signal in METACOGNITIVE_SIGNALS

        if used_signal:
            self.signal_used += 1

        # Teacher 根据信号调整
        old_strategy = self.current_strategy
        self.current_strategy = self.teacher.adapt_teaching(signal,
                                                             self.current_strategy)

        # 如果策略改变，重新尝试
        if self.current_strategy != old_strategy:
            utterance = self._describe_with_strategy(target, scene_features,
                                                      self.current_strategy)

        # 判断成功
        success = self._check_success(utterance, scene_features, target_idx)
        self.agent.update_confidence(success)

        if used_signal and success:
            self.signal_success += 1

        # 更新语言统计
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        if len(utterance) > 1:
            self.language.multi_symbol_games += 1
        self.language.record_usage(utterance, success)

        self.game_log.append({
            'target_idx': target_idx,
            'utterance': utterance,
            'success': success,
            'signal': signal,
            'uncertainty': uncertainty,
            'strategy': self.current_strategy,
        })
        return success

    def _describe_with_strategy(self, target: Dict[str, str],
                                 scene: List[Dict[str, str]],
                                 strategy: str) -> List[str]:
        """使用指定策略描述目标"""
        target_values = set(target.values())

        if strategy == 'basic':
            # 基本策略：只用单个符号
            for sym in target_values:
                count = sum(1 for obj in scene if sym in obj.values())
                if count == 1:
                    return [sym]
            return [list(target_values)[0]]

        elif strategy == 'simple':
            # 简单策略：用 1-2 个符号
            # 先尝试单符号
            for sym in target_values:
                count = sum(1 for obj in scene if sym in obj.values())
                if count == 1:
                    return [sym]
            # 再尝试 2 符号组合
            target_list = list(target_values)
            for i in range(len(target_list)):
                for j in range(i + 1, len(target_list)):
                    combo = [target_list[i], target_list[j]]
                    count = sum(1 for obj in scene
                               if all(s in obj.values() for s in combo))
                    if count == 1:
                        return combo
            return [target_list[0]]

        else:  # complex
            # 复杂策略：用任意数量的符号
            # 先尝试单符号
            for sym in target_values:
                count = sum(1 for obj in scene if sym in obj.values())
                if count == 1:
                    return [sym]
            # 再尝试 2 符号组合
            target_list = list(target_values)
            for i in range(len(target_list)):
                for j in range(i + 1, len(target_list)):
                    combo = [target_list[i], target_list[j]]
                    count = sum(1 for obj in scene
                               if all(s in obj.values() for s in combo))
                    if count == 1:
                        return combo
            # 再尝试 3 符号组合
            for i in range(len(target_list)):
                for j in range(i + 1, len(target_list)):
                    for k in range(j + 1, len(target_list)):
                        combo = [target_list[i], target_list[j], target_list[k]]
                        count = sum(1 for obj in scene
                                   if all(s in obj.values() for s in combo))
                        if count == 1:
                            return combo
            return target_list[:2]

    def _check_success(self, utterance: List[str],
                       scene: List[Dict[str, str]],
                       target_idx: int) -> bool:
        """检查描述是否成功标识目标"""
        if not utterance:
            return False

        # 计算每个物体的匹配度
        scores = []
        for i, obj in enumerate(scene):
            obj_values = set(obj.values())
            matches = sum(1 for s in utterance if s in obj_values)
            scores.append((i, matches))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] == target_idx and scores[0][1] > 0

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['signal_used'] = self.signal_used
        stats['signal_success'] = self.signal_success
        stats['signal_rate'] = self.signal_used / max(1, len(self.game_log))
        stats['help_requests'] = self.agent.help_requests
        stats['teacher_adaptations'] = self.teacher.adaptations
        return stats


def generate_metacognitive_scene(num_objects: int = 4) -> Tuple[List[Dict[str, str]], int]:
    """
    生成元认知场景

    场景设计：
    - 有歧义的场景（多个物体共享特征）
    - Agent 需要学习如何区分
    - 高不确定性 → 元认知信号
    """
    colors = list(COLORS)
    shapes = list(SHAPES)
    sizes = list(SIZES)

    np.random.shuffle(colors)
    np.random.shuffle(shapes)
    np.random.shuffle(sizes)

    scene = []
    for i in range(num_objects):
        obj = {
            'color': colors[i % len(colors)],
            'shape': shapes[i % len(shapes)],
            'size': sizes[i % len(sizes)],
        }
        scene.append(obj)

    target_idx = np.random.randint(0, len(scene))
    return scene, target_idx


def generate_difficult_scene(num_objects: int = 6) -> Tuple[List[Dict[str, str]], int]:
    """
    生成困难场景（高歧义）

    多个物体共享部分特征，增加不确定性
    """
    colors = list(COLORS)
    shapes = list(SHAPES)

    # 只用 2 种颜色和 2 种形状，增加歧义
    np.random.shuffle(colors)
    np.random.shuffle(shapes)
    selected_colors = colors[:2]
    selected_shapes = shapes[:2]

    scene = []
    for i in range(num_objects):
        obj = {
            'color': selected_colors[i % 2],
            'shape': selected_shapes[i % 2],
            'size': np.random.choice(list(SIZES)),
        }
        scene.append(obj)

    target_idx = np.random.randint(0, len(scene))
    return scene, target_idx


def test_metacognition():
    """测试元认知机制"""
    print("=== 元认知机制测试 ===")

    game = MetacognitiveGame()

    # 简单场景
    scene, target_idx = generate_metacognitive_scene()
    print(f"\n场景:")
    for i, obj in enumerate(scene):
        marker = " <-- TARGET" if i == target_idx else ""
        print(f"  {i}: {obj}{marker}")

    success = game.play_round(scene, target_idx)
    print(f"成功: {success}")
    print(f"信号: {game.game_log[-1]['signal']}")
    print(f"不确定性: {game.game_log[-1]['uncertainty']:.2f}")
    print(f"策略: {game.game_log[-1]['strategy']}")

    # 困难场景
    scene, target_idx = generate_difficult_scene()
    print(f"\n困难场景:")
    for i, obj in enumerate(scene):
        marker = " <-- TARGET" if i == target_idx else ""
        print(f"  {i}: {obj}{marker}")

    success = game.play_round(scene, target_idx)
    print(f"成功: {success}")
    print(f"信号: {game.game_log[-1]['signal']}")
    print(f"不确定性: {game.game_log[-1]['uncertainty']:.2f}")


if __name__ == '__main__':
    test_metacognition()
