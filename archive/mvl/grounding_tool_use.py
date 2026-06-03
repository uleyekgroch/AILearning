"""
Phase 22: 工具使用模块 —— 问题解决和规划

核心思想：
工具使用 = 理解物体可以服务于超越其身份的功能
- "棍子" 在目标是拿到高处东西时变成 "延伸器"
- 这需要目标依赖的物体重释和多步行动序列规划

涌现条件：
1. 直接行动无法达成目标
2. 必须使用工具
3. 用功能描述比外观描述更有效
4. "use", "for", "need" 从需要工具描述时涌现
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, _symbol_category,
    TOOL_MARKERS,
)


class Tool:
    """
    工具：具有物理特征和功能属性的物体

    物理特征：外观（颜色、形状、大小）
    功能属性：能做什么（reach, hit, contain, cut, push）
    有效性：对不同目标的有效性分数
    """

    def __init__(self, name: str, features: Dict[str, str],
                 affordances: Set[str],
                 effectiveness: Dict[str, float] = None):
        self.name = name
        self.features = features  # 外观特征
        self.affordances = affordances  # 功能（'reach', 'hit', 'contain', 'cut', 'push'）
        self.effectiveness = effectiveness or {}  # goal -> effectiveness (0-1)

    def to_symbols(self) -> List[str]:
        """外观特征符号"""
        return list(self.features.values())

    def to_functional_symbols(self) -> List[str]:
        """功能描述符号"""
        return list(self.affordances)

    def describe_functional(self) -> List[str]:
        """功能描述：affordance + feature"""
        symbols = list(self.affordances)
        symbols.extend(self.features.values())
        return symbols

    def is_effective_for(self, goal: str) -> float:
        """对特定目标的有效性"""
        return self.effectiveness.get(goal, 0.0)

    def __repr__(self):
        return f"Tool({self.name}, {self.affordances})"


class Goal:
    """
    目标：期望达到的世界状态

    属性：
    - required_affordance: 需要的功能类型
    - difficulty: 难度（0-1）
    - requires_tool: 是否需要工具
    """

    def __init__(self, name: str, required_affordance: str,
                 difficulty: float = 0.5, requires_tool: bool = True):
        self.name = name
        self.required_affordance = required_affordance
        self.difficulty = difficulty
        self.requires_tool = requires_tool

    def __repr__(self):
        return f"Goal({self.name}, needs={self.required_affordance})"


# ============================================================
# 工具-目标场景定义
# ============================================================

TOOL_TEMPLATES = [
    # (name, features, affordances, effectiveness)
    ('stick', {'color': 'brown', 'shape': 'long', 'size': 'thin'},
     {'reach', 'push'}, {'reach_object': 0.9, 'push_object': 0.7}),
    ('rock', {'color': 'gray', 'shape': 'round', 'size': 'heavy'},
     {'hit', 'break'}, {'break_wall': 0.8, 'hit_target': 0.6}),
    ('bowl', {'color': 'red', 'shape': 'round', 'size': 'hollow'},
     {'contain', 'carry'}, {'contain_water': 0.9, 'carry_food': 0.7}),
    ('rope', {'color': 'brown', 'shape': 'long', 'size': 'flexible'},
     {'reach', 'tie', 'pull'}, {'reach_object': 0.7, 'tie_together': 0.9, 'pull_object': 0.8}),
    ('knife', {'color': 'silver', 'shape': 'thin', 'size': 'sharp'},
     {'cut', 'slice'}, {'cut_rope': 0.9, 'slice_food': 0.8}),
    ('board', {'color': 'brown', 'shape': 'flat', 'size': 'wide'},
     {'bridge', 'cover', 'push'}, {'bridge_gap': 0.8, 'cover_hole': 0.7, 'push_object': 0.5}),
]

GOAL_TEMPLATES = [
    # (name, required_affordance, difficulty, requires_tool)
    ('reach_object', 'reach', 0.3, True),
    ('break_wall', 'hit', 0.5, True),
    ('contain_water', 'contain', 0.2, True),
    ('cut_rope', 'cut', 0.4, True),
    ('bridge_gap', 'bridge', 0.6, True),
    ('push_object', 'push', 0.2, False),  # 可以不用工具
    ('tie_together', 'tie', 0.4, True),
    ('pull_object', 'pull', 0.3, True),
]


def generate_tool_scenario(mode: str = 'direct',
                           num_tools: int = 4) -> Tuple[List[Tool], Goal, Optional[int]]:
    """
    生成工具使用场景

    参数：
        mode: 场景模式
            - 'direct': 不需要工具即可达成目标
            - 'single_tool': 需要一个工具（必须识别哪个）
            - 'tool_selection': 多个工具，必须选择正确的
            - 'sequential': 多个工具按顺序使用（计划通信）
        num_tools: 工具数量
    """
    # 选择目标
    if mode == 'direct':
        goal_data = [g for g in GOAL_TEMPLATES if not g[3]]  # requires_tool=False
        if not goal_data:
            goal_data = GOAL_TEMPLATES[:1]
        goal_idx = np.random.randint(0, len(goal_data))
        goal = Goal(*goal_data[goal_idx])
        correct_tool_idx = None
    else:
        goal_data = [g for g in GOAL_TEMPLATES if g[3]]  # requires_tool=True
        goal_idx = np.random.randint(0, len(goal_data))
        goal = Goal(*goal_data[goal_idx])

        # 选择工具
        available_templates = TOOL_TEMPLATES[:num_tools]
        tools = []
        for i, (name, features, affordances, effectiveness) in enumerate(available_templates):
            tools.append(Tool(name, features, affordances, effectiveness))

        # 找到正确的工具
        correct_tool_idx = None
        best_effectiveness = 0.0
        for i, tool in enumerate(tools):
            eff = tool.is_effective_for(goal.name)
            if eff > best_effectiveness:
                best_effectiveness = eff
                correct_tool_idx = i

        return tools, goal, correct_tool_idx

    # 对于 direct 模式，也创建工具（但不需要选择）
    available_templates = TOOL_TEMPLATES[:num_tools]
    tools = []
    for i, (name, features, affordances, effectiveness) in enumerate(available_templates):
        tools.append(Tool(name, features, affordances, effectiveness))

    return tools, goal, correct_tool_idx


class ToolCommunicationGame:
    """
    工具交流游戏

    游戏流程：
    1. 生成场景：工具 + 目标
    2. Speaker 描述应该使用哪个工具
    3. Listener 根据描述选择工具
    4. 如果直接描述（外观）无法区分，使用功能描述

    涌现压力：
    - 如果多个工具有相似外观，直接描述无法区分
    - 用功能描述（"use for reach"）可以区分
    - "use", "for" 从需要功能描述时涌现
    """

    def __init__(self):
        self.language = EmergingLanguage()
        self.game_log = []
        self.functional_used = 0
        self.functional_success = 0
        self.appearance_used = 0
        self.appearance_success = 0

    def play_round(self, tools: List[Tool], goal: Goal,
                   correct_idx: int = None) -> bool:
        if correct_idx is None:
            # 不需要工具
            self.language.total_games += 1
            self.language.total_successes += 1
            return True

        if correct_idx >= len(tools):
            return False

        target = tools[correct_idx]

        # 决定使用哪种描述策略
        # 检查外观是否能唯一标识目标
        appearance_unique = self._is_appearance_unique(target, tools)

        if appearance_unique:
            # 外观唯一，使用直接描述
            utterance = target.to_symbols()
            self.appearance_used += 1
        else:
            # 外观不唯一，使用功能描述
            utterance = self._describe_functional(target, goal)
            self.functional_used += 1

        if not utterance:
            return False

        # Listener 解释
        chosen_idx = self._listener_interpret(utterance, tools, goal)
        success = (chosen_idx == correct_idx)

        # 更新统计
        self.language.total_games += 1
        if success:
            self.language.total_successes += 1

        if not appearance_unique and success:
            self.functional_success += 1
        elif appearance_unique and success:
            self.appearance_success += 1

        self.language.record_usage(utterance, success)

        self.game_log.append({
            'correct_idx': correct_idx,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
            'used_functional': not appearance_unique,
            'goal': goal.name,
        })
        return success

    def _is_appearance_unique(self, target: Tool, tools: List[Tool]) -> bool:
        """检查目标工具的外观是否唯一"""
        target_features = set(target.to_symbols())
        for i, tool in enumerate(tools):
            if tool is target:
                continue
            tool_features = set(tool.to_symbols())
            # 如果有重叠特征，外观不唯一
            if len(target_features & tool_features) >= 2:
                return False
        return True

    def _describe_functional(self, target: Tool, goal: Goal) -> List[str]:
        """生成功能描述"""
        symbols = []
        # 添加功能标记
        if 'use' in TOOL_MARKERS:
            symbols.append('use')
        # 添加目标功能
        symbols.append(goal.required_affordance)
        # 添加 "for" 标记
        if 'for' in TOOL_MARKERS:
            symbols.append('for')
        # 添加工具的外观特征（辅助识别）
        symbols.extend(target.to_symbols())
        return symbols

    def _listener_interpret(self, utterance: List[str],
                            tools: List[Tool], goal: Goal) -> int:
        """Listener 解释描述"""
        scores = []
        for i, tool in enumerate(tools):
            score = 0.0

            # 功能匹配
            if goal.required_affordance in utterance:
                if goal.required_affordance in tool.affordances:
                    score += 3.0  # 功能匹配高权重

            # 外观匹配
            tool_features = set(tool.to_symbols())
            for s in utterance:
                if s in tool_features:
                    score += 1.0

            # 功能标记匹配
            if 'use' in utterance and tool.affordances:
                score += 0.5
            if 'for' in utterance and goal.required_affordance in tool.affordances:
                score += 0.5

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def get_stats(self) -> Dict:
        stats = self.language.get_stats()
        stats['functional_used'] = self.functional_used
        stats['functional_success'] = self.functional_success / max(1, self.functional_used)
        stats['appearance_used'] = self.appearance_used
        stats['appearance_success'] = self.appearance_success / max(1, self.appearance_used)
        return stats


class BaselineToolGame:
    """无功能描述的基线游戏（只使用外观描述）"""

    def __init__(self):
        self.language = EmergingLanguage()
        self.game_log = []

    def play_round(self, tools: List[Tool], goal: Goal,
                   correct_idx: int = None) -> bool:
        if correct_idx is None:
            self.language.total_games += 1
            self.language.total_successes += 1
            return True

        if correct_idx >= len(tools):
            return False

        target = tools[correct_idx]
        utterance = target.to_symbols()

        # 基线 Listener 只用外观匹配
        scores = []
        for i, tool in enumerate(tools):
            tool_features = set(tool.to_symbols())
            matches = sum(1 for s in utterance if s in tool_features)
            scores.append((i, matches))

        scores.sort(key=lambda x: x[1], reverse=True)
        chosen_idx = scores[0][0]
        success = (chosen_idx == correct_idx)

        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)

        self.game_log.append({
            'correct_idx': correct_idx,
            'utterance': utterance,
            'chosen': chosen_idx,
            'success': success,
        })
        return success

    def get_stats(self) -> Dict:
        return self.language.get_stats()


def test_tool_use():
    """测试工具使用机制"""
    print("=== 工具使用机制测试 ===")

    # 创建工具
    stick = Tool('stick', {'color': 'brown', 'shape': 'long', 'size': 'thin'},
                 {'reach', 'push'}, {'reach_object': 0.9, 'push_object': 0.7})
    rock = Tool('rock', {'color': 'gray', 'shape': 'round', 'size': 'heavy'},
                {'hit', 'break'}, {'break_wall': 0.8, 'hit_target': 0.6})

    print(f"\n工具: {stick}")
    print(f"  外观: {stick.to_symbols()}")
    print(f"  功能: {stick.to_functional_symbols()}")
    print(f"  reach_object 有效性: {stick.is_effective_for('reach_object')}")

    print(f"\n工具: {rock}")
    print(f"  外观: {rock.to_symbols()}")
    print(f"  功能: {rock.to_functional_symbols()}")

    # 创建目标
    goal = Goal('reach_object', 'reach', 0.3, True)
    print(f"\n目标: {goal}")
    print(f"  需要功能: {goal.required_affordance}")

    # 测试场景生成
    tools, goal, correct_idx = generate_tool_scenario(mode='tool_selection', num_tools=4)
    print(f"\n场景:")
    print(f"  工具: {[t.name for t in tools]}")
    print(f"  目标: {goal}")
    print(f"  正确工具: {tools[correct_idx].name if correct_idx is not None else 'None'}")


if __name__ == '__main__':
    test_tool_use()
