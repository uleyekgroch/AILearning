"""工具使用 / 功能推理 — 基于可供性(Affordance)的物体描述

核心思想：
工具使用 = 理解物体可以服务于超越其外观的功能。
"棍子" 在目标是拿到高处东西时变成 "延伸器"。
当外观描述无法区分工具时，功能描述（affordance-based）提供通信优势，
"use"/"for" 标记从这种需要中涌现。

源自 mvl/grounding_tool_use.py，numpy→torch 转换，适配 DDD 架构。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch


@dataclass
class Tool:
    """工具 — 物理特征 + 功能属性(Affordance) + 目标有效性

    features: 外观特征（颜色、形状、大小）
    affordances: 功能集合（reach, hit, contain, cut, push, ...）
    effectiveness: 对不同目标的有效性评分 (goal_name -> 0~1)
    """
    name: str
    features: Dict[str, str]
    affordances: Set[str]
    effectiveness: Dict[str, float] = field(default_factory=dict)

    def to_symbols(self) -> List[str]:
        """外观特征符号列表"""
        return list(self.features.values())

    def to_functional_symbols(self) -> List[str]:
        """功能描述符号列表"""
        return list(self.affordances)

    def is_effective_for(self, goal_name: str) -> float:
        """对特定目标的有效性，未知目标返回 0.0"""
        return self.effectiveness.get(goal_name, 0.0)


@dataclass
class Goal:
    """目标 — 期望达到的世界状态

    required_affordance: 需要的功能类型
    difficulty: 难度 (0~1)
    requires_tool: 是否必须使用工具
    """
    name: str
    required_affordance: str
    difficulty: float = 0.5
    requires_tool: bool = True


# 预定义工具模板 (name, features, affordances, effectiveness)
TOOL_TEMPLATES = [
    ('stick',
     {'color': 'brown', 'shape': 'long', 'size': 'thin'},
     {'reach', 'push'},
     {'reach_object': 0.9, 'push_object': 0.7}),

    ('rock',
     {'color': 'gray', 'shape': 'round', 'size': 'heavy'},
     {'hit', 'break'},
     {'break_wall': 0.8, 'hit_target': 0.6}),

    ('bowl',
     {'color': 'red', 'shape': 'round', 'size': 'hollow'},
     {'contain', 'carry'},
     {'contain_water': 0.9, 'carry_food': 0.7}),

    ('rope',
     {'color': 'brown', 'shape': 'long', 'size': 'flexible'},
     {'reach', 'tie', 'pull'},
     {'reach_object': 0.7, 'tie_together': 0.9, 'pull_object': 0.8}),

    ('knife',
     {'color': 'silver', 'shape': 'thin', 'size': 'sharp'},
     {'cut', 'slice'},
     {'cut_rope': 0.9, 'slice_food': 0.8}),

    ('board',
     {'color': 'brown', 'shape': 'flat', 'size': 'wide'},
     {'bridge', 'cover', 'push'},
     {'bridge_gap': 0.8, 'cover_hole': 0.7, 'push_object': 0.5}),
]

# 预定义目标模板 (name, required_affordance, difficulty, requires_tool)
GOAL_TEMPLATES = [
    ('reach_object', 'reach', 0.3, True),
    ('break_wall', 'hit', 0.5, True),
    ('contain_water', 'contain', 0.2, True),
    ('cut_rope', 'cut', 0.4, True),
    ('bridge_gap', 'bridge', 0.6, True),
    ('push_object', 'push', 0.2, False),   # 可以不用工具
    ('tie_together', 'tie', 0.4, True),
    ('pull_object', 'pull', 0.3, True),
]


class ToolUseModule:
    """工具使用模块 — 基于可供性的功能推理

    核心能力：
    1. select_tool — 从工具列表中选择最适合目标的工具
    2. describe_functional — 用功能（而非外观）描述工具
    3. is_appearance_unique — 检查外观是否足以区分工具
    4. evaluate_plan — 评估工具选择方案的质量
    """

    def __init__(self):
        self._selection_history: List[Dict] = []

    def select_tool(self, tools: List[Tool], goal: Goal) -> Optional[int]:
        """选择最适合目标的工具

        策略：优先匹配 affordance，再按 effectiveness 排序。
        如果目标不需要工具且没有匹配工具，返回 None。
        """
        if not tools:
            return None

        best_idx = None
        best_score = -1.0

        for i, tool in enumerate(tools):
            # affordance 必须匹配
            if goal.required_affordance not in tool.affordances:
                continue

            # effectiveness 作为主要评分
            eff = tool.is_effective_for(goal.name)
            # 综合评分 = effectiveness + affordance 匹配数
            affordance_bonus = len(tool.affordances) * 0.05
            score = eff + affordance_bonus

            if score > best_score:
                best_score = score
                best_idx = i

        # 如果不需要工具，返回 None 让调用者决定
        if best_idx is None and not goal.requires_tool:
            return None

        self._selection_history.append({
            'goal': goal.name,
            'selected': best_idx,
            'score': best_score,
        })
        return best_idx

    def describe_functional(self, tool: Tool, goal: Goal) -> List[str]:
        """功能描述 — affordance + 外观特征

        优先使用目标所需的 affordance，再补充外观特征。
        """
        symbols = []

        # 优先添加与目标匹配的 affordance
        if goal.required_affordance in tool.affordances:
            symbols.append(goal.required_affordance)

        # 补充其他 affordance
        for aff in sorted(tool.affordances):
            if aff not in symbols:
                symbols.append(aff)

        # 补充外观特征（辅助识别）
        symbols.extend(tool.to_symbols())

        return symbols

    def is_appearance_unique(self, target: Tool,
                             tools: List[Tool]) -> bool:
        """检查目标工具的外观是否在工具列表中唯一

        如果其他工具有 >= 2 个特征与目标重叠，则外观不唯一。
        """
        target_features = set(target.to_symbols())
        for tool in tools:
            if tool is target:
                continue
            other_features = set(tool.to_symbols())
            overlap = len(target_features & other_features)
            if overlap >= 2:
                return False
        return True

    def evaluate_plan(self, tools: List[Tool], goal: Goal,
                      selected_idx: int) -> float:
        """评估工具选择方案的质量

        评分维度：
        1. affordance 匹配 (0 或 1)
        2. effectiveness (0~1)
        3. 外观区分度 (0~1, 外观不唯一时功能描述更重要)

        综合评分 = 0.4 * affordance_match + 0.4 * effectiveness + 0.2 * distinctiveness
        """
        if selected_idx < 0 or selected_idx >= len(tools):
            return 0.0

        tool = tools[selected_idx]

        # affordance 匹配
        affordance_match = 1.0 if goal.required_affordance in tool.affordances else 0.0

        # effectiveness
        effectiveness = tool.is_effective_for(goal.name)

        # 外观区分度
        appearance_unique = self.is_appearance_unique(tool, tools)
        distinctiveness = 1.0 if appearance_unique else 0.3

        return (
            0.4 * affordance_match
            + 0.4 * effectiveness
            + 0.2 * distinctiveness
        )

    def get_stats(self) -> dict:
        return {
            'selection_count': len(self._selection_history),
            'recent_selections': self._selection_history[-10:],
        }

    def save_state(self) -> dict:
        return {
            'selection_history': self._selection_history[-200:],
        }

    def load_state(self, state: dict) -> None:
        self._selection_history = state.get('selection_history', [])
