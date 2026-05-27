"""
教师模块：社会交互中的引导者

这是Vygotsky的"脚手架"和Tomasello的"共同意图性"的实现。

教师不是直接告诉答案，而是在学习者的最近发展区(ZPD)内提供支持。
随着学习者能力的增长，脚手架被逐步撤除。

类比：
- 婴儿的父母指着小狗说"狗狗"
- 鹦鹉Alex的训练者展示"绿色钥匙"的正确命名
- 不是RLHF的二元偏好信号，而是真正的社会学习
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TeachingEpisode:
    """一次教学事件"""
    teacher_action: str
    target_object: Optional[Dict]
    learner_response: Optional[str]
    success: bool
    timestamp: int


class SimpleTeacher:
    """
    简单教师Agent

    实现基本的社会学习机制：
    1. 共同注意：指向物体，引导学习者注意力
    2. 命名：告诉学习者物体的名称
    3. 示范：展示如何与物体交互
    4. 脚手架：在学习者ZPD内提供支持

    这不是RLHF（二元偏好信号），
    而是Vygotsky意义上的脚手架。
    """

    def __init__(self):
        # 教师的知识库（预定义的物体知识）
        self.knowledge = {
            'red_circle': {'color': 'red', 'shape': 'circle', 'name': '红球'},
            'blue_square': {'color': 'blue', 'shape': 'square', 'name': '蓝方块'},
            'green_triangle': {'color': 'green', 'shape': 'triangle', 'name': '绿三角'},
            'yellow_circle': {'color': 'yellow', 'shape': 'circle', 'name': '黄球'},
        }

        # 教学历史
        self.teaching_history: List[TeachingEpisode] = []
        self.step_count = 0

        # 教学策略参数
        self.naming_probability = 0.3  # 命名概率
        self.demonstration_probability = 0.2  # 示范概率

        # 共同注意管理器
        self.joint_attention = JointAttentionManager()

        # 脚手架级别：1.0 = 最大支持, 0.0 = 无支持
        self.scaffold_level = 1.0

        # 上一次示范的动作（供 learner 模仿）
        self.last_demonstration_action = None

    def observe(self, env_observation: dict) -> Optional[Dict]:
        """
        观察环境

        教师观察环境中的物体，
        决定是否进行教学。
        """
        visible_objects = env_observation.get('visible_objects', [])

        if not visible_objects:
            return None

        # 选择最近的物体作为教学目标
        closest = min(visible_objects, key=lambda x: x['distance'])
        return closest

    def update_scaffold_level(self, grounded_symbols: int):
        """
        更新脚手架级别

        脚手架渐退：随着 learner 掌握更多符号，教师减少支持。
        scaffold_level = 1.0（0个符号）→ 0.0（10+个符号）
        """
        self.scaffold_level = max(0.0, 1.0 - grounded_symbols / 10.0)

    def decide_teaching_action(self, learner_stats: Dict,
                                target_object: Optional[Dict]) -> Optional[str]:
        """
        决定教学行动

        根据学习者的状态和脚手架级别选择教学策略。
        脚手架高时：频繁命名、明确指向
        脚手架低时：减少命名，增加组合/比较教学
        """
        if target_object is None:
            return None

        self.step_count += 1

        # 更新脚手架级别
        grounded_symbols = learner_stats.get('grounded_symbols', 0)
        self.update_scaffold_level(grounded_symbols)

        # 根据脚手架级别调整教学概率
        s = self.scaffold_level

        # 概率分布随脚手架级别变化
        prob_name = 0.05 + 0.45 * s          # 高脚手架时频繁命名
        prob_attend = 0.1 + 0.2 * s           # 共同注意
        prob_demo = 0.05 + 0.15 * s           # 示范
        prob_compose = 0.1 + 0.2 * (1 - s)    # 低脚手架时增加组合
        prob_compare = 0.05 + 0.25 * (1 - s)  # 低脚手架时增加比较

        # 归一化
        total = prob_name + prob_attend + prob_demo + prob_compose + prob_compare
        probs = [prob_name, prob_attend, prob_demo, prob_compose, prob_compare]
        probs = [p / total for p in probs]

        actions = ['name', 'attend', 'demonstrate', 'compose', 'compare']
        return np.random.choice(actions, p=probs)

    def execute_teaching(self, action: str, target_object: Dict,
                          learner) -> TeachingEpisode:
        """
        执行教学行动

        将教学决策转化为实际的交互。
        包含：共同注意建立、社会交互记录、模仿学习支持。
        """
        episode = TeachingEpisode(
            teacher_action=action,
            target_object=target_object,
            learner_response=None,
            success=False,
            timestamp=self.step_count
        )

        obj = target_object.get('object')
        if obj is None:
            return episode

        if action == 'attend':
            # 共同注意：教师指向物体，建立双向注意力
            self.joint_attention.establish(self, learner, target_object)
            learner.record_social_interaction()
            episode.success = True

        elif action == 'name':
            # 命名：告诉学习者物体的名称
            name = self._get_object_name(obj)
            if name:
                obs = obj.to_observation()
                learner.grounding.ground_from_social(name, obs, 'naming')
                learner.record_social_interaction()
                episode.learner_response = name
                episode.success = True

        elif action == 'demonstrate':
            # 示范：展示如何与物体交互
            # 计算示范动作（推动物体朝向目标方向）
            demo_action = self._get_demonstration_action(obj, learner)
            self.last_demonstration_action = demo_action
            learner.observe_teacher_demo(demo_action)
            learner.record_social_interaction()
            episode.success = True

        elif action == 'compose':
            # 组合：教学习者组合概念
            color_name = self._get_color_name(getattr(obj, 'color', None))
            shape_name = self._get_shape_name(getattr(obj, 'shape', None))
            if color_name and shape_name:
                composite = f"{color_name}{shape_name}"
                obs = obj.to_observation()
                learner.grounding.ground_from_social(composite, obs, 'composition')
                learner.record_social_interaction()
                episode.learner_response = composite
                episode.success = True

        elif action == 'compare':
            # 比较：教学习者比较不同物体
            learner.record_social_interaction()
            episode.success = True

        self.teaching_history.append(episode)
        return episode

    def _get_demonstration_action(self, obj, learner) -> int:
        """
        计算示范动作

        教师展示"正确的"动作：
        如果物体在 learner 上方 → 动作4（跳跃）
        如果物体在 learner 右侧 → 动作3（右移）
        如果物体在 learner 前方 → 动作0（前进）
        默认 → 动作6（推动）
        """
        # 获取 agent 位置（可能是 agent_x 或从 observation 中获取）
        agent_x = getattr(learner, 'agent_x', 5.0)
        agent_y = getattr(learner, 'agent_y', 5.0)
        agent_z = getattr(learner, 'agent_z', 1.0)

        dx = obj.x - agent_x
        dy = obj.y - agent_y
        dz = getattr(obj, 'z', 0.0) - agent_z

        # 选择最主要的方向
        if abs(dz) > abs(dx) and abs(dz) > abs(dy):
            return 4 if dz > 0 else 5  # 跳跃或下蹲
        elif abs(dx) > abs(dy):
            return 3 if dx > 0 else 2  # 右移或左移
        else:
            return 0 if dy > 0 else 1  # 前进或后退

    def get_demonstration_action(self) -> Optional[int]:
        """获取最近一次示范的动作（供外部使用）"""
        return self.last_demonstration_action

    def _get_object_name(self, obj) -> Optional[str]:
        """获取物体的名称"""
        color = getattr(obj, 'color', None)
        shape = getattr(obj, 'shape', None)
        for key, knowledge in self.knowledge.items():
            if (knowledge['color'] == color and
                knowledge['shape'] == shape):
                return knowledge['name']
        # 对于 PhysicsObjectAdapter（只有颜色，没有形状），
        # 用颜色作为名称
        if color and not shape:
            color_names = {'red': '红', 'blue': '蓝', 'green': '绿',
                          'yellow': '黄', 'brown': '棕', 'gray': '灰',
                          'black': '黑', 'white': '白'}
            return color_names.get(color, None)
        return None

    def _get_color_name(self, color: str) -> Optional[str]:
        """获取颜色的名称"""
        color_names = {
            'red': '红',
            'blue': '蓝',
            'green': '绿',
            'yellow': '黄'
        }
        return color_names.get(color)

    def _get_shape_name(self, shape: str) -> Optional[str]:
        """获取形状的名称"""
        shape_names = {
            'circle': '球',
            'square': '方块',
            'triangle': '三角'
        }
        return shape_names.get(shape)

    def get_teaching_stats(self) -> Dict:
        """获取教学统计"""
        if not self.teaching_history:
            return {'total_episodes': 0}

        total = len(self.teaching_history)
        successful = sum(1 for e in self.teaching_history if e.success)

        action_counts = {}
        for episode in self.teaching_history:
            action = episode.teacher_action
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            'total_episodes': total,
            'successful_episodes': successful,
            'success_rate': successful / total if total > 0 else 0,
            'action_distribution': action_counts
        }


class JointAttentionManager:
    """
    共同注意管理器

    共同注意 = 两个个体共同关注同一物体，
    并意识到彼此的注意力状态。

    这是人类独特认知能力的基础——
    不只是"看同一个东西"，
    而是"意识到我们都在看同一个东西"。
    """

    def __init__(self):
        self.current_focus = None
        self.attention_history = []

    def establish(self, teacher, learner, target_object):
        """建立共同注意"""
        self.current_focus = {
            'teacher': teacher,
            'learner': learner,
            'target': target_object,
            'established': True
        }
        self.attention_history.append(self.current_focus)

    def get_shared_focus(self) -> Optional[Dict]:
        """获取当前共享焦点"""
        return self.current_focus

    def is_attending(self) -> bool:
        """是否正在共同注意"""
        return self.current_focus is not None and self.current_focus['established']
