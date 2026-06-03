"""
丰富环境：更多的物体、更多的属性

这是为了测试学习体在更复杂环境中的学习能力。
类比：从一个只有几个玩具的房间 → 一个充满各种物体的幼儿园。
"""

import numpy as np
from typing import List
from environment import SimpleGridWorld, Object


def create_rich_world() -> SimpleGridWorld:
    """
    创建一个丰富的环境

    包含：
    - 更多物体（8个）
    - 更多颜色（4种）
    - 更多形状（3种）
    - 不同重量
    """
    env = SimpleGridWorld(12, 12)

    # 红色物体
    env.add_object(Object(0, 2, 2, 'red', 'circle', 1.0))
    env.add_object(Object(1, 3, 8, 'red', 'square', 1.5))

    # 蓝色物体
    env.add_object(Object(2, 5, 5, 'blue', 'square', 1.5))
    env.add_object(Object(3, 8, 2, 'blue', 'triangle', 0.8))

    # 绿色物体
    env.add_object(Object(4, 7, 3, 'green', 'triangle', 0.8))
    env.add_object(Object(5, 2, 9, 'green', 'circle', 1.2))

    # 黄色物体
    env.add_object(Object(6, 3, 7, 'yellow', 'circle', 1.2))
    env.add_object(Object(7, 9, 6, 'yellow', 'square', 1.0))

    return env


def create_learning_curriculum() -> List[dict]:
    """
    创建学习课程

    按照人类婴儿的学习顺序：
    1. 先学基本感知（颜色、形状）
    2. 再学简单命名（"红球"）
    3. 再学比较（"这个和那个一样"）
    4. 再学分类（"这些都是球"）
    """
    curriculum = [
        {
            'stage': 'sensorimotor',
            'focus': 'basic_perception',
            'goals': ['learn_colors', 'learn_shapes'],
            'duration': 100
        },
        {
            'stage': 'sensorimotor',
            'focus': 'object_permanence',
            'goals': ['track_objects', 'predict_movement'],
            'duration': 100
        },
        {
            'stage': 'pre_operational',
            'focus': 'symbolic_representation',
            'goals': ['name_objects', 'simple_combinations'],
            'duration': 150
        },
        {
            'stage': 'pre_operational',
            'focus': 'social_reference',
            'goals': ['follow_pointing', 'shared_attention'],
            'duration': 100
        }
    ]

    return curriculum
