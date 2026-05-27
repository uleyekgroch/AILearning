"""
环境复杂度评估器

评估环境的复杂度，用于指导模型选择。

复杂度维度：
1. 物体多样性：不同属性的数量
2. 空间复杂度：物体分布的分散程度
3. 交互复杂度：可能的动作-结果映射数量
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter, deque


class EnvironmentComplexityEstimator:
    """
    环境复杂度评估器

    评估环境的复杂度，返回0-1之间的复杂度分数。

    评估维度：
    1. 属性多样性（颜色、形状、重量）
    2. 空间分散度
    3. 物体密度
    """

    def __init__(self):
        # 复杂度权重
        self.weights = {
            'attribute_diversity': 0.4,
            'spatial_spread': 0.3,
            'object_density': 0.3
        }

    def estimate(self, objects: List[Dict], grid_size: Tuple[int, int]) -> float:
        """
        估计环境复杂度

        参数：
            objects: 物体列表，每个物体是一个字典
            grid_size: 网格大小 (width, height)

        返回：
            复杂度分数，0-1之间
        """
        if not objects:
            return 0.0

        # 1. 属性多样性
        attribute_diversity = self._compute_attribute_diversity(objects)

        # 2. 空间分散度
        spatial_spread = self._compute_spatial_spread(objects, grid_size)

        # 3. 物体密度
        object_density = self._compute_object_density(objects, grid_size)

        # 加权求和
        complexity = (
            self.weights['attribute_diversity'] * attribute_diversity +
            self.weights['spatial_spread'] * spatial_spread +
            self.weights['object_density'] * object_density
        )

        return min(1.0, complexity)

    def _compute_attribute_diversity(self, objects: List[Dict]) -> float:
        """
        计算属性多样性

        多样性 = 不同属性值的数量 / 可能的属性值总数
        """
        if not objects:
            return 0.0

        # 统计不同属性
        colors = set()
        shapes = set()
        weights = set()

        for obj in objects:
            if 'color' in obj:
                colors.add(obj['color'])
            if 'shape' in obj:
                shapes.add(obj['shape'])
            if 'weight' in obj:
                weights.add(obj['weight'])

        # 计算多样性（归一化到0-1）
        # 假设最多有10种颜色、5种形状、5种重量
        color_diversity = len(colors) / 10.0
        shape_diversity = len(shapes) / 5.0
        weight_diversity = len(weights) / 5.0

        # 平均多样性
        diversity = (color_diversity + shape_diversity + weight_diversity) / 3.0

        return min(1.0, diversity)

    def _compute_spatial_spread(self, objects: List[Dict], grid_size: Tuple[int, int]) -> float:
        """
        计算空间分散度

        分散度 = 物体位置的标准差 / 网格大小
        """
        if len(objects) < 2:
            return 0.0

        # 提取物体位置
        positions = []
        for obj in objects:
            if 'x' in obj and 'y' in obj:
                positions.append((obj['x'], obj['y']))

        if len(positions) < 2:
            return 0.0

        # 计算位置的标准差
        positions = np.array(positions)
        std_x = np.std(positions[:, 0])
        std_y = np.std(positions[:, 1])

        # 归一化到网格大小
        normalized_std_x = std_x / grid_size[0]
        normalized_std_y = std_y / grid_size[1]

        # 平均分散度
        spread = (normalized_std_x + normalized_std_y) / 2.0

        return min(1.0, spread)

    def _compute_object_density(self, objects: List[Dict], grid_size: Tuple[int, int]) -> float:
        """
        计算物体密度

        密度 = 物体数量 / 网格面积
        """
        grid_area = grid_size[0] * grid_size[1]
        density = len(objects) / grid_area

        # 归一化到0-1（假设最大密度是0.5）
        normalized_density = density / 0.5

        return min(1.0, normalized_density)


class AdaptiveModelSelector:
    """
    自适应模型选择器

    根据环境复杂度选择合适的模型。

    策略：
    - 低复杂度（<0.3）：线性模型
    - 中复杂度（0.3-0.7）：神经网络
    - 高复杂度（>0.7）：自适应神经网络
    """

    def __init__(self):
        self.estimator = EnvironmentComplexityEstimator()

        # 模型类型
        self.model_types = ['linear', 'neural_network', 'adaptive_nn']

        # 复杂度阈值
        self.thresholds = {
            'low': 0.3,
            'high': 0.7
        }

        # 历史记录
        self.complexity_history = []
        self.selection_history = []

    def select_model(self, objects: List[Dict], grid_size: Tuple[int, int]) -> str:
        """
        选择模型

        参数：
            objects: 物体列表
            grid_size: 网格大小

        返回：
            模型类型
        """
        # 估计复杂度
        complexity = self.estimator.estimate(objects, grid_size)

        # 选择模型
        if complexity < self.thresholds['low']:
            model_type = 'linear'
        elif complexity < self.thresholds['high']:
            model_type = 'neural_network'
        else:
            model_type = 'adaptive_nn'

        # 记录
        self.complexity_history.append(complexity)
        self.selection_history.append(model_type)

        return model_type

    def get_selection_reason(self, objects: List[Dict], grid_size: Tuple[int, int]) -> Dict:
        """
        获取选择原因

        返回复杂度分数和选择的模型。
        """
        complexity = self.estimator.estimate(objects, grid_size)

        if complexity < self.thresholds['low']:
            model_type = 'linear'
            reason = "环境简单，线性模型足够"
        elif complexity < self.thresholds['high']:
            model_type = 'neural_network'
            reason = "环境中等复杂，需要神经网络"
        else:
            model_type = 'adaptive_nn'
            reason = "环境复杂，需要自适应神经网络"

        return {
            'complexity': complexity,
            'model_type': model_type,
            'reason': reason
        }


class ErrorDrivenSelector:
    """
    基于预测误差的模型选择器

    与 AdaptiveModelSelector 的区别：
    - 旧方案：从环境属性估计复杂度 → 选择模型（ disconnected from learning ）
    - 新方案：从预测误差判断模型是否足够 → 决定是否需要更多容量

    关键机制：
    1. 迟滞：上阈值（升级）和下阈值（降级）不同，防止振荡
    2. 冷却期：切换后 N 步内不再评估，让新模型有时间学习
    3. 滑动窗口：用最近 N 步的平均误差做判断，减少噪声
    """

    def __init__(self, switch_up_threshold: float = 0.12,
                 switch_down_threshold: float = 0.03,
                 cooldown_steps: int = 100,
                 window_size: int = 20):
        self.switch_up_threshold = switch_up_threshold
        self.switch_down_threshold = switch_down_threshold
        self.cooldown_steps = cooldown_steps
        self.window_size = window_size

        self.error_history = deque(maxlen=200)
        self.steps_since_switch = cooldown_steps  # 初始允许立即切换
        self.switch_count = 0
        self.switch_log = []  # [(step, from, to, avg_error), ...]

    def should_switch(self, current_error: float,
                      current_model_type: str) -> Tuple[bool, str]:
        """
        判断是否需要切换模型

        参数：
            current_error: 当前预测误差
            current_model_type: 当前模型类型 ('linear' 或 'neural_network')

        返回：
            (should_switch, target_model_type)
        """
        self.error_history.append(current_error)
        self.steps_since_switch += 1

        # 冷却期内不评估
        if self.steps_since_switch < self.cooldown_steps:
            return False, current_model_type

        # 数据不足
        if len(self.error_history) < self.window_size:
            return False, current_model_type

        # 计算滑动窗口平均误差
        recent = list(self.error_history)[-self.window_size:]
        avg_error = np.mean(recent)

        # 迟滞逻辑：只升级，不降级
        # 一旦升级到 NN，就留在那里——降级会导致从零开始的学习震荡
        if current_model_type == 'linear':
            if avg_error > self.switch_up_threshold:
                self._record_switch(current_model_type, 'neural_network', avg_error)
                return True, 'neural_network'

        return False, current_model_type

    def _record_switch(self, from_type: str, to_type: str, avg_error: float):
        """记录切换"""
        self.steps_since_switch = 0
        self.switch_count += 1
        self.switch_log.append({
            'from': from_type,
            'to': to_type,
            'avg_error': avg_error,
            'step': len(self.error_history),
        })

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'switch_count': self.switch_count,
            'steps_since_switch': self.steps_since_switch,
            'avg_error': float(np.mean(list(self.error_history)[-20:])) if self.error_history else 0.0,
            'switch_log': self.switch_log,
        }
