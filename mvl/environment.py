"""
环境模块：2D网格世界

这是学习体的"身体"所在地。
不是静态数据集，而是可以交互的动态世界。
学习体通过行动改变环境，环境通过反馈改变学习体。

关键：行动-感知闭环
agent.act() → environment.step() → agent.perceive()
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Object:
    """可交互物体"""
    id: int
    x: int
    y: int
    color: str      # 颜色属性
    shape: str      # 形状属性
    weight: float   # 重量属性（影响推动难度）

    def to_observation(self) -> np.ndarray:
        """将物体转换为观测向量"""
        # 颜色编码 (one-hot)
        colors = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3}
        color_vec = np.zeros(4)
        if self.color in colors:
            color_vec[colors[self.color]] = 1.0

        # 形状编码 (one-hot)
        shapes = {'circle': 0, 'square': 1, 'triangle': 2}
        shape_vec = np.zeros(3)
        if self.shape in shapes:
            shape_vec[shapes[self.shape]] = 1.0

        # 位置和重量
        pos_weight = np.array([self.x / 10.0, self.y / 10.0, self.weight])

        return np.concatenate([color_vec, shape_vec, pos_weight])


class SimpleGridWorld:
    """
    简单的2D网格世界

    学习体可以：
    - 移动（上下左右）
    - 观察（看到附近的物体）
    - 交互（推动物体）

    环境提供：
    - 物体的位置和属性
    - 物理反馈（碰撞、推动）
    - 感知反馈（视野内的物体）
    """

    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.objects: List[Object] = []
        self.agent_x = width // 2
        self.agent_y = height // 2
        self.step_count = 0
        self.max_steps = 1000

    def add_object(self, obj: Object):
        """添加物体到环境"""
        self.objects.append(obj)

    def get_observation(self) -> dict:
        """
        获取当前观测

        观测包含：
        - agent位置
        - 视野内的物体（距离<3的物体）
        """
        visible_objects = []
        for obj in self.objects:
            dist = abs(obj.x - self.agent_x) + abs(obj.y - self.agent_y)
            if dist <= 3:  # 视野范围
                visible_objects.append({
                    'object': obj,
                    'relative_x': obj.x - self.agent_x,
                    'relative_y': obj.y - self.agent_y,
                    'distance': dist
                })

        return {
            'agent_position': np.array([self.agent_x / self.width,
                                       self.agent_y / self.height]),
            'visible_objects': visible_objects,
            'step': self.step_count
        }

    def step(self, action: int) -> Tuple[dict, float, bool]:
        """
        执行动作

        动作编码：
        0: 上移
        1: 下移
        2: 左移
        3: 右移
        4: 推（面向方向推动物体）

        返回：(新观测, 奖励, 是否结束)
        """
        self.step_count += 1

        # 执行移动
        dx, dy = 0, 0
        if action == 0:  # 上
            dy = -1
        elif action == 1:  # 下
            dy = 1
        elif action == 2:  # 左
            dx = -1
        elif action == 3:  # 右
            dx = 1

        # 尝试移动
        new_x = self.agent_x + dx
        new_y = self.agent_y + dy

        # 边界检查
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            # 检查是否与物体碰撞
            collision = False
            for obj in self.objects:
                if obj.x == new_x and obj.y == new_y:
                    collision = True
                    # 如果是推的动作，尝试推动物体
                    if action == 4:
                        push_x = obj.x + dx
                        push_y = obj.y + dy
                        if (0 <= push_x < self.width and
                            0 <= push_y < self.height):
                            # 检查推动位置是否空闲
                            blocked = False
                            for other in self.objects:
                                if other.x == push_x and other.y == push_y:
                                    blocked = True
                                    break
                            if not blocked:
                                obj.x = push_x
                                obj.y = push_y
                                self.agent_x = new_x
                                self.agent_y = new_y
                    break

            if not collision:
                self.agent_x = new_x
                self.agent_y = new_y

        # 获取新观测
        obs = self.get_observation()

        # 计算奖励（默认为0，好奇心驱动的学习不需要外在奖励）
        reward = 0.0

        # 检查是否结束
        done = self.step_count >= self.max_steps

        return obs, reward, done

    def reset(self) -> dict:
        """重置环境"""
        self.agent_x = self.width // 2
        self.agent_y = self.height // 2
        self.step_count = 0
        return self.get_observation()

    def render(self) -> str:
        """渲染环境为文本"""
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]

        # 放置物体
        for obj in self.objects:
            color_symbol = {'red': 'R', 'blue': 'B', 'green': 'G', 'yellow': 'Y'}
            shape_symbol = {'circle': 'o', 'square': 's', 'triangle': 't'}
            symbol = color_symbol.get(obj.color, '?')[0] + shape_symbol.get(obj.shape, '?')[0]
            if 0 <= obj.x < self.width and 0 <= obj.y < self.height:
                grid[obj.y][obj.x] = symbol

        # 放置agent
        if 0 <= self.agent_x < self.width and 0 <= self.agent_y < self.height:
            grid[self.agent_y][self.agent_x] = 'A'

        # 转换为字符串
        lines = []
        for row in grid:
            lines.append(' '.join(row))
        return '\n'.join(lines)


def create_simple_world() -> SimpleGridWorld:
    """创建一个简单的测试世界"""
    env = SimpleGridWorld(10, 10)

    # 添加一些物体
    env.add_object(Object(0, 2, 2, 'red', 'circle', 1.0))
    env.add_object(Object(1, 5, 5, 'blue', 'square', 1.5))
    env.add_object(Object(2, 7, 3, 'green', 'triangle', 0.8))
    env.add_object(Object(3, 3, 7, 'yellow', 'circle', 1.2))

    return env
