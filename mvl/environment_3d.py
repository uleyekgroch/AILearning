"""
3D物理环境：更复杂的世界

这是从2D网格世界到3D物理世界的升级。
类比：从一个平面的游戏板 → 一个充满物体的真实房间。

核心改进：
1. 3D空间：增加Z轴（高度）
2. 物理效果：重力、碰撞
3. 多模态感知：视觉、触觉、距离
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ObjectShape(Enum):
    """物体形状"""
    CUBE = "cube"
    SPHERE = "sphere"
    PYRAMID = "pyramid"


@dataclass
class Object3D:
    """3D物体"""
    id: int
    x: float
    y: float
    z: float
    color: str
    shape: ObjectShape
    size: float  # 大小（半径或边长的一半）
    weight: float  # 重量
    is_movable: bool = True  # 是否可移动

    def to_observation(self) -> np.ndarray:
        """
        转换为观测向量

        包含：位置(3) + 颜色编码(4) + 形状编码(3) + 大小(1) + 重量(1) = 12维
        """
        # 位置
        pos = np.array([self.x, self.y, self.z])

        # 颜色编码（one-hot）
        color_map = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3}
        color_vec = np.zeros(4)
        if self.color in color_map:
            color_vec[color_map[self.color]] = 1.0

        # 形状编码（one-hot）
        shape_map = {ObjectShape.CUBE: 0, ObjectShape.SPHERE: 1, ObjectShape.PYRAMID: 2}
        shape_vec = np.zeros(3)
        if self.shape in shape_map:
            shape_vec[shape_map[self.shape]] = 1.0

        # 大小和重量
        size_weight = np.array([self.size, self.weight])

        return np.concatenate([pos, color_vec, shape_vec, size_weight])


class PhysicsEngine:
    """
    简单的物理引擎

    实现基本的物理效果：
    1. 重力：物体向下掉落
    2. 碰撞检测：物体不能重叠
    3. 边界限制：物体不能超出环境边界
    """

    def __init__(self, gravity: float = 0.1, damping: float = 0.9):
        self.gravity = gravity
        self.damping = damping  # 阻尼系数

    def apply_gravity(self, obj: Object3D, ground_level: float = 0.0):
        """应用重力"""
        if obj.z > ground_level + obj.size:
            obj.z -= self.gravity
            # 确保不会穿过地面
            if obj.z < ground_level + obj.size:
                obj.z = ground_level + obj.size

    def check_collision(self, obj1: Object3D, obj2: Object3D) -> bool:
        """检查两个物体是否碰撞"""
        # 计算距离
        dx = obj1.x - obj2.x
        dy = obj1.y - obj2.y
        dz = obj1.z - obj2.z
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        # 如果距离小于两个物体的半径之和，则碰撞
        min_distance = obj1.size + obj2.size
        return distance < min_distance

    def resolve_collision(self, obj1: Object3D, obj2: Object3D):
        """解决碰撞"""
        # 计算方向向量
        dx = obj1.x - obj2.x
        dy = obj1.y - obj2.y
        dz = obj1.z - obj2.z
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        if distance == 0:
            return

        # 归一化方向向量
        dx /= distance
        dy /= distance
        dz /= distance

        # 计算分离距离
        min_distance = obj1.size + obj2.size
        separation = min_distance - distance

        # 分离物体（根据重量分配）
        total_weight = obj1.weight + obj2.weight
        if total_weight > 0:
            ratio1 = obj2.weight / total_weight
            ratio2 = obj1.weight / total_weight
        else:
            ratio1 = 0.5
            ratio2 = 0.5

        if obj1.is_movable:
            obj1.x += dx * separation * ratio1
            obj1.y += dy * separation * ratio1
            obj1.z += dz * separation * ratio1

        if obj2.is_movable:
            obj2.x -= dx * separation * ratio2
            obj2.y -= dy * separation * ratio2
            obj2.z -= dz * separation * ratio2

    def check_boundary(self, obj: Object3D, boundary: Tuple[float, float, float]):
        """检查并修正边界"""
        max_x, max_y, max_z = boundary

        # X轴边界
        if obj.x < obj.size:
            obj.x = obj.size
        elif obj.x > max_x - obj.size:
            obj.x = max_x - obj.size

        # Y轴边界
        if obj.y < obj.size:
            obj.y = obj.size
        elif obj.y > max_y - obj.size:
            obj.y = max_y - obj.size

        # Z轴边界（地面）
        if obj.z < obj.size:
            obj.z = obj.size
        elif obj.z > max_z - obj.size:
            obj.z = max_z - obj.size


class GridWorld3D:
    """
    3D网格世界

    从2D到3D的升级：
    1. 增加Z轴（高度）
    2. 物理效果（重力、碰撞）
    3. 多模态感知（视觉、触觉、距离）
    """

    def __init__(self, width: int = 10, height: int = 10, depth: int = 10):
        self.width = width
        self.height = height
        self.depth = depth

        # 物体列表
        self.objects: List[Object3D] = []

        # Agent位置
        self.agent_x = width / 2
        self.agent_y = height / 2
        self.agent_z = 1.0  # Agent站在地面上

        # 物理引擎
        self.physics = PhysicsEngine()

        # 步数
        self.step_count = 0

    def add_object(self, obj: Object3D):
        """添加物体"""
        self.objects.append(obj)

    def get_observation(self) -> Dict:
        """
        获取观测

        包含多种模态：
        1. 视觉：可见物体的特征
        2. 触觉：是否接触物体
        3. 距离：与物体的距离
        """
        # Agent位置
        agent_pos = np.array([self.agent_x, self.agent_y, self.agent_z])

        # 可见物体（在视野范围内）
        visible_objects = []
        touch_objects = []
        distances = []

        for obj in self.objects:
            # 计算距离
            dx = obj.x - self.agent_x
            dy = obj.y - self.agent_y
            dz = obj.z - self.agent_z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)

            distances.append(distance)

            # 视觉感知：距离小于视野范围
            vision_range = 5.0
            if distance < vision_range:
                visible_objects.append({
                    'object': obj,
                    'distance': distance,
                    'direction': np.array([dx, dy, dz]) / max(distance, 0.001)
                })

            # 触觉感知：距离小于接触范围
            touch_range = 1.5
            if distance < touch_range:
                touch_objects.append({
                    'object': obj,
                    'distance': distance,
                    'is_touching': distance < (obj.size + 0.5)
                })

        return {
            'agent_position': agent_pos,
            'visible_objects': visible_objects,
            'touch_objects': touch_objects,
            'distances': distances,
            'step_count': self.step_count
        }

    def step(self, action: int) -> Tuple[Dict, float, bool]:
        """
        执行动作

        动作：
        0: 前进 (y+)
        1: 后退 (y-)
        2: 左移 (x-)
        3: 右移 (x+)
        4: 跳跃 (z+)
        5: 下蹲 (z-)
        6: 推动（向前推物体）
        7: 拉动（向后拉物体）
        """
        # 移动速度
        move_speed = 1.0
        jump_speed = 2.0

        # 执行动作
        if action == 0:  # 前进
            self.agent_y = min(self.height - 1, self.agent_y + move_speed)
        elif action == 1:  # 后退
            self.agent_y = max(0, self.agent_y - move_speed)
        elif action == 2:  # 左移
            self.agent_x = max(0, self.agent_x - move_speed)
        elif action == 3:  # 右移
            self.agent_x = min(self.width - 1, self.agent_x + move_speed)
        elif action == 4:  # 跳跃
            self.agent_z = min(self.depth - 1, self.agent_z + jump_speed)
        elif action == 5:  # 下蹲
            self.agent_z = max(1.0, self.agent_z - move_speed)
        elif action == 6:  # 推动
            self._push_objects()
        elif action == 7:  # 拉动
            self._pull_objects()

        # 应用物理效果
        self._apply_physics()

        # 更新步数
        self.step_count += 1

        # 计算奖励（基于好奇心）
        reward = self._compute_reward()

        # 检查是否结束
        done = self.step_count >= 1000

        return self.get_observation(), reward, done

    def _push_objects(self):
        """推动前方的物体"""
        push_range = 2.0
        push_force = 1.0

        for obj in self.objects:
            if not obj.is_movable:
                continue

            # 计算距离
            dx = obj.x - self.agent_x
            dy = obj.y - self.agent_y
            dz = obj.z - self.agent_z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)

            # 如果在推动范围内
            if distance < push_range:
                # 计算推动方向（向前）
                push_direction = np.array([0, 1, 0])  # 向前推

                # 应用推力
                obj.x += push_direction[0] * push_force
                obj.y += push_direction[1] * push_force
                obj.z += push_direction[2] * push_force

    def _pull_objects(self):
        """拉动后方的物体"""
        pull_range = 2.0
        pull_force = 1.0

        for obj in self.objects:
            if not obj.is_movable:
                continue

            # 计算距离
            dx = obj.x - self.agent_x
            dy = obj.y - self.agent_y
            dz = obj.z - self.agent_z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)

            # 如果在拉动范围内
            if distance < pull_range:
                # 计算拉动方向（向后）
                pull_direction = np.array([0, -1, 0])  # 向后拉

                # 应用拉力
                obj.x += pull_direction[0] * pull_force
                obj.y += pull_direction[1] * pull_force
                obj.z += pull_direction[2] * pull_force

    def _apply_physics(self):
        """应用物理效果"""
        # 对每个物体应用物理
        for obj in self.objects:
            # 重力
            self.physics.apply_gravity(obj, ground_level=0.0)

            # 边界检查
            self.physics.check_boundary(obj, (self.width, self.height, self.depth))

        # 碰撞检测
        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                if self.physics.check_collision(self.objects[i], self.objects[j]):
                    self.physics.resolve_collision(self.objects[i], self.objects[j])

    def _compute_reward(self) -> float:
        """计算奖励（基于好奇心）"""
        # 简单的奖励：探索新区域
        # 这里可以集成好奇心模块
        return 0.0

    def reset(self):
        """重置环境"""
        self.agent_x = self.width / 2
        self.agent_y = self.height / 2
        self.agent_z = 1.0
        self.step_count = 0

        # 重置物体位置
        for obj in self.objects:
            # 随机位置
            obj.x = np.random.uniform(obj.size, self.width - obj.size)
            obj.y = np.random.uniform(obj.size, self.height - obj.size)
            obj.z = np.random.uniform(obj.size, self.depth - obj.size)

    def render(self):
        """渲染环境（简单文本输出）"""
        print(f"\n3D世界 ({self.width}x{self.height}x{self.depth})")
        print(f"Agent位置: ({self.agent_x:.1f}, {self.agent_y:.1f}, {self.agent_z:.1f})")
        print(f"物体数量: {len(self.objects)}")
        for obj in self.objects:
            print(f"  {obj.color} {obj.shape.value} at ({obj.x:.1f}, {obj.y:.1f}, {obj.z:.1f})")


def create_simple_3d_world() -> GridWorld3D:
    """创建一个简单的3D世界"""
    env = GridWorld3D(10, 10, 10)

    # 添加一些物体
    env.add_object(Object3D(0, 2, 2, 1, 'red', ObjectShape.SPHERE, 0.5, 1.0))
    env.add_object(Object3D(1, 5, 5, 1, 'blue', ObjectShape.CUBE, 0.5, 1.5))
    env.add_object(Object3D(2, 8, 3, 1, 'green', ObjectShape.PYRAMID, 0.5, 0.8))
    env.add_object(Object3D(3, 3, 8, 2, 'yellow', ObjectShape.SPHERE, 0.5, 1.2))

    return env


def create_rich_3d_world() -> GridWorld3D:
    """创建一个丰富的3D世界"""
    env = GridWorld3D(12, 12, 12)

    # 红色物体
    env.add_object(Object3D(0, 2, 2, 1, 'red', ObjectShape.SPHERE, 0.5, 1.0))
    env.add_object(Object3D(1, 3, 8, 2, 'red', ObjectShape.CUBE, 0.6, 1.5))

    # 蓝色物体
    env.add_object(Object3D(2, 5, 5, 1, 'blue', ObjectShape.CUBE, 0.5, 1.5))
    env.add_object(Object3D(3, 8, 2, 3, 'blue', ObjectShape.PYRAMID, 0.4, 0.8))

    # 绿色物体
    env.add_object(Object3D(4, 7, 3, 1, 'green', ObjectShape.PYRAMID, 0.5, 0.8))
    env.add_object(Object3D(5, 2, 9, 2, 'green', ObjectShape.SPHERE, 0.6, 1.2))

    # 黄色物体
    env.add_object(Object3D(6, 3, 7, 1, 'yellow', ObjectShape.SPHERE, 0.5, 1.2))
    env.add_object(Object3D(7, 9, 6, 2, 'yellow', ObjectShape.CUBE, 0.5, 1.0))

    return env
