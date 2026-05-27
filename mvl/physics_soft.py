"""
软体物理引擎

实现软体物体的物理效果：
1. 布料模拟
2. 绳子模拟
3. 软体碰撞和变形
4. 弹簧-质点系统

这是从刚体物理到软体物理的升级。
类比：从一个固体世界 → 一个有布料、绳子、软物体的世界。
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MassPoint:
    """质点"""
    x: float
    y: float
    z: float
    mass: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    is_fixed: bool = False  # 是否固定


@dataclass
class Spring:
    """弹簧"""
    point1_idx: int  # 质点1的索引
    point2_idx: int  # 质点2的索引
    rest_length: float  # 自然长度
    stiffness: float = 100.0  # 刚度
    damping: float = 1.0  # 阻尼


class Cloth:
    """
    布料模拟

    使用弹簧-质点系统模拟布料。
    """

    def __init__(self, width: float, height: float,
                 num_points_x: int = 10, num_points_y: int = 10,
                 mass: float = 1.0, stiffness: float = 100.0):
        self.width = width
        self.height = height
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y

        # 创建质点
        self.points: List[MassPoint] = []
        for i in range(num_points_x):
            for j in range(num_points_y):
                x = i * width / (num_points_x - 1)
                y = j * height / (num_points_y - 1)
                z = 0.0
                point = MassPoint(x, y, z, mass / (num_points_x * num_points_y))
                self.points.append(point)

        # 创建弹簧
        self.springs: List[Spring] = []

        # 结构弹簧（连接相邻质点）
        for i in range(num_points_x):
            for j in range(num_points_y):
                idx = i * num_points_y + j

                # 右邻居
                if i < num_points_x - 1:
                    right_idx = (i + 1) * num_points_y + j
                    rest_length = width / (num_points_x - 1)
                    self.springs.append(Spring(idx, right_idx, rest_length, stiffness))

                # 下邻居
                if j < num_points_y - 1:
                    down_idx = i * num_points_y + (j + 1)
                    rest_length = height / (num_points_y - 1)
                    self.springs.append(Spring(idx, down_idx, rest_length, stiffness))

                # 对角线弹簧（增加稳定性）
                if i < num_points_x - 1 and j < num_points_y - 1:
                    diag_idx = (i + 1) * num_points_y + (j + 1)
                    rest_length = np.sqrt((width / (num_points_x - 1))**2 +
                                         (height / (num_points_y - 1))**2)
                    self.springs.append(Spring(idx, diag_idx, rest_length, stiffness * 0.5))

        # 固定左上角的点
        self.points[0].is_fixed = True
        self.points[num_points_y - 1].is_fixed = True

    def update(self, dt: float, gravity: float = 9.81):
        """更新布料模拟"""
        # 应用重力
        for point in self.points:
            if not point.is_fixed:
                point.vz -= gravity * dt

        # 计算弹簧力
        for spring in self.springs:
            p1 = self.points[spring.point1_idx]
            p2 = self.points[spring.point2_idx]

            # 计算距离
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            dz = p2.z - p1.z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)

            if distance < 0.001:
                continue

            # 计算弹簧力
            displacement = distance - spring.rest_length
            force_magnitude = spring.stiffness * displacement

            # 计算阻尼力
            relative_velocity = np.array([
                p2.vx - p1.vx,
                p2.vy - p1.vy,
                p2.vz - p1.vz
            ])
            direction = np.array([dx, dy, dz]) / distance
            damping_force = spring.damping * np.dot(relative_velocity, direction)

            # 总力
            total_force = (force_magnitude + damping_force) * direction

            # 应用力
            if not p1.is_fixed:
                p1.vx += total_force[0] / p1.mass * dt
                p1.vy += total_force[1] / p1.mass * dt
                p1.vz += total_force[2] / p1.mass * dt
            if not p2.is_fixed:
                p2.vx -= total_force[0] / p2.mass * dt
                p2.vy -= total_force[1] / p2.mass * dt
                p2.vz -= total_force[2] / p2.mass * dt

        # 更新位置
        for point in self.points:
            if not point.is_fixed:
                point.x += point.vx * dt
                point.y += point.vy * dt
                point.z += point.vz * dt

                # 地面碰撞
                if point.z < 0:
                    point.z = 0
                    point.vz = -point.vz * 0.5  # 反弹

                # 应用阻尼
                point.vx *= 0.99
                point.vy *= 0.99
                point.vz *= 0.99

    def get_state(self) -> List[dict]:
        """获取布料状态"""
        states = []
        for point in self.points:
            states.append({
                'position': (point.x, point.y, point.z),
                'velocity': (point.vx, point.vy, point.vz),
                'mass': point.mass,
                'is_fixed': point.is_fixed
            })
        return states


class Rope:
    """
    绳子模拟

    使用弹簧-质点系统模拟绳子。
    """

    def __init__(self, length: float, num_points: int = 20,
                 mass: float = 1.0, stiffness: float = 200.0):
        self.length = length
        self.num_points = num_points

        # 创建质点
        self.points: List[MassPoint] = []
        for i in range(num_points):
            x = i * length / (num_points - 1)
            y = 0.0
            z = 0.0
            point = MassPoint(x, y, z, mass / num_points)
            self.points.append(point)

        # 创建弹簧
        self.springs: List[Spring] = []
        for i in range(num_points - 1):
            rest_length = length / (num_points - 1)
            self.springs.append(Spring(i, i + 1, rest_length, stiffness))

        # 固定第一个点
        self.points[0].is_fixed = True

    def update(self, dt: float, gravity: float = 9.81):
        """更新绳子模拟"""
        # 应用重力
        for point in self.points:
            if not point.is_fixed:
                point.vz -= gravity * dt

        # 计算弹簧力
        for spring in self.springs:
            p1 = self.points[spring.point1_idx]
            p2 = self.points[spring.point2_idx]

            # 计算距离
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            dz = p2.z - p1.z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)

            if distance < 0.001:
                continue

            # 计算弹簧力
            displacement = distance - spring.rest_length
            force_magnitude = spring.stiffness * displacement

            # 计算阻尼力
            relative_velocity = np.array([
                p2.vx - p1.vx,
                p2.vy - p1.vy,
                p2.vz - p1.vz
            ])
            direction = np.array([dx, dy, dz]) / distance
            damping_force = spring.damping * np.dot(relative_velocity, direction)

            # 总力
            total_force = (force_magnitude + damping_force) * direction

            # 应用力
            if not p1.is_fixed:
                p1.vx += total_force[0] / p1.mass * dt
                p1.vy += total_force[1] / p1.mass * dt
                p1.vz += total_force[2] / p1.mass * dt
            if not p2.is_fixed:
                p2.vx -= total_force[0] / p2.mass * dt
                p2.vy -= total_force[1] / p2.mass * dt
                p2.vz -= total_force[2] / p2.mass * dt

        # 更新位置
        for point in self.points:
            if not point.is_fixed:
                point.x += point.vx * dt
                point.y += point.vy * dt
                point.z += point.vz * dt

                # 地面碰撞
                if point.z < 0:
                    point.z = 0
                    point.vz = -point.vz * 0.5  # 反弹

                # 应用阻尼
                point.vx *= 0.99
                point.vy *= 0.99
                point.vz *= 0.99

    def get_state(self) -> List[dict]:
        """获取绳子状态"""
        states = []
        for point in self.points:
            states.append({
                'position': (point.x, point.y, point.z),
                'velocity': (point.vx, point.vy, point.vz),
                'mass': point.mass,
                'is_fixed': point.is_fixed
            })
        return states


class SoftBody:
    """
    软体物体

    使用弹簧-质点系统模拟软体。
    """

    def __init__(self, center_x: float, center_y: float, center_z: float,
                 radius: float = 1.0, num_points: int = 20,
                 mass: float = 1.0, stiffness: float = 150.0):
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.radius = radius

        # 创建质点（随机分布在球面上）
        self.points: List[MassPoint] = []
        for i in range(num_points):
            # 随机方向
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)

            x = center_x + radius * np.sin(phi) * np.cos(theta)
            y = center_y + radius * np.sin(phi) * np.sin(theta)
            z = center_z + radius * np.cos(phi)

            point = MassPoint(x, y, z, mass / num_points)
            self.points.append(point)

        # 创建弹簧（连接所有质点）
        self.springs: List[Spring] = []
        for i in range(num_points):
            for j in range(i + 1, num_points):
                p1 = self.points[i]
                p2 = self.points[j]
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                dz = p2.z - p1.z
                rest_length = np.sqrt(dx**2 + dy**2 + dz**2)
                self.springs.append(Spring(i, j, rest_length, stiffness))

    def update(self, dt: float, gravity: float = 9.81):
        """更新软体模拟"""
        # 应用重力
        for point in self.points:
            point.vz -= gravity * dt

        # 计算弹簧力
        for spring in self.springs:
            p1 = self.points[spring.point1_idx]
            p2 = self.points[spring.point2_idx]

            # 计算距离
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            dz = p2.z - p1.z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)

            if distance < 0.001:
                continue

            # 计算弹簧力
            displacement = distance - spring.rest_length
            force_magnitude = spring.stiffness * displacement

            # 计算阻尼力
            relative_velocity = np.array([
                p2.vx - p1.vx,
                p2.vy - p1.vy,
                p2.vz - p1.vz
            ])
            direction = np.array([dx, dy, dz]) / distance
            damping_force = spring.damping * np.dot(relative_velocity, direction)

            # 总力
            total_force = (force_magnitude + damping_force) * direction

            # 应用力
            p1.vx += total_force[0] / p1.mass * dt
            p1.vy += total_force[1] / p1.mass * dt
            p1.vz += total_force[2] / p1.mass * dt
            p2.vx -= total_force[0] / p2.mass * dt
            p2.vy -= total_force[1] / p2.mass * dt
            p2.vz -= total_force[2] / p2.mass * dt

        # 更新位置
        for point in self.points:
            point.x += point.vx * dt
            point.y += point.vy * dt
            point.z += point.vz * dt

            # 地面碰撞
            if point.z < 0:
                point.z = 0
                point.vz = -point.vz * 0.5  # 反弹

            # 应用阻尼
            point.vx *= 0.99
            point.vy *= 0.99
            point.vz *= 0.99

        # 更新中心位置
        if self.points:
            self.center_x = np.mean([p.x for p in self.points])
            self.center_y = np.mean([p.y for p in self.points])
            self.center_z = np.mean([p.z for p in self.points])

    def get_state(self) -> List[dict]:
        """获取软体状态"""
        states = []
        for point in self.points:
            states.append({
                'position': (point.x, point.y, point.z),
                'velocity': (point.vx, point.vy, point.vz),
                'mass': point.mass,
                'is_fixed': point.is_fixed
            })
        return states


def create_cloth_scene() -> Cloth:
    """创建一个布料场景"""
    cloth = Cloth(5.0, 5.0, 10, 10, mass=2.0, stiffness=100.0)
    return cloth


def create_rope_scene() -> Rope:
    """创建一个绳子场景"""
    rope = Rope(5.0, 20, mass=1.0, stiffness=200.0)
    return rope


def create_soft_body_scene() -> SoftBody:
    """创建一个软体场景"""
    soft_body = SoftBody(5.0, 5.0, 3.0, radius=1.0, num_points=20,
                        mass=2.0, stiffness=150.0)
    return soft_body
