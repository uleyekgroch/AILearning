"""
流体力学引擎

实现简单的流体效果：
1. 流体阻力
2. 浮力
3. 流体流动
4. 物体在流体中的行为

这是从刚体物理到流体物理的升级。
类比：从一个固体世界 → 一个有水、空气的世界。
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FluidProperties:
    """流体属性"""
    name: str
    density: float        # 密度
    viscosity: float      # 粘度
    drag_coefficient: float  # 阻力系数
    color: str = "blue"   # 颜色（用于可视化）


# 预定义流体
FLUIDS = {
    'water': FluidProperties("water", 1000.0, 0.001, 0.47, "blue"),
    'air': FluidProperties("air", 1.225, 0.000018, 0.47, "transparent"),
    'oil': FluidProperties("oil", 900.0, 0.1, 0.5, "yellow"),
    'honey': FluidProperties("honey", 1400.0, 10.0, 0.6, "gold"),
}


class FluidRegion:
    """
    流体区域

    定义流体在空间中的分布。
    """

    def __init__(self, fluid_type: str,
                 x_min: float, y_min: float, z_min: float,
                 x_max: float, y_max: float, z_max: float):
        self.fluid = FLUIDS[fluid_type]
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """检查点是否在流体区域内"""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)

    def get_depth(self, x: float, y: float, z: float) -> float:
        """获取点在流体中的深度"""
        if not self.contains_point(x, y, z):
            return 0.0
        return self.z_max - z

    def get_submerged_volume(self, x: float, y: float, z: float,
                            radius: float) -> float:
        """获取球体在流体中的浸没体积"""
        if not self.contains_point(x, y, z):
            return 0.0

        # 简化计算：假设球体完全浸没
        if z - radius >= self.z_min and z + radius <= self.z_max:
            return (4/3) * np.pi * radius**3

        # 部分浸没
        if z + radius > self.z_max:
            # 上部分露出
            h = self.z_max - (z - radius)
            h = max(0, min(2*radius, h))
            return np.pi * h**2 * (3*radius - h) / 3

        if z - radius < self.z_min:
            # 下部分超出
            h = (z + radius) - self.z_min
            h = max(0, min(2*radius, h))
            return np.pi * h**2 * (3*radius - h) / 3

        return (4/3) * np.pi * radius**3


class FluidForce:
    """
    流体力计算器

    计算流体对物体的作用力：
    1. 阻力（与速度相反）
    2. 浮力（向上）
    3. 升力（垂直于速度）
    """

    def __init__(self):
        pass

    def calculate_drag(self, velocity: np.ndarray, fluid: FluidProperties,
                      radius: float) -> np.ndarray:
        """
        计算流体阻力

        F_drag = 0.5 * ρ * v² * Cd * A
        """
        speed = np.linalg.norm(velocity)
        if speed < 0.001:
            return np.zeros(3)

        # 横截面积
        area = np.pi * radius**2

        # 阻力
        drag_magnitude = 0.5 * fluid.density * speed**2 * fluid.drag_coefficient * area

        # 阻力方向与速度相反
        drag_direction = -velocity / speed

        return drag_magnitude * drag_direction

    def calculate_buoyancy(self, submerged_volume: float, fluid: FluidProperties,
                          gravity: float = 9.81) -> np.ndarray:
        """
        计算浮力

        F_buoyancy = ρ_fluid * V_submerged * g
        """
        buoyancy_magnitude = fluid.density * submerged_volume * gravity
        return np.array([0, 0, buoyancy_magnitude])

    def calculate_viscous_force(self, velocity: np.ndarray, fluid: FluidProperties,
                               radius: float) -> np.ndarray:
        """
        计算粘性力（斯托克斯阻力）

        F_viscous = -6π * μ * r * v
        """
        return -6 * np.pi * fluid.viscosity * radius * velocity


class FluidEngine:
    """
    流体力学引擎

    整合流体区域和流体力计算。
    """

    def __init__(self):
        self.regions: List[FluidRegion] = []
        self.force_calculator = FluidForce()

    def add_region(self, region: FluidRegion):
        """添加流体区域"""
        self.regions.append(region)

    def get_fluid_at_point(self, x: float, y: float, z: float) -> Optional[FluidProperties]:
        """获取点所在的流体"""
        for region in self.regions:
            if region.contains_point(x, y, z):
                return region.fluid
        return None

    def calculate_forces(self, x: float, y: float, z: float,
                        vx: float, vy: float, vz: float,
                        radius: float, gravity: float = 9.81) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算流体对物体的所有力

        返回：(总力, 总扭矩)
        """
        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        # 检查物体是否在流体中
        fluid = self.get_fluid_at_point(x, y, z)
        if fluid is None:
            return total_force, total_torque

        velocity = np.array([vx, vy, vz])

        # 计算阻力
        drag = self.force_calculator.calculate_drag(velocity, fluid, radius)
        total_force += drag

        # 计算浮力
        for region in self.regions:
            if region.contains_point(x, y, z):
                submerged_volume = region.get_submerged_volume(x, y, z, radius)
                buoyancy = self.force_calculator.calculate_buoyancy(
                    submerged_volume, fluid, gravity
                )
                total_force += buoyancy

        # 计算粘性力
        viscous = self.force_calculator.calculate_viscous_force(velocity, fluid, radius)
        total_force += viscous

        return total_force, total_torque

    def apply_fluid_forces(self, body, gravity: float = 9.81):
        """
        将流体力应用到刚体

        参数：
            body: 刚体对象（需要有x, y, z, vx, vy, vz, radius, mass等属性）
            gravity: 重力加速度
        """
        force, torque = self.calculate_forces(
            body.x, body.y, body.z,
            body.vx, body.vy, body.vz,
            body.radius, gravity
        )

        # 应用力
        body.apply_force(force[0], force[1], force[2])

        # 应用扭矩
        body.apply_torque(torque[0], torque[1], torque[2])


def create_water_world() -> FluidEngine:
    """创建一个有水的世界"""
    engine = FluidEngine()

    # 添加一个水池
    water_region = FluidRegion('water',
                               x_min=0, y_min=0, z_min=0,
                               x_max=10, y_max=10, z_max=5)
    engine.add_region(water_region)

    return engine


def create_layered_fluid_world() -> FluidEngine:
    """创建一个有分层流体的世界"""
    engine = FluidEngine()

    # 底层：蜂蜜
    honey_region = FluidRegion('honey',
                               x_min=0, y_min=0, z_min=0,
                               x_max=10, y_max=10, z_max=2)
    engine.add_region(honey_region)

    # 中层：水
    water_region = FluidRegion('water',
                               x_min=0, y_min=0, z_min=2,
                               x_max=10, y_max=10, z_max=5)
    engine.add_region(water_region)

    # 上层：空气（默认）

    return engine
