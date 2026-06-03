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


# ============================================================
# SPH 粒子流体系统（Phase 48 新增）
# ============================================================

from collections import defaultdict


@dataclass
class FluidParticle:
    """流体粒子"""
    position: np.ndarray   # (3,)
    velocity: np.ndarray   # (3,)
    density: float = 0.0
    pressure: float = 0.0
    radius: float = 0.1
    mass: float = 0.1


class SpatialHash:
    """
    空间哈希：加速粒子邻域查找

    将空间划分为网格，每个粒子只与同格及相邻格的粒子交互。
    """

    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.cells = defaultdict(list)

    def _key(self, pos: np.ndarray) -> Tuple[int, int, int]:
        return (
            int(np.floor(pos[0] / self.cell_size)),
            int(np.floor(pos[1] / self.cell_size)),
            int(np.floor(pos[2] / self.cell_size)),
        )

    def clear(self):
        self.cells.clear()

    def insert(self, idx: int, pos: np.ndarray):
        self.cells[self._key(pos)].append(idx)

    def query_neighbors(self, pos: np.ndarray, radius: float) -> List[int]:
        """查找 pos 附近 radius 范围内的所有粒子索引"""
        key = self._key(pos)
        r_cells = int(np.ceil(radius / self.cell_size))
        neighbors = []
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                for dz in range(-r_cells, r_cells + 1):
                    cell = (key[0] + dx, key[1] + dy, key[2] + dz)
                    neighbors.extend(self.cells.get(cell, []))
        return neighbors


class FluidParticleSystem:
    """
    SPH 粒子流体系统

    简化光滑粒子流体动力学：
    1. 密度估计：核函数加权求和
    2. 压力计算：密度偏差 × 刚度
    3. 力计算：压力梯度 + 粘性 + 重力
    4. 积分：半隐式欧拉

    Agent 可学到的物理直觉：
    - 流体向下流动（重力）
    - 流体聚集在底部（压力平衡）
    - 流体遇到障碍物分流（粘附力）
    """

    def __init__(self, bounds: Tuple[float, float, float] = (10.0, 10.0, 5.0),
                 gravity: float = -9.8,
                 viscosity: float = 0.1,
                 rest_density: float = 1.0,
                 stiffness: float = 50.0,
                 smoothing_radius: float = 0.5,
                 particle_mass: float = 0.1,
                 damping: float = 0.98):
        self.bounds = np.array(bounds, dtype=np.float32)
        self.gravity = gravity
        self.viscosity = viscosity
        self.rest_density = rest_density
        self.stiffness = stiffness
        self.smoothing_radius = smoothing_radius
        self.particle_mass = particle_mass
        self.damping = damping

        self.particles: List[FluidParticle] = []
        self.spatial_hash = SpatialHash(smoothing_radius)
        self.step_count = 0

    def add_particles(self, center: np.ndarray, count: int,
                      spread: float = 0.5) -> List[int]:
        """在 center 附近添加 count 个粒子"""
        start_idx = len(self.particles)
        for _ in range(count):
            pos = center + np.random.uniform(-spread, spread, 3).astype(np.float32)
            pos = np.clip(pos, [0, 0, 0], self.bounds - 0.01)
            vel = np.zeros(3, dtype=np.float32)
            self.particles.append(FluidParticle(
                position=pos, velocity=vel, mass=self.particle_mass
            ))
        return list(range(start_idx, len(self.particles)))

    def step(self, dt: float = 0.02):
        """一步流体模拟"""
        if not self.particles:
            return

        # 1. 构建空间哈希
        self.spatial_hash.clear()
        for i, p in enumerate(self.particles):
            self.spatial_hash.insert(i, p.position)

        # 2. 计算密度和压力
        h = self.smoothing_radius
        for i, p in enumerate(self.particles):
            p.density = 0.0
            for j in self.spatial_hash.query_neighbors(p.position, h):
                if i == j:
                    continue
                dist = np.linalg.norm(p.position - self.particles[j].position)
                if dist < h:
                    w = (1.0 - dist / h) ** 2
                    p.density += self.particle_mass * w
            p.pressure = self.stiffness * max(0, p.density - self.rest_density)

        # 3. 计算力并更新速度
        for i, p in enumerate(self.particles):
            force = np.array([0.0, 0.0, self.gravity * p.mass], dtype=np.float32)

            for j in self.spatial_hash.query_neighbors(p.position, h):
                if i == j:
                    continue
                q = self.particles[j]
                diff = q.position - p.position
                dist = np.linalg.norm(diff)
                if dist < 1e-6 or dist >= h:
                    continue

                direction = diff / dist
                w = (1.0 - dist / h) ** 2

                # 压力梯度力
                force += -self.particle_mass * (
                    p.pressure + q.pressure
                ) / (2.0 * max(q.density, 1e-6)) * w * direction

                # 粘性力
                force += self.viscosity * self.particle_mass * (
                    q.velocity - p.velocity
                ) / max(q.density, 1e-6) * w

            p.velocity += (force / p.mass) * dt
            p.velocity *= self.damping
            p.position += p.velocity * dt

        # 4. 边界约束
        self._apply_boundaries()
        self.step_count += 1

    def _apply_boundaries(self):
        """边界约束：粒子碰到容器壁反弹"""
        bounce = 0.3
        for p in self.particles:
            for axis in range(3):
                if p.position[axis] < 0:
                    p.position[axis] = 0
                    p.velocity[axis] = abs(p.velocity[axis]) * bounce
                elif p.position[axis] > self.bounds[axis] - p.radius:
                    p.position[axis] = self.bounds[axis] - p.radius
                    p.velocity[axis] = -abs(p.velocity[axis]) * bounce

    def get_state(self) -> list:
        return [{
            'position': p.position.tolist(),
            'velocity': p.velocity.tolist(),
            'density': p.density,
            'pressure': p.pressure,
        } for p in self.particles]

    def get_level(self) -> float:
        """流体平均高度"""
        if not self.particles:
            return 0.0
        return float(np.mean([p.position[2] for p in self.particles]))

    def get_flow_direction(self) -> np.ndarray:
        """流体平均流动方向"""
        if not self.particles:
            return np.zeros(3)
        avg_vel = np.mean([p.velocity for p in self.particles], axis=0)
        norm = np.linalg.norm(avg_vel)
        return avg_vel / norm if norm > 1e-6 else np.zeros(3)

    def get_spread(self) -> float:
        """流体扩散程度"""
        if len(self.particles) < 2:
            return 0.0
        positions = np.array([p.position for p in self.particles])
        return float(np.mean(np.std(positions, axis=0)))

    def get_avg_speed(self) -> float:
        if not self.particles:
            return 0.0
        return float(np.mean([np.linalg.norm(p.velocity) for p in self.particles]))

    def get_features(self) -> dict:
        """提取流体特征（用于语言涌现）"""
        level = self.get_level()
        speed = self.get_avg_speed()
        spread = self.get_spread()
        return {
            'type': 'fluid',
            'level': 'high' if level > self.bounds[2] * 0.6 else
                     'low' if level < self.bounds[2] * 0.3 else 'mid',
            'motion': 'fast' if speed > 1.0 else
                      'moving' if speed > 0.1 else 'still',
            'spread': 'wide' if spread > 1.0 else
                      'narrow' if spread < 0.3 else 'medium',
            'particle_count': len(self.particles),
        }

    def collide_with_sphere(self, center: np.ndarray, radius: float,
                            velocity: np.ndarray = None):
        """流体粒子与球形物体碰撞"""
        if velocity is None:
            velocity = np.zeros(3)
        for p in self.particles:
            diff = p.position - center
            dist = np.linalg.norm(diff)
            min_dist = radius + p.radius
            if dist < min_dist and dist > 1e-6:
                normal = diff / dist
                p.position = center + normal * min_dist
                v_rel = p.velocity - velocity
                v_n = np.dot(v_rel, normal)
                if v_n < 0:
                    p.velocity = p.velocity - 1.5 * v_n * normal
