"""
刚体动力学引擎

实现更真实的物理效果：
1. 刚体碰撞检测
2. 碰撞响应（反弹、摩擦）
3. 旋转动力学
4. 弹性碰撞

这是从简单物理到真实物理的升级。
类比：从一个简单的游戏物理 → 一个真实的物理模拟器。
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MaterialType(Enum):
    """材料类型"""
    WOOD = "wood"      # 木头：中等弹性，中等摩擦
    METAL = "metal"    # 金属：低弹性，低摩擦
    RUBBER = "rubber"  # 橡胶：高弹性，高摩擦
    ICE = "ice"        # 冰：低弹性，极低摩擦
    STONE = "stone"    # 石头：低弹性，高摩擦


@dataclass
class Material:
    """材料属性"""
    name: str
    elasticity: float    # 弹性系数 (0-1)
    friction: float      # 摩擦系数 (0-1)
    density: float       # 密度
    restitution: float   # 恢复系数 (0-1)


# 预定义材料
MATERIALS = {
    MaterialType.WOOD: Material("wood", 0.5, 0.4, 0.6, 0.5),
    MaterialType.METAL: Material("metal", 0.3, 0.2, 7.8, 0.3),
    MaterialType.RUBBER: Material("rubber", 0.8, 0.7, 1.2, 0.8),
    MaterialType.ICE: Material("ice", 0.1, 0.05, 0.9, 0.1),
    MaterialType.STONE: Material("stone", 0.2, 0.6, 2.5, 0.2),
}


class RigidBody:
    """
    刚体

    包含：
    1. 位置、速度、加速度
    2. 旋转角度、角速度
    3. 质量、惯性
    4. 材料属性
    """

    def __init__(self, x: float, y: float, z: float,
                 radius: float = 0.5,
                 mass: float = 1.0,
                 material: MaterialType = MaterialType.WOOD):
        # 位置
        self.x = x
        self.y = y
        self.z = z

        # 速度
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        # 加速度
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0

        # 旋转（四元数表示）
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0

        # 角速度
        self.angular_vx = 0.0
        self.angular_vy = 0.0
        self.angular_vz = 0.0

        # 几何属性
        self.radius = radius

        # 物理属性
        self.mass = mass
        self.material = MATERIALS[material]

        # 计算惯性（简化为球体）
        self.inertia = 0.4 * mass * radius ** 2

        # 状态
        self.is_static = False  # 是否静态（不可移动）
        self.is_sleeping = False  # 是否休眠

    def apply_force(self, fx: float, fy: float, fz: float):
        """应用力"""
        if self.is_static or self.is_sleeping:
            return

        # F = ma, a = F/m
        self.ax += fx / self.mass
        self.ay += fy / self.mass
        self.az += fz / self.mass

    def apply_torque(self, tx: float, ty: float, tz: float):
        """应用扭矩"""
        if self.is_static or self.is_sleeping:
            return

        # 扭矩产生角加速度
        self.angular_vx += tx / self.inertia
        self.angular_vy += ty / self.inertia
        self.angular_vz += tz / self.inertia

    def update(self, dt: float):
        """更新状态"""
        if self.is_static or self.is_sleeping:
            return

        # 更新速度
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.vz += self.az * dt

        # 更新位置
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        # 更新旋转
        self.angle_x += self.angular_vx * dt
        self.angle_y += self.angular_vy * dt
        self.angle_z += self.angular_vz * dt

        # 应用阻尼
        damping = 0.99
        self.vx *= damping
        self.vy *= damping
        self.vz *= damping
        self.angular_vx *= damping
        self.angular_vy *= damping
        self.angular_vz *= damping

        # 重置加速度
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0

        # 检查是否应该休眠
        speed = np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        angular_speed = np.sqrt(self.angular_vx**2 + self.angular_vy**2 + self.angular_vz**2)
        if speed < 0.01 and angular_speed < 0.01:
            self.is_sleeping = True

    def wake_up(self):
        """唤醒休眠的物体"""
        self.is_sleeping = False


class CollisionDetector:
    """
    碰撞检测器

    实现：
    1. 球体-球体碰撞检测
    2. 球体-平面碰撞检测
    3. 碰撞信息计算
    """

    def detect_sphere_sphere(self, body1: RigidBody, body2: RigidBody) -> Optional[dict]:
        """
        检测两个球体是否碰撞

        返回碰撞信息或None
        """
        # 计算距离
        dx = body1.x - body2.x
        dy = body1.y - body2.y
        dz = body1.z - body2.z
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        # 检查是否碰撞
        min_distance = body1.radius + body2.radius
        if distance >= min_distance:
            return None

        # 计算碰撞法线
        if distance == 0:
            normal = np.array([1, 0, 0])  # 默认方向
        else:
            normal = np.array([dx, dy, dz]) / distance

        # 计算穿透深度
        penetration = min_distance - distance

        # 计算碰撞点
        contact_point = np.array([
            body1.x - normal[0] * body1.radius,
            body1.y - normal[1] * body1.radius,
            body1.z - normal[2] * body1.radius
        ])

        return {
            'body1': body1,
            'body2': body2,
            'normal': normal,
            'penetration': penetration,
            'contact_point': contact_point
        }

    def detect_sphere_plane(self, body: RigidBody, plane_normal: np.ndarray,
                           plane_point: np.ndarray) -> Optional[dict]:
        """
        检测球体是否与平面碰撞

        参数：
            body: 球体
            plane_normal: 平面法线（指向平面外）
            plane_point: 平面上一点
        """
        # 计算球心到平面的距离
        d = np.dot(np.array([body.x, body.y, body.z]) - plane_point, plane_normal)

        # 检查是否碰撞
        if d >= body.radius:
            return None

        # 计算穿透深度
        penetration = body.radius - d

        # 计算碰撞点
        contact_point = np.array([body.x, body.y, body.z]) - plane_normal * d

        return {
            'body': body,
            'normal': plane_normal,
            'penetration': penetration,
            'contact_point': contact_point
        }


class CollisionResolver:
    """
    碰撞响应器

    实现：
    1. 弹性碰撞
    2. 摩擦力
    3. 碰撞冲量
    """

    def resolve_sphere_sphere(self, collision: dict):
        """
        解决两个球体的碰撞

        使用冲量方法
        """
        body1 = collision['body1']
        body2 = collision['body2']
        normal = collision['normal']
        penetration = collision['penetration']

        # 唤醒休眠的物体
        body1.wake_up()
        body2.wake_up()

        # 分离物体
        separation = penetration / 2
        if not body1.is_static:
            body1.x += normal[0] * separation
            body1.y += normal[1] * separation
            body1.z += normal[2] * separation
        if not body2.is_static:
            body2.x -= normal[0] * separation
            body2.y -= normal[1] * separation
            body2.z -= normal[2] * separation

        # 计算相对速度
        relative_velocity = np.array([
            body1.vx - body2.vx,
            body1.vy - body2.vy,
            body1.vz - body2.vz
        ])

        # 计算相对速度在法线方向的分量
        relative_speed = np.dot(relative_velocity, normal)

        # 如果物体正在分离，不需要处理
        if relative_speed > 0:
            return

        # 计算恢复系数（两个材料的平均）
        restitution = (body1.material.restitution + body2.material.restitution) / 2

        # 计算冲量
        if body1.is_static:
            j = -(1 + restitution) * relative_speed / (1 / body2.mass)
        elif body2.is_static:
            j = -(1 + restitution) * relative_speed / (1 / body1.mass)
        else:
            j = -(1 + restitution) * relative_speed / (1 / body1.mass + 1 / body2.mass)

        # 应用冲量
        impulse = j * normal
        if not body1.is_static:
            body1.vx += impulse[0] / body1.mass
            body1.vy += impulse[1] / body1.mass
            body1.vz += impulse[2] / body1.mass
        if not body2.is_static:
            body2.vx -= impulse[0] / body2.mass
            body2.vy -= impulse[1] / body2.mass
            body2.vz -= impulse[2] / body2.mass

        # 应用摩擦力
        self._apply_friction(body1, body2, normal, j)

    def resolve_sphere_plane(self, collision: dict):
        """
        解决球体与平面的碰撞
        """
        body = collision['body']
        normal = collision['normal']
        penetration = collision['penetration']

        # 唤醒休眠的物体
        body.wake_up()

        # 分离物体
        if not body.is_static:
            body.x += normal[0] * penetration
            body.y += normal[1] * penetration
            body.z += normal[2] * penetration

        # 计算速度在法线方向的分量
        velocity = np.array([body.vx, body.vy, body.vz])
        normal_speed = np.dot(velocity, normal)

        # 如果物体正在分离，不需要处理
        if normal_speed > 0:
            return

        # 计算恢复系数
        restitution = body.material.restitution

        # 计算冲量
        j = -(1 + restitution) * normal_speed

        # 应用冲量
        impulse = j * normal
        if not body.is_static:
            body.vx += impulse[0] / body.mass
            body.vy += impulse[1] / body.mass
            body.vz += impulse[2] / body.mass

        # 应用摩擦力
        self._apply_friction_with_plane(body, normal, j)

    def _apply_friction(self, body1: RigidBody, body2: RigidBody,
                       normal: np.ndarray, normal_impulse: float):
        """应用摩擦力"""
        # 计算切线方向
        relative_velocity = np.array([
            body1.vx - body2.vx,
            body1.vy - body2.vy,
            body1.vz - body2.vz
        ])

        # 移除法线分量
        tangent = relative_velocity - np.dot(relative_velocity, normal) * normal
        tangent_length = np.linalg.norm(tangent)

        if tangent_length < 0.001:
            return

        tangent = tangent / tangent_length

        # 计算摩擦系数（两个材料的平均）
        friction = (body1.material.friction + body2.material.friction) / 2

        # 计算摩擦冲量
        friction_impulse = -friction * normal_impulse

        # 应用摩擦力
        if not body1.is_static:
            body1.vx += friction_impulse * tangent[0] / body1.mass
            body1.vy += friction_impulse * tangent[1] / body1.mass
            body1.vz += friction_impulse * tangent[2] / body1.mass
        if not body2.is_static:
            body2.vx -= friction_impulse * tangent[0] / body2.mass
            body2.vy -= friction_impulse * tangent[1] / body2.mass
            body2.vz -= friction_impulse * tangent[2] / body2.mass

    def _apply_friction_with_plane(self, body: RigidBody,
                                   normal: np.ndarray, normal_impulse: float):
        """应用与平面的摩擦力"""
        # 计算切线方向
        velocity = np.array([body.vx, body.vy, body.vz])
        tangent = velocity - np.dot(velocity, normal) * normal
        tangent_length = np.linalg.norm(tangent)

        if tangent_length < 0.001:
            return

        tangent = tangent / tangent_length

        # 计算摩擦系数
        friction = body.material.friction

        # 计算摩擦冲量
        friction_impulse = -friction * normal_impulse

        # 应用摩擦力
        if not body.is_static:
            body.vx += friction_impulse * tangent[0] / body.mass
            body.vy += friction_impulse * tangent[1] / body.mass
            body.vz += friction_impulse * tangent[2] / body.mass


class RigidBodyEngine:
    """
    刚体动力学引擎

    整合碰撞检测和响应。
    """

    def __init__(self, gravity: float = 9.81):
        self.gravity = gravity
        self.bodies: List[RigidBody] = []
        self.detector = CollisionDetector()
        self.resolver = CollisionResolver()

        # 地面平面
        self.ground_normal = np.array([0, 0, 1])
        self.ground_point = np.array([0, 0, 0])

    def add_body(self, body: RigidBody):
        """添加刚体"""
        self.bodies.append(body)

    def update(self, dt: float):
        """更新物理模拟"""
        # 应用重力
        for body in self.bodies:
            if not body.is_static:
                body.apply_force(0, 0, -self.gravity * body.mass)

        # 检测碰撞
        collisions = []

        # 球体-球体碰撞
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                collision = self.detector.detect_sphere_sphere(
                    self.bodies[i], self.bodies[j]
                )
                if collision:
                    collisions.append(('sphere_sphere', collision))

        # 球体-平面碰撞
        for body in self.bodies:
            collision = self.detector.detect_sphere_plane(
                body, self.ground_normal, self.ground_point
            )
            if collision:
                collisions.append(('sphere_plane', collision))

        # 解决碰撞
        for collision_type, collision in collisions:
            if collision_type == 'sphere_sphere':
                self.resolver.resolve_sphere_sphere(collision)
            elif collision_type == 'sphere_plane':
                self.resolver.resolve_sphere_plane(collision)

        # 更新所有物体
        for body in self.bodies:
            body.update(dt)

    def get_state(self) -> List[dict]:
        """获取所有物体的状态"""
        states = []
        for body in self.bodies:
            states.append({
                'position': (body.x, body.y, body.z),
                'velocity': (body.vx, body.vy, body.vz),
                'rotation': (body.angle_x, body.angle_y, body.angle_z),
                'angular_velocity': (body.angular_vx, body.angular_vy, body.angular_vz),
                'radius': body.radius,
                'mass': body.mass,
                'material': body.material.name,
                'is_sleeping': body.is_sleeping
            })
        return states


def create_simple_rigid_world() -> RigidBodyEngine:
    """创建一个简单的刚体世界"""
    engine = RigidBodyEngine(gravity=9.81)

    # 添加一些物体
    ball1 = RigidBody(2, 2, 5, radius=0.5, mass=1.0, material=MaterialType.RUBBER)
    ball2 = RigidBody(5, 5, 3, radius=0.5, mass=2.0, material=MaterialType.METAL)
    ball3 = RigidBody(8, 3, 4, radius=0.7, mass=1.5, material=MaterialType.WOOD)

    engine.add_body(ball1)
    engine.add_body(ball2)
    engine.add_body(ball3)

    return engine
