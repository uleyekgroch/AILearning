"""
3D 物理世界 v2：多形状、连续动作、丰富材质

增强：
1. 多形状物体：sphere, cube, cylinder（AABB 碰撞近似）
2. 连续动作空间：6 维连续向量 [force_xyz, torque, grip, throw]
3. 丰富材质：7 种（metal, wood, plastic, glass, rubber, stone, fabric）
4. 增强渲染：不同形状不同投影
5. 增强特征：14 维物体特征向量
6. 流体粒子系统（Phase 48）
7. 软体弹簧-质点系统（Phase 48）
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union


# ============================================================
# 材质系统（7 种）
# ============================================================

MATERIALS = {
    'metal':   {'density': 1.5, 'restitution': 0.6, 'friction': 0.3, 'frequency': 2000.0,
                'color': np.array([0.7, 0.7, 0.8]), 'hardness': 'hard'},
    'wood':    {'density': 1.0, 'restitution': 0.3, 'friction': 0.7, 'frequency': 300.0,
                'color': np.array([0.6, 0.4, 0.2]), 'hardness': 'hard'},
    'plastic': {'density': 0.8, 'restitution': 0.7, 'friction': 0.5, 'frequency': 800.0,
                'color': np.array([0.2, 0.6, 0.9]), 'hardness': 'soft'},
    'glass':   {'density': 1.2, 'restitution': 0.1, 'friction': 0.2, 'frequency': 3000.0,
                'color': np.array([0.6, 0.8, 1.0]), 'hardness': 'hard'},
    'rubber':  {'density': 0.6, 'restitution': 0.9, 'friction': 0.9, 'frequency': 200.0,
                'color': np.array([0.2, 0.2, 0.2]), 'hardness': 'elastic'},
    'stone':   {'density': 2.0, 'restitution': 0.2, 'friction': 0.8, 'frequency': 150.0,
                'color': np.array([0.5, 0.5, 0.5]), 'hardness': 'hard'},
    'fabric':  {'density': 0.3, 'restitution': 0.1, 'friction': 0.6, 'frequency': 100.0,
                'color': np.array([0.8, 0.7, 0.5]), 'hardness': 'soft'},
}

MATERIAL_NAMES = list(MATERIALS.keys())
MATERIAL_COLORS = {k: v['color'] for k, v in MATERIALS.items()}
MATERIAL_FREQUENCY = {k: v['frequency'] for k, v in MATERIALS.items()}
MATERIAL_RESTITUTION = {k: v['restitution'] for k, v in MATERIALS.items()}


# ============================================================
# 物理物体（支持多形状）
# ============================================================

@dataclass
class PhysicsObject3D:
    """3D 物理物体（支持 sphere, cube, cylinder）"""
    id: int
    position: np.ndarray      # (3,) x, y, z
    velocity: np.ndarray      # (3,) vx, vy, vz
    mass: float = 1.0         # 基础质量（× density = 实际质量）
    radius: float = 0.3       # 包围球半径（用于快速剔除）
    restitution: float = 0.5  # 弹性系数 0-1
    material: str = 'wood'    # 材质
    is_static: bool = False   # 是否静态
    on_ground: bool = False   # 是否在地面上
    shape: str = 'sphere'     # 'sphere', 'cube', 'cylinder'
    size: np.ndarray = None   # (3,) 半尺寸 [rx, ry, rz]（cube/cylinder 用）
    color: np.ndarray = None  # (3,) RGB 颜色（覆盖材质默认色）
    temperature: float = 20.0 # 温度

    def __post_init__(self):
        if self.size is None:
            self.size = np.array([self.radius, self.radius, self.radius])
        if self.color is None:
            self.color = MATERIAL_COLORS.get(self.material, np.array([0.5, 0.5, 0.5]))

    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    def get_effective_mass(self) -> float:
        """有效质量 = 基础质量 × 材质密度"""
        density = MATERIALS.get(self.material, {}).get('density', 1.0)
        return self.mass * density

    def get_bounding_radius(self) -> float:
        """包围球半径"""
        if self.shape == 'sphere':
            return self.size[0]
        elif self.shape == 'cube':
            return np.linalg.norm(self.size)
        else:  # cylinder
            return max(self.size[0], self.size[2])


# ============================================================
# 音频事件
# ============================================================

@dataclass
class AudioEvent3D:
    """3D 音频事件"""
    event_type: str       # 'collision', 'ground_hit', 'wall_hit', 'grab', 'drop', 'throw'
    amplitude: float      # 音量 0-1
    frequency: float      # 频率（材质决定音色）
    direction: np.ndarray  # (3,) 相对于 agent 的方向
    source_id: int = -1   # 源物体 id
    material: str = ''    # 材质（用于音色编码）


# ============================================================
# 观察数据
# ============================================================

@dataclass
class Observation3D:
    """3D 世界的观察数据"""
    visual: np.ndarray       # (64, 64, 3) RGB 图像
    depth: np.ndarray        # (64, 64) 深度图
    audio_events: List[AudioEvent3D]  # 音频事件列表
    proprioception: np.ndarray  # (10,) 本体感知
    # proprioception: [vx, vy, vz, facing_x, facing_y, speed, held_mass, held_flag, grip, throw_ready]


# ============================================================
# 动作定义
# ============================================================

class Action3D:
    """3D 离散动作枚举"""
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5
    TURN_LEFT = 6
    TURN_RIGHT = 7
    GRAB = 8
    DROP = 9
    THROW = 10
    PUSH = 11

    NAMES = ['forward', 'backward', 'left', 'right', 'up', 'down',
             'turn_left', 'turn_right', 'grab', 'drop', 'throw', 'push']

    COUNT = 12


# ============================================================
# 3D 物理引擎
# ============================================================

class PhysicsWorld3D:
    """
    3D 物理引擎 v2（纯 numpy）

    新增：
    - 多形状碰撞（sphere-sphere, sphere-cube, cube-cube）
    - 连续动作支持
    - 7 种材质
    - 增强渲染（不同形状不同投影）
    """

    def __init__(self, bounds: Tuple[float, float, float] = (10.0, 10.0, 5.0),
                 dt: float = 0.02, gravity: float = -9.8,
                 friction: float = 0.8, air_damping: float = 0.999):
        self.bounds = np.array(bounds, dtype=float)
        self.dt = dt
        self.gravity = gravity
        self.friction = friction
        self.air_damping = air_damping

        self.objects: List[PhysicsObject3D] = []
        self.next_id = 0

        # Agent 状态
        self.agent_pos = np.array([bounds[0] / 2, bounds[1] / 2, 0.5])
        self.agent_vel = np.zeros(3)
        self.agent_facing = 0.0  # 弧度
        self.agent_radius = 0.4
        self.held_object: Optional[int] = None
        self.grip_strength: float = 0.0  # 0-1
        self.throw_ready: float = 0.0    # 0-1

        # 音频事件缓存
        self._audio_events: List[AudioEvent3D] = []

        # 渲染参数
        self._fov = 60.0
        self._near = 0.1
        self._far = 20.0
        self._render_size = 64

        # 流体/软体系统（Phase 48）
        self._fluid_system = None
        self._soft_bodies = []

    def add_object(self, position: np.ndarray, mass: float = 1.0,
                   radius: float = 0.3, material: str = 'wood',
                   shape: str = 'sphere', size: Optional[np.ndarray] = None,
                   color: Optional[np.ndarray] = None,
                   velocity: Optional[np.ndarray] = None) -> int:
        """添加物体，返回 id"""
        if size is None:
            size = np.array([radius, radius, radius])

        # 计算实际弹性系数（材质 × 基础值）
        mat_rest = MATERIAL_RESTITUTION.get(material, 0.5)

        obj = PhysicsObject3D(
            id=self.next_id,
            position=np.array(position, dtype=float),
            velocity=np.zeros(3) if velocity is None else np.array(velocity, dtype=float),
            mass=mass,
            radius=radius,
            restitution=mat_rest,
            material=material,
            shape=shape,
            size=np.array(size, dtype=float),
            color=color if color is not None else MATERIAL_COLORS.get(material, np.array([0.5, 0.5, 0.5])),
        )
        self.objects.append(obj)
        self.next_id += 1
        return obj.id

    def add_random_objects(self, count: int = 5,
                           shapes: Optional[List[str]] = None,
                           materials: Optional[List[str]] = None) -> List[int]:
        """添加随机物体"""
        if shapes is None:
            shapes = ['sphere', 'cube', 'cylinder']
        if materials is None:
            materials = MATERIAL_NAMES

        ids = []
        for _ in range(count):
            pos = np.array([
                np.random.uniform(1.0, self.bounds[0] - 1.0),
                np.random.uniform(1.0, self.bounds[1] - 1.0),
                np.random.uniform(0.5, 2.0),
            ])
            mass = np.random.uniform(0.5, 5.0)
            radius = np.random.uniform(0.2, 0.5)
            mat = np.random.choice(materials)
            shape = np.random.choice(shapes)

            if shape == 'sphere':
                size = np.array([radius, radius, radius])
            elif shape == 'cube':
                size = np.array([radius, radius * np.random.uniform(0.5, 1.5), radius])
            else:  # cylinder
                size = np.array([radius, radius, radius * np.random.uniform(0.5, 2.0)])

            ids.append(self.add_object(pos, mass=mass, radius=radius,
                                       material=mat, shape=shape, size=size))
        return ids

    # ----------------------------------------------------------
    # 流体/软体接口（Phase 48）
    # ----------------------------------------------------------

    def add_fluid(self, center: np.ndarray, count: int = 50,
                  spread: float = 0.5, viscosity: float = 0.1) -> str:
        """添加流体粒子系统，返回 'fluid'"""
        from physics_fluid import FluidParticleSystem
        if self._fluid_system is None:
            self._fluid_system = FluidParticleSystem(
                bounds=tuple(self.bounds), viscosity=viscosity
            )
        self._fluid_system.add_particles(np.array(center, dtype=np.float32), count, spread)
        return 'fluid'

    def add_soft_body(self, center: np.ndarray, size: np.ndarray,
                      shape: str = 'box', mass: float = 1.0,
                      stiffness: float = 50.0) -> int:
        """添加软体，返回索引"""
        from physics_soft import SoftBox
        if shape == 'box':
            sb = SoftBox(center.tolist(), size.tolist(), mass, stiffness)
        else:
            sb = SoftBox(center.tolist(), size.tolist(), mass, stiffness)
        self._soft_bodies.append(sb)
        return len(self._soft_bodies) - 1

    def get_fluid_features(self) -> Optional[dict]:
        """获取流体特征"""
        if self._fluid_system and self._fluid_system.particles:
            return self._fluid_system.get_features()
        return None

    def get_soft_body_features(self) -> list:
        """获取所有软体特征"""
        features = []
        for i, sb in enumerate(self._soft_bodies):
            deform = sb.get_deformation() if hasattr(sb, 'get_deformation') else 0
            center = sb.get_center() if hasattr(sb, 'get_center') else np.zeros(3)
            features.append({
                'type': 'soft_body',
                'index': i,
                'deformation': 'deformed' if deform > 0.1 else 'stable',
                'height': 'high' if center[2] > self.bounds[2] * 0.6 else
                          'low' if center[2] < self.bounds[2] * 0.3 else 'mid',
            })
        return features

    def step(self, action) -> Observation3D:
        """
        执行一步物理模拟

        Args:
            action: int（离散动作）或 np.ndarray（6 维连续动作）
        """
        self._audio_events = []

        # 1. Agent 动作
        if isinstance(action, np.ndarray):
            self._apply_continuous_action(action)
        else:
            self._apply_action(action)

        # 2. 施加重力
        for obj in self.objects:
            if not obj.is_static and obj.id != self.held_object:
                obj.velocity[2] += self.gravity * self.dt

        self.agent_vel[2] += self.gravity * self.dt

        # 3. 更新位置
        for obj in self.objects:
            if not obj.is_static and obj.id != self.held_object:
                obj.position += obj.velocity * self.dt

        self.agent_pos += self.agent_vel * self.dt

        # 4. 碰撞检测 + 响应
        self._resolve_collisions()

        # 5. 地面约束
        self._ground_constraints()

        # 6. 边界约束
        self._boundary_constraints()

        # 7. 摩擦力
        self._apply_friction()

        # 8. 更新持有物体位置
        self._update_held_object()

        # 8.5 流体/软体更新（Phase 48）
        if self._fluid_system:
            self._fluid_system.step(self.dt)
            # 流体与刚体碰撞
            for obj in self.objects:
                if not obj.is_static:
                    self._fluid_system.collide_with_sphere(
                        obj.position, obj.get_bounding_radius(), obj.velocity
                    )
            # 流体与 Agent 碰撞
            self._fluid_system.collide_with_sphere(
                self.agent_pos, self.agent_radius, self.agent_vel
            )
        for sb in self._soft_bodies:
            sb.update(self.dt, abs(self.gravity))
            # 软体与刚体碰撞
            for obj in self.objects:
                if not obj.is_static:
                    sb.collide_with_sphere(
                        obj.position, obj.get_bounding_radius(), obj.velocity,
                        obj.get_effective_mass()
                    )

        # 9. 渲染
        visual, depth = self._render()
        audio = list(self._audio_events)
        proprio = self._get_proprioception()

        return Observation3D(
            visual=visual,
            depth=depth,
            audio_events=audio,
            proprioception=proprio,
        )

    # ----------------------------------------------------------
    # 连续动作
    # ----------------------------------------------------------

    def _apply_continuous_action(self, action: np.ndarray):
        """
        连续动作：6 维向量
        [force_x, force_y, force_z, torque_z, grip, throw_speed]
        范围：force/torque -1~1, grip/throw 0~1
        """
        action = np.clip(action, -1.0, 1.0)

        move_scale = 5.0  # m/s
        turn_scale = np.pi / 2  # rad/step

        # 力方向（世界坐标系）
        self.agent_vel[0] += action[0] * move_scale * self.dt * 50
        self.agent_vel[1] += action[1] * move_scale * self.dt * 50
        self.agent_vel[2] += action[2] * move_scale * self.dt * 50

        # 转向
        self.agent_facing += action[3] * turn_scale

        # grip: 0 = 松手, >0.5 = 抓取
        grip = max(0.0, action[3 + 1])  # action[4]
        throw_speed = max(0.0, action[3 + 2])  # action[5]

        self.grip_strength = grip
        self.throw_ready = throw_speed

        # grip 逻辑
        if grip > 0.5 and self.held_object is None:
            self._try_grab()
        elif grip < 0.1 and self.held_object is not None:
            self._try_drop()

        # throw 逻辑
        if throw_speed > 0.5 and self.held_object is not None:
            self._try_throw_with_speed(throw_speed * 10.0)

        # 速度衰减
        self.agent_vel *= 0.7

    def _apply_action(self, action: int):
        """离散动作"""
        move_speed = 2.0
        turn_speed = np.pi / 2

        fx = np.cos(self.agent_facing)
        fy = np.sin(self.agent_facing)

        if action == Action3D.FORWARD:
            self.agent_vel[0] = fx * move_speed
            self.agent_vel[1] = fy * move_speed
        elif action == Action3D.BACKWARD:
            self.agent_vel[0] = -fx * move_speed
            self.agent_vel[1] = -fy * move_speed
        elif action == Action3D.LEFT:
            self.agent_vel[0] = -fy * move_speed
            self.agent_vel[1] = fx * move_speed
        elif action == Action3D.RIGHT:
            self.agent_vel[0] = fy * move_speed
            self.agent_vel[1] = -fx * move_speed
        elif action == Action3D.UP:
            self.agent_vel[2] = move_speed
        elif action == Action3D.DOWN:
            self.agent_vel[2] = -move_speed
        elif action == Action3D.TURN_LEFT:
            self.agent_facing += turn_speed
        elif action == Action3D.TURN_RIGHT:
            self.agent_facing -= turn_speed
        elif action == Action3D.GRAB:
            self._try_grab()
        elif action == Action3D.DROP:
            self._try_drop()
        elif action == Action3D.THROW:
            self._try_throw()
        elif action == Action3D.PUSH:
            self._try_push()

        if action in (Action3D.FORWARD, Action3D.BACKWARD,
                      Action3D.LEFT, Action3D.RIGHT, Action3D.UP, Action3D.DOWN):
            self.agent_vel *= 0.5

    # ----------------------------------------------------------
    # Agent 交互
    # ----------------------------------------------------------

    def _try_grab(self):
        if self.held_object is not None:
            return

        facing_dir = np.array([np.cos(self.agent_facing), np.sin(self.agent_facing), 0.0])
        best_id = None
        best_dist = 1.5

        for obj in self.objects:
            if obj.is_static:
                continue
            to_obj = obj.position - self.agent_pos
            dist = np.linalg.norm(to_obj)
            if dist < best_dist:
                if dist > 0.01:
                    cos_angle = np.dot(to_obj[:2], facing_dir[:2]) / (
                        np.linalg.norm(to_obj[:2]) * np.linalg.norm(facing_dir[:2]) + 1e-8)
                    if cos_angle > 0.5:
                        best_dist = dist
                        best_id = obj.id

        if best_id is not None:
            self.held_object = best_id
            obj = self._get_object(best_id)
            self._audio_events.append(AudioEvent3D(
                event_type='grab', amplitude=0.3, frequency=500.0,
                direction=np.zeros(3), source_id=best_id,
                material=obj.material if obj else '',
            ))

    def _try_drop(self):
        if self.held_object is None:
            return

        obj = self._get_object(self.held_object)
        if obj:
            obj.velocity = self.agent_vel.copy() * 0.3
            self._audio_events.append(AudioEvent3D(
                event_type='drop', amplitude=0.2, frequency=400.0,
                direction=obj.position - self.agent_pos, source_id=obj.id,
                material=obj.material,
            ))
        self.held_object = None

    def _try_throw(self):
        self._try_throw_with_speed(8.0)

    def _try_throw_with_speed(self, speed: float):
        if self.held_object is None:
            return

        obj = self._get_object(self.held_object)
        if obj:
            facing_dir = np.array([np.cos(self.agent_facing), np.sin(self.agent_facing), 0.3])
            facing_dir = facing_dir / (np.linalg.norm(facing_dir) + 1e-8)
            obj.velocity = facing_dir * speed + self.agent_vel * 0.5
            self._audio_events.append(AudioEvent3D(
                event_type='throw', amplitude=0.5, frequency=600.0,
                direction=obj.position - self.agent_pos, source_id=obj.id,
                material=obj.material,
            ))
        self.held_object = None

    def _try_push(self):
        facing_dir = np.array([np.cos(self.agent_facing), np.sin(self.agent_facing), 0.0])
        push_force = 5.0

        for obj in self.objects:
            if obj.is_static:
                continue
            to_obj = obj.position - self.agent_pos
            dist = np.linalg.norm(to_obj)
            if dist < 1.5:
                cos_angle = np.dot(to_obj[:2], facing_dir[:2]) / (
                    np.linalg.norm(to_obj[:2]) * np.linalg.norm(facing_dir[:2]) + 1e-8)
                if cos_angle > 0.5:
                    effective_mass = obj.get_effective_mass()
                    impulse = facing_dir * push_force / max(effective_mass, 0.1)
                    obj.velocity += impulse
                    self._audio_events.append(AudioEvent3D(
                        event_type='push',
                        amplitude=0.4 * min(1.0, push_force / effective_mass),
                        frequency=MATERIAL_FREQUENCY.get(obj.material, 500.0),
                        direction=to_obj, source_id=obj.id,
                        material=obj.material,
                    ))
                    break

    # ----------------------------------------------------------
    # 碰撞检测（多形状）
    # ----------------------------------------------------------

    def _resolve_collisions(self):
        """多形状碰撞检测与响应"""
        all_bodies = list(self.objects)

        agent_sphere = PhysicsObject3D(
            id=-1, position=self.agent_pos.copy(),
            velocity=self.agent_vel.copy(),
            mass=2.0, radius=self.agent_radius,
            restitution=0.3, is_static=False,
        )

        for i in range(len(all_bodies)):
            for j in range(i + 1, len(all_bodies)):
                self._resolve_pair(all_bodies[i], all_bodies[j])

            if all_bodies[i].id != self.held_object:
                self._resolve_pair_agent(all_bodies[i], agent_sphere)

        self.agent_vel = agent_sphere.velocity.copy()

    def _check_collision(self, a: PhysicsObject3D, b: PhysicsObject3D) -> Tuple[bool, float, np.ndarray]:
        """
        多形状碰撞检测

        Returns:
            (colliding, penetration_depth, collision_normal)
        """
        # 快速剔除：包围球检测
        delta = b.position - a.position
        dist = np.linalg.norm(delta)
        bound_a = a.get_bounding_radius()
        bound_b = b.get_bounding_radius()

        if dist >= bound_a + bound_b or dist < 1e-6:
            return False, 0.0, np.zeros(3)

        # 具体形状碰撞
        if a.shape == 'sphere' and b.shape == 'sphere':
            return self._check_sphere_sphere(a, b)
        elif a.shape == 'sphere' and b.shape == 'cube':
            return self._check_sphere_cube(a, b)
        elif a.shape == 'cube' and b.shape == 'sphere':
            colliding, pen, normal = self._check_sphere_cube(b, a)
            return colliding, pen, -normal  # 法线反转
        elif a.shape == 'cube' and b.shape == 'cube':
            return self._check_cube_cube(a, b)
        else:
            # cylinder 和其他：用包围球近似
            return self._check_sphere_sphere_approx(a, b, dist)

    def _check_sphere_sphere(self, a: PhysicsObject3D, b: PhysicsObject3D) -> Tuple[bool, float, np.ndarray]:
        delta = b.position - a.position
        dist = np.linalg.norm(delta)
        min_dist = a.size[0] + b.size[0]

        if dist >= min_dist or dist < 1e-6:
            return False, 0.0, np.zeros(3)

        normal = delta / dist
        penetration = min_dist - dist
        return True, penetration, normal

    def _check_sphere_sphere_approx(self, a: PhysicsObject3D, b: PhysicsObject3D,
                                     dist: float) -> Tuple[bool, float, np.ndarray]:
        delta = b.position - a.position
        min_dist = a.get_bounding_radius() + b.get_bounding_radius()

        if dist >= min_dist or dist < 1e-6:
            return False, 0.0, np.zeros(3)

        normal = delta / dist
        penetration = min_dist - dist
        return True, penetration, normal

    def _check_sphere_cube(self, sphere: PhysicsObject3D,
                            cube: PhysicsObject3D) -> Tuple[bool, float, np.ndarray]:
        """球体 vs 立方体（AABB 近似）"""
        # 球心在立方体局部坐标系中的位置
        local = sphere.position - cube.position

        # AABB 半尺寸
        half = cube.size

        # 最近点（clamp）
        closest = np.clip(local, -half, half)

        # 最近点到球心的距离
        diff = local - closest
        dist = np.linalg.norm(diff)

        if dist >= sphere.size[0] or dist < 1e-6:
            # 球心在 AABB 内部
            if np.all(np.abs(local) < half):
                # 穿透：找最浅方向
                penetrations = half - np.abs(local)
                min_pen_axis = np.argmin(penetrations)
                normal = np.zeros(3)
                normal[min_pen_axis] = np.sign(local[min_pen_axis])
                return True, penetrations[min_pen_axis], normal
            return False, 0.0, np.zeros(3)

        normal = diff / dist
        penetration = sphere.size[0] - dist
        return True, penetration, normal

    def _check_cube_cube(self, a: PhysicsObject3D, b: PhysicsObject3D) -> Tuple[bool, float, np.ndarray]:
        """立方体 vs 立方体（AABB 重叠）"""
        delta = b.position - a.position
        overlap = a.size + b.size - np.abs(delta)

        if np.any(overlap <= 0):
            return False, 0.0, np.zeros(3)

        # 找最小穿透轴
        min_axis = np.argmin(overlap)
        normal = np.zeros(3)
        normal[min_axis] = np.sign(delta[min_axis])
        return True, overlap[min_axis], normal

    def _resolve_pair(self, a: PhysicsObject3D, b: PhysicsObject3D):
        """两个物体之间的碰撞响应"""
        if a.is_static and b.is_static:
            return

        colliding, penetration, normal = self._check_collision(a, b)
        if not colliding:
            return

        # 分离物体
        total_mass = a.get_effective_mass() + b.get_effective_mass()
        if a.is_static:
            b.position += normal * penetration
        elif b.is_static:
            a.position -= normal * penetration
        else:
            a.position -= normal * penetration * (b.get_effective_mass() / total_mass)
            b.position += normal * penetration * (a.get_effective_mass() / total_mass)

        # 相对速度
        rel_vel = a.velocity - b.velocity
        vel_along_normal = np.dot(rel_vel, normal)

        if vel_along_normal > 0:
            return

        e = min(a.restitution, b.restitution)
        j = -(1 + e) * vel_along_normal
        if a.is_static or b.is_static:
            j /= (1 / a.get_effective_mass() if not a.is_static else 1 / b.get_effective_mass())
        else:
            j /= (1 / a.get_effective_mass() + 1 / b.get_effective_mass())

        if not a.is_static and a.id != self.held_object:
            a.velocity += (j / a.get_effective_mass()) * normal
        if not b.is_static and b.id != self.held_object:
            b.velocity -= (j / b.get_effective_mass()) * normal

        # 碰撞音频
        impact_speed = abs(vel_along_normal)
        if impact_speed > 0.5:
            self._audio_events.append(AudioEvent3D(
                event_type='collision',
                amplitude=min(1.0, impact_speed / 5.0),
                frequency=MATERIAL_FREQUENCY.get(a.material, 500.0),
                direction=(a.position + b.position) / 2 - self.agent_pos,
                source_id=a.id, material=a.material,
            ))

    def _resolve_pair_agent(self, obj: PhysicsObject3D, agent: PhysicsObject3D):
        """物体与 agent 的碰撞"""
        colliding, penetration, normal = self._check_collision(obj, agent)
        if not colliding:
            return

        obj.position -= normal * penetration

        rel_vel = obj.velocity - agent.velocity
        vel_along_normal = np.dot(rel_vel, normal)
        if vel_along_normal > 0:
            return

        e = obj.restitution
        j = -(1 + e) * vel_along_normal / (1 / obj.get_effective_mass() + 1 / agent.mass)

        if obj.id != self.held_object:
            obj.velocity -= (j / obj.get_effective_mass()) * normal
        agent.velocity += (j / agent.mass) * normal

    # ----------------------------------------------------------
    # 约束
    # ----------------------------------------------------------

    def _ground_constraints(self):
        for obj in self.objects:
            if obj.is_static or obj.id == self.held_object:
                continue
            ground_z = obj.size[2]  # 物体底部高度
            if obj.position[2] - ground_z < 0:
                obj.position[2] = ground_z
                if obj.velocity[2] < 0:
                    obj.velocity[2] = -obj.velocity[2] * obj.restitution
                    if abs(obj.velocity[2]) < 0.3:
                        obj.velocity[2] = 0
                        obj.on_ground = True
                        self._audio_events.append(AudioEvent3D(
                            event_type='ground_hit',
                            amplitude=min(1.0, abs(obj.velocity[2]) / 3.0),
                            frequency=MATERIAL_FREQUENCY.get(obj.material, 500.0),
                            direction=obj.position - self.agent_pos,
                            source_id=obj.id, material=obj.material,
                        ))
            else:
                obj.on_ground = False

        if self.agent_pos[2] - self.agent_radius < 0:
            self.agent_pos[2] = self.agent_radius
            if self.agent_vel[2] < 0:
                self.agent_vel[2] = 0

    def _boundary_constraints(self):
        for obj in self.objects:
            if obj.is_static or obj.id == self.held_object:
                continue
            for axis in range(3):
                lo = obj.size[axis]
                hi = self.bounds[axis] - obj.size[axis]
                if obj.position[axis] < lo:
                    obj.position[axis] = lo
                    obj.velocity[axis] = abs(obj.velocity[axis]) * obj.restitution
                elif obj.position[axis] > hi:
                    obj.position[axis] = hi
                    obj.velocity[axis] = -abs(obj.velocity[axis]) * obj.restitution

        for axis in range(3):
            lo = self.agent_radius
            hi = self.bounds[axis] - self.agent_radius
            self.agent_pos[axis] = np.clip(self.agent_pos[axis], lo, hi)
            if self.agent_pos[axis] in (lo, hi):
                self.agent_vel[axis] = 0

    def _apply_friction(self):
        for obj in self.objects:
            if obj.is_static or obj.id == self.held_object:
                continue
            mat_friction = MATERIALS.get(obj.material, {}).get('friction', 0.5)
            if obj.on_ground:
                obj.velocity[0] *= (1.0 - mat_friction * 0.3)
                obj.velocity[1] *= (1.0 - mat_friction * 0.3)
            else:
                obj.velocity *= self.air_damping

        if self.agent_pos[2] - self.agent_radius < 0.05:
            self.agent_vel[0] *= self.friction
            self.agent_vel[1] *= self.friction

    def _update_held_object(self):
        if self.held_object is None:
            return
        obj = self._get_object(self.held_object)
        if obj:
            facing_dir = np.array([np.cos(self.agent_facing), np.sin(self.agent_facing), 0.0])
            obj.position = self.agent_pos + facing_dir * 0.8 + np.array([0, 0, 0.5])
            obj.velocity = self.agent_vel.copy()

    # ----------------------------------------------------------
    # 渲染（多形状）
    # ----------------------------------------------------------

    def _render(self) -> Tuple[np.ndarray, np.ndarray]:
        size = self._render_size
        visual = np.zeros((size, size, 3))
        depth_buf = np.full((size, size), self._far)

        cam_pos = self.agent_pos.copy()
        cam_dir = np.array([np.cos(self.agent_facing), np.sin(self.agent_facing), 0.0])

        forward = cam_dir / (np.linalg.norm(cam_dir) + 1e-8)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([0.0, 1.0, 0.0])
        else:
            right = right / right_norm
        up = np.cross(right, forward)

        fov_rad = np.radians(self._fov)
        f = 1.0 / np.tan(fov_rad / 2)

        for obj in self.objects:
            if obj.id == self.held_object:
                continue

            to_obj = obj.position - cam_pos
            x_cam = np.dot(to_obj, right)
            y_cam = np.dot(to_obj, up)
            z_cam = np.dot(to_obj, forward)

            if z_cam < self._near or z_cam > self._far:
                continue

            x_screen = (f) * x_cam / z_cam
            y_screen = f * y_cam / z_cam

            px = int((x_screen + 1) * size / 2)
            py = int((1 - y_screen) * size / 2)

            proj_radius = max(1, int(obj.get_bounding_radius() * f / z_cam * size / 2))

            color = obj.color.copy()
            brightness = max(0.3, 1.0 - z_cam / self._far)
            color = color * brightness

            # 按形状渲染
            if obj.shape == 'sphere':
                self._draw_sphere(visual, depth_buf, px, py, proj_radius, z_cam, color)
            elif obj.shape == 'cube':
                self._draw_cube(visual, depth_buf, px, py, proj_radius, z_cam, color)
            else:  # cylinder
                self._draw_cylinder(visual, depth_buf, px, py, proj_radius, z_cam, color)

        # 流体粒子渲染（Phase 48）
        if self._fluid_system:
            fluid_color = np.array([0.2, 0.4, 0.9])  # 蓝色
            for p in self._fluid_system.particles:
                to_pt = p.position - cam_pos
                z_cam = np.dot(to_pt, forward)
                if self._near < z_cam < self._far:
                    x_screen = f * np.dot(to_pt, right) / z_cam
                    y_screen = f * np.dot(to_pt, up) / z_cam
                    px = int((x_screen + 1) * size / 2)
                    py = int((1 - y_screen) * size / 2)
                    pr = max(1, int(0.1 * f / z_cam))
                    for dy in range(-pr, pr + 1):
                        for dx in range(-pr, pr + 1):
                            if dx*dx + dy*dy <= pr*pr:
                                sx, sy = px + dx, py + dy
                                if 0 <= sx < size and 0 <= sy < size:
                                    if z_cam < depth_buf[sy, sx]:
                                        brightness = max(0.3, 1.0 - z_cam / self._far)
                                        visual[sy, sx] = fluid_color * brightness
                                        depth_buf[sy, sx] = z_cam

        # 软体渲染（Phase 48）— 绘制节点和弹簧
        for sb in self._soft_bodies:
            soft_color = np.array([0.8, 0.5, 0.3])  # 橙色
            spring_color = np.array([0.5, 0.3, 0.2])  # 暗橙色
            # 绘制弹簧（线段）
            if hasattr(sb, 'springs') and hasattr(sb, 'points'):
                for spring in sb.springs:
                    p1 = sb.points[spring.point1_idx]
                    p2 = sb.points[spring.point2_idx]
                    pos1 = np.array([p1.x, p1.y, p1.z])
                    pos2 = np.array([p2.x, p2.y, p2.z])
                    # 简化：只绘制中点
                    mid = (pos1 + pos2) / 2
                    to_pt = mid - cam_pos
                    z_cam = np.dot(to_pt, forward)
                    if self._near < z_cam < self._far:
                        x_screen = f * np.dot(to_pt, right) / z_cam
                        y_screen = f * np.dot(to_pt, up) / z_cam
                        px = int((x_screen + 1) * size / 2)
                        py = int((1 - y_screen) * size / 2)
                        if 0 <= px < size and 0 <= py < size:
                            if z_cam < depth_buf[py, px]:
                                brightness = max(0.3, 1.0 - z_cam / self._far)
                                visual[py, px] = spring_color * brightness
                                depth_buf[py, px] = z_cam
                # 绘制节点
                for pt in sb.points:
                    pos = np.array([pt.x, pt.y, pt.z])
                    to_pt = pos - cam_pos
                    z_cam = np.dot(to_pt, forward)
                    if self._near < z_cam < self._far:
                        x_screen = f * np.dot(to_pt, right) / z_cam
                        y_screen = f * np.dot(to_pt, up) / z_cam
                        px = int((x_screen + 1) * size / 2)
                        py = int((1 - y_screen) * size / 2)
                        pr = max(1, int(0.15 * f / z_cam))
                        for dy in range(-pr, pr + 1):
                            for dx in range(-pr, pr + 1):
                                if dx*dx + dy*dy <= pr*pr:
                                    sx, sy = px + dx, py + dy
                                    if 0 <= sx < size and 0 <= sy < size:
                                        if z_cam < depth_buf[sy, sx]:
                                            brightness = max(0.3, 1.0 - z_cam / self._far)
                                            visual[sy, sx] = soft_color * brightness
                                            depth_buf[sy, sx] = z_cam

        # 地面网格
        for gx in range(int(self.bounds[0]) + 1):
            for gy in range(int(self.bounds[1]) + 1):
                ground_pt = np.array([float(gx), float(gy), 0.0])
                to_pt = ground_pt - cam_pos
                x_cam = np.dot(to_pt, right)
                y_cam = np.dot(to_pt, up)
                z_cam = np.dot(to_pt, forward)
                if self._near < z_cam < self._far:
                    x_screen = f * x_cam / z_cam
                    y_screen = f * y_cam / z_cam
                    px = int((x_screen + 1) * size / 2)
                    py = int((1 - y_screen) * size / 2)
                    if 0 <= px < size and 0 <= py < size:
                        if z_cam < depth_buf[py, px]:
                            visual[py, px] = np.array([0.15, 0.15, 0.15])
                            depth_buf[py, px] = z_cam

        depth_buf = np.clip(depth_buf / self._far, 0, 1)
        return visual, depth_buf

    def _draw_sphere(self, visual, depth_buf, px, py, radius, z_cam, color):
        """绘制圆形（球体投影）"""
        size = self._render_size
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    sx, sy = px + dx, py + dy
                    if 0 <= sx < size and 0 <= sy < size:
                        if z_cam < depth_buf[sy, sx]:
                            visual[sy, sx] = color
                            depth_buf[sy, sx] = z_cam

    def _draw_cube(self, visual, depth_buf, px, py, radius, z_cam, color):
        """绘制矩形（立方体投影）"""
        size = self._render_size
        w = int(radius * 0.9)
        h = int(radius * 1.1)
        for dy in range(-h, h + 1):
            for dx in range(-w, w + 1):
                sx, sy = px + dx, py + dy
                if 0 <= sx < size and 0 <= sy < size:
                    if z_cam < depth_buf[sy, sx]:
                        # 边缘稍微暗一点
                        edge = (abs(dx) == w or abs(dy) == h)
                        c = color * 0.7 if edge else color
                        visual[sy, sx] = c
                        depth_buf[sy, sx] = z_cam

    def _draw_cylinder(self, visual, depth_buf, px, py, radius, z_cam, color):
        """绘制椭圆（圆柱体投影）"""
        size = self._render_size
        rx = int(radius * 0.8)
        ry = int(radius * 1.2)
        for dy in range(-ry, ry + 1):
            for dx in range(-rx, rx + 1):
                if rx > 0 and ry > 0:
                    if (dx * dx) / (rx * rx + 1e-6) + (dy * dy) / (ry * ry + 1e-6) <= 1.0:
                        sx, sy = px + dx, py + dy
                        if 0 <= sx < size and 0 <= sy < size:
                            if z_cam < depth_buf[sy, sx]:
                                visual[sy, sx] = color
                                depth_buf[sy, sx] = z_cam

    # ----------------------------------------------------------
    # 本体感知（10 维）
    # ----------------------------------------------------------

    def _get_proprioception(self) -> np.ndarray:
        held_mass = 0.0
        held_flag = 0.0
        if self.held_object is not None:
            obj = self._get_object(self.held_object)
            if obj:
                held_mass = obj.get_effective_mass() / 10.0
                held_flag = 1.0

        speed = np.linalg.norm(self.agent_vel)
        facing_x = np.cos(self.agent_facing)
        facing_y = np.sin(self.agent_facing)

        return np.array([
            self.agent_vel[0] / 5.0,
            self.agent_vel[1] / 5.0,
            self.agent_vel[2] / 5.0,
            facing_x,
            facing_y,
            speed / 5.0,
            held_mass,
            held_flag,
            self.grip_strength,
            self.throw_ready,
        ])

    # ----------------------------------------------------------
    # 特征提取（用于语言涌现）
    # ----------------------------------------------------------

    def _get_object(self, obj_id: int) -> Optional[PhysicsObject3D]:
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None

    def get_object_features(self, obj_id: int) -> Dict:
        """获取物体的特征描述（14 维特征 → 字典）"""
        obj = self._get_object(obj_id)
        if obj is None:
            return {}

        features = {
            'material': obj.material,
            'shape': obj.shape,
            'size': 'big' if obj.mass > 2.0 else 'small',
            'weight': 'heavy' if obj.get_effective_mass() > 3.0 else (
                'light' if obj.get_effective_mass() < 1.0 else 'medium'),
            'bouncy': 'bouncy' if obj.restitution > 0.5 else 'stable',
            'hardness': MATERIALS.get(obj.material, {}).get('hardness', 'hard'),
        }

        speed = np.linalg.norm(obj.velocity)
        if speed > 3.0:
            features['motion'] = 'fast'
        elif speed > 0.5:
            features['motion'] = 'moving'
        else:
            features['motion'] = 'still'

        if obj.position[2] > 2.0:
            features['height'] = 'high'
        elif obj.position[2] < 0.5:
            features['height'] = 'low'
        else:
            features['height'] = 'mid'

        return features

    def get_nearby_objects(self, pos: np.ndarray, radius: float) -> List[Dict]:
        """获取指定位置附近的物体（不检查朝向）"""
        nearby = []
        for obj in self.objects:
            dist = np.linalg.norm(obj.position - pos)
            if dist < radius:
                features = self.get_object_features(obj.id)
                features['id'] = obj.id
                features['distance'] = dist
                nearby.append(features)
        return nearby

    def get_visible_objects(self) -> List[Dict]:
        visible = []
        cam_pos = self.agent_pos
        cam_dir = np.array([np.cos(self.agent_facing), np.sin(self.agent_facing), 0.0])

        for obj in self.objects:
            if obj.id == self.held_object:
                continue
            to_obj = obj.position - cam_pos
            dist = np.linalg.norm(to_obj)
            if dist > self._far:
                continue
            if dist > 0.01:
                cos_angle = np.dot(to_obj[:2], cam_dir[:2]) / (
                    np.linalg.norm(to_obj[:2]) * np.linalg.norm(cam_dir[:2]) + 1e-8)
                if cos_angle > 0.3:
                    features = self.get_object_features(obj.id)
                    features['id'] = obj.id
                    features['distance'] = dist
                    visible.append(features)

        return visible

    def reset(self):
        self.objects.clear()
        self.next_id = 0
        self.agent_pos = np.array([self.bounds[0] / 2, self.bounds[1] / 2, 0.5])
        self.agent_vel = np.zeros(3)
        self.agent_facing = 0.0
        self.held_object = None
        self.grip_strength = 0.0
        self.throw_ready = 0.0
        self._audio_events = []
        self._fluid_system = None
        self._soft_bodies = []
