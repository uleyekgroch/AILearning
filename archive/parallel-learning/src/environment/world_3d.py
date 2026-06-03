"""
统一学习系统 — 3D 物理世界

实现 IEnvironment 接口的 3D 世界 + 多 Agent 环境。
从 mvl 的 PhysicsWorld3D / MultiAgent3DEnv 移植，
全部使用 torch.Tensor，零 numpy。

9 阶段物理管线:
  apply_action → gravity → update_positions → collisions → ground/boundary →
  friction → held_object → fluid/soft → render
"""

import torch
import math
import random as _random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from src.core.interfaces import IEnvironment
from src.core.config import LearnerConfig
from .objects_3d import (
    PhysicsObject3D, AudioEvent3D, Observation3D, Action3D,
    MATERIALS, MATERIAL_NAMES, MATERIAL_COLORS, MATERIAL_FREQUENCY,
    MATERIAL_RESTITUTION, MATERIAL_FRICTION,
)
from .physics_3d import (
    CollisionDetector3D, CollisionResolver3D,
    FluidRegion, SoftBody,
)


# ============================================================
# Agent 状态
# ============================================================

@dataclass
class AgentState3D:
    """3D Agent 的状态"""
    id: int
    pos: torch.Tensor       # (3,)
    vel: torch.Tensor       # (3,)
    facing: float = 0.0
    radius: float = 0.4
    held_object: Optional[int] = None
    grip_strength: float = 0.0
    throw_ready: float = 0.0


# ============================================================
# 3D 物理世界（IEnvironment 兼容）
# ============================================================

class World3D(IEnvironment):
    """
    3D 物理世界，实现 IEnvironment 接口

    observe() 返回 Dict[str, torch.Tensor] 以兼容 Learner。
    """

    # 阶段 → 场景复杂度
    STAGE_CONFIG = {
        'sensorimotor': {
            'num_objects': 3,
            'shapes': ('sphere',),
            'materials': ('wood', 'rubber'),
        },
        'early_preoperational': {
            'num_objects': 5,
            'shapes': ('sphere', 'cube'),
            'materials': ('wood', 'plastic', 'rubber'),
        },
        'late_preoperational': {
            'num_objects': 7,
            'shapes': ('sphere', 'cube'),
            'materials': ('wood', 'plastic', 'metal', 'rubber'),
        },
        'early_concrete': {
            'num_objects': 8,
            'shapes': ('sphere', 'cube', 'cylinder'),
            'materials': ('wood', 'plastic', 'metal', 'glass'),
        },
        'late_concrete': {
            'num_objects': 10,
            'shapes': ('sphere', 'cube', 'cylinder'),
            'materials': tuple(MATERIAL_NAMES[:6]),
        },
        'early_formal': {
            'num_objects': 10,
            'shapes': ('sphere', 'cube', 'cylinder'),
            'materials': tuple(MATERIAL_NAMES),
        },
        'late_formal': {
            'num_objects': 12,
            'shapes': ('sphere', 'cube', 'cylinder'),
            'materials': tuple(MATERIAL_NAMES),
        },
        'adolescent': {
            'num_objects': 15,
            'shapes': ('sphere', 'cube', 'cylinder'),
            'materials': tuple(MATERIAL_NAMES),
        },
    }

    def __init__(
        self,
        bounds: Tuple[float, float, float] = (10.0, 10.0, 5.0),
        dt: float = 0.02,
        gravity: float = -9.8,
        friction: float = 0.8,
        air_damping: float = 0.999,
    ):
        self.bounds = torch.tensor(bounds, dtype=torch.float32)
        self.dt = dt
        self.gravity = gravity
        self.friction = friction
        self.air_damping = air_damping

        self.objects: List[PhysicsObject3D] = []
        self._next_id = 0

        # Agent 状态
        self.agent_pos = torch.tensor(
            [bounds[0] / 2, bounds[1] / 2, 0.5], dtype=torch.float32)
        self.agent_vel = torch.zeros(3, dtype=torch.float32)
        self.agent_facing = 0.0
        self.agent_radius = 0.4
        self.held_object: Optional[int] = None
        self.grip_strength = 0.0
        self.throw_ready = 0.0

        # 音频事件缓存
        self._audio_events: List[AudioEvent3D] = []

        # 渲染参数（_far 供 get_nearby_objects 使用）
        self._far = 20.0

        # 流体区域 / 软体
        self._fluid_regions: List[FluidRegion] = []
        self._soft_bodies: List[SoftBody] = []

        # 访问区域追踪（好奇心奖励）
        self._visited_cells: Set[Tuple[int, int, int]] = set()
        self._step_count = 0
        self._stage = 'sensorimotor'

    # ── 物体管理 ──────────────────────────────────────────────────

    def add_object(
        self,
        position: torch.Tensor,
        mass: float = 1.0,
        radius: float = 0.3,
        material: str = 'wood',
        shape: str = 'sphere',
        size: Optional[torch.Tensor] = None,
        color: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
    ) -> int:
        """添加物体，返回 id"""
        if size is None:
            size = torch.tensor([radius, radius, radius], dtype=torch.float32)

        mat_rest = MATERIAL_RESTITUTION.get(material, 0.5)
        mat_color = MATERIAL_COLORS.get(
            material, torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32))

        obj = PhysicsObject3D(
            id=self._next_id,
            position=position.float().clone(),
            velocity=torch.zeros(3, dtype=torch.float32) if velocity is None
                       else velocity.float().clone(),
            mass=mass,
            radius=radius,
            restitution=mat_rest,
            material=material,
            shape=shape,
            size=size.float().clone(),
            color=color.float().clone() if color is not None else mat_color.clone(),
        )
        self.objects.append(obj)
        self._next_id += 1
        return obj.id

    def add_fluid_region(
        self,
        min_corner: torch.Tensor,
        max_corner: torch.Tensor,
        name: str = 'water',
        density: float = 1000.0,
        viscosity: float = 0.001,
        drag_coefficient: float = 0.47,
    ) -> None:
        """添加流体区域"""
        self._fluid_regions.append(FluidRegion(
            min_corner=min_corner.float(),
            max_corner=max_corner.float(),
            name=name, density=density,
            viscosity=viscosity, drag_coefficient=drag_coefficient,
        ))

    def add_soft_body(
        self,
        center: torch.Tensor,
        size: torch.Tensor,
        mass: float = 1.0,
        stiffness: float = 50.0,
    ) -> int:
        """添加软体，返回索引"""
        sb = SoftBody(center=center.float(), size=size.float(),
                      mass=mass, stiffness=stiffness)
        self._soft_bodies.append(sb)
        return len(self._soft_bodies) - 1

    def _get_object(self, obj_id: int) -> Optional[PhysicsObject3D]:
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None

    # ── IEnvironment 接口 ─────────────────────────────────────────

    def observe(self) -> Dict[str, torch.Tensor]:
        """
        返回多模态观测字典，兼容 IEnvironment。

        键: 'visual' (N,), 'audio' (13,), 'position' (3,)
        """
        # visual: 物体观测拼接后平均
        if not self.objects:
            visual = torch.zeros(23)  # 2+2+3+5+7+4 对应 objects.py 的独热编码
        else:
            obs_vecs = [self._object_to_observation(obj) for obj in self.objects]
            visual = torch.stack(obs_vecs).mean(dim=0)

        # audio: 碰撞 + 速度统计 + 物体属性统计
        speeds = [obj.velocity.norm().item() for obj in self.objects]
        radii = [obj.radius for obj in self.objects]
        masses = [obj.mass for obj in self.objects]
        n_col = len(self._audio_events)
        n_obj = len(self.objects)
        audio = torch.tensor([
            float(n_col),
            sum(speeds) / max(len(speeds), 1),
            max(speeds) if speeds else 0.0,
            min(speeds) if speeds else 0.0,
            float(n_obj),
            self.agent_pos[0].item(),
            self.agent_pos[1].item(),
            sum(radii) / max(len(radii), 1),   # 平均半径
            max(radii) if radii else 0.0,
            min(radii) if radii else 0.0,
            sum(masses) / max(len(masses), 1),  # 平均质量
            max(masses) if masses else 0.0,
            min(masses) if masses else 0.0,
        ], dtype=torch.float32)

        return {
            'visual': visual,
            'audio': audio,
            'position': self.agent_pos.clone(),
        }

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool]:
        """
        执行动作，返回 (obs, reward, done)

        action: int 的 0-dim tensor（离散）或 (6,) 连续向量
        """
        self._step_count += 1
        self._audio_events = []

        # 1. Agent 动作
        if action.numel() == 1:
            self._apply_action(int(action.item()))
        elif action.numel() == 6:
            self._apply_continuous_action(action)
        else:
            self._apply_action(int(action.item()))

        # 2. 施加重力
        for obj in self.objects:
            if not obj.is_static and obj.id != self.held_object:
                obj.velocity[2] += self.gravity * self.dt
        self.agent_vel[2] += self.gravity * self.dt

        # 3. 更新位置
        for obj in self.objects:
            if not obj.is_static and obj.id != self.held_object:
                obj.position = obj.position + obj.velocity * self.dt
        self.agent_pos = self.agent_pos + self.agent_vel * self.dt

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

        # 8.5 流体/软体更新
        self._update_fluid_and_soft()

        # 好奇心奖励
        cell = (
            int(self.agent_pos[0].item()),
            int(self.agent_pos[1].item()),
            int(self.agent_pos[2].item()),
        )
        is_new = cell not in self._visited_cells
        self._visited_cells.add(cell)
        reward = 1.0 if is_new else 0.0

        max_cells = int(
            self.bounds[0].item() * self.bounds[1].item() * self.bounds[2].item())
        done = self._step_count >= 500 or len(self._visited_cells) >= max_cells

        obs = self.observe()
        return obs, reward, done

    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境"""
        self.objects.clear()
        self._next_id = 0
        self.agent_pos = torch.tensor(
            [self.bounds[0].item() / 2, self.bounds[1].item() / 2, 0.5],
            dtype=torch.float32)
        self.agent_vel = torch.zeros(3, dtype=torch.float32)
        self.agent_facing = 0.0
        self.held_object = None
        self.grip_strength = 0.0
        self.throw_ready = 0.0
        self._audio_events.clear()
        self._visited_cells.clear()
        self._step_count = 0
        self._fluid_regions.clear()
        self._soft_bodies.clear()
        return self.observe()

    def configure_for_stage(self, stage: str) -> None:
        """按发展阶段配置环境"""
        self._stage = stage
        self.reset()
        cfg = self.STAGE_CONFIG.get(stage, self.STAGE_CONFIG['sensorimotor'])

        for i in range(cfg['num_objects']):
            pos = torch.tensor([
                _random.uniform(1.0, self.bounds[0].item() - 1.0),
                _random.uniform(1.0, self.bounds[1].item() - 1.0),
                _random.uniform(0.5, 2.0),
            ], dtype=torch.float32)
            mass = _random.uniform(0.5, 5.0)
            radius = _random.uniform(0.2, 0.5)
            mat = _random.choice(cfg['materials'])
            shape = _random.choice(cfg['shapes'])

            if shape == 'sphere':
                size = torch.tensor([radius, radius, radius], dtype=torch.float32)
            elif shape == 'cube':
                ry = radius * _random.uniform(0.5, 1.5)
                size = torch.tensor([radius, ry, radius], dtype=torch.float32)
            else:
                rz = radius * _random.uniform(0.5, 2.0)
                size = torch.tensor([radius, radius, rz], dtype=torch.float32)

            self.add_object(pos, mass=mass, radius=radius,
                            material=mat, shape=shape, size=size)

    # ── 动作 ──────────────────────────────────────────────────────

    def _apply_action(self, action: int) -> None:
        """离散动作"""
        move_speed = 2.0
        turn_speed = math.pi / 2

        fx = math.cos(self.agent_facing)
        fy = math.sin(self.agent_facing)

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
                      Action3D.LEFT, Action3D.RIGHT,
                      Action3D.UP, Action3D.DOWN):
            self.agent_vel = self.agent_vel * 0.5

    def _apply_continuous_action(self, action_vec: torch.Tensor) -> None:
        """
        连续动作：6 维向量
        [force_x, force_y, force_z, torque_z, grip, throw_speed]
        """
        action = action_vec.float().clamp(-1.0, 1.0)

        move_scale = 5.0
        turn_scale = math.pi / 2
        scale = move_scale * self.dt * 50

        self.agent_vel[0] += action[0].item() * scale
        self.agent_vel[1] += action[1].item() * scale
        self.agent_vel[2] += action[2].item() * scale
        self.agent_facing += action[3].item() * turn_scale

        grip = max(0.0, action[4].item())
        throw_speed = max(0.0, action[5].item())
        self.grip_strength = grip
        self.throw_ready = throw_speed

        if grip > 0.5 and self.held_object is None:
            self._try_grab()
        elif grip < 0.1 and self.held_object is not None:
            self._try_drop()

        if throw_speed > 0.5 and self.held_object is not None:
            self._try_throw_with_speed(throw_speed * 10.0)

        self.agent_vel = self.agent_vel * 0.7

    # ── Agent 交互 ────────────────────────────────────────────────

    def _try_grab(self) -> None:
        """抓取前方物体"""
        if self.held_object is not None:
            return
        facing_dir = torch.tensor(
            [math.cos(self.agent_facing), math.sin(self.agent_facing), 0.0],
            dtype=torch.float32)
        best_id = None
        best_dist = 1.5

        for obj in self.objects:
            if obj.is_static:
                continue
            to_obj = obj.position - self.agent_pos
            dist = float(to_obj.norm())
            if dist < best_dist and dist > 0.01:
                cos_a = float(torch.dot(to_obj[:2], facing_dir[:2])) / (
                    to_obj[:2].norm() * facing_dir[:2].norm() + 1e-8)
                if cos_a > 0.5:
                    best_dist = dist
                    best_id = obj.id

        if best_id is not None:
            self.held_object = best_id
            obj = self._get_object(best_id)
            if obj:
                self._audio_events.append(AudioEvent3D(
                    event_type='grab', amplitude=0.3, frequency=500.0,
                    direction=torch.zeros(3, dtype=torch.float32),
                    source_id=best_id, material=obj.material,
                ))

    def _try_drop(self) -> None:
        """放下物体"""
        if self.held_object is None:
            return
        obj = self._get_object(self.held_object)
        if obj:
            obj.velocity = self.agent_vel.clone() * 0.3
            self._audio_events.append(AudioEvent3D(
                event_type='drop', amplitude=0.2, frequency=400.0,
                direction=obj.position - self.agent_pos,
                source_id=obj.id, material=obj.material,
            ))
        self.held_object = None

    def _try_throw(self) -> None:
        """投掷物体"""
        self._try_throw_with_speed(8.0)

    def _try_throw_with_speed(self, speed: float) -> None:
        """以指定速度投掷"""
        if self.held_object is None:
            return
        obj = self._get_object(self.held_object)
        if obj:
            facing_dir = torch.tensor(
                [math.cos(self.agent_facing), math.sin(self.agent_facing), 0.3],
                dtype=torch.float32)
            facing_dir = facing_dir / (facing_dir.norm() + 1e-8)
            obj.velocity = facing_dir * speed + self.agent_vel * 0.5
            self._audio_events.append(AudioEvent3D(
                event_type='throw', amplitude=0.5, frequency=600.0,
                direction=obj.position - self.agent_pos,
                source_id=obj.id, material=obj.material,
            ))
        self.held_object = None

    def _try_push(self) -> None:
        """推动前方物体"""
        facing_dir = torch.tensor(
            [math.cos(self.agent_facing), math.sin(self.agent_facing), 0.0],
            dtype=torch.float32)
        push_force = 5.0

        for obj in self.objects:
            if obj.is_static:
                continue
            to_obj = obj.position - self.agent_pos
            dist = float(to_obj.norm())
            if dist < 1.5:
                cos_a = float(torch.dot(to_obj[:2], facing_dir[:2])) / (
                    to_obj[:2].norm() * facing_dir[:2].norm() + 1e-8)
                if cos_a > 0.5:
                    eff_mass = obj.get_effective_mass()
                    impulse = facing_dir * push_force / max(eff_mass, 0.1)
                    obj.velocity = obj.velocity + impulse
                    self._audio_events.append(AudioEvent3D(
                        event_type='push',
                        amplitude=min(1.0, push_force / eff_mass) * 0.4,
                        frequency=MATERIAL_FREQUENCY.get(obj.material, 500.0),
                        direction=to_obj, source_id=obj.id,
                        material=obj.material,
                    ))
                    break

    # ── 碰撞 ──────────────────────────────────────────────────────

    def _resolve_collisions(self) -> None:
        """物体-物体 + Agent-物体碰撞（空间哈希加速）"""
        from .physics_3d import SpatialHash

        n = len(self.objects)
        if n < 2:
            # Still do agent-object collisions
            agent_obj = PhysicsObject3D(
                id=-1, position=self.agent_pos.clone(), velocity=self.agent_vel.clone(),
                mass=2.0, radius=self.agent_radius, restitution=0.3, material='rubber',
            )
            for obj in self.objects:
                if obj.is_static or obj.id == self.held_object:
                    continue
                info = CollisionDetector3D.detect_sphere_sphere(agent_obj, obj)
                if info is not None:
                    CollisionResolver3D.resolve_sphere_sphere(agent_obj, obj, info)
                    self.agent_vel = agent_obj.velocity.clone()
            return

        # 空间哈希收集候选对
        max_radius = max(obj.get_bounding_radius() for obj in self.objects)
        cell_size = max(max_radius * 2.5, 1.0)
        spatial = SpatialHash(cell_size=cell_size)

        for i, obj in enumerate(self.objects):
            spatial.insert(i, obj.position, obj.get_bounding_radius())

        # 只检测候选对
        for i, j in spatial.query_pairs(n):
            a = self.objects[i]
            b = self.objects[j]
            if a.is_static and b.is_static:
                continue
            info = CollisionDetector3D.detect_sphere_sphere(a, b)
            if info is not None:
                CollisionResolver3D.resolve_sphere_sphere(a, b, info)
                rel_vel = a.velocity - b.velocity
                impact = abs(float(torch.dot(rel_vel, info['normal'])))
                if impact > 0.5:
                    self._audio_events.append(AudioEvent3D(
                        event_type='collision',
                        amplitude=min(1.0, impact / 5.0),
                        frequency=MATERIAL_FREQUENCY.get(a.material, 500.0),
                        direction=(a.position + b.position) / 2 - self.agent_pos,
                        source_id=a.id, material=a.material,
                    ))

        # Agent-物体碰撞
        agent_obj = PhysicsObject3D(
            id=-1, position=self.agent_pos.clone(), velocity=self.agent_vel.clone(),
            mass=2.0, radius=self.agent_radius, restitution=0.3, material='rubber',
        )
        for obj in self.objects:
            if obj.is_static or obj.id == self.held_object:
                continue
            info = CollisionDetector3D.detect_sphere_sphere(agent_obj, obj)
            if info is not None:
                CollisionResolver3D.resolve_sphere_sphere(agent_obj, obj, info)
                self.agent_vel = agent_obj.velocity.clone()

    # ── 约束 ──────────────────────────────────────────────────────

    def _ground_constraints(self) -> None:
        """地面约束"""
        for obj in self.objects:
            if obj.is_static or obj.id == self.held_object:
                continue
            ground_z = float(obj.size[2])
            if obj.position[2] - ground_z < 0:
                obj.position[2] = ground_z
                if obj.velocity[2] < 0:
                    obj.velocity[2] = -obj.velocity[2] * obj.restitution
                    if abs(obj.velocity[2].item()) < 0.3:
                        obj.velocity[2] = 0
                        obj.on_ground = True
                        self._audio_events.append(AudioEvent3D(
                            event_type='ground_hit',
                            amplitude=min(1.0, abs(obj.velocity[2].item()) / 3.0),
                            frequency=MATERIAL_FREQUENCY.get(obj.material, 500.0),
                            direction=obj.position - self.agent_pos,
                            source_id=obj.id, material=obj.material,
                        ))
            else:
                obj.on_ground = False

        # Agent 地面
        if self.agent_pos[2] - self.agent_radius < 0:
            self.agent_pos[2] = self.agent_radius
            if self.agent_vel[2] < 0:
                self.agent_vel[2] = 0

    def _boundary_constraints(self) -> None:
        """边界约束"""
        for obj in self.objects:
            if obj.is_static or obj.id == self.held_object:
                continue
            for axis in range(3):
                lo = float(obj.size[axis])
                hi = self.bounds[axis].item() - lo
                if obj.position[axis] < lo:
                    obj.position[axis] = lo
                    obj.velocity[axis] = abs(obj.velocity[axis]) * obj.restitution
                elif obj.position[axis] > hi:
                    obj.position[axis] = hi
                    obj.velocity[axis] = -abs(obj.velocity[axis]) * obj.restitution

        for axis in range(3):
            lo = self.agent_radius
            hi = self.bounds[axis].item() - self.agent_radius
            self.agent_pos[axis] = self.agent_pos[axis].clamp(lo, hi)

    def _apply_friction(self) -> None:
        """摩擦力"""
        for obj in self.objects:
            if obj.is_static or obj.id == self.held_object:
                continue
            mat_fric = MATERIALS.get(obj.material, {}).get('friction', 0.5)
            if obj.on_ground:
                obj.velocity[0] *= (1.0 - mat_fric * 0.3)
                obj.velocity[1] *= (1.0 - mat_fric * 0.3)
            else:
                obj.velocity = obj.velocity * self.air_damping

        if self.agent_pos[2] - self.agent_radius < 0.05:
            self.agent_vel[0] *= self.friction
            self.agent_vel[1] *= self.friction

    def _update_held_object(self) -> None:
        """更新持有物体位置"""
        if self.held_object is None:
            return
        obj = self._get_object(self.held_object)
        if obj:
            facing_dir = torch.tensor(
                [math.cos(self.agent_facing), math.sin(self.agent_facing), 0.0],
                dtype=torch.float32)
            obj.position = self.agent_pos + facing_dir * 0.8 + torch.tensor(
                [0.0, 0.0, 0.5], dtype=torch.float32)
            obj.velocity = self.agent_vel.clone()

    def _update_fluid_and_soft(self) -> None:
        """更新流体区域和软体"""
        # 流体对物体的作用
        for fluid in self._fluid_regions:
            for obj in self.objects:
                if obj.is_static or obj.id == self.held_object:
                    continue
                if fluid.contains_point(obj.position):
                    # 阻力
                    drag = fluid.drag_coefficient * fluid.viscosity
                    obj.velocity = obj.velocity * max(0.0, 1.0 - drag * self.dt)
                    # 浮力（简化）
                    submerged = fluid.get_submerged_volume(
                        obj.position, obj.get_bounding_radius())
                    buoyancy = fluid.density * 9.8 * submerged * 1e-4
                    obj.velocity[2] += buoyancy * self.dt

            # 流体对 Agent
            if fluid.contains_point(self.agent_pos):
                drag = fluid.drag_coefficient * fluid.viscosity
                self.agent_vel = self.agent_vel * max(0.0, 1.0 - drag * self.dt)

        # 软体更新
        for sb in self._soft_bodies:
            sb.update(self.dt, abs(self.gravity))
            # 软体与刚体碰撞
            for obj in self.objects:
                if not obj.is_static:
                    sb.collide_with_sphere(
                        obj.position, obj.get_bounding_radius(),
                        obj.velocity, obj.get_effective_mass())
            # 软体与 Agent 碰撞
            sb.collide_with_sphere(
                self.agent_pos, self.agent_radius, self.agent_vel, 2.0)

    # ── 本体感知 ──────────────────────────────────────────────────

    def _get_proprioception(self) -> torch.Tensor:
        """
        本体感知 (10,) 向量

        [vx, vy, vz, facing_x, facing_y, speed, held_mass, held_flag, grip, throw_ready]
        """
        held_mass = 0.0
        held_flag = 0.0
        if self.held_object is not None:
            obj = self._get_object(self.held_object)
            if obj:
                held_mass = obj.get_effective_mass() / 10.0
                held_flag = 1.0

        speed = float(self.agent_vel.norm())
        facing_x = math.cos(self.agent_facing)
        facing_y = math.sin(self.agent_facing)

        return torch.tensor([
            self.agent_vel[0].item() / 5.0,
            self.agent_vel[1].item() / 5.0,
            self.agent_vel[2].item() / 5.0,
            facing_x,
            facing_y,
            speed / 5.0,
            held_mass,
            held_flag,
            self.grip_strength,
            self.throw_ready,
        ], dtype=torch.float32)

    # ── 特征提取（供语言游戏使用） ────────────────────────────────

    def get_object_features(self, obj_id: int) -> Dict:
        """获取物体特征描述"""
        obj = self._get_object(obj_id)
        if obj is None:
            return {}

        eff_mass = obj.get_effective_mass()
        speed = float(obj.velocity.norm())
        features = {
            'material': obj.material,
            'shape': obj.shape,
            'size': 'big' if obj.mass > 2.0 else 'small',
            'weight': ('heavy' if eff_mass > 3.0 else
                       'light' if eff_mass < 1.0 else 'medium'),
            'bouncy': 'bouncy' if obj.restitution > 0.5 else 'stable',
            'hardness': MATERIALS.get(obj.material, {}).get('hardness', 'hard'),
        }

        if speed > 3.0:
            features['motion'] = 'fast'
        elif speed > 0.5:
            features['motion'] = 'moving'
        else:
            features['motion'] = 'still'

        z = obj.position[2].item()
        b2 = self.bounds[2].item()
        if z > b2 * 0.6:
            features['height'] = 'high'
        elif z < b2 * 0.3:
            features['height'] = 'low'
        else:
            features['height'] = 'mid'

        return features

    def get_nearby_objects(self, pos: torch.Tensor, radius: float) -> List[Dict]:
        """获取指定位置附近的物体"""
        nearby = []
        for obj in self.objects:
            dist = float((obj.position - pos).norm())
            if dist < radius:
                features = self.get_object_features(obj.id)
                features['id'] = obj.id
                features['distance'] = dist
                nearby.append(features)
        return nearby

    def get_visible_objects(self) -> List[Dict]:
        """获取 Agent 可见物体（视野锥内）"""
        visible = []
        cam_dir = torch.tensor(
            [math.cos(self.agent_facing), math.sin(self.agent_facing), 0.0],
            dtype=torch.float32)

        for obj in self.objects:
            if obj.id == self.held_object:
                continue
            to_obj = obj.position - self.agent_pos
            dist = float(to_obj.norm())
            if dist > self._far:
                continue
            if dist > 0.01:
                cos_a = float(torch.dot(to_obj[:2], cam_dir[:2])) / (
                    to_obj[:2].norm() * cam_dir[:2].norm() + 1e-8)
                if cos_a > 0.3:
                    features = self.get_object_features(obj.id)
                    features['id'] = obj.id
                    features['distance'] = dist
                    visible.append(features)
        return visible

    # ── 内部辅助 ──────────────────────────────────────────────────

    def _object_to_observation(self, obj: PhysicsObject3D) -> torch.Tensor:
        """
        物体 → 观测向量

        [x, y, z, vx, vy, vz, shape_oh(3), material_oh(7), color(3)]
        共 3+3+3+7+3 = 19 维
        """
        parts = [obj.position, obj.velocity]

        shapes = ('sphere', 'cube', 'cylinder')
        shape_vec = torch.zeros(len(shapes), dtype=torch.float32)
        if obj.shape in shapes:
            shape_vec[shapes.index(obj.shape)] = 1.0
        parts.append(shape_vec)

        mat_vec = torch.zeros(len(MATERIAL_NAMES), dtype=torch.float32)
        if obj.material in MATERIAL_NAMES:
            mat_vec[MATERIAL_NAMES.index(obj.material)] = 1.0
        parts.append(mat_vec)

        parts.append(obj.color)
        return torch.cat(parts)


# ============================================================
# 多 Agent 3D 环境
# ============================================================

class MultiAgent3DEnv:
    """
    多 Agent 共享 3D 物理环境

    核心:
    - 一个共享 World3D（物体、碰撞、重力）
    - 每个 Agent 独立的位置/速度/朝向/持有物体
    - 空间邻近通信（距离 < comm_range）
    - Agent-agent 碰撞检测
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float] = (12.0, 12.0, 5.0),
        gravity: float = -9.8,
        comm_range: float = 3.0,
    ):
        self.world = World3D(bounds=bounds, gravity=gravity)
        self.bounds = bounds
        self.comm_range = comm_range

        self.agents: Dict[int, AgentState3D] = {}
        self._next_agent_id = 0

    def add_agent(self, pos: Optional[torch.Tensor] = None) -> int:
        """添加一个 Agent 到共享世界"""
        agent_id = self._next_agent_id
        self._next_agent_id += 1

        if pos is None:
            pos = torch.tensor([
                _random.uniform(1.0, self.bounds[0] - 1.0),
                _random.uniform(1.0, self.bounds[1] - 1.0),
                0.5,
            ], dtype=torch.float32)

        self.agents[agent_id] = AgentState3D(
            id=agent_id,
            pos=pos.float().clone(),
            vel=torch.zeros(3, dtype=torch.float32),
        )
        return agent_id

    def step(
        self,
        actions: Dict[int, torch.Tensor],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        执行所有 Agent 的动作，返回各自的观测

        actions: {agent_id: action_tensor}
        返回:    {agent_id: {visual, audio, position}}
        """
        # 1. 执行每个 agent 的动作
        for agent_id, action in actions.items():
            if agent_id not in self.agents:
                continue
            self._apply_agent_action(agent_id, action)

        # 2. 施加重力
        for obj in self.world.objects:
            if not obj.is_static:
                obj.velocity[2] += self.world.gravity * self.world.dt

        # 3. 更新物体位置
        for obj in self.world.objects:
            if not obj.is_static:
                obj.position = obj.position + obj.velocity * self.world.dt

        # 4. 更新 agent 位置
        for agent in self.agents.values():
            agent.pos = agent.pos + agent.vel * self.world.dt

        # 5. 碰撞
        self._resolve_all_collisions()

        # 6. 地面/边界约束
        self._ground_constraints()
        self._boundary_constraints()

        # 7. 摩擦
        self._apply_friction()

        # 8. 更新持有物体
        for agent in self.agents.values():
            if agent.held_object is not None:
                obj = self.world._get_object(agent.held_object)
                if obj:
                    facing_dir = torch.tensor(
                        [math.cos(agent.facing), math.sin(agent.facing), 0.3],
                        dtype=torch.float32)
                    obj.position = agent.pos + facing_dir * 0.8

        # 9. 生成观测
        observations = {}
        for agent_id in actions:
            if agent_id in self.agents:
                observations[agent_id] = self._get_agent_observation(agent_id)
        return observations

    def _apply_agent_action(self, agent_id: int, action: torch.Tensor) -> None:
        """执行单个 agent 的动作"""
        agent = self.agents[agent_id]

        if action.numel() == 6:
            self._apply_continuous(agent, action)
        else:
            self._apply_discrete(agent, int(action.item()))

    def _apply_continuous(self, agent: AgentState3D, action: torch.Tensor) -> None:
        """连续动作"""
        a = action.float().clamp(-1.0, 1.0)
        scale = 5.0 * self.world.dt * 50

        agent.vel[0] += a[0].item() * scale
        agent.vel[1] += a[1].item() * scale
        agent.vel[2] += a[2].item() * scale
        agent.facing += a[3].item() * math.pi / 2

        grip = max(0.0, a[4].item())
        throw_speed = max(0.0, a[5].item())
        agent.grip_strength = grip
        agent.throw_ready = throw_speed

        if grip > 0.5 and agent.held_object is None:
            self._try_grab(agent)
        elif grip < 0.1 and agent.held_object is not None:
            agent.held_object = None

        if throw_speed > 0.5 and agent.held_object is not None:
            self._try_throw(agent, throw_speed * 10.0)

        agent.vel = agent.vel * 0.7

    def _apply_discrete(self, agent: AgentState3D, action: int) -> None:
        """离散动作"""
        move_speed = 2.0
        turn_speed = math.pi / 2
        fx = math.cos(agent.facing)
        fy = math.sin(agent.facing)

        if action == 0:
            agent.vel[0] += fx * move_speed
            agent.vel[1] += fy * move_speed
        elif action == 1:
            agent.vel[0] -= fx * move_speed
            agent.vel[1] -= fy * move_speed
        elif action == 2:
            agent.vel[0] += -fy * move_speed
            agent.vel[1] += fx * move_speed
        elif action == 3:
            agent.vel[0] += fy * move_speed
            agent.vel[1] -= fx * move_speed
        elif action == 4:
            agent.vel[2] += move_speed
        elif action == 5:
            agent.vel[2] -= move_speed
        elif action == 6:
            agent.facing -= turn_speed
        elif action == 7:
            agent.facing += turn_speed
        elif action == 8:
            self._try_grab(agent)
        elif action == 9:
            agent.held_object = None
        elif action == 10:
            self._try_throw(agent, 8.0)
        elif action == 11:
            self._try_push(agent)

        agent.vel = agent.vel * 0.5

    def _try_grab(self, agent: AgentState3D) -> None:
        """尝试抓取"""
        if agent.held_object is not None:
            return
        facing_dir = torch.tensor(
            [math.cos(agent.facing), math.sin(agent.facing), 0.0],
            dtype=torch.float32)
        best_id = None
        best_dist = 1.5

        held_ids = {a.held_object for a in self.agents.values()
                    if a.held_object is not None}

        for obj in self.world.objects:
            if obj.is_static or obj.id in held_ids:
                continue
            to_obj = obj.position - agent.pos
            dist = float(to_obj.norm())
            if dist < best_dist and dist > 0.01:
                cos_a = float(torch.dot(to_obj[:2], facing_dir[:2])) / (
                    to_obj[:2].norm() * facing_dir[:2].norm() + 1e-8)
                if cos_a > 0.5:
                    best_dist = dist
                    best_id = obj.id

        if best_id is not None:
            agent.held_object = best_id

    def _try_throw(self, agent: AgentState3D, speed: float) -> None:
        """投掷物体"""
        if agent.held_object is None:
            return
        obj = self.world._get_object(agent.held_object)
        if obj:
            facing_dir = torch.tensor(
                [math.cos(agent.facing), math.sin(agent.facing), 0.3],
                dtype=torch.float32)
            facing_dir = facing_dir / (facing_dir.norm() + 1e-8)
            obj.velocity = facing_dir * speed + agent.vel * 0.5
        agent.held_object = None

    def _try_push(self, agent: AgentState3D) -> None:
        """推动物体"""
        facing_dir = torch.tensor(
            [math.cos(agent.facing), math.sin(agent.facing), 0.0],
            dtype=torch.float32)
        for obj in self.world.objects:
            if obj.is_static:
                continue
            to_obj = obj.position - agent.pos
            dist = float(to_obj.norm())
            if dist < 2.0 and dist > 0.01:
                cos_a = float(torch.dot(to_obj[:2], facing_dir[:2])) / (
                    to_obj[:2].norm() * facing_dir[:2].norm() + 1e-8)
                if cos_a > 0.5:
                    eff_mass = obj.get_effective_mass()
                    impulse = facing_dir * 8.0 / eff_mass
                    obj.velocity = obj.velocity + impulse * 0.1
                    break

    def _resolve_all_collisions(self) -> None:
        """物体-物体 + agent-物体 + agent-agent 碰撞（空间哈希加速）"""
        from .physics_3d import SpatialHash

        # 物体-物体（空间哈希）
        n = len(self.world.objects)
        if n >= 2:
            max_radius = max(obj.get_bounding_radius() for obj in self.world.objects)
            cell_size = max(max_radius * 2.5, 1.0)
            spatial = SpatialHash(cell_size=cell_size)
            for i, obj in enumerate(self.world.objects):
                spatial.insert(i, obj.position, obj.get_bounding_radius())
            for i, j in spatial.query_pairs(n):
                a = self.world.objects[i]
                b = self.world.objects[j]
                if a.is_static and b.is_static:
                    continue
                info = CollisionDetector3D.detect_sphere_sphere(a, b)
                if info is not None:
                    CollisionResolver3D.resolve_sphere_sphere(a, b, info)

        # Agent-物体
        for agent in self.agents.values():
            agent_sphere = PhysicsObject3D(
                id=-1, position=agent.pos.clone(), velocity=agent.vel.clone(),
                mass=1.0, radius=agent.radius, restitution=0.3, material='rubber',
            )
            for obj in self.world.objects:
                if not obj.is_static:
                    info = CollisionDetector3D.detect_sphere_sphere(agent_sphere, obj)
                    if info is not None:
                        CollisionResolver3D.resolve_sphere_sphere(agent_sphere, obj, info)
            agent.vel = agent_sphere.velocity.clone()

        # Agent-agent（数量通常不多，直接检测）
        agent_list = list(self.agents.values())
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a = agent_list[i]
                b = agent_list[j]
                diff = b.pos - a.pos
                dist = float(diff.norm())
                min_dist = a.radius + b.radius
                if dist < min_dist and dist > 1e-3:
                    normal = diff / dist
                    overlap = min_dist - dist
                    a.pos = a.pos - normal * overlap * 0.5
                    b.pos = b.pos + normal * overlap * 0.5
                    rel_vel = a.vel - b.vel
                    vn = float(torch.dot(rel_vel, normal))
                    if vn > 0:
                        a.vel = a.vel - normal * vn * 0.5
                        b.vel = b.vel + normal * vn * 0.5

    def _ground_constraints(self) -> None:
        """地面约束"""
        for agent in self.agents.values():
            if agent.pos[2] < agent.radius:
                agent.pos[2] = agent.radius
                if agent.vel[2] < 0:
                    agent.vel[2] = 0

        held_ids = {a.held_object for a in self.agents.values()
                    if a.held_object is not None}
        for obj in self.world.objects:
            if obj.is_static or obj.id in held_ids:
                continue
            gz = float(obj.size[2]) if obj.size is not None else obj.radius
            if obj.position[2] < gz:
                obj.position[2] = gz
                if obj.velocity[2] < 0:
                    obj.velocity[2] = -obj.velocity[2] * obj.restitution
                    if abs(obj.velocity[2].item()) < 0.3:
                        obj.velocity[2] = 0

    def _boundary_constraints(self) -> None:
        """边界约束"""
        r = 0.3
        for agent in self.agents.values():
            for axis in range(3):
                lo = r
                hi = self.bounds[axis] - r
                if agent.pos[axis] < lo:
                    agent.pos[axis] = lo
                    agent.vel[axis] = abs(agent.vel[axis]) * 0.3
                elif agent.pos[axis] > hi:
                    agent.pos[axis] = hi
                    agent.vel[axis] = -abs(agent.vel[axis]) * 0.3

    def _apply_friction(self) -> None:
        """地面摩擦"""
        for agent in self.agents.values():
            if agent.pos[2] <= agent.radius + 0.05:
                agent.vel[0] *= 0.85
                agent.vel[1] *= 0.85

    def _get_agent_observation(self, agent_id: int) -> Dict[str, torch.Tensor]:
        """获取指定 agent 的观测"""
        agent = self.agents[agent_id]

        # 临时设置 world 的 agent 状态用于渲染
        orig_pos = self.world.agent_pos.clone()
        orig_facing = self.world.agent_facing
        orig_held = self.world.held_object

        self.world.agent_pos = agent.pos.clone()
        self.world.agent_facing = agent.facing
        self.world.held_object = agent.held_object

        obs = self.world.observe()

        # 恢复
        self.world.agent_pos = orig_pos
        self.world.agent_facing = orig_facing
        self.world.held_object = orig_held

        return obs

    def get_nearby_agents(self, agent_id: int) -> List[int]:
        """获取指定 agent 附近的其他 agent"""
        if agent_id not in self.agents:
            return []
        agent = self.agents[agent_id]
        nearby = []
        for other_id, other in self.agents.items():
            if other_id == agent_id:
                continue
            dist = float((agent.pos - other.pos).norm())
            if dist < self.comm_range:
                nearby.append(other_id)
        return nearby

    def get_nearby_pairs(self) -> List[Tuple[int, int]]:
        """获取所有在通信范围内的 agent 对"""
        pairs = []
        ids = list(self.agents.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = self.agents[ids[i]]
                b = self.agents[ids[j]]
                dist = float((a.pos - b.pos).norm())
                if dist < self.comm_range:
                    pairs.append((ids[i], ids[j]))
        return pairs
