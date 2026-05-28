"""
多 Agent 共享 3D 物理环境

包装 PhysicsWorld3D，支持多个 Agent 在同一物理空间中共存、交互、协作。

设计：
- 一个共享 PhysicsWorld3D（物体、碰撞、重力）
- 每个 Agent 独立的位置/速度/朝向/持有物体
- 空间邻近通信（距离 < comm_range）
- Agent-agent 碰撞检测
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from environment_3d import (
    PhysicsWorld3D, PhysicsObject3D, Observation3D,
    AudioEvent3D, MATERIALS, MATERIAL_NAMES
)


@dataclass
class AgentState:
    """单个 Agent 的状态（在共享世界中）"""
    id: int
    pos: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 0.5]))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    facing: float = 0.0
    radius: float = 0.4
    held_object: Optional[int] = None
    grip_strength: float = 0.0
    throw_ready: float = 0.0


class MultiAgent3DEnv:
    """
    多 Agent 共享 3D 物理环境

    核心功能：
    - 多个 Agent 共享同一个物理世界
    - 空间邻近检测（通信、观察）
    - Agent-agent 碰撞
    - 每个 Agent 独立的观察视角
    """

    def __init__(self, bounds: Tuple[float, float, float] = (12.0, 12.0, 5.0),
                 gravity: float = -9.8, comm_range: float = 3.0):
        self.physics = PhysicsWorld3D(bounds=bounds, gravity=gravity)
        self.bounds = bounds
        self.comm_range = comm_range

        # Agent 状态
        self.agents: Dict[int, AgentState] = {}
        self._next_agent_id = 0

        # 通信记录
        self.comm_log: List[Dict] = []

    def add_agent(self, pos: Optional[np.ndarray] = None) -> int:
        """添加一个 Agent 到共享世界"""
        agent_id = self._next_agent_id
        self._next_agent_id += 1

        if pos is None:
            pos = np.array([
                np.random.uniform(1.0, self.bounds[0] - 1.0),
                np.random.uniform(1.0, self.bounds[1] - 1.0),
                0.5,
            ])

        self.agents[agent_id] = AgentState(id=agent_id, pos=pos.copy())
        return agent_id

    def step(self, actions: Dict[int, object]) -> Dict[int, Observation3D]:
        """
        执行所有 Agent 的动作，返回各自的观察

        Args:
            actions: {agent_id: action}，action 可以是 int 或 np.ndarray

        Returns:
            {agent_id: Observation3D}
        """
        observations = {}

        # 1. 执行每个 agent 的动作
        for agent_id, action in actions.items():
            if agent_id not in self.agents:
                continue
            self._apply_agent_action(agent_id, action)

        # 2. 施加重力到所有物体
        for obj in self.physics.objects:
            if not obj.is_static:
                obj.velocity[2] += self.physics.gravity * self.physics.dt

        # 3. 更新物体位置
        for obj in self.physics.objects:
            if not obj.is_static:
                obj.position += obj.velocity * self.physics.dt

        # 4. 更新 agent 位置
        for agent in self.agents.values():
            agent.pos += agent.vel * self.physics.dt

        # 5. 碰撞检测（物体-物体 + agent-物体 + agent-agent）
        self._resolve_all_collisions()

        # 6. 地面/边界约束
        self._ground_constraints()
        self._boundary_constraints()

        # 7. 摩擦
        self._apply_friction()

        # 8. 更新持有物体位置
        for agent in self.agents.values():
            if agent.held_object is not None:
                obj = self.physics._get_object(agent.held_object)
                if obj:
                    facing_dir = np.array([
                        np.cos(agent.facing), np.sin(agent.facing), 0.3
                    ])
                    obj.position = agent.pos + facing_dir * 0.8

        # 9. 为每个 agent 生成观察
        for agent_id in actions:
            if agent_id in self.agents:
                observations[agent_id] = self.get_agent_observation(agent_id)

        return observations

    def _apply_agent_action(self, agent_id: int, action):
        """执行单个 agent 的动作"""
        agent = self.agents[agent_id]

        if isinstance(action, np.ndarray):
            self._apply_continuous(agent, action)
        else:
            self._apply_discrete(agent, int(action))

    def _apply_continuous(self, agent: AgentState, action: np.ndarray):
        """连续动作"""
        action = np.clip(action, -1.0, 1.0)
        move_scale = 5.0
        turn_scale = np.pi / 2

        agent.vel[0] += action[0] * move_scale * self.physics.dt * 50
        agent.vel[1] += action[1] * move_scale * self.physics.dt * 50
        agent.vel[2] += action[2] * move_scale * self.physics.dt * 50
        agent.facing += action[3] * turn_scale

        grip = max(0.0, action[4])
        throw_speed = max(0.0, action[5])
        agent.grip_strength = grip
        agent.throw_ready = throw_speed

        if grip > 0.5 and agent.held_object is None:
            self._try_grab(agent)
        elif grip < 0.1 and agent.held_object is not None:
            self._try_drop(agent)

        if throw_speed > 0.5 and agent.held_object is not None:
            self._try_throw(agent, throw_speed * 10.0)

        agent.vel *= 0.7  # 速度衰减

    def _apply_discrete(self, agent: AgentState, action: int):
        """离散动作"""
        move_speed = 2.0
        turn_speed = np.pi / 2
        fx = np.cos(agent.facing)
        fy = np.sin(agent.facing)

        if action == 0:  # FORWARD
            agent.vel[0] += fx * move_speed
            agent.vel[1] += fy * move_speed
        elif action == 1:  # BACKWARD
            agent.vel[0] -= fx * move_speed
            agent.vel[1] -= fy * move_speed
        elif action == 2:  # LEFT
            agent.vel[0] += -fy * move_speed
            agent.vel[1] += fx * move_speed
        elif action == 3:  # RIGHT
            agent.vel[0] += fy * move_speed
            agent.vel[1] -= fx * move_speed
        elif action == 4:  # UP
            agent.vel[2] += move_speed
        elif action == 5:  # DOWN
            agent.vel[2] -= move_speed
        elif action == 6:  # TURN_LEFT
            agent.facing -= turn_speed
        elif action == 7:  # TURN_RIGHT
            agent.facing += turn_speed
        elif action == 8:  # GRAB
            self._try_grab(agent)
        elif action == 9:  # DROP
            self._try_drop(agent)
        elif action == 10:  # THROW
            self._try_throw(agent, 8.0)
        elif action == 11:  # PUSH
            self._try_push(agent)

        agent.vel *= 0.5

    def _try_grab(self, agent: AgentState):
        """尝试抓取前方物体"""
        if agent.held_object is not None:
            return
        facing_dir = np.array([np.cos(agent.facing), np.sin(agent.facing), 0.0])
        best_id = None
        best_dist = 1.5
        for obj in self.physics.objects:
            if obj.is_static:
                continue
            # 检查是否被其他 agent 持有
            held_by_other = any(
                a.held_object == obj.id for a in self.agents.values()
            )
            if held_by_other:
                continue
            to_obj = obj.position - agent.pos
            dist = np.linalg.norm(to_obj)
            if dist < best_dist and dist > 0.01:
                cos_angle = np.dot(to_obj[:2], facing_dir[:2]) / (
                    np.linalg.norm(to_obj[:2]) * np.linalg.norm(facing_dir[:2]) + 1e-8)
                if cos_angle > 0.5:
                    best_dist = dist
                    best_id = obj.id
        if best_id is not None:
            agent.held_object = best_id

    def _try_drop(self, agent: AgentState):
        """放下物体"""
        agent.held_object = None

    def _try_throw(self, agent: AgentState, speed: float):
        """扔出物体"""
        if agent.held_object is None:
            return
        obj = self.physics._get_object(agent.held_object)
        if obj:
            facing_dir = np.array([
                np.cos(agent.facing), np.sin(agent.facing), 0.3
            ])
            facing_dir = facing_dir / (np.linalg.norm(facing_dir) + 1e-8)
            obj.velocity = facing_dir * speed + agent.vel * 0.5
        agent.held_object = None

    def _try_push(self, agent: AgentState):
        """推前方物体"""
        facing_dir = np.array([np.cos(agent.facing), np.sin(agent.facing), 0.0])
        for obj in self.physics.objects:
            if obj.is_static:
                continue
            to_obj = obj.position - agent.pos
            dist = np.linalg.norm(to_obj)
            if dist < 2.0 and dist > 0.01:
                cos_angle = np.dot(to_obj[:2], facing_dir[:2]) / (
                    np.linalg.norm(to_obj[:2]) * np.linalg.norm(facing_dir[:2]) + 1e-8)
                if cos_angle > 0.5:
                    push_force = 8.0
                    effective_mass = obj.get_effective_mass()
                    impulse = facing_dir * push_force / effective_mass
                    obj.velocity += impulse * 0.1
                    break

    def _resolve_all_collisions(self):
        """碰撞检测：物体-物体 + agent-物体 + agent-agent"""
        # 物体-物体（复用 physics 的逻辑）
        for i in range(len(self.physics.objects)):
            for j in range(i + 1, len(self.physics.objects)):
                a = self.physics.objects[i]
                b = self.physics.objects[j]
                if not a.is_static or not b.is_static:
                    self.physics._resolve_pair(a, b)

        # Agent-物体
        for agent in self.agents.values():
            agent_sphere = PhysicsObject3D(
                id=-1, position=agent.pos.copy(), velocity=agent.vel.copy(),
                mass=1.0, radius=agent.radius, restitution=0.3, material='rubber',
            )
            for obj in self.physics.objects:
                if not obj.is_static:
                    self.physics._resolve_pair(agent_sphere, obj)
            agent.vel = agent_sphere.velocity.copy()

        # Agent-agent
        agent_list = list(self.agents.values())
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a = agent_list[i]
                b = agent_list[j]
                diff = b.pos - a.pos
                dist = np.linalg.norm(diff)
                min_dist = a.radius + b.radius
                if dist < min_dist and dist > 0.001:
                    normal = diff / dist
                    overlap = min_dist - dist
                    a.pos -= normal * overlap * 0.5
                    b.pos += normal * overlap * 0.5
                    # 简单弹性碰撞
                    rel_vel = a.vel - b.vel
                    vn = np.dot(rel_vel, normal)
                    if vn > 0:
                        a.vel -= normal * vn * 0.5
                        b.vel += normal * vn * 0.5

    def _ground_constraints(self):
        """地面约束（Agent + 物体）"""
        for agent in self.agents.values():
            if agent.pos[2] < agent.radius:
                agent.pos[2] = agent.radius
                if agent.vel[2] < 0:
                    agent.vel[2] = 0
            else:
                agent.vel[2] += self.physics.gravity * self.physics.dt

        # 物体地面约束
        held_ids = {a.held_object for a in self.agents.values()
                    if a.held_object is not None}
        for obj in self.physics.objects:
            if obj.is_static or obj.id in held_ids:
                continue
            ground_z = obj.size[2] if obj.size is not None else obj.radius
            if obj.position[2] < ground_z:
                obj.position[2] = ground_z
                if obj.velocity[2] < 0:
                    obj.velocity[2] = -obj.velocity[2] * obj.restitution
                    if abs(obj.velocity[2]) < 0.3:
                        obj.velocity[2] = 0

    def _boundary_constraints(self):
        """边界约束"""
        r = 0.3
        for agent in self.agents.values():
            for axis in range(3):
                lo = r if axis < 2 else r
                hi = self.bounds[axis] - r if axis < 2 else self.bounds[axis]
                if agent.pos[axis] < lo:
                    agent.pos[axis] = lo
                    agent.vel[axis] = abs(agent.vel[axis]) * 0.3
                elif agent.pos[axis] > hi:
                    agent.pos[axis] = hi
                    agent.vel[axis] = -abs(agent.vel[axis]) * 0.3

    def _apply_friction(self):
        """地面摩擦"""
        for agent in self.agents.values():
            if agent.pos[2] <= agent.radius + 0.05:
                agent.vel[0] *= 0.85
                agent.vel[1] *= 0.85

    def get_agent_observation(self, agent_id: int) -> Observation3D:
        """获取指定 agent 的观察（从其视角渲染）"""
        agent = self.agents[agent_id]

        # 临时设置 physics 的 agent 状态用于渲染
        orig_pos = self.physics.agent_pos.copy()
        orig_facing = self.physics.agent_facing
        orig_held = self.physics.held_object

        self.physics.agent_pos = agent.pos.copy()
        self.physics.agent_facing = agent.facing
        self.physics.held_object = agent.held_object

        # 从 agent 视角渲染
        visual, depth = self.physics._render()
        audio = list(self.physics._audio_events)

        # 本体感知（使用 agent 自己的状态）
        held_mass = 0.0
        held_flag = 0.0
        if agent.held_object is not None:
            obj = self.physics._get_object(agent.held_object)
            if obj:
                held_mass = obj.get_effective_mass() / 10.0
                held_flag = 1.0

        speed = np.linalg.norm(agent.vel)
        proprio = np.array([
            agent.vel[0] / 5.0,
            agent.vel[1] / 5.0,
            agent.vel[2] / 5.0,
            np.cos(agent.facing),
            np.sin(agent.facing),
            speed / 5.0,
            held_mass,
            held_flag,
            agent.grip_strength,
            agent.throw_ready,
        ])

        # 恢复
        self.physics.agent_pos = orig_pos
        self.physics.agent_facing = orig_facing
        self.physics.held_object = orig_held

        return Observation3D(
            visual=visual,
            depth=depth,
            audio_events=audio,
            proprioception=proprio,
        )

    def get_nearby_agents(self, agent_id: int) -> List[int]:
        """获取指定 agent 附近的其他 agent"""
        if agent_id not in self.agents:
            return []
        agent = self.agents[agent_id]
        nearby = []
        for other_id, other in self.agents.items():
            if other_id == agent_id:
                continue
            dist = np.linalg.norm(agent.pos - other.pos)
            if dist < self.comm_range:
                nearby.append(other_id)
        return nearby

    def get_nearby_pairs(self) -> List[Tuple[int, int]]:
        """获取所有在通信范围内的 agent 对"""
        pairs = []
        agent_list = list(self.agents.keys())
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a = self.agents[agent_list[i]]
                b = self.agents[agent_list[j]]
                dist = np.linalg.norm(a.pos - b.pos)
                if dist < self.comm_range:
                    pairs.append((agent_list[i], agent_list[j]))
        return pairs

    def get_all_agent_ids(self) -> List[int]:
        """获取所有 agent id"""
        return list(self.agents.keys())

    def get_agent_pos(self, agent_id: int) -> np.ndarray:
        """获取 agent 位置"""
        return self.agents[agent_id].pos.copy()

    def get_objects_near_agent(self, agent_id: int, radius: float = 5.0) -> List[Dict]:
        """获取 agent 附近的物体特征"""
        agent = self.agents[agent_id]
        return self.physics.get_nearby_objects(agent.pos, radius)
