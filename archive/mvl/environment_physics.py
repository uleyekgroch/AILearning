"""
真实物理环境

集成刚体、流体、软体物理效果。
这是从简单物理到真实物理的完整升级。

类比：从一个简单的游戏世界 → 一个充满各种物理效果的真实世界。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from physics_rigid import RigidBody, RigidBodyEngine, MaterialType, MATERIALS
from physics_fluid import FluidEngine, FluidRegion, create_water_world
from physics_soft import Cloth, Rope, SoftBody
from perception_visual import VisualField
from perception_auditory import AuditorySystem


class PhysicsObjectAdapter:
    """适配器：将刚体适配为 agent_3d 期望的物体接口"""

    # 材料到颜色的映射
    MATERIAL_COLOR = {
        'wood': 'brown',
        'metal': 'gray',
        'rubber': 'black',
        'ice': 'white',
        'stone': 'gray',
    }

    def __init__(self, body: RigidBody, obj_id: int):
        self.id = obj_id
        self.x = body.x
        self.y = body.y
        self.z = body.z
        self.size = body.radius
        self.weight = body.mass
        self.color = self.MATERIAL_COLOR.get(body.material.name, 'gray')
        self._body = body

    def to_observation(self) -> np.ndarray:
        """转换为12维观测向量：位置(3) + 颜色(4) + 形状(3) + 大小(1) + 重量(1)"""
        pos = np.array([self.x, self.y, self.z])
        color_map = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'brown': 0, 'gray': 1, 'black': 2, 'white': 3}
        color_vec = np.zeros(4)
        if self.color in color_map:
            color_vec[color_map[self.color]] = 1.0
        # 球体
        shape_vec = np.array([0, 1, 0])
        return np.concatenate([pos, color_vec, shape_vec, np.array([self.size, self.weight])])


class PhysicsEnvironment:
    """
    真实物理环境

    集成：
    1. 刚体动力学
    2. 流体力学
    3. 软体物理
    """

    def __init__(self, width: float = 10.0, height: float = 10.0, depth: float = 10.0):
        self.width = width
        self.height = height
        self.depth = depth

        # 物理引擎
        self.rigid_engine = RigidBodyEngine(gravity=9.81)
        self.fluid_engine = FluidEngine()

        # 软体物体
        self.cloths: List[Cloth] = []
        self.ropes: List[Rope] = []
        self.soft_bodies: List[SoftBody] = []

        # Agent位置
        self.agent_x = width / 2
        self.agent_y = height / 2
        self.agent_z = 1.0

        # 步数
        self.step_count = 0

        # 任务系统
        self.current_task = None

        # 感知系统
        self.visual_field = VisualField(grid_size=8, fov_degrees=90.0, max_range=8.0)
        self.auditory_system = AuditorySystem(max_events=8)

    def add_rigid_body(self, body: RigidBody):
        """添加刚体"""
        self.rigid_engine.add_body(body)

    def add_fluid_region(self, region: FluidRegion):
        """添加流体区域"""
        self.fluid_engine.add_region(region)

    def add_cloth(self, cloth: Cloth):
        """添加布料"""
        self.cloths.append(cloth)

    def add_rope(self, rope: Rope):
        """添加绳子"""
        self.ropes.append(rope)

    def add_soft_body(self, soft_body: SoftBody):
        """添加软体"""
        self.soft_bodies.append(soft_body)

    def get_observation(self) -> Dict:
        """
        获取观测

        包含兼容 agent_3d 的 visible_objects / touch_objects 格式，
        以及物理引擎原始状态。
        """
        agent_pos = np.array([self.agent_x, self.agent_y, self.agent_z])

        # 构建 visible_objects 和 touch_objects（兼容 agent_3d）
        visible_objects = []
        touch_objects = []
        vision_range = 8.0
        touch_range = 1.5

        for i, body in enumerate(self.rigid_engine.bodies):
            dx = body.x - self.agent_x
            dy = body.y - self.agent_y
            dz = body.z - self.agent_z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)

            adapter = PhysicsObjectAdapter(body, i)

            if distance < vision_range:
                visible_objects.append({
                    'object': adapter,
                    'distance': distance,
                    'direction': np.array([dx, dy, dz]) / max(distance, 0.001)
                })

            if distance < touch_range:
                touch_objects.append({
                    'object': adapter,
                    'distance': distance,
                    'is_touching': distance < (body.radius + 0.5)
                })

        # 物理引擎原始状态
        rigid_states = self.rigid_engine.get_state()

        fluid_regions = []
        for region in self.fluid_engine.regions:
            fluid_regions.append({
                'fluid_type': region.fluid.name,
                'bounds': (region.x_min, region.y_min, region.z_min,
                          region.x_max, region.y_max, region.z_max)
            })

        soft_states = []
        for cloth in self.cloths:
            soft_states.extend(cloth.get_state())
        for rope in self.ropes:
            soft_states.extend(rope.get_state())
        for soft_body in self.soft_bodies:
            soft_states.extend(soft_body.get_state())

        # 视觉场渲染
        agent_pos_arr = np.array([self.agent_x, self.agent_y, self.agent_z])
        visual = self.visual_field.render(
            agent_pos_arr, self.rigid_engine.bodies
        )

        # 听觉事件检测
        audio_events = self.auditory_system.detect_events(
            agent_pos_arr, self.rigid_engine.bodies, fluid_regions
        )
        audio_encoded = self.auditory_system.encode_events(audio_events)

        return {
            'agent_position': agent_pos,
            'visible_objects': visible_objects,
            'touch_objects': touch_objects,
            'rigid_bodies': rigid_states,
            'fluid_regions': fluid_regions,
            'soft_bodies': soft_states,
            'step_count': self.step_count,
            'visual_field': visual,          # (8, 8, 4) 视觉张量
            'audio_events': audio_encoded,   # (56,) 听觉编码向量
        }

    def step(self, action: int, dt: float = 0.01) -> Tuple[Dict, float, bool]:
        """
        执行动作

        动作：
        0: 前进 (y+)
        1: 后退 (y-)
        2: 左移 (x-)
        3: 右移 (x+)
        4: 跳跃 (z+)
        5: 下蹲 (z-)
        6: 推动
        7: 拉动
        """
        # 保存前一步观测（用于任务奖励计算）
        prev_obs = self.get_observation()

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

        # 更新物理模拟
        self._update_physics(dt)

        # 更新步数
        self.step_count += 1

        # 获取当前观测
        curr_obs = self.get_observation()

        # 计算奖励和终止条件
        if self.current_task is not None:
            reward, done = self.current_task.step(self, prev_obs, curr_obs)
        else:
            reward = self._compute_reward()
            done = self.step_count >= 1000

        return curr_obs, reward, done

    def _push_objects(self):
        """推动附近的物体（从 agent 指向物体的方向）"""
        push_range = 2.5
        push_force = 15.0

        for body in self.rigid_engine.bodies:
            # 计算距离
            dx = body.x - self.agent_x
            dy = body.y - self.agent_y
            dz = body.z - self.agent_z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)

            # 如果在推动范围内
            if distance < push_range and distance > 0.01:
                # 推动方向：从 agent 指向物体
                push_direction = np.array([dx, dy, dz]) / distance

                # 应用推力
                body.apply_force(
                    push_direction[0] * push_force,
                    push_direction[1] * push_force,
                    push_direction[2] * push_force
                )

    def _pull_objects(self):
        """拉动附近的物体（从物体指向 agent 的方向）"""
        pull_range = 2.5
        pull_force = 15.0

        for body in self.rigid_engine.bodies:
            # 计算距离
            dx = body.x - self.agent_x
            dy = body.y - self.agent_y
            dz = body.z - self.agent_z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)

            # 如果在拉动范围内
            if distance < pull_range and distance > 0.01:
                # 拉动方向：从物体指向 agent
                pull_direction = np.array([-dx, -dy, -dz]) / distance

                # 应用拉力
                body.apply_force(
                    pull_direction[0] * pull_force,
                    pull_direction[1] * pull_force,
                    pull_direction[2] * pull_force
                )

    def _update_physics(self, dt: float):
        """更新所有物理模拟"""
        # 更新刚体
        self.rigid_engine.update(dt)

        # 应用流体力到刚体
        for body in self.rigid_engine.bodies:
            self.fluid_engine.apply_fluid_forces(body)

        # 更新软体
        for cloth in self.cloths:
            cloth.update(dt)
        for rope in self.ropes:
            rope.update(dt)
        for soft_body in self.soft_bodies:
            soft_body.update(dt)

    def _compute_reward(self) -> float:
        """计算奖励"""
        # 简单的奖励：探索新区域
        return 0.0

    def reset(self):
        """重置环境"""
        self.agent_x = self.width / 2
        self.agent_y = self.height / 2
        self.agent_z = 1.0
        self.step_count = 0

        # 重置刚体
        for body in self.rigid_engine.bodies:
            body.x = np.random.uniform(1, self.width - 1)
            body.y = np.random.uniform(1, self.height - 1)
            body.z = np.random.uniform(3, self.depth - 1)
            body.vx = 0
            body.vy = 0
            body.vz = 0

        # 重置任务
        if self.current_task is not None:
            self.current_task.reset()
            self.current_task.setup(self)

    def render(self):
        """渲染环境（简单文本输出）"""
        print(f"\n物理环境 ({self.width}x{self.height}x{self.depth})")
        print(f"Agent位置: ({self.agent_x:.1f}, {self.agent_y:.1f}, {self.agent_z:.1f})")
        print(f"刚体数量: {len(self.rigid_engine.bodies)}")
        print(f"流体区域: {len(self.fluid_engine.regions)}")
        print(f"布料数量: {len(self.cloths)}")
        print(f"绳子数量: {len(self.ropes)}")
        print(f"软体数量: {len(self.soft_bodies)}")


def create_simple_physics_world() -> PhysicsEnvironment:
    """创建一个简单的物理世界"""
    env = PhysicsEnvironment(10, 10, 10)

    # 添加一些刚体
    ball1 = RigidBody(2, 2, 5, radius=0.5, mass=1.0, material=MaterialType.RUBBER)
    ball2 = RigidBody(5, 5, 3, radius=0.5, mass=2.0, material=MaterialType.METAL)
    ball3 = RigidBody(8, 3, 4, radius=0.7, mass=1.5, material=MaterialType.WOOD)

    env.add_rigid_body(ball1)
    env.add_rigid_body(ball2)
    env.add_rigid_body(ball3)

    # 添加一个水池
    water_region = FluidRegion('water',
                               x_min=0, y_min=0, z_min=0,
                               x_max=10, y_max=10, z_max=3)
    env.add_fluid_region(water_region)

    return env


def create_rich_physics_world() -> PhysicsEnvironment:
    """创建一个丰富的物理世界"""
    env = PhysicsEnvironment(12, 12, 12)

    # 添加各种刚体
    env.add_rigid_body(RigidBody(2, 2, 5, radius=0.5, mass=1.0, material=MaterialType.RUBBER))
    env.add_rigid_body(RigidBody(5, 5, 3, radius=0.5, mass=2.0, material=MaterialType.METAL))
    env.add_rigid_body(RigidBody(8, 3, 4, radius=0.7, mass=1.5, material=MaterialType.WOOD))
    env.add_rigid_body(RigidBody(3, 8, 6, radius=0.6, mass=1.2, material=MaterialType.ICE))
    env.add_rigid_body(RigidBody(7, 7, 2, radius=0.8, mass=3.0, material=MaterialType.STONE))

    # 添加流体区域
    env.add_fluid_region(FluidRegion('water',
                                     x_min=0, y_min=0, z_min=0,
                                     x_max=12, y_max=12, z_max=4))
    env.add_fluid_region(FluidRegion('oil',
                                     x_min=2, y_min=2, z_min=4,
                                     x_max=10, y_max=10, z_max=6))

    # 添加布料
    cloth = Cloth(4.0, 4.0, 8, 8, mass=1.0, stiffness=100.0)
    env.add_cloth(cloth)

    # 添加绳子
    rope = Rope(3.0, 15, mass=0.5, stiffness=200.0)
    env.add_rope(rope)

    # 添加软体
    soft_body = SoftBody(6.0, 6.0, 8.0, radius=1.0, num_points=15,
                        mass=2.0, stiffness=150.0)
    env.add_soft_body(soft_body)

    return env
