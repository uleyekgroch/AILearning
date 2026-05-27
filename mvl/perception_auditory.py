"""
听觉感知模块

检测环境中的声音事件。
声音由物理交互产生：
- 刚体碰撞：物体相撞时产生声音
- 流体流动：在流体区域内产生持续的环境声
- 物体移动：高速移动的物体产生声音

类比：婴儿听到声音，最初只是"响"，逐渐学会区分"碰撞声"和"水流声"。
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class SoundEvent:
    """单个声音事件"""
    sound_type: int       # 0=碰撞, 1=流体, 2=移动
    amplitude: float      # 强度 (0-1)
    direction: np.ndarray  # 从 agent 到声源的方向 (3D)
    distance: float        # 到声源的距离

    def to_vector(self) -> np.ndarray:
        """转换为4维向量：[type_onehot(3), amplitude(1)] + direction(3)"""
        type_onehot = np.zeros(3)
        type_onehot[self.sound_type] = 1.0
        return np.concatenate([type_onehot, [self.amplitude], self.direction])


class AuditorySystem:
    """
    听觉系统：检测和编码声音事件

    声音事件触发条件：
    1. 碰撞：两刚体距离 < sum_of_radii → 碰撞声
    2. 流体：刚体在流体区域内 → 流体声
    3. 移动：刚体速度 > 阈值 → 移动声
    """

    def __init__(self, max_events: int = 8, collision_threshold: float = 0.5,
                 velocity_threshold: float = 0.3):
        """
        Args:
            max_events: 最大同时声音事件数
            collision_threshold: 碰撞检测的距离余量
            velocity_threshold: 物体移动产生声音的速度阈值
        """
        self.max_events = max_events
        self.collision_threshold = collision_threshold
        self.velocity_threshold = velocity_threshold

        # 上一帧的物体状态（用于检测碰撞）
        self._prev_body_states = []

    def detect_events(self, agent_pos: np.ndarray,
                      rigid_bodies: list,
                      fluid_regions: list,
                      prev_rigid_bodies: Optional[list] = None) -> List[SoundEvent]:
        """
        检测当前帧的声音事件

        Args:
            agent_pos: agent 位置 (3,)
            rigid_bodies: 当前刚体列表
            fluid_regions: 流体区域列表
            prev_rigid_bodies: 上一帧刚体列表（用于碰撞检测）

        Returns:
            声音事件列表
        """
        events = []

        # 1. 碰撞检测
        collision_events = self._detect_collisions(agent_pos, rigid_bodies)
        events.extend(collision_events)

        # 2. 流体声检测
        fluid_events = self._detect_fluid_sounds(agent_pos, rigid_bodies, fluid_regions)
        events.extend(fluid_events)

        # 3. 移动声检测
        movement_events = self._detect_movement_sounds(agent_pos, rigid_bodies)
        events.extend(movement_events)

        # 限制事件数量（保留最强的）
        events.sort(key=lambda e: e.amplitude, reverse=True)
        return events[:self.max_events]

    def _detect_collisions(self, agent_pos: np.ndarray,
                           rigid_bodies: list) -> List[SoundEvent]:
        """检测刚体之间的碰撞"""
        events = []

        for i in range(len(rigid_bodies)):
            for j in range(i + 1, len(rigid_bodies)):
                body_a = rigid_bodies[i]
                body_b = rigid_bodies[j]

                # 计算距离
                dx = body_a.x - body_b.x
                dy = body_a.y - body_b.y
                dz = body_a.z - body_b.z
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                # 碰撞检测：距离 < 两半径之和 + 阈值
                collision_dist = body_a.radius + body_b.radius + self.collision_threshold
                if dist < collision_dist:
                    # 碰撞强度：基于相对速度
                    rel_speed = np.sqrt(
                        (body_a.vx - body_b.vx)**2 +
                        (body_a.vy - body_b.vy)**2 +
                        (body_a.vz - body_b.vz)**2
                    )
                    amplitude = np.clip(rel_speed / 10.0, 0.1, 1.0)

                    # 碰撞位置（中点）
                    collision_pos = np.array([
                        (body_a.x + body_b.x) / 2,
                        (body_a.y + body_b.y) / 2,
                        (body_a.z + body_b.z) / 2
                    ])

                    # 方向和距离
                    direction = collision_pos - agent_pos
                    distance = np.linalg.norm(direction)
                    if distance > 0.01:
                        direction /= distance
                    else:
                        direction = np.zeros(3)

                    events.append(SoundEvent(
                        sound_type=0,  # 碰撞
                        amplitude=amplitude,
                        direction=direction,
                        distance=distance
                    ))

        return events

    def _detect_fluid_sounds(self, agent_pos: np.ndarray,
                             rigid_bodies: list,
                             fluid_regions: list) -> List[SoundEvent]:
        """检测流体中的声音"""
        events = []

        for region in fluid_regions:
            bounds = region['bounds']  # (x_min, y_min, z_min, x_max, y_max, z_max)
            x_min, y_min, z_min, x_max, y_max, z_max = bounds

            # 检查是否有刚体在流体中
            for body in rigid_bodies:
                if (x_min <= body.x <= x_max and
                    y_min <= body.y <= y_max and
                    z_min <= body.z <= z_max):

                    # 计算物体在流体中的速度
                    speed = np.sqrt(body.vx**2 + body.vy**2 + body.vz**2)
                    # 流体声：基础强度 + 速度相关
                    amplitude = np.clip(0.2 + speed / 10.0, 0.1, 0.6)

                    # 方向：从 agent 到物体
                    direction = np.array([body.x - agent_pos[0],
                                          body.y - agent_pos[1],
                                          body.z - agent_pos[2]])
                    distance = np.linalg.norm(direction)
                    if distance > 0.01:
                        direction /= distance
                    else:
                        direction = np.zeros(3)

                    events.append(SoundEvent(
                        sound_type=1,  # 流体
                        amplitude=amplitude,
                        direction=direction,
                        distance=distance
                    ))

        return events

    def _detect_movement_sounds(self, agent_pos: np.ndarray,
                                rigid_bodies: list) -> List[SoundEvent]:
        """检测高速移动物体的声音"""
        events = []

        for body in rigid_bodies:
            speed = np.sqrt(body.vx**2 + body.vy**2 + body.vz**2)

            if speed > self.velocity_threshold:
                # 移动声强度：基于速度
                amplitude = np.clip(speed / 8.0, 0.1, 0.8)

                # 方向
                direction = np.array([body.x - agent_pos[0],
                                      body.y - agent_pos[1],
                                      body.z - agent_pos[2]])
                distance = np.linalg.norm(direction)
                if distance > 0.01:
                    direction /= distance
                else:
                    direction = np.zeros(3)

                events.append(SoundEvent(
                    sound_type=2,  # 移动
                    amplitude=amplitude,
                    direction=direction,
                    distance=distance
                ))

        return events

    def encode_events(self, events: List[SoundEvent]) -> np.ndarray:
        """
        编码声音事件为固定大小的向量

        输出：(max_events * 7,) 向量
        每个事件 7 维：type_onehot(3) + amplitude(1) + direction(3)

        如果事件不足 max_events，用零填充。
        """
        encoded = np.zeros(self.max_events * 7)

        for i, event in enumerate(events[:self.max_events]):
            start = i * 7
            encoded[start:start + 7] = event.to_vector()

        return encoded

    def get_encoding_dim(self) -> int:
        """返回编码向量的维度"""
        return self.max_events * 7
