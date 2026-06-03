"""
听觉处理系统 — 声音事件检测

将环境中的声音源信息编码为 (7,) 音频特征向量：
  [0-1]: 最近声源方向（sin/cos 编码）
  [2-3]: 最近声源距离 + 强度衰减
  [4-5]: 环境整体声压级 + 声音多样性
  [6]:   流体/碰撞等特殊声音标记

纯 torch 实现，零 numpy。
"""

import torch
import math
from typing import List, Dict, Tuple

from src.core.config import LearnerConfig
from src.core.device import get_device


class AuditorySystem:
    """
    听觉处理器：从环境声音源到音频特征

    输入：agent 位置、物体列表、流体列表
    输出：(7,) 音频事件向量
    """

    def __init__(self, config: LearnerConfig):
        self.config = config
        self.device = get_device(config.device)
        self.audio_dim = config.audio_dim  # 13
        self.hearing_radius = 6.0  # 听觉半径

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device)

    def detect_events(
        self,
        agent_pos: Tuple[float, float],
        bodies: List[Dict],
        fluids: List[Dict],
    ) -> torch.Tensor:
        """检测环境中的声音事件

        Args:
            agent_pos: (x, y) agent 位置
            bodies: 物体列表，每个物体为 dict:
                {'pos': (x, y), 'velocity': (vx, vy), 'type': int,
                 'mass': float, 'is_moving': bool}
            fluids: 流体列表，每个流体为 dict:
                {'pos': (x, y), 'velocity': (vx, vy), 'density': float}

        Returns:
            (7,) 音频事件向量
        """
        ax, ay = agent_pos
        features = torch.zeros(self.audio_dim, device=self.device)

        if not bodies and not fluids:
            return features

        # 收集所有声源
        sound_sources = []

        for body in bodies:
            bx, by = body.get('pos', (0.0, 0.0))
            vx, vy = body.get('velocity', (0.0, 0.0))
            is_moving = body.get('is_moving', False)
            mass = body.get('mass', 1.0)
            obj_type = body.get('type', 0)

            # 声音强度 = 速度 * 质量
            speed = math.sqrt(vx ** 2 + vy ** 2)
            intensity = speed * mass

            if is_moving or speed > 0.1:
                sound_sources.append({
                    'pos': (bx, by),
                    'intensity': intensity,
                    'direction': math.atan2(by - ay, bx - ax),
                    'is_fluid': False,
                    'type': obj_type,
                })

        for fluid in fluids:
            fx, fy = fluid.get('pos', (0.0, 0.0))
            vx, vy = fluid.get('velocity', (0.0, 0.0))
            density = fluid.get('density', 0.5)

            speed = math.sqrt(vx ** 2 + vy ** 2)
            intensity = speed * density * 2.0  # 流体声音更显著

            if speed > 0.01:
                sound_sources.append({
                    'pos': (fx, fy),
                    'intensity': intensity,
                    'direction': math.atan2(fy - ay, fx - ax),
                    'is_fluid': True,
                    'type': -1,
                })

        if not sound_sources:
            return features

        # 计算距离衰减
        for src in sound_sources:
            sx, sy = src['pos']
            dist = math.sqrt((sx - ax) ** 2 + (sy - ay) ** 2)
            src['distance'] = dist
            src['attenuated'] = src['intensity'] / (1.0 + dist)

        # 按衰减后强度排序，取最强声源
        sound_sources.sort(key=lambda s: s['attenuated'], reverse=True)
        nearest = sound_sources[0]

        # [0-1]: 最近声源方向（sin/cos 编码）
        features[0] = math.sin(nearest['direction'])
        features[1] = math.cos(nearest['direction'])

        # [2-3]: 距离 + 衰减后强度
        features[2] = min(nearest['distance'] / self.hearing_radius, 1.0)
        features[3] = min(nearest['attenuated'], 1.0)

        # [4-5]: 整体声压级 + 声音多样性
        total_intensity = sum(s['attenuated'] for s in sound_sources)
        features[4] = min(total_intensity / 5.0, 1.0)
        features[5] = min(len(sound_sources) / 5.0, 1.0)

        # [6]: 特殊声音标记（流体声 / 碰撞声）
        has_fluid = any(s['is_fluid'] for s in sound_sources[:3])
        has_collision = any(not s['is_fluid'] and s['attenuated'] > 0.5
                          for s in sound_sources[:3])
        if has_fluid:
            features[6] = 0.5
        if has_collision:
            features[6] += 0.5
        features[6] = min(features[6], 1.0)

        return features

    def detect_events_batch(
        self,
        agent_positions: torch.Tensor,
        body_lists: List[List[Dict]],
        fluid_lists: List[List[Dict]],
    ) -> torch.Tensor:
        """批量检测声音事件

        Args:
            agent_positions: (B, 2) agent 位置
            body_lists: 长度 B 的物体列表
            fluid_lists: 长度 B 的流体列表

        Returns:
            (B, 7) 批量音频特征
        """
        batch_size = agent_positions.shape[0]
        batch = torch.zeros(batch_size, self.audio_dim, device=self.device)

        for i in range(batch_size):
            pos = (agent_positions[i, 0].item(), agent_positions[i, 1].item())
            bodies = body_lists[i] if i < len(body_lists) else []
            fluids = fluid_lists[i] if i < len(fluid_lists) else []
            batch[i] = self.detect_events(pos, bodies, fluids)

        return batch
