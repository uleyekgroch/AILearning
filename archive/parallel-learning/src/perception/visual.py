"""
视觉处理系统 — 从环境到特征图

将 Agent 周围的环境状态渲染为 (4, 8, 8) 视觉特征图：
  Channel 0: 距离场（物体越近值越大）
  Channel 1: 方向场（基于 agent 朝向的角度编码）
  Channel 2: 物体类型（不同物体不同编码）
  Channel 3: 动态信息（物体速度/活动度）

纯 torch 实现，零 numpy。
"""

import torch
import math
from typing import List, Dict, Optional, Tuple

from src.core.config import LearnerConfig
from src.core.device import get_device


class VisualSystem:
    """
    视觉渲染器：将环境信息转为视觉特征图

    输入：agent 位置、朝向、环境中物体列表
    输出：(4, 8, 8) 视觉特征图
    """

    def __init__(self, config: LearnerConfig):
        self.config = config
        self.device = get_device(config.device)
        self.channels = config.visual_channels  # 4
        self.size = config.visual_size           # (8, 8)
        self.view_radius = 4.0  # 视野半径

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device)

    def render(
        self,
        agent_pos: Tuple[float, float],
        agent_facing: float,
        objects: List[Dict],
    ) -> torch.Tensor:
        """渲染视觉输入

        Args:
            agent_pos: (x, y) agent 位置
            agent_facing: 朝向角度（弧度），0 = 右，pi/2 = 上
            objects: 物体列表，每个物体为 dict:
                {'pos': (x, y), 'type': int, 'velocity': (vx, vy)}

        Returns:
            (4, 8, 8) 视觉特征图
        """
        ax, ay = agent_pos
        features = torch.zeros(self.channels, self.size[0], self.size[1],
                               device=self.device)

        # 创建坐标网格：以 agent 为中心
        rows = torch.arange(self.size[0], device=self.device, dtype=torch.float32)
        cols = torch.arange(self.size[1], device=self.device, dtype=torch.float32)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')

        # 网格坐标转世界坐标偏移（grid 中心 = agent 位置）
        center_r = (self.size[0] - 1) / 2.0
        center_c = (self.size[1] - 1) / 2.0
        dx = grid_c - center_c  # 世界 x 偏移
        dy = -(grid_r - center_r)  # 世界 y 偏移（行号反转 = y 向上）

        # Channel 0: 距离衰减场（中心最强，边缘衰减）
        dist_from_center = torch.sqrt(dx.pow(2) + dy.pow(2))
        features[0] = torch.exp(-0.5 * (dist_from_center / self.view_radius).pow(2))

        # Channel 1: 方向场（基于 agent 朝向的角度编码）
        angles = torch.atan2(dy, dx)
        # 将朝向方向编码为 1，其他方向按角度差衰减
        angle_diff = angles - agent_facing
        # 归一化到 [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        features[1] = torch.cos(angle_diff) * 0.5 + 0.5  # [0, 1]

        if not objects:
            return features

        # Channel 2 & 3: 从物体信息填充
        for obj in objects:
            obj_x, obj_y = obj.get('pos', (0.0, 0.0))
            obj_type = obj.get('type', 0)
            obj_vel = obj.get('velocity', (0.0, 0.0))

            # 物体相对 agent 的偏移
            rel_x = obj_x - ax
            rel_y = obj_y - ay

            # 计算物体在各网格单元的"影响力"
            cell_dx = dx - rel_x
            cell_dy = dy - rel_y
            cell_dist = torch.sqrt(cell_dx.pow(2) + cell_dy.pow(2) + 1e-8)

            # 物体影响力：距离越近越强，高斯衰减
            influence = torch.exp(-cell_dist.pow(2) / 2.0)

            # Channel 2: 物体类型编码（不同类型叠加）
            type_val = (obj_type + 1) / 10.0  # 归一化到 [0.1, ...]
            features[2] += influence * type_val

            # Channel 3: 物体运动强度
            speed = math.sqrt(obj_vel[0] ** 2 + obj_vel[1] ** 2)
            features[3] += influence * min(speed / 2.0, 1.0)

        # 归一化各通道到 [0, 1]
        for c in range(self.channels):
            c_max = features[c].max()
            if c_max > 1.0:
                features[c] = features[c] / c_max

        return features

    def render_batch(
        self,
        agent_positions: torch.Tensor,
        agent_facings: torch.Tensor,
        object_lists: List[List[Dict]],
    ) -> torch.Tensor:
        """批量渲染视觉输入

        Args:
            agent_positions: (B, 2) agent 位置
            agent_facings: (B,) 朝向角度
            object_lists: 长度 B 的物体列表

        Returns:
            (B, 4, 8, 8) 批量视觉特征图
        """
        batch_size = agent_positions.shape[0]
        batch = torch.zeros(batch_size, self.channels, self.size[0], self.size[1],
                            device=self.device)

        for i in range(batch_size):
            pos = (agent_positions[i, 0].item(), agent_positions[i, 1].item())
            facing = agent_facings[i].item()
            objects = object_lists[i] if i < len(object_lists) else []
            batch[i] = self.render(pos, facing, objects)

        return batch
