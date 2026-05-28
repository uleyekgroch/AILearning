"""
Phase 27/30: 多 Agent 共享网格世界

N 个 Agent 在同一环境中各自探索，
每个 Agent 有自己的位置和视野范围。

Phase 30 改进：
- 自适应网格大小（随 Agent 数量缩放）
- 均匀随机出生点（替换 4 角落）
- 空间索引加速观测
"""

import random
import numpy as np
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import Object


@dataclass
class AgentState:
    """单个 Agent 的状态"""
    agent_id: int
    x: int
    y: int
    fov_range: int = 3
    discovered_objects: Set[int] = field(default_factory=set)


class MultiAgentGridWorld:
    """多 Agent 共享网格世界"""

    def __init__(self, width: int = None, height: int = None, num_agents: int = 2):
        # 自适应网格大小：至少 base_size，随 Agent 数缩放
        if width is None:
            width = max(12, int(np.sqrt(num_agents) * 3))
        if height is None:
            height = width
        self.width = width
        self.height = height
        self.objects: List[Object] = []
        self.step_count = 0
        self.num_agents = num_agents

        # 空间索引：网格 cell 大小 = fov_range
        self._cell_size = 3
        self._spatial_index: Dict[tuple, List[int]] = defaultdict(list)

        # 均匀随机出生点（不重叠）
        occupied = set()
        self.agents: List[AgentState] = []
        for i in range(num_agents):
            for _ in range(100):  # 最多尝试 100 次
                px = random.randint(1, self.width - 2)
                py = random.randint(1, self.height - 2)
                if (px, py) not in occupied:
                    occupied.add((px, py))
                    break
            self.agents.append(AgentState(
                agent_id=i, x=px, y=py,
                fov_range=3,
            ))

    def _update_spatial_index(self):
        """更新空间索引"""
        self._spatial_index.clear()
        for idx, obj in enumerate(self.objects):
            cell = (obj.x // self._cell_size, obj.y // self._cell_size)
            self._spatial_index[cell].append(idx)

    def add_object(self, obj: Object):
        """添加物体"""
        obj.enrich_features()
        self.objects.append(obj)
        # 更新空间索引
        cell = (obj.x // self._cell_size, obj.y // self._cell_size)
        self._spatial_index[cell].append(len(self.objects) - 1)

    def get_agent_observation(self, agent_id: int) -> dict:
        """获取指定 Agent 的局部观测（使用空间索引加速）"""
        agent = self.agents[agent_id]
        visible = []

        # 只搜索邻近的网格 cell
        cx = agent.x // self._cell_size
        cy = agent.y // self._cell_size
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cell = (cx + dx, cy + dy)
                for obj_idx in self._spatial_index.get(cell, []):
                    obj = self.objects[obj_idx]
                    dist = abs(obj.x - agent.x) + abs(obj.y - agent.y)
                    if dist <= agent.fov_range:
                        visible.append({
                            'object': obj,
                            'relative_x': obj.x - agent.x,
                            'relative_y': obj.y - agent.y,
                            'distance': dist,
                        })
                        agent.discovered_objects.add(obj.id)

        return {
            'agent_id': agent_id,
            'agent_position': (agent.x, agent.y),
            'visible_objects': visible,
            'step': self.step_count,
        }

    def step_agent(self, agent_id: int, action: int) -> dict:
        """执行单个 Agent 的动作"""
        agent = self.agents[agent_id]
        dx, dy = 0, 0
        if action == 0: dy = -1      # 上
        elif action == 1: dy = 1     # 下
        elif action == 2: dx = -1    # 左
        elif action == 3: dx = 1     # 右

        new_x = agent.x + dx
        new_y = agent.y + dy

        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            agent.x = new_x
            agent.y = new_y

        return self.get_agent_observation(agent_id)

    def batch_step(self, actions: List[int]) -> List[dict]:
        """批量执行所有 Agent 的动作，返回所有观测"""
        observations = []
        for agent_id, action in enumerate(actions):
            obs = self.step_agent(agent_id, action)
            observations.append(obs)
        self.step_count += 1
        return observations

    def get_all_discoveries(self) -> Dict[int, Set[int]]:
        """获取每个 Agent 发现的物体 ID"""
        return {a.agent_id: a.discovered_objects.copy() for a in self.agents}

    def get_total_discoveries(self) -> Set[int]:
        """获取所有 Agent 发现的物体并集"""
        all_ids = set()
        for a in self.agents:
            all_ids.update(a.discovered_objects)
        return all_ids

    def render(self) -> str:
        """渲染环境"""
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]

        for obj in self.objects:
            sym = obj.color[0].upper() + obj.shape[0]
            if 0 <= obj.x < self.width and 0 <= obj.y < self.height:
                grid[obj.y][obj.x] = sym

        for agent in self.agents:
            if 0 <= agent.x < self.width and 0 <= agent.y < self.height:
                grid[agent.y][agent.x] = f'A{agent.agent_id}'

        lines = []
        for row in grid:
            lines.append(' '.join(row))
        return '\n'.join(lines)


def create_multi_agent_world(num_agents: int = 2) -> MultiAgentGridWorld:
    """创建测试世界（自适应网格大小）"""
    env = MultiAgentGridWorld(num_agents=num_agents)

    # 物体数量随网格面积缩放
    num_objects = max(10, env.width * env.height // 10)
    colors = ['red', 'blue', 'green', 'yellow']
    shapes = ['circle', 'square', 'triangle']
    materials = ['metal', 'fabric', 'wood', 'stone', 'plastic', 'glass']

    for oid in range(num_objects):
        x = random.randint(0, env.width - 1)
        y = random.randint(0, env.height - 1)
        color = colors[oid % len(colors)]
        shape = shapes[oid % len(shapes)]
        material = materials[oid % len(materials)]
        weight = 0.5 + (oid % 5) * 0.3
        env.add_object(Object(oid, x, y, color, shape, weight, material=material))

    return env
