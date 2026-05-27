"""
Phase 27: 多 Agent 共享网格世界

N 个 Agent 在同一环境中各自探索，
每个 Agent 有自己的位置和视野范围。
"""

import random
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

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

    def __init__(self, width: int = 12, height: int = 12, num_agents: int = 2):
        self.width = width
        self.height = height
        self.objects: List[Object] = []
        self.step_count = 0

        # 初始化 Agent 位置（均匀分布在四个角落区域）
        positions = [
            (1, 1), (width - 2, height - 2),
            (1, height - 2), (width - 2, 1),
        ]
        self.agents: List[AgentState] = []
        for i in range(num_agents):
            px, py = positions[i % len(positions)]
            self.agents.append(AgentState(
                agent_id=i, x=px, y=py,
                fov_range=3,
            ))

    def add_object(self, obj: Object):
        """添加物体"""
        obj.enrich_features()
        self.objects.append(obj)

    def get_agent_observation(self, agent_id: int) -> dict:
        """获取指定 Agent 的局部观测"""
        agent = self.agents[agent_id]
        visible = []
        for obj in self.objects:
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

        self.step_count += 1
        return self.get_agent_observation(agent_id)

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
    """创建测试世界"""
    env = MultiAgentGridWorld(12, 12, num_agents)

    # 散布各种材质的物体
    objects_data = [
        (0, 3, 3, 'red', 'circle', 1.5, 'metal'),
        (1, 8, 8, 'blue', 'square', 0.5, 'fabric'),
        (2, 5, 2, 'green', 'triangle', 1.0, 'wood'),
        (3, 10, 5, 'yellow', 'circle', 1.8, 'stone'),
        (4, 2, 9, 'red', 'square', 0.8, 'plastic'),
        (5, 7, 6, 'blue', 'circle', 1.2, 'glass'),
        (6, 1, 5, 'green', 'square', 0.6, 'fabric'),
        (7, 9, 10, 'yellow', 'triangle', 1.4, 'metal'),
        (8, 4, 7, 'red', 'triangle', 0.9, 'wood'),
        (9, 6, 1, 'blue', 'square', 1.1, 'stone'),
    ]
    for oid, x, y, color, shape, weight, material in objects_data:
        env.add_object(Object(oid, x, y, color, shape, weight, material=material))

    return env
