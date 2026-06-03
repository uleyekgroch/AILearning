"""
统一学习系统 — 2D 世界环境

实现 IEnvironment 接口，提供多模态观测、好奇心奖励、阶段配置。
"""

import torch
import random as _random
from typing import Dict, List, Tuple

from src.core.interfaces import IEnvironment
from src.core.config import LearnerConfig
from .physics import PhysicsEngine
from .objects import PhysicsObject


class World(IEnvironment):
    """统一环境接口"""

    # 阶段 → 场景复杂度
    STAGE_CONFIG = {
        'sensorimotor': {
            'num_objects': 3,
            'shapes': ('circle', 'square'),
            'colors': ('red', 'blue', 'green'),
            'materials': ('wood',),
            'sizes': ('small', 'medium'),
        },
        'early_preoperational': {
            'num_objects': 5,
            'shapes': ('circle', 'square', 'triangle'),
            'colors': ('red', 'blue', 'green', 'yellow'),
            'materials': ('wood', 'plastic'),
            'sizes': ('small', 'medium', 'big'),
        },
        'late_preoperational': {
            'num_objects': 7,
            'shapes': ('circle', 'square', 'triangle'),
            'colors': ('red', 'blue', 'green', 'yellow', 'orange'),
            'materials': ('wood', 'plastic', 'metal'),
            'sizes': ('tiny', 'small', 'medium', 'big'),
        },
        'early_concrete': {
            'num_objects': 8,
            'shapes': ('circle', 'square', 'triangle'),
            'colors': ('red', 'blue', 'green', 'yellow', 'orange', 'purple'),
            'materials': ('wood', 'plastic', 'metal', 'glass'),
            'sizes': ('tiny', 'small', 'medium', 'big', 'huge'),
        },
        'late_concrete': {
            'num_objects': 10,
            'shapes': ('circle', 'square', 'triangle'),
            'colors': PhysicsObject.COLORS,
            'materials': PhysicsObject.MATERIALS,
            'sizes': PhysicsObject.SIZES,
        },
        'early_formal': {
            'num_objects': 10,
            'shapes': PhysicsObject.SHAPES,
            'colors': PhysicsObject.COLORS,
            'materials': PhysicsObject.MATERIALS,
            'sizes': PhysicsObject.SIZES,
        },
        'late_formal': {
            'num_objects': 12,
            'shapes': PhysicsObject.SHAPES,
            'colors': PhysicsObject.COLORS,
            'materials': PhysicsObject.MATERIALS,
            'sizes': PhysicsObject.SIZES,
        },
        'adolescent': {
            'num_objects': 15,
            'shapes': PhysicsObject.SHAPES,
            'colors': PhysicsObject.COLORS,
            'materials': PhysicsObject.MATERIALS,
            'sizes': PhysicsObject.SIZES,
        },
    }

    def __init__(self, config: LearnerConfig):
        self.config = config
        self.physics = PhysicsEngine(bounds=(10.0, 10.0))
        self.agent_pos = torch.zeros(2)
        self.agent_vel = torch.zeros(2)
        self.visited_cells: set = set()
        self._stage = 'sensorimotor'
        self._step_count = 0

    # ── IEnvironment 接口 ──────────────────────────────────────────────

    def observe(self) -> Dict[str, torch.Tensor]:
        """返回多模态观测"""
        return self.physics.get_observations(self.agent_pos)

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool]:
        """
        执行动作，返回 (obs, reward, done)

        action: (action_dim,) 连续向量，前 2 维为移动方向
        reward: 好奇心奖励（探索新区域）
        """
        self._step_count += 1

        # 解析移动
        move = action[:2].float()
        self.agent_vel = move
        self.agent_pos = self.agent_pos + move * 0.5

        # 边界限制
        self.agent_pos = self.agent_pos.clamp(0, self.physics.bounds[0])

        # 记录访问区域（离散化为 1x1 格子）
        cell = (int(self.agent_pos[0].item()), int(self.agent_pos[1].item()))
        is_new = cell not in self.visited_cells
        self.visited_cells.add(cell)

        # 物理步进
        self.physics.step(dt=0.1)

        # 好奇心奖励
        reward = 1.0 if is_new else 0.0

        # done 条件：步数过多或探索完所有区域
        max_cells = int(self.physics.bounds[0] * self.physics.bounds[1])
        done = self._step_count >= 500 or len(self.visited_cells) >= max_cells

        obs = self.observe()
        return obs, reward, done

    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境，清空状态并随机放置物体"""
        self.agent_pos = torch.zeros(2)
        self.agent_vel = torch.zeros(2)
        self.visited_cells.clear()
        self._step_count = 0
        self.physics.objects.clear()
        return self.observe()

    def configure_for_stage(self, stage: str) -> None:
        """按发展阶段配置环境"""
        self._stage = stage
        self.reset()
        cfg = self.STAGE_CONFIG.get(stage, self.STAGE_CONFIG['sensorimotor'])

        for i in range(cfg['num_objects']):
            pos = torch.tensor([
                _random.uniform(1.0, self.physics.bounds[0] - 1.0),
                _random.uniform(1.0, self.physics.bounds[1] - 1.0),
            ])
            vel = torch.randn(2) * 0.3
            obj = PhysicsObject(
                obj_id=i,
                position=pos,
                velocity=vel,
                shape=_random.choice(cfg['shapes']),
                color=_random.choice(cfg['colors']),
                material=_random.choice(cfg['materials']),
                size=_random.choice(cfg['sizes']),
            )
            self.physics.add_object(obj)

    # ── 场景特征生成（供语言游戏使用） ──────────────────────────────────

    def generate_scene_features(self) -> List[Dict[str, str]]:
        """生成用于语言游戏的场景特征字典"""
        return [obj.get_properties() for obj in self.physics.objects]
