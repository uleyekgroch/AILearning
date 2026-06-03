"""
统一学习系统 — 3D 物理物体与数据结构

从 mvl 的 PhysicsObject3D / AudioEvent3D / Observation3D / Action3D 移植，
全部使用 torch.Tensor，零 numpy。

7 材质系统：metal, wood, plastic, glass, rubber, stone, fabric。
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================
# 材质系统（7 种）
# ============================================================

MATERIALS: Dict[str, Dict] = {
    'metal': {
        'density': 1.5,
        'restitution': 0.6,
        'friction': 0.3,
        'frequency': 2000.0,
        'color': torch.tensor([0.7, 0.7, 0.8], dtype=torch.float32),
        'hardness': 'hard',
    },
    'wood': {
        'density': 1.0,
        'restitution': 0.3,
        'friction': 0.7,
        'frequency': 300.0,
        'color': torch.tensor([0.6, 0.4, 0.2], dtype=torch.float32),
        'hardness': 'hard',
    },
    'plastic': {
        'density': 0.8,
        'restitution': 0.7,
        'friction': 0.5,
        'frequency': 800.0,
        'color': torch.tensor([0.2, 0.6, 0.9], dtype=torch.float32),
        'hardness': 'soft',
    },
    'glass': {
        'density': 1.2,
        'restitution': 0.1,
        'friction': 0.2,
        'frequency': 3000.0,
        'color': torch.tensor([0.6, 0.8, 1.0], dtype=torch.float32),
        'hardness': 'hard',
    },
    'rubber': {
        'density': 0.6,
        'restitution': 0.9,
        'friction': 0.9,
        'frequency': 200.0,
        'color': torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32),
        'hardness': 'elastic',
    },
    'stone': {
        'density': 2.0,
        'restitution': 0.2,
        'friction': 0.8,
        'frequency': 150.0,
        'color': torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
        'hardness': 'hard',
    },
    'fabric': {
        'density': 0.3,
        'restitution': 0.1,
        'friction': 0.6,
        'frequency': 100.0,
        'color': torch.tensor([0.8, 0.7, 0.5], dtype=torch.float32),
        'hardness': 'soft',
    },
}

MATERIAL_NAMES: List[str] = list(MATERIALS.keys())

MATERIAL_COLORS: Dict[str, torch.Tensor] = {k: v['color'] for k, v in MATERIALS.items()}

MATERIAL_FREQUENCY: Dict[str, float] = {k: v['frequency'] for k, v in MATERIALS.items()}

MATERIAL_RESTITUTION: Dict[str, float] = {k: v['restitution'] for k, v in MATERIALS.items()}

MATERIAL_FRICTION: Dict[str, float] = {k: v['friction'] for k, v in MATERIALS.items()}


# ============================================================
# 物理物体（支持多形状）
# ============================================================

@dataclass
class PhysicsObject3D:
    """
    3D 物理物体（支持 sphere, cube, cylinder）

    所有数值张量使用 float32。
    """
    id: int
    position: torch.Tensor       # (3,) x, y, z
    velocity: torch.Tensor       # (3,) vx, vy, vz
    mass: float = 1.0
    radius: float = 0.3
    restitution: float = 0.5
    material: str = 'wood'
    is_static: bool = False
    on_ground: bool = False
    shape: str = 'sphere'        # 'sphere' | 'cube' | 'cylinder'
    size: torch.Tensor = None    # (3,) 半尺寸 [rx, ry, rz]
    color: torch.Tensor = None   # (3,) RGB
    temperature: float = 20.0

    def __post_init__(self):
        if self.size is None:
            self.size = torch.tensor([self.radius, self.radius, self.radius],
                                     dtype=torch.float32)
        if self.color is None:
            default = MATERIAL_COLORS.get(self.material,
                                          torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32))
            self.color = default.clone()

    def kinetic_energy(self) -> float:
        """动能 = 0.5 * m * v^2"""
        return 0.5 * self.mass * float(torch.dot(self.velocity, self.velocity))

    def get_effective_mass(self) -> float:
        """有效质量 = 基础质量 x 材质密度"""
        density = MATERIALS.get(self.material, {}).get('density', 1.0)
        return self.mass * density

    def get_bounding_radius(self) -> float:
        """包围球半径"""
        if self.shape == 'sphere':
            return float(self.size[0])
        elif self.shape == 'cube':
            return float(self.size.norm())
        else:  # cylinder
            return float(max(self.size[0], self.size[2]))


# ============================================================
# 音频事件
# ============================================================

@dataclass
class AudioEvent3D:
    """
    3D 音频事件

    碰撞、落地、抓取、投掷等产生的声音信号。
    """
    event_type: str          # 'collision' | 'ground_hit' | 'wall_hit' | 'grab' | 'drop' | 'throw'
    amplitude: float         # 音量 0-1
    frequency: float         # 频率（材质决定音色）
    direction: torch.Tensor  # (3,) 相对于 agent 的方向
    source_id: int = -1
    material: str = ''


# ============================================================
# 观测数据
# ============================================================

@dataclass
class Observation3D:
    """
    3D 世界的观察数据

    visual:   (64, 64, 3) RGB 图像
    depth:    (64, 64)     深度图
    audio:    音频事件列表
    proprio:  (10,) 本体感知
    """
    visual: torch.Tensor               # (64, 64, 3)
    depth: torch.Tensor                # (64, 64)
    audio_events: List[AudioEvent3D]   # 音频事件列表
    proprio: torch.Tensor              # (10,)


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

    NAMES = [
        'forward', 'backward', 'left', 'right', 'up', 'down',
        'turn_left', 'turn_right', 'grab', 'drop', 'throw', 'push',
    ]

    COUNT = 12
