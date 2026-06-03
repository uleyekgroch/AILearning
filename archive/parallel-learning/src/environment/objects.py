"""
统一学习系统 — 物理物体

2D 物理世界中的物体表示。
所有数值属性使用 torch.Tensor，零 numpy。
"""

import torch
from typing import Dict, Optional


class PhysicsObject:
    """物理物体"""

    SHAPES = ('circle', 'square', 'triangle')
    MATERIALS = ('metal', 'wood', 'plastic', 'glass', 'rubber')
    COLORS = ('red', 'blue', 'green', 'yellow', 'orange', 'purple', 'white', 'black')
    SIZES = ('tiny', 'small', 'medium', 'big', 'huge')

    # 材质 → 质量系数
    MATERIAL_DENSITY = {
        'metal': 3.0, 'wood': 0.8, 'plastic': 0.5,
        'glass': 1.2, 'rubber': 0.6,
    }

    # 尺寸 → 半径映射
    SIZE_RADIUS = {
        'tiny': 0.2, 'small': 0.4, 'medium': 0.6, 'big': 0.8, 'huge': 1.0,
    }

    def __init__(
        self,
        obj_id: int,
        position: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        shape: str = 'circle',
        material: str = 'wood',
        mass: Optional[float] = None,
        color: str = 'red',
        size: str = 'medium',
    ):
        self.obj_id = obj_id
        self.position = position.float()
        self.velocity = (velocity.float() if velocity is not None
                         else torch.zeros_like(position))
        self.shape = shape
        self.material = material
        self.color = color
        self.size = size
        self.radius = self.SIZE_RADIUS.get(size, 0.6)
        self.mass = (mass if mass is not None
                     else self.MATERIAL_DENSITY.get(material, 1.0) * self.radius ** 2)

    def get_properties(self) -> Dict[str, str]:
        """返回可被语言指称的属性集合"""
        return {
            'color': self.color,
            'shape': self.shape,
            'size': self.size,
            'material': self.material,
        }

    def to_observation(self) -> torch.Tensor:
        """转为观测向量：[x, y, vx, vy, shape_oh(3), material_oh(5), color_oh(8), size_oh(5)]"""
        parts = [self.position, self.velocity]

        # 独热编码
        shape_vec = torch.zeros(len(self.SHAPES))
        shape_vec[self.SHAPES.index(self.shape)] = 1.0
        parts.append(shape_vec)

        mat_vec = torch.zeros(len(self.MATERIALS))
        mat_vec[self.MATERIALS.index(self.material)] = 1.0
        parts.append(mat_vec)

        color_vec = torch.zeros(len(self.COLORS))
        color_vec[self.COLORS.index(self.color)] = 1.0
        parts.append(color_vec)

        size_vec = torch.zeros(len(self.SIZES))
        size_vec[self.SIZES.index(self.size)] = 1.0
        parts.append(size_vec)

        return torch.cat(parts)
