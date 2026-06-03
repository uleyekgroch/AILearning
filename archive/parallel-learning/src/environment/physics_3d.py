"""
统一学习系统 — 3D 物理引擎

碰撞检测/解析、流体区域、软体弹簧-质点系统。
全部使用 torch.Tensor，零 numpy。

从 mvl 的 environment_3d.py / physics_fluid.py / physics_soft.py 移植。
"""

import torch
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .objects_3d import (
    PhysicsObject3D, AudioEvent3D,
    MATERIALS, MATERIAL_COLORS, MATERIAL_FREQUENCY, MATERIAL_RESTITUTION,
)


# ============================================================
# 碰撞检测
# ============================================================

class CollisionDetector3D:
    """
    3D 碰撞检测器

    支持 sphere-sphere 和 sphere-plane 碰撞。
    返回包含 normal / penetration / contact_point 的字典（torch 张量）。
    """

    @staticmethod
    def detect_sphere_sphere(
        a: PhysicsObject3D,
        b: PhysicsObject3D,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        球体 vs 球体碰撞检测

        返回 None 表示无碰撞；否则返回:
          normal:       (3,) 碰撞法线 a→b
          penetration:  ()   标量穿透深度
          contact_point:(3,) 接触点
        """
        delta = b.position - a.position
        dist = delta.norm()
        min_dist = a.get_bounding_radius() + b.get_bounding_radius()

        if dist >= min_dist or dist < 1e-6:
            return None

        normal = delta / dist
        penetration = min_dist - dist
        # 接触点：两球心连线上 a 表面处
        contact_point = a.position + normal * a.get_bounding_radius()

        return {
            'normal': normal,
            'penetration': torch.tensor(float(penetration), dtype=torch.float32),
            'contact_point': contact_point,
        }

# ============================================================
# 碰撞解析
# ============================================================

class CollisionResolver3D:
    """
    3D 碰撞解析器

    基于冲量的弹性碰撞，支持静态/动态物体。
    冲量公式：j = -(1+e) * v_rel_n / (1/m_a + 1/m_b)
    """

    @staticmethod
    def resolve_sphere_sphere(
        a: PhysicsObject3D,
        b: PhysicsObject3D,
        collision_info: Dict[str, torch.Tensor],
    ) -> None:
        """
        解析球体-球体碰撞，就地更新速度和位置。
        """
        normal = collision_info['normal']
        penetration = collision_info['penetration'].item()

        m_a = a.get_effective_mass()
        m_b = b.get_effective_mass()
        total_mass = m_a + m_b

        # 分离：按质量反比推开
        if a.is_static:
            b.position = b.position + normal * penetration
        elif b.is_static:
            a.position = a.position - normal * penetration
        else:
            a.position = a.position - normal * penetration * (m_b / total_mass)
            b.position = b.position + normal * penetration * (m_a / total_mass)

        # 相对速度
        rel_vel = a.velocity - b.velocity
        vel_along_normal = float(torch.dot(rel_vel, normal))

        if vel_along_normal > 0:
            return  # 正在远离

        # 弹性系数取较小值
        e = min(a.restitution, b.restitution)

        # 冲量大小
        if a.is_static:
            j = -(1 + e) * vel_along_normal / (1.0 / m_b)
        elif b.is_static:
            j = -(1 + e) * vel_along_normal / (1.0 / m_a)
        else:
            j = -(1 + e) * vel_along_normal / (1.0 / m_a + 1.0 / m_b)

        # 应用冲量
        if not a.is_static:
            a.velocity = a.velocity + (j / m_a) * normal
        if not b.is_static:
            b.velocity = b.velocity - (j / m_b) * normal


class SpatialHash:
    """空间哈希网格 — 碰撞检测加速 O(n²) → O(n)"""

    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        self._cells: Dict[Tuple[int, int, int], List[int]] = {}

    def clear(self) -> None:
        self._cells.clear()

    def _key(self, pos: torch.Tensor) -> Tuple[int, int, int]:
        return (
            int(math.floor(pos[0].item() / self.cell_size)),
            int(math.floor(pos[1].item() / self.cell_size)),
            int(math.floor(pos[2].item() / self.cell_size)),
        )

    def insert(self, idx: int, pos: torch.Tensor, radius: float = 0.0) -> None:
        """Insert object idx, covering all cells its bounding sphere touches."""
        min_key = self._key(pos - radius)
        max_key = self._key(pos + radius)
        for x in range(min_key[0], max_key[0] + 1):
            for y in range(min_key[1], max_key[1] + 1):
                for z in range(min_key[2], max_key[2] + 1):
                    key = (x, y, z)
                    if key not in self._cells:
                        self._cells[key] = []
                    self._cells[key].append(idx)

    def query_pairs(self, n: int) -> Set[Tuple[int, int]]:
        """Return all potentially colliding pairs (i, j) where i < j."""
        pairs: Set[Tuple[int, int]] = set()
        for cell_indices in self._cells.values():
            if len(cell_indices) < 2:
                continue
            for a in range(len(cell_indices)):
                for b in range(a + 1, len(cell_indices)):
                    i, j = cell_indices[a], cell_indices[b]
                    pair = (min(i, j), max(i, j))
                    pairs.add(pair)
        return pairs


# ============================================================
# 流体区域
# ============================================================

@dataclass
class FluidRegion:
    """
    轴对齐长方体流体区域

    提供浮力、阻力等流体力学查询。
    """
    min_corner: torch.Tensor   # (3,) 区域最小角
    max_corner: torch.Tensor   # (3,) 区域最大角
    name: str = 'water'
    density: float = 1000.0
    viscosity: float = 0.001
    drag_coefficient: float = 0.47

    def contains_point(self, pos: torch.Tensor) -> bool:
        """检查点是否在流体区域内"""
        return bool(
            (pos[0] >= self.min_corner[0]) and (pos[0] <= self.max_corner[0]) and
            (pos[1] >= self.min_corner[1]) and (pos[1] <= self.max_corner[1]) and
            (pos[2] >= self.min_corner[2]) and (pos[2] <= self.max_corner[2])
        )

    def get_depth(self, pos: torch.Tensor) -> float:
        """获取点在流体中的深度（从液面算起）"""
        if not self.contains_point(pos):
            return 0.0
        return float(self.max_corner[2] - pos[2])

    def get_submerged_volume(self, pos: torch.Tensor, radius: float) -> float:
        """
        计算球体在流体中的浸没体积

        完全浸没：4/3 * pi * r^3
        部分浸没：球冠体积公式 V = pi * h^2 * (3r - h) / 3
        """
        if not self.contains_point(pos):
            return 0.0

        full_vol = (4.0 / 3.0) * math.pi * radius ** 3

        # 完全浸没
        if (pos[2] - radius >= self.min_corner[2] and
                pos[2] + radius <= self.max_corner[2]):
            return full_vol

        # 部分浸没 — 上半露出
        if pos[2] + radius > self.max_corner[2]:
            h = float(self.max_corner[2] - (pos[2] - radius))
            h = max(0.0, min(2.0 * radius, h))
            return math.pi * h ** 2 * (3.0 * radius - h) / 3.0

        # 部分浸没 — 下半超出
        if pos[2] - radius < self.min_corner[2]:
            h = float((pos[2] + radius) - self.min_corner[2])
            h = max(0.0, min(2.0 * radius, h))
            return math.pi * h ** 2 * (3.0 * radius - h) / 3.0

        return full_vol


# 预定义流体区域模板（需要指定 min_corner / max_corner 实例化）
WATER = FluidRegion(
    min_corner=torch.zeros(3),
    max_corner=torch.zeros(3),
    name='water', density=1000.0, viscosity=0.001, drag_coefficient=0.47,
)

AIR = FluidRegion(
    min_corner=torch.zeros(3),
    max_corner=torch.zeros(3),
    name='air', density=1.225, viscosity=1.8e-5, drag_coefficient=0.47,
)

OIL = FluidRegion(
    min_corner=torch.zeros(3),
    max_corner=torch.zeros(3),
    name='oil', density=900.0, viscosity=0.1, drag_coefficient=0.5,
)

HONEY = FluidRegion(
    min_corner=torch.zeros(3),
    max_corner=torch.zeros(3),
    name='honey', density=1400.0, viscosity=10.0, drag_coefficient=0.6,
)


# ============================================================
# 软体弹簧-质点系统
# ============================================================

@dataclass
class MassPoint:
    """质点"""
    pos: torch.Tensor     # (3,)
    vel: torch.Tensor     # (3,)
    mass: float = 1.0
    is_fixed: bool = False


@dataclass
class Spring:
    """弹簧"""
    idx_a: int              # 质点 A 索引
    idx_b: int              # 质点 B 索引
    rest_length: float
    stiffness: float = 100.0
    damping: float = 1.0


class SoftBody:
    """
    软体弹簧-质点系统

    8 角节点 + 22 条弹簧（12 边 + 6 面对角线 + 4 体对角线）。
    使用批量化张量运算加速弹簧力计算。
    """

    def __init__(
        self,
        center: torch.Tensor,
        size: torch.Tensor,
        mass: float = 1.0,
        stiffness: float = 50.0,
        damping: float = 0.5,
        recovery_rate: float = 0.01,
    ):
        self.recovery_rate = recovery_rate

        # 8 个角节点
        cx, cy, cz = center.tolist()
        sx, sy, sz = size.tolist()
        corners = [
            (cx - sx, cy - sy, cz - sz),
            (cx + sx, cy - sy, cz - sz),
            (cx - sx, cy + sy, cz - sz),
            (cx + sx, cy + sy, cz - sz),
            (cx - sx, cy - sy, cz + sz),
            (cx + sx, cy - sy, cz + sz),
            (cx - sx, cy + sy, cz + sz),
            (cx + sx, cy + sy, cz + sz),
        ]
        node_mass = mass / 8.0
        self.points: List[MassPoint] = [
            MassPoint(
                pos=torch.tensor(c, dtype=torch.float32),
                vel=torch.zeros(3, dtype=torch.float32),
                mass=node_mass,
            )
            for c in corners
        ]

        # 弹簧：边(12) + 面对角线(6) + 体对角线(4) = 22
        self.springs: List[Spring] = []

        # 边弹簧
        edges = [
            (0, 1), (2, 3), (4, 5), (6, 7),  # x 方向
            (0, 2), (1, 3), (4, 6), (5, 7),  # y 方向
            (0, 4), (1, 5), (2, 6), (3, 7),  # z 方向
        ]
        for i, j in edges:
            rest = float((self.points[i].pos - self.points[j].pos).norm())
            self.springs.append(Spring(i, j, rest, stiffness, damping))

        # 面对角线弹簧
        face_diags = [
            (0, 3), (1, 2), (4, 7), (5, 6),
            (0, 5), (1, 4), (2, 7), (3, 6),
        ]
        seen: set = set()
        for i, j in face_diags:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                rest = float((self.points[i].pos - self.points[j].pos).norm())
                self.springs.append(Spring(i, j, rest, stiffness * 0.5, damping))
                if len(seen) >= 6:
                    break

        # 体对角线弹簧
        body_diags = [(0, 7), (1, 6), (2, 5), (3, 4)]
        for i, j in body_diags:
            rest = float((self.points[i].pos - self.points[j].pos).norm())
            self.springs.append(Spring(i, j, rest, stiffness * 0.3, damping))

        # 保存静止位置（用于恢复力）
        self.rest_positions = [p.pos.clone() for p in self.points]

    def update(self, dt: float, gravity: float) -> None:
        """
        更新软体：重力 → 弹簧力（批量张量） → 恢复力 → 积分 → 约束
        """
        n = len(self.points)
        device = self.points[0].pos.device

        # ── 堆叠位置/速度为张量 ──
        positions = torch.stack([p.pos for p in self.points])   # (n, 3)
        velocities = torch.stack([p.vel for p in self.points])  # (n, 3)
        masses = torch.tensor([p.mass for p in self.points],
                              dtype=torch.float32, device=device)  # (n,)

        # ── 重力 ──
        for i, p in enumerate(self.points):
            if not p.is_fixed:
                velocities[i, 2] -= abs(gravity) * dt

        # ── 弹簧力（向量化） ──
        num_springs = len(self.springs)
        if num_springs > 0:
            idx_a = torch.tensor([s.idx_a for s in self.springs], dtype=torch.long)
            idx_b = torch.tensor([s.idx_b for s in self.springs], dtype=torch.long)
            rest_lengths = torch.tensor([s.rest_length for s in self.springs],
                                        dtype=torch.float32)
            stiffnesses = torch.tensor([s.stiffness for s in self.springs],
                                       dtype=torch.float32)
            dampings = torch.tensor([s.damping for s in self.springs],
                                    dtype=torch.float32)

            # 弹簧端点位置/速度
            pos_a = positions[idx_a]  # (num_springs, 3)
            pos_b = positions[idx_b]
            vel_a = velocities[idx_a]
            vel_b = velocities[idx_b]

            delta = pos_b - pos_a                         # (num_springs, 3)
            dist = delta.norm(dim=1, keepdim=True).clamp(min=1e-4)  # (num_springs, 1)
            direction = delta / dist                      # (num_springs, 3)

            # 弹簧力
            displacement = dist.squeeze(1) - rest_lengths  # (num_springs,)
            force_mag = stiffnesses * displacement         # (num_springs,)

            # 阻尼力
            rel_vel = vel_b - vel_a
            damp_proj = (rel_vel * direction).sum(dim=1)   # (num_springs,)
            damp_force = dampings * damp_proj              # (num_springs,)

            total_scalar = force_mag + damp_force          # (num_springs,)
            total_force = total_scalar.unsqueeze(1) * direction  # (num_springs, 3)

            # 累加力到质点
            # 用 scatter_add 批量累加
            forces = torch.zeros(n, 3, dtype=torch.float32, device=device)
            forces.scatter_add_(0, idx_a.unsqueeze(1).expand(-1, 3), total_force)
            forces.scatter_add_(0, idx_b.unsqueeze(1).expand(-1, 3), -total_force)

            # 加速度 → 速度增量
            acc = forces / masses.unsqueeze(1).clamp(min=1e-6)
            velocities += acc * dt

        # ── 恢复力 ──
        rest_pos = torch.stack(self.rest_positions)  # (n, 3)
        recovery = (rest_pos - positions) * self.recovery_rate
        velocities += recovery

        # ── 写回 + 积分 + 约束 ──
        for i, p in enumerate(self.points):
            if p.is_fixed:
                continue
            p.vel = velocities[i] * 0.99  # 阻尼
            p.pos = positions[i] + p.vel * dt

            # 地面碰撞
            if p.pos[2] < 0:
                p.pos[2] = 0.0
                p.vel[2] = -p.vel[2] * 0.5

    def get_state(self) -> Dict[str, torch.Tensor]:
        """
        获取软体状态

        返回:
          positions: (n, 3)
          velocities: (n, 3)
          deformation: () 标量形变程度
        """
        positions = torch.stack([p.pos for p in self.points])
        velocities = torch.stack([p.vel for p in self.points])

        # 形变 = 当前位置与静止位置的平均距离
        rest = torch.stack(self.rest_positions)
        deformation = (positions - rest).norm(dim=1).mean()

        return {
            'positions': positions,
            'velocities': velocities,
            'deformation': deformation,
        }

    def collide_with_sphere(
        self,
        center: torch.Tensor,
        radius: float,
        velocity: Optional[torch.Tensor] = None,
        mass: float = 2.0,
    ) -> None:
        """软体节点与球形物体碰撞"""
        if velocity is None:
            velocity = torch.zeros(3, dtype=torch.float32)

        for p in self.points:
            diff = p.pos - center
            dist = float(diff.norm())
            min_dist = radius + 0.05

            if dist < min_dist and dist > 1e-6:
                n = diff / dist
                # 推出
                p.pos = center + n * min_dist
                # 反射速度
                v_rel = p.vel - velocity
                v_n = float(torch.dot(v_rel, n))
                if v_n < 0:
                    p.vel = p.vel - 1.2 * v_n * n

    def get_center(self) -> torch.Tensor:
        """软体质心"""
        return torch.stack([p.pos for p in self.points]).mean(dim=0)

    def get_deformation(self) -> float:
        """形变程度"""
        rest = torch.stack(self.rest_positions)
        current = torch.stack([p.pos for p in self.points])
        return float((current - rest).norm(dim=1).mean().item())
