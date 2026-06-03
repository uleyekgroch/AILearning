"""
统一学习系统 — 简化 2D 物理引擎

更新位置、碰撞检测、边界反弹。
零 numpy — 所有运算使用 torch。
"""

import torch
from typing import Dict, List, Tuple

from .objects import PhysicsObject


class PhysicsEngine:
    """简化 2D 物理引擎"""

    def __init__(self, bounds: Tuple[float, float] = (10.0, 10.0)):
        self.bounds = bounds
        self.objects: List[PhysicsObject] = []

    # ── 物体管理 ──────────────────────────────────────────────────────

    def add_object(self, obj: PhysicsObject) -> None:
        self.objects.append(obj)

    def remove_object(self, obj_id: int) -> None:
        self.objects = [o for o in self.objects if o.obj_id != obj_id]

    def get_by_id(self, obj_id: int) -> PhysicsObject | None:
        for o in self.objects:
            if o.obj_id == obj_id:
                return o
        return None

    # ── 物理步进 ──────────────────────────────────────────────────────

    def step(self, dt: float = 0.1) -> None:
        """物理步进：更新位置 → 碰撞响应 → 边界反弹"""
        # 1. 更新位置
        for obj in self.objects:
            obj.position = obj.position + obj.velocity * dt

        # 2. 碰撞检测与弹性响应
        collisions = self.detect_collisions()
        for i, j in collisions:
            self._resolve_collision(self.objects[i], self.objects[j])

        # 3. 边界反弹
        for obj in self.objects:
            self._boundary_bounce(obj)

    # ── 碰撞检测 ──────────────────────────────────────────────────────

    def detect_collisions(self) -> List[Tuple[int, int]]:
        """检测所有物体间的碰撞（圆形近似）"""
        collisions = []
        n = len(self.objects)
        for i in range(n):
            for j in range(i + 1, n):
                dist = (self.objects[i].position - self.objects[j].position).norm()
                min_dist = self.objects[i].radius + self.objects[j].radius
                if dist < min_dist:
                    collisions.append((i, j))
        return collisions

    # ── 观测 ──────────────────────────────────────────────────────────

    def get_observations(self, agent_pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        """返回多模态观测字典"""
        if not self.objects:
            return {
                'visual': torch.zeros(4),
                'audio': torch.zeros(13),
                'position': agent_pos.clone(),
            }

        # visual: 所有物体观测拼接后平均，保证固定维度
        obs_vecs = [obj.to_observation() for obj in self.objects]
        visual = torch.stack(obs_vecs).mean(dim=0)

        # audio: 碰撞数量 + 物体速度统计 + 物体属性统计
        collisions = self.detect_collisions()
        speeds = [obj.velocity.norm().item() for obj in self.objects]
        radii = [obj.radius for obj in self.objects]
        masses = [obj.mass for obj in self.objects]
        audio = torch.tensor([
            float(len(collisions)),
            sum(speeds) / max(len(speeds), 1),
            max(speeds) if speeds else 0.0,
            min(speeds) if speeds else 0.0,
            float(len(self.objects)),
            agent_pos[0].item(),
            agent_pos[1].item(),
            sum(radii) / max(len(radii), 1),   # 平均半径
            max(radii) if radii else 0.0,
            min(radii) if radii else 0.0,
            sum(masses) / max(len(masses), 1),  # 平均质量
            max(masses) if masses else 0.0,
            min(masses) if masses else 0.0,
        ])

        return {
            'visual': visual,
            'audio': audio,
            'position': agent_pos.clone(),
        }

    # ── 内部 ──────────────────────────────────────────────────────────

    def _resolve_collision(self, a: PhysicsObject, b: PhysicsObject) -> None:
        """简单弹性碰撞"""
        delta = a.position - b.position
        dist = delta.norm()
        if dist < 1e-8:
            delta = torch.randn_like(delta) * 0.01
            dist = delta.norm()

        normal = delta / dist
        overlap = a.radius + b.radius - dist

        # 分离
        a.position = a.position + normal * (overlap / 2)
        b.position = b.position - normal * (overlap / 2)

        # 速度交换（简化弹性碰撞）
        rel_vel = a.velocity - b.velocity
        vel_along_normal = torch.dot(rel_vel, normal)

        if vel_along_normal > 0:
            return  # 正在远离

        total_mass = a.mass + b.mass
        impulse = 2 * vel_along_normal / total_mass
        a.velocity = a.velocity - impulse * b.mass * normal
        b.velocity = b.velocity + impulse * a.mass * normal

    def _boundary_bounce(self, obj: PhysicsObject) -> None:
        """边界反弹"""
        for dim in range(obj.position.shape[0]):
            if obj.position[dim] - obj.radius < 0:
                obj.position[dim] = obj.radius
                obj.velocity[dim] = abs(obj.velocity[dim])
            elif obj.position[dim] + obj.radius > self.bounds[dim]:
                obj.position[dim] = self.bounds[dim] - obj.radius
                obj.velocity[dim] = -abs(obj.velocity[dim])
