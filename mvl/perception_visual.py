"""
视觉感知模块

从 agent 视角渲染视觉场（VisualField）。
使用射线投射（Ray Casting）模拟视觉：
- 从 agent 位置向视野范围发射射线
- 检测射线与物体的交叉
- 渲染为 NxN 像素网格（深度 + RGB + 占据）

类比：婴儿的眼睛接收到光线，大脑需要从这些原始光影中学会"看到"物体。
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


# 材料到 RGB 颜色的映射
MATERIAL_RGB = {
    'wood':   np.array([0.6, 0.3, 0.1]),   # 棕色
    'metal':  np.array([0.5, 0.5, 0.6]),   # 灰蓝
    'rubber': np.array([0.1, 0.1, 0.1]),   # 深灰/黑
    'ice':    np.array([0.8, 0.9, 1.0]),   # 淡蓝白
    'stone':  np.array([0.4, 0.4, 0.4]),   # 灰色
}

BACKGROUND_COLOR = np.array([0.05, 0.05, 0.1])  # 深蓝黑背景
MAX_DEPTH = 10.0  # 最大可视距离（归一化用）


class VisualField:
    """
    视觉场：从 agent 视角渲染环境

    使用射线投射将 3D 环境渲染为 2D 像素网格。
    每个像素包含 4 个通道：depth, R, G, B
    """

    def __init__(self, grid_size: int = 8, fov_degrees: float = 90.0,
                 max_range: float = 8.0):
        """
        Args:
            grid_size: 像素网格大小（grid_size x grid_size）
            fov_degrees: 视野角度
            max_range: 最大检测距离
        """
        self.grid_size = grid_size
        self.fov = np.radians(fov_degrees)
        self.max_range = max_range

        # 预计算射线方向（在 agent 坐标系中）
        self._precompute_rays()

    def _precompute_rays(self):
        """预计算射线方向向量（agent 朝向 +y）"""
        half_fov = self.fov / 2.0
        # 均匀分布在视野内的角度
        yaw_angles = np.linspace(-half_fov, half_fov, self.grid_size)   # 左右
        pitch_angles = np.linspace(-half_fov, half_fov, self.grid_size) # 上下

        self.ray_directions = []
        for pitch in pitch_angles:
            for yaw in yaw_angles:
                # 射线方向：以 +y 为前方
                # yaw 绕 z 轴旋转，pitch 绕 x 轴旋转
                dx = np.sin(yaw)
                dy = np.cos(yaw) * np.cos(pitch)
                dz = np.sin(pitch)
                direction = np.array([dx, dy, dz])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                self.ray_directions.append(direction)

        self.ray_directions = np.array(self.ray_directions)  # (grid_size*grid_size, 3)

    def render(self, agent_pos: np.ndarray,
               rigid_bodies: list,
               agent_facing: Optional[np.ndarray] = None) -> np.ndarray:
        """
        渲染视觉场

        Args:
            agent_pos: agent 位置 (3,)
            rigid_bodies: 刚体列表，每个有 x, y, z, radius, material (name)
            agent_facing: agent 朝向 (3,)，默认 +y

        Returns:
            visual_field: (grid_size, grid_size, 4) 数组
                          通道：[depth, R, G, B]
                          depth 归一化到 0-1（0=近，1=远/无物体）
                          RGB 0-1
        """
        if agent_facing is None:
            agent_facing = np.array([0.0, 1.0, 0.0])

        # 构建旋转矩阵（将射线从 agent 坐标系转换到世界坐标系）
        rot_matrix = self._facing_to_rotation(agent_facing)

        # 渲染每个像素
        pixels = np.zeros((self.grid_size * self.grid_size, 4))

        for i, local_dir in enumerate(self.ray_directions):
            # 转换到世界坐标系
            world_dir = rot_matrix @ local_dir

            # 射线检测
            hit_dist, hit_color = self._ray_cast(
                agent_pos, world_dir, rigid_bodies
            )

            if hit_dist is not None:
                # 命中物体
                depth_normalized = np.clip(hit_dist / self.max_range, 0.0, 1.0)
                pixels[i] = np.array([
                    depth_normalized,
                    hit_color[0],
                    hit_color[1],
                    hit_color[2]
                ])
            else:
                # 未命中 - 背景
                pixels[i] = np.array([1.0,
                                      BACKGROUND_COLOR[0],
                                      BACKGROUND_COLOR[1],
                                      BACKGROUND_COLOR[2]])

        return pixels.reshape(self.grid_size, self.grid_size, 4)

    def _ray_cast(self, origin: np.ndarray, direction: np.ndarray,
                  rigid_bodies: list) -> Tuple[Optional[float], np.ndarray]:
        """
        射线投射：检测射线与球体的交叉

        球体-射线交叉的经典算法。

        Returns:
            (distance, color) 或 (None, None) 如果未命中
        """
        closest_dist = None
        closest_color = None

        for body in rigid_bodies:
            # 从 body 获取位置和半径
            center = np.array([body.x, body.y, body.z])
            radius = body.radius

            # 射线-球体交叉检测
            dist = self._ray_sphere_intersect(origin, direction, center, radius)

            if dist is not None and dist > 0.01:  # 忽略太近的交叉
                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist
                    # 获取颜色
                    material_name = body.material.name if hasattr(body.material, 'name') else str(body.material)
                    closest_color = MATERIAL_RGB.get(material_name, np.array([0.5, 0.5, 0.5]))

        return closest_dist, closest_color

    def _ray_sphere_intersect(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                              sphere_center: np.ndarray, sphere_radius: float) -> Optional[float]:
        """
        射线-球体交叉检测

        数学推导：
        |ray_origin + t * ray_dir - sphere_center|^2 = sphere_radius^2
        令 oc = ray_origin - sphere_center
        a = dot(ray_dir, ray_dir) = 1 (归一化后)
        b = 2 * dot(oc, ray_dir)
        c = dot(oc, oc) - radius^2
        discriminant = b^2 - 4ac

        Returns:
            最近交叉点的 t 值，或 None
        """
        oc = ray_origin - sphere_center
        a = np.dot(ray_dir, ray_dir)  # 应该是 1
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - sphere_radius ** 2
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # 返回最小的正 t
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None

    def _facing_to_rotation(self, facing: np.ndarray) -> np.ndarray:
        """
        将朝向向量转换为旋转矩阵

        简化版：假设 agent 朝向 facing 方向，up 向量为 +z
        """
        forward = facing.copy()
        norm = np.linalg.norm(forward)
        if norm > 0:
            forward /= norm
        else:
            forward = np.array([0.0, 1.0, 0.0])

        up = np.array([0.0, 0.0, 1.0])

        # 右向量
        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            # forward 与 up 平行，选择另一个 up
            up = np.array([1.0, 0.0, 0.0])
            right = np.cross(forward, up)
            norm = np.linalg.norm(right)
        right /= norm

        # 修正 up
        up = np.cross(right, forward)

        # 旋转矩阵：列向量是 right, forward, up
        rot = np.column_stack([right, forward, up])
        return rot

    def flatten(self, visual_field: np.ndarray) -> np.ndarray:
        """将视觉场展平为一维向量"""
        return visual_field.flatten()
