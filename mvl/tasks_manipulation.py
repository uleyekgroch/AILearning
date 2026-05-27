"""
物体操控任务

任务层级：
1. PushToGoal（Level 1）：推物体到目标位置 → 理解惯性和摩擦
2. StackObjects（Level 2）：堆叠物体 → 理解重力和稳定性
3. SortByMaterial（Level 3）：按材质分类 → 理解材料属性
"""

import numpy as np
from typing import Dict
from tasks_base import PhysicsTask, TaskCategory, Difficulty


class PushToGoalTask(PhysicsTask):
    """
    推物体到目标位置（Level 1）

    需要理解的物理概念：
    - 物体有惯性（质量影响推动距离）
    - 摩擦力使物体减速
    - 必须在推动范围内才能影响物体

    奖励设计（密集）：
    - 主要：物体到目标的距离减少
    - 奖励：物体进入目标区域
    """

    def __init__(self, goal_pos: np.ndarray, goal_radius: float = 0.8,
                 target_body_idx: int = 0, max_steps: int = 300):
        super().__init__(TaskCategory.MANIPULATION, Difficulty.LEVEL_1, max_steps)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.goal_radius = goal_radius
        self.target_body_idx = target_body_idx
        self.initial_distance = 0.0

    def setup(self, env):
        """记录初始距离"""
        if self.target_body_idx < len(env.rigid_engine.bodies):
            body = env.rigid_engine.bodies[self.target_body_idx]
            self.initial_distance = np.sqrt(
                (body.x - self.goal_pos[0])**2 +
                (body.y - self.goal_pos[1])**2 +
                (body.z - self.goal_pos[2])**2
            )

    def _get_target_distance(self, env) -> float:
        """计算目标物体到目标位置的距离"""
        body = env.rigid_engine.bodies[self.target_body_idx]
        return np.sqrt(
            (body.x - self.goal_pos[0])**2 +
            (body.y - self.goal_pos[1])**2 +
            (body.z - self.goal_pos[2])**2
        )

    def compute_reward(self, env, prev_obs: Dict, curr_obs: Dict) -> float:
        dist = self._get_target_distance(env)

        # 进度奖励：距离减少的比例
        if self.initial_distance > 0.01:
            progress_reward = (self.initial_distance - dist) / self.initial_distance
        else:
            progress_reward = 0.0

        # 接近奖励
        proximity_bonus = 1.0 if dist < self.goal_radius else 0.0

        return 0.01 * progress_reward + proximity_bonus * 0.1

    def check_success(self, env) -> bool:
        return self._get_target_distance(env) < self.goal_radius

    def check_failure(self, env) -> bool:
        return False  # 没有失败条件，只有超时

    def _compute_progress(self, env) -> float:
        dist = self._get_target_distance(env)
        if self.initial_distance > 0.01:
            return max(0.0, 1.0 - dist / self.initial_distance)
        return 0.0


class StackObjectsTask(PhysicsTask):
    """
    堆叠物体（Level 2）

    需要理解的物理概念：
    - 重力把物体往下拉
    - 重的物体放在底部更稳定
    - 物体必须水平对齐才能堆叠
    - 碰撞防止物体占据同一空间

    奖励设计（密集）：
    - 每对成功堆叠的相邻物体获得奖励
    - 稳定性奖励：堆叠保持的时间越长，奖励越高
    """

    def __init__(self, num_objects: int = 3, max_steps: int = 500):
        super().__init__(TaskCategory.MANIPULATION, Difficulty.LEVEL_2, max_steps)
        self.num_objects = num_objects
        self.stable_steps = 0

    def setup(self, env):
        pass  # 物体已在环境中

    def _get_stacked_pairs(self, env) -> int:
        """计算成功堆叠的物体对数"""
        bodies = env.rigid_engine.bodies[:self.num_objects]
        if len(bodies) < 2:
            return 0

        sorted_bodies = sorted(bodies, key=lambda b: b.z)
        stacked_count = 0

        for i in range(1, len(sorted_bodies)):
            dx = sorted_bodies[i].x - sorted_bodies[i-1].x
            dy = sorted_bodies[i].y - sorted_bodies[i-1].y
            horiz_dist = np.sqrt(dx**2 + dy**2)
            threshold = sorted_bodies[i].radius + sorted_bodies[i-1].radius + 0.3

            if horiz_dist < threshold and sorted_bodies[i].z > sorted_bodies[i-1].z:
                stacked_count += 1

        return stacked_count

    def compute_reward(self, env, prev_obs: Dict, curr_obs: Dict) -> float:
        reward = 0.0

        # 堆叠奖励
        stacked = self._get_stacked_pairs(env)
        reward += stacked * 0.05

        # 稳定性奖励
        if self.check_success(env):
            self.stable_steps += 1
            reward += 0.1 * min(self.stable_steps / 10.0, 1.0)
        else:
            self.stable_steps = 0

        return reward

    def check_success(self, env) -> bool:
        bodies = env.rigid_engine.bodies[:self.num_objects]
        if len(bodies) < self.num_objects:
            return False

        # 检查所有物体都堆叠且静止
        sorted_bodies = sorted(bodies, key=lambda b: b.z)
        for i in range(1, len(sorted_bodies)):
            dx = sorted_bodies[i].x - sorted_bodies[i-1].x
            dy = sorted_bodies[i].y - sorted_bodies[i-1].y
            horiz_dist = np.sqrt(dx**2 + dy**2)
            threshold = sorted_bodies[i].radius + sorted_bodies[i-1].radius + 0.3

            if horiz_dist >= threshold:
                return False
            if sorted_bodies[i].z <= sorted_bodies[i-1].z + sorted_bodies[i-1].radius:
                return False

        # 检查稳定性：所有物体几乎静止
        for b in bodies:
            speed = np.sqrt(b.vx**2 + b.vy**2 + b.vz**2)
            if speed > 0.1:
                return False

        return True

    def check_failure(self, env) -> bool:
        return False

    def _compute_progress(self, env) -> float:
        stacked = self._get_stacked_pairs(env)
        max_pairs = self.num_objects - 1
        return stacked / max_pairs if max_pairs > 0 else 0.0


class SortByMaterialTask(PhysicsTask):
    """
    按材质分类物体（Level 3）

    需要理解的物理概念：
    - 材质决定质量（密度 × 体积）
    - 材质决定摩擦和弹性
    - Agent 必须从物理行为推断材质，而非颜色

    奖励设计：
    - 每个正确分类的物体获得奖励
    """

    def __init__(self, zones: Dict, max_steps: int = 500):
        """
        参数：
            zones: 分类区域 {material_name: (x_min, y_min, x_max, y_max)}
        """
        super().__init__(TaskCategory.MANIPULATION, Difficulty.LEVEL_3, max_steps)
        self.zones = zones
        # 每个刚体的目标材质（由 setup 设置）
        self.target_materials: Dict[int, str] = {}

    def setup(self, env):
        """为每个刚体分配目标区域"""
        for i, body in enumerate(env.rigid_engine.bodies):
            # 循环分配材质
            materials = list(self.zones.keys())
            self.target_materials[i] = materials[i % len(materials)]

    def _is_in_zone(self, x: float, y: float, zone: tuple) -> bool:
        """检查点是否在区域内"""
        x_min, y_min, x_max, y_max = zone
        return x_min <= x <= x_max and y_min <= y <= y_max

    def compute_reward(self, env, prev_obs: Dict, curr_obs: Dict) -> float:
        correct_count = 0
        for i, body in enumerate(env.rigid_engine.bodies):
            if i in self.target_materials:
                target_mat = self.target_materials[i]
                if target_mat in self.zones:
                    if self._is_in_zone(body.x, body.y, self.zones[target_mat]):
                        correct_count += 1
        return correct_count * 0.1

    def check_success(self, env) -> bool:
        for i, body in enumerate(env.rigid_engine.bodies):
            if i in self.target_materials:
                target_mat = self.target_materials[i]
                if target_mat in self.zones:
                    if not self._is_in_zone(body.x, body.y, self.zones[target_mat]):
                        return False
        return True

    def check_failure(self, env) -> bool:
        return False

    def _compute_progress(self, env) -> float:
        if not self.target_materials:
            return 0.0
        correct = 0
        for i, body in enumerate(env.rigid_engine.bodies):
            if i in self.target_materials:
                target_mat = self.target_materials[i]
                if target_mat in self.zones:
                    if self._is_in_zone(body.x, body.y, self.zones[target_mat]):
                        correct += 1
        return correct / len(self.target_materials)
