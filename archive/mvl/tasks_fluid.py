"""
流体导航任务

任务层级：
1. NavigateInFluid（Level 2）：在流体中导航到目标 → 理解阻力和浮力
2. AvoidHazardousFluid（Level 3）：避开危险流体 → 理解粘度和路径规划
"""

import numpy as np
from typing import Dict
from tasks_base import PhysicsTask, TaskCategory, Difficulty


class NavigateInFluidTask(PhysicsTask):
    """
    在流体中导航到目标位置（Level 2）

    需要理解的物理概念：
    - 流体阻力与运动方向相反
    - 浮力影响垂直位置
    - 不同流体（水 vs 蜂蜜）有不同的阻力系数
    - Agent 必须规划运动以应对流体阻力

    奖励设计（密集）：
    - 主要：到目标的距离减少
    - 奖励：到达目标
    """

    def __init__(self, goal_pos: np.ndarray, goal_radius: float = 1.0,
                 max_steps: int = 400):
        super().__init__(TaskCategory.FLUID_NAVIGATION, Difficulty.LEVEL_2, max_steps)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.goal_radius = goal_radius
        self.initial_distance = 0.0

    def setup(self, env):
        """记录初始距离"""
        agent_pos = np.array([env.agent_x, env.agent_y, env.agent_z])
        self.initial_distance = np.linalg.norm(agent_pos - self.goal_pos)

    def _get_agent_distance(self, env) -> float:
        """计算 agent 到目标的距离"""
        agent_pos = np.array([env.agent_x, env.agent_y, env.agent_z])
        return np.linalg.norm(agent_pos - self.goal_pos)

    def compute_reward(self, env, prev_obs: Dict, curr_obs: Dict) -> float:
        dist = self._get_agent_distance(env)

        # 进度奖励
        if self.initial_distance > 0.01:
            progress_reward = (self.initial_distance - dist) / self.initial_distance
        else:
            progress_reward = 0.0

        # 接近奖励
        proximity_bonus = 1.0 if dist < self.goal_radius else 0.0

        return 0.01 * progress_reward + proximity_bonus * 0.2

    def check_success(self, env) -> bool:
        return self._get_agent_distance(env) < self.goal_radius

    def check_failure(self, env) -> bool:
        return False

    def _compute_progress(self, env) -> float:
        dist = self._get_agent_distance(env)
        if self.initial_distance > 0.01:
            return max(0.0, 1.0 - dist / self.initial_distance)
        return 0.0


class AvoidHazardousFluidTask(PhysicsTask):
    """
    避开危险流体区域到达目标（Level 3）

    需要理解的物理概念：
    - 高粘度流体会大幅降低移动速度
    - Agent 必须从行为变化识别流体边界
    - 路径规划绕过障碍和流体区域

    奖励设计（密集）：
    - 主要：到目标的距离减少
    - 惩罚：处于危险区域内（每步 -0.05）
    - 奖励：到达目标
    """

    def __init__(self, goal_pos: np.ndarray, goal_radius: float = 1.0,
                 hazardous_fluid: str = 'honey', max_steps: int = 400):
        super().__init__(TaskCategory.FLUID_NAVIGATION, Difficulty.LEVEL_3, max_steps)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.goal_radius = goal_radius
        self.hazardous_fluid = hazardous_fluid
        self.initial_distance = 0.0

    def setup(self, env):
        agent_pos = np.array([env.agent_x, env.agent_y, env.agent_z])
        self.initial_distance = np.linalg.norm(agent_pos - self.goal_pos)

    def _get_agent_distance(self, env) -> float:
        agent_pos = np.array([env.agent_x, env.agent_y, env.agent_z])
        return np.linalg.norm(agent_pos - self.goal_pos)

    def _is_in_hazardous_fluid(self, env) -> bool:
        """检查 agent 是否在危险流体中"""
        for region in env.fluid_engine.regions:
            if region.fluid.name == self.hazardous_fluid:
                if region.contains_point(env.agent_x, env.agent_y, env.agent_z):
                    return True
        return False

    def compute_reward(self, env, prev_obs: Dict, curr_obs: Dict) -> float:
        dist = self._get_agent_distance(env)

        # 进度奖励
        if self.initial_distance > 0.01:
            progress_reward = (self.initial_distance - dist) / self.initial_distance
        else:
            progress_reward = 0.0

        # 接近奖励
        proximity_bonus = 1.0 if dist < self.goal_radius else 0.0

        # 危险区域惩罚
        hazard_penalty = -0.05 if self._is_in_hazardous_fluid(env) else 0.0

        return 0.01 * progress_reward + proximity_bonus * 0.2 + hazard_penalty

    def check_success(self, env) -> bool:
        return self._get_agent_distance(env) < self.goal_radius

    def check_failure(self, env) -> bool:
        return False

    def _compute_progress(self, env) -> float:
        dist = self._get_agent_distance(env)
        if self.initial_distance > 0.01:
            return max(0.0, 1.0 - dist / self.initial_distance)
        return 0.0
