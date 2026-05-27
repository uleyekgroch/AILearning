"""
风险敏感任务

三个任务测试不确定性感知决策：
1. CliffNavigationTask — 悬崖边缘导航（掉落=失败）
2. HazardousExplorationTask — 隐藏危险区探索（进入=失败）
3. RiskRewardTradeoffTask — 安全长路径 vs 危险短路径

这是系统中首次实现 check_failure() 返回 True 的任务。
之前所有任务的失败条件都是 False（永不失败）。
"""

import numpy as np
from typing import Dict, List, Tuple
from tasks_base import PhysicsTask, TaskCategory, Difficulty


class RiskTaskBase(PhysicsTask):
    """
    风险任务基类

    提供危险区检测的通用逻辑。
    子类只需定义危险区坐标和成功条件。
    """

    def __init__(self, category: TaskCategory, difficulty: Difficulty,
                 goal_pos: np.ndarray, goal_radius: float,
                 hazard_zones: List[Tuple[float, float, float, float]],
                 max_steps: int = 500):
        super().__init__(category, difficulty, max_steps)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.goal_radius = goal_radius
        self.hazard_zones = hazard_zones  # [(x_min, y_min, x_max, y_max), ...]
        self.initial_distance = 0.0
        self.hazard_steps = 0
        self.near_hazard_steps = 0

    def _get_agent_pos(self, env) -> np.ndarray:
        return np.array([env.agent_x, env.agent_y, env.agent_z])

    def _get_agent_distance(self, env) -> float:
        return np.linalg.norm(self._get_agent_pos(env) - self.goal_pos)

    def _is_in_hazard(self, env) -> bool:
        for (x_min, y_min, x_max, y_max) in self.hazard_zones:
            if x_min <= env.agent_x <= x_max and y_min <= env.agent_y <= y_max:
                return True
        return False

    def _nearest_hazard_distance(self, env) -> float:
        """到最近危险区边缘的距离"""
        min_dist = float('inf')
        for (x_min, y_min, x_max, y_max) in self.hazard_zones:
            cx = np.clip(env.agent_x, x_min, x_max)
            cy = np.clip(env.agent_y, y_min, y_max)
            d = np.sqrt((env.agent_x - cx)**2 + (env.agent_y - cy)**2)
            min_dist = min(min_dist, d)
        return min_dist

    def _compute_progress(self, env) -> float:
        dist = self._get_agent_distance(env)
        return max(0.0, 1.0 - dist / max(self.initial_distance, 0.01))


class CliffNavigationTask(RiskTaskBase):
    """
    悬崖导航任务

    布局（10×10）：
    - 起点：(3, 3)
    - 终点：(7, 7)
    - 悬崖：四边边缘窄带（宽度 1.0）
    - 安全区：中间区域 (1.0-9.0)

    FEP 优势：精度低时远离边缘，避免不确定性高的动作。
    MSE agent：无不确定性信息，可能走捷径靠近悬崖。
    """

    def __init__(self, max_steps: int = 200):
        hazard_zones = [
            (-0.5, -0.5, 0.8, 10.5),   # 左边缘
            (-0.5, -0.5, 10.5, 0.8),   # 下边缘
            (9.2, -0.5, 10.5, 10.5),   # 右边缘
            (-0.5, 9.2, 10.5, 10.5),   # 上边缘
        ]
        super().__init__(
            TaskCategory.RISK_NAVIGATION, Difficulty.LEVEL_2,
            goal_pos=np.array([7.0, 7.0, 1.0]),
            goal_radius=1.5,
            hazard_zones=hazard_zones,
            max_steps=max_steps,
        )

    def setup(self, env):
        env.agent_x = 3.0
        env.agent_y = 3.0
        env.agent_z = 1.0
        self.initial_distance = self._get_agent_distance(env)
        self.hazard_steps = 0
        self.near_hazard_steps = 0

    def compute_reward(self, env, prev_obs, curr_obs):
        dist = self._get_agent_distance(env)
        progress = (self.initial_distance - dist) / max(self.initial_distance, 0.01)

        reward = 0.05 * progress  # 更强的进度信号

        # 到达目标奖励
        if dist < self.goal_radius:
            reward += 1.0

        # 靠近悬崖惩罚（预警区域）
        near_dist = self._nearest_hazard_distance(env)
        if near_dist < 1.5:
            reward -= 0.03 * (1.5 - near_dist) / 1.5
            self.near_hazard_steps += 1

        return reward

    def check_success(self, env):
        return self._get_agent_distance(env) < self.goal_radius

    def check_failure(self, env):
        if self._is_in_hazard(env):
            self.hazard_steps += 1
            return True
        return False


class HazardousExplorationTask(RiskTaskBase):
    """
    危险探索任务

    布局：
    - 起点：(2, 5)
    - 终点：(8, 5)
    - 两个危险区在路径两侧（中间有安全通道）
    - agent 必须探索找到安全路径

    FEP 优势：惊奇度信号预警危险，精度低的区域谨慎探索。
    MSE agent：盲目探索，可能误入危险区。
    """

    def __init__(self, max_steps: int = 200):
        hazard_zones = [
            (3.5, 2.0, 5.5, 4.5),    # 危险区 1（下半部分）
            (5.5, 5.5, 7.5, 8.0),    # 危险区 2（上半部分）
        ]
        super().__init__(
            TaskCategory.RISK_NAVIGATION, Difficulty.LEVEL_3,
            goal_pos=np.array([8.0, 5.0, 1.0]),
            goal_radius=1.5,
            hazard_zones=hazard_zones,
            max_steps=max_steps,
        )

    def setup(self, env):
        env.agent_x = 2.0
        env.agent_y = 5.0
        env.agent_z = 1.0
        self.initial_distance = self._get_agent_distance(env)
        self.hazard_steps = 0
        self.near_hazard_steps = 0

    def compute_reward(self, env, prev_obs, curr_obs):
        dist = self._get_agent_distance(env)
        progress = (self.initial_distance - dist) / max(self.initial_distance, 0.01)
        reward = 0.05 * progress

        if dist < self.goal_radius:
            reward += 1.0

        # 谨慎探索奖励：靠近危险区但未进入
        near_dist = self._nearest_hazard_distance(env)
        if 1.0 < near_dist < 2.0:
            reward += 0.02  # 发现危险区边界
            self.near_hazard_steps += 1

        return reward

    def check_success(self, env):
        return self._get_agent_distance(env) < self.goal_radius

    def check_failure(self, env):
        if self._is_in_hazard(env):
            self.hazard_steps += 1
            return True
        return False


class RiskRewardTradeoffTask(RiskTaskBase):
    """
    风险-收益权衡任务

    布局：
    - 起点：(2, 2)
    - 终点：(8, 8)
    - 安全路径：沿边缘走（~12步）
    - 危险路径：对角线穿过中心（~8步），但中心有危险区 (4.5,4.5)-(5.5,5.5)

    FEP 优势：早期精度低时选安全路径，精度提高后探索危险路径。
    MSE agent：可能直接走对角线，不了解危险区位置。
    """

    def __init__(self, max_steps: int = 200):
        hazard_zones = [
            (4.5, 4.5, 5.5, 5.5),    # 中心危险区（小而精确）
        ]
        super().__init__(
            TaskCategory.RISK_NAVIGATION, Difficulty.LEVEL_2,
            goal_pos=np.array([8.0, 8.0, 1.0]),
            goal_radius=1.5,
            hazard_zones=hazard_zones,
            max_steps=max_steps,
        )
        self.path_type = 'unknown'

    def setup(self, env):
        env.agent_x = 2.0
        env.agent_y = 2.0
        env.agent_z = 1.0
        self.initial_distance = self._get_agent_distance(env)
        self.hazard_steps = 0
        self.near_hazard_steps = 0
        self.path_type = 'unknown'

    def compute_reward(self, env, prev_obs, curr_obs):
        dist = self._get_agent_distance(env)
        progress = (self.initial_distance - dist) / max(self.initial_distance, 0.01)
        reward = 0.05 * progress

        if dist < self.goal_radius:
            reward += 1.0

        # 速度奖励：每步进展越大，奖励越高
        # 这激励走短路径（对角线），创造风险-收益权衡
        reward += 0.01 * progress

        # 记录路径类型
        if env.agent_x > 4.0 and env.agent_y > 4.0 and self.path_type == 'unknown':
            self.path_type = 'risky'  # agent 进入了中心区域
        elif self.step_count > 50 and self.path_type == 'unknown':
            self.path_type = 'safe'   # agent 绕行了

        return reward

    def check_success(self, env):
        return self._get_agent_distance(env) < self.goal_radius

    def check_failure(self, env):
        if self._is_in_hazard(env):
            self.hazard_steps += 1
            return True
        return False
