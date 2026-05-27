"""
工具使用任务

任务层级：
1. IndirectPush（Level 2）：用工具推远处物体 → 理解动量传递
2. BridgeGap（Level 3）：用绳子/布料搭桥 → 理解软体力学
"""

import numpy as np
from typing import Dict
from tasks_base import PhysicsTask, TaskCategory, Difficulty


class IndirectPushTask(PhysicsTask):
    """
    用工具推远处物体到目标位置（Level 2）

    需要理解的物理概念：
    - 碰撞传递动量
    - 被推物体的质量和材质影响移动距离
    - 必须对齐工具、目标物体和目标位置

    奖励设计（密集）：
    - 主要：目标物体到目标的距离减少
    - 中间奖励：工具接触目标物体（动量传递检测）
    - 最终奖励：目标物体到达目标
    """

    def __init__(self, goal_pos: np.ndarray, tool_idx: int = 0,
                 target_idx: int = 1, goal_radius: float = 0.8,
                 max_steps: int = 400):
        super().__init__(TaskCategory.TOOL_USE, Difficulty.LEVEL_2, max_steps)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.tool_idx = tool_idx
        self.target_idx = target_idx
        self.goal_radius = goal_radius
        self.initial_target_distance = 0.0
        self.tool_touched_target = False

    def setup(self, env):
        """记录初始距离"""
        if self.target_idx < len(env.rigid_engine.bodies):
            target = env.rigid_engine.bodies[self.target_idx]
            self.initial_target_distance = np.sqrt(
                (target.x - self.goal_pos[0])**2 +
                (target.y - self.goal_pos[1])**2 +
                (target.z - self.goal_pos[2])**2
            )

    def _get_target_distance(self, env) -> float:
        """计算目标物体到目标位置的距离"""
        target = env.rigid_engine.bodies[self.target_idx]
        return np.sqrt(
            (target.x - self.goal_pos[0])**2 +
            (target.y - self.goal_pos[1])**2 +
            (target.z - self.goal_pos[2])**2
        )

    def _check_tool_target_contact(self, env) -> bool:
        """检查工具是否接触目标物体"""
        tool = env.rigid_engine.bodies[self.tool_idx]
        target = env.rigid_engine.bodies[self.target_idx]
        dist = np.sqrt(
            (tool.x - target.x)**2 +
            (tool.y - target.y)**2 +
            (tool.z - target.z)**2
        )
        return dist < tool.radius + target.radius + 0.1

    def compute_reward(self, env, prev_obs: Dict, curr_obs: Dict) -> float:
        reward = 0.0

        # 进度奖励：目标物体接近目标
        dist = self._get_target_distance(env)
        if self.initial_target_distance > 0.01:
            progress = (self.initial_target_distance - dist) / self.initial_target_distance
            reward += 0.01 * progress

        # 接触奖励：工具接触目标物体（首次）
        if self._check_tool_target_contact(env):
            if not self.tool_touched_target:
                reward += 0.3  # 首次接触奖励
                self.tool_touched_target = True

        # 到达目标奖励
        if dist < self.goal_radius:
            reward += 1.0

        return reward

    def check_success(self, env) -> bool:
        return self._get_target_distance(env) < self.goal_radius

    def check_failure(self, env) -> bool:
        return False

    def _compute_progress(self, env) -> float:
        dist = self._get_target_distance(env)
        if self.initial_target_distance > 0.01:
            return max(0.0, 1.0 - dist / self.initial_target_distance)
        return 0.0


class BridgeGapTask(PhysicsTask):
    """
    用绳子/布料搭桥跨越障碍到达目标（Level 3）

    需要理解的物理概念：
    - 软体在重力下会变形
    - 绳子如果两端固定可以跨越缝隙
    - Agent 可以站在绳子/布料上（布料提供更大支撑）

    奖励设计：
    - 中间奖励：绳子端点在缝隙两侧
    - 进度奖励：Agent 跨越缝隙
    - 最终奖励：Agent 到达目标
    """

    def __init__(self, goal_pos: np.ndarray, gap_x_min: float = 4.0,
                 gap_x_max: float = 6.0, goal_radius: float = 1.0,
                 max_steps: int = 500):
        super().__init__(TaskCategory.TOOL_USE, Difficulty.LEVEL_3, max_steps)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.gap_x_min = gap_x_min
        self.gap_x_max = gap_x_max
        self.goal_radius = goal_radius
        self.initial_distance = 0.0
        self.bridge_formed = False

    def setup(self, env):
        agent_pos = np.array([env.agent_x, env.agent_y, env.agent_z])
        self.initial_distance = np.linalg.norm(agent_pos - self.goal_pos)

    def _get_agent_distance(self, env) -> float:
        agent_pos = np.array([env.agent_x, env.agent_y, env.agent_z])
        return np.linalg.norm(agent_pos - self.goal_pos)

    def _check_bridge_formed(self, env) -> bool:
        """检查是否形成了桥（绳子/布料跨越缝隙）"""
        # 检查布料
        for cloth in env.cloths:
            points = cloth.points
            if not points:
                continue
            # 检查是否有端点在缝隙两侧
            left_points = [p for p in points if p.x < self.gap_x_min]
            right_points = [p for p in points if p.x > self.gap_x_max]
            if left_points and right_points:
                return True

        # 检查绳子
        for rope in env.ropes:
            points = rope.points
            if not points:
                continue
            left_points = [p for p in points if p.x < self.gap_x_min]
            right_points = [p for p in points if p.x > self.gap_x_max]
            if left_points and right_points:
                return True

        return False

    def compute_reward(self, env, prev_obs: Dict, curr_obs: Dict) -> float:
        reward = 0.0

        # 桥梁形成奖励
        if self._check_bridge_formed(env):
            if not self.bridge_formed:
                reward += 0.5  # 首次形成桥梁
                self.bridge_formed = True

        # 进度奖励
        dist = self._get_agent_distance(env)
        if self.initial_distance > 0.01:
            progress = (self.initial_distance - dist) / self.initial_distance
            reward += 0.01 * progress

        # 到达目标奖励
        if dist < self.goal_radius:
            reward += 1.0

        return reward

    def check_success(self, env) -> bool:
        return self._get_agent_distance(env) < self.goal_radius

    def check_failure(self, env) -> bool:
        return False

    def _compute_progress(self, env) -> float:
        dist = self._get_agent_distance(env)
        if self.initial_distance > 0.01:
            return max(0.0, 1.0 - dist / self.initial_distance)
        return 0.0
