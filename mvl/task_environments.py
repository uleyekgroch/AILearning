"""
任务环境工厂

创建预配置的物理环境和任务。
用于实验和测试。
"""

import numpy as np
from typing import Tuple

from environment_physics import PhysicsEnvironment, create_rich_physics_world
from physics_rigid import RigidBody, MaterialType
from physics_fluid import FluidRegion
from physics_soft import Cloth, Rope

from tasks_manipulation import PushToGoalTask, StackObjectsTask, SortByMaterialTask
from tasks_fluid import NavigateInFluidTask, AvoidHazardousFluidTask
from tasks_tool_use import IndirectPushTask, BridgeGapTask
from tasks_risk import CliffNavigationTask, HazardousExplorationTask, RiskRewardTradeoffTask


def create_push_to_goal_env() -> Tuple[PhysicsEnvironment, PushToGoalTask]:
    """Level 1: 推物体到目标位置（近距离）"""
    env = PhysicsEnvironment(10, 10, 10)

    # 物体在 agent 附近，目标也在附近
    body = RigidBody(5, 4, 1, radius=0.5, mass=1.0, material=MaterialType.WOOD)
    env.add_rigid_body(body)

    # 目标位置（近距离）
    goal = np.array([5.0, 6.0, 1.0])
    task = PushToGoalTask(goal_pos=goal, goal_radius=1.0, target_body_idx=0,
                           max_steps=500)

    # 绑定任务
    env.current_task = task
    task.setup(env)

    return env, task


def create_push_to_goal_env_easy() -> Tuple[PhysicsEnvironment, PushToGoalTask]:
    """Level 1: 推物体到目标位置（超近距离，agent 就在物体旁边）"""
    env = PhysicsEnvironment(10, 10, 10)

    # 物体就在 agent 旁边
    body = RigidBody(5, 5.5, 1, radius=0.5, mass=1.0, material=MaterialType.RUBBER)
    env.add_rigid_body(body)

    # Agent 在 (5, 5, 1)，物体在 (5, 5.5, 1)，目标在 (5, 7, 1)
    # Agent 需要向 y+ 移动靠近物体，然后推
    goal = np.array([5.0, 7.0, 1.0])
    task = PushToGoalTask(goal_pos=goal, goal_radius=1.0, target_body_idx=0,
                           max_steps=500)

    env.current_task = task
    task.setup(env)

    return env, task


def create_stack_objects_env() -> Tuple[PhysicsEnvironment, StackObjectsTask]:
    """Level 2: 堆叠3个不同质量的物体"""
    env = PhysicsEnvironment(10, 10, 10)

    # 添加不同质量的物体
    env.add_rigid_body(RigidBody(3, 5, 1, radius=0.5, mass=3.0, material=MaterialType.STONE))
    env.add_rigid_body(RigidBody(5, 5, 1, radius=0.5, mass=2.0, material=MaterialType.METAL))
    env.add_rigid_body(RigidBody(7, 5, 1, radius=0.5, mass=1.0, material=MaterialType.RUBBER))

    task = StackObjectsTask(num_objects=3)

    env.current_task = task
    task.setup(env)

    return env, task


def create_fluid_navigation_env() -> Tuple[PhysicsEnvironment, NavigateInFluidTask]:
    """Level 2: 在水中导航到目标"""
    env = PhysicsEnvironment(10, 10, 10)

    # 添加水区域
    env.add_fluid_region(FluidRegion('water',
                                      x_min=0, y_min=0, z_min=0,
                                      x_max=10, y_max=10, z_max=4))

    # 目标在水中
    goal = np.array([8.0, 8.0, 2.0])
    task = NavigateInFluidTask(goal_pos=goal, goal_radius=1.0)

    env.current_task = task
    task.setup(env)

    return env, task


def create_avoid_hazard_env() -> Tuple[PhysicsEnvironment, AvoidHazardousFluidTask]:
    """Level 3: 避开蜂蜜区域到达目标"""
    env = PhysicsEnvironment(12, 12, 12)

    # 添加蜂蜜区域（危险）
    env.add_fluid_region(FluidRegion('honey',
                                      x_min=4, y_min=4, z_min=0,
                                      x_max=8, y_max=8, z_max=6))

    # 目标在蜂蜜区域对面
    goal = np.array([10.0, 10.0, 1.0])
    task = AvoidHazardousFluidTask(goal_pos=goal, goal_radius=1.0,
                                    hazardous_fluid='honey')

    env.current_task = task
    task.setup(env)

    return env, task


def create_indirect_push_env() -> Tuple[PhysicsEnvironment, IndirectPushTask]:
    """Level 2: 用工具推远处物体到目标"""
    env = PhysicsEnvironment(10, 10, 10)

    # 工具物体（近处）
    tool = RigidBody(3, 5, 1, radius=0.5, mass=1.0, material=MaterialType.METAL)
    # 目标物体（远处）
    target = RigidBody(7, 5, 1, radius=0.5, mass=2.0, material=MaterialType.WOOD)

    env.add_rigid_body(tool)    # index 0
    env.add_rigid_body(target)  # index 1

    # 目标位置
    goal = np.array([9.0, 5.0, 1.0])
    task = IndirectPushTask(goal_pos=goal, tool_idx=0, target_idx=1, goal_radius=0.8)

    env.current_task = task
    task.setup(env)

    return env, task


def create_bridge_gap_env() -> Tuple[PhysicsEnvironment, BridgeGapTask]:
    """Level 3: 用绳子搭桥跨越缝隙"""
    env = PhysicsEnvironment(12, 10, 10)

    # 添加绳子
    rope = Rope(6.0, 15, mass=0.5, stiffness=200.0)
    env.add_rope(rope)

    # 缝隙在 x=4 到 x=8
    # 目标在缝隙对面
    goal = np.array([10.0, 5.0, 1.0])
    task = BridgeGapTask(goal_pos=goal, gap_x_min=4.0, gap_x_max=8.0,
                          goal_radius=1.0)

    env.current_task = task
    task.setup(env)

    return env, task


def create_no_task_env() -> PhysicsEnvironment:
    """对照组：相同物理环境，无任务"""
    return create_rich_physics_world()


def create_cliff_navigation_env() -> Tuple[PhysicsEnvironment, CliffNavigationTask]:
    """风险任务：悬崖导航（四边是悬崖，掉落=失败）"""
    env = PhysicsEnvironment(10, 10, 10)
    task = CliffNavigationTask()
    env.current_task = task
    task.setup(env)
    return env, task


def create_hazardous_exploration_env() -> Tuple[PhysicsEnvironment, HazardousExplorationTask]:
    """风险任务：危险探索（隐藏危险区，进入=失败）"""
    env = PhysicsEnvironment(10, 10, 10)
    task = HazardousExplorationTask()
    env.current_task = task
    task.setup(env)
    return env, task


def create_risk_reward_tradeoff_env() -> Tuple[PhysicsEnvironment, RiskRewardTradeoffTask]:
    """风险任务：风险-收益权衡（安全长路径 vs 危险短路径）"""
    env = PhysicsEnvironment(10, 10, 10)
    task = RiskRewardTradeoffTask()
    env.current_task = task
    task.setup(env)
    return env, task
