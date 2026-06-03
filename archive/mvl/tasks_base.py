"""
物理任务基础框架

设计原则：
1. 组合而非继承：任务叠加在 PhysicsEnvironment 之上
2. 好奇心为主，任务为辅：70% 好奇心 + 30% 任务奖励
3. 密集奖励：基于进度的连续信号

任务类型：
- 物体操控：推、堆叠、分类
- 流体导航：在流体中移动、避开危险区域
- 工具使用：间接推动物体、搭桥
"""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Tuple, Optional


class TaskCategory(Enum):
    """任务类别"""
    MANIPULATION = "manipulation"          # 物体操控
    FLUID_NAVIGATION = "fluid_navigation"  # 流体导航
    TOOL_USE = "tool_use"                  # 工具使用
    RISK_NAVIGATION = "risk_navigation"    # 风险导航


class Difficulty(Enum):
    """任务难度（对应发展阶段）"""
    LEVEL_1 = 1  # sensorimotor：基本感知和动作
    LEVEL_2 = 2  # pre_operational：符号表示和简单推理
    LEVEL_3 = 3  # concrete_operational：逻辑推理和分类


class PhysicsTask(ABC):
    """
    物理任务抽象基类

    子类需要实现：
    - setup(): 配置环境（添加物体、流体区域等）
    - compute_reward(): 计算任务奖励
    - check_success(): 检查是否成功
    - check_failure(): 检查是否失败
    - _compute_progress(): 计算完成进度
    """

    def __init__(self, category: TaskCategory, difficulty: Difficulty,
                 max_steps: int = 500):
        self.category = category
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.step_count = 0
        self.is_success = False
        self.is_failure = False

    @abstractmethod
    def setup(self, env) -> None:
        """配置环境"""
        pass

    @abstractmethod
    def compute_reward(self, env, prev_obs: Dict, curr_obs: Dict) -> float:
        """计算任务奖励"""
        pass

    @abstractmethod
    def check_success(self, env) -> bool:
        """检查是否成功"""
        pass

    @abstractmethod
    def check_failure(self, env) -> bool:
        """检查是否失败"""
        pass

    @abstractmethod
    def _compute_progress(self, env) -> float:
        """计算完成进度 [0, 1]"""
        pass

    def step(self, env, prev_obs: Dict, curr_obs: Dict) -> Tuple[float, bool]:
        """执行一步，返回 (reward, done)"""
        self.step_count += 1
        reward = self.compute_reward(env, prev_obs, curr_obs)
        self.is_success = self.check_success(env)
        self.is_failure = self.check_failure(env)
        timeout = self.step_count >= self.max_steps
        done = self.is_success or self.is_failure or timeout
        return reward, done

    def reset(self):
        """重置任务状态"""
        self.step_count = 0
        self.is_success = False
        self.is_failure = False

    def get_task_info(self, env) -> Dict:
        """获取任务信息（用于观测）"""
        return {
            'category': self.category.value,
            'difficulty': self.difficulty.value,
            'progress': self._compute_progress(env),
            'is_success': self.is_success,
            'is_failure': self.is_failure,
            'steps_remaining': self.max_steps - self.step_count,
        }


class TaskManager:
    """
    任务管理器

    根据发展阶段选择合适的任务，记录任务结果。
    """

    def __init__(self):
        self.task_history: List[Dict] = []
        self.success_rates: Dict[str, List[bool]] = {}

    def select_task(self, stage: str, task_registry: Dict) -> Optional[PhysicsTask]:
        """
        根据发展阶段选择任务

        参数：
            stage: 当前发展阶段 ('sensorimotor', 'pre_operational', 'concrete_operational')
            task_registry: 任务注册表 {Difficulty: [TaskClass, ...]}
        """
        stage_to_difficulty = {
            'sensorimotor': Difficulty.LEVEL_1,
            'pre_operational': Difficulty.LEVEL_2,
            'concrete_operational': Difficulty.LEVEL_3,
        }
        difficulty = stage_to_difficulty.get(stage, Difficulty.LEVEL_1)

        if difficulty not in task_registry or not task_registry[difficulty]:
            return None

        # 选择成功率最低的任务（需要更多练习）
        candidates = task_registry[difficulty]
        best_task = None
        best_rate = 1.0

        for task_class in candidates:
            name = task_class.__name__
            if name in self.success_rates and self.success_rates[name]:
                rate = sum(self.success_rates[name]) / len(self.success_rates[name])
            else:
                rate = 0.0  # 未尝试过的任务优先

            if rate < best_rate:
                best_rate = rate
                best_task = task_class

        if best_task is None:
            best_task = candidates[0]

        return best_task()

    def record_outcome(self, task_name: str, success: bool):
        """记录任务结果"""
        if task_name not in self.success_rates:
            self.success_rates[task_name] = []
        self.success_rates[task_name].append(success)

        self.task_history.append({
            'task': task_name,
            'success': success,
        })
