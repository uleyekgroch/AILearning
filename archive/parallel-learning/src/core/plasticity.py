"""
关键期可塑性 — 学习率的时间衰减调度

模拟神经发育中的关键期现象：学习率随发育时间从 1.0 衰减至 floor，
不同衰减曲线对应不同的发育模式。所有计算使用纯 torch 运算。

5 种调度:
  exponential — 指数衰减，最陡峭
  sigmoid     — S 曲线，中间段急降
  linear      — 线性衰减
  step        — 阶梯函数，70% 处骤降
  none        — 恒定 1.0（无衰减）
"""

import torch
from typing import Dict


# ── 调度函数 ──────────────────────────────────────────────────

def plasticity_exponential(step: int, total_steps: int, floor: float = 0.1) -> float:
    """指数衰减: exp(-5 × progress)

    早期快速下降，后期趋于平缓。
    progress ∈ [0, 1] 表示发育进度。
    """
    progress = torch.tensor(step / max(total_steps, 1), dtype=torch.float32)
    value = torch.exp(torch.tensor(-5.0) * progress)
    return max(value.item(), floor)


def plasticity_sigmoid(step: int, total_steps: int, floor: float = 0.1) -> float:
    """S 曲线衰减: 中点附近急降

    使用缩放后的 sigmoid，使得 progress=0 时≈1.0，progress=1 时接近 floor。
    陡降区域集中在 progress ≈ 0.4–0.6。
    """
    progress = torch.tensor(step / max(total_steps, 1), dtype=torch.float32)
    # 将 progress 映射到 sigmoid 输入域：中心在 0.5，斜率足够陡
    steepness = 12.0
    shifted = steepness * (progress - 0.5)
    raw = 1.0 / (1.0 + torch.exp(shifted))
    # raw 在 progress=0 时≈1.0, progress=1 时≈0，线性缩放到 [floor, 1.0]
    value = floor + (1.0 - floor) * raw
    return float(torch.clamp(value, min=floor, max=1.0).item())


def plasticity_linear(step: int, total_steps: int, floor: float = 0.1) -> float:
    """线性衰减: 从 1.0 直线降至 floor"""
    progress = torch.tensor(step / max(total_steps, 1), dtype=torch.float32)
    value = 1.0 - (1.0 - floor) * progress
    return float(torch.clamp(value, min=floor, max=1.0).item())


def plasticity_step(step: int, total_steps: int, floor: float = 0.1) -> float:
    """阶梯衰减: 70% 前保持 1.0，之后骤降到 floor

    模拟关键期的突然关闭。
    """
    progress = step / max(total_steps, 1)
    if progress < 0.7:
        return 1.0
    return floor


def plasticity_none(step: int, total_steps: int, floor: float = 0.1) -> float:
    """无衰减: 恒定可塑性 1.0"""
    return 1.0


# ── 调度注册表 ────────────────────────────────────────────────

PLASTICITY_SCHEDULES: Dict[str, type] = {
    'exponential': plasticity_exponential,
    'sigmoid': plasticity_sigmoid,
    'linear': plasticity_linear,
    'step': plasticity_step,
    'none': plasticity_none,
}


# ── 可塑性调度器 ─────────────────────────────────────────────

class PlasticityScheduler:
    """关键期可塑性调度器

    管理学习率的时间门控：根据选定的衰减曲线，
    随发育时间逐步降低可塑性系数。

    用法:
        scheduler = PlasticityScheduler('exponential', floor=0.1, total_steps=10000)
        for step in range(10000):
            lr = base_lr * scheduler.get_plasticity()
            scheduler.step()
    """

    def __init__(self, schedule_name: str = 'none', floor: float = 0.1,
                 total_steps: int = 10000):
        if schedule_name not in PLASTICITY_SCHEDULES:
            raise ValueError(
                f"未知调度 '{schedule_name}'，"
                f"可选: {list(PLASTICITY_SCHEDULES.keys())}"
            )
        self._schedule_fn = PLASTICITY_SCHEDULES[schedule_name]
        self._schedule_name = schedule_name
        self._floor = floor
        self._total_steps = max(total_steps, 1)
        self._current_step = 0

    def get_plasticity(self, step: int = None) -> float:
        """计算当前可塑性系数，返回 [floor, 1.0] 区间的浮点数

        Args:
            step: 指定步数，None 则使用内部计数器。

        Returns:
            当前可塑性值。
        """
        s = step if step is not None else self._current_step
        return self._schedule_fn(s, self._total_steps, self._floor)

    def step(self) -> None:
        """推进内部步数计数器"""
        self._current_step += 1

    def reset(self) -> None:
        """重置内部步数为 0"""
        self._current_step = 0

    @property
    def progress(self) -> float:
        """当前发育进度 [0, 1]"""
        return self._current_step / self._total_steps

    @property
    def schedule_name(self) -> str:
        """当前调度名称"""
        return self._schedule_name

    def save_state(self) -> dict:
        """序列化调度器状态"""
        return {
            'schedule_name': self._schedule_name,
            'floor': self._floor,
            'total_steps': self._total_steps,
            'current_step': self._current_step,
        }

    def load_state(self, state: dict) -> None:
        """从字典恢复调度器状态"""
        self._schedule_name = state['schedule_name']
        self._floor = state['floor']
        self._total_steps = max(state['total_steps'], 1)
        self._current_step = state['current_step']
        self._schedule_fn = PLASTICITY_SCHEDULES[self._schedule_name]
