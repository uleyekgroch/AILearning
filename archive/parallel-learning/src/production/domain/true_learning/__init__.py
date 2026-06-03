"""
真正的学习系统

从第一性原理出发，实现真正的学习能力：
1. 理解 - 理解含义
2. 推理 - 逻辑推理
3. 创造 - 创造新内容
4. 改进 - 自我改进
"""

from .understanding import UnderstandingEngine
from .reasoning import ReasoningEngine
from .creation import CreationEngine
from .improvement import ImprovementEngine
from .learner import TrueLearner

__all__ = [
    "UnderstandingEngine",
    "ReasoningEngine",
    "CreationEngine",
    "ImprovementEngine",
    "TrueLearner",
]
