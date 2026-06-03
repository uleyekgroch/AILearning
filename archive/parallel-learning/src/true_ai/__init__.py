"""
真正的AI系统

从第一性原理出发，实现真正的理解、推理、创造、改进能力
不是模仿LLM，而是实现真正的AI
"""

from .perception import PerceptionModule
from .understanding import UnderstandingModule
from .reasoning import ReasoningModule
from .creation import CreationModule
from .improvement import ImprovementModule
from .ai_system import TrueAISystem

__all__ = [
    "PerceptionModule",
    "UnderstandingModule",
    "ReasoningModule",
    "CreationModule",
    "ImprovementModule",
    "TrueAISystem",
]
