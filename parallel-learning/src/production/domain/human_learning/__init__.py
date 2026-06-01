"""
人类学习系统 - 从第一性原理出发

基于人类学习的本质：
1. 感知-预测循环
2. 记忆双重系统
3. STDP/Hebbian学习
4. 概念形成与推理
"""

from .predictive_system import PredictiveLearningSystem
from .memory_consolidation import MemoryConsolidation
from .biological_learning import BiologicalLearning
from .concept_formation import ConceptFormation

__all__ = [
    "PredictiveLearningSystem",
    "MemoryConsolidation",
    "BiologicalLearning",
    "ConceptFormation",
]
