"""推理模块 — 抽象、演绎、类比、归纳推理引擎"""

from src.reasoning.abstraction import AbstractionEngine, Category, PropertySchema, AbstractRule
from src.reasoning.deductive import DeductiveReasoner, LogicRule
from src.reasoning.analogical import AnalogicalReasoner, AnalogyMapping
from src.reasoning.inductive import InductiveReasoner
from src.reasoning.engine import ReasoningEngine
from src.reasoning.metaphor import MetaphorTracker, METAPHOR_MAPPINGS
from src.reasoning.theory_of_mind import TheoryOfMindModule, Perspective, PERSPECTIVE_MARKERS
from src.reasoning.causal import CausalReasoningModule, CausalRule
from src.reasoning.causal_unified import UnifiedCausalReasoner, CausalResult
from src.reasoning.counterfactual import CounterfactualModule, CounterfactualWorld, COUNTERFACTUAL_MARKERS
from src.reasoning.tool_use import ToolUseModule, Tool, Goal, TOOL_TEMPLATES, GOAL_TEMPLATES
from src.reasoning.questioning import QuestioningModule, QUESTION_MARKERS

__all__ = [
    'AbstractionEngine', 'Category', 'PropertySchema', 'AbstractRule',
    'DeductiveReasoner', 'LogicRule',
    'AnalogicalReasoner', 'AnalogyMapping',
    'InductiveReasoner',
    'ReasoningEngine',
    'MetaphorTracker', 'METAPHOR_MAPPINGS',
    'TheoryOfMindModule', 'Perspective', 'PERSPECTIVE_MARKERS',
    'CausalReasoningModule', 'CausalRule',
    'UnifiedCausalReasoner', 'CausalResult',
    'CounterfactualModule', 'CounterfactualWorld', 'COUNTERFACTUAL_MARKERS',
    'ToolUseModule', 'Tool', 'Goal', 'TOOL_TEMPLATES', 'GOAL_TEMPLATES',
    'QuestioningModule', 'QUESTION_MARKERS',
]
