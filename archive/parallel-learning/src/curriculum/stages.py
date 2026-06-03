"""
统一学习系统 — Piaget 发展阶段定义

8 个阶段与原 agent.py 的 DevelopmentEngine.STAGES 完全一致。
"""

from collections import OrderedDict
from src.core.config import StageDefinition


DEVELOPMENT_STAGES = OrderedDict({
    'sensorimotor': StageDefinition(
        name='感知运动阶段',
        age_range_months=(0, 24),
        description='通过感觉和运动与世界交互',
        promotion_criteria={
            'prediction_accuracy': 0.5,
            'exploration_diversity': 0.3,
            'total_steps': 50,
        },
        abilities=['basic_perception', 'simple_action', 'object_permanence'],
        limitations=['no_symbolic', 'no_abstract'],
    ),
    'early_preoperational': StageDefinition(
        name='前运算早期',
        age_range_months=(24, 48),
        description='符号功能出现，简单沟通',
        promotion_criteria={
            'symbol_count': 2,
            'social_reference': True,
            'total_steps': 200,
        },
        abilities=['symbolic_representation', 'simple_communication', 'egocentric_thinking'],
        limitations=['no_reversible_ops', 'no_abstract_logic', 'no_grammar'],
    ),
    'late_preoperational': StageDefinition(
        name='前运算后期',
        age_range_months=(48, 72),
        description='语法、否定、时态、基本叙事',
        promotion_criteria={
            'symbol_count': 4,
            'experience_count': 300,
            'classification_accuracy': 0.3,
            'total_steps': 500,
        },
        abilities=['grammar', 'negation', 'tense', 'basic_narrative', 'simple_classification'],
        limitations=['no_conservation', 'no_transitivity'],
    ),
    'early_concrete': StageDefinition(
        name='具体运算早期',
        age_range_months=(72, 96),
        description='守恒、分类、序列化',
        promotion_criteria={
            'classification_accuracy': 0.5,
            'conservation_test': True,
            'seriation_score': 0.3,
            'communication_success': 0.3,
            'total_steps': 800,
        },
        abilities=['conservation', 'classification', 'seriation', 'logical_reasoning'],
        limitations=['concrete_only', 'no_hypothetical'],
    ),
    'late_concrete': StageDefinition(
        name='具体运算后期',
        age_range_months=(96, 132),
        description='传递性、多步规划、类比推理',
        promotion_criteria={
            'abstract_reasoning_score': 0.3,
            'planning_score': 0.4,
            'classification_accuracy': 0.6,
            'communication_success': 0.5,
            'total_steps': 1200,
        },
        abilities=['transitivity', 'multi_step_planning', 'analogy', 'abstract_reasoning'],
        limitations=['no_hypothetical_deductive'],
    ),
    'early_formal': StageDefinition(
        name='形式运算早期',
        age_range_months=(132, 156),
        description='假设检验、反事实推理、系统性推理',
        promotion_criteria={
            'hypothesis_confirmed': 0.5,
            'counterfactual_diversity': 0.3,
            'abstract_reasoning_score': 0.4,
            'total_steps': 1800,
        },
        abilities=['hypothesis_testing', 'counterfactual_thinking', 'systematic_reasoning'],
        limitations=[],
    ),
    'late_formal': StageDefinition(
        name='形式运算后期',
        age_range_months=(156, 180),
        description='科学推理、视角协调、元认知',
        promotion_criteria={
            'meta_cognition': 0.5,
            'perspective_coordination': 0.4,
            'hypothesis_confirmed': 0.6,
            'total_steps': 2500,
        },
        abilities=['scientific_reasoning', 'perspective_coordination', 'meta_cognition'],
        limitations=[],
    ),
    'adolescent': StageDefinition(
        name='青少年阶段',
        age_range_months=(180, 204),
        description='抽象问题解决、道德推理、身份认同',
        promotion_criteria={},
        abilities=['abstract_problem_solving', 'moral_reasoning', 'identity_formation',
                    'hypothetical_deductive', 'scientific_thinking'],
        limitations=[],
    ),
})


def get_stage_names() -> list:
    """返回有序阶段名称列表"""
    return list(DEVELOPMENT_STAGES.keys())


def get_stage(name: str) -> StageDefinition:
    """获取阶段定义"""
    return DEVELOPMENT_STAGES[name]


def next_stage(name: str) -> str | None:
    """返回下一阶段名，若无则 None"""
    names = get_stage_names()
    idx = names.index(name)
    if idx < len(names) - 1:
        return names[idx + 1]
    return None
