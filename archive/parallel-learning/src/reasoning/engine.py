"""推理引擎 — 统一调度抽象、演绎、类比、归纳推理"""

from typing import Any, Dict, List, Optional

from src.knowledge.graph import KnowledgeGraph
from src.reasoning.abstraction import AbstractionEngine, AbstractRule
from src.reasoning.deductive import DeductiveReasoner
from src.reasoning.analogical import AnalogicalReasoner
from src.reasoning.inductive import InductiveReasoner


class ReasoningEngine:
    """推理引擎：根据任务类型分发到对应的推理子系统。

    支持的任务类型：
    - 'deduce': 演绎推理，从事实推导结论
    - 'analogy':  类比推理，解决 A:B :: C:? 问题
    - 'induce':   归纳推理，从观察中发现模式
    - 'categorize': 分类，判断实体属于哪个类别
    """

    def __init__(self, graph: KnowledgeGraph):
        self.knowledge = graph
        self.deductive = DeductiveReasoner(graph)
        self.analogical = AnalogicalReasoner(graph)
        self.inductive = InductiveReasoner(graph)
        self.abstraction = AbstractionEngine(graph)

    def reason(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """根据 task 类型分发推理。

        参数:
            task: 任务类型 ('deduce'/'analogy'/'induce'/'categorize')
            context: 任务上下文，不同任务需要不同的字段

        返回:
            包含推理结果的字典，至少包含 'success' 和 'result' 键
        """
        dispatch = {
            'deduce': self._handle_deduce,
            'analogy': self._handle_analogy,
            'induce': self._handle_induce,
            'categorize': self._handle_categorize,
        }

        handler = dispatch.get(task)
        if handler is None:
            return {
                'success': False,
                'error': f'未知任务类型: {task}，支持的类型: {list(dispatch.keys())}',
                'result': None,
            }

        try:
            return handler(context)
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'result': None,
            }

    def save_state(self) -> dict:
        """序列化推理引擎的完整状态"""
        state = {
            'deductive': {
                'rules': [
                    {
                        'premises': rule.premises,
                        'conclusion': rule.conclusion,
                        'confidence': rule.confidence,
                    }
                    for rule in self.deductive.rules
                ],
            },
            'abstraction': {
                'categories': {
                    name: {
                        'name': cat.name,
                        'members': cat.members,
                        'defining_properties': cat.defining_properties,
                        'confidence': cat.confidence,
                    }
                    for name, cat in self.abstraction.categories.items()
                },
                'property_schemas': {
                    pname: {
                        'property_name': schema.property_name,
                        'applicable_types': schema.applicable_types,
                        'observed_values': [str(v) for v in schema.observed_values],
                        'frequency': schema.frequency,
                    }
                    for pname, schema in self.abstraction.property_schemas.items()
                },
                'abstract_rules': [
                    {
                        'condition': rule.condition,
                        'conclusion': rule.conclusion,
                        'support': rule.support,
                        'confidence': rule.confidence,
                        'exceptions': rule.exceptions,
                    }
                    for rule in self.abstraction.abstract_rules
                ],
            },
        }
        return state

    def load_state(self, state: dict) -> None:
        """从序列化状态恢复推理引擎"""
        # 恢复演绎推理规则
        deductive_state = state.get('deductive', {})
        self.deductive.rules.clear()
        for rd in deductive_state.get('rules', []):
            from src.reasoning.deductive import LogicRule
            self.deductive.rules.append(LogicRule(
                premises=rd['premises'],
                conclusion=rd['conclusion'],
                confidence=rd.get('confidence', 1.0),
            ))

        # 恢复抽象引擎状态
        abs_state = state.get('abstraction', {})
        import torch

        # 恢复类别
        self.abstraction.categories.clear()
        for name, cd in abs_state.get('categories', {}).items():
            from src.reasoning.abstraction import Category
            # 原型需要重新计算（因为 tensor 不容易序列化）
            prototype = self.abstraction.compute_prototype(cd.get('members', []))
            self.abstraction.categories[name] = Category(
                name=cd['name'],
                members=cd['members'],
                defining_properties=cd['defining_properties'],
                prototype=prototype,
                confidence=cd.get('confidence', 1.0),
            )

        # 恢复属性模式
        self.abstraction.property_schemas.clear()
        for pname, sd in abs_state.get('property_schemas', {}).items():
            from src.reasoning.abstraction import PropertySchema
            self.abstraction.property_schemas[pname] = PropertySchema(
                property_name=sd['property_name'],
                applicable_types=sd['applicable_types'],
                observed_values=sd.get('observed_values', []),
                frequency=sd['frequency'],
            )

        # 恢复抽象规则
        self.abstraction.abstract_rules.clear()
        for rd in abs_state.get('abstract_rules', []):
            from src.reasoning.abstraction import AbstractRule
            self.abstraction.abstract_rules.append(AbstractRule(
                condition=rd['condition'],
                conclusion=rd['conclusion'],
                support=rd['support'],
                confidence=rd['confidence'],
                exceptions=rd.get('exceptions', []),
            ))

    # ── 内部分发方法 ──────────────────────────────────────────

    def _handle_deduce(self, context: Dict) -> Dict:
        """处理演绎推理任务"""
        facts = context.get('facts', [])
        max_depth = context.get('max_depth', 5)

        if not facts:
            return {'success': False, 'error': '缺少 facts 参数', 'result': None}

        derived = self.deductive.deduce(facts, max_depth=max_depth)
        return {
            'success': True,
            'result': {
                'input_facts': facts,
                'derived_conclusions': derived,
                'total_derived': len(derived),
            },
        }

    def _handle_analogy(self, context: Dict) -> Dict:
        """处理类比推理任务"""
        a = context.get('a')
        b = context.get('b')
        c = context.get('c')

        if a is None or b is None or c is None:
            return {'success': False, 'error': '缺少 a, b, c 参数', 'result': None}

        answer = self.analogical.solve_analogy(a, b, c)

        # 如果还提供了 domain_a / domain_b，同时做结构相似度分析
        similarity_result = None
        domain_a = context.get('domain_a')
        domain_b = context.get('domain_b')
        if domain_a and domain_b:
            mapping = self.analogical.find_structural_similarity(domain_a, domain_b)
            similarity_result = {
                'entity_map': mapping.entity_map,
                'confidence': mapping.confidence,
            }

        result = {
            'analogy': f'{a}:{b} :: {c}:{answer}',
            'answer': answer,
        }
        if similarity_result:
            result['structural_similarity'] = similarity_result

        return {'success': True, 'result': result}

    def _handle_induce(self, context: Dict) -> Dict:
        """处理归纳推理任务"""
        observations = context.get('observations', [])
        sequence = context.get('sequence', [])

        if not observations and not sequence:
            return {
                'success': False,
                'error': '缺少 observations 或 sequence 参数',
                'result': None,
            }

        result = {}

        # 模式发现
        if observations:
            rule = self.inductive.observe_pattern(observations)
            result['pattern'] = {
                'condition': rule.condition if rule else None,
                'conclusion': rule.conclusion if rule else None,
                'confidence': rule.confidence if rule else 0.0,
                'support': rule.support if rule else 0,
            } if rule else None

        # 序列预测
        if sequence:
            prediction = self.inductive.predict_next(sequence)
            result['prediction'] = prediction

        return {'success': True, 'result': result}

    def _handle_categorize(self, context: Dict) -> Dict:
        """处理分类任务"""
        entity_id = context.get('entity_id')
        instance_ids = context.get('instance_ids')  # 用于提取新类别

        if not entity_id and not instance_ids:
            return {
                'success': False,
                'error': '缺少 entity_id 或 instance_ids 参数',
                'result': None,
            }

        result = {}

        # 如果提供了 instance_ids，先提取类别
        if instance_ids:
            category = self.abstraction.extract_category(instance_ids)
            if category:
                result['extracted_category'] = {
                    'name': category.name,
                    'defining_properties': category.defining_properties,
                    'member_count': len(category.members),
                    'confidence': category.confidence,
                }

        # 如果提供了 entity_id，进行分类
        if entity_id:
            category_name = self.abstraction.get_category_for_entity(entity_id)
            result['category'] = category_name

            # 同时提取属性模式
            schemas = self.abstraction.extract_property_pattern([entity_id])
            result['property_patterns'] = [
                {
                    'property': s.property_name,
                    'frequency': s.frequency,
                    'types': s.applicable_types,
                }
                for s in schemas
            ]

        return {'success': True, 'result': result}
