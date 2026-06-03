"""抽象能力 — 从具体实例中提取类别、属性模式和抽象规则"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from src.knowledge.graph import KnowledgeGraph


@dataclass
class Category:
    """类别：一组具有共同属性的实体的抽象"""
    name: str
    members: List[str]  # 实体 ID 列表
    defining_properties: Dict[str, Any]  # 定义该类别的属性
    prototype: Optional[torch.Tensor]  # 均值向量（原型）
    confidence: float


@dataclass
class PropertySchema:
    """属性模式：某个属性在多个实体中的出现规律"""
    property_name: str
    applicable_types: List[str]  # 该属性适用的实体类型
    observed_values: List[Any]  # 观察到的值
    frequency: int  # 出现频率


@dataclass
class AbstractRule:
    """抽象规则：从多个示例中归纳出的条件-结论规则"""
    condition: Dict[str, Any]
    conclusion: Dict[str, Any]
    support: int  # 支持该规则的示例数量
    confidence: float
    exceptions: List[str]  # 例外情况描述


class AbstractionEngine:
    """抽象引擎：从具体实例提取类别、属性模式和抽象规则"""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.categories: Dict[str, Category] = {}
        self.property_schemas: Dict[str, PropertySchema] = {}
        self.abstract_rules: List[AbstractRule] = []

    def extract_category(self, instance_ids: List[str]) -> Optional[Category]:
        """从一组实体提取共同属性作为类别定义。

        算法：
        1. 获取所有实体的属性字典
        2. 计算属性的交集（所有实体都有的属性及其相同值）
        3. 如果交集非空，创建 Category 并计算原型向量
        """
        if not instance_ids:
            return None

        # 收集有效实体
        entities = []
        for eid in instance_ids:
            entity = self.graph.get_entity(eid)
            if entity is not None:
                entities.append(entity)

        if len(entities) < 2:
            return None  # 至少需要 2 个实体才能形成类别

        # 计算共同属性交集
        common_props = self.find_common_properties(instance_ids)
        if not common_props:
            return None  # 没有共同属性，无法形成有意义的类别

        # 生成类别名称：基于共同属性的哈希
        sorted_keys = sorted(common_props.keys())
        category_name = 'cat_' + '_'.join(
            f'{k}={common_props[k]}' for k in sorted_keys
        )

        # 计算原型向量
        prototype = self.compute_prototype(instance_ids)

        # 置信度 = 共同属性数量 / 平均属性数量
        avg_prop_count = sum(len(e.properties) for e in entities) / len(entities)
        confidence = min(1.0, len(common_props) / max(avg_prop_count, 1.0))

        category = Category(
            name=category_name,
            members=list(instance_ids),
            defining_properties=common_props,
            prototype=prototype,
            confidence=confidence,
        )

        # 存入内部字典
        self.categories[category_name] = category
        return category

    def extract_property_pattern(self, entity_ids: List[str]) -> List[PropertySchema]:
        """统计每个属性在多少实体中出现，高频属性提取为 PropertySchema。

        算法：
        1. 遍历所有实体的所有属性
        2. 统计每个属性名出现的频率和观察到的值
        3. 记录具有该属性的实体类型
        4. 返回所有观察到的 PropertySchema 列表
        """
        # 统计数据结构：属性名 -> {值列表, 实体类型列表, 频率}
        prop_data: Dict[str, Dict[str, Any]] = {}

        for eid in entity_ids:
            entity = self.graph.get_entity(eid)
            if entity is None:
                continue

            for prop_name, prop_value in entity.properties.items():
                if prop_name not in prop_data:
                    prop_data[prop_name] = {
                        'values': [],
                        'types': [],
                        'frequency': 0,
                    }
                data = prop_data[prop_name]
                data['values'].append(prop_value)
                if entity.type not in data['types']:
                    data['types'].append(entity.type)
                data['frequency'] += 1

        # 构建 PropertySchema 列表
        schemas = []
        for prop_name, data in prop_data.items():
            schema = PropertySchema(
                property_name=prop_name,
                applicable_types=data['types'],
                observed_values=data['values'],
                frequency=data['frequency'],
            )
            schemas.append(schema)
            # 同时更新内部缓存
            self.property_schemas[prop_name] = schema

        return schemas

    def generalize_from_examples(self, examples: List[Dict]) -> Optional[AbstractRule]:
        """从多个示例中归纳出抽象规则。

        每个示例格式: {'conditions': {key: value, ...}, 'outcome': {key: value, ...}}
        算法：找 conditions 中所有示例都相同的不变部分作为规则条件，
        outcome 中所有示例都相同的部分作为结论。
        """
        if not examples:
            return None

        # 收集所有 conditions 和 outcomes
        all_conditions = [ex.get('conditions', {}) for ex in examples]
        all_outcomes = [ex.get('outcome', {}) for ex in examples]

        if not any(all_conditions) or not any(all_outcomes):
            return None

        # 找 conditions 中不变的部分（所有示例的交集）
        invariant_conditions = dict(all_conditions[0])
        for cond in all_conditions[1:]:
            keys_to_remove = []
            for key, value in invariant_conditions.items():
                if key not in cond or cond[key] != value:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del invariant_conditions[key]

        # 找 outcome 中不变的部分
        invariant_outcome = dict(all_outcomes[0])
        for outcome in all_outcomes[1:]:
            keys_to_remove = []
            for key, value in invariant_outcome.items():
                if key not in outcome or outcome[key] != value:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del invariant_outcome[key]

        if not invariant_conditions or not invariant_outcome:
            return None  # 没有不变的部分，无法形成规则

        # 收集例外：不符合规则的示例描述
        exceptions = []
        for i, ex in enumerate(examples):
            cond = ex.get('conditions', {})
            outcome = ex.get('outcome', {})
            # 检查是否匹配归纳出的规则
            cond_match = all(cond.get(k) == v for k, v in invariant_conditions.items())
            out_match = all(outcome.get(k) == v for k, v in invariant_outcome.items())
            if cond_match and not out_match:
                exceptions.append(f'示例{i}: {cond} -> {outcome}')

        # 置信度 = 符合规则的示例数 / 总示例数
        support = len(examples)
        violation_count = len(exceptions)
        confidence = (support - violation_count) / support if support > 0 else 0.0

        rule = AbstractRule(
            condition=invariant_conditions,
            conclusion=invariant_outcome,
            support=support,
            confidence=confidence,
            exceptions=exceptions,
        )
        self.abstract_rules.append(rule)
        return rule

    def get_category_for_entity(self, entity_id: str) -> Optional[str]:
        """检查实体属于哪个已知类别（属性匹配度最高的）。

        算法：遍历所有已知类别，计算实体属性与类别定义属性的重合度，
        返回匹配度最高的类别名称。
        """
        entity = self.graph.get_entity(entity_id)
        if entity is None:
            return None

        if not self.categories:
            return None

        best_category = None
        best_score = 0.0

        for cat_name, category in self.categories.items():
            # 计算属性匹配分数
            matched = 0
            total = len(category.defining_properties)
            if total == 0:
                continue

            for prop_key, prop_value in category.defining_properties.items():
                if entity.properties.get(prop_key) == prop_value:
                    matched += 1

            score = matched / total
            # 加权：类别置信度 * 属性匹配度
            weighted_score = score * category.confidence

            if weighted_score > best_score:
                best_score = weighted_score
                best_category = cat_name

        # 至少匹配 50% 的定义属性才算属于该类别
        if best_category is not None and best_score >= 0.5:
            return best_category
        return None

    def find_common_properties(self, entity_ids: List[str]) -> Dict[str, Any]:
        """找一组实体的共同属性（交集）。

        算法：取所有实体都拥有且值相同的属性键值对。
        """
        if not entity_ids:
            return {}

        # 收集有效实体
        entities = []
        for eid in entity_ids:
            entity = self.graph.get_entity(eid)
            if entity is not None:
                entities.append(entity)

        if not entities:
            return {}

        # 以第一个实体的属性为基准，逐步求交集
        common = dict(entities[0].properties)

        for entity in entities[1:]:
            keys_to_remove = []
            for key, value in common.items():
                if key not in entity.properties or entity.properties[key] != value:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del common[key]

        return common

    def compute_prototype(self, entity_ids: List[str]) -> Optional[torch.Tensor]:
        """计算一组实体的 embedding 均值向量作为原型。

        算法：收集所有有 embedding 的实体，取逐元素均值。
        """
        embeddings = []
        for eid in entity_ids:
            entity = self.graph.get_entity(eid)
            if entity is not None and entity.embedding is not None:
                embeddings.append(entity.embedding)

        if not embeddings:
            return None

        # 堆叠并取均值
        stacked = torch.stack(embeddings)
        prototype = stacked.mean(dim=0)
        return prototype
