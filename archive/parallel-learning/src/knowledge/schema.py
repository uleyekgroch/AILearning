"""
常识知识库数据模型 - Production Commonsense Knowledge Data Model

基于2024-2025年最新研究设计：
- ConceptNet的知识表示
- ATOMIC的事件模型
- PrimeNet的概念原型
- 可扩展的Schema设计

核心设计：
1. 实体类型系统（Type Hierarchy）
2. 关系类型系统（Relation Taxonomy）
3. 本体设计（Ontology）
4. 元数据模型（Metadata）
"""

from enum import Enum
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json


# ============================================================================
# Triple 类定义
# ============================================================================

@dataclass
class Triple:
    """三元组"""
    subject: str                                    # 主语
    relation: str                                   # 关系
    object: str                                     # 宾语
    confidence: float = 1.0                     # 置信度
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> Tuple[str, str, str]:
        """转换为元组"""
        return (self.subject, self.relation, self.object)

    def __hash__(self):
        return hash((self.subject, self.relation, self.object))

    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return (self.subject == other.subject and
                self.relation == other.relation and
                self.object == other.object)


# ============================================================================
# 1. 实体类型系统 (Entity Type System)
# ============================================================================

class EntityType(Enum):
    """核心实体类型（基于ConceptNet和PrimeNet）"""
    # 物理实体
    PHYSICAL_OBJECT = "PhysicalObject"      # 物理对象
    LOCATION = "Location"                   # 地点
    ARTIFACT = "Artifact"                   # 人工物

    # 生物实体
    PERSON = "Person"                       # 人物
    ORGANISM = "Organism"                  # 生物体
    ANIMAL = "Animal"                      # 动物
    PLANT = "Plant"                        # 植物
    BODY_PART = "BodyPart"                 # 身体部位

    # 抽象实体
    CONCEPT = "Concept"                     # 概念
    EMOTION = "Emotion"                     # 情绪
    IDEA = "Idea"                          # 想法
    EVENT = "Event"                         # 事件
    ACTIVITY = "Activity"                 # 活动

    # 时间实体
    TIME = "Time"                          # 时间
    DATE = "Date"                          # 日期
    DURATION = "Duration"                  # 时长

    # 数量实体
    QUANTITY = "Quantity"                  # 数量
    MEASURE = "Measure"                    # 度量
    NUMBER = "Number"                      # 数字


class EntitySubType(Enum):
    """实体子类型（细化分类）"""
    # 物理对象子类型
    FOOD = "Food"                          # 食物
    TOOL = "Tool"                          # 工具
    FURNITURE = "Furniture"                # 家具
    CLOTHING = "Clothing"                  # 衣物
    VEHICLE = "Vehicle"                    # 车辆

    # 人物子类型
    PROFESSION = "Profession"            # 职业
    ROLE = "Role"                         # 角色
    FICTIONAL = "Fictional"              # 虚构人物

    # 动物子类型
    DOMESTIC = "Domestic"                 # 驯养动物
    WILD = "Wild"                         # 野生动物
    INSECT = "Insect"                     # 昆虫

    # 植物子类型
    TREE = "Tree"                         # 树木
    FLOWER = "Flower"                     # 花卉
    VEGETABLE = "Vegetable"               # 蔬菜
    FRUIT = "Fruit"                       # 水果


# ============================================================================
# 2. 关系类型系统 (Relation Type System)
# ============================================================================

class RelationType(Enum):
    """核心关系类型（基于ConceptNet）"""
    # 属性关系
    PROPERTY = "Property"                  # 属性
    HAS_PROPERTY = "HasProperty"          # 具有属性
    CAPABLE_OF = "CapableOf"              # 能够

    # 空间关系
    AT_LOCATION = "AtLocation"             # 位于
    NEAR = "Near"                         # 靠近
    PART_OF = "PartOf"                    # 组成部分
    CONTAINS = "Contains"                  # 包含

    # 时间关系
    BEFORE = "Before"                     # 在...之前
    AFTER = "After"                       # 在...之后
    DURING = "During"                     # 在...期间

    # 因果关系
    CAUSE = "Cause"                       # 导致
    EFFECT = "Effect"                     # 结果
    ENABLES = "Enables"                   # 使能
    PREVENTS = "Prevents"                  # 阻止

    # 功能关系
    USED_FOR = "UsedFor"                   # 用于
    FUNCTION = "Function"                 # 功能
    PURPOSE = "Purpose"                   # 目的

    # 社会关系
    RELATED_TO = "RelatedTo"              # 相关
    SIMILAR_TO = "SimilarTo"              # 相似
    OPPOSITE_OF = "OppositeOf"             # 相反

    # 语义关系
    IS_A = "IsA"                          # 是一种
    INSTANCE_OF = "InstanceOf"            # 是...的实例
    DEFINED_AS = "DefinedAs"               # 定义为

    # 情感关系
    DESIRES = "Desires"                    # 渴望
    FEARS = "Fears"                       # 害怕
    LIKES = "Likes"                       # 喜欢
    DISLIKES = "Dislikes"                 # 不喜欢


# ATOMIC事件关系（基于ATOMIC数据集）
class AtomicRelationType(Enum):
    """ATOMIC事件关系类型"""
    WANT = "HasProperty"                   # 想要
    HATE = "HasProperty"                    # 讨厌
    CAUSE = "Causes"                      # 导致
    DESIRE = "Desires"                     # 渴望
    REACT = "Reacts"                      # 反应
    CREATED_BY = "CreatedBy"               # 由...创建
    IS = "Is"                             # 是


# ============================================================================
# 3. 本体设计 (Ontology)
# ============================================================================

class Ontology:
    """
    本体 - 定义知识结构

    功能：
    - 类型继承层次
    - 约束定义
    - 推理规则
    """

    def __init__(self):
        # 类型继承层次
        self.type_hierarchy: Dict[str, Set[str]] = defaultdict(set)
        self._init_type_hierarchy()

        # 关系约束
        self.relation_constraints: Dict[str, RelationConstraint] = {}
        self._init_relation_constraints()

        # 推理规则
        self.inference_rules: List[InferenceRule] = []
        self._init_inference_rules()

    def _init_type_hierarchy(self):
        """初始化类型层次"""
        # === 基础类型层次 ===

        # 生物体层次（生物体继承自物理对象）
        self.type_hierarchy["Organism"] = {"PhysicalObject"}

        # 具体生物类型
        self.type_hierarchy["Person"] = {"Organism"}  # 人物是生物体（通过继承获得PhysicalObject）
        self.type_hierarchy["Animal"] = {"Organism"}   # 动物是生物体
        self.type_hierarchy["Plant"] = {"Organism"}     # 植物是生物体
        self.type_hierarchy["BodyPart"] = {"PhysicalObject"}  # 身体部位是物理对象

        # 对象类型
        self.type_hierarchy["Artifact"] = {"PhysicalObject"}  # 人工物是物理对象
        self.type_hierarchy["Location"] = {"PhysicalObject"}   # 地点是物理对象
        self.type_hierarchy["Food"] = {"PhysicalObject"}       # 食物是物理对象
        self.type_hierarchy["Tool"] = {"PhysicalObject", "Artifact"}  # 工具既是物理对象也是人工物

        # 抽象实体层次
        self.type_hierarchy["Event"] = {"Concept"}      # 事件是一种概念
        self.type_hierarchy["Activity"] = {"Event"}     # 活动是一种事件
        self.type_hierarchy["Emotion"] = {"Concept"}    # 情绪是一种概念
        self.type_hierarchy["Idea"] = {"Concept"}        # 想法是一种概念

        # 时间实体层次
        self.type_hierarchy["Date"] = {"Time"}           # 日期是一种时间
        self.type_hierarchy["Duration"] = {"Time"}       # 时长是一种时间

        # 数量实体层次
        self.type_hierarchy["Measure"] = {"Quantity"}    # 度量是一种数量
        self.type_hierarchy["Number"] = {"Quantity"}    # 数字是一种数量

    def _init_relation_constraints(self):
        """初始化关系约束"""
        # AtLocation约束：主语必须是物理对象
        self.relation_constraints[RelationType.AT_LOCATION] = RelationConstraint(
            subject_types={EntityType.PHYSICAL_OBJECT, EntityType.PERSON, EntityType.ANIMAL, EntityType.PLANT},
            object_types={EntityType.LOCATION, EntityType.PHYSICAL_OBJECT}
        )

        # PartOf约束：整体和部分必须是同类
        self.relation_constraints[RelationType.PART_OF] = RelationConstraint(
            subject_types=None,  # 允许任意
            object_types=None,  # 允许任意
            transitive=True  # 传递性
        )

        # Cause约束：因果关系
        self.relation_constraints[RelationType.CAUSE] = RelationConstraint(
            subject_types=None,
            object_types={EntityType.EVENT, EntityType.ACTIVITY},
            transitive=False
        )

    def _init_inference_rules(self):
        """初始化推理规则"""
        # 传递性规则
        self.inference_rules.append(InferenceRule(
            name="part_of_transitivity",
            relation=RelationType.PART_OF,
            rule_type=InferenceRuleType.TRANSITIVE
        ))

        # 对称性规则
        self.inference_rules.append(InferenceRule(
            name="near_symmetry",
            relation=RelationType.NEAR,
            rule_type=InferenceRuleType.SYMMETRIC
        ))

        # 反对关系
        self.inference_rules.append(InferenceRule(
            name="opposite_asymmetry",
            relation=RelationType.OPPOSITE_OF,
            rule_type=InferenceRuleType.ASYMMETRIC
        ))

    def is_valid_triple(self, subject: str, relation: str, obj: str,
                        subject_type: EntityType, object_type: EntityType) -> bool:
        """验证三元组是否有效"""
        # 检查关系约束
        if relation in self.relation_constraints:
            constraint = self.relation_constraints[relation]
            return constraint.is_valid(subject_type, object_type)
        return True

    def get_inherited_types(self, entity_type: EntityType) -> Set[EntityType]:
        """获取继承的类型"""
        inherited = {entity_type}
        # 获取枚举的value作为类型层次中查找的键
        type_name = entity_type.value if isinstance(entity_type, EntityType) else entity_type

        # 递归获取父类型
        for parent in self.type_hierarchy.get(type_name, set()):
            # 通过枚举的value来查找枚举成员
            parent_type = None
            for et in EntityType:
                if et.value == parent:
                    parent_type = et
                    break

            if parent_type:
                inherited.update(self.get_inherited_types(parent_type))

        return inherited

    def infer_types(self, entity: str, relations: List[Tuple[str, str]]) -> Set[EntityType]:
        """基于关系推断实体类型"""
        possible_types = set(EntityType)

        # 基于关系约束缩小类型范围
        for rel, obj in relations:
            for rel_type, constraint in self.relation_constraints.items():
                if rel == rel_type.name:
                    # 移除不符合约束的类型
                    possible_types &= constraint.subject_types

        return possible_types


# ============================================================================
# 4. 约束和规则数据类
# ============================================================================

class InferenceRuleType(Enum):
    """推理规则类型"""
    TRANSITIVE = "transitive"         # 传递性 (A->B, B->C => A->C)
    SYMMETRIC = "symmetric"           # 对称性 (A->B => B->A)
    ASYMMETRIC = "asymmetric"         # 非对称 (A->B !=> B->A)
    INVERSE = "inverse"               # 逆关系 (A->B => B<-A)


@dataclass
class RelationConstraint:
    """关系约束"""
    subject_types: Optional[Set[EntityType]] = None  # 主语允许的类型
    object_types: Optional[Set[EntityType]] = None   # 宾语允许的类型
    transitive: bool = False                      # 是否传递性
    symmetric: bool = False                      # 是否对称性


@dataclass
class InferenceRule:
    """推理规则"""
    name: str
    relation: RelationType
    rule_type: InferenceRuleType
    confidence: float = 1.0


# ============================================================================
# 5. 实体和关系数据类
# ============================================================================

@dataclass
class Entity:
    """实体"""
    id: str                                          # 唯一标识
    text: str                                        # 文本表示
    entity_type: EntityType                          # 实体类型
    sub_type: Optional[EntitySubType] = None         # 子类型
    properties: Dict[str, Any] = field(default_factory=dict)  # 属性
    confidence: float = 1.0                         # 置信度
    sources: List[str] = field(default_factory=list)     # 来源

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'id': self.id,
            'text': self.text,
            'entity_type': self.entity_type.value,
            'sub_type': self.sub_type.value if self.sub_type else None,
            'properties': self.properties,
            'confidence': self.confidence,
            'sources': self.sources
        }


@dataclass
class Relation:
    """关系"""
    id: str                                          # 唯一标识
    relation_type: RelationType                     # 关系类型
    subject: str                                     # 主语
    object: str                                      # 宾语
    properties: Dict[str, Any] = field(default_factory=dict)  # 属性
    confidence: float = 1.0                         # 置信度
    sources: List[str] = field(default_factory=list)     # 来源

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'id': self.id,
            'relation_type': self.relation_type.value,
            'subject': self.subject,
            'object': self.object,
            'properties': self.properties,
            'confidence': self.confidence,
            'sources': self.sources
        }


# ============================================================================
# 6. 知识条目（Knowledge Entry）
# ============================================================================

@dataclass
class KnowledgeEntry:
    """知识条目 - 基本知识单元"""
    id: str
    content: str                                     # 内容描述
    entry_type: str                                  # 类型（fact/rule/concept）
    entities: List[str] = field(default_factory=list)      # 涉及实体
    relations: List[Relation] = field(default_factory=list)  # 涉及关系
    confidence: float = 1.0                         # 置信度
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def to_triples(self) -> List[Triple]:
        """转换为三元组列表"""
        triples = []
        for rel in self.relations:
            triples.append(Triple(
                subject=rel.subject,
                relation=rel.relation_type.value,
                object=rel.object,
                confidence=rel.confidence,
                metadata=rel.properties
            ))
        return triples

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'id': self.id,
            'content': self.content,
            'entry_type': self.entry_type,
            'entities': self.entities,
            'relations': [r.to_dict() for r in self.relations],
            'confidence': self.confidence,
            'metadata': self.metadata
        }


# ============================================================================
# 7. Schema定义和验证
# ============================================================================

class CommonsenseSchema:
    """
    常识知识库Schema

    定义：
    - 数据格式规范
    - 验证规则
    - 一致性约束
    """

    def __init__(self):
        self.ontology = Ontology()
        self.validators = []

        # 注册验证器
        self._init_validators()

    def _init_validators(self):
        """初始化验证器"""
        # 实体验证器
        self.validators.append(EntityValidator(self.ontology))

        # 关系验证器
        self.validators.append(RelationValidator(self.ontology))

        # 三元组验证器
        self.validators.append(TripleValidator(self.ontology))

    def validate_entity(self, entity: Entity) -> Tuple[bool, List[str]]:
        """验证实体

        Returns:
            (is_valid, error_messages)
        """
        errors = []
        for validator in self.validators:
            if hasattr(validator, 'validate_entity'):
                valid, errs = validator.validate_entity(entity)
                if not valid:
                    errors.extend(errs)

        return (len(errors) == 0, errors)

    def validate_relation(self, relation: Relation) -> Tuple[bool, List[str]]:
        """验证关系"""
        errors = []
        for validator in self.validators:
            if hasattr(validator, 'validate_relation'):
                valid, errs = validator.validate_relation(relation)
                if not valid:
                    errors.extend(errs)

        return (len(errors) == 0, errors)

    def validate_triple(self, subject: str, relation: str,
                        obj: str, subject_type: EntityType,
                        object_type: EntityType) -> Tuple[bool, List[str]]:
        """验证三元组"""
        errors = []
        for validator in self.validators:
            if hasattr(validator, 'validate_triple'):
                valid, errs = validator.validate_triple(
                    subject, relation, obj, subject_type, object_type
                )
                if not valid:
                    errors.extend(errs)

        # 本体验证
        valid = self.ontology.is_valid_triple(
            subject, relation, obj, subject_type, object_type
        )
        if not valid:
            errors.append(f"本体验证失败: {subject} {relation} {obj}")

        return (len(errors) == 0, errors)


# ============================================================================
# 8. 验证器
# ============================================================================

class EntityValidator:
    """实体验证器"""

    def __init__(self, ontology: Ontology):
        self.ontology = ontology

    def validate_entity(self, entity: Entity) -> Tuple[bool, List[str]]:
        """验证实体"""
        errors = []

        # 检查必需字段
        if not entity.id:
            errors.append("实体ID不能为空")
        if not entity.text:
            errors.append("实体文本不能为空")

        # 检查类型有效性
        if isinstance(entity.entity_type, str):
            try:
                entity.entity_type = EntityType[entity.entity_type]
            except ValueError:
                errors.append(f"无效的实体类型: {entity.entity_type}")

        return (len(errors) == 0, errors)


class RelationValidator:
    """关系验证器"""

    def __init__(self, ontology: Ontology):
        self.ontology = ontology

    def validate_relation(self, relation: Relation) -> Tuple[bool, List[str]]:
        """验证关系"""
        errors = []

        # 检查必需字段
        if not relation.id:
            errors.append("关系ID不能为空")
        if not relation.subject or not relation.object:
            errors.append("主语和宾语不能为空")

        # 检查关系类型有效性
        if isinstance(relation.relation_type, str):
            try:
                relation.relation_type = RelationType[relation.relation_type]
            except ValueError:
                errors.append(f"无效的关系类型: {relation.relation_type}")

        return (len(errors) == 0, errors)


class TripleValidator:
    """三元组验证器"""

    def __init__(self, ontology: Ontology):
        self.ontology = ontology

    def validate_triple(self, subject: str, relation: str,
                        obj: str, subject_type: EntityType,
                        object_type: EntityType) -> Tuple[bool, List[str]]:
        """验证三元组"""
        errors = []

        # 基本检查
        if not subject or not relation or not obj:
            errors.append("三元组的任一部分不能为空")
            return (False, errors)

        # 本体验证
        if not self.ontology.is_valid_triple(
            subject, relation, obj, subject_type, object_type
        ):
            errors.append(f"关系约束验证失败: {subject} {relation} {obj}")

        return (len(errors) == 0, errors)


# ============================================================================
# 10. 导出和导入
# ============================================================================

class KnowledgeSchemaExporter:
    """知识Schema导出器"""

    @staticmethod
    def export_schema(schema: CommonsenseSchema) -> Dict:
        """导出Schema为JSON格式"""
        return {
            'entity_types': [t.value for t in EntityType],
            'entity_sub_types': [t.value for t in EntitySubType],
            'relation_types': [t.value for t in RelationType],
            'atomic_relations': [t.value for t in AtomicRelationType],
            'type_hierarchy': dict(schema.ontology.type_hierarchy),
            'relation_constraints': {
                name: {
                    'subject_types': [t.value for t in c.subject_types] if c.subject_types else [],
                    'object_types': [t.value for t in c.object_types] if c.object_types else [],
                    'transitive': c.transitive,
                    'symmetric': c.symmetric
                }
                for name, c in schema.ontology.relation_constraints.items()
            },
            'inference_rules': [
                {
                    'name': rule.name,
                    'relation': rule.relation.value,
                    'rule_type': rule.rule_type.value,
                    'confidence': rule.confidence
                }
                for rule in schema.ontology.inference_rules
            ]
        }


if __name__ == '__main__':
    print("=== 常识知识库数据模型 ===")
    print()
    print("核心组件:")
    print("- 实体类型系统: 15+核心类型")
    print("- 关系类型系统: 20+核心关系")
    print("- 本体设计: 继承层次、约束、规则")
    print("- Schema验证: 多层验证机制")
    print()
    print("支持的操作:")
    print("- 实体/关系验证")
    print("- 类型推断")
    print("- 约束检查")
    print("- 规则推理")
