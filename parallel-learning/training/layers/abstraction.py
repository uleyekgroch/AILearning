"""Layer 3: 概念抽象层

从具体实例中抽象出概念，建立概念层次。

核心能力：
1. 概念形成 — 从多个实例中提取共同特征
2. 层次抽象 — 具体→基本→抽象→元
3. 概念继承 — 子概念继承父概念的属性
4. 概念组合 — 组合简单概念形成复杂概念

运行方式：
    python training/layers/abstraction.py
"""

import re
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Concept:
    """概念 — 不是词条，是心智中的抽象表示"""
    name: str
    level: str = "concrete"  # concrete, basic, abstract, meta
    examples: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    confidence: float = 0.0
    formation_count: int = 0  # 被提及次数

    def add_example(self, example: str):
        """添加实例"""
        if example not in self.examples:
            self.examples.append(example)
            self.formation_count += 1

    def add_property(self, prop_name: str, prop_value: Any):
        """添加属性"""
        self.properties[prop_name] = prop_value

    def add_relation(self, relation: str, target: str):
        """添加关系"""
        if target not in self.relations[relation]:
            self.relations[relation].append(target)

    def inherit_from_parent(self, parent: 'Concept'):
        """从父概念继承属性"""
        for prop_name, prop_value in parent.properties.items():
            if prop_name not in self.properties:
                self.properties[prop_name] = prop_value

        for relation, targets in parent.relations.items():
            for target in targets:
                if target not in self.relations[relation]:
                    self.relations[relation].append(target)


@dataclass
class AbstractionRule:
    """抽象规则 — 如何从具体到抽象"""
    pattern: str  # 匹配模式
    abstract_level: str  # 抽象级别
    category: str  # 类别
    properties: Dict[str, Any] = field(default_factory=dict)  # 继承的属性


class ConceptAbstraction:
    """概念抽象层

    核心能力：
    - 从实例中形成概念
    - 建立概念层次
    - 属性继承
    """

    def __init__(self):
        # 概念库
        self.concepts: Dict[str, Concept] = {}

        # 抽象规则
        self.abstraction_rules = self._load_abstraction_rules()

        # 概念层次
        self.hierarchy = {
            'meta': set(),      # 元概念：存在、时间、空间
            'abstract': set(),  # 抽象概念：动物、植物、物质
            'basic': set(),     # 基本概念：猫、狗、水
            'concrete': set(),  # 具体概念：这只猫、那杯水
        }

        # 统计
        self.stats = {
            'total_concepts': 0,
            'abstractions_made': 0,
            'inheritances_applied': 0,
        }

    def _load_abstraction_rules(self) -> List[AbstractionRule]:
        """加载抽象规则"""
        return [
            # 生物分类（只匹配完整的动物名，不匹配人名）
            AbstractionRule(pattern='^猫$|^狗$|^猪$|^牛$|^羊$|^马$', abstract_level='abstract',
                          category='哺乳动物', properties={'有毛': True, '胎生': True, '恒温': True}),
            AbstractionRule(pattern='^鸟$|^鸡$|^鸭$|^鹅$|^鹰$', abstract_level='abstract',
                          category='鸟类', properties={'有羽毛': True, '卵生': True, '会飞': True}),
            AbstractionRule(pattern='^鱼$|^鲨$|^鲸$|^豚$', abstract_level='abstract',
                          category='水生动物', properties={'有鳞': True, '卵生': True, '水生': True}),
            AbstractionRule(pattern='^哺乳动物$|^鸟类$|^水生动物$|^爬行动物$', abstract_level='meta',
                          category='动物', properties={'有生命': True, '能运动': True}),

            # 物质分类
            AbstractionRule(pattern='^水$|^油$|^酒$|^奶$', abstract_level='abstract',
                          category='液体', properties={'液态': True, '可流动': True}),
            AbstractionRule(pattern='^铁$|^铜$|^金$|^银$|^铝$', abstract_level='abstract',
                          category='金属', properties={'导电': True, '有光泽': True, '延展': True}),
            AbstractionRule(pattern='^木$|^纸$|^布$|^棉$', abstract_level='abstract',
                          category='有机材料', properties={'可燃': True, '来自生物': True}),

            # 语言分类
            AbstractionRule(pattern='^Python$|^Java$|^C\+\+$|^JavaScript$', abstract_level='abstract',
                          category='编程语言', properties={'可编程': True, '有语法': True}),
            AbstractionRule(pattern='^中文$|^英文$|^日文$|^法文$', abstract_level='abstract',
                          category='自然语言', properties={'可交流': True, '自然演化': True}),

            # 学科分类
            AbstractionRule(pattern='^物理$|^化学$|^生物$|^数学$', abstract_level='abstract',
                          category='自然科学', properties={'研究自然': True, '可实验': True}),
            AbstractionRule(pattern='^历史$|^文学$|^哲学$|^艺术$', abstract_level='abstract',
                          category='人文科学', properties={'研究人类': True, '可诠释': True}),

            # 地理分类
            AbstractionRule(pattern='^中国$|^美国$|^日本$|^英国$', abstract_level='abstract',
                          category='国家', properties={'有主权': True, '有领土': True}),
            AbstractionRule(pattern='^北京$|^上海$|^东京$|^纽约$', abstract_level='abstract',
                          category='城市', properties={'有城市化': True, '有人口': True}),
        ]

    def form_concept(self, name: str, context: str = "", level: str = "concrete") -> Concept:
        """形成概念"""
        if name not in self.concepts:
            self.concepts[name] = Concept(name=name, level=level)
            self.hierarchy[level].add(name)
            self.stats['total_concepts'] += 1

        concept = self.concepts[name]
        concept.add_example(context)
        concept.confidence = min(1.0, concept.formation_count * 0.1)

        # 尝试应用抽象规则
        self._apply_abstraction_rules(name, concept)

        return concept

    def _apply_abstraction_rules(self, name: str, concept: Concept):
        """应用抽象规则"""
        for rule in self.abstraction_rules:
            if re.search(rule.pattern, name):
                # 创建或获取抽象概念
                abstract_name = rule.category
                if abstract_name not in self.concepts:
                    self.form_concept(abstract_name, level=rule.abstract_level)

                abstract_concept = self.concepts[abstract_name]

                # 建立层次关系
                if concept.parent is None:
                    concept.parent = abstract_name
                    abstract_concept.children.append(name)

                    # 继承属性
                    for prop_name, prop_value in rule.properties.items():
                        concept.add_property(prop_name, prop_value)

                    self.stats['abstractions_made'] += 1

                break

    def abstract_up(self, concept_name: str) -> List[str]:
        """向上抽象 — 从具体到抽象"""
        path = []
        current = concept_name

        while current and current in self.concepts:
            path.append(current)
            current = self.concepts[current].parent

        return path

    def specialize_down(self, concept_name: str) -> List[str]:
        """向下特化 — 从抽象到具体"""
        if concept_name not in self.concepts:
            return []

        result = []
        queue = [concept_name]

        while queue:
            current = queue.pop(0)
            result.append(current)

            if current in self.concepts:
                queue.extend(self.concepts[current].children)

        return result

    def find_common_ancestor(self, concept1: str, concept2: str) -> Optional[str]:
        """找到两个概念的共同祖先"""
        path1 = set(self.abstract_up(concept1))
        path2 = set(self.abstract_up(concept2))

        common = path1 & path2
        if common:
            # 返回最近的共同祖先
            for name in self.abstract_up(concept1):
                if name in common:
                    return name

        return None

    def get_inherited_properties(self, concept_name: str) -> Dict[str, Any]:
        """获取概念的所有属性（包括继承的）"""
        if concept_name not in self.concepts:
            return {}

        properties = {}
        current = concept_name

        while current and current in self.concepts:
            concept = self.concepts[current]
            # 添加当前概念的属性（不覆盖已有的）
            for prop_name, prop_value in concept.properties.items():
                if prop_name not in properties:
                    properties[prop_name] = prop_value
                    self.stats['inheritances_applied'] += 1

            current = concept.parent

        return properties

    def learn_from_text(self, text: str):
        """从文本中学习概念"""
        # 提取实体
        entities = self._extract_entities(text)

        # 形成概念
        for entity in entities:
            self.form_concept(entity, context=text[:100])

    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        # 分隔符
        separators = r'[，。！？；：、\s的了是在有位于属于包括使用产生导致引起为了因为所以如果那么但是而且或者而但]'

        parts = re.split(separators, text)
        entities = []

        for part in parts:
            part = part.strip()
            # 中文实体
            zh_matches = re.findall(r'([一-鿿]{2,6})', part)
            for entity in zh_matches:
                if self._validate_entity(entity):
                    entities.append(entity)

            # 英文实体
            en_matches = re.findall(r'([A-Z][a-zA-Z]+)', part)
            for entity in en_matches:
                if self._validate_entity(entity):
                    entities.append(entity)

        return list(set(entities))

    def _validate_entity(self, text: str) -> bool:
        """验证是否是有效的实体"""
        if len(text) < 2:
            return False

        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样可以能够应该已经正在将要')
        if text in stopwords:
            return False

        return True

    def query(self, question: str) -> Dict:
        """查询概念知识"""
        entities = self._extract_entities(question)

        results = {
            'concepts': [],
            'hierarchies': [],
            'common_ancestors': [],
        }

        for entity in entities:
            if entity in self.concepts:
                concept = self.concepts[entity]

                # 获取概念信息
                results['concepts'].append({
                    'name': entity,
                    'level': concept.level,
                    'parent': concept.parent,
                    'children': concept.children[:5],
                    'properties': self.get_inherited_properties(entity),
                    'examples': concept.examples[:3],
                })

                # 获取抽象层次
                hierarchy = self.abstract_up(entity)
                results['hierarchies'].append(hierarchy)

        # 如果有多个实体，找共同祖先
        if len(entities) >= 2:
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    if e1 in self.concepts and e2 in self.concepts:
                        ancestor = self.find_common_ancestor(e1, e2)
                        if ancestor:
                            results['common_ancestors'].append({
                                'concepts': [e1, e2],
                                'ancestor': ancestor,
                            })

        return results

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'hierarchy_sizes': {k: len(v) for k, v in self.hierarchy.items()},
        }


def test_concept_abstraction():
    """测试概念抽象"""
    print("=" * 70)
    print("概念抽象层测试")
    print("=" * 70)

    abstraction = ConceptAbstraction()

    # 测试文本
    test_texts = [
        "猫是一种哺乳动物。",
        "狗是人类的朋友。",
        "Python是一种编程语言。",
        "太阳是太阳系的中心。",
        "牛顿是物理学家。",
    ]

    with open('abstraction_test.txt', 'w', encoding='utf-8') as f:
        f.write('概念抽象层测试\n')
        f.write('=' * 70 + '\n')

        for text in test_texts:
            f.write(f'\n输入: {text}\n')
            abstraction.learn_from_text(text)

        # 显示概念层次
        f.write('\n' + '=' * 70 + '\n')
        f.write('概念层次\n')
        f.write('=' * 70 + '\n')

        for level, concepts in abstraction.hierarchy.items():
            if concepts:
                f.write(f'\n{level}:\n')
                for concept_name in concepts:
                    concept = abstraction.concepts[concept_name]
                    f.write(f'  {concept_name}')
                    if concept.parent:
                        f.write(f' (父: {concept.parent})')
                    if concept.children:
                        f.write(f' (子: {", ".join(concept.children[:3])})')
                    f.write('\n')

        # 测试抽象
        f.write('\n' + '=' * 70 + '\n')
        f.write('抽象测试\n')
        f.write('=' * 70 + '\n')

        test_concepts = ['猫', 'Python', '太阳']
        for concept_name in test_concepts:
            f.write(f'\n{concept_name} 的抽象层次:\n')
            hierarchy = abstraction.abstract_up(concept_name)
            f.write(f'  {" → ".join(hierarchy)}\n')

            f.write(f'  继承的属性:\n')
            properties = abstraction.get_inherited_properties(concept_name)
            for prop_name, prop_value in properties.items():
                f.write(f'    {prop_name}: {prop_value}\n')

        # 测试共同祖先
        f.write('\n' + '=' * 70 + '\n')
        f.write('共同祖先测试\n')
        f.write('=' * 70 + '\n')

        pairs = [('猫', '狗'), ('Python', 'Java')]
        for c1, c2 in pairs:
            ancestor = abstraction.find_common_ancestor(c1, c2)
            if ancestor:
                f.write(f'\n{c1} 和 {c2} 的共同祖先: {ancestor}\n')
            else:
                f.write(f'\n{c1} 和 {c2} 没有找到共同祖先\n')

        # 统计
        f.write('\n' + '=' * 70 + '\n')
        f.write('统计\n')
        f.write('=' * 70 + '\n')
        stats = abstraction.get_stats()
        for k, v in stats.items():
            f.write(f'  {k}: {v}\n')

    print('Written to abstraction_test.txt')


if __name__ == '__main__':
    test_concept_abstraction()
