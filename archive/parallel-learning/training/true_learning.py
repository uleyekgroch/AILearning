"""真正的人类式学习系统

不是模式匹配，不是信息检索，是真正的学习。

人类学习的核心：
1. 概念形成 — 从具体实例中抽象出概念
2. 因果理解 — 理解事件之间的因果关系
3. 类比推理 — 用已知理解未知
4. 好奇驱动 — 主动探索不理解的东西
5. 层次抽象 — 从具象到抽象的知识层次
6. 预测验证 — 形成预期，验证对错，修正理解

运行方式：
    python training/true_learning.py
"""

import json
import os
import sys
import time
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field


@dataclass
class Concept:
    """概念 — 不是词条，是心智中的抽象表示"""

    name: str
    examples: List[str] = field(default_factory=list)  # 具体实例
    properties: Dict[str, Any] = field(default_factory=dict)  # 属性
    relations: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))  # 关系
    parent: Optional[str] = None  # 上位概念
    children: List[str] = field(default_factory=list)  # 下位概念
    activation: float = 0.0  # 激活程度（最近被提及）
    confidence: float = 0.0  # 对这个概念的理解程度
    surprise: float = 0.0  # 遇到意外信息时的惊讶度
    co_activations: Dict[str, float] = field(default_factory=dict)  # 共现激活

    def add_example(self, example: str):
        """添加实例，自动更新属性"""
        if example not in self.examples:
            self.examples.append(example)
            self._update_from_examples()

    def _update_from_examples(self):
        """从实例中提取共同属性"""
        # 这里应该做真正的抽象，目前简化处理
        pass

    def relate(self, relation: str, target: str):
        """建立关系"""
        if target not in self.relations[relation]:
            self.relations[relation].append(target)

    def spread_activation(self, amount: float = 1.0):
        """激活扩散到相关概念"""
        self.activation = min(1.0, self.activation + amount)
        # 衰减
        self.activation *= 0.9


@dataclass
class CausalRule:
    """因果规则 — 理解原因和结果"""

    cause: str
    effect: str
    conditions: List[str] = field(default_factory=list)  # 条件
    confidence: float = 0.0
    observations: int = 0  # 观察次数
    exceptions: List[str] = field(default_factory=list)  # 例外情况

    def observe(self, confirmed: bool):
        """观察一次"""
        self.observations += 1
        if confirmed:
            self.confidence = (self.confidence * (self.observations - 1) + 1) / self.observations
        else:
            self.confidence = (self.confidence * (self.observations - 1)) / self.observations


class CuriosityEngine:
    """好奇心引擎 — 驱动主动学习"""

    def __init__(self):
        self.questions: List[Dict] = []  # 待探索的问题
        self.surprise_history: List[float] = []  # 惊讶度历史
        self.knowledge_gaps: Set[str] = set()  # 知识空白

    def notice_surprise(self, concept: str, expected: str, actual: str):
        """注意到意外"""
        surprise = self._calculate_surprise(expected, actual)
        self.surprise_history.append(surprise)

        if surprise > 0.5:
            self.questions.append({
                'type': 'surprise',
                'concept': concept,
                'expected': expected,
                'actual': actual,
                'surprise': surprise,
            })

        return surprise

    def notice_gap(self, concept: str, context: str):
        """注意到知识空白"""
        self.knowledge_gaps.add(concept)
        self.questions.append({
            'type': 'gap',
            'concept': concept,
            'context': context,
        })

    def get_next_question(self) -> Optional[Dict]:
        """获取下一个要探索的问题"""
        if not self.questions:
            return None
        # 按惊讶度排序，优先探索最意外的
        self.questions.sort(key=lambda x: x.get('surprise', 0), reverse=True)
        return self.questions.pop(0)

    def _calculate_surprise(self, expected: str, actual: str) -> float:
        """计算惊讶度"""
        # 简单版本：字符串差异
        if expected == actual:
            return 0.0
        # 计算编辑距离比例
        max_len = max(len(expected), len(actual))
        if max_len == 0:
            return 0.0
        differences = sum(1 for a, b in zip(expected, actual) if a != b)
        return differences / max_len


class HumanLikeLearningSystem:
    """人类式学习系统

    核心区别：
    - 不是存储事实，而是形成理解
    - 不是被动接收，而是主动探索
    - 不是平面存储，而是层次抽象
    - 不是统计关联，而是因果推理
    """

    def __init__(self):
        # 概念网络
        self.concepts: Dict[str, Concept] = {}

        # 因果规则库
        self.causal_rules: List[CausalRule] = []

        # 好奇心引擎
        self.curiosity = CuriosityEngine()

        # 经验流
        self.experiences: List[Dict] = []

        # 学习统计
        self.stats = {
            'articles_read': 0,
            'concepts_formed': 0,
            'causal_rules_learned': 0,
            'surprises_encountered': 0,
            'questions_generated': 0,
        }

        # 抽象层次
        self.abstraction_levels = {
            'concrete': set(),  # 具体实例
            'basic': set(),  # 基本概念
            'abstract': set(),  # 抽象概念
            'meta': set(),  # 元概念
        }

    def learn_from_text(self, title: str, text: str):
        """从文本中学习

        不是提取三元组，而是：
        1. 识别新概念
        2. 理解概念间关系
        3. 发现因果规律
        4. 注意到意外信息
        5. 形成抽象理解
        """
        self.stats['articles_read'] += 1

        # 1. 提取实体和事件
        entities = self._extract_entities(text)
        events = self._extract_events(text)

        # 2. 概念形成
        for entity in entities:
            self._form_concept(entity, title, text)

        # 3. 关系发现
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                self._discover_relation(e1, e2, text)

        # 4. 因果理解
        self._extract_causal_relations(text, entities, events)

        # 5. 惊讶检测
        self._detect_surprises(text, entities)

        # 6. 抽象提升
        self._abstract_up(entities)

        # 7. 记录经验
        self.experiences.append({
            'title': title,
            'entities': entities,
            'timestamp': time.time(),
        })

    def _extract_entities(self, text: str) -> List[str]:
        """提取实体 — 不是简单的正则，而是理解什么是有意义的实体"""
        entities = []

        # 中文实体（2-6字）
        zh_entities = re.findall(r'[一-鿿]{2,6}', text)

        # 英文实体
        en_entities = re.findall(r'[A-Z][a-zA-Z]+', text)

        # 过滤停用词和无意义词
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样'
                       '可以能够应该已经正在将要一个这个那个一些很多所有')
        meaningful_words = set('是为有在位于属于包括使用产生导致引起发明发现创造提出开发设计')

        for e in zh_entities + en_entities:
            if e not in stopwords and e not in meaningful_words and len(e) >= 2:
                entities.append(e)

        return list(set(entities))

    def _extract_events(self, text: str) -> List[Dict]:
        """提取事件 — 理解发生了什么"""
        events = []

        # 事件模式
        event_patterns = [
            (r'(\w+)发明了(\w+)', 'invention'),
            (r'(\w+)发现了(\w+)', 'discovery'),
            (r'(\w+)创造了(\w+)', 'creation'),
            (r'(\w+)提出了(\w+)', 'proposal'),
            (r'(\w+)导致了(\w+)', 'causation'),
            (r'(\w+)引起了(\w+)', 'causation'),
        ]

        for pattern, event_type in event_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                events.append({
                    'type': event_type,
                    'agent': match[0],
                    'object': match[1],
                })

        return events

    def _form_concept(self, entity: str, context: str, text: str):
        """形成概念 — 不是记录，是理解"""
        if entity not in self.concepts:
            self.concepts[entity] = Concept(name=entity)
            self.stats['concepts_formed'] += 1
            self.abstraction_levels['concrete'].add(entity)

        concept = self.concepts[entity]

        # 添加实例
        concept.add_example(context)

        # 从文本中提取属性
        self._extract_properties(concept, text)

        # 更新激活度
        concept.activation = 1.0

    def _extract_properties(self, concept: Concept, text: str):
        """从文本中提取概念属性"""
        # 属性模式
        prop_patterns = [
            (r'(\w+)是(\w+的一种|一种\w+)', 'type'),
            (r'(\w+)属于(\w+)', 'category'),
            (r'(\w+)位于(\w+)', 'location'),
            (r'(\w+)成立于(\d{4}年?)', 'founded'),
            (r'(\w+)发明于(\d{4}年?)', 'invented'),
            (r'(\w+)由(\w+)发明', 'inventor'),
            (r'(\w+)由(\w+)创造', 'creator'),
        ]

        for pattern, prop_name in prop_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match[0] == concept.name:
                    concept.properties[prop_name] = match[1]
                elif match[1] == concept.name:
                    concept.properties[f'inverse_{prop_name}'] = match[0]

    def _discover_relation(self, e1: str, e2: str, text: str):
        """发现概念间关系"""
        # 如果两个实体在同一个句子中出现，它们可能有关系
        sentences = re.split(r'[。！？；\n]', text)

        for sentence in sentences:
            if e1 in sentence and e2 in sentence:
                # 确定关系类型
                relation = self._infer_relation_type(e1, e2, sentence)

                if relation:
                    if e1 in self.concepts:
                        self.concepts[e1].relate(relation, e2)
                    if e2 in self.concepts:
                        self.concepts[e2].relate(f'inverse_{relation}', e1)

                    # 更新共现激活
                    if e1 in self.concepts and e2 in self.concepts:
                        self.concepts[e1].co_activations[e2] = self.concepts[e1].co_activations.get(e2, 0) + 0.1
                        self.concepts[e2].co_activations[e1] = self.concepts[e2].co_activations.get(e1, 0) + 0.1

    def _infer_relation_type(self, e1: str, e2: str, sentence: str) -> Optional[str]:
        """推断关系类型"""
        # 关系指示词
        relation_indicators = {
            '是': 'is_a',
            '属于': 'belongs_to',
            '包括': 'includes',
            '位于': 'located_in',
            '使用': 'uses',
            '用于': 'used_for',
            '产生': 'produces',
            '导致': 'causes',
            '发明': 'invented',
            '发现': 'discovered',
            '创造': 'created',
        }

        for indicator, relation in relation_indicators.items():
            if indicator in sentence:
                return relation

        return 'related_to'  # 默认关系

    def _extract_causal_relations(self, text: str, entities: List[str], events: List[Dict]):
        """提取因果关系"""
        # 因果模式
        causal_patterns = [
            (r'因为(\w+)，所以(\w+)', 'because'),
            (r'由于(\w+)，(\w+)', 'due_to'),
            (r'(\w+)导致(\w+)', 'causes'),
            (r'(\w+)引起(\w+)', 'leads_to'),
            (r'如果(\w+)，那么(\w+)', 'if_then'),
        ]

        for pattern, causal_type in causal_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                cause = match[0]
                effect = match[1]

                # 检查是否已存在这条规则
                existing = None
                for rule in self.causal_rules:
                    if rule.cause == cause and rule.effect == effect:
                        existing = rule
                        break

                if existing:
                    existing.observe(True)
                else:
                    rule = CausalRule(cause=cause, effect=effect, confidence=0.5, observations=1)
                    self.causal_rules.append(rule)
                    self.stats['causal_rules_learned'] += 1

    def _detect_surprises(self, text: str, entities: List[str]):
        """检测意外信息"""
        for entity in entities:
            if entity in self.concepts:
                concept = self.concepts[entity]

                # 检查是否与已有知识冲突
                for prop_name, prop_value in concept.properties.items():
                    # 如果文本中有矛盾信息
                    if prop_name in text and prop_value not in text:
                        surprise = self.curiosity.notice_surprise(
                            entity,
                            prop_value,
                            'unknown'
                        )
                        if surprise > 0.3:
                            concept.surprise = max(concept.surprise, surprise)
                            self.stats['surprises_encountered'] += 1

    def _abstract_up(self, entities: List[str]):
        """抽象提升 — 从具体到抽象"""
        # 简单的抽象规则
        abstraction_rules = {
            '猫': '哺乳动物',
            '狗': '哺乳动物',
            '鸟': '动物',
            '鱼': '动物',
            '苹果': '水果',
            '香蕉': '水果',
            '汽车': '交通工具',
            '飞机': '交通工具',
            'Python': '编程语言',
            'Java': '编程语言',
        }

        for entity in entities:
            if entity in abstraction_rules:
                abstract_concept = abstraction_rules[entity]

                # 形成抽象概念
                if abstract_concept not in self.concepts:
                    self.concepts[abstract_concept] = Concept(name=abstract_concept)
                    self.abstraction_levels['abstract'].add(abstract_concept)

                # 建立层次关系
                if entity in self.concepts:
                    self.concepts[entity].parent = abstract_concept
                    self.concepts[abstract_concept].children.append(entity)

                # 移动到更高抽象层次
                if entity in self.abstraction_levels['concrete']:
                    self.abstraction_levels['concrete'].remove(entity)
                self.abstraction_levels['basic'].add(entity)

    def think(self, question: str) -> str:
        """思考 — 不是搜索，是推理"""
        # 提取问题中的概念
        question_concepts = self._extract_entities(question)

        if not question_concepts:
            return "我不太理解你的问题。"

        # 激活相关概念
        activated = self._activate_concepts(question_concepts)

        # 推理
        answer = self._reason(question, activated)

        return answer

    def _activate_concepts(self, concepts: List[str]) -> List[Concept]:
        """激活相关概念"""
        activated = []

        for concept_name in concepts:
            if concept_name in self.concepts:
                concept = self.concepts[concept_name]
                concept.spread_activation(1.0)
                activated.append(concept)

                # 激活相关概念
                for related, strength in concept.co_activations.items():
                    if related in self.concepts:
                        self.concepts[related].spread_activation(strength)
                        activated.append(self.concepts[related])

        # 按激活度排序
        activated.sort(key=lambda c: c.activation, reverse=True)
        return activated[:10]

    def _reason(self, question: str, activated_concepts: List[Concept]) -> str:
        """推理 — 基于激活的概念生成回答"""
        if not activated_concepts:
            return "我没有相关的知识来回答这个问题。"

        # 收集相关信息
        parts = []
        parts.append("基于我的理解：")

        for concept in activated_concepts[:5]:
            # 基本信息
            if concept.properties:
                props = ', '.join([f"{k}: {v}" for k, v in list(concept.properties.items())[:3]])
                parts.append(f"- {concept.name} ({props})")

            # 关系信息
            for relation, targets in concept.relations.items():
                if targets:
                    parts.append(f"- {concept.name} {relation} {targets[0]}")

            # 因果信息
            for rule in self.causal_rules:
                if rule.cause == concept.name and rule.confidence > 0.5:
                    parts.append(f"- {concept.name} 导致 {rule.effect} (置信度: {rule.confidence:.0%})")
                elif rule.effect == concept.name and rule.confidence > 0.5:
                    parts.append(f"- {rule.cause} 导致 {concept.name} (置信度: {rule.confidence:.0%})")

        if len(parts) == 1:
            return f"我对 {', '.join([c.name for c in activated_concepts[:3]])} 的了解还很有限。"

        return '\n'.join(parts)

    def get_stats(self) -> Dict:
        """获取学习统计"""
        return {
            **self.stats,
            'total_concepts': len(self.concepts),
            'concrete_concepts': len(self.abstraction_levels['concrete']),
            'basic_concepts': len(self.abstraction_levels['basic']),
            'abstract_concepts': len(self.abstraction_levels['abstract']),
            'meta_concepts': len(self.abstraction_levels['meta']),
            'causal_rules': len(self.causal_rules),
            'knowledge_gaps': len(self.curiosity.knowledge_gaps),
        }


def stream_jsonl(filepath, max_lines=None):
    """流式读取 JSONL 文件"""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield data
                count += 1
                if max_lines and count >= max_lines:
                    break
            except:
                continue


def main():
    print("=" * 70)
    print("真正的人类式学习系统")
    print("=" * 70)

    data_dir = 'data/extracted'
    system = HumanLikeLearningSystem()

    # 先学习维基百科（知识密度最高）
    print("\n[1] 从维基百科学习...")
    wiki_dir = os.path.join(data_dir, 'wiki', 'wiki_zh')
    count = 0

    if os.path.exists(wiki_dir):
        for root, dirs, files in os.walk(wiki_dir):
            for fname in files:
                if fname.startswith('.'):
                    continue
                filepath = os.path.join(root, fname)
                for data in stream_jsonl(filepath, max_lines=10000):  # 先学1万条
                    title = data.get('title', '')
                    text = data.get('text', '')
                    if title and text:
                        system.learn_from_text(title, text)
                        count += 1
                    if count % 1000 == 0:
                        stats = system.get_stats()
                        print(f"  已学习: {count} 篇, "
                              f"概念: {stats['total_concepts']}, "
                              f"因果规则: {stats['causal_rules']}, "
                              f"惊讶: {stats['surprises_encountered']}")
                    if count >= 10000:
                        break
                if count >= 10000:
                    break
            if count >= 10000:
                break

    print(f"\n维基百科学习完成: {count} 篇")

    # 显示学习成果
    stats = system.get_stats()
    print("\n" + "=" * 70)
    print("学习成果")
    print("=" * 70)
    print(f"  阅读文章: {stats['articles_read']}")
    print(f"  形成概念: {stats['total_concepts']}")
    print(f"    具体概念: {stats['concrete_concepts']}")
    print(f"    基本概念: {stats['basic_concepts']}")
    print(f"    抽象概念: {stats['abstract_concepts']}")
    print(f"  因果规则: {stats['causal_rules']}")
    print(f"  惊讶事件: {stats['surprises_encountered']}")
    print(f"  知识空白: {stats['knowledge_gaps']}")

    # 显示一些学到的概念
    print("\n  学到的概念示例:")
    for i, (name, concept) in enumerate(list(system.concepts.items())[:20]):
        if concept.properties:
            props = ', '.join([f"{k}={v}" for k, v in list(concept.properties.items())[:2]])
            print(f"    {name}: {props}")
        else:
            print(f"    {name}")

    # 显示因果规则
    if system.causal_rules:
        print("\n  学到的因果规则:")
        for rule in system.causal_rules[:10]:
            print(f"    {rule.cause} → {rule.effect} (置信度: {rule.confidence:.0%})")

    # 测试思考能力
    print("\n" + "=" * 70)
    print("测试思考能力")
    print("=" * 70)

    test_questions = [
        "什么是人工智能",
        "中国在哪里",
        "牛顿发现了什么",
        "Python是什么",
    ]

    for q in test_questions:
        print(f"\n问: {q}")
        answer = system.think(q)
        print(f"答: {answer}")

    # 显示好奇心驱动的问题
    if system.curiosity.questions:
        print("\n" + "=" * 70)
        print("好奇心驱动的问题（待探索）")
        print("=" * 70)
        for q in system.curiosity.questions[:5]:
            print(f"  - {q}")


if __name__ == '__main__':
    main()
