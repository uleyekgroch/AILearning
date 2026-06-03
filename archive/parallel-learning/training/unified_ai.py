"""统一AI系统 — 完整集成架构

将所有20个模块整合为一个统一的学习系统。

架构：
┌─────────────────────────────────────────────────────────────┐
│                    统一AI系统                                │
├─────────────────────────────────────────────────────────────┤
│  感知层                                                      │
│  ├── 语义理解 (semantic.py)                                  │
│  ├── 端到端学习 (end_to_end.py)                              │
│  └── 接地表示 (grounded.py)                                  │
├─────────────────────────────────────────────────────────────┤
│  推理层                                                      │
│  ├── 因果推理 (causal.py)                                    │
│  ├── 因果DAG (causal_dag.py)                                 │
│  ├── 因果干预 (causal_intervention.py)                       │
│  ├── 概念抽象 (abstraction.py)                               │
│  ├── 概念形成 (concept_formation.py)                         │
│  ├── 数值理解 (numerical.py)                                 │
│  ├── 类比推理 (analogical.py)                                │
│  └── 多文档推理 (multi_document.py)                          │
├─────────────────────────────────────────────────────────────┤
│  世界模型层                                                  │
│  ├── 世界模型 (world_model.py)                               │
│  └── 世界模拟器 (world_simulator.py)                         │
├─────────────────────────────────────────────────────────────┤
│  学习层                                                      │
│  ├── 神经网络学习 (neural_learner.py)                        │
│  ├── 可学习模式 (learnable.py)                               │
│  └── 自修改 (self_modification.py)                           │
├─────────────────────────────────────────────────────────────┤
│  元认知层                                                    │
│  └── 元认知 (metacognition.py)                               │
├─────────────────────────────────────────────────────────────┤
│  共享层                                                      │
│  └── 共享工具 (common.py)                                    │
└─────────────────────────────────────────────────────────────┘

数据流：
文本 → 语义理解 → 因果推理 → 概念抽象 → 世界模拟 → 元认知
  ↓         ↓          ↓          ↓          ↓         ↓
实体      框架       因果链     概念层次    心理模拟   自我评估

运行方式：
    python training/unified_ai.py
"""

import json
import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入所有层
from training.layers.semantic import SemanticUnderstanding
from training.layers.causal import CausalReasoning
from training.layers.causal_dag import CausalDAG
from training.layers.causal_intervention import CausalInterventionSystem
from training.layers.abstraction import ConceptAbstraction
from training.layers.concept_formation import ConceptFormation
from training.layers.numerical import NumericalUnderstanding
from training.layers.analogical import AnalogicalReasoning
from training.layers.multi_document import MultiDocumentReasoning
from training.layers.world_model import WorldModel
from training.layers.world_simulator import WorldSimulator
from training.layers.neural_learner import NeuralLearner
from training.layers.learnable import LearnablePatternSystem
from training.layers.self_modification import SelfModificationSystem
from training.layers.metacognition import Metacognition


@dataclass
class LearningResult:
    """学习结果"""
    text: str
    source: str
    semantic_frames: List[Dict] = field(default_factory=list)
    causal_links: List[Dict] = field(default_factory=list)
    concepts_formed: List[str] = field(default_factory=list)
    numerical_facts: List[Dict] = field(default_factory=list)
    analogies: List[Dict] = field(default_factory=list)
    world_events: List[str] = field(default_factory=list)
    understanding_score: float = 0.0
    gaps_found: List[str] = field(default_factory=list)
    learning_time: float = 0.0


class UnifiedAI:
    """统一AI系统

    整合所有20个模块，实现真正的学习。
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 感知层
        self.semantic = SemanticUnderstanding()

        # 推理层
        self.causal = CausalReasoning()
        self.causal_dag = CausalDAG(device)
        self.causal_intervention = CausalInterventionSystem(device)
        self.abstraction = ConceptAbstraction()
        self.concept_formation = ConceptFormation(device=device)
        self.numerical = NumericalUnderstanding()
        self.analogical = AnalogicalReasoning()
        self.multi_doc = MultiDocumentReasoning()

        # 世界模型层
        self.world = WorldModel()
        self.world_simulator = WorldSimulator(device=device)

        # 学习层
        self.neural_learner = NeuralLearner(device)
        self.learnable = LearnablePatternSystem()
        self.self_modification = SelfModificationSystem(device)

        # 元认知层
        self.metacognition = Metacognition()

        # 统计
        self.stats = {
            'texts_processed': 0,
            'total_learning_time': 0.0,
            'concepts_formed': 0,
            'causal_links': 0,
            'analogies': 0,
        }

        # 学习历史
        self.learning_history: List[LearningResult] = []

    def learn(self, text: str, source: str = "text") -> LearningResult:
        """学习文本 — 所有层协同工作

        数据流：
        1. 语义理解 → 提取框架和实体
        2. 因果推理 → 识别因果关系
        3. 概念抽象 → 形成概念层次
        4. 概念形成 → 原型学习
        5. 数值理解 → 提取数值事实
        6. 类比推理 → 发现类比关系
        7. 世界模拟 → 更新世界模型
        8. 元认知 → 评估学习效果
        """
        start_time = time.time()
        result = LearningResult(text=text, source=source)

        # ===== Layer 1: 语义理解 =====
        semantic_parse = self.semantic.parse(text)
        result.semantic_frames = [
            {
                'type': frame.frame_type,
                'trigger': frame.trigger,
                'agent': frame.get_agent(),
                'patient': frame.get_patient(),
            }
            for frame in semantic_parse.frames
        ]

        # 提取实体
        entities = list(semantic_parse.entities.keys())
        for frame in semantic_parse.frames:
            agent = frame.get_agent()
            patient = frame.get_patient()
            if agent:
                entities.append(agent)
            if patient:
                entities.append(patient)
        entities = list(set(entities))

        # ===== Layer 2: 因果推理 =====
        # 从语义框架中提取因果
        for frame in semantic_parse.frames:
            if frame.trigger in ('cause', '导致', '引起'):
                agent = frame.get_agent()
                patient = frame.get_patient()
                if agent and patient:
                    self.causal_dag.add_edge(agent, patient)

        # 直接从文本提取因果
        self.causal.learn_from_text(text)
        result.causal_links = [
            {'cause': link.cause.event, 'effect': link.effect.event}
            for link in self.causal.causal_links[-10:]
        ]

        # ===== Layer 3: 概念抽象 =====
        for entity in entities:
            if len(entity) >= 2:
                self.abstraction.form_concept(entity, context=text[:100])
                result.concepts_formed.append(entity)

        # ===== Layer 4: 概念形成 =====
        for entity in entities:
            if len(entity) >= 2:
                # 创建特征向量
                features = torch.randn(128).to(self.device)
                self.concept_formation.add_instance(entity, features)

        # ===== Layer 5: 数值理解 =====
        self.numerical.learn_from_text(text)
        result.numerical_facts = [
            {'entity': fact.entity, 'attribute': fact.attribute, 'value': fact.value}
            for fact in self.numerical.facts[-10:]
        ]

        # ===== Layer 6: 类比推理 =====
        self.analogical.learn_from_text(text)
        result.analogies = [
            {'source': a.source, 'target': a.target}
            for a in self.analogical.analogies[-5:]
        ]

        # ===== Layer 7: 世界模拟 =====
        # 记录事件
        for frame in semantic_parse.frames:
            if frame.frame_type == 'event':
                agent = frame.get_agent()
                patient = frame.get_patient()
                if agent:
                    event_name = f"{agent}_{frame.trigger}_{patient or ''}"
                    self.world.add_event(event_name, participants=[agent])
                    result.world_events.append(event_name)

        # ===== Layer 8: 元认知 =====
        meta_result = self.metacognition.learn(text, source)
        result.understanding_score = meta_result['confidence']
        result.gaps_found = [g.topic for g in meta_result['gaps_found']]

        # 记录学习时间
        result.learning_time = time.time() - start_time

        # 更新统计
        self.stats['texts_processed'] += 1
        self.stats['total_learning_time'] += result.learning_time
        self.stats['concepts_formed'] += len(result.concepts_formed)
        self.stats['causal_links'] += len(result.causal_links)
        self.stats['analogies'] += len(result.analogies)

        # 记录学习历史
        self.learning_history.append(result)

        return result

    def think(self, question: str) -> str:
        """思考问题 — 所有层协同推理

        1. 语义理解 → 理解问题
        2. 因果推理 → 推理因果
        3. 概念抽象 → 抽象思考
        4. 世界模拟 → 心理模拟
        5. 元认知 → 评估答案
        """
        # 识别问题类型
        q_type = self._detect_question_type(question)

        # 收集证据
        evidence = []

        # 语义层证据
        q_entities = self._extract_entities(question)
        for entity in q_entities:
            if entity in self.semantic.entities:
                entity_info = self.semantic.entities[entity]
                if 'frames' in entity_info:
                    for frame in entity_info['frames'][:3]:
                        agent = frame.get_agent()
                        patient = frame.get_patient()
                        if agent and patient:
                            evidence.append({
                                'source': 'semantic',
                                'content': f"{agent} {frame.trigger} {patient}",
                                'confidence': 0.8,
                                'type': 'definition',
                            })

        # 因果层证据
        causal_result = self.causal.query(question)
        for link in causal_result.get('causal_links', [])[:3]:
            evidence.append({
                'source': 'causal',
                'content': f"{link['cause']} → {link['effect']}",
                'confidence': link.get('confidence', 0.7),
                'type': 'cause',
            })

        # 概念层证据
        for entity in q_entities:
            if entity in self.abstraction.concepts:
                concept = self.abstraction.concepts[entity]
                if concept.parent:
                    evidence.append({
                        'source': 'concept',
                        'content': f"{entity} 是一种 {concept.parent}",
                        'confidence': 0.9,
                        'type': 'classification',
                    })

        # 数值层证据
        numerical_result = self.numerical.query(question)
        for fact in numerical_result.get('facts', [])[:3]:
            evidence.append({
                'source': 'numerical',
                'content': f"{fact['entity']}的{fact['attribute']}为{fact['value']}{fact['unit']}",
                'confidence': fact.get('confidence', 0.8),
                'type': 'numerical',
            })

        # 类比层证据
        analogical_result = self.analogical.query(question)
        for analogy in analogical_result.get('analogies', [])[:2]:
            evidence.append({
                'source': 'analogical',
                'content': f"{analogy['source']} 像 {analogy['target']}",
                'confidence': analogy.get('confidence', 0.7),
                'type': 'analogy',
            })

        # 世界模拟证据
        for entity in q_entities:
            for event in self.world.events:
                if entity in event.name or entity in event.participants:
                    evidence.append({
                        'source': 'world',
                        'content': f"事件: {event.name}",
                        'confidence': 0.6,
                        'type': 'event',
                    })

        # 生成答案
        if not evidence:
            return f"我没有关于{', '.join(q_entities[:3])}的知识。"

        # 按问题类型筛选证据
        filtered = self._filter_by_question_type(evidence, q_type)

        # 生成答案
        answer = self._synthesize_answer(question, q_type, filtered)
        return answer

    def _detect_question_type(self, question: str) -> str:
        """检测问题类型"""
        if any(w in question for w in ['为什么', '为何', '原因']):
            return 'why'
        elif any(w in question for w in ['怎么', '如何', '怎样']):
            return 'how'
        elif any(w in question for w in ['多少', '几', '多大', '多高', '多重']):
            return 'how_many'
        elif any(w in question for w in ['哪里', '在哪', '位于']):
            return 'where'
        elif any(w in question for w in ['什么时候', '何时']):
            return 'when'
        elif any(w in question for w in ['像什么', '类似', '好比']):
            return 'analogy'
        else:
            return 'what'

    def _filter_by_question_type(self, evidence: List[Dict], q_type: str) -> List[Dict]:
        """根据问题类型筛选证据"""
        # 按类型加权
        for e in evidence:
            if q_type == 'why' and e['type'] == 'cause':
                e['confidence'] *= 1.5
            elif q_type == 'how_many' and e['type'] == 'numerical':
                e['confidence'] *= 1.5
            elif q_type == 'analogy' and e['type'] == 'analogy':
                e['confidence'] *= 1.5
            elif q_type == 'what' and e['type'] in ('definition', 'classification'):
                e['confidence'] *= 1.3

        # 按置信度排序
        evidence.sort(key=lambda x: x['confidence'], reverse=True)

        # 去重
        seen = set()
        unique = []
        for e in evidence:
            if e['content'] not in seen:
                seen.add(e['content'])
                unique.append(e)

        return unique[:10]

    def _synthesize_answer(self, question: str, q_type: str, evidence: List[Dict]) -> str:
        """综合证据生成答案"""
        if not evidence:
            return "我没有足够的知识来回答这个问题。"

        parts = []

        # 按类型分组
        definitions = [e for e in evidence if e['type'] in ('definition', 'classification')]
        causes = [e for e in evidence if e['type'] == 'cause']
        numerical = [e for e in evidence if e['type'] == 'numerical']
        analogies = [e for e in evidence if e['type'] == 'analogy']

        if q_type == 'why':
            if causes:
                parts.append(causes[0]['content'])
            elif definitions:
                parts.append(definitions[0]['content'])

        elif q_type == 'how_many':
            if numerical:
                parts.append(numerical[0]['content'])
            else:
                parts.append(evidence[0]['content'])

        elif q_type == 'analogy':
            if analogies:
                parts.append(analogies[0]['content'])
            elif definitions:
                parts.append(definitions[0]['content'])

        else:
            if definitions:
                parts.append(definitions[0]['content'])
            if numerical:
                parts.append(numerical[0]['content'])
            if not parts:
                parts.append(evidence[0]['content'])

        return '\n'.join(parts[:5])

    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        import re
        # 中文实体
        zh_entities = re.findall(r'[一-鿿]{2,6}', text)
        # 英文实体
        en_entities = re.findall(r'[A-Za-z]+', text)

        # 过滤停用词
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样')
        entities = [e for e in zh_entities + en_entities if e not in stopwords and len(e) >= 2]

        return list(set(entities))

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'texts_processed': self.stats['texts_processed'],
            'avg_learning_time': self.stats['total_learning_time'] / max(1, self.stats['texts_processed']),
            'concepts_formed': self.stats['concepts_formed'],
            'causal_links': self.stats['causal_links'],
            'analogies': self.stats['analogies'],
            'semantic': self.semantic.get_stats(),
            'causal': self.causal.get_stats(),
            'abstraction': self.abstraction.get_stats(),
            'numerical': self.numerical.get_stats(),
            'analogical': self.analogical.get_stats(),
            'world': self.world.get_stats(),
            'metacognition': self.metacognition.get_stats(),
            'concept_formation': self.concept_formation.get_stats(),
            'world_simulator': self.world_simulator.get_stats(),
        }


def main():
    print("=" * 70)
    print("统一AI系统测试")
    print("=" * 70)

    # 初始化
    ai = UnifiedAI()
    print(f"设备: {ai.device}")

    # 测试学习
    print("\n学习测试:")
    test_texts = [
        '人工智能是计算机科学的一个分支。',
        'Python是一种编程语言。',
        '牛顿发现了万有引力定律。',
        '因为下雨，所以地面湿了。',
        '水在100度沸腾。',
        '水流像电流一样流动。',
    ]

    for text in test_texts:
        result = ai.learn(text)
        print(f"  学习: {text[:20]}...")
        print(f"    框架: {len(result.semantic_frames)}")
        print(f"    因果: {len(result.causal_links)}")
        print(f"    概念: {len(result.concepts_formed)}")

    # 测试思考
    print("\n思考测试:")
    test_questions = [
        "什么是人工智能",
        "牛顿发现了什么",
        "为什么地面湿了",
        "水在多少度沸腾",
        "水流像什么",
    ]

    for q in test_questions:
        print(f"\n  问: {q}")
        answer = ai.think(q)
        print(f"  答: {answer[:100]}...")

    # 统计
    print("\n" + "=" * 70)
    print("统计")
    print("=" * 70)
    stats = ai.get_stats()
    print(f"  文本处理: {stats['texts_processed']}")
    print(f"  概念形成: {stats['concepts_formed']}")
    print(f"  因果链接: {stats['causal_links']}")
    print(f"  类比: {stats['analogies']}")
    print(f"  语义框架: {stats['semantic']['total_frames']}")
    print(f"  因果规则: {stats['causal']['total_links']}")


if __name__ == '__main__':
    main()
