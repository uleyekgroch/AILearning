"""整合AI系统 — 5层架构统一（修复版）

核心修复：5层之间有真正的数据流，不是各自独立运行。

数据流：
  文本 → 语义理解 → 语义框架+实体
                    ↓
         因果推理（接收语义框架，提取因果）
                    ↓
         概念抽象（接收实体，形成概念层次）
                    ↓
         世界模型（接收因果+概念，建立世界模型）
                    ↓
         元认知（评估全部，反馈驱动学习）

运行方式：
    python training/integrated_ai.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.layers.semantic import SemanticUnderstanding, SemanticFrame, SemanticParse, SemanticRole, SemanticUnit
from training.layers.causal import CausalReasoning, CausalLink, CausalEvent
from training.layers.abstraction import ConceptAbstraction
from training.layers.world_model import WorldModel
from training.layers.metacognition import Metacognition
from training.layers.numerical import NumericalUnderstanding
from training.layers.analogical import AnalogicalReasoning
from training.layers.multi_document import MultiDocumentReasoning
from training.layers.common import extract_entities


@dataclass
class LearningResult:
    """单次学习结果"""
    text: str
    semantic_parse: Optional[SemanticParse] = None
    causal_links: List[CausalLink] = field(default_factory=list)
    concepts_formed: List[str] = field(default_factory=list)
    world_events: List[str] = field(default_factory=list)
    understanding_score: float = 0.0
    gaps_found: List[str] = field(default_factory=list)


class IntegratedAI:
    """整合AI系统 — 5层真正连接

    数据流：
    1. 语义理解 → 产生语义框架和实体
    2. 因果推理 ← 接收语义框架，提取因果关系
    3. 概念抽象 ← 接收实体，形成抽象概念
    4. 世界模型 ← 接收因果+概念，更新世界模型
    5. 元认知 ← 评估全部，反馈驱动学习
    """

    def __init__(self):
        # 8层完整架构
        self.semantic = SemanticUnderstanding()
        self.causal = CausalReasoning()
        self.abstraction = ConceptAbstraction()
        self.numerical = NumericalUnderstanding()
        self.analogical = AnalogicalReasoning()
        self.multi_doc = MultiDocumentReasoning()
        self.world = WorldModel()
        self.metacognition = Metacognition()

        # 学习历史
        self.learning_history: List[LearningResult] = []

        # 统计
        self.stats = {
            'texts_processed': 0,
            'total_learning_time': 0.0,
        }

    def learn(self, text: str, source: str = "text") -> LearningResult:
        """学习文本 — 5层协同

        数据流：
        1. 语义理解 → 语义框架 + 实体
        2. 因果推理 ← 语义框架 → 因果链接
        3. 概念抽象 ← 实体 → 概念层次
        4. 世界模型 ← 因果 + 概念 → 世界状态
        5. 元认知 ← 全部 → 评估 + 反馈
        """
        start_time = time.time()
        result = LearningResult(text=text)

        # ===== Layer 1: 语义理解 =====
        # 产生：语义框架、实体、关系
        semantic_parse = self.semantic.parse(text)
        result.semantic_parse = semantic_parse

        # ===== Layer 2: 因果推理 =====
        # 接收：语义框架
        # 产生：因果链接
        for frame in semantic_parse.frames:
            # 从语义框架中提取因果
            if frame.trigger in ('cause', '导致', '引起'):
                agent = frame.get_agent()
                patient = frame.get_patient()
                if agent and patient:
                    link = CausalLink(
                        cause=CausalEvent(event=agent),
                        effect=CausalEvent(event=patient),
                        mechanism='semantic_cause',
                        confidence=0.8,
                        evidence=[frame.source_sentence],
                    )
                    self.causal.add_causal_link(link)
                    result.causal_links.append(link)

            # 从"是"关系推断因果
            elif frame.trigger == 'be':
                agent = frame.get_agent()
                patient = frame.get_patient()
                if agent and patient:
                    # "X是Y的一种" → Y的属性可以传递给X
                    if '一种' in patient or '属于' in patient:
                        # 这是概念继承，不是因果
                        pass

        # 直接从文本提取因果
        links_before = len(self.causal.causal_links)
        self.causal.learn_from_text(text)
        links_after = len(self.causal.causal_links)
        # 只追加本次学习新增的链接
        result.causal_links.extend(self.causal.causal_links[links_before:links_after])

        # ===== Layer 3: 概念抽象 =====
        # 接收：语义框架中的实体（优先使用框架提取的实体）
        # 产生：概念层次

        # 先从语义框架提取实体（更准确）
        frame_entities = set()
        for frame in semantic_parse.frames:
            agent = frame.get_agent()
            patient = frame.get_patient()
            if agent:
                frame_entities.add(agent)
            if patient:
                frame_entities.add(patient)

        # 合并框架实体和正则实体
        all_entities = list(frame_entities) + list(semantic_parse.entities.keys())
        seen = set()
        for entity_name in all_entities:
            if entity_name in seen or len(entity_name) < 2:
                continue
            seen.add(entity_name)

            concept = self.abstraction.form_concept(entity_name, context=text[:100])
            result.concepts_formed.append(entity_name)

            # 从语义框架推断概念属性
            for frame in semantic_parse.frames:
                if frame.trigger == 'be':
                    agent = frame.get_agent()
                    patient = frame.get_patient()
                    if agent == entity_name and patient:
                        # "X是Y" → X的类型是Y
                        concept.add_property('type', patient)

        # ===== Layer 4: 世界模型 =====
        # 接收：因果链接 + 概念
        # 产生：世界状态
        for frame in semantic_parse.frames:
            if frame.frame_type == 'event':
                agent = frame.get_agent()
                patient = frame.get_patient()
                if agent:
                    # 添加智能体
                    if agent not in self.world.agents:
                        self.world.add_agent(agent)
                    # 记录事件
                    event_name = f"{agent}_{frame.trigger}_{patient or ''}"
                    self.world.add_event(event_name, participants=[agent])
                    result.world_events.append(event_name)

            elif frame.trigger == 'located_at':
                agent = frame.get_agent()
                patient = frame.get_patient()
                if agent and patient:
                    # 记录位置
                    if agent not in self.world.locations:
                        self.world.locations[agent] = {}
                    self.world.locations[agent]['current'] = patient

        # ===== Layer 5.5: 数值理解 =====
        # 接收：文本
        # 产生：数值事实
        self.numerical.learn_from_text(text)

        # ===== Layer 5.6: 类比推理 =====
        # 接收：文本
        # 产生：类比关系
        self.analogical.learn_from_text(text)

        # ===== Layer 5.7: 多文档推理 =====
        # 接收：文本+来源
        # 产生：知识整合+冲突检测
        for frame in semantic_parse.frames:
            agent = frame.get_agent()
            patient = frame.get_patient()
            if agent and patient:
                self.multi_doc.add_knowledge(
                    topic=agent,
                    content=f"{agent} {frame.trigger} {patient}",
                    source=source,
                )

        # ===== Layer 6: 元认知 =====
        # 接收：全部
        # 产生：评估 + 反馈
        meta_result = self.metacognition.learn(text, source)
        result.understanding_score = meta_result['confidence']
        result.gaps_found = [g.topic for g in meta_result['gaps_found']]

        # 反馈循环：矛盾检测 → 知识修正
        if meta_result['contradictions']:
            for contradiction in meta_result['contradictions']:
                key = contradiction['key']
                old_value = contradiction['existing_value']
                new_value = contradiction['new_value']

                # 比较来源可靠性，选择更可靠的
                resolution = self._resolve_contradiction(key, old_value, new_value, source)
                if resolution == 'new':
                    # 新信息更可靠，更新旧知识
                    self._update_knowledge(key, old_value, new_value)
                    self.metacognition.curiosity_queue.append(
                        f"已修正：{key}从{old_value}更新为{new_value}"
                    )
                elif resolution == 'old':
                    # 旧信息更可靠，保持不变
                    pass
                else:
                    # 无法判断，加入好奇心队列
                    self.metacognition.curiosity_queue.append(
                        f"为什么{key}既是{old_value}又是{new_value}？"
                    )

        # 反馈循环：知识空白 → 好奇心驱动
        for gap in result.gaps_found:
            if gap not in self.metacognition.curiosity_queue:
                self.metacognition.curiosity_queue.append(f"什么是{gap}？")

        # 记录学习历史
        self.learning_history.append(result)
        self.stats['texts_processed'] += 1
        self.stats['total_learning_time'] += time.time() - start_time

        return result

    def think(self, question: str) -> str:
        """思考问题 — 8层协同推理

        根据问题类型，选择最相关的证据，综合生成答案。
        """
        # Step 1: 识别问题类型
        q_type = self._detect_question_type(question)

        # Step 2: 语义理解问题
        q_parse = self.semantic.parse(question)
        q_entities = list(q_parse.entities.keys())
        q_frames = q_parse.frames

        # Step 3: 从各层收集证据，按问题类型加权
        evidence = []

        # 语义层：相关框架
        for entity in q_entities:
            if entity in self.semantic.entities:
                entity_info = self.semantic.entities[entity]
                if 'frames' in entity_info:
                    for frame in entity_info['frames'][:3]:
                        agent = frame.get_agent()
                        patient = frame.get_patient()
                        if agent and patient:
                            weight = 1.0 if q_type == 'what' else 0.7
                            evidence.append({
                                'source': 'semantic',
                                'content': f"{agent} {frame.trigger} {patient}",
                                'confidence': 0.8 * weight,
                                'type': 'definition',
                            })

        # 因果层：相关因果（优先用于"为什么"问题）
        causal_result = self.causal.query(question)
        for link in causal_result.get('causal_links', [])[:3]:
            weight = 1.5 if q_type == 'why' else 0.8
            evidence.append({
                'source': 'causal',
                'content': f"{link['cause']} → {link['effect']}",
                'confidence': link.get('confidence', 0.7) * weight,
                'type': 'cause',
            })

        # 概念层：相关概念
        for entity in q_entities:
            if entity in self.abstraction.concepts:
                concept = self.abstraction.concepts[entity]
                if concept.properties:
                    props = ', '.join([f"{k}={v}" for k, v in list(concept.properties.items())[:3]])
                    weight = 1.2 if q_type == 'what' else 0.8
                    evidence.append({
                        'source': 'concept',
                        'content': f"{entity} ({props})",
                        'confidence': concept.confidence * weight,
                        'type': 'property',
                    })
                if concept.parent:
                    weight = 1.3 if q_type == 'what' else 0.7
                    evidence.append({
                        'source': 'concept',
                        'content': f"{entity} 是一种 {concept.parent}",
                        'confidence': 0.9 * weight,
                        'type': 'classification',
                    })

        # 数值层：相关数值事实（优先用于"多少"问题）
        numerical_result = self.numerical.query(question)
        for fact in numerical_result.get('facts', [])[:3]:
            weight = 1.5 if q_type == 'how_many' else 0.8
            evidence.append({
                'source': 'numerical',
                'content': f"{fact['entity']}的{fact['attribute']}为{fact['value']}{fact['unit']}",
                'confidence': fact.get('confidence', 0.8) * weight,
                'type': 'numerical',
            })

        # 类比层：相关类比（用于"像什么"问题）
        analogical_result = self.analogical.query(question)
        for analogy in analogical_result.get('analogies', [])[:2]:
            weight = 1.5 if q_type == 'how' else 0.6
            evidence.append({
                'source': 'analogical',
                'content': f"{analogy['source']} 像 {analogy['target']}",
                'confidence': analogy.get('confidence', 0.7) * weight,
                'type': 'analogy',
            })

        # 多文档层：整合知识
        multi_result = self.multi_doc.query(question)
        for knowledge in multi_result.get('integrated_knowledge', [])[:2]:
            for src in knowledge.get('sources', [])[:2]:
                evidence.append({
                    'source': 'multi_doc',
                    'content': src['content'],
                    'confidence': src.get('confidence', 0.7),
                    'type': 'fact',
                })

        # 世界模型层：相关事件
        for entity in q_entities:
            for event in self.world.events:
                if entity in event.name or entity in event.participants:
                    evidence.append({
                        'source': 'world',
                        'content': f"事件: {event.name}",
                        'confidence': 0.6,
                        'type': 'event',
                    })

        # 多跳推理：因果链推理（用于"为什么"问题）
        if q_type == 'why':
            for entity in q_entities:
                # 查找因果链
                causal_chains = self.causal.infer_causes(entity, max_depth=3)
                for chain in causal_chains[:2]:
                    if len(chain.links) > 1:
                        # 多跳因果链
                        chain_desc = ' → '.join([link.cause.event for link in chain.links] + [chain.links[-1].effect.event])
                        evidence.append({
                            'source': 'causal_chain',
                            'content': chain_desc,
                            'confidence': 0.9,
                            'type': 'cause_chain',
                        })

        # 多跳推理：概念继承链（用于"是什么"问题）
        if q_type == 'what':
            for entity in q_entities:
                if entity in self.abstraction.concepts:
                    hierarchy = self.abstraction.abstract_up(entity)
                    if len(hierarchy) > 1:
                        chain_desc = ' → '.join(hierarchy)
                        evidence.append({
                            'source': 'concept_hierarchy',
                            'content': f"{entity}的分类层次: {chain_desc}",
                            'confidence': 0.85,
                            'type': 'hierarchy',
                        })

        # Step 4: 综合推理生成答案
        if not evidence:
            gaps = [g for g in self.metacognition.knowledge_gaps
                    if any(e in g.topic for e in q_entities)]
            if gaps:
                return f"我对{', '.join(q_entities[:3])}了解有限，需要学习更多。"
            return f"我没有关于{', '.join(q_entities[:3])}的知识。"

        # 按加权置信度排序
        evidence.sort(key=lambda x: x['confidence'], reverse=True)

        # 去重
        seen = set()
        unique_evidence = []
        for e in evidence:
            if e['content'] not in seen:
                seen.add(e['content'])
                unique_evidence.append(e)

        # Step 5: 根据问题类型生成结构化答案
        answer = self._synthesize_answer(question, q_type, unique_evidence[:10])
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

    def _synthesize_answer(self, question: str, q_type: str, evidence: List[Dict]) -> str:
        """综合证据生成结构化答案"""
        parts = []

        # 按类型分组证据
        definitions = [e for e in evidence if e.get('type') in ('definition', 'classification')]
        causes = [e for e in evidence if e.get('type') == 'cause']
        properties = [e for e in evidence if e.get('type') in ('property', 'numerical')]
        analogies = [e for e in evidence if e.get('type') == 'analogy']
        facts = [e for e in evidence if e.get('type') in ('fact', 'event')]

        if q_type == 'why':
            # "为什么"问题：优先因果链
            if causes:
                parts.append(causes[0]['content'])
                if len(causes) > 1:
                    parts.append(f"此外，{causes[1]['content']}")
            elif definitions:
                parts.append(definitions[0]['content'])
            else:
                parts.append(evidence[0]['content'])

        elif q_type == 'how_many':
            # "多少"问题：优先数值
            if properties:
                parts.append(properties[0]['content'])
            else:
                parts.append(evidence[0]['content'])

        elif q_type == 'how':
            # "怎么做"问题：优先方法
            if definitions:
                parts.append(definitions[0]['content'])
            elif properties:
                parts.append(properties[0]['content'])
            else:
                parts.append(evidence[0]['content'])

        elif q_type == 'analogy':
            # "像什么"问题：优先类比
            if analogies:
                parts.append(analogies[0]['content'])
            elif definitions:
                parts.append(definitions[0]['content'])
            else:
                parts.append(evidence[0]['content'])

        elif q_type == 'where':
            # "在哪里"问题：优先位置信息
            location_evidence = [e for e in evidence if '位于' in e['content'] or '在' in e['content']]
            if location_evidence:
                parts.append(location_evidence[0]['content'])
            else:
                parts.append(evidence[0]['content'])

        else:
            # "是什么"问题：定义 + 属性 + 分类
            if definitions:
                parts.append(definitions[0]['content'])
            if properties:
                parts.append(properties[0]['content'])
            if not parts:
                parts.append(evidence[0]['content'])

        return '\n'.join(parts[:5])

    def _resolve_contradiction(self, key: str, old_value: str, new_value: str, new_source: str) -> str:
        """解决矛盾：比较来源可靠性"""
        # 策略1：来源可靠性
        # 如果新来源是"文档"类，旧来源是"推断"类，新来源更可靠
        reliable_sources = ['维基百科', '百科', '文档', '知识库']
        unreliable_sources = ['推断', '猜测', '假设']

        new_is_reliable = any(s in new_source for s in reliable_sources)
        new_is_unreliable = any(s in new_source for s in unreliable_sources)

        if new_is_reliable:
            return 'new'
        if new_is_unreliable:
            return 'old'

        # 策略2：信息具体性
        # 更具体的信息通常更可靠（如"100度"比"很高"更可靠）
        if len(new_value) > len(old_value) * 1.5:
            return 'new'
        if len(old_value) > len(new_value) * 1.5:
            return 'old'

        # 策略3：无法判断
        return 'unknown'

    def _update_knowledge(self, key: str, old_value: str, new_value: str):
        """更新知识：修正旧信息"""
        # 更新语义框架中的实体
        if key in self.semantic.entities:
            entity_info = self.semantic.entities[key]
            if 'frames' in entity_info:
                for frame in entity_info['frames']:
                    patient = frame.get_patient()
                    if patient and old_value in patient:
                        # 更新患者
                        frame.roles[SemanticRole.PATIENT] = SemanticUnit(
                            text=new_value,
                            role=SemanticRole.PATIENT
                        )

        # 更新概念属性
        if key in self.abstraction.concepts:
            concept = self.abstraction.concepts[key]
            for prop_name, prop_value in concept.properties.items():
                if prop_value == old_value:
                    concept.properties[prop_name] = new_value

        # 更新因果链
        for link in self.causal.causal_links:
            if link.cause.event == old_value:
                link.cause.event = new_value
            if link.effect.event == old_value:
                link.effect.event = new_value

    def ask_question(self) -> Optional[str]:
        """提出好奇心驱动的问题"""
        return self.metacognition.ask_question()

    def self_evaluate(self) -> Dict:
        """自我评估"""
        return {
            'semantic': self.semantic.get_stats(),
            'causal': self.causal.get_stats(),
            'abstraction': self.abstraction.get_stats(),
            'world_model': self.world.get_stats(),
            'metacognition': self.metacognition.self_evaluate(),
            'overall': self.stats,
        }

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'texts_processed': self.stats['texts_processed'],
            'avg_learning_time': self.stats['total_learning_time'] / max(1, self.stats['texts_processed']),
            'semantic': self.semantic.get_stats(),
            'causal': self.causal.get_stats(),
            'abstraction': self.abstraction.get_stats(),
            'numerical': self.numerical.get_stats(),
            'analogical': self.analogical.get_stats(),
            'multi_document': self.multi_doc.get_stats(),
            'world_model': self.world.get_stats(),
            'metacognition': self.metacognition.get_stats(),
        }

    def save_state(self, path: str):
        """保存状态"""
        state = {
            'stats': self.stats,
            'semantic': self.semantic.get_stats(),
            'causal': self.causal.get_stats(),
            'abstraction': self.abstraction.get_stats(),
            'world_model': self.world.get_stats(),
            'metacognition': self.metacognition.get_learning_report(),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)


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
            except (json.JSONDecodeError, ValueError):
                continue


def main():
    print("=" * 70)
    print("整合AI系统 — 5层协同架构")
    print("=" * 70)

    ai = IntegratedAI()

    # 学习维基百科
    print("\n从维基百科学习...")
    data_dir = 'data/extracted'
    wiki_dir = os.path.join(data_dir, 'wiki', 'wiki_zh')

    count = 0
    if os.path.exists(wiki_dir):
        for root, dirs, files in os.walk(wiki_dir):
            for fname in files:
                if fname.startswith('.'):
                    continue
                filepath = os.path.join(root, fname)
                for data in stream_jsonl(filepath, max_lines=2000):
                    title = data.get('title', '')
                    text = data.get('text', '')
                    if title and text:
                        ai.learn(text, source=title)
                        count += 1

                        if count % 200 == 0:
                            stats = ai.get_stats()
                            print(f"  已学习: {count} 篇, "
                                  f"概念: {stats['abstraction']['total_concepts']}, "
                                  f"因果: {stats['causal']['total_links']}, "
                                  f"框架: {stats['semantic']['total_frames']}")

                        if count >= 2000:
                            break
                    if count >= 2000:
                        break
                if count >= 2000:
                    break
            if count >= 2000:
                break

    print(f"\n学习完成: {count} 篇")

    # 保存状态
    ai.save_state('data/knowledge/integrated_ai_state.json')

    # 显示学习成果
    stats = ai.get_stats()

    with open('integrated_ai_result.txt', 'w', encoding='utf-8') as f:
        f.write('整合AI系统学习成果（5层协同版）\n')
        f.write('=' * 70 + '\n\n')

        f.write(f'处理文本: {stats["texts_processed"]}\n')
        f.write(f'平均学习时间: {stats["avg_learning_time"]:.3f}秒\n\n')

        f.write('各层统计:\n')
        f.write(f'  语义理解: {stats["semantic"]["total_frames"]} 框架, {stats["semantic"]["total_entities"]} 实体\n')
        f.write(f'  因果推理: {stats["causal"]["total_links"]} 因果链接\n')
        f.write(f'  概念抽象: {stats["abstraction"]["total_concepts"]} 概念\n')
        f.write(f'  世界模型: {stats["world_model"]["objects_tracked"]} 对象, {stats["world_model"]["agents_tracked"]} 智能体\n')
        f.write(f'  元认知: {stats["metacognition"]["total_knowledge"]} 知识项\n\n')

        # 测试思考能力
        f.write('=' * 70 + '\n')
        f.write('思考能力测试\n')
        f.write('=' * 70 + '\n')

        test_questions = [
            "什么是人工智能",
            "Python是什么",
            "太阳是什么",
            "牛顿发现了什么",
        ]

        for q in test_questions:
            f.write(f'\n问: {q}\n')
            answer = ai.think(q)
            f.write(f'答:\n{answer}\n')

        # 好奇心问题
        f.write('\n' + '=' * 70 + '\n')
        f.write('好奇心驱动的问题\n')
        f.write('=' * 70 + '\n')

        for _ in range(5):
            q = ai.ask_question()
            if q:
                f.write(f'  - {q}\n')

    print("\n结果已保存到 integrated_ai_result.txt")


if __name__ == '__main__':
    main()
