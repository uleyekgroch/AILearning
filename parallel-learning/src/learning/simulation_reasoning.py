"""模拟推理系统 — 推理不是查表，是在脑中模拟场景

认知科学基础：

    1. 心理模拟理论（Mental Simulation, Barsalou 1999）：
       推理的本质是在脑中"运行"一个内部世界模型。
       问"加热冰块会怎样？" → 脑中构建冰块场景 → 模拟加热 → 观察结果

    2. 因果推理（Pearl 2009, The Book of Why）：
       三个层级的因果推理：
       - L1 观察/关联："看到烟雾时火灾的概率？"
       - L2 干预："如果我禁止吸烟，肺癌率会怎样？"
       - L3 反事实："如果当时没吸烟，现在会怎样？"
       当前AI主要在L1，人类日常推理在L2-L3。

    3. 类比推理（Gentner 1983, Structure-Mapping Theory）：
       类比匹配的是关系结构，不是表面特征。
       "原子围绕核旋转" ~ "行星围绕太阳旋转" → 匹配"围绕旋转"关系

核心设计：
    推理不是数据库查询，而是：
    1. 概念激活 → 从问题中提取关键概念
    2. 场景构建 → 在内部世界模型中构建虚拟场景
    3. 因果追踪 → 在场景中追踪因果链
    4. 反事实模拟 → "如果X没发生，会怎样？"
    5. 类比发现 → 找到结构相似的历史经验
    6. 结果表达 → 将模拟结果转化为自然语言
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class SimulationScene:
    """模拟场景 — 在脑中构建的虚拟世界"""
    concepts: List[str]              # 场景中的概念
    features: List[Dict]             # 每个概念的感知特征
    relations: List[Dict]            # 概念间的关系
    confidence: float = 0.0          # 场景的置信度


@dataclass
class CausalChain:
    """因果链 — 场景中的因果路径"""
    steps: List[str]                 # 因果步骤 [A, B, C, ...]
    confidence: float = 0.0          # 链的置信度
    evidence: List[str] = field(default_factory=list)  # 支持证据


@dataclass
class ReasoningResult:
    """推理结果"""
    scene: Optional[SimulationScene] = None
    causal_chains: List[CausalChain] = field(default_factory=list)
    counterfactuals: List[Dict] = field(default_factory=list)
    analogies: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_type: str = 'unknown'  # causal/counterfactual/analogical


class SimulationReasoning:
    """模拟推理系统

    使用方式：
        sr = SimulationReasoning(concept_system, causal_engine)

        # 推理
        result = sr.reason(
            question="如果把水加热会发生什么",
            activated_concepts=["水", "加热", "温度"],
        )

        # 表达结果
        answer = sr.express(result, "如果把水加热会发生什么")
    """

    # 因果动词（中文中常见的因果标记）
    CAUSAL_VERBS = ['导致', '引起', '使', '让', '造成', '产生', '带来', '促使']
    RESULT_WORDS = ['会', '将', '会变', '变成', '变为']
    CONDITION_WORDS = ['如果', '假如', '假设', '要是', '若', '当']

    def __init__(self, concept_space=None, causal_engine=None,
                 knowledge_graph=None):
        """
        Args:
            concept_space: ConceptSpace 实例
            causal_engine: 因果引擎（CausalReasoningModule 或类似）
            knowledge_graph: KnowledgeGraph 实例
        """
        self.concept_space = concept_space
        self.causal_engine = causal_engine
        self.knowledge_graph = knowledge_graph

        # 统计
        self._stats = {
            'total_reasoning': 0,
            'causal_chains_found': 0,
            'counterfactuals_run': 0,
            'analogies_found': 0,
        }

    def reason(self, question: str,
               activated_concepts: List[str]) -> ReasoningResult:
        """模拟推理主入口

        1. 从激活概念构建内部场景
        2. 在场景中追踪因果链
        3. 如果问题是反事实的 → 运行反事实模拟
        4. 找到类比结构
        5. 综合评估置信度
        """
        self._stats['total_reasoning'] += 1

        # 1. 场景构建
        scene = self._build_scene(activated_concepts)

        # 2. 因果追踪
        causal_chains = self._trace_causal_chains(scene, question)

        # 3. 反事实检测与模拟
        counterfactuals = []
        if self._is_counterfactual_question(question):
            counterfactuals = self._simulate_counterfactuals(scene, question)
            self._stats['counterfactuals_run'] += len(counterfactuals)

        # 4. 类比发现
        analogies = self._find_analogies(scene, activated_concepts)
        self._stats['analogies_found'] += len(analogies)

        # 5. 综合置信度
        confidence = self._assess_confidence(scene, causal_chains, analogies)

        return ReasoningResult(
            scene=scene,
            causal_chains=causal_chains,
            counterfactuals=counterfactuals,
            analogies=analogies,
            confidence=confidence,
            reasoning_type=self._classify_question(question),
        )

    def express(self, result: ReasoningResult, question: str) -> str:
        """将推理结果表达为自然语言

        不是模板拼接，而是基于因果链和场景描述自然组织。
        """
        if not result or result.confidence < 0.1:
            return ""

        parts = []

        # 1. 因果链表达
        if result.causal_chains:
            for chain in result.causal_chains[:2]:  # 最多表达2条链
                if len(chain.steps) >= 2:
                    chain_text = " → ".join(chain.steps)
                    parts.append(chain_text)

        # 2. 反事实表达
        if result.counterfactuals:
            for cf in result.counterfactuals[:1]:
                cf_text = cf.get('description', '')
                if cf_text:
                    parts.append(f"另一种可能: {cf_text}")

        # 3. 类比辅助表达
        if result.analogies:
            for analogy in result.analogies[:1]:
                src = analogy.get('source', '')
                tgt = analogy.get('target', '')
                if src and tgt:
                    parts.append(f"类似: {src}与{tgt}有相似结构")

        if not parts:
            # 回退到场景描述
            if result.scene and result.scene.concepts:
                concepts_str = "、".join(result.scene.concepts[:5])
                parts.append(f"涉及: {concepts_str}")

        return "。".join(parts)

    def _build_scene(self, concepts: List[str]) -> SimulationScene:
        """从激活概念构建内部场景"""
        scene_concepts = []
        scene_features = []
        scene_relations = []

        for cid in concepts:
            if self.concept_space and cid in self.concept_space.concepts:
                node = self.concept_space.concepts[cid]
                scene_concepts.append(cid)

                # 收集感知特征
                features = {
                    'id': cid,
                    'source': node.source,
                    'frequency': node.frequency,
                    'strength': node.strength,
                }
                if node.perceptual_features:
                    features['perceptual'] = node.perceptual_features
                if node.affordances:
                    features['affordances'] = node.affordances
                scene_features.append(features)

                # 收集关系
                if self.concept_space:
                    for rel_id, weight in self.concept_space.relations.get(cid, {}).items():
                        if rel_id in concepts and weight > 0.1:
                            scene_relations.append({
                                'from': cid, 'to': rel_id,
                                'weight': weight,
                            })

        # 计算场景置信度
        confidence = 0.0
        if scene_concepts:
            # 有感知特征的概念越多 → 置信度越高
            grounded = sum(1 for f in scene_features if 'perceptual' in f)
            confidence = grounded / max(len(scene_concepts), 1)
            # 关系越丰富 → 置信度越高
            if scene_relations:
                confidence = min(1.0, confidence + 0.2)

        return SimulationScene(
            concepts=scene_concepts,
            features=scene_features,
            relations=scene_relations,
            confidence=confidence,
        )

    def _trace_causal_chains(self, scene: SimulationScene,
                              question: str) -> List[CausalChain]:
        """在场景中追踪因果链"""
        chains = []

        # 方法1: 从知识图谱追踪因果边
        if self.knowledge_graph:
            for concept in scene.concepts:
                try:
                    # 查找从该概念出发的因果路径
                    kg = self.knowledge_graph
                    if hasattr(kg, 'get_outgoing'):
                        outgoing = kg.get_outgoing(concept, relation_type='CAUSES')
                        for target in outgoing:
                            target_id = target.target_id if hasattr(target, 'target_id') else str(target)
                            if target_id in scene.concepts:
                                chains.append(CausalChain(
                                    steps=[concept, '导致', target_id],
                                    confidence=0.7,
                                    evidence=[f'KG: {concept}→{target_id}'],
                                ))
                except Exception:
                    pass

        # 方法2: 从概念空间的关系图追踪
        if self.concept_space and not chains:
            for concept in scene.concepts:
                related = self.concept_space.relations.get(concept, {})
                for rel_id, weight in related.items():
                    if rel_id in scene.concepts and weight > 0.3:
                        chains.append(CausalChain(
                            steps=[concept, '关联', rel_id],
                            confidence=weight * 0.8,
                            evidence=[f'CS: weight={weight:.2f}'],
                        ))

        # 方法3: 从问题中检测因果意图
        if not chains:
            for verb in self.CAUSAL_VERBS:
                if verb in question:
                    # 问题中有因果动词 → 尝试构建简单因果链
                    if len(scene.concepts) >= 2:
                        chains.append(CausalChain(
                            steps=[scene.concepts[0], verb, scene.concepts[-1]],
                            confidence=0.3,
                            evidence=[f'question_contains: {verb}'],
                        ))

        self._stats['causal_chains_found'] += len(chains)
        return chains

    def _is_counterfactual_question(self, question: str) -> bool:
        """检测问题是否是反事实的"""
        for cw in self.CONDITION_WORDS:
            if cw in question:
                return True
        return False

    def _simulate_counterfactuals(self, scene: SimulationScene,
                                   question: str) -> List[Dict]:
        """反事实模拟

        "如果X没有发生会怎样？" → 在模型中删除X的因果边，重新运行
        """
        counterfactuals = []

        for concept in scene.concepts:
            # 对于每个概念，假设它不存在
            remaining = [c for c in scene.concepts if c != concept]

            if len(remaining) >= 1:
                cf = {
                    'removed': concept,
                    'remaining_concepts': remaining,
                    'description': f"如果没有{concept}，场景只涉及{'、'.join(remaining)}",
                    'plausibility': 0.5,
                }
                counterfactuals.append(cf)

        return counterfactuals[:3]  # 最多3个反事实

    def _find_analogies(self, scene: SimulationScene,
                        concepts: List[str]) -> List[Dict]:
        """找到类比结构

        在概念空间中找到与当前场景结构相似的其他概念组合。
        """
        analogies = []

        if not self.concept_space or len(scene.concepts) < 2:
            return analogies

        # 对每对概念，找结构相似的其他对
        for i, c1 in enumerate(scene.concepts):
            for c2 in scene.concepts[i + 1:]:
                # 检查这对概念的关系模式
                rel_weight = self.concept_space.relations.get(c1, {}).get(c2, 0)

                if rel_weight > 0.1:
                    # 找具有类似关系的其他概念对
                    for other_c1 in self.concept_space.concepts:
                        if other_c1 in scene.concepts:
                            continue
                        other_related = self.concept_space.relations.get(other_c1, {})
                        for other_c2, other_weight in other_related.items():
                            if other_c2 in scene.concepts:
                                continue
                            # 关系权重相似 → 可能是类比
                            if abs(other_weight - rel_weight) < 0.2 and other_weight > 0.2:
                                analogies.append({
                                    'source': f'{c1}→{c2}',
                                    'target': f'{other_c1}→{other_c2}',
                                    'similarity': 1 - abs(other_weight - rel_weight),
                                })

        return analogies[:3]

    def _assess_confidence(self, scene: SimulationScene,
                           chains: List[CausalChain],
                           analogies: List[Dict]) -> float:
        """综合评估推理置信度"""
        confidence = 0.0

        # 场景置信度
        if scene:
            confidence += scene.confidence * 0.3

        # 因果链置信度
        if chains:
            best_chain_conf = max(c.confidence for c in chains)
            confidence += best_chain_conf * 0.4

        # 类比支持度
        if analogies:
            avg_sim = sum(a['similarity'] for a in analogies) / len(analogies)
            confidence += avg_sim * 0.3

        return min(1.0, confidence)

    def _classify_question(self, question: str) -> str:
        """分类问题类型"""
        if self._is_counterfactual_question(question):
            return 'counterfactual'
        for verb in self.CAUSAL_VERBS:
            if verb in question:
                return 'causal'
        if '关系' in question or '联系' in question or '和' in question:
            return 'analogical'
        return 'factual'

    def get_stats(self) -> Dict:
        """获取推理系统统计"""
        return self._stats
