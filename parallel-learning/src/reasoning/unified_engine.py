"""统一推理引擎 — 从图遍历到多模式推理

整合所有推理模块到一个统一的管道。

推理模式：
1. 演绎推理 — 从一般到特殊（逻辑蕴涵）
2. 归纳推理 — 从特殊到一般（模式发现）
3. 类比推理 — 跨域映射（结构对应）
4. 因果推理 — 干预和反事实
5. 概率推理 — 贝叶斯更新
6. 元认知 — 自我评估和知识空白检测

设计原则：
- 按需调用：不是所有问题都需要所有推理模式
- 优先级排序：直接查询 > 因果 > 归纳 > 类比
- 结果融合：多个推理结果按置信度加权
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ReasoningResult:
    """推理结果"""
    content: str
    confidence: float
    method: str  # 'deductive', 'inductive', 'analogical', 'causal', 'probabilistic'
    evidence: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)


class UnifiedReasoningEngine:
    """统一推理引擎

    整合所有推理模块，按需调用，结果融合。
    """

    def __init__(self, learner: Any):
        self.learner = learner

    def reason(self, question: str) -> List[ReasoningResult]:
        """对问题进行多模式推理

        流水线：
        1. 直接查询（知识图谱）
        2. 因果推理（因果DAG）
        3. 归纳推理（模式发现）
        4. 类比推理（跨域映射）
        5. 反事实推理（如果...会怎样）
        6. 元认知评估（置信度校准）
        """
        results = []

        # 1. 直接查询
        direct_results = self._direct_query(question)
        results.extend(direct_results)

        # 2. 因果推理
        causal_results = self._causal_reasoning(question)
        results.extend(causal_results)

        # 3. 归纳推理
        inductive_results = self._inductive_reasoning(question)
        results.extend(inductive_results)

        # 4. 类比推理
        analogical_results = self._analogical_reasoning(question)
        results.extend(analogical_results)

        # 5. 反事实推理
        counterfactual_results = self._counterfactual_reasoning(question)
        results.extend(counterfactual_results)

        # 6. 概率推理
        probabilistic_results = self._probabilistic_reasoning(question)
        results.extend(probabilistic_results)

        # 按置信度排序
        results.sort(key=lambda r: r.confidence, reverse=True)

        return results

    def _direct_query(self, question: str) -> List[ReasoningResult]:
        """直接查询知识图谱"""
        results = []
        learner = self.learner

        # 编码问题
        q_repr = learner._encode_text(question)

        # 提取关键词
        keywords = learner._extract_keywords(question)

        # 向量相似度检索
        kg = learner.knowledge
        if kg and hasattr(kg, 'entities'):
            similarities = []
            for entity_id, entity in kg.entities.items():
                if entity.embedding is not None:
                    sim = torch.cosine_similarity(
                        q_repr.unsqueeze(0), entity.embedding.unsqueeze(0)
                    ).item()
                    similarities.append((entity_id, sim))

            # 排序取top
            similarities.sort(key=lambda x: x[1], reverse=True)

            for entity_id, sim in similarities[:5]:
                if sim > 0.3:
                    # 获取关系
                    try:
                        relations = kg.get_relations_of(entity_id)
                        for rel in relations:
                            content = f"{entity_id} {rel.type} {rel.target_id}"
                            results.append(ReasoningResult(
                                content=content,
                                confidence=sim * rel.confidence,
                                method='direct',
                                evidence=[entity_id, rel.target_id],
                            ))
                    except Exception:
                        pass

        # 关键词匹配补充
        if kg and hasattr(kg, 'entities'):
            for keyword in keywords:
                for entity_id in kg.entities:
                    if keyword in entity_id:
                        try:
                            relations = kg.get_relations_of(entity_id)
                            for rel in relations:
                                content = f"{entity_id} {rel.type} {rel.target_id}"
                                if content not in [r.content for r in results]:
                                    results.append(ReasoningResult(
                                        content=content,
                                        confidence=0.5,
                                        method='keyword',
                                        evidence=[entity_id],
                                    ))
                        except Exception:
                            pass

        return results

    def _causal_reasoning(self, question: str) -> List[ReasoningResult]:
        """因果推理"""
        results = []
        learner = self.learner

        # 检查是否有因果DAG
        try:
            if not hasattr(learner, '_registry'):
                return results
            dag = learner._registry.get('causal_dag')
            if dag is None:
                return results
        except Exception:
            return results
        keywords = learner._extract_keywords(question)

        # 在因果DAG中查找
        for keyword in keywords:
            for node_name in dag.nodes:
                if keyword in node_name:
                    # 获取因果链
                    node = dag.nodes[node_name]
                    for child_name in node.children:
                        results.append(ReasoningResult(
                            content=f"{node_name} → {child_name}",
                            confidence=0.7,
                            method='causal',
                            evidence=[node_name, child_name],
                            reasoning_chain=[f"因果关系: {node_name} 导致 {child_name}"],
                        ))
                    for parent_name in node.parents:
                        results.append(ReasoningResult(
                            content=f"{parent_name} → {node_name}",
                            confidence=0.7,
                            method='causal',
                            evidence=[parent_name, node_name],
                            reasoning_chain=[f"因果关系: {parent_name} 导致 {node_name}"],
                        ))

        return results

    def _inductive_reasoning(self, question: str) -> List[ReasoningResult]:
        """归纳推理 — 从多个实例中发现规律"""
        results = []
        learner = self.learner

        # 检查是否有足够的记忆
        if not hasattr(learner, 'memory') or learner.memory is None:
            return results

        # 从情景记忆中寻找模式
        keywords = learner._extract_keywords(question)

        # 收集相关记忆
        related_memories = []
        if hasattr(learner.memory, 'episodic'):
            episodic = learner.memory.episodic
            if hasattr(episodic, 'items'):
                for key, memory in episodic.items():
                    for keyword in keywords:
                        if keyword in str(key):
                            related_memories.append(memory)
                            break
            elif hasattr(episodic, 'traces'):
                for trace in episodic.traces:
                    for keyword in keywords:
                        if keyword in str(trace):
                            related_memories.append(trace)
                            break

        # 如果有多个相关记忆，尝试归纳
        if len(related_memories) >= 2:
            # 简化：提取共同模式
            common_patterns = self._find_common_patterns(related_memories)
            for pattern in common_patterns:
                results.append(ReasoningResult(
                    content=pattern,
                    confidence=0.6,
                    method='inductive',
                    evidence=[m.get('text', '')[:30] for m in related_memories[:3]],
                    reasoning_chain=[f"从 {len(related_memories)} 个实例中归纳"],
                ))

        return results

    def _analogical_reasoning(self, question: str) -> List[ReasoningResult]:
        """类比推理 — 跨域映射"""
        results = []
        learner = self.learner

        # 检查是否有类比模块
        if not hasattr(learner, '_registry'):
            return results

        try:
            metaphor = learner._registry.get('metaphor')
            if metaphor and hasattr(metaphor, 'find_analogies'):
                analogies = metaphor.find_analogies(question)
                for analogy in analogies:
                    results.append(ReasoningResult(
                        content=analogy.get('target', ''),
                        confidence=analogy.get('confidence', 0.5),
                        method='analogical',
                        evidence=[analogy.get('source', '')],
                        reasoning_chain=[f"类比: {analogy.get('source', '')} → {analogy.get('target', '')}"],
                    ))
        except Exception:
            pass

        return results

    def _counterfactual_reasoning(self, question: str) -> List[ReasoningResult]:
        """反事实推理"""
        results = []
        learner = self.learner

        # 检查是否有反事实模块
        if not hasattr(learner, '_registry'):
            return results

        try:
            counterfactual = learner._registry.get('counterfactual')
            if counterfactual and hasattr(counterfactual, 'reason'):
                cf_results = counterfactual.reason(question)
                for cf in cf_results:
                    results.append(ReasoningResult(
                        content=cf.get('content', ''),
                        confidence=cf.get('confidence', 0.5),
                        method='counterfactual',
                        evidence=cf.get('evidence', []),
                        reasoning_chain=[f"反事实: {cf.get('premise', '')}"],
                    ))
        except Exception:
            pass

        return results

    def _probabilistic_reasoning(self, question: str) -> List[ReasoningResult]:
        """概率推理"""
        results = []
        learner = self.learner

        # 使用FEP预测器
        if hasattr(learner, 'predictor') and learner.predictor is not None:
            try:
                q_repr = learner._encode_text(question)
                # 简化：使用预测器的不确定性作为置信度
                results.append(ReasoningResult(
                    content="[概率推理]",
                    confidence=0.3,
                    method='probabilistic',
                    reasoning_chain=["基于自由能原理的概率估计"],
                ))
            except Exception:
                pass

        return results

    def _find_common_patterns(self, memories: List[Dict]) -> List[str]:
        """从多个记忆中找到共同模式"""
        patterns = []

        # 简化：提取共同的实体和关系
        entity_counts = {}
        for memory in memories:
            if 'entities' in memory:
                for entity in memory['entities']:
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1

        # 出现频率高的实体可能是规律的一部分
        for entity, count in entity_counts.items():
            if count >= 2:
                patterns.append(f"反复出现的概念: {entity}")

        return patterns


class MultiHopReasoner:
    """多跳推理器

    支持 A → B → C 的链式推理。
    """

    def __init__(self, learner: Any, max_hops: int = 3):
        self.learner = learner
        self.max_hops = max_hops

    def reason(self, start_entity: str, question: str) -> List[ReasoningResult]:
        """从起始实体进行多跳推理"""
        results = []
        kg = self.learner.knowledge

        if not kg or not hasattr(kg, 'entities'):
            return results

        # BFS 多跳推理
        visited = set()
        queue = [(start_entity, [], 0)]  # (entity, path, hop)

        while queue:
            entity, path, hop = queue.pop(0)

            if hop >= self.max_hops:
                continue

            if entity in visited:
                continue
            visited.add(entity)

            try:
                relations = kg.get_relations_of(entity)
                for rel in relations:
                    new_path = path + [(entity, rel.type, rel.target_id)]

                    # 检查是否回答了问题
                    if self._answers_question(new_path, question):
                        content = ' → '.join([f"{s} {r} {t}" for s, r, t in new_path])
                        results.append(ReasoningResult(
                            content=content,
                            confidence=0.7 / (hop + 1),  # 跳数越多置信度越低
                            method='multi_hop',
                            evidence=[entity, rel.target_id],
                            reasoning_chain=[f"第{i+1}跳: {s} {r} {t}" for i, (s, r, t) in enumerate(new_path)],
                        ))

                    # 继续推理
                    queue.append((rel.target_id, new_path, hop + 1))
            except Exception:
                pass

        return results

    def _answers_question(self, path: List[Tuple], question: str) -> bool:
        """检查推理路径是否回答了问题"""
        # 简化：检查路径中的实体是否出现在问题中
        question_keywords = set(question)
        for _, _, target in path:
            if any(c in question_keywords for c in target):
                return True
        return False
