"""功能性概念系统 — 概念不是标签，是功能性表征

认知科学基础：

    1. Gibson 可供性理论 (Affordance Theory, 1977)：
       概念不是被动描述"是什么"，而是主动表达"能做什么"。
       "杯子" = {可以装水, 可以握住, 可以喝, 易碎}
       这比 "杯子" = [0.2, -0.5, 0.8, ...] 有意义得多。

    2. Carey 快速映射 (Fast Mapping, 1978)：
       儿童只需一次接触就能大致理解新词含义。
       背后的认知偏差：
       - 互斥性：每个对象一个标签（如果已知"狗"，"斑点狗"必须是子类）
       - 整体对象：新标签指向整个物体（不是部分）
       - 形状偏差：按形状分类（不是颜色或材质）

    3. Paivio 双重编码理论 (Dual Coding, 1971)：
       概念同时有感知编码和语言编码。
       "红色" = {视觉体验（红色光波），语言标签（"红色"二字）}
       双重编码的记忆比单一编码强2-3倍。

    4. Barsalou 知觉符号系统 (Perceptual Symbol Systems, 1999)：
       所有概念都是感知运动经验的模拟（simulation）。
       想到"红色"时，视觉皮层真的被部分激活。

核心设计：
    概念节点不再是 {id, vector}，
    而是 {id, vector, perceptual_features, affordances, usage_contexts}。
    每个概念都知道自己"看起来像什么"、"能做什么"、"在什么场景下使用"。
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class FunctionalConceptSystem:
    """功能性概念系统

    使用方式：
        # 初始化（依赖概念空间和核心知识系统）
        fcs = FunctionalConceptSystem(concept_space, core_knowledge)

        # 从感知经验中形成概念
        node = fcs.form_concept(
            label="红色",
            perceptual_features={'type': 'color', 'rgb': (255, 0, 0)},
            affordances=['描述颜色', '区分物体'],
        )

        # 快速映射（一次接触）
        node = fcs.fast_map(
            label="蓝色",
            context={'known_labels': ['红色'], 'object': {...}},
            perceptual_input={'type': 'color', 'rgb': (0, 0, 255)},
        )

        # 接地到感知经验
        fcs.ground_concept("红色", {'visual': tensor, 'objects_with_color': 3})

        # 按可供性检索
        containers = fcs.retrieve_by_affordance("装水")
    """

    def __init__(self, concept_space, core_knowledge, encoder=None):
        """
        Args:
            concept_space: ConceptSpace 实例
            core_knowledge: CoreKnowledgeSystem 实例
            encoder: 可选的文本编码器（用于编码概念向量）
        """
        self.concept_space = concept_space
        self.core_knowledge = core_knowledge
        self.encoder = encoder

        # 快速映射缓冲区（暂存部分理解的概念）
        self.fast_map_buffer: Dict[str, Dict] = {}

        # 统计
        self._stats = {
            'concepts_formed': 0,
            'fast_maps': 0,
            'groundings': 0,
            'affordance_queries': 0,
        }

    def form_concept(self,
                     label: str,
                     perceptual_features: Dict = None,
                     usage_context: Dict = None,
                     affordances: List[str] = None,
                     source: str = 'perception',
                     vector: torch.Tensor = None) -> Optional[object]:
        """形成一个功能性概念

        流程：
        1. 用核心知识先验评估标签
        2. 创建概念向量（编码或随机）
        3. 注册到概念空间（带感知锚点）
        4. 填充功能性字段（感知特征、可供性、使用场景）

        Args:
            label: 概念标签（如"红色"）
            perceptual_features: 感知特征字典
            usage_context: 使用场景
            affordances: 可供性列表
            source: 来源（perception/text/fast_map）
            vector: 可选的预计算向量

        Returns:
            ConceptNode 或 None
        """
        if not label or len(label) < 1:
            return None

        # 1. 先验评估
        prior_score = self.core_knowledge.score_concept_candidate({
            'label': label,
            'frequency': 1,
            'features': {'has_perceptual_binding': perceptual_features is not None},
        })

        # 2. 创建向量
        if vector is not None:
            vec = vector.detach().clone()
        elif self.encoder is not None:
            try:
                with torch.no_grad():
                    vec = self.encoder(label).detach().clone()
            except Exception:
                vec = torch.randn(self.concept_space.dim)
        else:
            vec = torch.randn(self.concept_space.dim)

        vec = F.normalize(vec, p=2, dim=0)

        # 3. 推断可供性（如果未提供）
        if affordances is None:
            affordances = self._infer_affordances(label, perceptual_features or {})

        # 4. 注册到概念空间
        anchor = f'{source}:{label}'
        try:
            node = self.concept_space.register(
                label,
                vector=vec,
                source=source,
                sensory_anchors=[anchor],
            )
        except Exception:
            return None

        # 5. 填充功能性字段
        if perceptual_features:
            node.perceptual_features.update(perceptual_features)
        if affordances:
            for aff in affordances:
                if aff not in node.affordances:
                    node.affordances.append(aff)
        if usage_context:
            node.usage_contexts.append(usage_context)

        self._stats['concepts_formed'] += 1
        return node

    def fast_map(self, label: str, context: Dict,
                 perceptual_input: Dict = None) -> Optional[object]:
        """快速映射 — 听一次就大致理解

        实现 Carey & Bartlett (1978) 的快速映射机制。

        认知偏差驱动：
        1. 互斥性偏差：如果已知标签 → 推断新标签是新类别
        2. 整体对象偏差：标签指向整个对象
        3. 形状偏差：按形状分类

        Args:
            label: 新标签
            context: 上下文 {'known_labels': [...], 'object': {...}, ...}
            perceptual_input: 感知输入

        Returns:
            概念节点（可能是部分理解）
        """
        self._stats['fast_maps'] += 1
        biases = self.core_knowledge.provide_learning_biases()

        # 1. 检查是否与已知概念有互斥关系
        known_labels = context.get('known_labels', [])
        related_known = self._find_related_known(label, known_labels)

        if related_known:
            # 互斥性偏差：这是已知概念的一个变体或属性
            concept = self._form_by_exclusion(label, related_known, biases, context, perceptual_input)
        else:
            # 全新概念：整体对象偏差
            concept = self._form_by_whole_object(label, perceptual_input, biases)

        if concept:
            # 暂存到快速映射缓冲区（部分理解，待后续巩固）
            self.fast_map_buffer[label] = {
                'node_id': concept.id,
                'context': context,
                'confidence': 0.5,  # 快速映射初始置信度较低
            }

        return concept

    def ground_concept(self, concept_id: str,
                       perceptual_experience: Dict) -> bool:
        """将概念接地到感知经验

        让概念不再是抽象标签，而是与具体感知体验绑定。
        每次接地都会更新概念的感知特征和感知编码。

        Args:
            concept_id: 概念ID
            perceptual_experience: {
                'visual': torch.Tensor,     # 视觉特征
                'audio': torch.Tensor,      # 听觉特征
                'haptic': torch.Tensor,     # 触觉特征
                'description': str,         # 文本描述
                'objects': List[str],       # 关联物体
            }

        Returns:
            是否成功接地
        """
        if concept_id not in self.concept_space.concepts:
            return False

        node = self.concept_space.concepts[concept_id]
        self._stats['groundings'] += 1

        # 1. 更新感知特征
        for key, value in perceptual_experience.items():
            if key != 'description' and value is not None:
                if isinstance(value, torch.Tensor):
                    node.perceptual_features[key] = value.detach().cpu().tolist()
                else:
                    node.perceptual_features[key] = value

        # 2. 更新感知锚点
        exp_hash = hash(str(sorted(
            (k, str(v)[:50]) for k, v in perceptual_experience.items()
        )))
        anchor = f'exp:{exp_hash}'
        if anchor not in node.sensory_anchors:
            node.sensory_anchors.append(anchor)

        # 3. 融合感知编码到概念向量
        if 'visual' in perceptual_experience:
            visual = perceptual_experience['visual']
            if isinstance(visual, torch.Tensor):
                visual = visual.to(node.vector.device).flatten()
                if visual.shape[0] == node.vector.shape[0]:
                    # 感知编码融合（70%语言 + 30%感知）
                    node.vector = F.normalize(
                        0.7 * node.vector + 0.3 * F.normalize(visual, p=2, dim=0),
                        p=2, dim=0
                    )

        # 4. 增强强度
        node.strength = min(node.strength + 0.05, 2.0)

        return True

    def retrieve_by_affordance(self, action_goal: str,
                                top_k: int = 10) -> List[Tuple[str, float]]:
        """根据可供性检索概念

        "能用什么来装水？" → [杯子, 碗, 桶, ...]

        Args:
            action_goal: 目标动作/功能描述
            top_k: 返回数量

        Returns:
            [(concept_id, relevance_score), ...]
        """
        self._stats['affordance_queries'] += 1
        results = []

        for cid, node in self.concept_space.concepts.items():
            # 精确匹配
            if action_goal in node.affordances:
                results.append((cid, 1.0))
                continue

            # 模糊匹配（语义相似度）
            for aff in node.affordances:
                # 简单的关键词重叠
                overlap = len(set(action_goal) & set(aff))
                if overlap > 0:
                    score = overlap / max(len(action_goal), len(aff))
                    results.append((cid, score * 0.5))

        # 去重并排序
        seen = {}
        for cid, score in results:
            if cid not in seen or seen[cid] < score:
                seen[cid] = score

        sorted_results = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def retrieve_by_context(self, context: Dict,
                             top_k: int = 10) -> List[Tuple[str, float]]:
        """根据使用场景检索概念

        "厨房里有什么？" → [锅, 碗, 刀, ...]
        """
        context_keys = set()
        for key, value in context.items():
            if isinstance(value, str):
                context_keys.add(value)
            elif isinstance(value, list):
                context_keys.update(str(v) for v in value)

        results = []
        for cid, node in self.concept_space.concepts.items():
            best_score = 0.0
            for uc in node.usage_contexts:
                uc_keys = set(str(v) for v in uc.values())
                overlap = len(context_keys & uc_keys)
                if overlap > 0:
                    score = overlap / max(len(context_keys), 1)
                    best_score = max(best_score, score)

            # 感知特征匹配
            if node.perceptual_features:
                for key in context_keys:
                    if key in str(node.perceptual_features):
                        best_score = max(best_score, 0.3)

            if best_score > 0:
                results.append((cid, best_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_concept_profile(self, concept_id: str) -> Dict:
        """获取概念的完整档案

        返回一个概念的所有信息：向量、感知特征、可供性、使用场景、关系等。
        """
        if concept_id not in self.concept_space.concepts:
            return {}

        node = self.concept_space.concepts[concept_id]
        related = self.concept_space.get_related(concept_id, top_k=5)

        return {
            'id': node.id,
            'source': node.source,
            'frequency': node.frequency,
            'strength': node.strength,
            'perceptual_features': node.perceptual_features,
            'affordances': node.affordances,
            'usage_contexts': len(node.usage_contexts),
            'sensory_anchors': len(node.sensory_anchors),
            'related_concepts': [(rid, round(score, 3)) for rid, score in related],
        }

    # ===== 内部方法 =====

    def _infer_affordances(self, label: str, features: Dict) -> List[str]:
        """从感知特征推断可供性"""
        affordances = []
        feat_type = features.get('type', '')

        # 基于类型的可供性推断
        type_affordances = {
            'color': ['描述颜色', '区分物体', '识别属性'],
            'shape': ['描述形状', '分类物体', '识别几何', '判断空间'],
            'material': ['描述材质', '判断属性', '评估触感'],
            'size': ['描述大小', '比较物体', '评估空间需求'],
            'discipline': ['分类知识', '组织概念', '描述领域'],
            'action': ['描述过程', '预测结果', '规划步骤'],
        }

        if feat_type in type_affordances:
            affordances.extend(type_affordances[feat_type])

        # 基于标签的启发式推断
        label_affordances = {
            '红': ['标记颜色', '表示警告'],
            '蓝': ['标记颜色', '表示冷静'],
            '绿': ['标记颜色', '表示生长'],
            '圆': ['描述形状', '判断旋转'],
            '方': ['描述形状', '判断堆叠'],
            '大': ['比较大小', '评估空间'],
            '小': ['比较大小', '评估精度'],
        }

        for key, affs in label_affordances.items():
            if key in label:
                for aff in affs:
                    if aff not in affordances:
                        affordances.append(aff)

        return affordances

    def _find_related_known(self, label: str, known_labels: List[str]) -> List[str]:
        """找到与标签相关的已知概念"""
        related = []
        for known in known_labels:
            # 子串关系
            if label in known or known in label:
                related.append(known)
            # 共享字符
            overlap = len(set(label) & set(known))
            if overlap >= max(len(label), len(known)) * 0.5:
                if known not in related:
                    related.append(known)
        return related

    def _form_by_exclusion(self, label, related_known, biases, context, perceptual_input):
        """通过互斥性偏差形成概念"""
        # 新标签是已知概念的变体/子类
        # 从已知概念继承部分向量，但添加独特性
        related_node = None
        for rk in related_known:
            if rk in self.concept_space.concepts:
                related_node = self.concept_space.concepts[rk]
                break

        if related_node:
            # 继承部分向量 + 随机偏移
            base_vec = related_node.vector.clone()
            offset = torch.randn_like(base_vec) * 0.3
            vec = F.normalize(base_vec + offset, p=2, dim=0)

            # 继承部分可供性
            inherited_affordances = list(related_node.affordances[:2])
        else:
            vec = None
            inherited_affordances = []

        return self.form_concept(
            label=label,
            perceptual_features=perceptual_input,
            usage_context={'formed_by': 'exclusion', 'related_to': related_known},
            affordances=inherited_affordances,
            source='fast_map_exclusion',
            vector=vec,
        )

    def _form_by_whole_object(self, label, perceptual_input, biases):
        """通过整体对象偏差形成概念"""
        return self.form_concept(
            label=label,
            perceptual_features=perceptual_input,
            usage_context={'formed_by': 'whole_object'},
            source='fast_map_whole_object',
        )

    def get_stats(self) -> Dict:
        """获取功能性概念系统统计"""
        total_concepts = len(self.concept_space.concepts)
        with_perceptual = sum(
            1 for n in self.concept_space.concepts.values()
            if n.perceptual_features
        )
        with_affordances = sum(
            1 for n in self.concept_space.concepts.values()
            if n.affordances
        )

        return {
            **self._stats,
            'total_concepts': total_concepts,
            'with_perceptual_features': with_perceptual,
            'with_affordances': with_affordances,
            'fast_map_buffer_size': len(self.fast_map_buffer),
            'grounding_rate': (self._stats['groundings'] / max(total_concepts, 1)),
        }
