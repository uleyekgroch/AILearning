"""概念形成器 — 从具体经验中归纳抽象概念

核心思想：像婴儿一样从多个具体例子中归纳出抽象概念。
看10个球 → 形成"球"的概念 → 能识别新的球。

概念形成的三个层次：
1. 感知层：聚类相似经验 → 原型概念
2. 抽象层：从多个原型中提取共同特征 → 抽象概念
3. 层次层：建立概念之间的 is-a / part-of 关系 → 概念层次

与现有系统的区别：
- 现有：预定义概念（OXFORD_3000）
- 新系统：从经验中自动发现和形成概念
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class Concept:
    """概念"""
    id: str
    name: str
    prototype: Dict[str, float]    # 原型特征
    examples: List[Dict] = field(default_factory=list)
    parent: Optional[str] = None   # 上位概念
    children: List[str] = field(default_factory=list)
    confidence: float = 0.5
    example_count: int = 0

    def similarity(self, features: Dict[str, float]) -> float:
        """计算特征与概念原型的相似度"""
        common = set(self.prototype.keys()) & set(features.keys())
        if not common:
            return 0.0
        # 只比较数值特征
        numeric_keys = [k for k in common
                       if isinstance(self.prototype[k], (int, float))
                       and isinstance(features[k], (int, float))]
        if not numeric_keys:
            # 比较字符串特征的相等率
            str_keys = [k for k in common if isinstance(self.prototype[k], str)]
            if not str_keys:
                return 0.0
            matches = sum(1 for k in str_keys if self.prototype[k] == features[k])
            return matches / len(str_keys)
        dot = sum(self.prototype[k] * features[k] for k in numeric_keys)
        norm1 = math.sqrt(sum(v**2 for v in self.prototype.values() if isinstance(v, (int, float))))
        norm2 = math.sqrt(sum(v**2 for v in features.values() if isinstance(v, (int, float))))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


class ConceptFormer:
    """概念形成器 — 从经验中归纳抽象概念"""

    def __init__(self, similarity_threshold: float = 0.7,
                 min_examples: int = 3):
        self.similarity_threshold = similarity_threshold
        self.min_examples = min_examples

        # 概念库
        self.concepts: Dict[str, Concept] = {}
        self._concept_counter = 0

        # 概念层次
        self.hierarchy: Dict[str, List[str]] = defaultdict(list)

        # 统计
        self.total_formations = 0
        self.total_merges = 0

    def observe(self, features: Dict[str, float],
                context: Dict[str, Any] = None) -> str:
        """观察一个新经验，尝试形成或更新概念

        Args:
            features: 观察到的特征
            context: 上下文信息

        Returns:
            匹配或新创建的概念ID
        """
        # 1. 找到最相似的已有概念
        best_concept, best_sim = self._find_best_match(features)

        if best_concept and best_sim > self.similarity_threshold:
            # 2a. 更新已有概念
            self._update_concept(best_concept, features)
            return best_concept.id
        else:
            # 2b. 创建新概念
            concept = self._create_concept(features, context)
            return concept.id

    def abstract(self, concept_ids: List[str]) -> Optional[Concept]:
        """从多个具体概念中抽象出一般概念

        归纳推理：多个球 → "球"的概念
        """
        concepts = [self.concepts[cid] for cid in concept_ids if cid in self.concepts]
        if len(concepts) < 2:
            return None

        # 提取共同特征
        common_features = self._extract_common_features(concepts)
        if not common_features:
            return None

        # 检查是否已有抽象概念
        for concept in self.concepts.values():
            if concept.example_count >= self.min_examples:
                sim = concept.similarity(common_features)
                if sim > 0.8:
                    # 已有类似抽象概念，合并
                    for c in concepts:
                        if c.id not in concept.children:
                            concept.children.append(c.id)
                            c.parent = concept.id
                    self.total_merges += 1
                    return concept

        # 创建新的抽象概念
        self._concept_counter += 1
        abstract_concept = Concept(
            id=f"abstract_{self._concept_counter}",
            name=f"abstract_concept_{self._concept_counter}",
            prototype=common_features,
            example_count=len(concepts),
            confidence=0.6,
        )

        # 建立层次关系
        for c in concepts:
            abstract_concept.children.append(c.id)
            c.parent = abstract_concept.id

        self.concepts[abstract_concept.id] = abstract_concept
        self.total_formations += 1

        return abstract_concept

    def specialize(self, concept_id: str,
                   specific_features: Dict[str, float]) -> Optional[Concept]:
        """从抽象概念中特化出具体概念

        演绎推理："球" + 红色 → "红球"
        """
        parent = self.concepts.get(concept_id)
        if not parent:
            return None

        # 合并父概念特征和特化特征
        specialized = dict(parent.prototype)
        specialized.update(specific_features)

        self._concept_counter += 1
        child = Concept(
            id=f"special_{self._concept_counter}",
            name=f"specialized_{parent.name}",
            prototype=specialized,
            parent=parent.id,
            confidence=0.5,
        )

        parent.children.append(child.id)
        self.concepts[child.id] = child
        self.total_formations += 1

        return child

    def get_concept_tree(self) -> Dict:
        """获取概念层次树"""
        roots = [c for c in self.concepts.values() if c.parent is None]
        tree = {}
        for root in roots:
            tree[root.id] = self._build_subtree(root)
        return tree

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_concepts': len(self.concepts),
            'total_formations': self.total_formations,
            'total_merges': self.total_merges,
            'root_concepts': sum(1 for c in self.concepts.values() if c.parent is None),
            'avg_examples': sum(c.example_count for c in self.concepts.values()) / max(len(self.concepts), 1),
        }

    # ── 内部方法 ──────────────────────────────────────────────

    def _find_best_match(self, features: Dict[str, float]) -> Tuple[Optional[Concept], float]:
        """找到最相似的已有概念"""
        best = None
        best_sim = 0.0

        for concept in self.concepts.values():
            sim = concept.similarity(features)
            if sim > best_sim:
                best_sim = sim
                best = concept

        return best, best_sim

    def _create_concept(self, features: Dict[str, float],
                        context: Dict = None) -> Concept:
        """创建新概念"""
        self._concept_counter += 1
        concept = Concept(
            id=f"concept_{self._concept_counter}",
            name=context.get('name', f'concept_{self._concept_counter}') if context else f'concept_{self._concept_counter}',
            prototype=dict(features),
            examples=[features],
            example_count=1,
            confidence=0.3,
        )
        self.concepts[concept.id] = concept
        self.total_formations += 1
        return concept

    def _update_concept(self, concept: Concept, features: Dict[str, float]):
        """更新已有概念的原型"""
        concept.example_count += 1
        concept.examples.append(features)
        if len(concept.examples) > 10:
            concept.examples = concept.examples[-10:]

        # 更新原型（滑动平均，只处理数值）
        n = concept.example_count
        for key, value in features.items():
            if key in concept.prototype:
                if isinstance(value, (int, float)) and isinstance(concept.prototype[key], (int, float)):
                    concept.prototype[key] = (
                        concept.prototype[key] * (n-1) / n + value / n
                    )
                elif isinstance(value, str):
                    # 字符串：保留最常见的值
                    pass
            else:
                concept.prototype[key] = value

        # 增加置信度
        concept.confidence = min(1.0, concept.confidence + 0.05)

    def _extract_common_features(self, concepts: List[Concept]) -> Dict[str, float]:
        """提取多个概念的共同特征"""
        if not concepts:
            return {}

        # 找到所有概念都有的特征
        common_keys = set(concepts[0].prototype.keys())
        for c in concepts[1:]:
            common_keys &= set(c.prototype.keys())

        if not common_keys:
            return {}

        # 计算共同特征（数值取平均，字符串取众数）
        common = {}
        for key in common_keys:
            values = [c.prototype[key] for c in concepts]
            if all(isinstance(v, (int, float)) for v in values):
                common[key] = sum(values) / len(values)
            else:
                # 字符串：取最常见的值
                from collections import Counter
                counts = Counter(str(v) for v in values)
                common[key] = counts.most_common(1)[0][0]

        return common

    def _build_subtree(self, concept: Concept) -> Dict:
        """递归构建子树"""
        tree = {
            'id': concept.id,
            'name': concept.name,
            'confidence': concept.confidence,
            'examples': concept.example_count,
            'children': {},
        }
        for child_id in concept.children:
            child = self.concepts.get(child_id)
            if child:
                tree['children'][child_id] = self._build_subtree(child)
        return tree
