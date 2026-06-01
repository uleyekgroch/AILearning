"""
概念形成系统 - 从第一性原理出发

人类概念形成的本质：
1. 从具体到抽象
2. 从简单到复杂
3. 通过类比和隐喻
4. 概念之间形成层次结构
"""

import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """概念"""
    concept_id: str
    name: str
    features: Dict[str, float]  # 特征向量
    examples: List[Any] = field(default_factory=list)
    parent_concepts: Set[str] = field(default_factory=set)
    child_concepts: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    abstraction_level: int = 0  # 抽象层次


@dataclass
class Analogy:
    """类比"""
    source_concept: str
    target_concept: str
    similarity: float
    mapping: Dict[str, str]  # 特征映射
    confidence: float


class ConceptHierarchy:
    """概念层次结构

    实现从具体到抽象的概念层次：
    - 具体概念（如：狗、猫）
    - 中层概念（如：动物）
    - 抽象概念（如：生物）
    """

    def __init__(self):
        """初始化概念层次"""
        self.concepts: Dict[str, Concept] = {}
        self.hierarchy: Dict[str, Set[str]] = {}  # parent -> children

    def add_concept(self, concept: Concept) -> None:
        """
        添加概念

        Args:
            concept: 概念对象
        """
        self.concepts[concept.concept_id] = concept

        # 更新层次结构
        if concept.parent_concepts:
            for parent_id in concept.parent_concepts:
                if parent_id not in self.hierarchy:
                    self.hierarchy[parent_id] = set()
                self.hierarchy[parent_id].add(concept.concept_id)

    def get_children(self, concept_id: str) -> List[Concept]:
        """
        获取子概念

        Args:
            concept_id: 概念ID

        Returns:
            子概念列表
        """
        child_ids = self.hierarchy.get(concept_id, set())
        return [self.concepts[cid] for cid in child_ids if cid in self.concepts]

    def get_parents(self, concept_id: str) -> List[Concept]:
        """
        获取父概念

        Args:
            concept_id: 概念ID

        Returns:
            父概念列表
        """
        concept = self.concepts.get(concept_id)
        if not concept:
            return []

        return [self.concepts[pid] for pid in concept.parent_concepts if pid in self.concepts]

    def get_all_ancestors(self, concept_id: str) -> List[Concept]:
        """
        获取所有祖先概念

        Args:
            concept_id: 概念ID

        Returns:
            祖先概念列表
        """
        ancestors = []
        visited = set()

        def dfs(cid):
            if cid in visited:
                return
            visited.add(cid)

            concept = self.concepts.get(cid)
            if concept:
                ancestors.append(concept)
                for parent_id in concept.parent_concepts:
                    dfs(parent_id)

        dfs(concept_id)
        return ancestors

    def get_abstraction_level(self, concept_id: str) -> int:
        """
        获取抽象层次

        Args:
            concept_id: 概念ID

        Returns:
            抽象层次
        """
        ancestors = self.get_all_ancestors(concept_id)
        return len(ancestors)


class FeatureExtractor:
    """特征提取器

    从具体实例中提取特征，形成概念
    """

    def __init__(self, feature_dim: int = 16):
        """
        初始化特征提取器

        Args:
            feature_dim: 特征维度
        """
        self.feature_dim = feature_dim

    def extract(self, example: Any) -> Dict[str, float]:
        """
        提取特征

        Args:
            example: 实例

        Returns:
            特征字典
        """
        # 简化实现：基于哈希生成特征
        if isinstance(example, str):
            features = {}
            for i, char in enumerate(example[:self.feature_dim]):
                features[f"f_{i}"] = ord(char) / 255.0
            return features
        elif isinstance(example, (int, float)):
            return {"f_0": float(example)}
        else:
            return {"f_0": 0.0}

    def compute_similarity(self, features1: Dict[str, float],
                          features2: Dict[str, float]) -> float:
        """
        计算特征相似度

        Args:
            features1: 特征1
            features2: 特征2

        Returns:
            相似度 [0, 1]
        """
        # 获取共同特征
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0

        # 计算余弦相似度
        vec1 = np.array([features1[k] for k in common_keys])
        vec2 = np.array([features2[k] for k in common_keys])

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


class AnalogyEngine:
    """类比引擎

    实现结构映射类比：
    - 发现源概念和目标概念之间的相似性
    - 建立特征映射
    - 推断新知识
    """

    def __init__(self, feature_extractor: FeatureExtractor):
        """
        初始化类比引擎

        Args:
            feature_extractor: 特征提取器
        """
        self.feature_extractor = feature_extractor

    def find_analogy(self, source: Concept, target: Concept,
                    threshold: float = 0.5) -> Optional[Analogy]:
        """
        寻找类比

        Args:
            source: 源概念
            target: 目标概念
            threshold: 相似度阈值

        Returns:
            类比对象，如果不存在返回None
        """
        # 计算特征相似度
        similarity = self.feature_extractor.compute_similarity(
            source.features, target.features
        )

        if similarity < threshold:
            return None

        # 建立特征映射
        mapping = self._build_mapping(source.features, target.features)

        return Analogy(
            source_concept=source.concept_id,
            target_concept=target.concept_id,
            similarity=similarity,
            mapping=mapping,
            confidence=similarity
        )

    def _build_mapping(self, features1: Dict[str, float],
                      features2: Dict[str, float]) -> Dict[str, str]:
        """
        建立特征映射

        Args:
            features1: 特征1
            features2: 特征2

        Returns:
            映射字典
        """
        mapping = {}

        # 简化实现：按相似度匹配
        sorted1 = sorted(features1.items(), key=lambda x: x[1], reverse=True)
        sorted2 = sorted(features2.items(), key=lambda x: x[1], reverse=True)

        for (k1, v1), (k2, v2) in zip(sorted1, sorted2):
            if abs(v1 - v2) < 0.3:  # 相似特征
                mapping[k1] = k2

        return mapping


class ConceptFormation:
    """概念形成系统

    实现人类概念形成的核心机制：
    1. 从具体实例中提取特征
    2. 形成概念
    3. 建立概念层次
    4. 通过类比扩展知识
    """

    def __init__(self, feature_dim: int = 16):
        """
        初始化概念形成系统

        Args:
            feature_dim: 特征维度
        """
        self.feature_dim = feature_dim

        # 概念层次
        self.hierarchy = ConceptHierarchy()

        # 特征提取器
        self.feature_extractor = FeatureExtractor(feature_dim)

        # 类比引擎
        self.analogy_engine = AnalogyEngine(self.feature_extractor)

        # 统计
        self.stats = {
            'total_concepts': 0,
            'total_analogies': 0,
            'abstraction_levels': {},
        }

    def form_concept(self, name: str, examples: List[Any],
                    parent_concepts: Set[str] = None) -> Concept:
        """
        形成概念

        Args:
            name: 概念名称
            examples: 实例列表
            parent_concepts: 父概念ID集合

        Returns:
            概念对象
        """
        # 生成概念ID
        concept_id = f"concept_{name}_{datetime.now().timestamp()}"

        # 从实例中提取特征
        all_features = {}
        for example in examples:
            features = self.feature_extractor.extract(example)
            for k, v in features.items():
                if k not in all_features:
                    all_features[k] = []
                all_features[k].append(v)

        # 平均特征
        avg_features = {k: np.mean(v) for k, v in all_features.items()}

        # 计算抽象层次
        abstraction_level = 0
        if parent_concepts:
            parent_levels = [
                self.hierarchy.get_abstraction_level(pid)
                for pid in parent_concepts
                if pid in self.hierarchy.concepts
            ]
            abstraction_level = max(parent_levels) + 1 if parent_levels else 1

        # 创建概念
        concept = Concept(
            concept_id=concept_id,
            name=name,
            features=avg_features,
            examples=examples,
            parent_concepts=parent_concepts or set(),
            abstraction_level=abstraction_level
        )

        # 添加到层次结构
        self.hierarchy.add_concept(concept)

        # 更新统计
        self.stats['total_concepts'] += 1
        level = abstraction_level
        self.stats['abstraction_levels'][level] = self.stats['abstraction_levels'].get(level, 0) + 1

        logger.info(f"Formed concept: {name} (level {abstraction_level})")
        return concept

    def find_analogy(self, concept_id1: str, concept_id2: str,
                    threshold: float = 0.5) -> Optional[Analogy]:
        """
        寻找类比

        Args:
            concept_id1: 概念1 ID
            concept_id2: 概念2 ID
            threshold: 相似度阈值

        Returns:
            类比对象
        """
        concept1 = self.hierarchy.concepts.get(concept_id1)
        concept2 = self.hierarchy.concepts.get(concept_id2)

        if not concept1 or not concept2:
            return None

        analogy = self.analogy_engine.find_analogy(concept1, concept2, threshold)

        if analogy:
            self.stats['total_analogies'] += 1

        return analogy

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """
        获取概念

        Args:
            concept_id: 概念ID

        Returns:
            概念对象
        """
        return self.hierarchy.concepts.get(concept_id)

    def get_children(self, concept_id: str) -> List[Concept]:
        """
        获取子概念

        Args:
            concept_id: 概念ID

        Returns:
            子概念列表
        """
        return self.hierarchy.get_children(concept_id)

    def get_parents(self, concept_id: str) -> List[Concept]:
        """
        获取父概念

        Args:
            concept_id: 概念ID

        Returns:
            父概念列表
        """
        return self.hierarchy.get_parents(concept_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'total_concepts_in_hierarchy': len(self.hierarchy.concepts),
        }
