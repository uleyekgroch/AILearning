"""概念形成 — 原型理论

不是属性交集，是真正的概念形成。

核心能力：
1. 原型计算 — 从实例中提取典型特征
2. 典型性评估 — 评估实例的典型程度
3. 渐进类别成员 — 模糊边界
4. 概念层次 — 从具体到抽象

运行方式：
    python training/layers/concept_formation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class Concept:
    """概念 — 基于原型理论"""
    name: str
    prototype: torch.Tensor  # 原型向量
    features: Dict[str, float] = field(default_factory=dict)  # 典型特征
    examples: List[torch.Tensor] = field(default_factory=list)  # 实例
    typicality: Dict[str, float] = field(default_factory=dict)  # 典型性
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    abstraction_level: str = "basic"  # concrete, basic, abstract, superordinate


class PrototypeNetwork(nn.Module):
    """原型网络 — 学习原型表示"""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码为原型空间"""
        return self.encoder(x)


class ConceptFormation:
    """概念形成系统

    基于原型理论：
    - 概念由原型表示
    - 实例与原型的相似度决定典型性
    - 概念边界是模糊的
    """

    def __init__(self, feature_dim: int = 128, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.feature_dim = feature_dim

        # 原型网络
        self.prototype_net = PrototypeNetwork(feature_dim, 64, 32).to(self.device)

        # 概念库
        self.concepts: Dict[str, Concept] = {}

        # 实例库
        self.instances: Dict[str, List[torch.Tensor]] = defaultdict(list)

        # 统计
        self.stats = {
            'concepts_formed': 0,
            'instances_processed': 0,
            'abstractions_made': 0,
        }

    def add_instance(self, concept_name: str, features: torch.Tensor):
        """添加实例"""
        if features.dim() == 1:
            features = features.unsqueeze(0)

        # 存储实例
        self.instances[concept_name].append(features.squeeze(0).to(self.device))
        self.stats['instances_processed'] += 1

        # 如果有足够的实例，形成概念
        if len(self.instances[concept_name]) >= 3:
            self._form_concept(concept_name)

    def _form_concept(self, concept_name: str):
        """形成概念"""
        instances = self.instances[concept_name]

        # 计算原型（平均值）
        prototype = torch.stack(instances).mean(dim=0).to(self.device)

        # 编码到原型空间
        with torch.no_grad():
            prototype_encoded = self.prototype_net(prototype.unsqueeze(0)).squeeze(0)

        # 计算典型性
        typicality = {}
        for i, instance in enumerate(instances):
            similarity = F.cosine_similarity(
                instance.unsqueeze(0),
                prototype.unsqueeze(0)
            ).item()
            typicality[f"instance_{i}"] = similarity

        # 形成概念
        concept = Concept(
            name=concept_name,
            prototype=prototype_encoded,
            typicality=typicality,
            abstraction_level="basic",
        )

        self.concepts[concept_name] = concept
        self.stats['concepts_formed'] += 1

    def compute_typicality(self, concept_name: str, instance: torch.Tensor) -> float:
        """计算实例的典型性"""
        if concept_name not in self.concepts:
            return 0.0

        concept = self.concepts[concept_name]

        # 编码实例
        instance = instance.to(self.device)
        with torch.no_grad():
            instance_encoded = self.prototype_net(instance.unsqueeze(0)).squeeze(0)

        # 计算与原型的相似度
        similarity = F.cosine_similarity(
            instance_encoded.unsqueeze(0),
            concept.prototype.unsqueeze(0)
        ).item()

        return similarity

    def categorize(self, instance: torch.Tensor, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """将实例分类到概念"""
        results = []

        for concept_name, concept in self.concepts.items():
            typicality = self.compute_typicality(concept_name, instance)
            if typicality >= threshold:
                results.append((concept_name, typicality))

        # 按典型性排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def abstract_up(self, concept_name: str) -> Optional[str]:
        """向上抽象"""
        if concept_name not in self.concepts:
            return None

        concept = self.concepts[concept_name]
        return concept.parent

    def specialize_down(self, concept_name: str) -> List[str]:
        """向下特化"""
        if concept_name not in self.concepts:
            return []

        concept = self.concepts[concept_name]
        return concept.children

    def form_abstract_concept(self, children: List[str], abstract_name: str):
        """从子概念形成抽象概念"""
        if not children:
            return

        # 收集子概念的原型
        prototypes = []
        for child_name in children:
            if child_name in self.concepts:
                prototypes.append(self.concepts[child_name].prototype)

        if not prototypes:
            return

        # 计算抽象原型
        abstract_prototype = torch.stack(prototypes).mean(dim=0)

        # 形成抽象概念
        abstract_concept = Concept(
            name=abstract_name,
            prototype=abstract_prototype,
            abstraction_level="abstract",
            children=children,
        )

        self.concepts[abstract_name] = abstract_concept

        # 更新子概念的父概念
        for child_name in children:
            if child_name in self.concepts:
                self.concepts[child_name].parent = abstract_name

        self.stats['abstractions_made'] += 1

    def query(self, question: str) -> Dict:
        """查询概念"""
        # 提取关键词
        keywords = self._extract_keywords(question)

        results = {
            'concepts': [],
            'typicality': [],
            'hierarchy': [],
        }

        # 搜索概念
        for keyword in keywords:
            for name, concept in self.concepts.items():
                if keyword in name:
                    results['concepts'].append({
                        'name': name,
                        'level': concept.abstraction_level,
                        'parent': concept.parent,
                        'children': concept.children,
                        'num_examples': len(concept.examples),
                    })

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        import re
        # 中文关键词
        zh_keywords = re.findall(r'[一-鿿]{2,6}', text)
        # 英文关键词
        en_keywords = re.findall(r'[a-zA-Z]+', text)

        # 过滤停用词
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样')
        keywords = [k for k in zh_keywords + en_keywords if k not in stopwords and len(k) >= 2]

        return keywords

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'total_concepts': len(self.concepts),
            'total_instances': sum(len(v) for v in self.instances.values()),
        }


def test_concept_formation():
    """测试概念形成"""
    print("=" * 70)
    print("概念形成测试")
    print("=" * 70)

    formation = ConceptFormation(feature_dim=32)

    # 添加实例
    print("\n1. 添加实例:")

    # 猫的实例
    for i in range(5):
        features = torch.randn(32)
        features[0] = 0.8  # 有毛
        features[1] = 0.9  # 四条腿
        features[2] = 0.3  # 小型
        formation.add_instance('猫', features)

    # 狗的实例
    for i in range(5):
        features = torch.randn(32)
        features[0] = 0.8  # 有毛
        features[1] = 0.9  # 四条腿
        features[2] = 0.6  # 中型
        formation.add_instance('狗', features)

    print(f"  猫实例: {len(formation.instances['猫'])}")
    print(f"  狗实例: {len(formation.instances['狗'])}")

    # 形成概念
    print("\n2. 形成概念:")
    print(f"  概念数: {len(formation.concepts)}")
    for name, concept in formation.concepts.items():
        print(f"    {name}: {concept.abstraction_level}")

    # 测试典型性
    print("\n3. 典型性测试:")
    cat_instance = torch.randn(32)
    cat_instance[0] = 0.8
    cat_instance[1] = 0.9
    cat_instance[2] = 0.3

    cat_typicality = formation.compute_typicality('猫', cat_instance)
    dog_typicality = formation.compute_typicality('狗', cat_instance)
    print(f"  猫实例对猫的典型性: {cat_typicality:.3f}")
    print(f"  猫实例对狗的典型性: {dog_typicality:.3f}")

    # 测试分类
    print("\n4. 分类测试:")
    categories = formation.categorize(cat_instance, threshold=0.5)
    for name, score in categories:
        print(f"  {name}: {score:.3f}")

    # 形成抽象概念
    print("\n5. 抽象概念:")
    formation.form_abstract_concept(['猫', '狗'], '哺乳动物')
    print(f"  概念数: {len(formation.concepts)}")

    # 测试层次
    print("\n6. 概念层次:")
    hierarchy = formation.abstract_up('猫')
    print(f"  猫的父概念: {hierarchy}")

    # 统计
    print("\n统计:")
    stats = formation.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    test_concept_formation()
