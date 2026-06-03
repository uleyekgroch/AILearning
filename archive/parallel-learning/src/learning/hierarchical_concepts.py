#!/usr/bin/env python3
"""层次概念体系 (Hierarchical Concept Taxonomy)

基于论文:
- PMC 2023: "Learning and Representation of Hierarchical Concepts in
  Hippocampus and Prefrontal Cortex"
  mPFC跟踪层次概念知识的积累，海马体支持逐次更新。
- eLife 2023: "Distinct Hippocampal and Cortical Contributions in the
  Representation of Hierarchical Knowledge"
  前海马体和mPFC编码层次位置，下皮层区域提供不同贡献。
- Trends in Cognitive Sciences 2025: "A Hierarchical Model of Early Brain
  Functional Network Development"
  脑网络从感觉→情感→认知的层次发展路径。

核心思想:
  概念不是扁平的，而是有层次结构的：
  - 超ordinate (上层): 动物、交通工具、科学
  - 基础层 (basic): 狗、汽车、物理学
  - 下层 (subordinate): 金毛、特斯拉、量子力学

  儿童学习概念的顺序:
  1. 先学基础层（"狗"）— 最常用的分类层级
  2. 再学上层（"动物"）— 需要更多抽象
  3. 最后学下层（"金毛"）— 需要专业知识

  Rosch et al. 1976的发现:
  - 基础层是"最佳"分类层级（信息量最大，区分度最高）
  - 上层太抽象，下层太具体

  对学习系统的意义:
  - 知识图谱从扁平变层次化
  - 支持不同粒度的推理（"这是一种动物" vs "这是一种金毛犬"）
  - 支持泛化（"狗会叫" → "动物可能发出声音"）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import math


@dataclass
class ConceptNode:
    """概念节点

    每个节点代表一个概念，在层次结构中有位置。
    """
    concept_id: str               # 唯一标识
    name: str                      # 概念名称
    level: int                     # 层级深度 (0=根, 1=超ordinate, 2=基础, 3=下层)
    parent_id: Optional[str]       # 父节点
    children: Set[str]             # 子节点集合
    embedding: torch.Tensor        # 概念嵌入
    specificity: float             # 特异性(0-1, 越具体越高)
    abstractness: float            # 抽象度(0-1, 越抽象越高)
    instance_count: int = 0        # 实例计数（遇到次数）


class HierarchicalConceptSystem:
    """层次概念体系

    核心功能:
    1. 自动发现概念的层次关系（通过嵌入相似度和包含关系）
    2. 支持不同粒度的推理
    3. 沿层次结构上下传播知识
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 概念节点存储
        self.nodes: Dict[str, ConceptNode] = {}

        # 快速查找索引
        self.name_to_id: Dict[str, str] = {}
        self.level_index: Dict[int, Set[str]] = {}  # level → {node_ids}

        # 上行/下行传播缓存
        self._propagation_cache: Dict[str, List[str]] = {}

        # 统计
        self.stats = {
            'nodes_created': 0,
            'hierarchies_formed': 0,
            'upward_propagations': 0,
            'downward_propagations': 0,
            'generalizations': 0,
        }

    def add_concept(self, name: str, embedding: torch.Tensor,
                    parent_name: str = None) -> str:
        """添加概念到层次结构

        如果提供了parent_name，概念被放在父节点下。
        如果没有，系统尝试自动找到合适的层次位置。
        """
        # 检查是否已存在
        if name in self.name_to_id:
            node = self.nodes[self.name_to_id[name]]
            # 更新嵌入（EMA）
            alpha = 1.0 / (1 + node.instance_count)
            with torch.no_grad():
                node.embedding = (1 - alpha) * node.embedding + alpha * embedding.detach()
            node.instance_count += 1
            return node.concept_id

        # 确定层级
        if parent_name and parent_name in self.name_to_id:
            parent_id = self.name_to_id[parent_name]
            parent = self.nodes[parent_id]
            level = parent.level + 1
            specificity = min(1.0, parent.specificity + 0.2)
            abstractness = max(0.0, parent.abstractness - 0.2)
        else:
            # 自动确定：新概念默认在基础层(level=2)
            level = 2
            specificity = 0.5
            abstractness = 0.5
            parent_id = None

        # 创建节点
        node_id = f"c_{self.stats['nodes_created']}"
        node = ConceptNode(
            concept_id=node_id,
            name=name,
            level=level,
            parent_id=parent_id,
            children=set(),
            embedding=embedding.detach().clone(),
            specificity=specificity,
            abstractness=abstractness,
            instance_count=1,
        )

        self.nodes[node_id] = node
        self.name_to_id[name] = node_id

        # 更新索引
        if level not in self.level_index:
            self.level_index[level] = set()
        self.level_index[level].add(node_id)

        # 更新父节点
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children.add(node_id)
            self.stats['hierarchies_formed'] += 1

        self.stats['nodes_created'] += 1
        self._propagation_cache.clear()  # 清缓存

        return node_id

    def auto_classify(self, name: str, embedding: torch.Tensor) -> str:
        """自动分类：为概念找到最合适的层次位置

        策略:
        1. 找到嵌入最相似的已有概念
        2. 如果相似度>0.7 → 作为兄弟（同层）
        3. 如果相似度0.4-0.7 → 作为子节点（下层）
        4. 如果相似度<0.4 → 创建新的顶层分类
        """
        emb = embedding.to(self.device)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        best_match = None
        best_sim = 0.0

        for nid, node in self.nodes.items():
            n_emb = node.embedding.unsqueeze(0) if node.embedding.dim() == 1 else node.embedding
            sim = F.cosine_similarity(emb, n_emb.to(self.device)).item()
            if sim > best_sim:
                best_sim = sim
                best_match = node

        if best_match and best_sim > 0.7:
            # 同层兄弟
            if best_match.parent_id:
                parent = self.nodes[best_match.parent_id]
                return self.add_concept(name, embedding, parent_name=parent.name)
            else:
                return self.add_concept(name, embedding)
        elif best_match and best_sim > 0.4:
            # 作为子节点
            return self.add_concept(name, embedding, parent_name=best_match.name)
        else:
            # 新分类
            return self.add_concept(name, embedding)

    def get_ancestors(self, name: str) -> List[str]:
        """获取概念的所有祖先（从下到上）"""
        if name not in self.name_to_id:
            return []

        ancestors = []
        node_id = self.name_to_id[name]
        current = self.nodes.get(node_id)
        while current and current.parent_id:
            parent = self.nodes.get(current.parent_id)
            if parent:
                ancestors.append(parent.name)
            current = parent
        return ancestors

    def get_descendants(self, name: str) -> List[str]:
        """获取概念的所有后代"""
        if name not in self.name_to_id:
            return []

        descendants = []
        node_id = self.name_to_id[name]
        node = self.nodes.get(node_id)
        if not node:
            return []

        # BFS遍历子树
        queue = list(node.children)
        while queue:
            child_id = queue.pop(0)
            child = self.nodes.get(child_id)
            if child:
                descendants.append(child.name)
                queue.extend(child.children)

        return descendants

    def propagate_upward(self, name: str, attribute: str, value: str,
                        confidence: float = 0.8) -> List[str]:
        """向上传播属性（归纳推理）

        如果"金毛"有属性"会游泳"，
        则"狗"可能有属性"会游泳"（置信度降低），
        "动物"可能有属性"会游泳"（置信度再降低）。

        Returns:
            传播到的祖先列表
        """
        ancestors = self.get_ancestors(name)
        propagated = []

        for i, ancestor_name in enumerate(ancestors):
            # 置信度随层级递减
            decay = 0.8 ** (i + 1)
            prop_confidence = confidence * decay

            if prop_confidence > 0.2:
                propagated.append(f"{ancestor_name} 可能有{attribute}{value} "
                                f"(置信度:{prop_confidence:.2f})")
                self.stats['upward_propagations'] += 1

        return propagated

    def propagate_downward(self, name: str, attribute: str, value: str,
                          confidence: float = 0.8) -> List[str]:
        """向下传播属性（演绎推理）

        如果"动物"有属性"需要呼吸"，
        则"狗"一定有属性"需要呼吸"，
        "金毛"一定有属性"需要呼吸"。

        Returns:
            传播到的后代列表
        """
        descendants = self.get_descendants(name)
        propagated = []

        for desc_name in descendants:
            propagated.append(f"{desc_name} {attribute}{value}")
            self.stats['downward_propagations'] += 1

        return propagated

    def find_common_ancestor(self, name_a: str, name_b: str) -> Optional[str]:
        """找两个概念的最近共同祖先"""
        ancestors_a = self.get_ancestors(name_a)
        ancestors_b = set(self.get_ancestors(name_b))

        for ancestor in ancestors_a:
            if ancestor in ancestors_b:
                return ancestor
        return None

    def get_generalization(self, name: str) -> Optional[str]:
        """获取概念的上层泛化"""
        if name not in self.name_to_id:
            return None
        node = self.nodes[self.name_to_id[name]]
        if node.parent_id and node.parent_id in self.nodes:
            self.stats['generalizations'] += 1
            return self.nodes[node.parent_id].name
        return None

    def get_stats(self) -> Dict:
        levels = {}
        for level, node_ids in self.level_index.items():
            levels[f'level_{level}'] = len(node_ids)

        return {
            **self.stats,
            'total_concepts': len(self.nodes),
            'hierarchy_depth': max(self.level_index.keys()) if self.level_index else 0,
            **levels,
        }
