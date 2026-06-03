"""
语义记忆：从重复经验中抽象出概念

通过聚类将相似经验归入同一概念，维护概念质心。
当新经验与已有概念质心的距离超过阈值时创建新概念。
"""

from typing import Dict, List
import torch

from src.core.interfaces import IMemory
from src.core.device import get_device, to_device


class SemanticMemory(IMemory):
    """语义记忆：概念抽象与存储"""

    def __init__(self, dim: int = 40, device=None,
                 new_concept_threshold: float = 0.5,
                 min_examples_to_keep: int = 2):
        self.dim = dim
        self.device = device or get_device()
        self.new_concept_threshold = new_concept_threshold
        self.min_examples_to_keep = min_examples_to_keep
        self.concepts: Dict[str, Dict] = {}
        self._concept_counter: int = 0

    def store(self, representation: torch.Tensor, metadata: Dict) -> None:
        """尝试归入已有概念，或创建新概念"""
        rep = to_device(representation.clone().detach(), self.device)

        if len(self.concepts) == 0:
            self._create_concept(rep, metadata)
            return

        # 计算与所有概念质心的余弦相似度
        labels = list(self.concepts.keys())
        centroids = torch.stack(
            [self.concepts[l]['centroid'] for l in labels]
        )  # (K, dim)

        rep_norm = rep / (rep.norm() + 1e-8)
        centroids_norm = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)
        similarities = torch.matmul(centroids_norm, rep_norm)  # (K,)

        best_idx = similarities.argmax().item()
        best_sim = similarities[best_idx].item()

        if best_sim > (1.0 - self.new_concept_threshold):
            # 归入最近的概念
            self._update_concept(labels[best_idx], rep, metadata)
        else:
            # 创建新概念
            self._create_concept(rep, metadata)

    def _create_concept(self, representation: torch.Tensor,
                        metadata: Dict) -> None:
        """创建新概念"""
        self._concept_counter += 1
        label = f"concept_{self._concept_counter}"
        self.concepts[label] = {
            'centroid': representation.clone(),
            'count': 1,
            'examples': 1,
            'label': label,
        }

    def _update_concept(self, label: str, representation: torch.Tensor,
                        metadata: Dict) -> None:
        """更新已有概念的质心"""
        concept = self.concepts[label]
        n = concept['count']
        # 增量更新质心：centroid = (centroid * n + rep) / (n + 1)
        concept['centroid'] = (concept['centroid'] * n + representation) / (n + 1)
        concept['count'] = n + 1
        concept['examples'] += 1

    def retrieve(self, cue: torch.Tensor, k: int = 5) -> List[Dict]:
        """检索最近的概念"""
        if len(self.concepts) == 0:
            return []

        cue = to_device(cue.clone().detach(), self.device)
        k = min(k, len(self.concepts))

        labels = list(self.concepts.keys())
        centroids = torch.stack(
            [self.concepts[l]['centroid'] for l in labels]
        )

        cue_norm = cue / (cue.norm() + 1e-8)
        centroids_norm = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)
        similarities = torch.matmul(centroids_norm, cue_norm)

        topk_values, topk_indices = torch.topk(similarities, k)

        results = []
        for idx, sim in zip(topk_indices.tolist(), topk_values.tolist()):
            label = labels[idx]
            concept = self.concepts[label]
            results.append({
                'label': label,
                'similarity': sim,
                'count': concept['count'],
                'centroid': concept['centroid'],
            })
        return results

    def consolidate(self) -> Dict:
        """清理低频概念"""
        before = len(self.concepts)
        removed = 0

        to_remove = []
        for label, concept in self.concepts.items():
            if concept['examples'] < self.min_examples_to_keep:
                to_remove.append(label)

        for label in to_remove:
            del self.concepts[label]
            removed += 1

        return {
            'removed': removed,
            'remaining': len(self.concepts),
            'before': before,
        }

    def get_concept_graph(self) -> Dict[str, List[str]]:
        """返回概念间的关联图（基于质心相似度）"""
        if len(self.concepts) == 0:
            return {}

        labels = list(self.concepts.keys())
        centroids = torch.stack(
            [self.concepts[l]['centroid'] for l in labels]
        )

        norms = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = torch.matmul(norms, norms.T)

        graph: Dict[str, List[str]] = {}
        for i, label_i in enumerate(labels):
            graph[label_i] = []
            for j, label_j in enumerate(labels):
                if i != j and sim_matrix[i, j].item() > 0.5:
                    graph[label_i].append(label_j)

        return graph

    def __len__(self):
        return len(self.concepts)
