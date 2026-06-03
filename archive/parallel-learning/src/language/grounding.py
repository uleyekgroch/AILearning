"""
符号接地模块：从感知到抽象的渐进过程（PyTorch 版）

从 mvl/agent.py 的 GroundingModule 移植，用 torch 替代 numpy。

三层接地：
1. 感知聚类 -> 原始概念
2. 概念组合 -> 复合符号
3. 社会标注 -> 语言符号
"""

from typing import Dict, List, Optional, Tuple

import torch

from src.core.device import get_device


class GroundingModule:
    """
    符号接地模块

    符号的意义来自感知-行动经验，不是来自其他符号。
    """

    def __init__(self, obs_dim: int = 40, device: str = 'auto'):
        self.obs_dim = obs_dim
        self.device = get_device(device)

        # 感知聚类 -> 原始概念
        self.perceptual_clusters: Dict[int, Dict] = {}
        self.cluster_counter: int = 0

        # 符号 -> 概念映射
        self.symbol_mappings: Dict[str, Dict] = {}

        # 概念层级
        self.concept_hierarchy: Dict[str, List[str]] = {}

        # 聚类中心张量缓存（用于 torch.cdist 快速距离计算）
        self._centroid_matrix: Optional[torch.Tensor] = None
        self._centroid_ids: List[int] = []
        self._dirty: bool = True

    # -------------------------------------------------------------------
    # 感知接地
    # -------------------------------------------------------------------

    def ground_from_perception(self, observation: torch.Tensor) -> int:
        """
        从感知经验中建立概念

        将相似的感知经验聚类为"概念"。
        """
        obs = observation.to(self.device).float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        best_cluster = None
        best_distance = float('inf')

        if self._centroid_matrix is not None and len(self._centroid_ids) > 0:
            # 快速路径：用 torch.cdist 批量计算距离
            dists = torch.cdist(obs, self._centroid_matrix).squeeze(0)  # (num_clusters,)
            min_idx = torch.argmin(dists).item()
            best_distance = dists[min_idx].item()
            best_cluster = self._centroid_ids[min_idx]

        # 距离阈值：超过 0.5 创建新聚类
        if best_distance > 0.5 or best_cluster is None:
            cluster_id = self.cluster_counter
            self.cluster_counter += 1
            self.perceptual_clusters[cluster_id] = {
                'centroid': obs.squeeze(0).detach().cpu(),
                'count': 1,
            }
            self._dirty = True
        else:
            cluster_id = best_cluster
            cluster = self.perceptual_clusters[cluster_id]
            cluster['count'] += 1
            old_centroid = cluster['centroid'].cpu()
            n = cluster['count']
            obs_cpu = obs.squeeze(0).cpu()
            new_centroid = (old_centroid * (n - 1) + obs_cpu) / n
            cluster['centroid'] = new_centroid.detach()

        if self._dirty:
            self._rebuild_centroid_cache()
        return cluster_id

    def ground_from_social(self, symbol: str, referent: torch.Tensor, context: str = ""):
        """
        从社会交互中接地符号

        将符号与当前感知经验关联。
        """
        cluster_id = self.ground_from_perception(referent)

        if symbol not in self.symbol_mappings:
            self.symbol_mappings[symbol] = {
                'referent_clusters': [cluster_id],
                'confidence': 1.0,
                'contexts': [context] if context else [],
                'usage_count': 1,
            }
        else:
            mapping = self.symbol_mappings[symbol]
            if cluster_id not in mapping['referent_clusters']:
                mapping['referent_clusters'].append(cluster_id)
            mapping['usage_count'] += 1
            mapping['confidence'] = min(1.0, mapping['confidence'] + 0.1)
            if context and context not in mapping['contexts']:
                mapping['contexts'].append(context)

    # -------------------------------------------------------------------
    # 批量接地
    # -------------------------------------------------------------------

    def ground_from_perception_batch(self, observations: torch.Tensor) -> list:
        """批量感知接地——一次 cdist 处理多个观测

        Args:
            observations: (batch, obs_dim) 张量
        Returns:
            cluster_ids: List[int]，每个观测对应的聚类 ID
        """
        obs = observations.to(self.device).float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        batch_size = obs.shape[0]
        cluster_ids = []

        if self._centroid_matrix is not None and len(self._centroid_ids) > 0:
            # 批量距离计算
            dists = torch.cdist(obs, self._centroid_matrix)  # (batch, num_clusters)
            min_dists, min_indices = dists.min(dim=1)

            for i in range(batch_size):
                if min_dists[i].item() > 0.5:
                    # 创建新聚类
                    cid = self.cluster_counter
                    self.cluster_counter += 1
                    self.perceptual_clusters[cid] = {
                        'centroid': obs[i].detach().cpu(),
                        'count': 1,
                    }
                    cluster_ids.append(cid)
                else:
                    # 更新已有聚类
                    cid = self._centroid_ids[min_indices[i].item()]
                    cluster = self.perceptual_clusters[cid]
                    cluster['count'] += 1
                    n = cluster['count']
                    old_c = cluster['centroid'].cpu()
                    new_c = (old_c * (n - 1) + obs[i].cpu()) / n
                    cluster['centroid'] = new_c.detach()
                    cluster_ids.append(cid)
        else:
            # 没有聚类，全部创建新的
            for i in range(batch_size):
                cid = self.cluster_counter
                self.cluster_counter += 1
                self.perceptual_clusters[cid] = {
                    'centroid': obs[i].detach().cpu(),
                    'count': 1,
                }
                cluster_ids.append(cid)

        self._dirty = True
        self._rebuild_centroid_cache()
        return cluster_ids

    def ground_symbols_batch(self, symbols_contexts: list, observations: torch.Tensor):
        """批量社会标注

        Args:
            symbols_contexts: [(symbol, context), ...]
            observations: (batch, obs_dim) 或者 (obs_dim,)
        """
        if not symbols_contexts:
            return

        obs = observations.to(self.device).float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        cluster_ids = self.ground_from_perception_batch(obs)

        for (symbol, context), cid in zip(symbols_contexts, cluster_ids):
            if symbol not in self.symbol_mappings:
                self.symbol_mappings[symbol] = {
                    'referent_clusters': [cid],
                    'confidence': 1.0,
                    'contexts': [context] if context else [],
                    'usage_count': 1,
                }
            else:
                mapping = self.symbol_mappings[symbol]
                if cid not in mapping['referent_clusters']:
                    mapping['referent_clusters'].append(cid)
                mapping['usage_count'] += 1
                mapping['confidence'] = min(1.0, mapping['confidence'] + 0.1)
                if context and context not in mapping['contexts']:
                    mapping['contexts'].append(context)

    # -------------------------------------------------------------------
    # 查询 API
    # -------------------------------------------------------------------

    def get_symbol_meaning(self, symbol: str) -> Optional[Dict]:
        """获取符号的含义"""
        return self.symbol_mappings.get(symbol)

    def get_grounded_symbols(self) -> List[str]:
        """获取所有已接地的符号"""
        return list(self.symbol_mappings.keys())

    def find_similar_concepts(self, observation: torch.Tensor, top_k: int = 3) -> List[Tuple[int, float]]:
        """找到与给定观测最相似的概念"""
        obs = observation.to(self.device).float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if self._centroid_matrix is None or len(self._centroid_ids) == 0:
            return []

        dists = torch.cdist(obs, self._centroid_matrix).squeeze(0)
        similarities = []
        for i, cluster_id in enumerate(self._centroid_ids):
            distance = dists[i].item()
            similarity = 1.0 / (1.0 + distance)
            similarities.append((cluster_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    # -------------------------------------------------------------------
    # 内部辅助
    # -------------------------------------------------------------------

    def _rebuild_centroid_cache(self):
        """重建聚类中心矩阵缓存"""
        if not self._dirty:
            return
        self._centroid_ids = list(self.perceptual_clusters.keys())
        if self._centroid_ids:
            centroids = [
                self.perceptual_clusters[cid]['centroid'].to(self.device)
                for cid in self._centroid_ids
            ]
            self._centroid_matrix = torch.stack(centroids).float()
        else:
            self._centroid_matrix = None
        self._dirty = False
