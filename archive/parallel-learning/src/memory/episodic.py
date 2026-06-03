"""
情景记忆：带时间戳和衰减强度的经验痕迹

基于 Ebbinghaus 遗忘曲线实现指数衰减。
痕迹强度 = 初始强度 × forgetting_rate^age
检索时使用 strength × similarity 加权排序。
"""

from typing import Dict, List
import torch

from src.core.interfaces import IMemory
from src.core.device import get_device, to_device


class EpisodicMemory(IMemory):
    """情景记忆：带时间戳和衰减强度的经验痕迹"""

    def __init__(self, max_traces: int = 1000, dim: int = 40,
                 forgetting_rate: float = 0.95, device=None):
        self.max_traces = max_traces
        self.dim = dim
        self.forgetting_rate = forgetting_rate
        self.device = device or get_device()
        self.traces: List[Dict] = []
        self._timestamp_counter: int = 0

    def store(self, representation: torch.Tensor, metadata: Dict) -> None:
        """存入新的经验痕迹，初始 strength=1.0"""
        rep = to_device(representation.clone().detach(), self.device)

        self._timestamp_counter += 1
        trace = {
            'repr': rep,
            'meta': metadata,
            'timestamp': self._timestamp_counter,
            'strength': 1.0,
        }
        self.traces.append(trace)

        # 超容量时移除最弱的
        if len(self.traces) > self.max_traces:
            self._evict_weakest()

    def _evict_weakest(self):
        """驱逐强度最低的痕迹"""
        if not self.traces:
            return
        min_idx = min(range(len(self.traces)),
                      key=lambda i: self.traces[i]['strength'])
        self.traces.pop(min_idx)

    def retrieve(self, cue: torch.Tensor, k: int = 5) -> List[Dict]:
        """按 strength × similarity 加权检索"""
        if len(self.traces) == 0:
            return []

        cue = to_device(cue.clone().detach(), self.device)
        k = min(k, len(self.traces))

        # 构建表示矩阵和强度向量
        reps = torch.stack([t['repr'] for t in self.traces])  # (N, dim)
        strengths = torch.tensor(
            [t['strength'] for t in self.traces],
            device=self.device, dtype=torch.float32
        )  # (N,)

        # 余弦相似度
        cue_norm = cue / (cue.norm() + 1e-8)
        reps_norm = reps / (reps.norm(dim=1, keepdim=True) + 1e-8)
        similarities = torch.matmul(reps_norm, cue_norm)  # (N,)

        # 加权分数 = strength × similarity
        scores = strengths * similarities

        # 取 top-k
        topk_scores, topk_indices = torch.topk(scores, k)

        results = []
        for idx, score in zip(topk_indices.tolist(), topk_scores.tolist()):
            t = self.traces[idx]
            results.append({
                'metadata': t['meta'],
                'similarity': similarities[idx].item(),
                'strength': t['strength'],
                'score': score,
                'representation': t['repr'],
            })
        return results

    def consolidate(self) -> Dict:
        """重播巩固：聚类吸引 + 强度恢复

        核心策略：
        1. 计算痕迹间的相似度矩阵
        2. 找到每个痕迹的邻居（相似度 > threshold）
        3. 对有足够邻居的痕迹：计算邻居质心，做表示向量吸引
        4. 恢复被成功聚类的痕迹强度，削弱噪声痕迹
        """
        if len(self.traces) == 0:
            return {'replayed': 0, 'strengthened': 0, 'weakened': 0}

        n = len(self.traces)
        reps = torch.stack([t['repr'] for t in self.traces])  # (N, dim)
        norms = reps / (reps.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = torch.matmul(norms, norms.T)  # (N, N)

        # 每个痕迹找到与它最近邻的 min_neighbors 个邻居
        min_neighbors = 2
        neighbor_threshold = 0.3

        strengthened = 0
        weakened = 0

        for i in range(n):
            # 找到与痕迹 i 相似的邻居（排除自身）
            sims = sim_matrix[i].clone()
            sims[i] = -1.0  # 排除自身
            neighbor_mask = sims > neighbor_threshold
            neighbor_count = neighbor_mask.sum().item()

            if neighbor_count >= min_neighbors:
                # 有足够邻居：做质心吸引
                neighbor_indices = neighbor_mask.nonzero(as_tuple=True)[0].tolist()
                neighbor_reprs = torch.stack(
                    [self.traces[j]['repr'] for j in neighbor_indices] +
                    [self.traces[i]['repr']]
                )
                centroid = neighbor_reprs.mean(dim=0)
                centroid = centroid / (centroid.norm() + 1e-8)

                # 质心吸引
                attraction_rate = 0.2
                old_repr = self.traces[i]['repr']
                new_repr = (1.0 - attraction_rate) * old_repr + attraction_rate * centroid
                new_repr = new_repr / (new_repr.norm() + 1e-8)
                self.traces[i]['repr'] = new_repr

                # 恢复/加强强度
                self.traces[i]['strength'] = max(self.traces[i]['strength'], 1.0) * 1.1
                strengthened += 1
            else:
                # 孤立痕迹：削弱
                self.traces[i]['strength'] *= 0.85
                weakened += 1

        return {
            'replayed': strengthened,
            'strengthened': strengthened,
            'weakened': weakened,
        }

    def decay(self):
        """对所有痕迹应用遗忘衰减：strength *= forgetting_rate"""
        for trace in self.traces:
            trace['strength'] *= self.forgetting_rate

    def get_forgetting_curve(self) -> Dict[int, float]:
        """返回 timestamp → avg_strength 映射"""
        curve: Dict[int, float] = {}
        counts: Dict[int, int] = {}

        for trace in self.traces:
            ts = trace['timestamp']
            if ts not in curve:
                curve[ts] = 0.0
                counts[ts] = 0
            curve[ts] += trace['strength']
            counts[ts] += 1

        for ts in curve:
            curve[ts] /= counts[ts]

        return curve

    def __len__(self):
        return len(self.traces)
