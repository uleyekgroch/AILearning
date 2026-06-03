"""
工作记忆：容量 7±2 的短期缓冲区，FIFO 驱逐

基于 Baddeley 的工作记忆模型，实现有限容量的短期存储。
超出容量时自动驱逐最早存入的项（FIFO）。
"""

from collections import deque
from typing import Dict, List
import torch

from src.core.interfaces import IMemory
from src.core.device import get_device, to_device


class WorkingMemory(IMemory):
    """工作记忆：容量有限的短期缓冲区"""

    def __init__(self, capacity: int = 7, dim: int = 40, device=None):
        self.capacity = capacity
        self.dim = dim
        self.device = device or get_device()
        # deque 自动 FIFO 驱逐
        self.items: deque = deque(maxlen=capacity)

    def store(self, representation: torch.Tensor, metadata: Dict) -> None:
        """存储新项，超容量时 FIFO 驱逐最旧项"""
        rep = to_device(representation.clone().detach(), self.device)
        self.items.append((rep, metadata))

    def retrieve(self, cue: torch.Tensor, k: int = 5) -> List[Dict]:
        """按余弦相似度检索最近的 k 项"""
        if len(self.items) == 0:
            return []

        cue = to_device(cue.clone().detach(), self.device)
        k = min(k, len(self.items))

        # 构建表示矩阵
        reps = torch.stack([item[0] for item in self.items])  # (N, dim)
        cue_norm = cue / (cue.norm() + 1e-8)
        reps_norm = reps / (reps.norm(dim=1, keepdim=True) + 1e-8)

        # 余弦相似度
        similarities = torch.matmul(reps_norm, cue_norm)  # (N,)

        # 取 top-k
        topk_values, topk_indices = torch.topk(similarities, k)

        results = []
        for idx, sim in zip(topk_indices.tolist(), topk_values.tolist()):
            _, meta = self.items[idx]
            results.append({
                'metadata': meta,
                'similarity': sim,
                'representation': self.items[idx][0],
            })
        return results

    def consolidate(self) -> Dict:
        """返回所有项供上一层（情景记忆）提取，然后清空"""
        items = list(self.items)
        count = len(items)
        self.items.clear()
        return {'transferred': count, 'items': items}

    def clear(self):
        """清空工作记忆"""
        self.items.clear()

    def __len__(self):
        return len(self.items)
