"""
连续概念学习 —— 离散类别从连续感知输入中涌现（PyTorch 版）

核心思想：智能体接收连续感知输入，但需要形成离散类别。
通信压力促使类别边界涌现——双方需要用有限的离散符号描述连续世界。

参考：
- Rosch (1978) 原型理论
- Levinson (2000) 语言相对论与类别学习
"""

from typing import Dict, List, Optional

import torch

from src.core.device import get_device


class ContinuousConceptSpace:
    """从连续感知输入中形成离散类别"""

    def __init__(self, feature_dim: int = 10, device: str = 'auto'):
        self.feature_dim = feature_dim
        self.device = get_device(device)

        # 类别原型：label -> 原型向量 (CPU 上存储，计算时移到 device)
        self._prototypes: Dict[str, torch.Tensor] = {}
        # 类别成员计数
        self._counts: Dict[str, int] = {}
        # 类别成员累积向量（用于增量更新原型）
        self._sums: Dict[str, torch.Tensor] = {}

        # 分配阈值：距离最近原型超过此值时创建新类别
        self._threshold: float = 1.5
        # 标签计数器（自动命名用）
        self._label_counter: int = 0

        # 原型矩阵缓存
        self._proto_matrix: Optional[torch.Tensor] = None
        self._proto_labels: List[str] = []
        self._dirty: bool = True

    # -------------------------------------------------------------------
    # 观察 & 分类
    # -------------------------------------------------------------------

    def observe(self, features: torch.Tensor, label: str = None) -> str:
        """
        观察连续特征向量，分配到类别

        如果提供 label，则更新该类别原型；否则分配到最近类别或创建新类别。

        Args:
            features: 连续特征向量
            label: 可选类别标签

        Returns:
            分配的类别标签
        """
        feat = features.to(self.device).float()
        if feat.dim() == 0:
            feat = feat.unsqueeze(0)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)

        # 指定标签：更新已有类别或创建新类别
        if label is not None:
            return self._assign_with_label(feat, label)

        # 自动分配：找最近原型
        self._rebuild_proto_cache()

        if self._proto_matrix is None:
            # 第一个观察 -> 新类别
            return self._create_category(feat)

        dists = torch.cdist(feat, self._proto_matrix).squeeze(0)
        min_idx = torch.argmin(dists).item()
        min_dist = dists[min_idx].item()

        if min_dist > self._threshold:
            return self._create_category(feat)
        else:
            assigned_label = self._proto_labels[min_idx]
            self._update_prototype(assigned_label, feat)
            return assigned_label

    # -------------------------------------------------------------------
    # 查询 API
    # -------------------------------------------------------------------

    def get_prototypes(self) -> Dict[str, torch.Tensor]:
        """返回所有类别原型的副本"""
        return {k: v.clone() for k, v in self._prototypes.items()}

    def get_category_purity(self) -> Dict[str, float]:
        """
        类别纯度：每个类别成员到原型的平均距离的倒数

        Returns:
            label -> purity (0 ~ 1)，值越高越纯净
        """
        purity = {}
        for label in self._prototypes:
            if self._counts.get(label, 0) <= 1:
                purity[label] = 1.0
            else:
                # 使用累积距离的近似：从 sum 和 count 推算
                # 纯度 = 1 / (1 + avg_distance)，avg_distance 用原型范数近似
                proto_norm = self._prototypes[label].norm().item()
                purity[label] = 1.0 / (1.0 + proto_norm * 0.1)
        return purity

    def align_categories(self, other: 'ContinuousConceptSpace') -> float:
        """
        测量两个类别系统的对齐度（0 ~ 1）

        使用最优匹配：为每个自身类别找到对方最近的类别，计算平均相似度。

        Args:
            other: 另一个 ContinuousConceptSpace

        Returns:
            对齐度，0 = 完全不对齐，1 = 完全对齐
        """
        my_protos = self.get_prototypes()
        other_protos = other.get_prototypes()

        if not my_protos or not other_protos:
            return 0.0

        # 构建距离矩阵
        my_labels = list(my_protos.keys())
        other_labels = list(other_protos.keys())

        my_mat = torch.stack([my_protos[l].to(self.device) for l in my_labels])
        other_mat = torch.stack(
            [other_protos[l].to(self.device) for l in other_labels]
        )

        dists = torch.cdist(my_mat, other_mat)  # (n_me, n_other)
        # 转换为相似度
        sims = 1.0 / (1.0 + dists)

        # 贪心最优匹配
        used = set()
        total_sim = 0.0
        n_matched = 0

        # 从每个自身类别出发，找对方最佳匹配
        for i in range(len(my_labels)):
            sorted_idx = torch.argsort(sims[i], descending=True)
            for j in sorted_idx:
                j_item = j.item()
                if j_item not in used:
                    used.add(j_item)
                    total_sim += sims[i, j_item].item()
                    n_matched += 1
                    break

        if n_matched == 0:
            return 0.0

        # 用较大类别集合归一化
        max_categories = max(len(my_labels), len(other_labels))
        return total_sim / max_categories

    def fuzzy_classify(self, features: torch.Tensor) -> Dict[str, float]:
        """
        模糊分类：返回所有类别的隶属概率

        使用 softmax 将距离转换为概率分布。

        Args:
            features: 连续特征向量

        Returns:
            label -> membership probability
        """
        self._rebuild_proto_cache()

        if self._proto_matrix is None:
            return {}

        feat = features.to(self.device).float()
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)

        dists = torch.cdist(feat, self._proto_matrix).squeeze(0)  # (n_cats,)
        # 距离越小越相似 -> 取负作为 logits
        logits = -dists
        probs = torch.softmax(logits, dim=0)

        return {
            label: probs[i].item()
            for i, label in enumerate(self._proto_labels)
        }

    # -------------------------------------------------------------------
    # 序列化
    # -------------------------------------------------------------------

    def save_state(self) -> dict:
        """将模块状态序列化为字典"""
        return {
            'prototypes': {
                k: v.tolist() for k, v in self._prototypes.items()
            },
            'counts': dict(self._counts),
            'sums': {
                k: v.tolist() for k, v in self._sums.items()
            },
            'threshold': self._threshold,
            'label_counter': self._label_counter,
            'feature_dim': self.feature_dim,
        }

    def load_state(self, state: dict) -> None:
        """从字典恢复模块状态"""
        self._threshold = state.get('threshold', self._threshold)
        self._label_counter = state.get('label_counter', 0)
        self.feature_dim = state.get('feature_dim', self.feature_dim)

        self._prototypes = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in state.get('prototypes', {}).items()
        }
        self._counts = {
            k: int(v) for k, v in state.get('counts', {}).items()
        }
        self._sums = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in state.get('sums', {}).items()
        }

        self._dirty = True

    # -------------------------------------------------------------------
    # 内部辅助
    # -------------------------------------------------------------------

    def _assign_with_label(self, feat: torch.Tensor, label: str) -> str:
        """使用指定标签分配类别"""
        if label in self._prototypes:
            self._update_prototype(label, feat)
        else:
            # 创建新类别
            self._prototypes[label] = feat.squeeze(0).detach().cpu()
            self._sums[label] = feat.squeeze(0).detach().cpu()
            self._counts[label] = 1
            self._dirty = True
        return label

    def _update_prototype(self, label: str, feat: torch.Tensor) -> None:
        """增量更新类别原型（均值）"""
        feat_cpu = feat.squeeze(0).detach().cpu()

        if label not in self._sums:
            self._sums[label] = feat_cpu.clone()
            self._counts[label] = 1
        else:
            self._sums[label] = self._sums[label] + feat_cpu
            self._counts[label] += 1

        # 原型 = 累积和 / 计数
        self._prototypes[label] = (
            self._sums[label] / self._counts[label]
        )
        self._dirty = True

    def _create_category(self, feat: torch.Tensor) -> str:
        """创建新类别"""
        label = f"C{self._label_counter}"
        self._label_counter += 1
        self._prototypes[label] = feat.squeeze(0).detach().cpu()
        self._sums[label] = feat.squeeze(0).detach().cpu()
        self._counts[label] = 1
        self._dirty = True
        return label

    def _rebuild_proto_cache(self) -> None:
        """重建原型矩阵缓存"""
        if not self._dirty:
            return
        self._proto_labels = list(self._prototypes.keys())
        if self._proto_labels:
            self._proto_matrix = torch.stack(
                [self._prototypes[l].to(self.device).float()
                 for l in self._proto_labels]
            )
        else:
            self._proto_matrix = None
        self._dirty = False
