"""对比学习训练器 — 让编码器学会区分相关与不相关概念

核心问题（Phase 4 要解决的）：
  未经对比训练的 Transformer 编码器将所有文本映射为高度相似的向量。
  cos("数学", "物理学") ≈ cos("数学", "苹果") ≈ 0.6-0.8
  → 概念空间无法有效区分相关 vs 不相关概念
  → 激活扩散推理退化为随机噪声

解决思路 — InfoNCE 对比损失：
  正样本对（应该近）：同一文本中出现的概念、统计学习发现的共现概念
  负样本对（应该远）：不同文本中的概念、随机采样

  L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))

  其中 τ 是温度参数，控制分布的锐度。
  τ=0.07 → 很sharp，迫使模型精确区分
  τ=0.5  → 很soft，允许更多模糊

设计参考：
  - SimCLR (Chen et al. 2020): 数据增强 + 大batch对比
  - MoCo (He et al. 2020): 动量编码器 + 内存银行
  - DeCLUTR (Giorgi et al. 2021): 文本段落的自监督对比

我们的简化版（在线学习，无需大batch）：
  1. 内存银行存储近期概念嵌入（避免需要大batch）
  2. 每次学习新文本时，用当前概念与内存银行中的概念构建正负对
  3. 统计学习的共现关系提供高质量正样本
  4. 硬负例挖掘：关注"看起来相关但实际不相关"的对
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque


@dataclass
class ContrastiveStats:
    """对比学习统计"""
    total_updates: int = 0
    total_loss: float = 0.0
    avg_loss: float = 0.0
    positive_sims: List[float] = field(default_factory=list)
    negative_sims: List[float] = field(default_factory=list)
    last_cos_gap: float = 0.0  # cos(positive) - cos(negative) 差距


class MemoryBank:
    """概念嵌入内存银行

    维护一个滑动窗口的概念嵌入，用于构建负样本。
    这避免了需要大 batch size 的限制——在任何时刻，
    内存中的所有概念都可以作为负样本候选。

    参考 MoCo 的 queue-based memory bank。
    """

    def __init__(self, capacity: int = 512, dim: int = 128):
        self.capacity = capacity
        self.dim = dim

        # 用 deque 维护 (concept_id, vector, text_id) 的滑动窗口
        self._entries: deque = deque(maxlen=capacity)

        # 快速查找索引
        self._concept_to_entries: Dict[str, List[int]] = defaultdict(list)

    def push(self, concept_id: str, vector: torch.Tensor, text_id: int = 0):
        """推入一个概念嵌入

        如果该概念已存在，更新其向量（巩固）。
        如果不存在，添加新条目。
        """
        vector = vector.detach().cpu().clone()

        # 检查是否已有该概念
        existing_indices = self._concept_to_entries.get(concept_id, [])
        if existing_indices:
            # 更新已有条目（取移动平均）
            idx = existing_indices[0]
            if idx < len(self._entries):
                old_vec = self._entries[idx][1]
                new_vec = 0.7 * old_vec + 0.3 * vector
                self._entries[idx] = (concept_id, new_vec, text_id)
                return

        # 新条目
        self._entries.append((concept_id, vector, text_id))
        # 重建索引（deque 满了会自动淘汰旧条目）
        self._rebuild_index()

    def _rebuild_index(self):
        """重建概念到索引的映射"""
        self._concept_to_entries.clear()
        for i, (cid, _, _) in enumerate(self._entries):
            self._concept_to_entries[cid].append(i)

    def get_negative_samples(self, exclude_concepts: Set[str],
                             k: int = 7) -> List[Tuple[str, torch.Tensor]]:
        """获取负样本（排除指定概念）

        返回 k 个不在 exclude_concepts 中的概念及其向量。
        使用随机采样。
        """
        candidates = []
        for i, (cid, vec, _) in enumerate(self._entries):
            if cid not in exclude_concepts:
                candidates.append((cid, vec))

        if not candidates:
            return []

        # 随机采样
        k = min(k, len(candidates))
        return random.sample(candidates, k)

    def get_hard_negatives(self, anchor: torch.Tensor,
                          exclude_concepts: Set[str],
                          k: int = 7) -> List[Tuple[str, torch.Tensor]]:
        """获取硬负例 — 与 anchor 最相似但不应该相关的概念

        硬负例挖掘是对比学习的关键：
        普通负样本太容易区分（cos≈0），梯度信号弱。
        硬负样本与 anchor 相似度高但实际不相关，提供更强的梯度。
        """
        candidates = []
        anchor_cpu = anchor.detach().cpu()  # 统一到 CPU 做比较
        for i, (cid, vec, _) in enumerate(self._entries):
            if cid not in exclude_concepts:
                sim = F.cosine_similarity(
                    anchor_cpu.unsqueeze(0), vec.unsqueeze(0)
                ).item()
                candidates.append((cid, vec, sim))

        if not candidates:
            return []

        # 按相似度降序排列，取 top-k（最相似的负样本 = 最难的负样本）
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(cid, vec) for cid, vec, _ in candidates[:k]]

    def get_all_vectors(self) -> Optional[torch.Tensor]:
        """获取所有向量（用于批量计算）"""
        if not self._entries:
            return None
        return torch.stack([vec for _, vec, _ in self._entries])

    def get_concept_ids(self) -> List[str]:
        """获取所有概念ID"""
        return [cid for cid, _, _ in self._entries]

    def __len__(self):
        return len(self._entries)


class ContrastiveTrainer:
    """对比学习训练器

    使用方式：
        trainer = ContrastiveTrainer(encoder, dim=128)

        # 每学习一条文本时：
        concepts = ["数学", "研究", "数量"]  # 从文本中提取的概念
        cooccurrence_pairs = [("数学", "研究"), ("研究", "数量")]  # 共现对
        loss = trainer.train_step(concepts, cooccurrence_pairs)

        # 损失反向传播更新编码器

    核心算法：
        1. 正样本对构建：
           - 同文本概念对（所有组合）
           - 统计学习的共现关系（PMI > 0 的概念对）

        2. 负样本对构建：
           - 内存银行中的随机概念
           - 硬负例：与 anchor 相似但不相关的概念

        3. InfoNCE 损失：
           对每个 anchor i，正样本 j，计算：
           L_i = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))

           分子 = 与正样本的相似度（应该大）
           分母 = 与所有样本（正+负）的相似度之和
    """

    def __init__(self, encoder: nn.Module, dim: int = 128,
                 temperature: float = 0.07,
                 memory_bank_size: int = 512,
                 negatives_per_positive: int = 7,
                 learning_rate: float = 5e-4,
                 device: str = 'cpu'):
        self.encoder = encoder
        self.dim = dim
        self.temperature = temperature
        self.negatives_per_positive = negatives_per_positive
        self.device = device

        # 内存银行
        self.memory_bank = MemoryBank(
            capacity=memory_bank_size,
            dim=dim,
        )

        # 统计
        self.stats = ContrastiveStats()

        # 文本计数器（用于跟踪 text_id）
        self._text_counter = 0

        # 当前文本的概念列表（用于构建正样本）
        self._current_text_concepts: List[str] = []

    def _encode(self, text: str) -> torch.Tensor:
        """编码文本为向量（带梯度）"""
        vec = self.encoder(text)
        return F.normalize(vec, p=2, dim=0)

    def _encode_no_grad(self, text: str) -> torch.Tensor:
        """编码文本为向量（不带梯度，用于内存银行）"""
        with torch.no_grad():
            vec = self.encoder(text)
        return F.normalize(vec.detach().cpu(), p=2, dim=0)

    def build_positive_pairs(self, concepts: List[str],
                             cooccurrence_pairs: List[Tuple[str, str]] = None) -> List[Tuple[str, str]]:
        """构建正样本对

        正样本来源：
        1. 同一文本中出现的所有概念对（局部共现）
        2. 统计学习的共现关系（全局共现，PMI > 0）

        返回去重的正样本对列表。
        """
        positive_pairs = set()

        # 来源1：同文本概念对
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                positive_pairs.add((concepts[i], concepts[j]))

        # 来源2：统计学习的共现关系
        if cooccurrence_pairs:
            for a, b in cooccurrence_pairs:
                if a != b:
                    positive_pairs.add((a, b))

        return list(positive_pairs)

    def info_nce_loss(self, anchor_vec: torch.Tensor,
                      positive_vecs: List[torch.Tensor],
                      negative_vecs: List[torch.Tensor]) -> torch.Tensor:
        """计算 InfoNCE 对比损失

        L = -log(exp(sim(anchor, pos)/τ) / Σ exp(sim(anchor, all)/τ))

        Args:
            anchor_vec: 锚点向量 (dim,)，带梯度
            positive_vecs: 正样本向量列表，每个 (dim,)
            negative_vecs: 负样本向量列表，每个 (dim,)

        Returns:
            标量损失（带梯度）
        """
        if not positive_vecs or not negative_vecs:
            return torch.tensor(0.0, device=self.device)

        # 所有正样本相似度 (P,)
        pos_sims = torch.stack([
            F.cosine_similarity(anchor_vec.unsqueeze(0), p.unsqueeze(0))
            for p in positive_vecs
        ]) / self.temperature

        # 所有负样本相似度 (N,)
        neg_sims = torch.stack([
            F.cosine_similarity(anchor_vec.unsqueeze(0), n.unsqueeze(0))
            for n in negative_vecs
        ]) / self.temperature

        # 记录统计
        with torch.no_grad():
            pos_mean = pos_sims.mean().item() * self.temperature
            neg_mean = neg_sims.mean().item() * self.temperature
            self.stats.positive_sims.append(pos_mean)
            self.stats.negative_sims.append(neg_mean)
            self.stats.last_cos_gap = pos_mean - neg_mean

        # InfoNCE: 对每个正样本计算 -log(exp(pos) / (exp(pos) + Σexp(neg)))
        # 使用 log-sum-exp 数值稳定计算
        total_loss = torch.tensor(0.0, device=self.device)

        for pos_sim in pos_sims:
            # logits = [pos_sim, neg_sim_1, neg_sim_2, ...]
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])

            # log-softmax = logits - logsumexp(logits)
            log_softmax = pos_sim - torch.logsumexp(logits, dim=0)

            total_loss = total_loss - log_softmax

        # 平均
        loss = total_loss / len(pos_sims)

        return loss

    def train_step(self, concepts: List[str],
                   cooccurrence_pairs: List[Tuple[str, str]] = None,
                   optimizer: torch.optim.Optimizer = None) -> Optional[float]:
        """执行一步对比学习训练

        这是核心方法，在每条文本学习后调用。

        关键设计：先填充内存银行，再训练。
        - 前几次调用：只有概念被推入内存银行，不训练（冷启动）
        - 内存银行 >= 10 个概念后：开始对比学习训练

        Args:
            concepts: 当前文本的概念列表
            cooccurrence_pairs: 统计学习发现的共现关系
            optimizer: 编码器优化器

        Returns:
            损失值（float），如果无法训练返回 None
        """
        if len(concepts) < 2:
            return None

        self._text_counter += 1

        # ===== 步骤0：先编码概念并填充内存银行（冷启动关键！）=====
        concept_vecs_no_grad = {}
        for c in concepts:
            concept_vecs_no_grad[c] = self._encode_no_grad(c)
            self.memory_bank.push(c, concept_vecs_no_grad[c], self._text_counter)

        # 也填充共现关系中的概念
        if cooccurrence_pairs:
            for a, b in cooccurrence_pairs:
                for c in [a, b]:
                    if c not in concept_vecs_no_grad:
                        concept_vecs_no_grad[c] = self._encode_no_grad(c)
                        self.memory_bank.push(c, concept_vecs_no_grad[c], self._text_counter)

        # 冷启动检查：内存银行不够大，无法构建足够负样本
        if len(self.memory_bank) < 10:
            return None

        # ===== 步骤1：构建正样本对 =====
        positive_pairs = self.build_positive_pairs(concepts, cooccurrence_pairs)
        if not positive_pairs:
            return None

        # ===== 步骤2：带梯度编码概念（用于损失计算）=====
        self.encoder.train()
        if optimizer:
            optimizer.zero_grad()

        # 概念向量缓存
        concept_vecs_grad = {}  # 带梯度
        concept_vecs_detached = {}  # detach 后

        for c in concepts:
            if c not in concept_vecs_grad:
                vec = self._encode(c)
                concept_vecs_grad[c] = vec
                concept_vecs_detached[c] = vec.detach()

        # 编码正样本中的额外概念
        for a, b in positive_pairs:
            for c in [a, b]:
                if c not in concept_vecs_grad:
                    concept_vecs_grad[c] = self._encode(c)
                    concept_vecs_detached[c] = concept_vecs_grad[c].detach()

        # ===== 步骤3：对每个正样本对计算 InfoNCE 损失 =====
        total_loss = torch.tensor(0.0, device=self.device)
        valid_pairs = 0

        # 正样本关系映射
        pos_map: Dict[str, Set[str]] = defaultdict(set)
        for a, b in positive_pairs:
            pos_map[a].add(b)
            pos_map[b].add(a)

        for anchor_id, positive_id in positive_pairs[:10]:
            anchor_vec = concept_vecs_grad[anchor_id]

            # 正样本向量
            pos_vecs = [concept_vecs_detached[positive_id]]

            # 负样本：内存银行中不是正样本的概念
            exclude = pos_map.get(anchor_id, set()) | {anchor_id}

            # 混合策略：50% 随机 + 50% 硬负例
            n_random = max(1, self.negatives_per_positive // 2)
            n_hard = self.negatives_per_positive - n_random

            neg_samples = self.memory_bank.get_negative_samples(
                exclude_concepts=exclude, k=n_random
            )

            # 硬负例（如果内存银行足够大）
            if len(self.memory_bank) > 20:
                hard_negs = self.memory_bank.get_hard_negatives(
                    anchor=concept_vecs_detached[anchor_id],
                    exclude_concepts=exclude,
                    k=n_hard,
                )
                neg_samples.extend(hard_negs)

            if not neg_samples:
                continue

            neg_vecs = [vec.to(self.device) for _, vec in neg_samples]

            # 计算 InfoNCE 损失
            loss = self.info_nce_loss(anchor_vec, pos_vecs, neg_vecs)
            total_loss = total_loss + loss
            valid_pairs += 1

        if valid_pairs == 0:
            self.encoder.eval()
            return None

        avg_loss = total_loss / valid_pairs

        # ===== 步骤4：反向传播 =====
        if avg_loss.requires_grad and avg_loss.item() > 1e-8:
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            if optimizer:
                optimizer.step()

        self.encoder.eval()

        # 5. 更新统计（概念已在步骤0中推入内存银行）
        loss_val = avg_loss.item()
        self.stats.total_updates += 1
        self.stats.total_loss += loss_val
        self.stats.avg_loss = self.stats.total_loss / self.stats.total_updates

        return loss_val

    def get_stats(self) -> Dict:
        """获取训练统计"""
        stats = {
            'total_updates': self.stats.total_updates,
            'avg_loss': round(self.stats.avg_loss, 4),
            'memory_bank_size': len(self.memory_bank),
            'last_cos_gap': round(self.stats.last_cos_gap, 4),
        }
        if self.stats.positive_sims:
            stats['avg_positive_sim'] = round(
                sum(self.stats.positive_sims[-50:]) / len(self.stats.positive_sims[-50:]), 4
            )
        if self.stats.negative_sims:
            stats['avg_negative_sim'] = round(
                sum(self.stats.negative_sims[-50:]) / len(self.stats.negative_sims[-50:]), 4
            )
        return stats

    def replay_epochs(self, positive_pairs: List[Tuple[str, str]],
                      optimizer: torch.optim.Optimizer = None,
                      n_epochs: int = 5) -> Dict:
        """回放训练 — 用累积的高质量概念对做多轮对比训练

        解决在线学习数据不足的问题：
        - 在线学习时每个文本只训练一次，梯度信号太弱
        - 回放机制反复使用已积累的概念对，等效于扩大了训练数据

        类比人脑的睡眠巩固：白天学到的经验在夜间反复回放加强。

        Args:
            positive_pairs: 高质量正样本对列表 [(a, b), ...]
            optimizer: 编码器优化器
            n_epochs: 回放轮数

        Returns:
            训练统计
        """
        if not positive_pairs or len(self.memory_bank) < 10:
            return {'epochs': 0, 'total_loss': 0.0}

        all_concepts = list(set(c for pair in positive_pairs for c in pair))
        epoch_losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            valid_count = 0

            # 随机打乱正样本对
            shuffled_pairs = list(positive_pairs)
            random.shuffle(shuffled_pairs)

            for anchor_id, positive_id in shuffled_pairs[:20]:  # 每轮最多20对
                # 编码 anchor（带梯度）
                self.encoder.train()
                if optimizer:
                    optimizer.zero_grad()

                anchor_vec = self._encode(anchor_id)
                pos_vec = self._encode(positive_id).detach()

                # 负样本
                exclude = {anchor_id, positive_id}
                neg_samples = self.memory_bank.get_negative_samples(
                    exclude_concepts=exclude, k=self.negatives_per_positive
                )

                # 硬负例
                if len(self.memory_bank) > 20:
                    hard_negs = self.memory_bank.get_hard_negatives(
                        anchor=anchor_vec.detach().cpu(),
                        exclude_concepts=exclude,
                        k=max(1, self.negatives_per_positive // 2),
                    )
                    neg_samples.extend(hard_negs)

                if not neg_samples:
                    continue

                neg_vecs = [vec.to(self.device) for _, vec in neg_samples]

                # InfoNCE 损失
                loss = self.info_nce_loss(anchor_vec, [pos_vec], neg_vecs)

                if loss.requires_grad and loss.item() > 1e-8:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    if optimizer:
                        optimizer.step()

                epoch_loss += loss.item()
                valid_count += 1

            if valid_count > 0:
                epoch_losses.append(epoch_loss / valid_count)

            self.encoder.eval()

            # 每轮后更新内存银行中的向量（反映编码器变化）
            with torch.no_grad():
                for cid in all_concepts:
                    new_vec = F.normalize(self.encoder(cid).detach().cpu(), p=2, dim=0)
                    self.memory_bank.push(cid, new_vec)

        self.stats.total_updates += n_epochs

        return {
            'epochs': n_epochs,
            'total_loss': round(sum(epoch_losses) / max(1, len(epoch_losses)), 4),
            'concepts_used': len(all_concepts),
            'pairs_used': len(positive_pairs),
        }

    def diagnose_encoding_quality(self, test_pairs: List[Tuple[str, str, str]]) -> Dict:
        """诊断编码质量

        test_pairs: [(text_a, text_b, category), ...]
        category: 'related' | 'unrelated'

        返回：
        - 各对的余弦相似度
        - related 组 vs unrelated 组的平均相似度
        - GAP（区分度指标）
        """
        related_sims = []
        unrelated_sims = []
        pair_results = []

        for a, b, category in test_pairs:
            with torch.no_grad():
                va = F.normalize(self.encoder(a), p=2, dim=0)
                vb = F.normalize(self.encoder(b), p=2, dim=0)
                sim = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()

            pair_results.append({
                'pair': f"('{a}', '{b}')",
                'category': category,
                'similarity': round(sim, 4),
            })

            if category == 'related':
                related_sims.append(sim)
            else:
                unrelated_sims.append(sim)

        avg_related = sum(related_sims) / max(1, len(related_sims))
        avg_unrelated = sum(unrelated_sims) / max(1, len(unrelated_sims))
        gap = avg_related - avg_unrelated

        return {
            'pairs': pair_results,
            'avg_related_sim': round(avg_related, 4),
            'avg_unrelated_sim': round(avg_unrelated, 4),
            'gap': round(gap, 4),
            'quality': 'excellent' if gap > 0.4 else 'good' if gap > 0.2 else 'weak' if gap > 0.1 else 'poor',
        }
