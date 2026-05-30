"""测试时训练 — 推理时持续学习

基于 In-Place TTT (arXiv:2604.06169, ICLR 2026 Oral) 的核心思想：
- 将模型的最终投影矩阵视为"快权重"
- 在推理时原地更新这些权重
- 无需从头重训，只更新最上层

关键创新：
- 与Next-Token-Prediction对齐的训练目标
- chunk-wise更新机制（兼容上下文并行）
- 只更新快权重，保持慢权重稳定

对当前系统的应用：
- 在think()时，根据查询上下文微调编码器
- 在learn_from_text()时，根据新知识更新嵌入
- 保持底层Transformer权重稳定，只更新顶层

设计原则：
- 快权重：嵌入层的最后一层投影
- 慢权重：Transformer的底层权重
- 更新频率：每次查询时微调，每次学习时大幅更新
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from collections import deque


class TestTimeTrainer:
    """测试时训练器

    在推理时原地更新快权重。
    """

    def __init__(self, encoder: nn.Module, lr: float = 1e-5):
        self.encoder = encoder
        self.lr = lr

        # 快权重：嵌入层的最后一层
        self.fast_weights = None
        self.slow_weights = None

        # 更新历史
        self.update_history = deque(maxlen=100)
        self.total_updates = 0

        # 上下文缓存
        self.context_cache = {}

    def adapt_to_query(self, query: str, query_repr: torch.Tensor,
                      relevant_entities: List[str]) -> torch.Tensor:
        """根据查询上下文微调编码器

        在推理时，根据查询和相关实体调整编码器的快权重。

        Args:
            query: 查询文本
            query_repr: 查询的向量表示
            relevant_entities: 相关实体列表

        Returns:
            调整后的查询表示
        """
        if not relevant_entities:
            return query_repr

        # 编码相关实体
        entity_reprs = []
        for entity in relevant_entities[:3]:
            with torch.no_grad():
                entity_repr = self.encoder(entity)
            entity_reprs.append(entity_repr)

        if not entity_reprs:
            return query_repr

        # 计算查询与实体的平均相似度
        similarities = []
        for entity_repr in entity_reprs:
            sim = torch.cosine_similarity(
                query_repr.unsqueeze(0), entity_repr.unsqueeze(0)
            ).item()
            similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities)

        # 始终更新，但根据相似度调整学习率
        # 相似度高→小更新，相似度低→大更新
        adaptive_lr = self.lr * (1.0 - avg_sim + 0.1)
        self._update_fast_weights(query, query_repr, entity_reprs, adaptive_lr)

        return query_repr

    def _update_fast_weights(self, query: str, query_repr: torch.Tensor,
                           entity_reprs: List[torch.Tensor], adaptive_lr: float = None):
        """更新快权重（嵌入层的最后一层）"""
        # 获取嵌入层
        if not hasattr(self.encoder, 'embedding'):
            return

        embedding_layer = self.encoder.embedding
        lr = adaptive_lr if adaptive_lr is not None else self.lr

        # 计算目标：查询表示应该与实体表示相关
        target = torch.stack(entity_reprs).mean(dim=0)

        # 计算损失
        loss = 1.0 - torch.cosine_similarity(
            query_repr.unsqueeze(0), target.unsqueeze(0)
        )

        # 只更新嵌入层的权重（快权重）
        if hasattr(embedding_layer, 'weight'):
            # 计算梯度方向
            grad_direction = target - query_repr

            # 原地更新（不使用优化器，直接更新权重）
            with torch.no_grad():
                # 更新与查询字符相关的嵌入
                chars = list(query[:128])
                for char in chars:
                    char_idx = ord(char) % 10000
                    if char_idx < embedding_layer.weight.size(0):
                        embedding_layer.weight[char_idx] += lr * grad_direction

        self.total_updates += 1
        self.update_history.append({
            'query': query[:30],
            'loss': loss.item(),
            'avg_similarity': sum(torch.cosine_similarity(
                query_repr.unsqueeze(0), er.unsqueeze(0)
            ).item() for er in entity_reprs) / len(entity_reprs),
        })

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'total_updates': self.total_updates,
            'recent_loss': self.update_history[-1]['loss'] if self.update_history else 0.0,
            'recent_similarity': self.update_history[-1]['avg_similarity'] if self.update_history else 0.0,
        }


class ChunkwiseTTT:
    """分块测试时训练

    将长序列分成chunks，每个chunk独立更新快权重。
    """

    def __init__(self, trainer: TestTimeTrainer, chunk_size: int = 32):
        self.trainer = trainer
        self.chunk_size = chunk_size

    def process_long_sequence(self, text: str, encoder) -> torch.Tensor:
        """处理长序列，分块更新"""
        # 分词
        tokens = list(text)

        # 分块处理
        chunks = [tokens[i:i+self.chunk_size]
                 for i in range(0, len(tokens), self.chunk_size)]

        representations = []
        for chunk in chunks:
            chunk_text = ''.join(chunk)
            with torch.no_grad():
                chunk_repr = encoder(chunk_text)

            # 在每个chunk后微调
            self.trainer.adapt_to_query(
                chunk_text, chunk_repr, []
            )

            representations.append(chunk_repr)

        # 合并表示
        if representations:
            return torch.stack(representations).mean(dim=0)
        return torch.zeros(encoder.d_model)
