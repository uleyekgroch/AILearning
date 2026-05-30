#!/usr/bin/env python3
"""睡眠回放巩固 (Sleep Replay Consolidation)

基于论文:
- Nature Communications 2022: "Sleep-like unsupervised replay reduces catastrophic forgetting"
- NeuroDream 2025: "Sleep-Inspired Memory Consolidation Framework"
- Science 2024: "The time course and organization of hippocampal replay"
- eLife 2024: "A unifying account of replay as context-driven memory reactivation"

核心思想:
  人类睡眠期间，海马体重放白天经历：
  1. 慢波睡眠(SWS): 按时间顺序回放，将具体事件从海马体转移到皮层
  2. REM睡眠: 随机组合回放，创造性重组记忆，发现新联系
  3. 纺锤波(Spindles): 选择性巩固重要记忆

  生成式回放(Generative Replay):
  不是简单复制旧记忆，而是用当前模型重新生成旧数据的表征。
  这样新知识不会覆盖旧知识（防止灾难性遗忘），
  同时旧记忆被"翻译"到新的表征空间。

  对学习系统的意义:
  - consolidate()不只是压缩，而是真正的回放+重组
  - 防止学新忘旧（灾难性遗忘的核心解决方案）
  - 通过REM阶段的随机组合发现隐含联系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import random
import math


@dataclass
class ReplayEpisode:
    """回放片段"""
    content: str                # 原始文本
    embedding: torch.Tensor     # 嵌入向量
    importance: float           # 重要性分数
    access_count: int           # 被访问次数
    timestamp: int              # 学习时间
    replay_count: int = 0       # 被回放次数
    strengthened: bool = False  # 是否已巩固


class GenerativeReplayModel(nn.Module):
    """生成式回放模型

    学习旧数据的分布，在回放时生成近似旧数据的表征，
    而不是存储原始数据。这样新参数不会与旧数据冲突。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # 自编码器：学习压缩+重建
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码+解码（重建）"""
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def generate(self, num_samples: int = 1) -> torch.Tensor:
        """从学习到的分布中生成样本"""
        # 从标准正态采样，解码为记忆表征
        z = torch.randn(num_samples, self.d_model // 4)
        return self.decode(z)


class SleepReplaySystem:
    """睡眠回放巩固系统

    模拟人类睡眠的三个阶段：
    1. SWS（慢波睡眠）: 按重要性回放，转移到长期记忆
    2. REM（快速眼动）: 随机组合，创造性发现
    3. Spindles（纺锤波）: 选择性加强高重要性记忆

    使用方法：
    1. 学习时调用 record_episode() 记录经历
    2. 定期调用 sleep_cycle() 执行完整睡眠周期
    3. 系统自动选择回放策略并更新记忆
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 记忆存储（海马体: 快速存储区）
        self.episodic_memory: List[ReplayEpisode] = []

        # 长期记忆（皮层: 稳定存储区）
        self.semantic_memory: List[ReplayEpisode] = []

        # 生成式回放模型
        self.gen_model = GenerativeReplayModel(d_model).to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.gen_model.parameters(), lr=1e-4)

        # 回放统计
        self.stats = {
            'sws_replays': 0,
            'rem_combinations': 0,
            'spindle_consolidations': 0,
            'episodes_forgotten': 0,
            'creative_links_found': 0,
        }

        # 最大容量
        self.max_episodic = 10000
        self.max_semantic = 50000

        # 时间戳计数器
        self._time_counter = 0

    def record_episode(self, text: str, embedding: torch.Tensor,
                       importance: float = 0.5) -> None:
        """记录一个学习经历到海马体（情节记忆）

        Args:
            text: 原始文本
            embedding: 文本的嵌入向量
            importance: 重要性分数（0-1）
        """
        self._time_counter += 1

        episode = ReplayEpisode(
            content=text,
            embedding=embedding.detach().to(self.device),
            importance=importance,
            access_count=0,
            timestamp=self._time_counter,
        )

        self.episodic_memory.append(episode)

        # 超过容量时触发部分巩固
        if len(self.episodic_memory) > self.max_episodic:
            self._partial_consolidation(batch_size=100)

        # 训练生成式模型
        self._train_gen_model(embedding.detach())

    def _train_gen_model(self, embedding: torch.Tensor):
        """训练生成式回放模型"""
        self.gen_model.train()
        emb = embedding.to(self.device)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        # 自编码器重建
        reconstructed = self.gen_model(emb)
        loss = F.mse_loss(reconstructed, emb)

        self.gen_optimizer.zero_grad()
        loss.backward()
        self.gen_optimizer.step()

    def sleep_cycle(self, knowledge_graph=None, encoder_fn=None) -> Dict:
        """执行完整睡眠周期

        模拟一个晚上的睡眠：SWS → REM → Spindles
        每个阶段有不同的回放策略。

        Args:
            knowledge_graph: 知识图谱（用于建立新联系）
            encoder_fn: 编码函数（用于生成新表征）

        Returns:
            统计信息
        """
        if len(self.episodic_memory) < 2:
            return {'status': 'insufficient_memories', 'stats': self.stats}

        results = {}

        # 阶段1: SWS — 按重要性回放
        sws_result = self._sws_replay(knowledge_graph)
        results['sws'] = sws_result

        # 阶段2: REM — 创造性组合
        rem_result = self._rem_replay(knowledge_graph, encoder_fn)
        results['rem'] = rem_result

        # 阶段3: Spindles — 选择性巩固
        spindle_result = self._spindle_consolidation()
        results['spindles'] = spindle_result

        return results

    def _sws_replay(self, knowledge_graph=None) -> Dict:
        """慢波睡眠回放 — 按时间顺序+重要性回放

        特征:
        - 优先回放高重要性+高不确定性的记忆
        - 将情节记忆转移到语义记忆
        - 训练生成式模型重建旧记忆
        """
        if not self.episodic_memory:
            return {'replayed': 0}

        # 按重要性排序（高→低）
        scored = []
        for ep in self.episodic_memory:
            # 重要性权重 + 新鲜度衰减
            freshness = 1.0 / (1.0 + ep.replay_count * 0.1)
            score = ep.importance * freshness
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 回放前N个
        replay_count = min(len(scored), max(10, len(scored) // 5))
        replayed = []

        for i in range(replay_count):
            score, episode = scored[i]
            episode.replay_count += 1
            episode.access_count += 1
            self.stats['sws_replays'] += 1

            # 生成式回放：用当前模型重建旧记忆
            with torch.no_grad():
                emb = episode.embedding.unsqueeze(0) if episode.embedding.dim() == 1 else episode.embedding.unsqueeze(0)
                reconstructed = self.gen_model(emb.to(self.device)).squeeze(0)

            # 更新嵌入为重建版本（适应新的表征空间）
            episode.embedding = reconstructed

            # 检查是否可以转移到语义记忆
            if episode.replay_count >= 3:
                episode.strengthened = True
                # 转移到皮层（语义记忆）
                self.semantic_memory.append(episode)
                replayed.append(episode.content[:50])

        # 清理已转移的记忆
        self.episodic_memory = [
            ep for ep in self.episodic_memory if not ep.strengthened
        ]

        return {
            'replayed': replay_count,
            'transferred_to_semantic': len(replayed),
            'episodic_remaining': len(self.episodic_memory),
        }

    def _rem_replay(self, knowledge_graph=None, encoder_fn=None) -> Dict:
        """REM回放 — 创造性随机组合

        特征:
        - 随机配对不同记忆
        - 通过相似度发现隐含联系
        - 模拟"梦中灵感"

        这是创造力的重要来源：把不相关的记忆组合在一起，
        有时会发现意想不到的联系。
        """
        if len(self.episodic_memory) + len(self.semantic_memory) < 2:
            return {'combinations': 0}

        # 合并所有记忆用于组合
        all_memories = self.episodic_memory + self.semantic_memory
        num_combinations = min(20, len(all_memories) // 2)
        links_found = 0

        for _ in range(num_combinations):
            # 随机选择两个不相关的记忆
            idx_a, idx_b = random.sample(range(len(all_memories)), 2)
            mem_a = all_memories[idx_a]
            mem_b = all_memories[idx_b]

            # 计算跨域相似度
            emb_a = mem_a.embedding
            emb_b = mem_b.embedding
            if emb_a.dim() == 1:
                emb_a = emb_a.unsqueeze(0)
            if emb_b.dim() == 1:
                emb_b = emb_b.unsqueeze(0)

            similarity = F.cosine_similarity(emb_a.to(self.device),
                                             emb_b.to(self.device)).item()

            # 发现新联系：中等相似度的配对最有价值
            # 太相似 = 已知联系，太不相似 = 无关
            if 0.3 < similarity < 0.7:
                links_found += 1
                self.stats['creative_links_found'] += 1

                # 如果有知识图谱，建立新联系
                if knowledge_graph is not None and encoder_fn is not None:
                    self._discover_link(mem_a, mem_b, similarity,
                                       knowledge_graph, encoder_fn)

            self.stats['rem_combinations'] += 1

        return {
            'combinations': num_combinations,
            'links_found': links_found,
        }

    def _discover_link(self, mem_a: ReplayEpisode, mem_b: ReplayEpisode,
                       similarity: float, knowledge_graph, encoder_fn):
        """发现并记录两个记忆之间的隐含联系"""
        try:
            # 提取关键实体
            from src.knowledge.entity import Entity
            from src.knowledge.relation import Relation

            # 用文本前20字符作为实体ID
            entity_a = mem_a.content[:20].strip()
            entity_b = mem_b.content[:20].strip()

            if len(entity_a) < 2 or len(entity_b) < 2:
                return

            # 添加关联关系
            knowledge_graph.add_relation(Relation(
                source_id=entity_a,
                target_id=entity_b,
                type=f'隐含关联({similarity:.2f})',
                confidence=similarity * 0.5,  # 较低置信度
            ))
        except Exception:
            pass

    def _spindle_consolidation(self) -> Dict:
        """纺锤波巩固 — 选择性加强重要记忆

        特征:
        - 加强高重要性记忆的嵌入
        - 衰减低重要性记忆（遗忘曲线）
        - 计算记忆保留率
        """
        consolidated = 0
        forgotten = 0

        # 处理语义记忆（长期）
        to_keep = []
        for ep in self.semantic_memory:
            # Ebbinghaus遗忘曲线: R = e^(-t/S)
            # S = strength (由importance和replay_count决定)
            strength = ep.importance * (1 + ep.replay_count * 0.5)
            time_elapsed = self._time_counter - ep.timestamp
            retention = math.exp(-time_elapsed / (strength * 1000))

            if retention > 0.1:
                # 加强：提高嵌入向量的范数
                with torch.no_grad():
                    ep.embedding = ep.embedding * (1.0 + 0.01 * retention)
                to_keep.append(ep)
                consolidated += 1
            else:
                forgotten += 1
                self.stats['episodes_forgotten'] += 1

        self.semantic_memory = to_keep

        # 限制语义记忆容量
        if len(self.semantic_memory) > self.max_semantic:
            self.semantic_memory.sort(key=lambda ep: ep.importance, reverse=True)
            self.semantic_memory = self.semantic_memory[:self.max_semantic]

        self.stats['spindle_consolidations'] += consolidated

        return {
            'consolidated': consolidated,
            'forgotten': forgotten,
            'semantic_memory_size': len(self.semantic_memory),
            'episodic_memory_size': len(self.episodic_memory),
        }

    def _partial_consolidation(self, batch_size: int = 100):
        """部分巩固：当情节记忆接近满时触发"""
        # 按重要性排序
        self.episodic_memory.sort(key=lambda ep: ep.importance, reverse=True)

        # 转移最不重要的
        to_transfer = self.episodic_memory[-batch_size:]
        self.episodic_memory = self.episodic_memory[:-batch_size]

        for ep in to_transfer:
            if ep.importance > 0.3:
                ep.strengthened = True
                self.semantic_memory.append(ep)

    def generate_replay_batch(self, batch_size: int = 16) -> torch.Tensor:
        """生成回放批次 — 用于防止灾难性遗忘

        在学习新知识时，同时回放旧记忆的生成版本，
        确保新学习不会覆盖旧知识。

        Returns:
            batch: (batch_size, d_model) 的回放嵌入
        """
        self.gen_model.eval()
        with torch.no_grad():
            batch = self.gen_model.generate(batch_size)
        return batch

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'episodic_memory': len(self.episodic_memory),
            'semantic_memory': len(self.semantic_memory),
        }
