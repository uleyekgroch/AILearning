#!/usr/bin/env python3
"""跨域迁移学习 (Cross-Domain Transfer Learning)

基于论文:
- Science China 2025: "Temporal development mechanisms enable cross-domain
  continual learning"
- Cognitive Science 2020: "Relation learning in a neurocomputational
  architecture supports analogical inference"
- AAAI 2011: "Transfer Learning by Structural Analogy"

核心思想:
  人类最强大的学习能力之一是跨域迁移：
  "太阳系像原子" → 把宏观物理的直觉迁移到微观世界
  "心脏像泵"     → 把机械原理迁移到生物系统

  迁移学习的核心机制:
  1. 关系抽象(Relational Abstraction): 提取关系结构，忽略表面特征
  2. 结构映射(Structure Mapping): 找到源域和目标域之间的对应关系
  3. 投射迁移(Projection Transfer): 将源域的知识投射到目标域

  Gentner的结构映射理论(Structure-Mapping Theory):
  - 相同关系优先：关系结构匹配比表面特征匹配更重要
  - 系统性偏好：连接多个关系的映射优于孤立映射
  - 一对一映射：源域和目标域的元素一一对应

  对学习系统的意义:
  - 在A领域学到的抽象关系自动迁移到B领域
  - 减少需要的训练数据（不需要每个领域从头学）
  - 支持创造性推理（"从未直接学过，但可以类比推出"）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math


@dataclass
class RelationalPattern:
    """关系模式

    抽象的关系结构，不依赖具体实体。
    例如: "A 导致 B" 可以匹配 "下雨导致地面湿" 和 "吸烟导致肺癌"
    """
    pattern_id: str
    relation_type: str               # 关系类型（如"导致"、"组成"）
    source_schemas: List[str]         # 来自哪些领域的图式
    confidence: float                 # 迁移置信度
    usage_count: int = 0              # 使用次数
    success_count: int = 0            # 成功次数


@dataclass
class TransferHypothesis:
    """迁移假设

    从源域到目标域的一个迁移假设。
    例如: "力导致加速度" → "压力导致产量变化"
    """
    source_domain: str
    target_domain: str
    source_relation: Tuple[str, str, str]  # (subject, relation, obj)
    projected_relation: Tuple[str, str, str]
    confidence: float
    source_evidence: int = 0


class CrossDomainTransfer:
    """跨域迁移学习系统

    工作流程:
    1. 积累(Accumulate): 从学习中收集跨域的关系模式
    2. 抽象(Abstract): 将具体关系抽象为关系模式
    3. 匹配(Match): 找到可以迁移的源域-目标域对
    4. 迁移(Transfer): 将源域知识投射到目标域
    5. 验证(Verify): 检验迁移结果是否正确
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 关系模式库（跨域抽象）
        self.patterns: Dict[str, RelationalPattern] = {}

        # 领域到嵌入的映射
        self.domain_embeddings: Dict[str, torch.Tensor] = {}

        # 迁移历史
        self.transfer_history: List[TransferHypothesis] = []

        # 关系类型到模式的映射
        self.relation_to_patterns: Dict[str, List[str]] = {}

        # 统计
        self.stats = {
            'patterns_learned': 0,
            'transfers_attempted': 0,
            'transfers_succeeded': 0,
            'knowledge_gained': 0,
        }

    def learn_relation(self, relation_type: str, domain: str,
                       confidence: float = 0.5):
        """学习一个关系模式

        当系统在任何领域学到新关系时调用。
        关系类型会被抽象存储，后续可以迁移到其他领域。
        """
        # 创建或更新关系模式
        pattern_key = relation_type
        if pattern_key not in self.patterns:
            pattern = RelationalPattern(
                pattern_id=f"rp_{self.stats['patterns_learned']}",
                relation_type=relation_type,
                source_schemas=[domain],
                confidence=confidence,
            )
            self.patterns[pattern_key] = pattern
            self.stats['patterns_learned'] += 1

            # 更新索引
            if relation_type not in self.relation_to_patterns:
                self.relation_to_patterns[relation_type] = []
            self.relation_to_patterns[relation_type].append(pattern.pattern_id)
        else:
            # 更新已有模式
            pattern = self.patterns[pattern_key]
            if domain not in pattern.source_schemas:
                pattern.source_schemas.append(domain)
            pattern.confidence = min(1.0, pattern.confidence + 0.05)
            pattern.usage_count += 1

    def register_domain(self, domain: str, embedding: torch.Tensor):
        """注册一个知识领域"""
        self.domain_embeddings[domain] = embedding.detach().to(self.device)

    def find_transfer_candidates(self, target_domain: str,
                                  target_embedding: torch.Tensor,
                                  top_k: int = 3) -> List[TransferHypothesis]:
        """找到可以从其他领域迁移到目标领域的知识

        核心算法:
        1. 计算目标领域与所有已知领域的相似度
        2. 找到最相似的源领域
        3. 提取源领域中的关系模式
        4. 生成迁移假设
        """
        if not self.domain_embeddings:
            return []

        target_emb = target_embedding.to(self.device)
        if target_emb.dim() == 1:
            target_emb = target_emb.unsqueeze(0)

        # 计算领域相似度
        domain_sims = []
        for domain, domain_emb in self.domain_embeddings.items():
            if domain == target_domain:
                continue
            d_emb = domain_emb.unsqueeze(0) if domain_emb.dim() == 1 else domain_emb
            sim = F.cosine_similarity(target_emb, d_emb).item()
            if sim > 0.3:
                domain_sims.append((domain, sim))

        domain_sims.sort(key=lambda x: x[1], reverse=True)

        # 为每个相似领域生成迁移假设
        hypotheses = []
        for source_domain, domain_sim in domain_sims[:top_k]:
            for pattern_key, pattern in self.patterns.items():
                if source_domain in pattern.source_schemas:
                    # 计算迁移置信度 = 领域相似度 × 模式置信度
                    transfer_confidence = domain_sim * pattern.confidence
                    if transfer_confidence > 0.3:
                        hypothesis = TransferHypothesis(
                            source_domain=source_domain,
                            target_domain=target_domain,
                            source_relation=('', pattern.relation_type, ''),
                            projected_relation=('', pattern.relation_type, ''),
                            confidence=transfer_confidence,
                            source_evidence=pattern.usage_count,
                        )
                        hypotheses.append(hypothesis)

        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:top_k]

    def execute_transfer(self, hypothesis: TransferHypothesis,
                         knowledge_graph=None, encoder_fn=None) -> Dict:
        """执行迁移

        将源域的关系模式投射到目标域，在知识图谱中创建新关系。

        Returns:
            迁移结果
        """
        self.stats['transfers_attempted'] += 1

        # 这里需要具体的实体映射
        # 简化版：只记录迁移模式，不做实体级投射
        result = {
            'source': hypothesis.source_domain,
            'target': hypothesis.target_domain,
            'relation_type': hypothesis.source_relation[1],
            'confidence': hypothesis.confidence,
            'transferred': False,
        }

        # 如果置信度足够高，可以在知识图谱中创建"推测性"关系
        if hypothesis.confidence > 0.5 and knowledge_graph is not None:
            try:
                from src.knowledge.relation import Relation
                knowledge_graph.add_relation(Relation(
                    source_id=hypothesis.target_domain,
                    target_id=hypothesis.source_domain,
                    type=f'类比迁移({hypothesis.source_relation[1]})',
                    confidence=hypothesis.confidence * 0.3,  # 低置信度（是推测）
                ))
                result['transferred'] = True
                self.stats['knowledge_gained'] += 1
            except Exception:
                pass

        self.transfer_history.append(hypothesis)
        return result

    def verify_transfer(self, hypothesis: TransferHypothesis,
                        verified: bool) -> None:
        """验证迁移结果

        如果迁移成功，增强该关系模式的置信度。
        如果失败，降低置信度。
        """
        pattern_key = hypothesis.source_relation[1]
        if pattern_key in self.patterns:
            pattern = self.patterns[pattern_key]
            if verified:
                pattern.success_count += 1
                pattern.confidence = min(1.0, pattern.confidence + 0.1)
                self.stats['transfers_succeeded'] += 1
            else:
                pattern.confidence = max(0.1, pattern.confidence - 0.05)

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'domains_registered': len(self.domain_embeddings),
            'transfer_success_rate': (
                self.stats['transfers_succeeded'] / max(1, self.stats['transfers_attempted'])
            ),
        }
