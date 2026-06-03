"""可微分知识提取器 — 从正则到学习驱动

替代硬编码正则模式，通过梯度学习提取实体和关系。

核心组件：
1. 实体提取器 — 序列标注（BIO标签）
2. 关系分类器 — 给定实体对，预测关系类型
3. 因果发现器 — 从干预中学习因果
4. 数值提取器 — 通用数值理解

设计原则：
- 可微分：通过梯度优化
- 从交互学习：不依赖预标注数据
- 语言无关：支持中英文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# BIO标签
BIO_TAGS = ['O', 'B-ENT', 'I-ENT']
BIO_TAG_TO_ID = {tag: i for i, tag in enumerate(BIO_TAGS)}


# 关系类型
RELATION_TYPES = [
    '是', '属于', '位于', '发明', '发现', '使用', '导致',
    '温度', '长度', '重量', '时间', '颜色', '形状', '大小',
    '部分', '整体', '因果', '相似', '对比', '包含',
    '拥有', '制造', '运动', '状态', '属性', '其他',
]
REL_TO_ID = {rel: i for i, rel in enumerate(RELATION_TYPES)}


@dataclass
class ExtractedTriple:
    """提取的三元组"""
    subject: str
    relation: str
    obj: str
    confidence: float
    method: str  # 'learned' or 'regex'


class EntityExtractor(nn.Module):
    """实体提取器 — 序列标注

    使用 Transformer 编码器 + 分类头。
    每个token预测 BIO 标签。
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 分类头：预测 BIO 标签
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, len(BIO_TAGS)),
        )

        # 字符嵌入（支持更大词表）
        self.char_embedding = nn.Embedding(50000, d_model, padding_idx=0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """预测 BIO 标签

        Args:
            token_ids: (batch, seq_len) token IDs

        Returns:
            (batch, seq_len, n_tags) 标签概率
        """
        # 嵌入
        x = self.char_embedding(token_ids)

        # Transformer 编码
        x = self.encoder(x)

        # 分类
        logits = self.classifier(x)
        return F.softmax(logits, dim=-1)


class RelationClassifier(nn.Module):
    """关系分类器

    给定两个实体的表示，预测它们之间的关系类型。
    """

    def __init__(self, d_model: int = 128, n_relations: int = len(RELATION_TYPES)):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_relations),
        )

    def forward(self, subject_repr: torch.Tensor, object_repr: torch.Tensor,
                context_repr: torch.Tensor) -> torch.Tensor:
        """预测关系类型

        Args:
            subject_repr: (d_model,) 主体表示
            object_repr: (d_model,) 客体表示
            context_repr: (d_model,) 上下文表示

        Returns:
            (n_relations,) 关系概率
        """
        combined = torch.cat([subject_repr, object_repr, context_repr])
        logits = self.classifier(combined)
        return F.softmax(logits, dim=-1)


class NumericalExtractor(nn.Module):
    """数值提取器

    从文本中提取数值及其单位。
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        # 数值分类头
        self.value_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # 单位分类头
        self.unit_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 10),  # 10种单位类型
        )

    def forward(self, context_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测数值和单位

        Returns:
            value: 预测的数值
            unit_probs: 单位概率分布
        """
        value = self.value_predictor(context_repr)
        unit_probs = F.softmax(self.unit_classifier(context_repr), dim=-1)
        return value, unit_probs


class LearnableKnowledgeExtractor:
    """可微分知识提取器

    整合实体提取、关系分类、数值提取。
    通过交互学习，不依赖预标注数据。
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 模型
        self.entity_extractor = EntityExtractor(d_model).to(self.device)
        self.relation_classifier = RelationClassifier(d_model).to(self.device)
        self.numerical_extractor = NumericalExtractor(d_model).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.entity_extractor.parameters()) +
            list(self.relation_classifier.parameters()) +
            list(self.numerical_extractor.parameters()),
            lr=1e-4
        )

        # 正则回退模式（用于冷启动）
        self._regex_patterns = [
            (r'(.{1,10}?)是(.{2,30})', '是'),
            (r'(.{1,10}?)属于(.{2,20})', '属于'),
            (r'(.{1,10}?)位于(.{2,20})', '位于'),
            (r'(.{1,10}?)发明了?(.{2,20})', '发明'),
            (r'(.{1,10}?)发现了?(.{2,20})', '发现'),
            (r'(.{1,10}?)使用(.{2,20})', '使用'),
            (r'(.{1,10}?)导致(.{2,20})', '导致'),
            (r'(.{1,10}?)在(\d+[\.\d]*摄氏度.{1,10})', '温度'),
        ]

        # 学习统计
        self._extraction_count = 0
        self._learned_patterns = []

    def extract_triples(self, text: str, text_repr: torch.Tensor,
                       encoder=None) -> List[ExtractedTriple]:
        """提取三元组

        混合策略：
        1. 冷启动：使用正则模式
        2. 有足够数据后：使用学习到的模型
        3. 两种方法的结果合并去重
        """
        triples = []

        # 方法1：正则回退（冷启动）
        regex_triples = self._extract_by_regex(text)
        triples.extend(regex_triples)

        # 方法2：学习到的模型（有足够数据后）
        if self._extraction_count >= 10 and encoder is not None:
            learned_triples = self._extract_by_model(text, text_repr, encoder)
            triples.extend(learned_triples)

        self._extraction_count += 1

        # 去重
        seen = set()
        unique_triples = []
        for t in triples:
            key = (t.subject, t.relation, t.obj)
            if key not in seen:
                seen.add(key)
                unique_triples.append(t)

        return unique_triples

    def _extract_by_regex(self, text: str) -> List[ExtractedTriple]:
        """正则提取（冷启动）"""
        triples = []
        for pattern, relation in self._regex_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                subject = match[0].strip()
                obj = match[1].strip()
                if 1 <= len(subject) <= 15 and 2 <= len(obj) <= 30:
                    triples.append(ExtractedTriple(
                        subject=subject,
                        relation=relation,
                        obj=obj,
                        confidence=0.5,
                        method='regex',
                    ))
        return triples

    def _extract_by_model(self, text: str, text_repr: torch.Tensor,
                         encoder) -> List[ExtractedTriple]:
        """模型提取（学习后）"""
        triples = []

        # 使用编码器获取每个字符的表示
        token_ids = encoder.tokenizer.encode(text, encoder.max_len)
        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            # 实体提取
            entity_probs = self.entity_extractor(x)
            entity_tags = entity_probs.argmax(dim=-1).squeeze(0)

            # 提取实体
            entities = self._extract_entities_from_tags(text, entity_tags, token_ids)

            # 关系分类
            for i, (subj, subj_start, subj_end) in enumerate(entities):
                for j, (obj, obj_start, obj_end) in enumerate(entities):
                    if i != j:
                        # 获取实体和上下文表示
                        subj_repr = self._get_span_repr(text_repr, subj_start, subj_end)
                        obj_repr = self._get_span_repr(text_repr, obj_start, obj_end)

                        # 关系分类
                        rel_probs = self.relation_classifier(subj_repr, obj_repr, text_repr)
                        rel_id = rel_probs.argmax().item()
                        rel_name = RELATION_TYPES[rel_id]
                        confidence = rel_probs[rel_id].item()

                        if confidence > 0.3:
                            triples.append(ExtractedTriple(
                                subject=subj,
                                relation=rel_name,
                                obj=obj,
                                confidence=confidence,
                                method='learned',
                            ))

        return triples

    def _extract_entities_from_tags(self, text: str, tags: torch.Tensor,
                                   token_ids: List[int]) -> List[Tuple[str, int, int]]:
        """从 BIO 标签提取实体"""
        entities = []
        current_entity = []
        current_start = -1

        chars = list(text)

        for i, tag_id in enumerate(tags):
            if i >= len(chars):
                break

            tag = BIO_TAGS[tag_id.item()]

            if tag == 'B-ENT':
                if current_entity:
                    entities.append((''.join(current_entity), current_start, i - 1))
                current_entity = [chars[i]]
                current_start = i
            elif tag == 'I-ENT' and current_entity:
                current_entity.append(chars[i])
            else:
                if current_entity:
                    entities.append((''.join(current_entity), current_start, i - 1))
                    current_entity = []
                    current_start = -1

        if current_entity:
            entities.append((''.join(current_entity), current_start, len(chars) - 1))

        return entities

    def _get_span_repr(self, text_repr: torch.Tensor, start: int, end: int,
                      token_reprs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取文本区间的表示

        如果有token级表示，做真正的区间池化；
        否则用全文表示的加权近似。
        """
        if token_reprs is not None and token_reprs.dim() >= 2:
            # 真正的区间池化
            seq_len = token_reprs.size(0)
            start = max(0, min(start, seq_len - 1))
            end = max(start, min(end, seq_len))
            span = token_reprs[start:end]
            if span.size(0) > 0:
                return span.mean(dim=0)

        # 回退：用位置加权的全文表示
        weight = (end - start) / max(text_repr.size(0), 1)
        return text_repr * weight

    def train_on_feedback(self, text: str, correct_triples: List[Tuple[str, str, str]],
                         encoder=None):
        """从反馈中学习

        当用户纠正提取结果时，使用正确标签训练模型。
        """
        if encoder is None:
            return

        self.entity_extractor.train()
        self.relation_classifier.train()
        self.optimizer.zero_grad()

        # 编码文本
        token_ids = encoder.tokenizer.encode(text, encoder.max_len)
        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # 实体提取损失
        entity_probs = self.entity_extractor(x)

        # 生成正确标签（简化：假设正确三元组中的实体在文本中）
        correct_tags = torch.zeros(len(token_ids), dtype=torch.long, device=self.device)
        for subj, _, obj in correct_triples:
            # 在文本中找到实体位置
            subj_pos = text.find(subj)
            obj_pos = text.find(obj)
            if subj_pos >= 0:
                for k in range(subj_pos, min(subj_pos + len(subj), len(token_ids))):
                    correct_tags[k] = BIO_TAG_TO_ID['B-ENT' if k == subj_pos else 'I-ENT']
            if obj_pos >= 0:
                for k in range(obj_pos, min(obj_pos + len(obj), len(token_ids))):
                    correct_tags[k] = BIO_TAG_TO_ID['B-ENT' if k == obj_pos else 'I-ENT']

        entity_loss = F.cross_entropy(
            entity_probs.view(-1, len(BIO_TAGS)),
            correct_tags[:entity_probs.size(1)]
        )

        # 关系分类损失
        text_repr = encoder.forward(text)
        relation_loss = torch.tensor(0.0, device=self.device)
        for subj, rel, obj in correct_triples:
            subj_repr = encoder.forward(subj)
            obj_repr = encoder.forward(obj)
            rel_probs = self.relation_classifier(subj_repr, obj_repr, text_repr)

            # 找到正确关系的ID
            if rel in REL_TO_ID:
                target = torch.tensor([REL_TO_ID[rel]], device=self.device)
                relation_loss = relation_loss + F.cross_entropy(
                    rel_probs.unsqueeze(0), target
                )

        # 总损失
        total_loss = entity_loss + relation_loss
        total_loss.backward()
        self.optimizer.step()

        self.entity_extractor.eval()
        self.relation_classifier.eval()
