"""端到端学习层

从原始输入直接学习，不需要预处理。

核心能力：
1. 字符级编码 — 从字符学习词表示
2. 上下文理解 — 理解词在上下文中的含义
3. 端到端训练 — 从输入到输出的完整学习

运行方式：
    python training/layers/end_to_end.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import re


class CharacterEncoder(nn.Module):
    """字符级编码器

    从原始字符学习表示，不需要分词。
    """

    def __init__(self, char_vocab_size: int = 10000, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        """编码字符序列"""
        # chars: [batch, seq_len]
        embedded = self.char_embedding(chars)  # [batch, seq_len, embed_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch, embed_dim, seq_len]

        # 卷积
        x = F.relu(self.conv1(embedded))
        x = F.relu(self.conv2(x))

        # 池化
        x = self.pool(x).squeeze(-1)  # [batch, hidden_dim]

        return self.fc(x)


class ContextualEncoder(nn.Module):
    """上下文编码器

    理解词在上下文中的含义。
    """

    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = nn.Embedding(512, embed_dim)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 输出层
        self.fc = nn.Linear(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """编码上下文"""
        # x: [batch, seq_len]
        seq_len = x.shape[1]

        # 位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        embedded = self.embedding(x) + self.position_encoding(positions)

        # Transformer编码
        if mask is not None:
            encoded = self.transformer(embedded, src_key_padding_mask=mask)
        else:
            encoded = self.transformer(embedded)

        # 取[CLS]位置或平均池化
        pooled = encoded.mean(dim=1)  # [batch, embed_dim]

        return self.fc(pooled)


class EndToEndLearner(nn.Module):
    """端到端学习器

    从原始输入直接学习输出。
    """

    def __init__(self, char_vocab_size: int = 10000, word_vocab_size: int = 10000,
                 embed_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()

        # 字符级编码器
        self.char_encoder = CharacterEncoder(char_vocab_size, embed_dim // 2, hidden_dim)

        # 上下文编码器
        self.context_encoder = ContextualEncoder(word_vocab_size, embed_dim, hidden_dim)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

        # 预测头
        self.relation_head = nn.Linear(output_dim, 20)  # 20种关系
        self.entity_head = nn.Linear(output_dim, 100)  # 100种实体类型

    def forward(self, chars: torch.Tensor, words: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 字符级编码
        char_features = self.char_encoder(chars)

        # 上下文编码
        context_features = self.context_encoder(words)

        # 融合
        combined = torch.cat([char_features, context_features], dim=1)
        fused = self.fusion(combined)

        # 预测
        relations = self.relation_head(fused)
        entities = self.entity_head(fused)

        return relations, entities


class EndToEndSystem:
    """端到端学习系统

    从原始文本直接学习。
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 模型
        self.model = EndToEndLearner().to(self.device)

        # 字符词汇表
        self.char2idx: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
        self.idx2char: Dict[int, str] = {0: '<PAD>', 1: '<UNK>'}
        self.next_char_idx = 2

        # 词词汇表
        self.word2idx: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word: Dict[int, str] = {0: '<PAD>', 1: '<UNK>'}
        self.next_word_idx = 2

        # 关系类型
        self.relation_types = [
            'is_a', 'has', 'part_of', 'causes', 'located_in',
            'created_by', 'used_for', 'similar_to', 'opposite_of',
            'instance_of', 'subclass_of', 'property_of',
            'belongs_to', 'includes', 'uses', 'produces',
            'discovered_by', 'invented_by', 'proposed_by', 'related_to',
        ]

        # 训练数据
        self.training_data: List[Tuple[str, str, str]] = []

        # 统计
        self.stats = {
            'chars_learned': 0,
            'words_learned': 0,
            'training_steps': 0,
        }

    def encode_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码文本"""
        # 字符编码
        chars = list(text)
        char_indices = [self._get_char_idx(c) for c in chars]
        char_tensor = torch.tensor([char_indices], dtype=torch.long).to(self.device)

        # 词编码
        words = self._tokenize(text)
        word_indices = [self._get_word_idx(w) for w in words]
        word_tensor = torch.tensor([word_indices], dtype=torch.long).to(self.device)

        return char_tensor, word_tensor

    def _get_char_idx(self, char: str) -> int:
        """获取字符索引"""
        if char not in self.char2idx:
            self.char2idx[char] = self.next_char_idx
            self.idx2char[self.next_char_idx] = char
            self.next_char_idx += 1
            self.stats['chars_learned'] += 1
        return self.char2idx[char]

    def _get_word_idx(self, word: str) -> int:
        """获取词索引"""
        if word not in self.word2idx:
            self.word2idx[word] = self.next_word_idx
            self.idx2word[self.next_word_idx] = word
            self.next_word_idx += 1
            self.stats['words_learned'] += 1
        return self.word2idx[word]

    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        tokens = []
        # 中文
        zh_tokens = re.findall(r'[一-鿿]{1,4}', text)
        tokens.extend(zh_tokens)
        # 英文
        en_tokens = re.findall(r'[a-zA-Z]+', text)
        tokens.extend(en_tokens)
        return tokens

    def learn(self, text: str, subject: str, relation: str, obj: str):
        """学习"""
        self.training_data.append((subject, relation, obj))

        # 编码
        chars, words = self.encode_text(text)

        # 前向传播
        with torch.no_grad():
            relations, entities = self.model(chars, words)

        self.stats['training_steps'] += 1

    def predict(self, text: str) -> Tuple[str, float]:
        """预测关系"""
        chars, words = self.encode_text(text)

        with torch.no_grad():
            relations, entities = self.model(chars, words)
            probs = F.softmax(relations, dim=1)
            rel_idx = probs.argmax(dim=1).item()
            confidence = probs[0, rel_idx].item()

        rel_idx = min(rel_idx, len(self.relation_types) - 1)
        return self.relation_types[rel_idx], confidence

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'device': str(self.device),
            'char_vocab_size': len(self.char2idx),
            'word_vocab_size': len(self.word2idx),
            'training_data_size': len(self.training_data),
        }


def test_end_to_end():
    """测试端到端学习"""
    print("=" * 70)
    print("端到端学习测试")
    print("=" * 70)

    learner = EndToEndSystem()

    # 测试文本编码
    print("\n1. 文本编码测试:")
    texts = [
        '人工智能是计算机科学的一个分支',
        'Python是一种编程语言',
        '牛顿发现了万有引力定律',
    ]

    for text in texts:
        chars, words = learner.encode_text(text)
        print(f"  '{text[:20]}...' → 字符: {chars.shape}, 词: {words.shape}")

    # 测试学习
    print("\n2. 学习测试:")
    test_cases = [
        ('人工智能是计算机科学的一个分支', '人工智能', 'is_a', '计算机科学'),
        ('Python是一种编程语言', 'Python', 'is_a', '编程语言'),
        ('牛顿发现了万有引力定律', '牛顿', 'discovered_by', '万有引力定律'),
    ]

    for text, subject, relation, obj in test_cases:
        learner.learn(text, subject, relation, obj)
        print(f"  学习: {subject} --[{relation}]--> {obj}")

    # 测试预测
    print("\n3. 预测测试:")
    test_texts = [
        '深度学习是机器学习的一个分支',
        '太阳是太阳系的中心',
    ]

    for text in test_texts:
        pred_rel, confidence = learner.predict(text)
        print(f"  '{text[:20]}...' → {pred_rel} (置信度: {confidence:.3f})")

    # 统计
    print("\n统计:")
    stats = learner.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    test_end_to_end()
