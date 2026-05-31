"""可学习文本编码器 — 从字符级到语义级

替代 hash-based 编码和 bag-of-characters。

核心组件：
1. BPE分词器 — 从语料学习子词单元
2. 位置编码 — 正弦位置编码保留词序
3. 多头自注意力 — 上下文感知的表示
4. 学习到的嵌入 — 通过梯度优化

设计原则：
- 从零学习，不依赖预训练
- 支持中文和英文
- 可通过交互持续改进
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter


class BPETokenizer:
    """字节对编码(BPE)分词器

    从语料中学习子词单元。
    中文按字分割，英文按BPE合并。
    """

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._trained = False

        # 特殊token
        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        self.BOS = '<BOS>'
        self.EOS = '<EOS>'

    def _pre_tokenize(self, text: str) -> List[str]:
        """预分词：中文按字，英文按空格"""
        tokens = []
        i = 0
        while i < len(text):
            c = text[i]
            if '一' <= c <= '鿿':
                # 中文字符：直接作为token
                tokens.append(c)
                i += 1
            elif c.isascii() and c.isalpha():
                # 英文单词
                j = i
                while j < len(text) and text[j].isascii() and text[j].isalpha():
                    j += 1
                tokens.append(text[i:j])
                i = j
            elif c.isdigit():
                j = i
                while j < len(text) and text[j].isdigit():
                    j += 1
                tokens.append(text[i:j])
                i = j
            else:
                # 标点等
                tokens.append(c)
                i += 1
        return tokens

    def train(self, corpus: List[str], max_merges: int = 1000):
        """从语料学习BPE合并规则"""
        # 初始词表：所有单字符
        word_freq: Counter = Counter()
        for text in corpus:
            tokens = self._pre_tokenize(text)
            for t in tokens:
                word_freq[t] += 1

        # 初始词表
        vocab = set()
        for word in word_freq:
            for c in word:
                vocab.add(c)

        # 添加特殊token
        special = [self.PAD, self.UNK, self.BOS, self.EOS]
        for s in special:
            vocab.add(s)

        # BPE合并
        for _ in range(max_merges):
            if len(vocab) >= self.vocab_size:
                break

            # 统计相邻对频率
            pair_freq: Counter = Counter()
            for word, freq in word_freq.items():
                symbols = list(word)
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_freq[pair] += freq

            if not pair_freq:
                break

            # 找最频繁的对
            best_pair = pair_freq.most_common(1)[0][0]
            self.merges.append(best_pair)

            # 合并
            new_word_freq = {}
            merged = best_pair[0] + best_pair[1]
            vocab.add(merged)

            for word, freq in word_freq.items():
                symbols = list(word)
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if (i < len(symbols) - 1 and
                        symbols[i] == best_pair[0] and
                        symbols[i + 1] == best_pair[1]):
                        new_symbols.append(merged)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                new_word_freq[''.join(new_symbols)] = freq

            word_freq = new_word_freq

        # 构建词表
        self.token_to_id = {}
        self.id_to_token = {}
        for i, token in enumerate(sorted(vocab)):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        self._trained = True

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        """将文本编码为token ID序列"""
        if not self._trained:
            # 未训练时使用字符级编码
            tokens = self._pre_tokenize(text)
        else:
            tokens = self._pre_tokenize(text)
            # 应用BPE合并
            for merge in self.merges:
                merged = merge[0] + merge[1]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1 and
                        tokens[i] == merge[0] and
                        tokens[i + 1] == merge[1]):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

        # 转换为ID
        ids = [self.token_to_id.get(self.BOS, 0)]
        for t in tokens[:max_len - 2]:
            ids.append(self.token_to_id.get(t, self.token_to_id.get(self.UNK, 1)))
        ids.append(self.token_to_id.get(self.EOS, 0))

        # 填充
        while len(ids) < max_len:
            ids.append(self.token_to_id.get(self.PAD, 0))

        return ids[:max_len]

    @property
    def vocab_len(self) -> int:
        return max(len(self.token_to_id), 1)


class PositionalEncoding(nn.Module):
    """正弦位置编码

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        batch_size, seq_len, _ = x.shape

        # 线性变换
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)


class TransformerBlock(nn.Module):
    """Transformer编码器块

    LayerNorm → MultiHeadAttention → Residual
    LayerNorm → FFN → Residual
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差
        x = x + self.attn(self.norm1(x), mask)
        # FFN + 残差
        x = x + self.ffn(self.norm2(x))
        return x


class LearnableTextEncoder(nn.Module):
    """可学习文本编码器

    完整的编码流水线：
    1. BPE分词 → token IDs
    2. 嵌入层 → 向量序列
    3. 位置编码 → 保留词序
    4. Transformer → 上下文感知
    5. 池化 → 固定维度输出
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 max_len: int = 128, vocab_size: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # BPE分词器
        self.tokenizer = BPETokenizer(vocab_size)

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])

        # 输出层归一化
        self.norm = nn.LayerNorm(d_model)

        # 训练状态
        self._is_trained = False

    def train_tokenizer(self, corpus: List[str], max_merges: int = 1000):
        """训练BPE分词器"""
        self.tokenizer.train(corpus, max_merges)
        # 更新嵌入层大小
        new_vocab = self.tokenizer.vocab_len
        if new_vocab > self.embedding.num_embeddings:
            new_emb = nn.Embedding(new_vocab, self.d_model, padding_idx=0)
            # 复制已有权重
            with torch.no_grad():
                new_emb.weight[:self.embedding.num_embeddings] = self.embedding.weight
            self.embedding = new_emb
        self._is_trained = True

    def forward(self, text: str, return_attention: bool = False) -> torch.Tensor:
        """编码单个文本

        Args:
            text: 输入文本
            return_attention: 是否返回注意力权重

        Returns:
            (d_model,) 固定维度向量
        """
        # 分词
        token_ids = self.tokenizer.encode(text, self.max_len)
        ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.embedding.weight.device)

        # 用PAD的真实ID构建mask
        # PAD ID不一定是0！BPE tokenizer中 PAD='<PAD>' 的ID由sorted vocab决定
        pad_id = self.tokenizer.token_to_id.get(self.tokenizer.PAD, 0)
        non_pad_mask = (ids_tensor != pad_id)  # True for non-PAD positions
        pad_mask = non_pad_mask.unsqueeze(1).unsqueeze(2)  # (1, 1, 1, max_len) for attention

        # 嵌入 + 位置编码
        x = self.embedding(ids_tensor) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Transformer编码（使用正确的padding mask）
        for layer in self.layers:
            x = layer(x, pad_mask)

        x = self.norm(x)

        # 池化：只用非PAD位置的平均（用PAD的真实ID判断）
        pool_mask = non_pad_mask.float().unsqueeze(-1)  # (1, max_len, 1)
        pooled = (x * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp(min=1)

        return pooled.squeeze(0)

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """批量编码文本（真正的批处理）

        Returns:
            (batch, d_model) 批量向量
        """
        if len(texts) == 0:
            return torch.zeros(0, self.d_model)

        # 批量分词
        batch_token_ids = [self.tokenizer.encode(text, self.max_len) for text in texts]
        ids_tensor = torch.tensor(batch_token_ids, dtype=torch.long, device=self.embedding.weight.device)

        # 用PAD的真实ID构建mask
        pad_id = self.tokenizer.token_to_id.get(self.tokenizer.PAD, 0)
        non_pad_mask = (ids_tensor != pad_id)
        pad_mask = non_pad_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, max_len)

        # 嵌入 + 位置编码
        x = self.embedding(ids_tensor) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Transformer编码（使用正确的padding mask）
        for layer in self.layers:
            x = layer(x, pad_mask)

        x = self.norm(x)

        # 池化：只用非PAD位置
        pool_mask = non_pad_mask.float().unsqueeze(-1)  # (batch, max_len, 1)
        pooled = (x * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp(min=1)

        return pooled

    def similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的语义相似度"""
        v1 = self.forward(text1)
        v2 = self.forward(text2)
        return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'embedding': self.embedding.state_dict(),
            'layers': [l.state_dict() for l in self.layers],
            'norm': self.norm.state_dict(),
            'tokenizer_merges': self.tokenizer.merges,
            'tokenizer_token_to_id': self.tokenizer.token_to_id,
        }, path)

    def load(self, path: str):
        """加载模型"""
        data = torch.load(path, map_location=self.embedding.weight.device)
        self.embedding.load_state_dict(data['embedding'])
        for layer, state in zip(self.layers, data['layers']):
            layer.load_state_dict(state)
        self.norm.load_state_dict(data['norm'])
        self.tokenizer.merges = data['tokenizer_merges']
        self.tokenizer.token_to_id = data['tokenizer_token_to_id']
        self.tokenizer.id_to_token = {v: k for k, v in self.tokenizer.token_to_id.items()}
        self.tokenizer._trained = True


def test_encoder():
    """测试编码器"""
    encoder = LearnableTextEncoder(d_model=128, n_heads=4, n_layers=2)

    # 训练分词器
    corpus = [
        '人工智能是计算机科学的一个分支',
        '牛顿发现了万有引力定律',
        '下雨导致地面湿了',
        '水在100摄氏度沸腾',
        'The cat sat on the mat',
        'Dogs are loyal animals',
    ]
    encoder.train_tokenizer(corpus)
    print(f'Vocab size: {encoder.tokenizer.vocab_len}')

    # 编码测试
    v1 = encoder.forward('人工智能')
    v2 = encoder.forward('下雨')
    print(f'v1 shape: {v1.shape}')
    print(f'v1 != v2: {not torch.allclose(v1, v2)}')

    # 相似度测试
    sim_same = encoder.similarity('人工智能', '机器学习')
    sim_diff = encoder.similarity('人工智能', '下雨')
    print(f'Sim(人工智能, 机器学习): {sim_same:.4f}')
    print(f'Sim(人工智能, 下雨): {sim_diff:.4f}')

    # 词序测试
    v_dog_bite = encoder.forward('狗咬人')
    v_man_bite = encoder.forward('人咬狗')
    print(f'狗咬人 != 人咬狗: {not torch.allclose(v_dog_bite, v_man_bite)}')


if __name__ == '__main__':
    test_encoder()
