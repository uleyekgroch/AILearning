"""文本编码器 — hash-based 编码，输出与 MultiModalEncoder 兼容的格式

将句子编码为 {'visual': (4,8,8), 'auditory': (7,), 'position': (2,)}，
可以直接喂给 PredictiveCodingEngine 学习。

设计理念：不用预训练嵌入，用 hash(word) 生成确定性伪随机张量。
PredictiveCodingEngine 通过 Hebbian 学习发现上下文规律 — 从零学习。
"""

import hashlib
import re
import struct
from typing import Dict, List, Tuple

import torch

from src.core.device import get_device


def _hash_to_floats(seed: str, n: int) -> List[float]:
    """用 SHA-256 从 seed 生成 n 个 [-1, 1] 浮点数"""
    h = hashlib.sha256(seed.encode('utf-8')).digest()
    values = []
    for i in range(n):
        byte_idx = (i * 4) % len(h)
        chunk = h[byte_idx:byte_idx + 4]
        if len(chunk) < 4:
            chunk = chunk + h[:4 - len(chunk)]
        val = struct.unpack('f', chunk)[0]
        values.append((val % 2.0) - 1.0)
    return values


def _hash_to_tensor(seed: str, shape: Tuple[int, ...]) -> torch.Tensor:
    """生成确定性 hash 张量"""
    n = 1
    for s in shape:
        n *= s
    values = _hash_to_floats(seed, n)
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def _char_bigrams(text: str) -> List[str]:
    t = text.lower()
    return [t[i:i+2] for i in range(len(t) - 1)]


# 简单 POS 规则（无 spaCy 回退）
_CONTENT_POS = {'noun', 'verb', 'adj', 'adv'}
_POS_SUFFIXES = {
    'noun': {'tion', 'ment', 'ness', 'ity', 'ance', 'ence', 'er', 'or', 'ist'},
    'verb': {'ate', 'ize', 'ify', 'ing', 'ed'},
    'adj': {'ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ic'},
    'adv': {'ly'},
}
_DET_WORDS = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'some', 'any', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
_PREP_WORDS = {'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'of', 'about', 'into', 'through', 'over', 'under', 'between'}


def _guess_pos(word: str) -> str:
    if word in _DET_WORDS:
        return 'det'
    if word in _PREP_WORDS:
        return 'prep'
    for pos, suffixes in _POS_SUFFIXES.items():
        for suf in suffixes:
            if word.endswith(suf) and len(word) > len(suf) + 1:
                return pos
    return 'other'


def _pos_distribution(words: List[str]) -> List[float]:
    """返回 7 维 POS 分布 [noun, verb, adj, adv, prep, det, other]"""
    counts = {'noun': 0, 'verb': 0, 'adj': 0, 'adv': 0, 'prep': 0, 'det': 0, 'other': 0}
    for w in words:
        pos = _guess_pos(w)
        counts[pos] += 1
    total = max(len(words), 1)
    return [counts[k] / total for k in ['noun', 'verb', 'adj', 'adv', 'prep', 'det', 'other']]


class TextEncoder:
    """将文本编码为与 MultiModalEncoder 兼容的格式"""

    def __init__(self, device: str = 'auto'):
        self.device = get_device(device)

    def encode_sentence(self, sentence: str) -> Dict[str, torch.Tensor]:
        """编码一个句子

        Returns:
            {'visual': (4,8,8), 'auditory': (7,), 'position': (2,)}
        """
        words = _tokenize(sentence)
        if not words:
            return self._empty()

        visual = self._encode_visual(sentence, words)
        auditory = self._encode_auditory(words)
        position = self._encode_position(sentence, words)

        return {
            'visual': visual.to(self.device),
            'auditory': auditory.to(self.device),
            'position': position.to(self.device),
        }

    def encode_paragraph(self, text: str, max_words: int = 16) -> List[Dict[str, torch.Tensor]]:
        """将段落编码为多个句子向量"""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        results = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                results.append(self.encode_sentence(sent))
        return results

    def _empty(self) -> Dict[str, torch.Tensor]:
        return {
            'visual': torch.zeros(4, 8, 8, device=self.device),
            'auditory': torch.zeros(7, device=self.device),
            'position': torch.zeros(2, device=self.device),
        }

    def _encode_visual(self, sentence: str, words: List[str]) -> torch.Tensor:
        """4 通道 8x8 视觉张量

        Ch0: 前 8 词的 hash 向量
        Ch1: 9-16 词（或字符 bigram 指纹）
        Ch2: 字符 bigram 指纹
        Ch3: 结构特征（标点密度、大写模式等）
        """
        grid = torch.zeros(4, 8, 8)

        # Ch0: 前 8 词 hash 向量
        for i, word in enumerate(words[:8]):
            row, col = i // 2, (i % 2) * 4
            vec = _hash_to_floats(f'w:{word}', 4)
            for j, v in enumerate(vec):
                if row < 8 and col + j < 8:
                    grid[0, row, col + j] = v

        # Ch1: 9-16 词或溢出
        for i, word in enumerate(words[8:16]):
            row, col = i // 2, (i % 2) * 4
            vec = _hash_to_floats(f'w:{word}', 4)
            for j, v in enumerate(vec):
                if row < 8 and col + j < 8:
                    grid[1, row, col + j] = v

        # 如果词不足 8，用 bigram 填充 Ch1
        if len(words) <= 8:
            bigrams = _char_bigrams(sentence)
            for i, bg in enumerate(bigrams[:16]):
                row, col = i // 4, i % 4
                if row < 8 and col < 8:
                    val = hash(bg) % 1000 / 500.0 - 1.0
                    grid[1, row, col * 2] = val

        # Ch2: 字符 bigram 指纹
        bigrams = _char_bigrams(sentence.lower())
        bg_positions = set()
        for bg in bigrams[:32]:
            h = hash(bg)
            row = h % 8
            col = (h // 8) % 8
            bg_positions.add((row, col))
            grid[2, row, col] += 0.3

        for row, col in bg_positions:
            grid[2, row, col] = min(grid[2, row, col].item(), 1.0)

        # Ch3: 结构特征
        punct_count = sum(1 for c in sentence if c in '.,!?;:\'"()-')
        caps_count = sum(1 for c in sentence if c.isupper())
        word_count = len(words)
        avg_len = sum(len(w) for w in words) / max(word_count, 1)
        has_period = 1.0 if '.' in sentence else 0.0
        has_comma = 1.0 if ',' in sentence else 0.0
        has_question = 1.0 if '?' in sentence else 0.0
        has_exclaim = 1.0 if '!' in sentence else 0.0

        struct_vals = [
            min(punct_count / 5.0, 1.0),
            min(caps_count / max(word_count, 1), 1.0),
            min(word_count / 20.0, 1.0),
            min(avg_len / 10.0, 1.0),
            has_period, has_comma, has_question, has_exclaim,
        ]
        for i, v in enumerate(struct_vals):
            row, col = i // 4, i % 4
            if row < 8:
                grid[3, row, col * 2] = v
                grid[3, row, col * 2 + 1] = v

        return grid

    def _encode_auditory(self, words: List[str]) -> torch.Tensor:
        """7 维 POS 分布向量 [noun, verb, adj, adv, prep, det, other]"""
        dist = _pos_distribution(words)
        return torch.tensor(dist, dtype=torch.float32)

    def _encode_position(self, sentence: str, words: List[str]) -> torch.Tensor:
        """2 维 [句长/20, 平均词长/10]"""
        word_count = len(words)
        avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
        return torch.tensor([
            min(word_count / 20.0, 1.0),
            min(avg_word_len / 10.0, 1.0),
        ], dtype=torch.float32)
