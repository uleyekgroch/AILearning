"""听力技能 — 语音特征估算与听力匹配

从文本估算语音特征（无 TTS 依赖），生成听力练习。
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.content.text_unit import TextUnit


@dataclass
class ListenItem:
    """一道听力题"""
    target: str             # 目标句子
    text_options: List[str] # 文本选项（含正确答案）
    audio_features: List[float] = field(default_factory=list)  # 语音特征
    source: str = ''

    @property
    def correct_idx(self) -> Optional[int]:
        for i, opt in enumerate(self.text_options):
            if opt == target:
                return i
        return None


def _estimate_phonetic_features(text: str) -> List[float]:
    """从文本估算语音特征（字符级启发式）

    Returns:
        7 维向量: [vowels, fricatives, plosives, nasals, liquids, glides, silence_ratio]
    """
    t = text.lower()
    if not t:
        return [0.0] * 7

    vowels = sum(1 for c in t if c in 'aeiou')
    fricatives = sum(1 for c in t if c in 'fsvzhx')
    plosives = sum(1 for c in t if c in 'ptkbdg')
    nasals = sum(1 for c in t if c in 'mnng')
    liquids = sum(1 for c in t if c in 'lr')
    glides = sum(1 for c in t if c in 'wy')
    silence = sum(1 for c in t if c in ' .,!?;:')
    total = max(len(t), 1)

    return [
        vowels / total,
        fricatives / total,
        plosives / total,
        nasals / total,
        liquids / total,
        glides / total,
        silence / total,
    ]


def generate_listen_exercise(
    text_unit: TextUnit,
    num_options: int = 3,
    distractor_pool: Optional[List[str]] = None,
) -> List[ListenItem]:
    """生成听力练习题

    给出语音特征，让学习者从文本选项中选出匹配的句子。
    """
    items = []
    sentences = text_unit.sentences or [text_unit.text]

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent.split()) < 3:
            continue

        features = _estimate_phonetic_features(sent)

        # 干扰项：从其他句子或干扰池中选
        distractors = []
        if distractor_pool:
            candidates = [s for s in distractor_pool if s != sent]
            distractors = candidates[:num_options - 1]
        else:
            # 用同文本中的其他句子
            for other in sentences:
                if other != sent and len(distractors) < num_options - 1:
                    distractors.append(other)

        # 如果干扰项不够，构造简单干扰
        while len(distractors) < num_options - 1:
            words = re.findall(r"[a-z']+", sent.lower())
            random_idx = len(distractors)
            if len(words) > 3:
                shuffled = list(words)
                shuffled[-1] = words[random_idx % len(words)]
                distractors.append(' '.join(shuffled))
            else:
                break

        options = distractors + [sent]

        # 打乱选项
        import random
        random.shuffle(options)

        items.append(ListenItem(
            target=sent,
            text_options=options,
            audio_features=features,
            source=text_unit.source,
        ))

    return items


def text_to_audio_features(text: str) -> List[float]:
    """公开接口：将文本转为 7 维语音特征"""
    return _estimate_phonetic_features(text)


def listen_and_match(
    audio_features: List[float],
    options: List[str],
) -> Optional[int]:
    """根据语音特征从选项中选择最匹配的句子

    用语音特征向量的余弦相似度匹配。
    """
    if not options:
        return None

    import math

    target_vec = audio_features
    best_idx = 0
    best_sim = -1.0

    for i, opt in enumerate(options):
        opt_features = _estimate_phonetic_features(opt)
        sim = _cosine_sim(target_vec, opt_features)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    return best_idx


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)
