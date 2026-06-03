"""写作技能 — 句子重组、造句练习

打乱词序让学习者重排，评估词序准确性和完整度。
"""

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from src.content.text_unit import TextUnit


@dataclass
class ScrambleItem:
    """一道句子重组题"""
    original: str           # 原始句子
    target_order: List[str] # 正确词序
    scrambled: List[str]    # 打乱后的词序
    source: str = ''


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def generate_scramble(
    text_unit: TextUnit,
    num_items: int = 0,
) -> List[ScrambleItem]:
    """从文本单元生成句子重组题"""
    items = []
    sentences = text_unit.sentences or [text_unit.text]
    count = 0

    for sent in sentences:
        words = _tokenize(sent)
        if len(words) < 3:
            continue

        scrambled = list(words)
        # 确保打乱后与原序不同
        for _ in range(10):
            random.shuffle(scrambled)
            if scrambled != words:
                break

        items.append(ScrambleItem(
            original=sent,
            target_order=words,
            scrambled=scrambled,
            source=text_unit.source,
        ))
        count += 1
        if num_items > 0 and count >= num_items:
            break

    return items


def evaluate_sentence(
    produced: List[str],
    target: List[str],
) -> Dict[str, float]:
    """评估造句/重组结果

    Returns:
        word_overlap: 用词重叠率
        order_accuracy: 词序准确率
        completeness: 完整度
        score: 综合得分
    """
    if not target:
        return {'word_overlap': 0.0, 'order_accuracy': 0.0, 'completeness': 0.0, 'score': 0.0}

    produced_set = set(produced)
    target_set = set(target)

    # 用词重叠
    overlap = len(produced_set & target_set) / max(len(target_set), 1)

    # 完整度：是否包含了所有目标词
    completeness = len(produced_set & target_set) / max(len(target), 1)

    # 词序准确率：最长公共子序列比例
    order_accuracy = _lcs_ratio(produced, target)

    # 综合得分
    score = 0.3 * overlap + 0.4 * order_accuracy + 0.3 * completeness

    return {
        'word_overlap': round(overlap, 3),
        'order_accuracy': round(order_accuracy, 3),
        'completeness': round(completeness, 3),
        'score': round(score, 3),
    }


def _lcs_ratio(a: List[str], b: List[str]) -> float:
    """最长公共子序列比例"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0

    # 简化：用 2 行 DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    lcs_len = prev[n]
    return lcs_len / max(m, n)


def generate_free_writing_prompt(
    known_words: Set[str],
    min_words: int = 4,
    max_words: int = 8,
) -> Tuple[str, Set[str]]:
    """生成自由造句提示：给出几个词让学习者造句"""
    candidates = [w for w in known_words if len(w) > 2]
    if len(candidates) < 3:
        return '', set()

    n = random.randint(min_words, min(max_words, len(candidates)))
    chosen = set(random.sample(candidates, n))
    prompt = 'Use these words to make a sentence: ' + ', '.join(sorted(chosen))
    return prompt, chosen
