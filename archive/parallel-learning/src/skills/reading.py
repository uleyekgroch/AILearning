"""阅读技能 — 完形填空（cloze test）

从文本单元中挖空关键词，让学习者填写。
优先挖空：新词 > 内容词 > 等间隔词。
"""

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from src.content.text_unit import TextUnit


@dataclass
class ClozeItem:
    """一道完形填空题"""
    original: str           # 原始句子
    blanked: str            # 挖空后的句子
    target_word: str        # 被挖的词
    options: List[str] = field(default_factory=list)  # 选项（含正确答案）
    source: str = ''

    @property
    def correct_idx(self) -> Optional[int]:
        for i, opt in enumerate(self.options):
            if opt == self.target_word:
                return i
        return None


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def _is_content_word(word: str) -> bool:
    """判断是否为内容词（非功能词）"""
    function_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'am', 'do', 'does', 'did', 'have', 'has', 'had', 'will', 'would',
        'shall', 'should', 'can', 'could', 'may', 'might', 'must',
        'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'of',
        'and', 'or', 'but', 'not', 'no', 'if', 'so', 'as', 'than',
        'this', 'that', 'these', 'those', 'it', 'its', 'i', 'me', 'my',
        'he', 'she', 'him', 'her', 'his', 'we', 'us', 'our', 'they', 'them', 'their',
    }
    return word not in function_words and len(word) > 2


def generate_cloze(
    text_unit: TextUnit,
    blank_ratio: float = 0.2,
    known_vocabulary: Optional[Set[str]] = None,
    distractor_pool: Optional[Set[str]] = None,
) -> List[ClozeItem]:
    """从文本单元生成完形填空题

    优先挖空顺序：新词 > 内容词 > 其他词
    """
    words = _tokenize(text_unit.text)
    if not words:
        return []

    known = known_vocabulary or set()
    new_words = text_unit.new_words

    # 为每个词计算优先级
    priorities = []
    for i, w in enumerate(words):
        if w in new_words:
            priorities.append((i, w, 3))  # 最高：新词
        elif _is_content_word(w) and w not in known:
            priorities.append((i, w, 2))  # 内容词且未掌握
        elif _is_content_word(w):
            priorities.append((i, w, 1))  # 内容词
        else:
            priorities.append((i, w, 0))  # 功能词

    # 按优先级排序
    priorities.sort(key=lambda x: x[2], reverse=True)

    # 选择要挖空的词
    num_blanks = max(1, int(len(words) * blank_ratio))
    blanks = set()
    for idx, word, _ in priorities:
        if len(blanks) >= num_blanks:
            break
        blanks.add(idx)

    # 生成干扰项池
    all_words = distractor_pool or (text_unit.unique_words | known)
    all_words = all_words - {words[i] for i in blanks}

    items = []
    for idx in sorted(blanks):
        target = words[idx]
        distractors = random.sample(
            list(all_words - {target}),
            min(3, len(all_words) - 1) if len(all_words) > 1 else []
        )
        options = distractors + [target]
        random.shuffle(options)

        # 构建挖空句子
        blanked_words = list(words)
        blanked_words[idx] = '___'
        blanked = ' '.join(blanked_words)

        # 恢复原始大小写的近似
        items.append(ClozeItem(
            original=text_unit.text,
            blanked=blanked,
            target_word=target,
            options=options,
            source=text_unit.source,
        ))

    return items


def generate_sentence_cloze(
    sentences: List[str],
    known_vocabulary: Optional[Set[str]] = None,
) -> List[ClozeItem]:
    """为句子列表生成完形填空"""
    items = []
    for sent in sentences:
        words = _tokenize(sent)
        unique = set(words)
        new = unique - (known_vocabulary or set())
        unit = TextUnit(
            text=sent,
            sentences=[sent],
            words=words,
            unique_words=unique,
            new_words=new,
        )
        items.extend(generate_cloze(unit, blank_ratio=0.15, known_vocabulary=known_vocabulary))
    return items
