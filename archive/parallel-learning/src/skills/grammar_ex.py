"""语法练习 — POS 模式提取、填空、纠错

从文本中提取语法模式，生成语法练习题。
"""

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.content.text_unit import TextUnit


@dataclass
class GrammarPattern:
    """语法模式"""
    pos_sequence: List[str]  # POS 标签序列
    words: List[str]         # 示例词
    example: str             # 完整例句
    frequency: int = 1       # 出现频率


@dataclass
class GrammarExercise:
    """一道语法练习题"""
    pattern: GrammarPattern
    prompt: str              # 练习提示
    target: str              # 正确答案
    options: List[str] = field(default_factory=list)
    exercise_type: str = 'completion'  # completion | correction | identification


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def _simple_pos_tag(word: str) -> str:
    """简单 POS 标注（基于规则）"""
    determiners = {'the', 'a', 'an', 'this', 'that', 'these', 'those'}
    prepositions = {'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'of'}
    pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    be_verbs = {'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
    aux_verbs = {'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'shall', 'should', 'can', 'could', 'may', 'might', 'must'}
    conjunctions = {'and', 'or', 'but', 'so', 'yet', 'nor', 'for', 'because', 'although'}

    if word in determiners:
        return 'DET'
    if word in prepositions:
        return 'PREP'
    if word in pronouns:
        return 'PRON'
    if word in be_verbs:
        return 'BE'
    if word in aux_verbs:
        return 'AUX'
    if word in conjunctions:
        return 'CONJ'
    if word.endswith('ly') and len(word) > 3:
        return 'ADV'
    if word.endswith(('tion', 'ment', 'ness', 'ity', 'ance', 'ence', 'er', 'or')) and len(word) > 4:
        return 'NOUN'
    if word.endswith(('ing', 'ed', 'ate', 'ize', 'ify')) and len(word) > 3:
        return 'VERB'
    if word.endswith(('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ic')) and len(word) > 3:
        return 'ADJ'
    if word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
        return 'NOUN'

    return 'WORD'


def extract_patterns(
    text_unit: TextUnit,
    min_freq: int = 1,
    max_len: int = 5,
) -> List[GrammarPattern]:
    """从文本单元提取语法模式"""
    sentences = text_unit.sentences or [text_unit.text]
    pattern_counts: Dict[Tuple[str, ...], List[Tuple[List[str], str]]] = {}

    for sent in sentences:
        words = _tokenize(sent)
        if len(words) < 2:
            continue

        pos_tags = [_simple_pos_tag(w) for w in words]

        # 提取所有长度 2-max_len 的 n-gram 模式
        for n in range(2, min(max_len + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                pos_seq = tuple(pos_tags[i:i + n])
                word_seq = words[i:i + n]
                example = ' '.join(word_seq)

                if pos_seq not in pattern_counts:
                    pattern_counts[pos_seq] = []
                pattern_counts[pos_seq].append((word_seq, example))

    # 统计频率并生成模式
    patterns = []
    for pos_seq, examples in pattern_counts.items():
        freq = len(examples)
        if freq >= min_freq:
            best_example = max(examples, key=lambda x: len(x[1]))
            patterns.append(GrammarPattern(
                pos_sequence=list(pos_seq),
                words=best_example[0],
                example=best_example[1],
                frequency=freq,
            ))

    patterns.sort(key=lambda p: p.frequency, reverse=True)
    return patterns


def generate_pattern_completion(
    pattern: GrammarPattern,
    known_words: Optional[Set[str]] = None,
) -> GrammarExercise:
    """按语法模式生成填空题：给出 POS 模式，挖空一个词"""
    words = list(pattern.words)
    if not words:
        return GrammarExercise(pattern=pattern, prompt='', target='', options=[])

    # 选择挖空位置（优先中间或末尾）
    blank_pos = len(words) // 2 if len(words) > 2 else 0
    target_word = words[blank_pos]
    target_pos = pattern.pos_sequence[blank_pos]

    # 构建提示
    prompt_parts = []
    for i, (w, pos) in enumerate(zip(words, pattern.pos_sequence)):
        if i == blank_pos:
            prompt_parts.append(f'[{pos}]')
        else:
            prompt_parts.append(w)
    prompt = ' '.join(prompt_parts)

    # 生成干扰项
    distractor_pool = known_words or set()
    same_pos = [w for w in distractor_pool if _simple_pos_tag(w) == target_pos and w != target_word]
    distractors = random.sample(same_pos, min(3, len(same_pos)))

    # 如果干扰项不够，用简单的占位
    while len(distractors) < 3:
        distractors.append(f'_{target_pos.lower()}_{len(distractors)}')

    options = distractors + [target_word]
    random.shuffle(options)

    return GrammarExercise(
        pattern=pattern,
        prompt=prompt,
        target=target_word,
        options=options,
        exercise_type='completion',
    )


def generate_error_correction(
    text_unit: TextUnit,
    error_rate: float = 0.3,
) -> List[GrammarExercise]:
    """生成语法纠错题

    策略：交换相邻词、改单复数、改时态等。
    """
    exercises = []
    sentences = text_unit.sentences or [text_unit.text]

    for sent in sentences:
        words = _tokenize(sent)
        if len(words) < 4:
            continue

        # 交换两个相邻词
        error_words = list(words)
        swap_pos = random.randint(0, len(words) - 2)
        error_words[swap_pos], error_words[swap_pos + 1] = error_words[swap_pos + 1], error_words[swap_pos]

        error_sent = ' '.join(error_words)

        # 简单 POS 模式
        pos_tags = [_simple_pos_tag(w) for w in words]

        pattern = GrammarPattern(
            pos_sequence=pos_tags,
            words=words,
            example=sent,
        )

        exercises.append(GrammarExercise(
            pattern=pattern,
            prompt=f'Find and fix the error: "{error_sent}"',
            target=sent,
            options=[sent],
            exercise_type='correction',
        ))

    return exercises
