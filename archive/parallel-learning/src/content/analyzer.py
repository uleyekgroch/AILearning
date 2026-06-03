"""文本难度分析 — 纯 Python，用 textstat"""

import re
from typing import List, Set, Dict, Optional

from src.content.text_unit import TextUnit


def _split_sentences(text: str) -> List[str]:
    """按 .!? 分句，保留非空句"""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if s.strip()]


def _tokenize(text: str) -> List[str]:
    """简单分词：小写 + 去标点"""
    return re.findall(r"[a-z']+", text.lower())


def _avg_sentence_length(sentences: List[str]) -> float:
    if not sentences:
        return 0.0
    lengths = [len(_tokenize(s)) for s in sentences]
    return sum(lengths) / len(lengths)


class DifficultyAnalyzer:
    """分析文本难度"""

    def __init__(self, known_vocabulary: Optional[Set[str]] = None):
        self.known_vocabulary = known_vocabulary or set()

    def analyze(self, text: str) -> Dict[str, float]:
        """返回难度指标"""
        sentences = _split_sentences(text)
        words = _tokenize(text)

        # textstat 指标（可选）
        fk_grade = 0.0
        try:
            import textstat
            fk_grade = textstat.flesch_kincaid_grade(text)
        except Exception:
            pass

        avg_sent_len = _avg_sentence_length(sentences)
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)

        unique = set(words)
        unknown = unique - self.known_vocabulary
        unknown_ratio = len(unknown) / max(len(unique), 1)

        # 综合难度
        fk_norm = min(1.0, max(0.0, fk_grade / 12.0))
        sent_norm = min(1.0, avg_sent_len / 20.0)
        difficulty = 0.3 * fk_norm + 0.2 * sent_norm + 0.5 * unknown_ratio

        return {
            'flesch_kincaid_grade': fk_grade,
            'avg_sentence_length': avg_sent_len,
            'avg_word_length': avg_word_len,
            'unique_words': len(unique),
            'unknown_words': len(unknown),
            'unknown_ratio': unknown_ratio,
            'difficulty': difficulty,
        }

    def classify_level(self, text: str) -> str:
        """分级：elementary / intermediate / advanced"""
        d = self.analyze(text)['difficulty']
        if d < 0.3:
            return 'elementary'
        if d < 0.6:
            return 'intermediate'
        return 'advanced'

    def extract_learnable_units(
        self, text: str, source: str = '', max_words: int = 80,
    ) -> List[TextUnit]:
        """将文本拆成可学习单元（每单元 ≤ max_words）"""
        sentences = _split_sentences(text)
        if not sentences:
            return []

        units = []
        current_sents: List[str] = []
        current_words = 0

        for sent in sentences:
            sent_words = _tokenize(sent)
            if current_words + len(sent_words) > max_words and current_sents:
                units.append(self._make_unit(current_sents, source))
                current_sents = []
                current_words = 0
            current_sents.append(sent)
            current_words += len(sent_words)

        if current_sents:
            units.append(self._make_unit(current_sents, source))

        return units

    def identify_new_vocabulary(self, text: str) -> List[str]:
        """找出文本中未掌握的词"""
        words = set(_tokenize(text))
        return sorted(words - self.known_vocabulary)

    def _make_unit(self, sentences: List[str], source: str) -> TextUnit:
        text = ' '.join(sentences)
        words = _tokenize(text)
        unique = set(words)
        new = unique - self.known_vocabulary
        difficulty = self.analyze(text)['difficulty']
        level = 'elementary' if difficulty < 0.3 else ('intermediate' if difficulty < 0.6 else 'advanced')
        return TextUnit(
            text=text, sentences=sentences, words=words,
            unique_words=unique, new_words=new,
            difficulty=difficulty, level=level, source=source,
        )
