"""
感知模块 - 从第一性原理出发

人类感知的本质：
1. 接收信息
2. 提取特征
3. 识别模式

不是简单的输入处理，而是主动感知
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class PerceptionResult:
    """感知结果"""
    raw_input: str              # 原始输入
    features: Dict[str, Any]    # 提取的特征
    patterns: List[str]         # 识别的模式
    confidence: float           # 置信度


class PerceptionModule:
    """感知模块

    从第一性原理出发：
    - 不是简单的输入处理
    - 而是主动感知
    - 提取有意义的特征
    """

    def __init__(self):
        """初始化感知模块"""
        # 特征提取器
        self.feature_extractors = {
            'text': self._extract_text_features,
            'number': self._extract_number_features,
        }

        # 模式识别器
        self.pattern_recognizers = {
            'word': self._recognize_word_patterns,
            'structure': self._recognize_structure_patterns,
        }

    def perceive(self, input_data: str) -> PerceptionResult:
        """
        感知输入

        Args:
            input_data: 输入数据

        Returns:
            感知结果
        """
        # 提取特征
        features = self._extract_features(input_data)

        # 识别模式
        patterns = self._recognize_patterns(input_data, features)

        # 计算置信度
        confidence = self._compute_confidence(features, patterns)

        return PerceptionResult(
            raw_input=input_data,
            features=features,
            patterns=patterns,
            confidence=confidence
        )

    def _extract_features(self, input_data: str) -> Dict[str, Any]:
        """
        提取特征

        Args:
            input_data: 输入数据

        Returns:
            特征字典
        """
        features = {}

        # 文本特征
        features['length'] = len(input_data)
        features['words'] = input_data.split()
        features['word_count'] = len(features['words'])

        # 语言特征
        features['has_chinese'] = any('一' <= c <= '鿿' for c in input_data)
        features['has_english'] = any('a' <= c.lower() <= 'z' for c in input_data)
        features['has_numbers'] = any(c.isdigit() for c in input_data)

        # 结构特征
        features['has_punctuation'] = any(c in '.,!?;:' for c in input_data)
        features['sentence_count'] = input_data.count('。') + input_data.count('.') + input_data.count('！') + input_data.count('？')

        return features

    def _recognize_patterns(self, input_data: str, features: Dict[str, Any]) -> List[str]:
        """
        识别模式

        Args:
            input_data: 输入数据
            features: 特征

        Returns:
            模式列表
        """
        patterns = []

        # 问题模式
        if '?' in input_data or '？' in input_data:
            patterns.append('question')

        # 命令模式
        if input_data.startswith(('请', '帮我', '告诉')):
            patterns.append('command')

        # 陈述模式
        if features['sentence_count'] > 0 and 'question' not in patterns:
            patterns.append('statement')

        # 中文模式
        if features['has_chinese']:
            patterns.append('chinese')

        # 英文模式
        if features['has_english']:
            patterns.append('english')

        return patterns

    def _compute_confidence(self, features: Dict[str, Any], patterns: List[str]) -> float:
        """
        计算置信度

        Args:
            features: 特征
            patterns: 模式

        Returns:
            置信度 [0, 1]
        """
        # 基础置信度
        confidence = 0.5

        # 根据特征调整
        if features['word_count'] > 0:
            confidence += 0.1

        if features['sentence_count'] > 0:
            confidence += 0.1

        # 根据模式调整
        if patterns:
            confidence += 0.1 * len(patterns)

        return min(1.0, confidence)

    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """提取文本特征"""
        return {
            'length': len(text),
            'words': text.split(),
        }

    def _extract_number_features(self, text: str) -> Dict[str, Any]:
        """提取数字特征"""
        numbers = [int(s) for s in text.split() if s.isdigit()]
        return {
            'numbers': numbers,
            'count': len(numbers),
        }

    def _recognize_word_patterns(self, text: str) -> List[str]:
        """识别单词模式"""
        return text.split()

    def _recognize_structure_patterns(self, text: str) -> List[str]:
        """识别结构模式"""
        patterns = []
        if '。' in text or '.' in text:
            patterns.append('sentence')
        if '，' in text or ',' in text:
            patterns.append('clause')
        return patterns
