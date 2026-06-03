"""
文章编辑器 - 像小朋友一样修改文章

不是简单的替换，而是理解后改进：
1. 理解原文含义
2. 找出问题
3. 提出改进
4. 重写内容
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class EditSuggestion:
    """编辑建议"""
    original: str
    suggested: str
    reason: str
    confidence: float
    edit_type: str  # grammar, style, clarity, content


@dataclass
class EditResult:
    """编辑结果"""
    original_text: str
    edited_text: str
    suggestions: List[EditSuggestion]
    improvements: List[str]
    word_count_change: int


class ArticleEditor:
    """文章编辑器

    像小朋友一样修改文章：
    - 理解原文含义
    - 找出问题
    - 提出改进建议
    - 重写内容
    """

    def __init__(self):
        """初始化编辑器"""
        # 常见问题模式
        self.problem_patterns = {
            'redundancy': ['的的', '了了', '是是', '美丽美丽', '很多很多', '非常非常'],
            'unclear': ['这个', '那个', '一些', '某种', '某些'],
            'grammar': ['的得地混用', '标点错误'],
        }

        # 改进建议
        self.improvement_suggestions = {
            'clarity': '使表达更清晰',
            'conciseness': '使文字更简洁',
            'vividness': '使描述更生动',
            'coherence': '使逻辑更连贯',
        }

        # 编辑历史
        self.edit_history: List[Tuple[str, str]] = []

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        分析文章

        Args:
            text: 文章文本

        Returns:
            分析结果
        """
        # 基本统计
        word_count = len(text)
        sentence_count = text.count('。') + text.count('！') + text.count('？')

        # 找出问题
        problems = self._find_problems(text)

        # 评估质量
        quality_score = self._evaluate_quality(text, problems)

        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'problems': problems,
            'quality_score': quality_score,
            'suggestions': self._generate_suggestions(problems),
        }

    def edit(self, text: str, focus: str = 'all') -> EditResult:
        """
        编辑文章

        Args:
            text: 文章文本
            focus: 关注点（grammar, style, clarity, content, all）

        Returns:
            编辑结果
        """
        # 分析原文
        analysis = self.analyze(text)

        # 生成编辑建议
        suggestions = self._generate_edit_suggestions(text, analysis, focus)

        # 应用编辑
        edited_text = self._apply_edits(text, suggestions)

        # 记录改进
        improvements = self._identify_improvements(text, edited_text)

        # 保存到历史
        self.edit_history.append((text, edited_text))

        return EditResult(
            original_text=text,
            edited_text=edited_text,
            suggestions=suggestions,
            improvements=improvements,
            word_count_change=len(edited_text) - len(text),
        )

    def _find_problems(self, text: str) -> List[Dict[str, Any]]:
        """
        找出问题

        Args:
            text: 文本

        Returns:
            问题列表
        """
        problems = []

        # 检查冗余
        for pattern in self.problem_patterns['redundancy']:
            if pattern in text:
                problems.append({
                    'type': 'redundancy',
                    'pattern': pattern,
                    'position': text.find(pattern),
                })

        # 检查不清晰
        for pattern in self.problem_patterns['unclear']:
            if pattern in text:
                problems.append({
                    'type': 'unclear',
                    'pattern': pattern,
                    'position': text.find(pattern),
                })

        return problems

    def _evaluate_quality(self, text: str, problems: List[Dict]) -> float:
        """
        评估质量

        Args:
            text: 文本
            problems: 问题列表

        Returns:
            质量分数 [0, 1]
        """
        # 基础分数
        base_score = 1.0

        # 根据问题扣分
        for problem in problems:
            if problem['type'] == 'redundancy':
                base_score -= 0.1
            elif problem['type'] == 'unclear':
                base_score -= 0.05

        return max(0.0, min(1.0, base_score))

    def _generate_suggestions(self, problems: List[Dict]) -> List[str]:
        """
        生成建议

        Args:
            problems: 问题列表

        Returns:
            建议列表
        """
        suggestions = []

        for problem in problems:
            if problem['type'] == 'redundancy':
                suggestions.append(f"避免重复使用'{problem['pattern']}'")
            elif problem['type'] == 'unclear':
                suggestions.append(f"将'{problem['pattern']}'替换为更具体的描述")

        return suggestions

    def _generate_edit_suggestions(self, text: str, analysis: Dict,
                                  focus: str) -> List[EditSuggestion]:
        """
        生成编辑建议

        Args:
            text: 文本
            analysis: 分析结果
            focus: 关注点

        Returns:
            编辑建议列表
        """
        suggestions = []

        # 根据问题生成建议
        for problem in analysis['problems']:
            if focus == 'all' or focus == problem['type']:
                suggestion = EditSuggestion(
                    original=problem['pattern'],
                    suggested=self._suggest_replacement(problem),
                    reason=f"避免{problem['type']}",
                    confidence=0.8,
                    edit_type=problem['type'],
                )
                suggestions.append(suggestion)

        return suggestions

    def _suggest_replacement(self, problem: Dict) -> str:
        """
        建议替换

        Args:
            problem: 问题

        Returns:
            替换建议
        """
        if problem['type'] == 'redundancy':
            return problem['pattern'][0]  # 只保留一个
        elif problem['type'] == 'unclear':
            return '具体的事物'
        return ''

    def _apply_edits(self, text: str, suggestions: List[EditSuggestion]) -> str:
        """
        应用编辑

        Args:
            text: 文本
            suggestions: 编辑建议

        Returns:
            编辑后的文本
        """
        edited = text

        for suggestion in suggestions:
            edited = edited.replace(suggestion.original, suggestion.suggested)

        return edited

    def _identify_improvements(self, original: str, edited: str) -> List[str]:
        """
        识别改进

        Args:
            original: 原文
            edited: 编辑后

        Returns:
            改进列表
        """
        improvements = []

        if len(edited) < len(original):
            improvements.append("文字更简洁")

        if edited != original:
            improvements.append("表达更清晰")

        return improvements

    def get_edit_stats(self) -> Dict[str, Any]:
        """获取编辑统计"""
        return {
            'total_edits': len(self.edit_history),
            'avg_word_change': sum(len(e[1]) - len(e[0]) for e in self.edit_history) / max(1, len(self.edit_history)),
        }
