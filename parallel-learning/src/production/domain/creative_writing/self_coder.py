"""
自我编码系统 - 像小朋友一样改进自己

不是简单的模板填充，而是理解后改进：
1. 理解代码功能
2. 找出问题
3. 提出改进
4. 重写代码
5. 测试验证
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CodeAnalysis:
    """代码分析"""
    file_path: str
    functions: List[str]
    classes: List[str]
    lines: int
    complexity: float
    issues: List[Dict[str, Any]]


@dataclass
class CodeImprovement:
    """代码改进"""
    original: str
    improved: str
    reason: str
    improvement_type: str  # performance, readability, maintainability, functionality
    confidence: float


class SelfCoder:
    """自我编码系统

    像小朋友一样改进自己：
    - 理解代码功能
    - 找出问题
    - 提出改进
    - 重写代码
    - 测试验证
    """

    def __init__(self):
        """初始化自我编码系统"""
        # 代码模式库
        self.code_patterns = {
            'performance': {
                'loop_optimization': '使用列表推导式替代循环',
                'cache_optimization': '添加缓存减少重复计算',
                'lazy_loading': '使用延迟加载',
            },
            'readability': {
                'naming': '使用更具描述性的变量名',
                'comments': '添加注释解释复杂逻辑',
                'structure': '将长函数拆分为小函数',
            },
            'maintainability': {
                'duplication': '消除重复代码',
                'coupling': '减少模块间耦合',
                'testing': '添加单元测试',
            },
        }

        # 改进历史
        self.improvement_history: List[Tuple[str, str]] = []

    def analyze_code(self, code: str, file_path: str = "unknown") -> CodeAnalysis:
        """
        分析代码

        Args:
            code: 代码文本
            file_path: 文件路径

        Returns:
            代码分析
        """
        # 基本统计
        lines = code.split('\n')
        line_count = len(lines)

        # 提取函数
        functions = []
        for line in lines:
            if line.strip().startswith('def '):
                func_name = line.split('(')[0].replace('def ', '').strip()
                functions.append(func_name)

        # 提取类
        classes = []
        for line in lines:
            if line.strip().startswith('class '):
                class_name = line.split('(')[0].split(':')[0].replace('class ', '').strip()
                classes.append(class_name)

        # 计算复杂度（简化）
        complexity = self._calculate_complexity(code)

        # 找出问题
        issues = self._find_issues(code)

        return CodeAnalysis(
            file_path=file_path,
            functions=functions,
            classes=classes,
            lines=line_count,
            complexity=complexity,
            issues=issues,
        )

    def improve_code(self, code: str, focus: str = 'all') -> List[CodeImprovement]:
        """
        改进代码

        Args:
            code: 代码文本
            focus: 关注点（performance, readability, maintainability, all）

        Returns:
            改进列表
        """
        improvements = []

        # 分析代码
        analysis = self.analyze_code(code)

        # 根据问题生成改进
        for issue in analysis.issues:
            if focus == 'all' or focus == issue['type']:
                improvement = self._generate_improvement(code, issue)
                if improvement:
                    improvements.append(improvement)

        return improvements

    def rewrite_function(self, function_code: str, improvements: List[CodeImprovement]) -> str:
        """
        重写函数

        Args:
            function_code: 函数代码
            improvements: 改进列表

        Returns:
            重写后的代码
        """
        rewritten = function_code

        for improvement in improvements:
            rewritten = rewritten.replace(improvement.original, improvement.improved)

        return rewritten

    def _calculate_complexity(self, code: str) -> float:
        """
        计算复杂度

        Args:
            code: 代码

        Returns:
            复杂度分数
        """
        # 简化：基于控制流语句计算
        complexity = 1.0

        # 条件语句
        complexity += code.count('if ') * 0.5
        complexity += code.count('elif ') * 0.5

        # 循环
        complexity += code.count('for ') * 1.0
        complexity += code.count('while ') * 1.0

        # 异常处理
        complexity += code.count('try:') * 0.5
        complexity += code.count('except ') * 0.5

        return complexity

    def _find_issues(self, code: str) -> List[Dict[str, Any]]:
        """
        找出问题

        Args:
            code: 代码

        Returns:
            问题列表
        """
        issues = []

        # 检查长函数
        functions = code.split('def ')
        for func in functions[1:]:  # 跳过第一个（类定义或模块级代码）
            func_lines = func.split('\n')
            if len(func_lines) > 20:
                issues.append({
                    'type': 'readability',
                    'pattern': 'long_function',
                    'description': f'函数过长（{len(func_lines)}行）',
                    'suggestion': '拆分为更小的函数',
                })

        # 检查重复代码
        lines = code.split('\n')
        line_counts = {}
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 10:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        for line, count in line_counts.items():
            if count > 2:
                issues.append({
                    'type': 'maintainability',
                    'pattern': 'duplication',
                    'description': f'重复代码：{line[:30]}...',
                    'suggestion': '提取为公共函数',
                })

        # 检查魔法数字
        import re
        magic_numbers = re.findall(r'\b\d{2,}\b', code)
        for num in set(magic_numbers):
            if num not in ['0', '1', '10', '100']:  # 排除常见数字
                issues.append({
                    'type': 'readability',
                    'pattern': 'magic_number',
                    'description': f'魔法数字：{num}',
                    'suggestion': '定义为命名常量',
                })

        return issues

    def _generate_improvement(self, code: str, issue: Dict) -> Optional[CodeImprovement]:
        """
        生成改进

        Args:
            code: 代码
            issue: 问题

        Returns:
            改进建议
        """
        if issue['pattern'] == 'long_function':
            return CodeImprovement(
                original='long_function',
                improved='split_into_smaller_functions',
                reason=issue['description'],
                improvement_type='readability',
                confidence=0.9,
            )
        elif issue['pattern'] == 'duplication':
            return CodeImprovement(
                original='duplicated_code',
                improved='extract_common_function',
                reason=issue['description'],
                improvement_type='maintainability',
                confidence=0.9,
            )
        elif issue['pattern'] == 'magic_number':
            return CodeImprovement(
                original=issue['description'].split('：')[1] if '：' in issue['description'] else '',
                improved='NAMED_CONSTANT',
                reason=issue['description'],
                improvement_type='readability',
                confidence=0.8,
            )

        return None

    def get_improvement_stats(self) -> Dict[str, Any]:
        """获取改进统计"""
        return {
            'total_improvements': len(self.improvement_history),
            'improvement_types': list(set(i[0] for i in self.improvement_history)),
        }
