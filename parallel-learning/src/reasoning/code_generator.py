"""代码生成引擎 — 从需求到代码

实现代码生成的核心能力：
1. 需求理解：理解用户的代码需求
2. 模式匹配：从已有代码中学习模式
3. 代码生成：生成符合需求的代码
4. 代码验证：验证生成的代码是否正确

设计原则：
- 从简单到复杂：先生成简单代码，再逐步复杂化
- 模式复用：从已有代码中学习模式
- 增量生成：逐步构建代码，而非一次性生成
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CodePattern:
    """代码模式"""
    name: str
    template: str
    description: str
    examples: List[str] = field(default_factory=list)
    complexity: int = 1  # 1-5


@dataclass
class GeneratedCode:
    """生成的代码"""
    code: str
    language: str
    description: str
    patterns_used: List[str]
    complexity: int


class CodeGenerator:
    """代码生成引擎

    从需求描述生成代码。
    """

    def __init__(self):
        # 代码模式库
        self.patterns: Dict[str, CodePattern] = {}

        # 生成历史
        self.generation_history: List[GeneratedCode] = []

        # 初始化基础模式
        self._init_basic_patterns()

    def _init_basic_patterns(self):
        """初始化基础代码模式"""
        # Python基础模式
        self.patterns['variable'] = CodePattern(
            name='variable',
            template='{name} = {value}',
            description='变量赋值',
            examples=['x = 10', 'name = "hello"'],
            complexity=1,
        )

        self.patterns['function'] = CodePattern(
            name='function',
            template='def {name}({params}):\n    {body}',
            description='函数定义',
            examples=['def add(a, b):\n    return a + b'],
            complexity=2,
        )

        self.patterns['if_else'] = CodePattern(
            name='if_else',
            template='if {condition}:\n    {true_branch}\nelse:\n    {false_branch}',
            description='条件判断',
            examples=['if x > 0:\n    print("positive")\nelse:\n    print("negative")'],
            complexity=2,
        )

        self.patterns['for_loop'] = CodePattern(
            name='for_loop',
            template='for {item} in {iterable}:\n    {body}',
            description='for循环',
            examples=['for i in range(10):\n    print(i)'],
            complexity=2,
        )

        self.patterns['list_comprehension'] = CodePattern(
            name='list_comprehension',
            template='[{expr} for {item} in {iterable}]',
            description='列表推导式',
            examples=['[x**2 for x in range(10)]'],
            complexity=3,
        )

        self.patterns['class'] = CodePattern(
            name='class',
            template='class {name}:\n    def __init__(self{params}):\n        {init_body}\n\n    def {method}(self{params}):\n        {method_body}',
            description='类定义',
            examples=['class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y'],
            complexity=4,
        )

    def generate(self, requirement: str, language: str = 'python') -> GeneratedCode:
        """根据需求生成代码

        Args:
            requirement: 需求描述
            language: 目标语言

        Returns:
            生成的代码
        """
        # 分析需求
        patterns_needed = self._analyze_requirement(requirement)

        # 生成代码
        code = self._generate_from_patterns(patterns_needed, requirement)

        # 创建结果
        result = GeneratedCode(
            code=code,
            language=language,
            description=requirement,
            patterns_used=[p.name for p in patterns_needed],
            complexity=max(p.complexity for p in patterns_needed) if patterns_needed else 1,
        )

        self.generation_history.append(result)
        return result

    def _analyze_requirement(self, requirement: str) -> List[CodePattern]:
        """分析需求，确定需要的代码模式"""
        patterns = []
        req_lower = requirement.lower()

        # 变量
        if any(keyword in req_lower for keyword in ['定义', '创建', '变量']):
            patterns.append(self.patterns['variable'])

        # 函数
        if any(keyword in req_lower for keyword in ['函数', '方法', '计算', '转换']):
            patterns.append(self.patterns['function'])

        # 条件
        if any(keyword in req_lower for keyword in ['如果', '判断', '条件', '是否']):
            patterns.append(self.patterns['if_else'])

        # 循环
        if any(keyword in req_lower for keyword in ['循环', '遍历', '迭代', '重复']):
            patterns.append(self.patterns['for_loop'])

        # 类
        if any(keyword in req_lower for keyword in ['类', '对象', '面向对象']):
            patterns.append(self.patterns['class'])

        # 如果没有匹配到任何模式，使用函数模式
        if not patterns:
            patterns.append(self.patterns['function'])

        return patterns

    def _generate_from_patterns(self, patterns: List[CodePattern],
                               requirement: str) -> str:
        """从模式生成代码"""
        if not patterns:
            return f"# TODO: {requirement}"

        # 简化：使用第一个模式的模板
        pattern = patterns[0]

        # 根据需求填充模板
        if pattern.name == 'function':
            return self._generate_function(requirement)
        elif pattern.name == 'variable':
            return self._generate_variable(requirement)
        elif pattern.name == 'if_else':
            return self._generate_if_else(requirement)
        elif pattern.name == 'for_loop':
            return self._generate_for_loop(requirement)
        elif pattern.name == 'class':
            return self._generate_class(requirement)
        else:
            return f"# {requirement}\n{pattern.template}"

    def _generate_function(self, requirement: str) -> str:
        """生成函数代码"""
        # 提取函数名
        func_name = 'process'
        if '计算' in requirement:
            func_name = 'calculate'
        elif '转换' in requirement:
            func_name = 'convert'
        elif '验证' in requirement:
            func_name = 'validate'

        return f'''def {func_name}(data):
    """
    {requirement}
    """
    # TODO: 实现具体逻辑
    result = data
    return result'''

    def _generate_variable(self, requirement: str) -> str:
        """生成变量代码"""
        return f'# {requirement}\nresult = None'

    def _generate_if_else(self, requirement: str) -> str:
        """生成条件代码"""
        return f'''# {requirement}
if condition:
    # 条件为真时执行
    pass
else:
    # 条件为假时执行
    pass'''

    def _generate_for_loop(self, requirement: str) -> str:
        """生成循环代码"""
        return f'''# {requirement}
for item in items:
    # 处理每个item
    pass'''

    def _generate_class(self, requirement: str) -> str:
        """生成类代码"""
        return f'''class DataProcessor:
    """
    {requirement}
    """

    def __init__(self):
        self.data = None

    def process(self, data):
        """处理数据"""
        self.data = data
        return self.data

    def get_result(self):
        """获取结果"""
        return self.data'''

    def add_pattern(self, pattern: CodePattern):
        """添加新的代码模式"""
        self.patterns[pattern.name] = pattern

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'patterns': len(self.patterns),
            'generations': len(self.generation_history),
            'avg_complexity': sum(g.complexity for g in self.generation_history) / max(len(self.generation_history), 1),
        }
