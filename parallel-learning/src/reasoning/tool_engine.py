"""工具调用引擎 — 从注册到实际使用

实现工具调用的核心能力：
1. 工具发现：识别何时需要工具
2. 工具选择：选择合适的工具
3. 工具执行：调用工具并获取结果
4. 结果整合：将工具结果整合到推理中

设计原则：
- 工具是推理的延伸，不是替代
- 工具调用应该是可解释的
- 工具失败应该有回退策略
"""

import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, str]  # 参数名 -> 参数类型
    function: Callable  # 工具函数
    examples: List[Dict] = field(default_factory=list)  # 使用示例


@dataclass
class ToolCall:
    """工具调用记录"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    success: bool
    timestamp: float


class ToolEngine:
    """工具调用引擎

    管理和调用工具，将工具结果整合到推理中。
    """

    def __init__(self):
        # 注册的工具
        self.tools: Dict[str, Tool] = {}

        # 调用历史
        self.call_history: List[ToolCall] = []

        # 工具使用统计
        self.usage_stats: Dict[str, int] = {}

    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
        self.usage_stats[tool.name] = 0

    def unregister_tool(self, name: str):
        """注销工具"""
        if name in self.tools:
            del self.tools[name]
            del self.usage_stats[name]

    def detect_tool_need(self, question: str, context: Dict) -> Optional[str]:
        """检测是否需要工具

        根据问题类型和上下文判断是否需要工具。
        """
        question_lower = question.lower()

        # 数学计算
        if any(op in question_lower for op in ['计算', '多少', '等于', '加', '减', '乘', '除']):
            if 'calculator' in self.tools:
                return 'calculator'

        # 代码执行
        if any(keyword in question_lower for keyword in ['运行', '执行', '代码', 'python']):
            if 'code_executor' in self.tools:
                return 'code_executor'

        # 搜索
        if any(keyword in question_lower for keyword in ['搜索', '查找', '最新', '现在']):
            if 'search' in self.tools:
                return 'search'

        # 文件操作
        if any(keyword in question_lower for keyword in ['读取', '写入', '文件']):
            if 'file_ops' in self.tools:
                return 'file_ops'

        return None

    def select_tool(self, question: str, available_tools: List[str]) -> Optional[str]:
        """选择最合适的工具

        基于问题内容和工具描述选择最佳工具。
        """
        if not available_tools:
            return None

        # 简化：选择第一个可用的工具
        return available_tools[0]

    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolCall:
        """调用工具

        Args:
            tool_name: 工具名称
            parameters: 工具参数

        Returns:
            工具调用结果
        """
        if tool_name not in self.tools:
            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                result=f"Error: Tool '{tool_name}' not found",
                success=False,
                timestamp=0.0,
            )

        tool = self.tools[tool_name]

        try:
            # 调用工具函数
            result = tool.function(**parameters)

            # 记录调用
            call = ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                result=result,
                success=True,
                timestamp=0.0,
            )
            self.call_history.append(call)
            self.usage_stats[tool_name] += 1

            return call

        except Exception as e:
            # 记录失败
            call = ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                result=str(e),
                success=False,
                timestamp=0.0,
            )
            self.call_history.append(call)

            return call

    def integrate_result(self, question: str, tool_call: ToolCall) -> str:
        """将工具结果整合到推理中

        Args:
            question: 原始问题
            tool_call: 工具调用结果

        Returns:
            整合后的答案
        """
        if not tool_call.success:
            return f"工具调用失败: {tool_call.result}"

        # 根据工具类型格式化结果
        if tool_call.tool_name == 'calculator':
            return f"计算结果: {tool_call.result}"
        elif tool_call.tool_name == 'code_executor':
            return f"代码执行结果: {tool_call.result}"
        elif tool_call.tool_name == 'search':
            return f"搜索结果: {tool_call.result}"
        else:
            return f"工具'{tool_call.tool_name}'的结果: {tool_call.result}"

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'registered_tools': len(self.tools),
            'total_calls': len(self.call_history),
            'successful_calls': sum(1 for c in self.call_history if c.success),
            'usage_stats': self.usage_stats.copy(),
        }

    def get_report(self) -> str:
        """获取报告"""
        stats = self.get_stats()
        lines = [
            "=== 工具调用引擎报告 ===",
            f"注册工具: {stats['registered_tools']}",
            f"总调用次数: {stats['total_calls']}",
            f"成功调用: {stats['successful_calls']}",
        ]

        if self.tools:
            lines.append("")
            lines.append("可用工具:")
            for name, tool in self.tools.items():
                lines.append(f"  {name}: {tool.description}")

        return '\n'.join(lines)
