"""
改进模块 - 从第一性原理出发

人类改进的本质：
1. 分析问题
2. 理解原因
3. 生成方案
4. 实施改进
5. 评估效果

设计原则：
- 单一职责：只负责改进
- 简洁清晰：代码易于理解
- 可测试：接口明确，易于测试
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class Problem:
    """问题"""
    description: str            # 问题描述
    severity: str               # 严重程度
    location: str               # 位置


@dataclass
class ImprovementPlan:
    """改进计划"""
    problem: Problem            # 问题
    solution: str               # 解决方案
    steps: List[str]            # 步骤
    expected_outcome: str       # 预期结果


@dataclass
class ImprovementResult:
    """改进结果"""
    plan: ImprovementPlan       # 改进计划
    success: bool               # 是否成功
    actual_outcome: str         # 实际结果
    lessons: List[str]          # 经验教训


class ImprovementModule:
    """改进模块

    职责：
    - 错误分析
    - 经验学习
    - 自我优化

    设计原则：
    - 单一职责
    - 简洁清晰
    - 可测试
    """

    def __init__(self):
        """初始化改进模块"""
        # 问题库
        self.problems: List[Problem] = []

        # 改进历史
        self.improvement_history: List[ImprovementResult] = []

        # 经验库
        self.experiences: List[Dict[str, Any]] = []

    def identify_problem(self, description: str, severity: str,
                        location: str) -> Problem:
        """
        识别问题

        Args:
            description: 问题描述
            severity: 严重程度
            location: 位置

        Returns:
            问题对象
        """
        problem = Problem(
            description=description,
            severity=severity,
            location=location
        )

        self.problems.append(problem)
        return problem

    def generate_plan(self, problem: Problem) -> ImprovementPlan:
        """
        生成改进计划

        Args:
            problem: 问题

        Returns:
            改进计划
        """
        # 分析问题
        root_cause = self._analyze_root_cause(problem)

        # 生成解决方案
        solution = self._generate_solution(problem, root_cause)

        # 生成步骤
        steps = self._generate_steps(solution)

        # 预期结果
        expected_outcome = f"通过{solution}，问题应该得到解决"

        return ImprovementPlan(
            problem=problem,
            solution=solution,
            steps=steps,
            expected_outcome=expected_outcome
        )

    def implement_plan(self, plan: ImprovementPlan,
                      actual_outcome: str = None) -> ImprovementResult:
        """
        实施改进计划

        Args:
            plan: 改进计划
            actual_outcome: 实际结果

        Returns:
            改进结果
        """
        # 评估成功
        success = self._evaluate_success(plan, actual_outcome)

        # 提取经验
        lessons = self._extract_lessons(plan, success)

        result = ImprovementResult(
            plan=plan,
            success=success,
            actual_outcome=actual_outcome or plan.expected_outcome,
            lessons=lessons
        )

        self.improvement_history.append(result)

        # 保存经验
        self.experiences.append({
            'problem': plan.problem.description,
            'solution': plan.solution,
            'success': success,
            'lessons': lessons,
        })

        return result

    def learn_from_experience(self) -> List[str]:
        """
        从经验中学习

        Returns:
            学到的教训
        """
        lessons = []

        for exp in self.experiences:
            if exp['success']:
                lessons.append(f"成功经验: {exp['solution']}")
            else:
                lessons.append(f"失败教训: {exp['problem']}")

        return lessons

    def _analyze_root_cause(self, problem: Problem) -> str:
        """
        分析根本原因

        Args:
            problem: 问题

        Returns:
            根本原因
        """
        return f"{problem.location}中的{problem.description}"

    def _generate_solution(self, problem: Problem, root_cause: str) -> str:
        """
        生成解决方案

        Args:
            problem: 问题
            root_cause: 根本原因

        Returns:
            解决方案
        """
        return f"解决{root_cause}的方法"

    def _generate_steps(self, solution: str) -> List[str]:
        """
        生成步骤

        Args:
            solution: 解决方案

        Returns:
            步骤列表
        """
        return [
            f"分析{solution}",
            f"设计{solution}",
            f"实施{solution}",
            f"验证{solution}",
        ]

    def _evaluate_success(self, plan: ImprovementPlan,
                         actual_outcome: str) -> bool:
        """
        评估成功

        Args:
            plan: 计划
            actual_outcome: 实际结果

        Returns:
            是否成功
        """
        if actual_outcome is None:
            return True

        # 简单匹配
        return len(actual_outcome) > 0

    def _extract_lessons(self, plan: ImprovementPlan, success: bool) -> List[str]:
        """
        提取经验

        Args:
            plan: 计划
            success: 是否成功

        Returns:
            经验列表
        """
        lessons = []

        if success:
            lessons.append(f"成功: {plan.solution}")
        else:
            lessons.append(f"失败: {plan.problem.description}")

        return lessons
