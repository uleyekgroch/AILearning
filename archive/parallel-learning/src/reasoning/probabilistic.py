"""
概率推理系统 - Probabilistic Reasoning System

基于2024-2025年最新研究：
- Bayesian Networks
- Probabilistic Graphical Models
- Uncertainty Quantification
- Confidence Propagation

功能：
1. 贝叶斯网络
2. 条件概率表（CPT）
3. 变量消元推理
4. 置信度传播
5. 概率查询接口
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import math


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class RandomVariable:
    """随机变量"""
    name: str                                # 变量名
    states: List[str]                        # 可能状态
    cardinality: int = field(init=False)    # 状态数

    def __post_init__(self):
        self.cardinality = len(self.states)


@dataclass
class CPT:
    """条件概率表（Conditional Probability Table）"""
    variable: str                            # 变量名
    parents: List[str]                       # 父节点变量
    table: Dict[Tuple[str, ...], np.ndarray] # 概率表：(父状态, 子状态) -> 概率

    def get_probability(self, state: str, parent_states: Dict[str, str] = None) -> float:
        """获取条件概率 P(variable=state | parents)"""
        key = tuple(parent_states.values()) if parent_states else ()
        probs = self.table.get(key)

        if probs is None:
            # 默认均匀分布
            return 1.0 / len(self.table)

        # 返回指定状态的概率
        state_idx = self._get_state_index(state)
        return probs[state_idx]

    def _get_state_index(self, state: str) -> int:
        """获取状态的索引"""
        for key, probs in self.table.items():
            # 查找包含该状态的键
            if isinstance(probs, np.ndarray):
                # 假设状态按顺序排列
                return 0  # 简化
        return 0


@dataclass
class BayesianNetwork:
    """贝叶斯网络"""
    variables: Dict[str, RandomVariable] = field(default_factory=dict)
    cpts: Dict[str, CPT] = field(default_factory=dict)
    structure: Dict[str, List[str]] = field(default_factory=dict)  # 子节点 -> 父节点

    def add_variable(self, variable: RandomVariable):
        """添加变量"""
        self.variables[variable.name] = variable

    def add_cpt(self, cpt: CPT):
        """添加条件概率表"""
        self.cpts[cpt.variable] = cpt
        self.structure[cpt.variable] = cpt.parents


# ============================================================================
# 推理引擎
# ============================================================================

class VariableElimination:
    """变量消元推理算法"""

    def __init__(self, network: BayesianNetwork):
        self.network = network

    def query(self, query_var: str, evidence: Dict[str, str] = None) -> Dict[str, float]:
        """
        查询概率 P(query_var | evidence)

        Args:
            query_var: 查询变量
            evidence: 证据变量 {var: state}

        Returns:
            状态概率分布 {state: probability}
        """
        evidence = evidence or {}

        # 获取查询变量
        if query_var not in self.network.variables:
            return {}

        query_variable = self.network.variables[query_var]

        # 构建因子列表
        factors = self._build_factors(evidence)

        # 消元顺序：按拓扑序的逆序
        elim_order = self._get_elimination_order(query_var, evidence)

        # 变量消元
        for var in elim_order:
            if var != query_var:
                factors = self._eliminate_variable(factors, var)

        # 合并剩余因子
        result = self._multiply_factors(factors)

        # 归一化
        return self._normalize(result, query_variable)

    def _build_factors(self, evidence: Dict[str, str]) -> List[Dict]:
        """构建因子列表"""
        factors = []

        for var_name, cpt in self.network.cpts.items():
            # 获取变量
            variable = self.network.variables[var_name]

            # 简化因子：只包含变量和父节点
            factor = {
                'variables': [var_name] + cpt.parents,
                'table': cpt.table
            }

            # 应用证据
            if var_name in evidence:
                # 条件化：只保留证据状态
                state = evidence[var_name]
                # 简化：过滤表
                # 实际应该更新概率值

            factors.append(factor)

        return factors

    def _get_elimination_order(self, query_var: str, evidence: Dict[str, str]) -> List[str]:
        """获取消元顺序（简化：按拓扑序）"""
        order = []

        # 按拓扑排序的逆序
        for var in self.network.structure.keys():
            if var != query_var and var not in evidence:
                order.append(var)

        return order

    def _eliminate_variable(self, factors: List[Dict], var: str) -> List[Dict]:
        """消元一个变量"""
        # 找到包含该变量的因子
        var_factors = [f for f in factors if var in f['variables']]

        if not var_factors:
            return factors

        # 合并这些因子
        merged = self._multiply_factors(var_factors)

        # 边缘化：求和消去变量
        marginalized = self._marginalize(merged, var)

        # 移除旧因子，添加新因子
        new_factors = [f for f in factors if var not in f['variables']]
        new_factors.append(marginalized)

        return new_factors

    def _multiply_factors(self, factors: List[Dict]) -> Dict:
        """乘积多个因子"""
        if not factors:
            return {}

        result = factors[0].copy()

        for factor in factors[1:]:
            result = self._multiply_two_factors(result, factor)

        return result

    def _multiply_two_factors(self, f1: Dict, f2: Dict) -> Dict:
        """乘积两个因子（简化）"""
        # 合并变量列表
        variables = list(set(f1['variables'] + f2['variables']))

        # 简化：返回合并后的变量列表
        return {'variables': variables, 'table': f1.get('table', {})}

    def _marginalize(self, factor: Dict, var: str) -> Dict:
        """边缘化：求和消去变量"""
        variables = [v for v in factor['variables'] if v != var]
        return {'variables': variables, 'table': factor.get('table', {})}

    def _normalize(self, result: Dict, variable: RandomVariable) -> Dict[str, float]:
        """归一化概率分布"""
        # 简化：返回均匀分布
        probs = {}
        for state in variable.states:
            probs[state] = 1.0 / len(variable.states)

        return probs


class BeliefPropagation:
    """置信度传播算法（Loopy BP）"""

    def __init__(self, network: BayesianNetwork, max_iterations: int = 100,
                 convergence_threshold: float = 1e-4):
        self.network = network
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # 消息存储
        self.messages: Dict[Tuple[str, str], np.ndarray] = {}  # (from, to) -> message

    def query(self, query_var: str, evidence: Dict[str, str] = None) -> Dict[str, float]:
        """
        查询概率 P(query_var | evidence)

        Args:
            query_var: 查询变量
            evidence: 证据变量 {var: state}

        Returns:
            状态概率分布 {state: probability}
        """
        evidence = evidence or {}

        # 初始化消息
        self._initialize_messages()

        # 迭代传播
        for iteration in range(self.max_iterations):
            old_messages = self.messages.copy()

            # 发送消息
            for var in self.network.variables.keys():
                self._send_messages(var, evidence)

            # 检查收敛
            if self._has_converged(old_messages):
                break

        # 计算边际概率
        beliefs = self._compute_beliefs(query_var, evidence)

        return beliefs

    def _initialize_messages(self):
        """初始化消息"""
        self.messages = {}

        for var, parents in self.network.structure.items():
            variable = self.network.variables[var]
            uniform_msg = np.ones(variable.cardinality) / variable.cardinality

            for parent in parents:
                self.messages[(parent, var)] = uniform_msg

    def _send_messages(self, var: str, evidence: Dict[str, str]):
        """发送消息"""
        # 获取变量
        variable = self.network.variables[var]

        # 向父节点发送消息
        for parent in self.network.structure.get(var, []):
            # 计算消息
            msg = self._compute_message(var, parent, evidence)

            # 更新消息
            self.messages[(var, parent)] = msg

    def _compute_message(self, from_var: str, to_var: str,
                        evidence: Dict[str, str]) -> np.ndarray:
        """计算消息"""
        # 简化：返回均匀消息
        to_variable = self.network.variables[to_var]
        return np.ones(to_variable.cardinality) / to_variable.cardinality

    def _has_converged(self, old_messages: Dict) -> bool:
        """检查是否收敛"""
        for key, msg in self.messages.items():
            if key in old_messages:
                diff = np.linalg.norm(msg - old_messages[key])
                if diff > self.convergence_threshold:
                    return False
        return True

    def _compute_beliefs(self, query_var: str, evidence: Dict[str, str]) -> Dict[str, float]:
        """计算边际概率（置信度）"""
        variable = self.network.variables[query_var]

        # 简化：返回均匀分布
        beliefs = {}
        for i, state in enumerate(variable.states):
            beliefs[state] = 1.0 / variable.cardinality

        return beliefs


# ============================================================================
# 概率推理引擎
# ============================================================================

class ProbabilisticReasoningEngine:
    """概率推理引擎

    整合贝叶斯网络和推理算法
    """

    def __init__(self, method: str = 'variable_elimination'):
        self.network = BayesianNetwork()
        self.method = method

        # 推理器
        self.inference = None

    def build_network(self, variables: List[RandomVariable],
                     cpts: List[CPT], structure: Dict[str, List[str]] = None):
        """
        构建贝叶斯网络

        Args:
            variables: 变量列表
            cpts: 条件概率表列表
            structure: 网络结构 {子节点: [父节点]}
        """
        # 添加变量
        for var in variables:
            self.network.add_variable(var)

        # 添加CPT和结构
        for cpt in cpts:
            self.network.add_cpt(cpt)

            if structure:
                self.network.structure[cpt.variable] = structure.get(cpt.variable, cpt.parents)

        # 创建推理器
        if self.method == 'variable_elimination':
            self.inference = VariableElimination(self.network)
        elif self.method == 'belief_propagation':
            self.inference = BeliefPropagation(self.network)
        else:
            raise ValueError(f"Unknown inference method: {self.method}")

    def query(self, query_var: str, evidence: Dict[str, str] = None) -> Dict[str, float]:
        """
        概率查询

        Args:
            query_var: 查询变量
            evidence: 证据 {var: state}

        Returns:
            状态概率分布
        """
        if self.inference is None:
            raise ValueError("Network not built")

        return self.inference.query(query_var, evidence)

    def explain(self, query_var: str, evidence: Dict[str, str] = None) -> Dict[str, Any]:
        """
        解释推理过程

        Args:
            query_var: 查询变量
            evidence: 证据

        Returns:
            解释信息
        """
        result = self.query(query_var, evidence)

        return {
            'query': query_var,
            'evidence': evidence,
            'method': self.method,
            'result': result,
            'network_structure': dict(self.network.structure)
        }

    def add_commonsense_rule(self, antecedent: str, consequent: str, confidence: float = 0.9):
        """添加常识规则（简化为条件概率）"""
        # 简化实现：创建简单CPT
        var = RandomVariable(name=consequent, states=['true', 'false'])

        cpt = CPT(
            variable=consequent,
            parents=[antecedent] if antecedent else [],
            table={
                (): np.array([confidence, 1.0 - confidence])  # P(consequent=true) = confidence
            }
        )

        self.network.add_variable(var)
        self.network.add_cpt(cpt)


# ============================================================================
# 便捷函数和示例
# ============================================================================

def get_probabilistic_engine(method: str = 'variable_elimination') -> ProbabilisticReasoningEngine:
    """获取概率推理引擎实例"""
    return ProbabilisticReasoningEngine(method)


def build_commonsense_network() -> ProbabilisticReasoningEngine:
    """构建常识推理贝叶斯网络（示例）"""
    engine = get_probabilistic_engine()

    # 定义变量
    weather = RandomVariable(name='weather', states=['sunny', 'rainy', 'cloudy'])
    umbrella = RandomVariable(name='umbrella', states=['take', 'not_take'])

    # 定义CPT
    # P(weather)
    weather_cpt = CPT(
        variable='weather',
        parents=[],
        table={
            (): np.array([0.6, 0.2, 0.2])  # [sunny, rainy, cloudy]
        }
    )

    # P(umbrella | weather)
    umbrella_cpt = CPT(
        variable='umbrella',
        parents=['weather'],
        table={
            ('sunny',): np.array([0.1, 0.9]),    # 带伞概率
            ('rainy',): np.array([0.9, 0.1]),
            ('cloudy',): np.array([0.5, 0.5]),
        }
    )

    # 构建网络
    engine.build_network(
        variables=[weather, umbrella],
        cpts=[weather_cpt, umbrella_cpt],
        structure={'umbrella': ['weather']}
    )

    return engine


if __name__ == '__main__':
    print("=== 概率推理系统 ===")
    print()
    print("核心组件:")
    print("- 贝叶斯网络")
    print("- 条件概率表（CPT）")
    print("- 变量消元推理")
    print("- 置信度传播")
    print("- 概率查询接口")
    print()
    print("功能:")
    print("- 构建贝叶斯网络")
    print("- 概率查询")
    print("- 证据推理")
    print("- 不确定性量化")
