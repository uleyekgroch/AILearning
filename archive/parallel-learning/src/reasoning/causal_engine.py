"""因果推理引擎 — 从DAG遍历到do-calculus

实现Pearl的因果推理层次：
1. 关联层：P(Y|X) — 观察到X时Y的概率
2. 干预层：P(Y|do(X)) — 干预X时Y的概率
3. 反事实层：P(Y_x|X', Y') — 如果X没有发生，Y会怎样

核心能力：
- 因果发现：从观察数据中学习因果结构
- 干预推理：计算干预的效果
- 反事实推理：回答"如果...会怎样"
"""

import torch
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CausalNode:
    """因果节点"""
    name: str
    value: float = 0.0
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    mechanism: str = 'linear'  # 因果机制类型


@dataclass
class CausalEdge:
    """因果边"""
    source: str
    target: str
    strength: float = 1.0
    mechanism: str = 'linear'


class CausalEngine:
    """因果推理引擎

    实现Pearl因果推理的三个层次。
    """

    def __init__(self):
        # 因果图
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []

        # 观察数据
        self.observations: List[Dict[str, float]] = []

        # 因果机制参数
        self.mechanisms: Dict[str, Dict] = {}

    def add_node(self, name: str, mechanism: str = 'linear'):
        """添加因果节点"""
        if name not in self.nodes:
            self.nodes[name] = CausalNode(name=name, mechanism=mechanism)

    def add_edge(self, source: str, target: str, strength: float = 1.0):
        """添加因果边"""
        self.add_node(source)
        self.add_node(target)

        self.edges.append(CausalEdge(source=source, target=target, strength=strength))

        # 更新节点关系
        self.nodes[source].children.append(target)
        self.nodes[target].parents.append(source)

    def observe(self, data: Dict[str, float]):
        """记录观察数据"""
        self.observations.append(data)

        # 自动发现因果关系
        self._discover_causality(data)

    def _discover_causality(self, data: Dict[str, float]):
        """从观察数据中发现因果关系"""
        # 简化：如果A和B同时出现且A在B之前，认为A可能导致B
        keys = list(data.keys())
        for i, key_a in enumerate(keys):
            for key_b in keys[i+1:]:
                if data[key_a] > 0.5 and data[key_b] > 0.5:
                    # 检查是否已有因果边
                    has_edge = any(
                        e.source == key_a and e.target == key_b
                        for e in self.edges
                    )
                    if not has_edge:
                        self.add_edge(key_a, key_b, strength=0.5)

    def do_intervention(self, node: str, value: float) -> Dict[str, float]:
        """do-calculus干预推理

        计算干预node=value时，所有其他节点的值。

        Args:
            node: 要干预的节点
            value: 干预值

        Returns:
            所有节点的值
        """
        # 初始化所有节点值
        values = {n: 0.0 for n in self.nodes}

        # 设置干预节点的值
        values[node] = value

        # 按拓扑顺序传播
        visited = set()
        self._propagate_intervention(node, value, values, visited)

        return values

    def _propagate_intervention(self, node: str, value: float,
                               values: Dict[str, float], visited: Set[str]):
        """传播干预效果"""
        if node in visited:
            return
        visited.add(node)

        # 传播到子节点
        for child_name in self.nodes[node].children:
            # 计算子节点的值
            child_value = self._compute_child_value(child_name, values)
            values[child_name] = child_value

            # 递归传播
            self._propagate_intervention(child_name, child_value, values, visited)

    def _compute_child_value(self, child_name: str, values: Dict[str, float]) -> float:
        """计算子节点的值"""
        child = self.nodes[child_name]
        if not child.parents:
            return values.get(child_name, 0.0)

        # 简化：取父节点值的加权平均
        total = 0.0
        count = 0
        for parent_name in child.parents:
            parent_value = values.get(parent_name, 0.0)
            # 找到边的强度
            edge_strength = 1.0
            for edge in self.edges:
                if edge.source == parent_name and edge.target == child_name:
                    edge_strength = edge.strength
                    break
            total += parent_value * edge_strength
            count += 1

        return total / max(count, 1)

    def counterfactual(self, observed: Dict[str, float],
                      intervention_node: str, intervention_value: float) -> Dict[str, float]:
        """反事实推理

        给定观察到的数据，如果intervention_node的值是intervention_value，
        其他节点会怎样？

        Args:
            observed: 观察到的数据
            intervention_node: 干预节点
            intervention_value: 干预值

        Returns:
            反事实结果
        """
        # 步骤1：从观察数据中学习因果机制参数
        self._learn_mechanisms(observed)

        # 步骤2：使用学到的机制进行反事实推理
        result = {}

        # 复制观察数据
        for node, value in observed.items():
            result[node] = value

        # 应用干预
        result[intervention_node] = intervention_value

        # 使用学到的机制重新计算受影响的节点
        for node in self.nodes:
            if node == intervention_node:
                continue
            if intervention_node in self.nodes[node].parents:
                # 直接受影响的节点
                result[node] = self._apply_mechanism(node, result)

        return result

    def _learn_mechanisms(self, observed: Dict[str, float]):
        """从观察数据中学习因果机制"""
        for node_name, node in self.nodes.items():
            if node.parents:
                # 学习线性机制的参数
                parent_values = [observed.get(p, 0.0) for p in node.parents]
                if parent_values:
                    # 简化：使用最小二乘法学习权重
                    target_value = observed.get(node_name, 0.0)
                    if sum(parent_values) > 0:
                        weights = [target_value / sum(parent_values)] * len(parent_values)
                    else:
                        weights = [0.0] * len(parent_values)
                    self.mechanisms[node_name] = {
                        'type': 'linear',
                        'weights': weights,
                        'parents': node.parents,
                    }

    def _apply_mechanism(self, node_name: str, values: Dict[str, float]) -> float:
        """应用学到的因果机制"""
        if node_name not in self.mechanisms:
            return values.get(node_name, 0.0)

        mechanism = self.mechanisms[node_name]
        if mechanism['type'] == 'linear':
            total = 0.0
            for parent, weight in zip(mechanism['parents'], mechanism['weights']):
                total += values.get(parent, 0.0) * weight
            return total

        return values.get(node_name, 0.0)

    def get_causal_chain(self, source: str, target: str) -> List[List[str]]:
        """获取从source到target的所有因果链"""
        chains = []
        self._dfs_chains(source, target, [], chains, set())
        return chains

    def _dfs_chains(self, current: str, target: str, path: List[str],
                   chains: List[List[str]], visited: Set[str]):
        """DFS搜索因果链"""
        if current in visited:
            return

        path = path + [current]
        visited.add(current)

        if current == target:
            chains.append(path)
            return

        for child in self.nodes[current].children:
            self._dfs_chains(child, target, path, chains, visited)

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'observations': len(self.observations),
            'mechanisms': len(self.mechanisms),
        }
