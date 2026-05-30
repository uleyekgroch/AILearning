"""因果DAG — 从相关统计到干预推理

不是统计共现，是真正的因果推理。

核心能力：
1. 因果图构建 — 学习变量间的因果关系
2. 干预推理 — do(X=x) 干预
3. 反事实推理 — 如果X没有发生会怎样

运行方式：
    python training/layers/causal_dag.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class CausalNode:
    """因果图节点"""
    name: str
    value: float = 0.0
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    mechanism: Optional[nn.Module] = None  # 因果机制


class CausalMechanism(nn.Module):
    """因果机制网络 — 学习父节点如何影响子节点"""

    def __init__(self, parent_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(parent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, parent_values: torch.Tensor) -> torch.Tensor:
        """预测子节点值"""
        x = F.relu(self.fc1(parent_values))
        return self.fc2(x)


class CausalDAG:
    """因果有向无环图

    实现真正的因果推理，不只是相关性统计。
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 节点
        self.nodes: Dict[str, CausalNode] = {}

        # 边
        self.edges: List[Tuple[str, str]] = []  # (cause, effect)

        # 观测数据
        self.observations: Dict[str, List[float]] = defaultdict(list)

        # 统计
        self.stats = {
            'nodes': 0,
            'edges': 0,
            'interventions': 0,
            'counterfactuals': 0,
        }

    def add_node(self, name: str) -> CausalNode:
        """添加节点"""
        if name not in self.nodes:
            self.nodes[name] = CausalNode(name=name)
            self.stats['nodes'] += 1
        return self.nodes[name]

    def add_edge(self, cause: str, effect: str):
        """添加边（因果关系）"""
        cause_node = self.add_node(cause)
        effect_node = self.add_node(effect)

        # 检查是否形成环
        if self._would_create_cycle(cause, effect):
            return False

        cause_node.children.append(effect)
        effect_node.parents.append(cause)
        self.edges.append((cause, effect))
        self.stats['edges'] += 1

        # 创建因果机制
        parent_dim = len(effect_node.parents)
        effect_node.mechanism = CausalMechanism(parent_dim).to(self.device)

        return True

    def _would_create_cycle(self, cause: str, effect: str) -> bool:
        """检查是否会形成环"""
        # BFS检查effect是否能到达cause
        visited = set()
        queue = [effect]

        while queue:
            current = queue.pop(0)
            if current == cause:
                return True
            if current in visited:
                continue
            visited.add(current)

            if current in self.nodes:
                for child in self.nodes[current].children:
                    queue.append(child)

        return False

    def observe(self, observations: Dict[str, float]):
        """记录观测"""
        for name, value in observations.items():
            self.observations[name].append(value)
            if name in self.nodes:
                self.nodes[name].value = value

    def learn_mechanisms(self):
        """学习因果机制"""
        for name, node in self.nodes.items():
            if not node.parents or not node.mechanism:
                continue

            # 收集父节点值
            parent_values = []
            for parent_name in node.parents:
                parent_values.append(self.observations.get(parent_name, [0.0]))

            if not parent_values:
                continue

            # 训练机制网络
            parent_tensor = torch.tensor(parent_values, dtype=torch.float32).T.to(self.device)
            target_tensor = torch.tensor(self.observations.get(name, [0.0]), dtype=torch.float32).to(self.device)

            # 简单训练
            optimizer = torch.optim.Adam(node.mechanism.parameters(), lr=1e-3)
            for _ in range(100):
                pred = node.mechanism(parent_tensor)
                loss = F.mse_loss(pred.squeeze(), target_tensor[:pred.shape[0]])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def do_intervention(self, variable: str, value: float) -> Dict[str, float]:
        """do(X=x) 干预

        切断所有指向X的边，设置X=value，推断其他变量的变化。
        """
        self.stats['interventions'] += 1

        # 设置干预值
        if variable in self.nodes:
            self.nodes[variable].value = value

        # 按拓扑顺序传播
        result = {variable: value}
        visited = {variable}

        # BFS传播
        queue = [variable]
        while queue:
            current = queue.pop(0)

            if current in self.nodes:
                for child_name in self.nodes[current].children:
                    if child_name in visited:
                        continue

                    child_node = self.nodes[child_name]
                    if child_node.mechanism:
                        # 收集父节点值
                        parent_values = []
                        for parent_name in child_node.parents:
                            parent_values.append(result.get(parent_name, self.nodes[parent_name].value))

                        parent_tensor = torch.tensor([parent_values], dtype=torch.float32).to(self.device)

                        # 预测
                        with torch.no_grad():
                            predicted = child_node.mechanism(parent_tensor)
                            result[child_name] = predicted.item()
                    else:
                        # 如果没有机制，使用简单传播
                        result[child_name] = result.get(current, 0.0)

                    visited.add(child_name)
                    queue.append(child_name)

        return result

    def counterfactual(self, observations: Dict[str, float],
                      intervention: Dict[str, float]) -> Dict[str, float]:
        """反事实推理

        给定实际观测，如果干预了X，其他变量会怎样？
        """
        self.stats['counterfactuals'] += 1

        # 1. 使用实际观测推断噪声
        noise = {}
        for name, value in observations.items():
            if name in self.nodes and self.nodes[name].parents:
                # 计算噪声 = 实际值 - 预测值
                node = self.nodes[name]
                if node.mechanism:
                    parent_values = []
                    for parent_name in node.parents:
                        parent_values.append(observations.get(parent_name, 0.0))

                    parent_tensor = torch.tensor([parent_values], dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        predicted = node.mechanism(parent_tensor)
                        noise[name] = value - predicted.item()

        # 2. 应用干预并传播
        result = dict(observations)
        for var, val in intervention.items():
            result[var] = val

        # 3. 使用噪声重建反事实
        for name, node in self.nodes.items():
            if name in intervention:
                continue
            if node.parents and node.mechanism:
                parent_values = []
                for parent_name in node.parents:
                    parent_values.append(result.get(parent_name, 0.0))

                parent_tensor = torch.tensor([parent_values], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    predicted = node.mechanism(parent_tensor)
                    result[name] = predicted.item() + noise.get(name, 0.0)

        return result

    def query(self, question: str) -> Dict:
        """查询因果知识"""
        # 提取关键词
        keywords = self._extract_keywords(question)

        results = {
            'nodes': [],
            'edges': [],
            'paths': [],
        }

        # 搜索节点
        for keyword in keywords:
            for name, node in self.nodes.items():
                if keyword in name:
                    results['nodes'].append({
                        'name': name,
                        'parents': node.parents,
                        'children': node.children,
                        'value': node.value,
                    })

        # 搜索边
        for cause, effect in self.edges:
            if any(kw in cause or kw in effect for kw in keywords):
                results['edges'].append({
                    'cause': cause,
                    'effect': effect,
                })

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        import re
        # 中文关键词
        zh_keywords = re.findall(r'[一-鿿]{2,6}', text)
        # 英文关键词
        en_keywords = re.findall(r'[a-zA-Z]+', text)

        # 过滤停用词
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样')
        keywords = [k for k in zh_keywords + en_keywords if k not in stopwords and len(k) >= 2]

        return keywords

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'observations': {k: len(v) for k, v in self.observations.items()},
        }


def test_causal_dag():
    """测试因果DAG"""
    print("=" * 70)
    print("因果DAG测试")
    print("=" * 70)

    dag = CausalDAG()

    # 构建因果图
    print("\n1. 构建因果图:")
    dag.add_edge('下雨', '地面湿')
    dag.add_edge('地面湿', '滑倒')
    dag.add_edge('全球变暖', '冰川融化')
    dag.add_edge('冰川融化', '海平面上升')

    print(f"  节点: {len(dag.nodes)}")
    print(f"  边: {len(dag.edges)}")

    # 记录观测
    print("\n2. 记录观测:")
    observations = [
        {'下雨': 1.0, '地面湿': 1.0, '滑倒': 0.0},
        {'下雨': 0.0, '地面湿': 0.0, '滑倒': 0.0},
        {'下雨': 1.0, '地面湿': 1.0, '滑倒': 1.0},
        {'全球变暖': 1.0, '冰川融化': 1.0, '海平面上升': 0.5},
    ]

    for obs in observations:
        dag.observe(obs)
        print(f"  观测: {obs}")

    # 学习因果机制
    print("\n3. 学习因果机制:")
    dag.learn_mechanisms()
    print(f"  学习完成")

    # 干预推理
    print("\n4. 干预推理:")
    result = dag.do_intervention('下雨', 1.0)
    print(f"  do(下雨=1.0):")
    for var, val in result.items():
        print(f"    {var}: {val:.2f}")

    # 反事实推理
    print("\n5. 反事实推理:")
    actual = {'下雨': 1.0, '地面湿': 1.0, '滑倒': 0.0}
    intervention = {'下雨': 0.0}
    counterfactual = dag.counterfactual(actual, intervention)
    print(f"  实际: {actual}")
    print(f"  干预: {intervention}")
    print(f"  反事实: {counterfactual}")

    # 查询
    print("\n6. 查询测试:")
    result = dag.query('下雨导致什么')
    print(f"  节点: {len(result['nodes'])}")
    print(f"  边: {len(result['edges'])}")

    # 统计
    print("\n统计:")
    stats = dag.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    test_causal_dag()
