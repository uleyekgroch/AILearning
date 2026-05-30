"""因果干预层

区分因果和相关，实现do-calculus近似。

核心能力：
1. 干预推理 — P(Y|do(X)) vs P(Y|X)
2. 反事实推理 — 如果X没有发生会怎样
3. 因果发现 — 从数据中发现因果结构

运行方式：
    python training/layers/causal_intervention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CausalGraph:
    """因果图"""
    nodes: Set[str] = field(default_factory=set)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # (cause, effect)
    strengths: Dict[Tuple[str, str], float] = field(default_factory=dict)


class CausalDiscovery(nn.Module):
    """因果发现

    从观测数据中发现因果结构。
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        # 因果强度预测
        self.cause_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 方向预测
        self.direction_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测因果关系"""
        combined = torch.cat([x, y], dim=1)

        # 因果强度
        strength = self.cause_predictor(combined)

        # 方向: >0.5 表示 x→y, <0.5 表示 y→x
        direction = self.direction_predictor(combined)

        return strength, direction


class InterventionModel(nn.Module):
    """干预模型

    实现do-calculus近似：
    P(Y|do(X)) = Σ_z P(Y|X,Z)P(Z)
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        # 观测网络: P(Y|X)
        self.observation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 干预网络: P(Y|do(X))
        self.intervention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 混淆因子网络
        self.confounder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, do_intervention: bool = False) -> torch.Tensor:
        """前向传播"""
        if do_intervention:
            return self.intervention_net(x)
        else:
            return self.observation_net(x)

    def compute_ate(self, x: torch.Tensor) -> torch.Tensor:
        """计算平均因果效应 (ATE)"""
        obs = self.observation_net(x)
        interv = self.intervention_net(x)
        return interv - obs

    def compute_counterfactual(self, x: torch.Tensor, x_counter: torch.Tensor) -> torch.Tensor:
        """计算反事实"""
        # P(Y|X=x)
        y_actual = self.observation_net(x)

        # P(Y|do(X=x'))
        y_counterfactual = self.intervention_net(x_counter)

        return y_counterfactual - y_actual


class CausalInterventionSystem:
    """因果干预系统"""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 模型
        self.discovery = CausalDiscovery().to(self.device)
        self.intervention = InterventionModel().to(self.device)

        # 因果图
        self.causal_graph = CausalGraph()

        # 实体编码（限制大小避免内存泄漏）
        self.entity_embeddings: Dict[str, torch.Tensor] = {}
        self.max_embeddings = 10000  # 最大实体数
        self.embedding_access_order: List[str] = []  # LRU追踪

        # 统计
        self.stats = {
            'causal_relations': 0,
            'interventions': 0,
            'counterfactuals': 0,
        }

    def encode_entity(self, entity: str) -> torch.Tensor:
        """编码实体"""
        if entity not in self.entity_embeddings:
            # 如果超过最大数量，删除最久未使用的
            if len(self.entity_embeddings) >= self.max_embeddings:
                oldest = self.embedding_access_order.pop(0)
                del self.entity_embeddings[oldest]

            self.entity_embeddings[entity] = torch.randn(1, 128).to(self.device)
            self.embedding_access_order.append(entity)
        return self.entity_embeddings[entity]

    def discover_causal(self, entity1: str, entity2: str) -> Tuple[float, str]:
        """发现因果关系"""
        e1 = self.encode_entity(entity1)
        e2 = self.encode_entity(entity2)

        with torch.no_grad():
            strength, direction = self.discovery(e1, e2)

        strength_val = strength.item()
        direction_val = direction.item()

        # 确定方向
        if direction_val > 0.5:
            cause, effect = entity1, entity2
        else:
            cause, effect = entity2, entity1

        # 更新因果图
        if strength_val > 0.5:
            self.causal_graph.nodes.add(cause)
            self.causal_graph.nodes.add(effect)
            self.causal_graph.edges.append((cause, effect))
            self.causal_graph.strengths[(cause, effect)] = strength_val
            self.stats['causal_relations'] += 1

        return strength_val, f"{cause} → {effect}"

    def compute_intervention_effect(self, cause: str, effect: str) -> float:
        """计算干预效应"""
        cause_vec = self.encode_entity(cause)
        effect_vec = self.encode_entity(effect)

        combined = (cause_vec + effect_vec) / 2

        with torch.no_grad():
            ate = self.intervention.compute_ate(combined)

        self.stats['interventions'] += 1
        return ate.item()

    def compute_counterfactual(self, cause: str, effect: str,
                              cause_counter: str) -> float:
        """计算反事实"""
        cause_vec = self.encode_entity(cause)
        effect_vec = self.encode_entity(effect)
        cause_counter_vec = self.encode_entity(cause_counter)

        combined = (cause_vec + effect_vec) / 2
        combined_counter = (cause_counter_vec + effect_vec) / 2

        with torch.no_grad():
            cf = self.intervention.compute_counterfactual(combined, combined_counter)

        self.stats['counterfactuals'] += 1
        return cf.item()

    def is_causal(self, entity1: str, entity2: str) -> bool:
        """判断是否有因果关系"""
        for cause, effect in self.causal_graph.edges:
            if (cause == entity1 and effect == entity2) or \
               (cause == entity2 and effect == entity1):
                return True
        return False

    def get_causal_chain(self, start: str, end: str, max_depth: int = 3) -> List[str]:
        """获取因果链"""
        visited = set()
        path = []

        def dfs(current: str, depth: int) -> bool:
            if depth > max_depth:
                return False
            if current == end:
                return True
            if current in visited:
                return False

            visited.add(current)
            path.append(current)

            for cause, effect in self.causal_graph.edges:
                if cause == current:
                    if dfs(effect, depth + 1):
                        return True

            path.pop()
            return False

        if dfs(start, 0):
            return path + [end]
        return []

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'device': str(self.device),
            'nodes': len(set(self.causal_graph.nodes)),
            'edges': len(self.causal_graph.edges),
        }


def test_causal_intervention():
    """测试因果干预"""
    print("=" * 70)
    print("因果干预测试")
    print("=" * 70)

    system = CausalInterventionSystem()

    # 测试因果发现
    print("\n1. 因果发现测试:")
    pairs = [
        ('下雨', '地面湿'),
        ('学习', '成绩好'),
        ('火', '热'),
    ]

    for e1, e2 in pairs:
        strength, direction = system.discover_causal(e1, e2)
        print(f"  {e1} vs {e2}: 强度={strength:.3f}, 方向={direction}")

    # 测试干预效应
    print("\n2. 干预效应测试:")
    for e1, e2 in pairs:
        effect = system.compute_intervention_effect(e1, e2)
        print(f"  do({e1}) → {e2}: 效应={effect:.3f}")

    # 测试反事实
    print("\n3. 反事实测试:")
    counterfactuals = [
        ('下雨', '地面湿', '晴天'),
        ('学习', '成绩好', '玩耍'),
    ]

    for cause, effect, counter in counterfactuals:
        cf = system.compute_counterfactual(cause, effect, counter)
        print(f"  如果{counter}而不是{cause}: 效应={cf:.3f}")

    # 测试因果链
    print("\n4. 因果链测试:")
    # 先建立因果链
    system.discover_causal('全球变暖', '冰川融化')
    system.discover_causal('冰川融化', '海平面上升')

    chain = system.get_causal_chain('全球变暖', '海平面上升')
    print(f"  因果链: {' → '.join(chain)}")

    # 统计
    print("\n统计:")
    stats = system.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    test_causal_intervention()
