"""演绎推理 — 基于规则和事实的逻辑推导"""

from dataclasses import dataclass, field
from typing import List, Optional, Set

from src.knowledge.graph import KnowledgeGraph


@dataclass
class LogicRule:
    """逻辑规则：一组前提推导出一个结论"""
    premises: List[str]  # 前提命题列表，如 ['A is_a Bird', 'A can_fly']
    conclusion: str       # 结论命题，如 'A is_a FlyingAnimal'
    confidence: float = 1.0


class DeductiveReasoner:
    """演绎推理器：基于已知的逻辑规则，从给定事实出发推导新结论。"""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.rules: List[LogicRule] = []

    def add_rule(self, premises: List[str], conclusion: str) -> None:
        """添加一条逻辑规则"""
        rule = LogicRule(premises=premises, conclusion=conclusion)
        self.rules.append(rule)

    def deduce(self, facts: List[str], max_depth: int = 5) -> List[str]:
        """从 facts 出发，反复应用 rules 推导新结论。

        算法：前向链推理（forward chaining）
        1. 将初始 facts 放入已知事实集
        2. 在每一轮中，检查所有规则的前提是否满足
        3. 如果满足，将结论加入已知事实集
        4. 重复直到没有新结论产生或达到最大深度
        """
        known: Set[str] = set(facts)
        derived: List[str] = []  # 本次推导出的新结论

        for _ in range(max_depth):
            new_derived_in_round = False

            for rule in self.rules:
                # 检查该规则的所有前提是否都在已知事实中
                if all(premise in known for premise in rule.premises):
                    # 结论还未知则添加
                    if rule.conclusion not in known:
                        known.add(rule.conclusion)
                        derived.append(rule.conclusion)
                        new_derived_in_round = True

            # 没有新结论则停止
            if not new_derived_in_round:
                break

        return derived

    def transitivity_check(self, a: str, b: str, c: str,
                           relation_type: str) -> bool:
        """在知识图谱中检查传递性：a→b 且 b→c（同一关系类型）。

        算法：查询 a 的出边中是否有类型为 relation_type 且目标为 b 的边，
        同时查询 b 的出边中是否有类型为 relation_type 且目标为 c 的边。
        """
        # 检查 a -> b
        a_to_b = False
        for rel in self.graph.get_relations_of(a, direction='out'):
            if rel.type == relation_type and rel.target_id == b:
                a_to_b = True
                break

        if not a_to_b:
            return False

        # 检查 b -> c
        for rel in self.graph.get_relations_of(b, direction='out'):
            if rel.type == relation_type and rel.target_id == c:
                return True

        return False

    def verify_chain(self, premises: List[str], conclusion: str) -> bool:
        """检查从 premises 到 conclusion 是否存在推理链。

        算法：广度优先搜索
        1. 从 premises 出发，逐步应用规则扩展已知事实
        2. 如果在任何步骤中产生了 conclusion，返回 True
        3. 如果推理链耗尽仍未找到，返回 False
        """
        known: Set[str] = set(premises)

        if conclusion in known:
            return True

        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                if all(p in known for p in rule.premises):
                    if rule.conclusion not in known:
                        known.add(rule.conclusion)
                        changed = True
                        if rule.conclusion == conclusion:
                            return True

        return False
