"""创造性引擎 — 从无到有

实现创造性思维的核心能力：
1. 概念组合：将已有概念组合成新概念
2. 类比推理：从一个领域映射到另一个领域
3. 反事实想象：想象不存在的情况
4. 约束放松：放松限制以产生新想法

设计原则：
- 创造性 = 新颖性 + 有用性
- 从已有知识出发，而非凭空创造
- 渐进式创新：小步改进，而非革命性突破
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import random


@dataclass
class CreativeIdea:
    """创造性想法"""
    description: str
    novelty: float  # 0-1
    usefulness: float  # 0-1
    source_concepts: List[str]
    generation_method: str


@dataclass
class ConceptCombination:
    """概念组合"""
    concept_a: str
    concept_b: str
    result: str
    relationship: str


class CreativityEngine:
    """创造性引擎

    生成新的概念、想法和解决方案。
    """

    def __init__(self):
        # 概念库
        self.concepts: Dict[str, Dict] = {}

        # 组合历史
        self.combinations: List[ConceptCombination] = []

        # 生成的想法
        self.ideas: List[CreativeIdea] = []

        # 创造性策略
        self.strategies = {
            'combine': self._combine_concepts,
            'analogy': self._analogy_transfer,
            'counterfactual': self._counterfactual_imagine,
            'relax_constraint': self._relax_constraint,
        }

    def add_concept(self, name: str, properties: Dict):
        """添加概念"""
        self.concepts[name] = properties

    def combine_concepts(self, concept_a: str, concept_b: str,
                        relationship: str = 'merge') -> CreativeIdea:
        """组合两个概念

        Args:
            concept_a: 第一个概念
            concept_b: 第二个概念
            relationship: 组合关系

        Returns:
            创造性想法
        """
        # 生成新概念
        new_concept = f"{concept_a}_{concept_b}"

        # 计算新颖性
        novelty = self._calculate_novelty(new_concept)

        # 计算有用性
        usefulness = self._calculate_usefulness(concept_a, concept_b)

        # 创建想法
        idea = CreativeIdea(
            description=f"组合'{concept_a}'和'{concept_b}'形成'{new_concept}'",
            novelty=novelty,
            usefulness=usefulness,
            source_concepts=[concept_a, concept_b],
            generation_method='combine',
        )

        # 记录组合
        self.combinations.append(ConceptCombination(
            concept_a=concept_a,
            concept_b=concept_b,
            result=new_concept,
            relationship=relationship,
        ))

        self.ideas.append(idea)
        return idea

    def _combine_concepts(self, concepts: List[str]) -> CreativeIdea:
        """组合概念策略"""
        if len(concepts) < 2:
            return CreativeIdea(
                description="需要至少两个概念进行组合",
                novelty=0.0,
                usefulness=0.0,
                source_concepts=concepts,
                generation_method='combine',
            )

        return self.combine_concepts(concepts[0], concepts[1])

    def _analogy_transfer(self, source: str, target: str) -> CreativeIdea:
        """类比迁移策略"""
        # 从源领域映射到目标领域
        new_idea = f"将'{source}'的原理应用到'{target}'"

        idea = CreativeIdea(
            description=new_idea,
            novelty=0.7,
            usefulness=0.6,
            source_concepts=[source, target],
            generation_method='analogy',
        )

        self.ideas.append(idea)
        return idea

    def _counterfactual_imagine(self, situation: str) -> CreativeIdea:
        """反事实想象策略"""
        # 想象不存在的情况
        new_idea = f"如果'{situation}'不存在会怎样？"

        idea = CreativeIdea(
            description=new_idea,
            novelty=0.8,
            usefulness=0.5,
            source_concepts=[situation],
            generation_method='counterfactual',
        )

        self.ideas.append(idea)
        return idea

    def _relax_constraint(self, constraint: str) -> CreativeIdea:
        """放松约束策略"""
        # 放松限制以产生新想法
        new_idea = f"如果'{constraint}'的限制不存在会怎样？"

        idea = CreativeIdea(
            description=new_idea,
            novelty=0.6,
            usefulness=0.7,
            source_concepts=[constraint],
            generation_method='relax_constraint',
        )

        self.ideas.append(idea)
        return idea

    def _calculate_novelty(self, concept: str) -> float:
        """计算新颖性"""
        # 检查是否已存在
        if concept in self.concepts:
            return 0.1

        # 检查组合历史
        for combo in self.combinations:
            if combo.result == concept:
                return 0.3

        # 新概念
        return 0.9

    def _calculate_usefulness(self, concept_a: str, concept_b: str) -> float:
        """计算有用性"""
        # 简化：基于概念的属性数量
        props_a = len(self.concepts.get(concept_a, {}))
        props_b = len(self.concepts.get(concept_b, {}))

        # 属性越多，组合越有用
        return min(1.0, (props_a + props_b) / 10.0)

    def generate_ideas(self, topic: str, n_ideas: int = 3) -> List[CreativeIdea]:
        """生成多个创造性想法

        Args:
            topic: 主题
            n_ideas: 要生成的想法数量

        Returns:
            创造性想法列表
        """
        ideas = []

        # 使用不同策略生成想法
        strategies = list(self.strategies.values())

        for i in range(n_ideas):
            strategy = strategies[i % len(strategies)]

            if strategy == self._combine_concepts:
                # 随机选择两个概念
                concept_list = list(self.concepts.keys())
                if len(concept_list) >= 2:
                    a, b = random.sample(concept_list, 2)
                    idea = strategy([a, b])
                else:
                    idea = strategy([topic, 'new'])
            elif strategy == self._analogy_transfer:
                idea = strategy(topic, 'new_domain')
            elif strategy == self._counterfactual_imagine:
                idea = strategy(topic)
            elif strategy == self._relax_constraint:
                idea = strategy(f'{topic}的限制')
            else:
                idea = CreativeIdea(
                    description=f"关于'{topic}'的创造性想法",
                    novelty=0.5,
                    usefulness=0.5,
                    source_concepts=[topic],
                    generation_method='random',
                )

            ideas.append(idea)

        return ideas

    def get_best_ideas(self, n: int = 5) -> List[CreativeIdea]:
        """获取最佳想法"""
        # 按新颖性+有用性排序
        sorted_ideas = sorted(
            self.ideas,
            key=lambda x: x.novelty + x.usefulness,
            reverse=True,
        )
        return sorted_ideas[:n]

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'concepts': len(self.concepts),
            'combinations': len(self.combinations),
            'ideas': len(self.ideas),
            'avg_novelty': sum(i.novelty for i in self.ideas) / max(len(self.ideas), 1),
            'avg_usefulness': sum(i.usefulness for i in self.ideas) / max(len(self.ideas), 1),
        }

    def get_report(self) -> str:
        """获取报告"""
        stats = self.get_stats()
        lines = [
            "=== 创造性引擎报告 ===",
            f"概念数量: {stats['concepts']}",
            f"组合数量: {stats['combinations']}",
            f"想法数量: {stats['ideas']}",
            f"平均新颖性: {stats['avg_novelty']:.2f}",
            f"平均有用性: {stats['avg_usefulness']:.2f}",
        ]

        best = self.get_best_ideas(3)
        if best:
            lines.append("")
            lines.append("最佳想法:")
            for idea in best:
                lines.append(f"  [{idea.generation_method}] {idea.description}")
                lines.append(f"    新颖性: {idea.novelty:.2f}, 有用性: {idea.usefulness:.2f}")

        return '\n'.join(lines)
