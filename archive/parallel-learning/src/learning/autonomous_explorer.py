"""自主探索器 — 好奇心 + 知识缺口驱动

核心思想：系统自己决定学什么，不是被动遍历数据。
像婴儿一样：对新事物好奇→主动探索→发现规律→形成概念。

探索策略：
1. 惊讶度驱动：预测误差高的领域值得探索
2. 新颖度驱动：未探索的领域值得探索
3. 可学习性驱动：误差在下降的领域值得继续
4. 知识缺口驱动：孤立/低置信度概念需要补充
"""

import random
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExplorationTarget:
    """探索目标"""
    description: str
    priority: float           # 优先级 [0, 1]
    reason: str               # 选择原因
    category: str             # curiosity / novelty / gap / learnability


@dataclass
class ExplorationResult:
    """探索结果"""
    target: ExplorationTarget
    action_taken: Dict[str, Any]
    outcome: Dict[str, float]
    surprise: float
    new_knowledge: List[str]  # 新发现的知识


class AutonomousExplorer:
    """自主探索器 — 决定学什么

    与现有系统的区别：
    - 现有：遍历预定义词表
    - 新系统：基于好奇心和知识缺口自主选择
    """

    def __init__(self, exploration_rate: float = 0.3):
        self.exploration_rate = exploration_rate

        # 探索历史
        self.exploration_history: List[ExplorationResult] = []
        self.visited_areas: Dict[str, int] = {}

        # 好奇心状态
        self.curiosity_scores: Dict[str, float] = {}
        self.surprise_history: List[float] = []

        # 知识缺口
        self.knowledge_gaps: List[str] = []

    def select_exploration_target(self, world_model, knowledge_graph,
                                   available_actions: List[Dict]) -> ExplorationTarget:
        """选择下一个探索目标

        综合考虑：
        1. 惊讶度：世界模型预测误差高的领域
        2. 新颖度：未探索过的动作/状态
        3. 知识缺口：知识图谱中孤立/低置信度的概念
        4. 可学习性：误差在下降的领域
        """
        candidates = []

        # 1. 惊讶度驱动
        if world_model.avg_surprise > 0.3:
            candidates.append(ExplorationTarget(
                description="探索高惊讶度领域",
                priority=world_model.avg_surprise,
                reason=f"平均惊讶度 {world_model.avg_surprise:.3f}，需要更新世界模型",
                category='curiosity',
            ))

        # 2. 新颖度驱动
        for action in available_actions:
            action_key = str(sorted(action.items()))
            visits = self.visited_areas.get(action_key, 0)
            novelty = 1.0 / (1.0 + visits)
            if novelty > 0.5:
                candidates.append(ExplorationTarget(
                    description=f"探索新动作: {action}",
                    priority=novelty * 0.8,
                    reason=f"访问次数 {visits}，新颖度 {novelty:.3f}",
                    category='novelty',
                ))

        # 3. 知识缺口驱动
        if knowledge_graph:
            gaps = self._identify_gaps(knowledge_graph)
            for gap in gaps[:3]:
                candidates.append(ExplorationTarget(
                    description=f"补充知识缺口: {gap}",
                    priority=0.7,
                    reason="知识图谱中的孤立/低置信度概念",
                    category='gap',
                ))

        # 4. 可学习性驱动
        if world_model.total_observations > 50:
            learnability = self._compute_learnability(world_model)
            if learnability > 0.5:
                candidates.append(ExplorationTarget(
                    description="继续学习当前领域",
                    priority=learnability * 0.6,
                    reason=f"可学习性 {learnability:.3f}，学习效果好",
                    category='learnability',
                ))

        # 如果没有候选，随机探索
        if not candidates:
            candidates.append(ExplorationTarget(
                description="随机探索",
                priority=0.5,
                reason="没有明确的探索方向",
                category='random',
            ))

        # 选择优先级最高的
        candidates.sort(key=lambda t: t.priority, reverse=True)
        return candidates[0]

    def explore(self, target: ExplorationTarget, environment,
                world_model, knowledge_graph) -> ExplorationResult:
        """执行一次探索

        Args:
            target: 探索目标
            environment: 环境（提供 observe/step 接口）
            world_model: 世界模型
            knowledge_graph: 知识图谱

        Returns:
            探索结果
        """
        # 1. 根据目标选择动作
        action = self._select_action(target, environment)

        # 2. 执行动作
        action_key = str(sorted(action.items()))
        self.visited_areas[action_key] = self.visited_areas.get(action_key, 0) + 1

        # 3. 观察结果
        if hasattr(environment, 'step'):
            observation = environment.step(action)
        else:
            observation = environment

        # 4. 更新世界模型
        state = observation.get('state', {})
        next_state = observation.get('next_state', {})
        surprise = world_model.observe(state, action, next_state)

        # 5. 更新知识图谱
        new_knowledge = self._update_knowledge(
            knowledge_graph, state, action, next_state, surprise
        )

        # 6. 记录结果
        result = ExplorationResult(
            target=target,
            action_taken=action,
            outcome=next_state,
            surprise=surprise,
            new_knowledge=new_knowledge,
        )
        self.exploration_history.append(result)
        self.surprise_history.append(surprise)

        return result

    def get_exploration_stats(self) -> Dict:
        """获取探索统计"""
        if not self.exploration_history:
            return {'total': 0}

        surprises = [r.surprise for r in self.exploration_history]
        return {
            'total_explorations': len(self.exploration_history),
            'avg_surprise': sum(surprises) / len(surprises),
            'max_surprise': max(surprises),
            'unique_areas': len(self.visited_areas),
            'knowledge_discovered': sum(len(r.new_knowledge) for r in self.exploration_history),
            'category_counts': self._count_categories(),
        }

    # ── 内部方法 ──────────────────────────────────────────────

    def _select_action(self, target: ExplorationTarget,
                       environment) -> Dict[str, Any]:
        """根据探索目标选择具体动作"""
        if target.category == 'random':
            if hasattr(environment, 'sample_action'):
                return environment.sample_action()
            return {'type': 'random', 'value': random.random()}

        if target.category == 'curiosity':
            # 选择最可能导致惊讶的动作
            return {'type': 'explore', 'intensity': random.uniform(0.5, 1.0)}

        if target.category == 'novelty':
            # 选择最不熟悉的方向
            return {'type': 'novel', 'direction': random.randint(0, 7)}

        if target.category == 'gap':
            return {'type': 'investigate', 'target': target.description}

        return {'type': 'default'}

    def _identify_gaps(self, knowledge_graph) -> List[str]:
        """识别知识图谱中的缺口"""
        gaps = []

        if not hasattr(knowledge_graph, 'entities'):
            return gaps

        for entity_id, entity in knowledge_graph.entities.items():
            # 低置信度
            if hasattr(entity, 'confidence') and entity.confidence < 0.3:
                gaps.append(f"低置信度: {entity_id}")

            # 孤立节点
            if hasattr(knowledge_graph, 'get_related'):
                related = knowledge_graph.get_related(entity_id)
                if not related:
                    gaps.append(f"孤立概念: {entity_id}")

        return gaps[:5]

    def _compute_learnability(self, world_model) -> float:
        """计算可学习性 = 学习进展 / 总观察数"""
        if world_model.total_observations < 20:
            return 0.5

        # 看最近的惊讶度趋势
        recent = self.surprise_history[-20:]
        if len(recent) < 10:
            return 0.5

        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)

        # 惊讶度下降 = 可学习性高
        if first_half > 0:
            progress = (first_half - second_half) / first_half
            return max(0, min(1, 0.5 + progress))

        return 0.5

    def _update_knowledge(self, knowledge_graph, state, action,
                          next_state, surprise) -> List[str]:
        """更新知识图谱"""
        new_knowledge = []

        if not knowledge_graph:
            return new_knowledge

        # 如果惊讶度高，可能发现了新知识
        if surprise > 0.5:
            # 记录状态转移为知识
            for key, value in next_state.items():
                if key not in state or abs(next_state[key] - state.get(key, 0)) > 0.1:
                    new_knowledge.append(f"发现: {key} = {value}")

        return new_knowledge

    def _count_categories(self) -> Dict[str, int]:
        """统计各类别探索次数"""
        counts = {}
        for r in self.exploration_history:
            cat = r.target.category
            counts[cat] = counts.get(cat, 0) + 1
        return counts
