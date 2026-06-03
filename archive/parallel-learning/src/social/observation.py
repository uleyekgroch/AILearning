"""观察学习 — 通过观察他人行为来学习"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.knowledge.graph import KnowledgeGraph
from src.knowledge.entity import Entity
from src.knowledge.relation import Relation, REL_CAUSES


@dataclass
class Observation:
    agent_id: str
    action: str
    context: Dict[str, Any]
    outcome: str
    reward: float
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            'agent_id': self.agent_id,
            'action': self.action,
            'context': self.context,
            'outcome': self.outcome,
            'reward': self.reward,
            'timestamp': self.timestamp,
        }


class ObservationLearner:
    """观察学习模块

    通过观察其他 Agent 的行为和结果来学习。
    核心机制：记录观察 → 提取成功模式 → 评估模仿倾向。
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.observations: List[Observation] = []
        self._agent_stats: Dict[str, Dict] = defaultdict(
            lambda: {'total': 0, 'success': 0, 'rewards': []}
        )

    def observe_agent(self, agent_id: str, action: str,
                      context: Dict, outcome: str,
                      reward: float, timestamp: float = 0.0) -> None:
        """记录对某 Agent 行为的观察"""
        obs = Observation(
            agent_id=agent_id, action=action,
            context=context, outcome=outcome,
            reward=reward, timestamp=timestamp,
        )
        self.observations.append(obs)

        self._agent_stats[agent_id]['total'] += 1
        if reward > 0:
            self._agent_stats[agent_id]['success'] += 1
        self._agent_stats[agent_id]['rewards'].append(reward)

    def extract_successful_strategies(self, min_count: int = 3) -> List[Dict]:
        """提取被多个 Agent 验证的成功模式"""
        action_outcomes: Dict[str, List[Dict]] = defaultdict(list)

        for obs in self.observations:
            if obs.reward > 0:
                key = obs.action
                action_outcomes[key].append({
                    'agent': obs.agent_id,
                    'context': obs.context,
                    'outcome': obs.outcome,
                    'reward': obs.reward,
                })

        strategies = []
        for action, cases in action_outcomes.items():
            if len(cases) >= min_count:
                avg_reward = sum(c['reward'] for c in cases) / len(cases)
                contexts = [c['context'] for c in cases]
                common_context = self._extract_common(contexts)
                strategies.append({
                    'action': action,
                    'success_count': len(cases),
                    'avg_reward': avg_reward,
                    'common_context': common_context,
                    'confidence': min(1.0, len(cases) / 10.0),
                })

        return sorted(strategies, key=lambda s: s['avg_reward'], reverse=True)

    def should_imitate(self, agent_id: str, context: Optional[Dict] = None) -> float:
        """计算对某 Agent 在某情境下的模仿倾向

        返回 0~1 的模仿概率，基于：
        - 该 Agent 的历史成功率
        - 当前情境与该 Agent 成功情境的匹配度
        """
        stats = self._agent_stats.get(agent_id)
        if stats is None or stats['total'] == 0:
            return 0.0

        base_rate = stats['success'] / stats['total']

        if context:
            successful_obs = [
                o for o in self.observations
                if o.agent_id == agent_id and o.reward > 0
            ]
            if successful_obs:
                context_matches = sum(
                    1 for o in successful_obs
                    if self._context_similarity(o.context, context) > 0.5
                )
                context_bonus = min(0.3, context_matches / len(successful_obs) * 0.3)
                base_rate = min(1.0, base_rate + context_bonus)

        return base_rate

    def get_observation_count(self, agent_id: Optional[str] = None) -> int:
        if agent_id:
            return sum(1 for o in self.observations if o.agent_id == agent_id)
        return len(self.observations)

    # ── 持久化 ────────────────────────────────────────────────

    def save_state(self) -> dict:
        return {
            'observations': [o.to_dict() for o in self.observations],
        }

    def load_state(self, state: dict) -> None:
        self.observations.clear()
        self._agent_stats.clear()
        for od in state.get('observations', []):
            obs = Observation(
                agent_id=od['agent_id'], action=od['action'],
                context=od.get('context', {}), outcome=od['outcome'],
                reward=od.get('reward', 0.0),
                timestamp=od.get('timestamp', 0.0),
            )
            self.observations.append(obs)
            self._agent_stats[obs.agent_id]['total'] += 1
            if obs.reward > 0:
                self._agent_stats[obs.agent_id]['success'] += 1
            self._agent_stats[obs.agent_id]['rewards'].append(obs.reward)

    # ── 内部方法 ──────────────────────────────────────────────

    def _extract_common(self, contexts: List[Dict]) -> Dict:
        if not contexts:
            return {}
        common = dict(contexts[0])
        for ctx in contexts[1:]:
            keys_to_remove = [k for k in common if common.get(k) != ctx.get(k)]
            for k in keys_to_remove:
                del common[k]
        return common

    def _context_similarity(self, a: Dict, b: Dict) -> float:
        if not a or not b:
            return 0.0
        shared = set(a.keys()) & set(b.keys())
        if not shared:
            return 0.0
        matches = sum(1 for k in shared if a[k] == b[k])
        return matches / len(shared)
