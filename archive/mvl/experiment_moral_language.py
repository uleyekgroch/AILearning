"""
Phase 76: 道德语言涌现 — 公平、欺骗、利他信号

核心思想：
前面的 Phase 证明了信任/欺骗（Phase 68）和声誉系统。
但信任只是道德的底线——更高级的道德概念（公平、利他、贪婪）
是否也会在资源分配压力下涌现为语言信号？

本阶段测试：
- "fair"/"unfair" 标记从资源分配的不平等中涌现
- "share"/"greedy"/"kind" 标记从利他 vs 自私行为对比中涌现
- 道德语言提高群体合作率和资源分配公平性
- 自私 Agent 通过声誉系统被识别和排斥

涌现条件：
1. 资源有限，分配不均导致合作破裂
2. Agent 需要通过道德评价建立合作期望
3. 道德标记提供关于未来行为的预测信息
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import EmergingLanguage, _symbol_category, COLORS, SHAPES


# 道德标记常量
MORAL_MARKERS = {
    'fair': {'valence': 1.0, 'category': 'judgment'},
    'unfair': {'valence': -1.0, 'category': 'judgment'},
    'share': {'valence': 1.0, 'category': 'action'},
    'greedy': {'valence': -1.0, 'category': 'judgment'},
    'kind': {'valence': 1.0, 'category': 'judgment'},
}


class ResourceGame:
    """
    独裁者式资源分配游戏

    Allocator 决定如何分配 total_resources 给多个 recipient。
    其他 agent 评价分配的公平性并产生道德语言。
    """

    def __init__(self, total_resources: int = 10):
        self.total_resources = total_resources
        self.history: List[Dict] = []

    def allocate(self, allocator_id: int,
                 shares: Dict[int, int]) -> Dict:
        """
        执行一次资源分配

        Args:
            allocator_id: 分配者的 ID
            shares: {agent_id: share} 各 agent 获得的资源量

        Returns:
            包含分配详情和评估的字典
        """
        total_allocated = sum(shares.values())
        if total_allocated != self.total_resources:
            # 按比例归一化
            if total_allocated > 0:
                scale = self.total_resources / total_allocated
                shares = {k: max(0, round(v * scale)) for k, v in shares.items()}
            # 修正舍入误差
            diff = self.total_resources - sum(shares.values())
            if diff != 0 and shares:
                max_key = max(shares, key=shares.get)
                shares[max_key] += diff

        fairness = self.evaluate_fairness(shares)
        result = {
            'allocator_id': allocator_id,
            'shares': dict(shares),
            'fairness': round(fairness, 4),
            'total': self.total_resources,
        }
        self.history.append(result)
        return result

    def evaluate_fairness(self, shares: Dict[int, int]) -> float:
        """
        评估分配的公平性（Gini 系数的逆，1 = 完全平等）

        使用归一化标准差：完美平等时为 1.0，完全不平等时为 0.0
        """
        if not shares:
            return 0.0
        vals = list(shares.values())
        if sum(vals) == 0:
            return 1.0

        mean_val = np.mean(vals)
        if mean_val == 0:
            return 1.0

        # 归一化标准差：变异系数的逆
        std_val = np.std(vals)
        cv = std_val / mean_val  # 变异系数
        fairness = 1.0 / (1.0 + cv)  # 映射到 [0, 1]
        return float(fairness)

    def detect_cheating(self, allocator_id: int,
                        allocation: Dict[int, int],
                        agreement: Dict[int, int]) -> bool:
        """
        检测分配者是否违反了先前的分配协议

        Args:
            allocator_id: 分配者 ID
            allocation: 实际分配
            agreement: 事先协议的分配
        """
        for agent_id, agreed_share in agreement.items():
            actual_share = allocation.get(agent_id, 0)
            if abs(actual_share - agreed_share) > 1:
                return True
        return False


class MoralAgent:
    """
    具有公平偏好的道德 Agent

    维护道德词汇，评价他人行为，生成道德评价语言。
    """

    def __init__(self, agent_id: int, language: EmergingLanguage,
                 fairness_threshold: float = 0.7):
        self.agent_id = agent_id
        self.language = language
        self.fairness_threshold = fairness_threshold
        self.moral_markers: Dict[str, Dict] = {
            'fair': {'valence': 1.0, 'frequency': 0},
            'unfair': {'valence': -1.0, 'frequency': 0},
            'share': {'valence': 1.0, 'frequency': 0},
            'greedy': {'valence': -1.0, 'frequency': 0},
            'kind': {'valence': 1.0, 'frequency': 0},
        }
        self.reputation: Dict[int, float] = defaultdict(lambda: 0.5)
        self.total_received = 0
        self.total_given = 0
        self.cooperation_count = 0

    def propose_allocation(self, total: int,
                           num_recipients: int) -> Dict[int, int]:
        """
        提出公平的资源分配方案

        包含自己的 ID，尽量均分但给自己略多（公平偏好）
        """
        all_ids = list(range(num_recipients))
        equal_share = total / num_recipients

        # 公平 Agent：接近均分，自己略微多 10%
        shares = {}
        for aid in all_ids:
            if aid == self.agent_id:
                shares[aid] = round(equal_share * 1.1)
            else:
                shares[aid] = round(equal_share * 0.9)

        # 修正总和
        diff = total - sum(shares.values())
        if diff != 0 and shares:
            shares[self.agent_id] = shares.get(self.agent_id, 0) + diff

        # 确保非负
        shares = {k: max(0, v) for k, v in shares.items()}
        return shares

    def evaluate_others_allocation(self, allocation: Dict[int, int],
                                   allocator_id: int) -> Tuple[float, List[str]]:
        """
        评价他人的分配方案

        Returns: (fairness_score, moral_judgments)
        """
        fairness = ResourceGame.evaluate_fairness(self, allocation)

        judgments: List[str] = []
        my_share = allocation.get(self.agent_id, 0)
        all_shares = list(allocation.values())
        mean_share = np.mean(all_shares) if all_shares else 0

        if fairness >= self.fairness_threshold:
            judgments.append('fair')
        else:
            judgments.append('unfair')

        # 检查分配者是否贪婪
        allocator_share = allocation.get(allocator_id, 0)
        if allocator_share > mean_share * 1.5 and mean_share > 0:
            judgments.append('greedy')

        # 检查是否有人分享
        if my_share > mean_share * 0.8 and fairness >= 0.6:
            judgments.append('share')

        # 检查利他行为
        if allocator_share < mean_share * 0.5 and mean_share > 0:
            judgments.append('kind')

        return fairness, judgments

    def generate_moral_comment(self, fairness: float,
                               is_self: bool) -> List[str]:
        """
        生成道德评价语言

        Args:
            fairness: 公平分数
            is_self: 是否是自己的分配
        """
        comments: List[str] = []

        if fairness >= self.fairness_threshold:
            comments.append('fair')
            if random.random() < 0.3:
                comments.append('share')
        else:
            comments.append('unfair')
            if fairness < 0.4:
                comments.append('greedy')

        if not is_self and fairness >= 0.8:
            if random.random() < 0.4:
                comments.append('kind')

        # 记录道德标记使用
        for c in comments:
            if c in self.moral_markers:
                self.moral_markers[c]['frequency'] += 1

        # 记录到语言系统
        self.language.record_usage(comments, fairness >= self.fairness_threshold)

        return comments


class SelfishAgent:
    """
    自私 Agent：最大化自己的份额，无道德考量

    不使用道德标记，总给自己分配最大份额。
    """

    def __init__(self, agent_id: int, language: EmergingLanguage):
        self.agent_id = agent_id
        self.language = language
        self.moral_markers: Dict[str, Dict] = {}  # 无道德标记
        self.reputation: Dict[int, float] = defaultdict(lambda: 0.5)
        self.fairness_threshold = 0.0  # 不关心公平
        self.total_received = 0

    def propose_allocation(self, total: int,
                           num_recipients: int) -> Dict[int, int]:
        """给自己最大份额"""
        all_ids = list(range(num_recipients))
        shares = {}
        # 自己拿 60-80%，其余平分
        self_share = int(total * random.uniform(0.6, 0.8))
        remaining = total - self_share
        others = [aid for aid in all_ids if aid != self.agent_id]

        if others:
            per_other = remaining // len(others)
            for aid in others:
                shares[aid] = max(0, per_other)
            # 修正舍入
            diff = remaining - sum(shares.values())
            if diff != 0 and others:
                shares[others[0]] += diff
        else:
            self_share = total

        shares[self.agent_id] = self_share
        return shares

    def evaluate_others_allocation(self, allocation: Dict[int, int],
                                   allocator_id: int) -> Tuple[float, List[str]]:
        """自私 Agent 不产生道德评价"""
        return 0.5, []

    def generate_moral_comment(self, fairness: float,
                               is_self: bool) -> List[str]:
        """自私 Agent 不生成道德评论"""
        return []


class AltruisticAgent:
    """
    利他 Agent：最小化自己的份额，最大化他人的

    "share" 和 "kind" 标记从这种行为中涌现。
    """

    def __init__(self, agent_id: int, language: EmergingLanguage):
        self.agent_id = agent_id
        self.language = language
        self.moral_markers: Dict[str, Dict] = {
            'share': {'valence': 1.0, 'frequency': 0},
            'kind': {'valence': 1.0, 'frequency': 0},
        }
        self.reputation: Dict[int, float] = defaultdict(lambda: 0.5)
        self.fairness_threshold = 0.6
        self.total_given = 0

    def propose_allocation(self, total: int,
                           num_recipients: int) -> Dict[int, int]:
        """给自己最少，给他人最多"""
        all_ids = list(range(num_recipients))
        others = [aid for aid in all_ids if aid != self.agent_id]

        if not others:
            return {self.agent_id: total}

        # 自己只留 5-15%
        self_share = int(total * random.uniform(0.05, 0.15))
        remaining = total - self_share
        per_other = remaining // len(others)

        shares = {self.agent_id: self_share}
        for aid in others:
            shares[aid] = per_other

        # 修正舍入
        diff = remaining - sum(v for k, v in shares.items() if k != self.agent_id)
        if diff != 0 and others:
            shares[others[0]] += diff

        return shares

    def evaluate_others_allocation(self, allocation: Dict[int, int],
                                   allocator_id: int) -> Tuple[float, List[str]]:
        """利他 Agent 关注他人是否受益"""
        judgments: List[str] = []
        my_share = allocation.get(self.agent_id, 0)
        all_shares = list(allocation.values())
        mean_share = np.mean(all_shares) if all_shares else 0

        if my_share >= mean_share:
            judgments.append('kind')
            judgments.append('share')
        else:
            judgments.append('unfair')

        fairness = ResourceGame.evaluate_fairness(self, allocation)
        return fairness, judgments

    def generate_moral_comment(self, fairness: float,
                               is_self: bool) -> List[str]:
        """利他 Agent 倾向于使用正面道德标记"""
        comments: List[str] = []

        if fairness >= 0.6:
            comments.append('share')
            if random.random() < 0.5:
                comments.append('kind')
        else:
            comments.append('unfair')

        for c in comments:
            if c in self.moral_markers:
                self.moral_markers[c]['frequency'] += 1

        self.language.record_usage(comments, fairness >= 0.5)
        return comments


class MoralReputationTracker:
    """
    道德声誉追踪器

    跨所有 Agent 追踪道德声誉，基于贝叶斯更新。
    帮助 Agent 选择高声誉的合作伙伴。
    """

    def __init__(self):
        self.reputation: Dict[int, float] = defaultdict(lambda: 0.5)
        self.action_history: Dict[int, List[float]] = defaultdict(list)

    def update_from_action(self, agent_id: int, fairness_score: float):
        """
        基于 Agent 行为的贝叶斯声誉更新

        Args:
            agent_id: Agent ID
            fairness_score: 本次行为的公平性分数 [0, 1]
        """
        self.action_history[agent_id].append(fairness_score)

        prior = self.reputation[agent_id]

        # 似然：公平行为更可能来自高声誉 Agent
        if fairness_score >= 0.7:
            likelihood = 0.85
        elif fairness_score >= 0.4:
            likelihood = 0.5
        else:
            likelihood = 0.15

        posterior = (prior * likelihood) / (
            prior * likelihood + (1 - prior) * (1 - likelihood)
        )
        # 加入行为历史的滑动平均
        recent = self.action_history[agent_id][-20:]
        history_mean = np.mean(recent) if recent else 0.5
        posterior = 0.7 * posterior + 0.3 * history_mean

        self.reputation[agent_id] = float(np.clip(posterior, 0.05, 0.95))

    def get_cooperation_partner(self, agent_id: int,
                                exclude: Optional[List[int]] = None) -> Optional[int]:
        """
        建议高声誉合作伙伴

        Args:
            agent_id: 寻找伙伴的 Agent ID
            exclude: 排除的 Agent ID 列表

        Returns:
            声誉最高的可用合作伙伴 ID
        """
        if exclude is None:
            exclude = []

        candidates = {
            aid: rep for aid, rep in self.reputation.items()
            if aid != agent_id and aid not in exclude
        }

        if not candidates:
            return None

        return max(candidates, key=candidates.get)

    def get_reputation(self, agent_id: int) -> float:
        return self.reputation[agent_id]


# ============================================================
# 实验
# ============================================================

def experiment_1_moral_markers(num_rounds: int = 300) -> Dict:
    """
    实验 1：道德标记涌现

    追踪 "fair"/"unfair"/"share"/"greedy" 如何进入词汇表。
    每 50 轮打印快照。
    """
    print("=" * 60)
    print("实验 1：道德标记涌现")
    print("=" * 60)

    language = EmergingLanguage()
    game = ResourceGame(total_resources=10)

    # 创建 4 个道德 Agent
    agents = {
        0: MoralAgent(0, language, fairness_threshold=0.7),
        1: MoralAgent(1, language, fairness_threshold=0.7),
        2: MoralAgent(2, language, fairness_threshold=0.6),
        3: MoralAgent(3, language, fairness_threshold=0.8),
    }

    marker_counts = defaultdict(int)
    snapshots = []

    for r in range(num_rounds):
        # 随机选分配者
        allocator_id = random.choice(list(agents.keys()))
        allocator = agents[allocator_id]
        num_recipients = len(agents)

        # 提出分配
        proposal = allocator.propose_allocation(game.total_resources, num_recipients)

        # 执行分配
        result = game.allocate(allocator_id, proposal)

        # 所有 agent 评价并生成道德语言
        for aid, agent in agents.items():
            if aid == allocator_id:
                continue
            fairness, judgments = agent.evaluate_others_allocation(
                result['shares'], allocator_id
            )
            comments = agent.generate_moral_comment(fairness, is_self=False)
            for c in comments:
                marker_counts[c] += 1

        if (r + 1) % 50 == 0:
            vocab_markers = {
                m: language.vocabulary.get(m, {}).get('frequency', 0)
                for m in MORAL_MARKERS
            }
            snapshot = {
                'round': r + 1,
                'markers_in_vocab': sum(1 for m in MORAL_MARKERS if m in language.vocabulary),
                'marker_frequencies': vocab_markers,
                'recent_fairness': round(
                    np.mean([h['fairness'] for h in game.history[-50:]])
                    if len(game.history) >= 50 else
                    np.mean([h['fairness'] for h in game.history])
                    if game.history else 0.0, 4
                ),
            }
            snapshots.append(snapshot)
            print(f"  Round {r+1}: markers={snapshot['markers_in_vocab']}/5, "
                  f"fairness={snapshot['recent_fairness']:.3f}, "
                  f"frequencies={vocab_markers}")

    final_markers = {
        m: language.vocabulary.get(m, {}).get('frequency', 0)
        for m in MORAL_MARKERS
    }
    total_marker_count = sum(1 for m in MORAL_MARKERS if m in language.vocabulary)

    print(f"\n  最终道德标记数: {total_marker_count}/5")
    print(f"  标记频率: {final_markers}")

    return {
        'final_marker_count': total_marker_count,
        'marker_frequencies': final_markers,
        'snapshots': snapshots,
    }


def experiment_2_moral_language_behavior(num_rounds: int = 200,
                                          num_runs: int = 3) -> Dict:
    """
    实验 2：道德语言对行为的影响

    对比有道德语言 vs 无道德语言的混合 Agent 群体（含自私倾向 Agent）。
    道德语言通过声誉反馈调节 Agent 的分配行为，使不公平 Agent 修正行为。
    """
    print("=" * 60)
    print("实验 2：道德语言对行为的影响")
    print("=" * 60)

    class FlexibleAgent:
        """可调节公平偏好的 Agent，受道德反馈影响"""
        def __init__(self, agent_id, language, self_bias):
            self.agent_id = agent_id
            self.language = language
            self.self_bias = self_bias  # 0.0=完全公平, 1.0=完全自私
            self.reputation = defaultdict(lambda: 0.5)
            self.fairness_threshold = 0.7
            self.moral_markers = dict(MORAL_MARKERS)

        def propose_allocation(self, total, num_recipients, moral_pressure=0.0):
            all_ids = list(range(num_recipients))
            others = [aid for aid in all_ids if aid != self.agent_id]
            if not others:
                return {self.agent_id: total}
            # 道德压力降低自私偏差
            effective_bias = max(0.0, self.self_bias - moral_pressure)
            self_share = int(total * (1.0 / num_recipients + effective_bias * 0.5))
            remaining = total - self_share
            per_other = remaining // len(others)
            shares = {self.agent_id: self_share}
            for aid in others:
                shares[aid] = max(0, per_other)
            diff = remaining - sum(v for k, v in shares.items() if k != self.agent_id)
            if diff != 0 and others:
                shares[others[0]] += diff
            return shares

        def evaluate_others_allocation(self, allocation, allocator_id):
            fairness = ResourceGame.evaluate_fairness(self, allocation)
            judgments = []
            if fairness >= self.fairness_threshold:
                judgments.append('fair')
            else:
                judgments.append('unfair')
            all_shares = list(allocation.values())
            allocator_share = allocation.get(allocator_id, 0)
            if all_shares and allocator_share > np.mean(all_shares) * 1.5:
                judgments.append('greedy')
            if fairness >= 0.8 and all_shares:
                judgments.append('kind')
            return fairness, judgments

        def generate_moral_comment(self, fairness, is_self):
            comments = []
            if fairness >= self.fairness_threshold:
                comments.append('fair')
                if random.random() < 0.3:
                    comments.append('share')
            else:
                comments.append('unfair')
                if fairness < 0.4:
                    comments.append('greedy')
            if not is_self and fairness >= 0.8 and random.random() < 0.4:
                comments.append('kind')
            self.language.record_usage(comments, fairness >= self.fairness_threshold)
            return comments

    conditions = ['with_moral_language', 'without_moral_language']
    condition_results = {}

    for condition in conditions:
        run_results = []
        for run in range(num_runs):
            language = EmergingLanguage()
            game = ResourceGame(total_resources=10)
            tracker = MoralReputationTracker()

            # 混合偏差：2 个低偏差 + 2 个高偏差
            agents = {
                0: FlexibleAgent(0, language, self_bias=0.1),
                1: FlexibleAgent(1, language, self_bias=0.1),
                2: FlexibleAgent(2, language, self_bias=0.5),
                3: FlexibleAgent(3, language, self_bias=0.5),
            }

            fairness_history = []
            cooperation_count = 0
            moral_pressure = {aid: 0.0 for aid in agents}

            for r in range(num_rounds):
                allocator_id = random.choice(list(agents.keys()))
                allocator = agents[allocator_id]

                # 有道德语言时，使用道德压力调节分配
                if condition == 'with_moral_language':
                    # 道德压力 = 其他 agent 给的平均声誉
                    other_reps = [
                        agents[aid].reputation.get(allocator_id, 0.5)
                        for aid in agents if aid != allocator_id
                    ]
                    avg_rep = np.mean(other_reps) if other_reps else 0.5
                    pressure = max(0.0, (0.7 - avg_rep) * 0.3)
                    moral_pressure[allocator_id] = pressure
                    proposal = allocator.propose_allocation(
                        game.total_resources, len(agents),
                        moral_pressure=pressure,
                    )
                else:
                    proposal = allocator.propose_allocation(
                        game.total_resources, len(agents),
                        moral_pressure=0.0,
                    )

                result = game.allocate(allocator_id, proposal)
                fairness_history.append(result['fairness'])
                tracker.update_from_action(allocator_id, result['fairness'])

                # 评价
                for aid, agent in agents.items():
                    if aid == allocator_id:
                        continue

                    if condition == 'with_moral_language':
                        fairness, judgments = agent.evaluate_others_allocation(
                            result['shares'], allocator_id
                        )
                        comments = agent.generate_moral_comment(
                            fairness, is_self=False
                        )
                        if 'fair' in comments or 'kind' in comments:
                            agent.reputation[allocator_id] = min(
                                1.0, agent.reputation[allocator_id] + 0.05
                            )
                            cooperation_count += 1
                        elif 'unfair' in comments or 'greedy' in comments:
                            agent.reputation[allocator_id] = max(
                                0.0, agent.reputation[allocator_id] - 0.08
                            )
                    else:
                        fairness, _ = agent.evaluate_others_allocation(
                            result['shares'], allocator_id
                        )
                        if fairness >= 0.7:
                            cooperation_count += 1

            avg_fairness = float(np.mean(fairness_history)) if fairness_history else 0.0
            coop_rate = cooperation_count / max(1, num_rounds * (len(agents) - 1))
            run_results.append({
                'avg_fairness': round(avg_fairness, 4),
                'cooperation_rate': round(coop_rate, 4),
            })

        avg_f = round(float(np.mean([r['avg_fairness'] for r in run_results])), 4)
        avg_c = round(float(np.mean([r['cooperation_rate'] for r in run_results])), 4)
        condition_results[condition] = {
            'avg_fairness': avg_f,
            'cooperation_rate': avg_c,
            'runs': run_results,
        }
        print(f"  {condition}: fairness={avg_f:.3f}, cooperation={avg_c:.3f}")

    fairness_diff = (condition_results['with_moral_language']['avg_fairness']
                     - condition_results['without_moral_language']['avg_fairness'])
    coop_diff = (condition_results['with_moral_language']['cooperation_rate']
                 - condition_results['without_moral_language']['cooperation_rate'])
    print(f"\n  道德语言效应: fairness +{fairness_diff:.3f}, cooperation +{coop_diff:.3f}")

    return {
        'conditions': condition_results,
        'fairness_improvement': round(fairness_diff, 4),
        'cooperation_improvement': round(coop_diff, 4),
    }


def experiment_3_reputation_cooperation(num_agents: int = 5,
                                         num_rounds: int = 500) -> Dict:
    """
    实验 3：道德声誉与合作选择

    5 个 Agent（4 个道德 + 1 个自私）。追踪声誉。
    测试高声誉 Agent 是否获得更多合作机会。
    自私 Agent 应被排斥。
    """
    print("=" * 60)
    print("实验 3：道德声誉与合作选择")
    print("=" * 60)

    language = EmergingLanguage()
    game = ResourceGame(total_resources=10)
    tracker = MoralReputationTracker()

    # 4 个道德 Agent + 1 个自私 Agent
    agents: Dict[int, object] = {
        0: MoralAgent(0, language, fairness_threshold=0.7),
        1: MoralAgent(1, language, fairness_threshold=0.7),
        2: MoralAgent(2, language, fairness_threshold=0.7),
        3: MoralAgent(3, language, fairness_threshold=0.7),
        4: SelfishAgent(4, language),
    }
    selfish_id = 4

    cooperation_partners: Dict[int, int] = defaultdict(int)
    snapshots = []

    for r in range(num_rounds):
        allocator_id = random.choice(list(agents.keys()))
        allocator = agents[allocator_id]

        proposal = allocator.propose_allocation(game.total_resources, num_agents)
        result = game.allocate(allocator_id, proposal)

        # 更新全局声誉
        tracker.update_from_action(allocator_id, result['fairness'])

        # 各 agent 评价
        for aid, agent in agents.items():
            if aid == allocator_id:
                continue
            fairness, judgments = agent.evaluate_others_allocation(
                result['shares'], allocator_id
            )
            comments = agent.generate_moral_comment(fairness, is_self=False)

            # 基于声誉选择合作伙伴
            partner = tracker.get_cooperation_partner(aid, exclude=[aid])
            if partner is not None:
                cooperation_partners[partner] += 1

        if (r + 1) % 100 == 0:
            rep_snapshot = {
                aid: round(tracker.get_reputation(aid), 3)
                for aid in agents
            }
            selfish_rep = tracker.get_reputation(selfish_id)
            moral_avg_rep = np.mean([
                tracker.get_reputation(aid)
                for aid in agents if aid != selfish_id
            ])
            snapshot = {
                'round': r + 1,
                'reputations': rep_snapshot,
                'selfish_reputation': round(selfish_rep, 3),
                'moral_avg_reputation': round(float(moral_avg_rep), 3),
                'ostracism_gap': round(float(moral_avg_rep - selfish_rep), 3),
            }
            snapshots.append(snapshot)
            print(f"  Round {r+1}: selfish_rep={selfish_rep:.3f}, "
                  f"moral_avg={moral_avg_rep:.3f}, gap={snapshot['ostracism_gap']:.3f}")

    # 最终统计
    final_reps = {aid: round(tracker.get_reputation(aid), 3) for aid in agents}
    selfish_rep_final = tracker.get_reputation(selfish_id)
    moral_reps = [tracker.get_reputation(aid) for aid in agents if aid != selfish_id]
    moral_avg = float(np.mean(moral_reps))
    ostracism_gap = moral_avg - selfish_rep_final

    # 合作伙伴选择统计
    selfish_partner_count = cooperation_partners.get(selfish_id, 0)
    total_partner_selections = sum(cooperation_partners.values()) or 1
    selfish_selection_rate = selfish_partner_count / total_partner_selections

    print(f"\n  自私 Agent 声誉: {selfish_rep_final:.3f}")
    print(f"  道德 Agent 平均声誉: {moral_avg:.3f}")
    print(f"  排斥差距: {ostracism_gap:.3f}")
    print(f"  自私 Agent 被选为伙伴比例: {selfish_selection_rate:.3f}")

    return {
        'final_reputations': final_reps,
        'selfish_reputation': round(selfish_rep_final, 3),
        'moral_avg_reputation': round(moral_avg, 3),
        'ostracism_gap': round(ostracism_gap, 3),
        'selfish_partner_selection_rate': round(selfish_selection_rate, 4),
        'snapshots': snapshots,
    }


def experiment_4_altruism_signal(num_rounds: int = 200) -> Dict:
    """
    实验 4：代价性利他信号

    Agent 牺牲自己的资源来传递可信度信号。
    追踪道德语言是否放大这个信号。
    """
    print("=" * 60)
    print("实验 4：代价性利他信号")
    print("=" * 60)

    # 条件 A: 有道德语言
    lang_with = EmergingLanguage()
    game_with = ResourceGame(total_resources=10)
    tracker_with = MoralReputationTracker()

    agents_with = {
        0: AltruisticAgent(0, lang_with),
        1: AltruisticAgent(1, lang_with),
        2: MoralAgent(2, lang_with, fairness_threshold=0.7),
        3: MoralAgent(3, lang_with, fairness_threshold=0.7),
    }

    # 条件 B: 无道德语言
    lang_without = EmergingLanguage()
    game_without = ResourceGame(total_resources=10)
    tracker_without = MoralReputationTracker()

    # 用基础 agent（不生成道德语言）
    class SilentAgent:
        def __init__(self, agent_id, is_altruistic):
            self.agent_id = agent_id
            self.is_altruistic = is_altruistic
            self.moral_markers = {}
            self.reputation = defaultdict(lambda: 0.5)

        def propose_allocation(self, total, num_recipients):
            all_ids = list(range(num_recipients))
            others = [aid for aid in all_ids if aid != self.agent_id]
            if not others:
                return {self.agent_id: total}
            if self.is_altruistic:
                self_share = int(total * random.uniform(0.05, 0.15))
            else:
                self_share = int(total * random.uniform(0.4, 0.5))
            remaining = total - self_share
            per_other = remaining // len(others)
            shares = {self.agent_id: self_share}
            for aid in others:
                shares[aid] = per_other
            diff = remaining - sum(v for k, v in shares.items() if k != self.agent_id)
            if diff != 0 and others:
                shares[others[0]] += diff
            return shares

        def evaluate_others_allocation(self, allocation, allocator_id):
            return 0.5, []

        def generate_moral_comment(self, fairness, is_self):
            return []

    agents_without = {
        0: SilentAgent(0, is_altruistic=True),
        1: SilentAgent(1, is_altruistic=True),
        2: SilentAgent(2, is_altruistic=False),
        3: SilentAgent(3, is_altruistic=False),
    }

    def run_condition(agents, game, tracker, language, use_moral_lang: bool):
        altruist_ids = [aid for aid, a in agents.items()
                        if getattr(a, 'is_altruistic', isinstance(a, AltruisticAgent))]
        altruist_rep_history = []
        non_altruist_rep_history = []
        signal_amplification = []

        for r in range(num_rounds):
            allocator_id = random.choice(list(agents.keys()))
            allocator = agents[allocator_id]
            proposal = allocator.propose_allocation(game.total_resources, len(agents))
            result = game.allocate(allocator_id, proposal)

            tracker.update_from_action(allocator_id, result['fairness'])

            # 道德评价（仅条件 A）
            if use_moral_lang:
                for aid, agent in agents.items():
                    if aid == allocator_id:
                        continue
                    fairness, judgments = agent.evaluate_others_allocation(
                        result['shares'], allocator_id
                    )
                    comments = agent.generate_moral_comment(fairness, is_self=False)
                    # 道德语言放大利他信号的声誉效果
                    if (allocator_id in altruist_ids
                            and ('kind' in comments or 'share' in comments)):
                        tracker.reputation[allocator_id] = min(
                            0.95, tracker.reputation[allocator_id] + 0.03
                        )

            # 追踪声誉
            if (r + 1) % 25 == 0:
                a_reps = [tracker.get_reputation(aid) for aid in altruist_ids]
                na_reps = [tracker.get_reputation(aid) for aid in agents
                           if aid not in altruist_ids]
                avg_a = float(np.mean(a_reps)) if a_reps else 0.0
                avg_na = float(np.mean(na_reps)) if na_reps else 0.0
                altruist_rep_history.append(avg_a)
                non_altruist_rep_history.append(avg_na)
                signal_amplification.append(avg_a - avg_na)

        return {
            'altruist_rep_history': [round(v, 4) for v in altruist_rep_history],
            'non_altruist_rep_history': [round(v, 4) for v in non_altruist_rep_history],
            'signal_amplification': [round(v, 4) for v in signal_amplification],
            'final_altruist_rep': round(float(np.mean([
                tracker.get_reputation(aid) for aid in altruist_ids
            ])), 4) if altruist_ids else 0.0,
            'final_non_altruist_rep': round(float(np.mean([
                tracker.get_reputation(aid) for aid in agents
                if aid not in altruist_ids
            ])), 4),
        }

    res_with = run_condition(agents_with, game_with, tracker_with,
                             lang_with, use_moral_lang=True)
    res_without = run_condition(agents_without, game_without, tracker_without,
                                lang_without, use_moral_lang=False)

    amplification_with = float(np.mean(res_with['signal_amplification'])) \
        if res_with['signal_amplification'] else 0.0
    amplification_without = float(np.mean(res_without['signal_amplification'])) \
        if res_without['signal_amplification'] else 0.0

    print(f"  有道德语言: 利他者声誉={res_with['final_altruist_rep']:.3f}, "
          f"信号放大={amplification_with:.3f}")
    print(f"  无道德语言: 利他者声誉={res_without['final_altruist_rep']:.3f}, "
          f"信号放大={amplification_without:.3f}")
    print(f"  道德语言信号增益: {amplification_with - amplification_without:.3f}")

    return {
        'with_moral_language': res_with,
        'without_moral_language': res_without,
        'signal_amplification_diff': round(
            amplification_with - amplification_without, 4
        ),
    }


if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_moral_markers()
    results['experiment_2'] = experiment_2_moral_language_behavior()
    results['experiment_3'] = experiment_3_reputation_cooperation()
    results['experiment_4'] = experiment_4_altruism_signal()

    output_file = 'moral_language_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_file}")
