"""
Phase 73: 社会规范语言 — 礼貌、禁忌、"应该"/"不行"/"请"涌现

核心思想：
社会规范通过惩罚和奖励机制内化为语言标记。
当违反规范导致惩罚时，Agent 学会使用"应该"/"请"/"不行"等标记。
规范感知型 Agent 在合作场景中优于自私 Agent。

涌现条件：
1. 资源分配需要公平性（违反公平 → 惩罚）
2. 规范标记使用可减少惩罚（功能性压力）
3. 多 Agent 交互中规范通过社会学习传播
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import EmergingLanguage, COLORS, SHAPES


# ============================================================
# 常量
# ============================================================

# 规范标记
NORM_MARKERS = {
    'should',       # 应该做某事
    'please',       # 请求/礼貌
    'not_allowed',  # 禁忌/不允许
    'share',        # 分享标记
    'fair',         # 公平标记
    'sorry',        # 道歉标记
    'thank',        # 感谢标记
}

# 规范类型
NORM_TYPES = {
    'equal_split',    # 平均分配
    'ask_permission', # 请求许可
    'share_resource', # 分享资源
    'no_greed',       # 禁止贪婪
}


# ============================================================
# ResourceScenario
# ============================================================

class ResourceScenario:
    """
    资源分配场景

    Agent 们需要分配一组资源。
    规范约束分配行为：违反规范会受到惩罚。
    """

    def __init__(self, total_resources: int, num_agents: int):
        self.total_resources = total_resources
        self.num_agents = num_agents
        self.norms = {
            'equal_split': True,
            'ask_permission': True,
        }

    def allocate(self, shares: List[int]) -> bool:
        """检查分配是否有效（总和 = 总资源，且所有份额非负）"""
        if len(shares) != self.num_agents:
            return False
        if any(s < 0 for s in shares):
            return False
        return sum(shares) == self.total_resources

    def evaluate_fairness(self, shares: List[int]) -> float:
        """
        评估公平性

        1.0 = 完全平等
        0.0 = 完全不平等（一人拿走全部）

        使用 Gini 系数的互补值。
        """
        if not shares or sum(shares) == 0:
            return 0.0
        n = len(shares)
        if n <= 1:
            return 1.0
        total = sum(shares)
        # Gini 系数
        sorted_s = sorted(shares)
        cum = 0.0
        for i, s in enumerate(sorted_s):
            cum += (n - i) * s
        gini = (n + 1 - 2 * cum / total) / n
        # 公平性 = 1 - Gini
        fairness = max(0.0, min(1.0, 1.0 - gini))
        return fairness


# ============================================================
# NormAwareAgent
# ============================================================

class NormAwareAgent:
    """
    规范感知 Agent

    知道社会规范，能生成规范标记（should/please/not_allowed/share）。
    通过惩罚学习规范遵从。
    """

    def __init__(self, agent_id: int, language: EmergingLanguage,
                 learning_rate: float = 0.1):
        self.agent_id = agent_id
        self.language = language
        self.learning_rate = learning_rate

        # 规范知识：规范名 → 遵从率
        self.norms_learned: Dict[str, float] = {
            name: 0.1 for name in NORM_TYPES
        }

        # 规范标记使用统计
        self.norm_markers: Dict[str, Dict] = {
            'should': {'used': 0, 'successes': 0},
            'please': {'used': 0, 'successes': 0},
            'not_allowed': {'used': 0, 'successes': 0},
            'share': {'used': 0, 'successes': 0},
        }

        # 累计收益
        self.total_payoff = 0.0
        self.total_penalty = 0.0

    def propose_allocation(self, resources: int,
                           num_agents: int) -> Tuple[List[int], List[str]]:
        """
        提出资源分配方案

        返回 (shares, norm_markers_used)
        规范内化程度越高，分配越公平，使用的规范标记越多。
        """
        markers_used = []

        # 基于规范内化程度决定分配策略
        equal_compliance = self.norms_learned.get('equal_split', 0.1)
        share_compliance = self.norms_learned.get('share_resource', 0.1)
        no_greed_compliance = self.norms_learned.get('no_greed', 0.1)

        # 综合规范遵从度
        overall_compliance = np.mean([
            equal_compliance, share_compliance, no_greed_compliance
        ])

        # 分配：在自私和公平之间按遵从度插值
        base = resources // num_agents
        remainder = resources % num_agents

        # 自私分配：自己拿大部分，其他人只拿 1
        min_give = 1
        selfish_share = resources - min_give * (num_agents - 1)
        selfish_shares = [min_give] * num_agents
        selfish_shares[self.agent_id % num_agents] = selfish_share

        # 公平分配
        fair_shares = [base] * num_agents
        # 余数随机分给某人
        for i in range(remainder):
            fair_shares[i % num_agents] += 1

        # 按遵从度混合
        shares = []
        for i in range(num_agents):
            s = (1 - overall_compliance) * selfish_shares[i] + \
                overall_compliance * fair_shares[i]
            shares.append(max(0, int(round(s))))

        # 修正总和
        diff = resources - sum(shares)
        if diff != 0:
            shares[self.agent_id % num_agents] += diff

        # 根据规范内化程度添加标记
        if equal_compliance > 0.3 and random.random() < equal_compliance:
            markers_used.append('should')

        if share_compliance > 0.2 and random.random() < share_compliance:
            markers_used.append('share')

        # 请求许可标记
        ask_compliance = self.norms_learned.get('ask_permission', 0.1)
        if ask_compliance > 0.3 and random.random() < ask_compliance:
            markers_used.append('please')

        # 禁忌标记：如果分配不够公平
        fairness = 1.0
        if shares:
            ideal = resources / num_agents
            if ideal > 0:
                deviations = [abs(s - ideal) / ideal for s in shares]
                fairness = 1.0 - np.mean(deviations)
                fairness = max(0.0, min(1.0, fairness))

        if fairness < 0.5 and no_greed_compliance > 0.3:
            markers_used.append('not_allowed')

        return shares, markers_used

    def evaluate_allocation(self, shares: List[int],
                            num_agents: int) -> Tuple[float, List[str]]:
        """
        评估分配方案的公平性

        返回 (fairness_score, norm_violations)
        """
        if not shares:
            return 0.0, ['no_allocation']

        total = sum(shares)
        if total == 0:
            return 0.0, ['zero_resources']

        ideal = total / num_agents
        violations = []

        # 检查公平性
        for i, s in enumerate(shares):
            if ideal > 0 and (s - ideal) / ideal > 0.5:
                violations.append(f'greedy_agent_{i}')
            if s == 0 and total > 0:
                violations.append('excluded_agent')

        # 计算公平性分数
        if ideal > 0:
            deviations = [abs(s - ideal) / ideal for s in shares]
            fairness = 1.0 - np.mean(deviations)
        else:
            fairness = 1.0

        fairness = max(0.0, min(1.0, fairness))
        return fairness, violations

    def learn_norm(self, violation_type: str, penalty: float):
        """
        从惩罚中学习规范

        更新对应规范的内化程度。
        """
        # 根据违规类型映射到规范
        norm_map = {
            'unequal': 'equal_split',
            'greedy': 'no_greed',
            'no_permission': 'ask_permission',
            'no_share': 'share_resource',
            'greedy_agent': 'no_greed',
            'excluded_agent': 'equal_split',
            'fair_violation': 'equal_split',
        }

        # 找到对应规范
        norm_name = None
        for key, name in norm_map.items():
            if key in violation_type:
                norm_name = name
                break

        if norm_name is None:
            norm_name = 'equal_split'

        # 惩罚越大，学习越快
        lr = self.learning_rate * min(1.0, penalty / 5.0 + 0.2)
        self.norms_learned[norm_name] = min(
            1.0, self.norms_learned.get(norm_name, 0.1) + lr
        )

        # 也会提升其他规范（泛化）
        for name in self.norms_learned:
            if name != norm_name:
                self.norms_learned[name] = min(
                    1.0,
                    self.norms_learned[name] + lr * 0.3
                )

    def receive_payoff(self, amount: float, penalty: float = 0.0):
        """接收收益和惩罚"""
        self.total_payoff += amount
        self.total_penalty += penalty


# ============================================================
# NormEnforcement
# ============================================================

class NormEnforcement:
    """
    规范执行机制

    检测违规并施加惩罚。
    """

    def __init__(self, fairness_threshold: float = 0.6,
                 penalty_rate: float = 2.0):
        self.fairness_threshold = fairness_threshold
        self.penalty_rate = penalty_rate
        self.penalty_history: List[Dict] = []

    def check_compliance(self, agent_id: int,
                         action: Dict) -> Tuple[bool, float]:
        """
        检查 Agent 行为是否合规

        action 包含：shares, fairness, used_markers
        返回 (compliant, penalty_amount)
        """
        shares = action.get('shares', [])
        fairness = action.get('fairness', 1.0)
        markers = action.get('markers', [])

        compliant = True
        penalty = 0.0

        # 检查公平性
        if fairness < self.fairness_threshold:
            compliant = False
            unfairness = self.fairness_threshold - fairness
            penalty += self.penalty_rate * unfairness

        # 使用规范标记可以减轻惩罚
        marker_discount = len(markers) * 0.15
        penalty = max(0, penalty - marker_discount)

        # 记录
        self.penalty_history.append({
            'agent_id': agent_id,
            'compliant': compliant,
            'fairness': round(fairness, 4),
            'penalty': round(penalty, 4),
            'markers': markers,
        })

        return compliant, penalty


# ============================================================
# BaselineNormAgent
# ============================================================

class BaselineNormAgent:
    """
    基线 Agent：无规范感知，纯自私策略

    最大化个人份额，不使用规范标记。
    """

    def __init__(self, agent_id: int, language: EmergingLanguage):
        self.agent_id = agent_id
        self.language = language
        self.total_payoff = 0.0
        self.total_penalty = 0.0

    def propose_allocation(self, resources: int,
                           num_agents: int) -> Tuple[List[int], List[str]]:
        """纯自私分配：自己拿最多"""
        min_share = max(1, resources // (num_agents * 3))
        remaining = resources - min_share * (num_agents - 1)

        shares = [min_share] * num_agents
        shares[self.agent_id % num_agents] = remaining

        # 不使用规范标记
        return shares, []

    def receive_payoff(self, amount: float, penalty: float = 0.0):
        self.total_payoff += amount
        self.total_penalty += penalty


# ============================================================
# NormLearningGame
# ============================================================

class NormLearningGame:
    """
    规范学习游戏

    多轮资源分配 + 规范执行。
    Agent 通过惩罚学习规范，规范标记涌现。
    """

    def __init__(self, agents, scenario: ResourceScenario,
                 enforcement: NormEnforcement):
        self.agents = agents
        self.scenario = scenario
        self.enforcement = enforcement

        # 统计
        self.rounds_played = 0
        self.total_fairness = 0.0
        self.compliance_rate = 0.0
        self.compliant_count = 0

        # 标记使用历史
        self.marker_history: List[Dict] = []

    def play_round(self) -> Dict:
        """进行一轮资源分配"""
        resources = self.scenario.total_resources
        num_agents = self.scenario.num_agents

        # 每个 Agent 提出分配方案
        proposals = []
        for agent in self.agents:
            if isinstance(agent, NormAwareAgent):
                shares, markers = agent.propose_allocation(resources, num_agents)
            else:
                shares, markers = agent.propose_allocation(resources, num_agents)
            proposals.append((shares, markers))

        # 选择一个方案（轮流或随机）
        proposer_idx = self.rounds_played % len(self.agents)
        chosen_shares, chosen_markers = proposals[proposer_idx]

        # 验证分配有效性
        if not self.scenario.allocate(chosen_shares):
            chosen_shares = [resources // num_agents] * num_agents
            remainder = resources % num_agents
            chosen_shares[0] += remainder

        # 评估公平性
        fairness = self.scenario.evaluate_fairness(chosen_shares)

        # 规范检查
        action = {
            'shares': chosen_shares,
            'fairness': fairness,
            'markers': chosen_markers,
        }
        compliant, penalty = self.enforcement.check_compliance(
            proposer_idx, action
        )

        # 分配收益和惩罚
        for i, agent in enumerate(self.agents):
            share = chosen_shares[i] if i < len(chosen_shares) else 0
            agent_penalty = penalty if i == proposer_idx else 0.0
            agent.receive_payoff(float(share), agent_penalty)

            # 规范感知 Agent 从惩罚中学习
            if isinstance(agent, NormAwareAgent) and not compliant:
                agent.learn_norm('fair_violation', penalty)
                # 更新标记统计
                for m in chosen_markers:
                    if m in agent.norm_markers:
                        agent.norm_markers[m]['used'] += 1
                        if compliant:
                            agent.norm_markers[m]['successes'] += 1
            elif isinstance(agent, NormAwareAgent):
                for m in chosen_markers:
                    if m in agent.norm_markers:
                        agent.norm_markers[m]['used'] += 1
                        agent.norm_markers[m]['successes'] += 1

        # 记录语言使用
        all_symbols = chosen_markers + [
            str(s) for s in chosen_shares if str(s) in COLORS or str(s) in SHAPES
        ]
        if all_symbols:
            success = compliant and fairness > 0.5
            for agent in self.agents:
                agent.language.record_usage(all_symbols, success)

        # 更新统计
        self.rounds_played += 1
        self.total_fairness += fairness
        if compliant:
            self.compliant_count += 1

        # 标记历史
        marker_counts = defaultdict(int)
        for m in chosen_markers:
            marker_counts[m] += 1

        self.marker_history.append({
            'round': self.rounds_played,
            'fairness': round(fairness, 4),
            'compliant': compliant,
            'markers': dict(marker_counts),
        })

        return {
            'shares': chosen_shares,
            'fairness': fairness,
            'compliant': compliant,
            'penalty': penalty,
            'markers': chosen_markers,
        }


# ============================================================
# 实验 1：规范标记涌现
# ============================================================

def experiment_1_norm_markers(num_rounds: int = 300) -> Dict:
    """
    追踪 "should"/"please"/"not_allowed" 进入词汇表

    300 轮资源分配游戏，每 50 轮打印快照。
    预期：规范标记逐渐出现在词汇中。
    """
    print("=" * 60)
    print("实验 1：规范标记涌现")
    print("=" * 60)

    language = EmergingLanguage()
    scenario = ResourceScenario(total_resources=20, num_agents=4)
    enforcement = NormEnforcement(fairness_threshold=0.6, penalty_rate=2.0)

    agents = [
        NormAwareAgent(i, language, learning_rate=0.1)
        for i in range(4)
    ]

    game = NormLearningGame(agents, scenario, enforcement)
    snapshots = []

    for r in range(num_rounds):
        game.play_round()

        if (r + 1) % 50 == 0:
            # 统计词汇中的规范标记
            markers_in_vocab = sum(
                1 for m in NORM_MARKERS if m in language.vocabulary
            )
            vocab_size = len(language.vocabulary)
            avg_fairness = game.total_fairness / max(1, game.rounds_played)
            compliance = game.compliant_count / max(1, game.rounds_played)

            snapshot = {
                'round': r + 1,
                'markers_in_vocab': markers_in_vocab,
                'total_markers': len(NORM_MARKERS),
                'vocab_size': vocab_size,
                'avg_fairness': round(avg_fairness, 4),
                'compliance_rate': round(compliance, 4),
            }
            snapshots.append(snapshot)

            print(f"  Round {r+1}: 标记={markers_in_vocab}/{len(NORM_MARKERS)}, "
                  f"词汇={vocab_size}, 公平={avg_fairness:.3f}, "
                  f"遵从={compliance:.3f}")

    # 最终规范内化程度
    final_norms = {}
    for agent in agents:
        for name, rate in agent.norms_learned.items():
            if name not in final_norms:
                final_norms[name] = []
            final_norms[name].append(rate)

    avg_norms = {
        name: round(float(np.mean(rates)), 4)
        for name, rates in final_norms.items()
    }

    print(f"\n  最终规范内化: {avg_norms}")

    return {
        'snapshots': snapshots,
        'final_norms': avg_norms,
        'final_markers_in_vocab': sum(
            1 for m in NORM_MARKERS if m in language.vocabulary
        ),
    }


# ============================================================
# 实验 2：规范内化
# ============================================================

def experiment_2_norm_internalization(num_rounds: int = 200,
                                       num_runs: int = 3) -> Dict:
    """
    对比规范感知 Agent vs 自私 Agent

    预期：
    - 规范感知 Agent 公平性分数更高
    - 规范感知 Agent 遵从率更高
    - 规范感知 Agent 总收益（扣除惩罚后）可能更高
    """
    print("=" * 60)
    print("实验 2：规范内化对比")
    print("=" * 60)

    norm_results = []
    selfish_results = []

    for run in range(num_runs):
        # ---- 规范感知组 ----
        lang_norm = EmergingLanguage()
        scenario = ResourceScenario(total_resources=20, num_agents=4)
        enforcement = NormEnforcement(fairness_threshold=0.6, penalty_rate=2.0)

        norm_agents = [
            NormAwareAgent(i, lang_norm, learning_rate=0.1)
            for i in range(4)
        ]
        norm_game = NormLearningGame(norm_agents, scenario, enforcement)

        norm_fairness = []
        norm_compliance = []

        for r in range(num_rounds):
            result = norm_game.play_round()
            if (r + 1) % 50 == 0:
                avg_f = norm_game.total_fairness / max(1, norm_game.rounds_played)
                comp = norm_game.compliant_count / max(1, norm_game.rounds_played)
                norm_fairness.append(avg_f)
                norm_compliance.append(comp)

        norm_payoffs = [a.total_payoff - a.total_penalty for a in norm_agents]

        # ---- 自私组（基线） ----
        lang_self = EmergingLanguage()
        scenario_s = ResourceScenario(total_resources=20, num_agents=4)
        enforcement_s = NormEnforcement(fairness_threshold=0.6, penalty_rate=2.0)

        selfish_agents = [
            BaselineNormAgent(i, lang_self)
            for i in range(4)
        ]
        selfish_game = NormLearningGame(selfish_agents, scenario_s, enforcement_s)

        self_fairness = []
        self_compliance = []

        for r in range(num_rounds):
            result = selfish_game.play_round()
            if (r + 1) % 50 == 0:
                avg_f = selfish_game.total_fairness / max(1, selfish_game.rounds_played)
                comp = selfish_game.compliant_count / max(1, selfish_game.rounds_played)
                self_fairness.append(avg_f)
                self_compliance.append(comp)

        self_payoffs = [a.total_payoff - a.total_penalty for a in selfish_agents]

        norm_results.append({
            'fairness': norm_fairness,
            'compliance': norm_compliance,
            'avg_payoff': float(np.mean(norm_payoffs)),
            'total_penalty': float(sum(a.total_penalty for a in norm_agents)),
        })
        selfish_results.append({
            'fairness': self_fairness,
            'compliance': self_compliance,
            'avg_payoff': float(np.mean(self_payoffs)),
            'total_penalty': float(sum(a.total_penalty for a in selfish_agents)),
        })

        print(f"  Run {run+1}: "
              f"规范组公平={np.mean(norm_fairness):.3f} vs "
              f"自私组公平={np.mean(self_fairness):.3f}, "
              f"规范组惩罚={norm_results[-1]['total_penalty']:.1f} vs "
              f"自私组惩罚={selfish_results[-1]['total_penalty']:.1f}")

    # 汇总
    avg_norm_fairness = float(np.mean([r['fairness'][-1] for r in norm_results]))
    avg_self_fairness = float(np.mean([r['fairness'][-1] for r in selfish_results]))
    avg_norm_payoff = float(np.mean([r['avg_payoff'] for r in norm_results]))
    avg_self_payoff = float(np.mean([r['avg_payoff'] for r in selfish_results]))
    avg_norm_compliance = float(np.mean([r['compliance'][-1] for r in norm_results]))
    avg_self_compliance = float(np.mean([r['compliance'][-1] for r in selfish_results]))

    print(f"\n  平均公平性: 规范={avg_norm_fairness:.3f}, 自私={avg_self_fairness:.3f}")
    print(f"  平均遵从率: 规范={avg_norm_compliance:.3f}, 自私={avg_self_compliance:.3f}")
    print(f"  平均净收益: 规范={avg_norm_payoff:.1f}, 自私={avg_self_payoff:.1f}")

    return {
        'norm_aware': {
            'avg_fairness': round(avg_norm_fairness, 4),
            'avg_compliance': round(avg_norm_compliance, 4),
            'avg_payoff': round(avg_norm_payoff, 4),
        },
        'baseline': {
            'avg_fairness': round(avg_self_fairness, 4),
            'avg_compliance': round(avg_self_compliance, 4),
            'avg_payoff': round(avg_self_payoff, 4),
        },
        'fairness_advantage': round(avg_norm_fairness - avg_self_fairness, 4),
        'compliance_advantage': round(avg_norm_compliance - avg_self_compliance, 4),
    }


# ============================================================
# 实验 3：规范传播
# ============================================================

def experiment_3_norm_transmission(num_agents: int = 5,
                                    num_rounds: int = 300) -> Dict:
    """
    规范传播：1 个规范感知 Agent + 4 个 naive Agent

    通过社会学习，规范从"老师"传播到"学生"。
    追踪：每个 Agent 的规范内化程度变化。
    """
    print("=" * 60)
    print("实验 3：规范传播")
    print("=" * 60)

    language = EmergingLanguage()
    scenario = ResourceScenario(total_resources=25, num_agents=num_agents)
    enforcement = NormEnforcement(fairness_threshold=0.55, penalty_rate=2.5)

    agents = []

    # 1 个经验丰富的 Agent（高初始规范内化）
    teacher = NormAwareAgent(0, language, learning_rate=0.15)
    for norm_name in teacher.norms_learned:
        teacher.norms_learned[norm_name] = 0.7
    agents.append(teacher)

    # 4 个 naive Agent（低初始规范内化）
    for i in range(1, num_agents):
        student = NormAwareAgent(i, language, learning_rate=0.12)
        for norm_name in student.norms_learned:
            student.norms_learned[norm_name] = 0.05
        agents.append(student)

    game = NormLearningGame(agents, scenario, enforcement)

    # 追踪每个 Agent 的规范内化
    transmission_snapshots = []

    for r in range(num_rounds):
        game.play_round()

        # 社会学习：naive Agent 观察并从经验丰富 Agent 学习
        if (r + 1) % 10 == 0:
            for i in range(1, num_agents):
                student = agents[i]
                # 向老师看齐（部分学习）
                for norm_name in student.norms_learned:
                    teacher_rate = teacher.norms_learned[norm_name]
                    gap = teacher_rate - student.norms_learned[norm_name]
                    if gap > 0:
                        transfer = gap * 0.03 * (1.0 / num_agents)
                        student.norms_learned[norm_name] = min(
                            1.0,
                            student.norms_learned[norm_name] + transfer
                        )

        if (r + 1) % 60 == 0:
            norms_by_agent = {}
            for i, agent in enumerate(agents):
                avg_norm = float(np.mean(list(agent.norms_learned.values())))
                norms_by_agent[f'agent_{i}'] = round(avg_norm, 4)

            transmission_snapshots.append({
                'round': r + 1,
                'norms_by_agent': norms_by_agent,
                'avg_fairness': round(
                    game.total_fairness / max(1, game.rounds_played), 4
                ),
            })

            role = "老师" if 0 in norms_by_agent else ""
            vals = list(norms_by_agent.values())
            print(f"  Round {r+1}: 老师={vals[0]:.3f}, "
                  f"学生均值={np.mean(vals[1:]):.3f}, "
                  f"公平={transmission_snapshots[-1]['avg_fairness']:.3f}")

    # 最终分析
    final_teacher = float(np.mean(list(teacher.norms_learned.values())))
    final_students = [
        float(np.mean(list(agents[i].norms_learned.values())))
        for i in range(1, num_agents)
    ]

    print(f"\n  老师最终: {final_teacher:.3f}")
    print(f"  学生最终: 均值={np.mean(final_students):.3f}, "
          f"范围=[{min(final_students):.3f}, {max(final_students):.3f}]")

    return {
        'transmission_snapshots': transmission_snapshots,
        'final_teacher_norm': round(final_teacher, 4),
        'final_student_norms': [round(v, 4) for v in final_students],
        'avg_student_norm': round(float(np.mean(final_students)), 4),
        'norm_gap': round(final_teacher - np.mean(final_students), 4),
    }


# ============================================================
# 实验 4：规范冲突
# ============================================================

def experiment_4_norm_conflict(num_rounds: int = 200) -> Dict:
    """
    规范冲突：两组 Agent 有不同规范相遇

    A 组：重视公平分配（equal_split 高）
    B 组：重视请求许可（ask_permission 高）

    追踪：冲突解决和规范趋同。
    """
    print("=" * 60)
    print("实验 4：规范冲突与趋同")
    print("=" * 60)

    language = EmergingLanguage()
    scenario = ResourceScenario(total_resources=20, num_agents=4)
    enforcement = NormEnforcement(fairness_threshold=0.5, penalty_rate=1.5)

    # A 组：公平导向（2 个 Agent）
    group_a = []
    for i in range(2):
        agent = NormAwareAgent(i, language, learning_rate=0.12)
        agent.norms_learned['equal_split'] = 0.8
        agent.norms_learned['share_resource'] = 0.7
        agent.norms_learned['ask_permission'] = 0.2
        agent.norms_learned['no_greed'] = 0.6
        group_a.append(agent)

    # B 组：许可导向（2 个 Agent）
    group_b = []
    for i in range(2):
        agent = NormAwareAgent(i + 2, language, learning_rate=0.12)
        agent.norms_learned['equal_split'] = 0.2
        agent.norms_learned['share_resource'] = 0.3
        agent.norms_learned['ask_permission'] = 0.8
        agent.norms_learned['no_greed'] = 0.4
        group_b.append(agent)

    all_agents = group_a + group_b
    game = NormLearningGame(all_agents, scenario, enforcement)

    snapshots = []

    for r in range(num_rounds):
        game.play_round()

        # 跨组社会学习
        if (r + 1) % 20 == 0:
            for a_agent in group_a:
                for b_agent in group_b:
                    for norm_name in a_agent.norms_learned:
                        b_rate = b_agent.norms_learned[norm_name]
                        a_rate = a_agent.norms_learned[norm_name]
                        # 双向影响
                        diff = a_rate - b_rate
                        transfer = diff * 0.02
                        b_agent.norms_learned[norm_name] = min(
                            1.0, max(0.0, b_rate + transfer)
                        )
                        a_agent.norms_learned[norm_name] = min(
                            1.0, max(0.0, a_rate - transfer * 0.5)
                        )

        if (r + 1) % 40 == 0:
            # 计算各组平均规范
            a_norms = {}
            b_norms = {}
            for norm_name in NORM_TYPES:
                a_vals = [a.norms_learned[norm_name] for a in group_a]
                b_vals = [a.norms_learned[norm_name] for a in group_b]
                a_norms[norm_name] = round(float(np.mean(a_vals)), 4)
                b_norms[norm_name] = round(float(np.mean(b_vals)), 4)

            # 计算规范距离
            distances = [
                abs(a_norms[n] - b_norms[n])
                for n in NORM_TYPES
            ]
            avg_distance = float(np.mean(distances))

            snapshot = {
                'round': r + 1,
                'group_a_norms': a_norms,
                'group_b_norms': b_norms,
                'norm_distance': round(avg_distance, 4),
                'fairness': round(
                    game.total_fairness / max(1, game.rounds_played), 4
                ),
            }
            snapshots.append(snapshot)

            print(f"  Round {r+1}: 规范距离={avg_distance:.3f}, "
                  f"A公平={a_norms.get('equal_split', 0):.3f}, "
                  f"B公平={b_norms.get('equal_split', 0):.3f}, "
                  f"A许可={a_norms.get('ask_permission', 0):.3f}, "
                  f"B许可={b_norms.get('ask_permission', 0):.3f}")

    # 最终趋同分析
    final_a = {
        n: float(np.mean([a.norms_learned[n] for a in group_a]))
        for n in NORM_TYPES
    }
    final_b = {
        n: float(np.mean([a.norms_learned[n] for a in group_b]))
        for n in NORM_TYPES
    }

    final_distance = float(np.mean([
        abs(final_a[n] - final_b[n]) for n in NORM_TYPES
    ]))

    initial_distance = 0.6  # 估计初始距离

    print(f"\n  初始规范距离: ~{initial_distance:.3f}")
    print(f"  最终规范距离: {final_distance:.3f}")
    print(f"  趋同率: {1 - final_distance / initial_distance:.3f}")

    return {
        'snapshots': snapshots,
        'initial_distance': round(initial_distance, 4),
        'final_distance': round(final_distance, 4),
        'convergence_rate': round(
            max(0, 1 - final_distance / initial_distance), 4
        ),
        'final_group_a': {k: round(v, 4) for k, v in final_a.items()},
        'final_group_b': {k: round(v, 4) for k, v in final_b.items()},
    }


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_norm_markers()
    results['experiment_2'] = experiment_2_norm_internalization()
    results['experiment_3'] = experiment_3_norm_transmission()
    results['experiment_4'] = experiment_4_norm_conflict()

    with open('social_norms_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 social_norms_results.json")
