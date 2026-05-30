"""
Phase 82: 所有权与财产概念涌现

核心思想：
前面的 Phase 证明了道德语言（Phase 76）、社会规范（Phase 77）。
但道德和规范只约束了"应该如何分配"——更基础的财产概念（我的/你的/共享）
是否会在资源竞争压力下自发涌现为语言标记？

本阶段测试：
- "mine"/"yours"/"ours" 等所有权标记从资源争夺中涌现
- "share"/"give"/"take" 转移标记提高资源分配效率
- 所有权语言降低冲突率、促进共享
- 适度的资源稀缺性最大化共享标记的使用频率

涌现条件：
1. 资源有限且有价值，Agent 需要声明所有权以避免冲突
2. 竞争性声索需要通过语言协调解决
3. 所有权转移（give/take）提高整体资源分配效率
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import EmergingLanguage, COLORS, SHAPES, SIZES, MATERIALS


# 所有权标记常量
OWNERSHIP_MARKERS = {
    'mine', 'yours', 'ours', 'theirs',
    'share', 'take', 'give', 'keep',
    'want', 'not_yours',
}

# 声索强度权重
CLAIM_STRENGTH = {
    'mine': 1.0,
    'want': 0.6,
    'keep': 0.8,
    'not_yours': 0.5,
}


class ResourceItem:
    """带所有权的资源物品"""

    def __init__(self, item_id: str, properties: Dict,
                 owner: Optional[int] = None, value: float = 1.0):
        self.item_id = item_id
        self.properties = properties  # {color, shape, size, material, ...}
        self.owner = owner            # agent_id 或 None（无主）
        self.value = value

    def describe(self) -> str:
        """生成资源描述字符串"""
        parts = [f"{v}" for v in self.properties.values()]
        return '-'.join(parts)


class OwnershipAgent:
    """
    具有所有权概念的 Agent

    维护所有权词汇，声明和认可资源所有权，
    并在冲突中使用语言标记协调解决。
    """

    def __init__(self, agent_id: int, language: EmergingLanguage):
        self.agent_id = agent_id
        self.language = language
        self.owned_items: List[str] = []
        self.claimed_items: List[str] = []
        self.claim_history: Dict[str, List[str]] = defaultdict(list)
        self.marker_effectiveness: Dict[str, Dict] = {
            m: {'uses': 0, 'successes': 0} for m in OWNERSHIP_MARKERS
        }

    def claim_resource(self, resource: ResourceItem,
                       language: EmergingLanguage) -> str:
        """
        声明对资源的所有权，生成所有权话语

        Args:
            resource: 目标资源
            language: 共享语言系统

        Returns:
            所有权标记（如 'mine', 'want'）
        """
        # 基于资源价值和已有物品决定声索强度
        if resource.value > 0.7 or len(self.owned_items) < 2:
            marker = 'mine'
        elif resource.value > 0.4:
            marker = random.choice(['mine', 'want', 'keep'])
        else:
            marker = random.choice(['want', 'keep'])

        self.claimed_items.append(resource.item_id)
        self.claim_history[resource.item_id].append(marker)

        # 记录语言使用
        description = [marker] + [
            v for v in resource.properties.values()
        ]
        language.record_usage([marker], True)

        return marker

    def recognize_ownership(self, utterance: str) -> Optional[str]:
        """
        解析话语中的所有权标记

        Returns:
            识别到的标记或 None
        """
        for marker in OWNERSHIP_MARKERS:
            if marker in utterance.lower():
                return marker
        return None

    def respond_to_claim(self, claim: str,
                         own_claims: List[str]) -> str:
        """
        对他人的声索做出回应

        Args:
            claim: 对方的声索标记
            own_claims: 自己已有的声索

        Returns:
            回应标记（'ok', 'not_yours', 'mine'）
        """
        if claim in ('mine', 'keep') and len(own_claims) >= 2:
            # 自己已有足够资源，让步
            return 'ok'
        elif claim in ('want',) and random.random() < 0.6:
            # 弱声索，有概率竞争
            return random.choice(['ok', 'mine'])
        else:
            # 竞争
            return random.choice(['not_yours', 'mine'])

    def share_resource(self, resource: ResourceItem) -> str:
        """
        生成共享话语

        Returns:
            共享标记（'share', 'ours'）
        """
        marker = random.choice(['share', 'ours'])
        self.language.record_usage([marker], True)
        return marker

    def update_from_outcome(self, utterance: str, conflict_resolved: bool):
        """
        根据结果更新标记有效性追踪

        Args:
            utterance: 使用过的标记
            conflict_resolved: 冲突是否被成功解决
        """
        marker = self.recognize_ownership(utterance)
        if marker and marker in self.marker_effectiveness:
            self.marker_effectiveness[marker]['uses'] += 1
            if conflict_resolved:
                self.marker_effectiveness[marker]['successes'] += 1


class ConflictResolver:
    """
    冲突解决器

    解决竞争性声索：先到先得，或声索强度相同时共享。
    追踪冲突频率和标记使用模式。
    """

    def __init__(self):
        self.conflicts: List[Dict] = []
        self.resolutions: Dict[str, int] = defaultdict(int)
        self.marker_in_conflict: Dict[str, int] = defaultdict(int)

    def resolve(self, claims: Dict[int, Tuple[str, float]],
                resource: ResourceItem) -> Tuple[Optional[int], str]:
        """
        解决多个 Agent 对同一资源的竞争性声索

        Args:
            claims: {agent_id: (marker, claim_strength)}
            resource: 争议资源

        Returns:
            (winner_id 或 None, 解决方式描述)
        """
        if len(claims) <= 1:
            # 无冲突
            if claims:
                winner = list(claims.keys())[0]
                return winner, 'first_claim'
            return None, 'no_claims'

        # 记录冲突
        conflict_markers = [m for m, _ in claims.values()]
        for m in conflict_markers:
            self.marker_in_conflict[m] += 1

        # 按声索强度排序
        sorted_claims = sorted(
            claims.items(),
            key=lambda x: x[1][1],
            reverse=True,
        )
        top_agent, (top_marker, top_strength) = sorted_claims[0]
        second_agent, (second_marker, second_strength) = sorted_claims[1]

        if top_strength > second_strength + 0.2:
            # 明确的优势声索
            resolution = 'stronger_claim'
            winner = top_agent
        elif abs(top_strength - second_strength) <= 0.2:
            # 声索强度接近 → 概率性分配或共享
            if random.random() < 0.4:
                resolution = 'sharing'
                winner = None  # 共享，无人独占
            else:
                resolution = 'contested'
                winner = random.choice([top_agent, second_agent])
        else:
            resolution = 'weaker_yields'
            winner = top_agent

        self.conflicts.append({
            'resource_id': resource.item_id,
            'claimants': list(claims.keys()),
            'markers': conflict_markers,
            'resolution': resolution,
            'winner': winner,
        })
        self.resolutions[resolution] += 1

        return winner, resolution


class ResourceScenario:
    """
    资源场景生成器

    创建包含有主和无主资源的场景。
    """

    def __init__(self):
        self.resources: List[ResourceItem] = []
        self.agents: List[OwnershipAgent] = []

    @staticmethod
    def generate_scenario(num_resources: int = 5,
                          num_agents: int = 3) -> 'ResourceScenario':
        """
        生成资源场景

        部分资源预设所有权，部分无主可供争夺。
        """
        scenario = ResourceScenario()
        colors = list(COLORS)[:4]
        shapes = list(SHAPES)[:4]
        sizes = list(SIZES)[:3]
        materials = list(MATERIALS)[:3]

        for i in range(num_resources):
            props = {
                'color': random.choice(colors),
                'shape': random.choice(shapes),
                'size': random.choice(sizes),
            }
            value = round(random.uniform(0.2, 1.0), 2)
            # ~40% 的资源预设所有权
            owner = random.choice(
                list(range(num_agents)) + [None, None]
            )
            item = ResourceItem(
                item_id=f"res_{i}",
                properties=props,
                owner=owner,
                value=value,
            )
            scenario.resources.append(item)

        return scenario

    def distribute_randomly(self):
        """随机分配初始所有权"""
        for res in self.resources:
            if res.owner is None:
                res.owner = random.randint(0, len(self.agents) - 1)


class OwnershipGame:
    """
    所有权游戏

    生成场景 → Agent 声索资源 → 解决冲突 → 追踪结果。
    度量：冲突率、标记涌现、共享行为。
    """

    def __init__(self, num_agents: int = 3, num_resources: int = 5):
        self.num_agents = num_agents
        self.num_resources = num_resources
        self.language = EmergingLanguage()
        self.resolver = ConflictResolver()
        self.agents: Dict[int, OwnershipAgent] = {
            i: OwnershipAgent(i, self.language)
            for i in range(num_agents)
        }
        self.round_history: List[Dict] = []

    def play_round(self) -> Dict:
        """
        执行一轮所有权游戏

        Returns:
            本轮结果统计
        """
        scenario = ResourceScenario.generate_scenario(
            self.num_resources, self.num_agents
        )

        # Agent 对无主资源发起声索
        unclaimed = [r for r in scenario.resources if r.owner is None]
        claims_by_resource: Dict[str, Dict[int, Tuple[str, float]]] = defaultdict(dict)

        for res in unclaimed:
            # 每个 Agent 决定是否声索
            for aid, agent in self.agents.items():
                if random.random() < 0.6 + res.value * 0.3:
                    marker = agent.claim_resource(res, self.language)
                    strength = CLAIM_STRENGTH.get(marker, 0.5)
                    claims_by_resource[res.item_id][aid] = (marker, strength)

        # 解决冲突并分配
        acquisitions = 0
        conflicts = 0
        sharings = 0
        outcomes: List[Dict] = []

        for res in scenario.resources:
            if res.owner is not None:
                # 已有所有者，直接归属
                self.agents[res.owner].owned_items.append(res.item_id)
                acquisitions += 1
                continue

            claims = claims_by_resource.get(res.item_id, {})
            if not claims:
                continue

            winner, resolution = self.resolver.resolve(claims, res)

            if winner is not None:
                res.owner = winner
                self.agents[winner].owned_items.append(res.item_id)
                acquisitions += 1
            elif resolution == 'sharing':
                # 共享：多个 Agent 共同使用
                for aid in claims:
                    self.agents[aid].owned_items.append(res.item_id)
                sharings += 1
                # 记录共享标记
                self.language.record_usage(['share', 'ours'], True)
            else:
                acquisitions += 1

            if len(claims) > 1:
                conflicts += 1

            # 更新 Agent 对结果的追踪
            for aid, (marker, _) in claims.items():
                resolved = (winner == aid or resolution == 'sharing')
                self.agents[aid].update_from_outcome(marker, resolved)

            outcomes.append({
                'resource': res.item_id,
                'claims': {str(aid): marker for aid, (marker, _) in claims.items()},
                'winner': winner,
                'resolution': resolution,
            })

        total_possible_claims = len(unclaimed)
        conflict_rate = conflicts / max(1, total_possible_claims)
        result = {
            'conflicts': conflicts,
            'conflict_rate': round(conflict_rate, 4),
            'acquisitions': acquisitions,
            'sharings': sharings,
            'outcomes': outcomes,
        }
        self.round_history.append(result)
        return result


class BaselineNoOwnershipGame:
    """
    基线游戏：无所有权语言

    Agent 随机争夺资源，无所有权标记协调。
    预期更高的冲突率、无共享行为。
    """

    def __init__(self, num_agents: int = 3, num_resources: int = 5):
        self.num_agents = num_agents
        self.num_resources = num_resources
        self.round_history: List[Dict] = []

    def play_round(self) -> Dict:
        """执行一轮基线游戏（无语言协调）"""
        scenario = ResourceScenario.generate_scenario(
            self.num_resources, self.num_agents
        )

        unclaimed = [r for r in scenario.resources if r.owner is None]
        conflicts = 0
        acquisitions = 0

        for res in unclaimed:
            # 随机争夺：多个 Agent 可能同时抢
            claimants = [
                i for i in range(self.num_agents)
                if random.random() < 0.5
            ]
            if len(claimants) > 1:
                conflicts += 1
                # 随机选择获胜者
                winner = random.choice(claimants)
                res.owner = winner
            elif claimants:
                res.owner = claimants[0]
            acquisitions += 1

        conflict_rate = conflicts / max(1, len(unclaimed))
        result = {
            'conflicts': conflicts,
            'conflict_rate': round(conflict_rate, 4),
            'acquisitions': acquisitions,
            'sharings': 0,  # 基线无共享
        }
        self.round_history.append(result)
        return result


# ============================================================
# 实验
# ============================================================

def experiment_1_ownership_markers(num_rounds: int = 300) -> Dict:
    """
    实验 1：所有权标记涌现

    3 个 Agent，每轮 5 个资源，300 轮。
    追踪所有权标记从竞争中涌现的过程。
    每 50 轮打印快照。
    """
    print("=" * 60)
    print("实验 1：所有权标记涌现")
    print("=" * 60)

    game = OwnershipGame(num_agents=3, num_resources=5)
    snapshots = []

    for r in range(num_rounds):
        result = game.play_round()

        if (r + 1) % 50 == 0:
            vocab_markers = {
                m: game.language.vocabulary.get(m, {}).get('frequency', 0)
                for m in OWNERSHIP_MARKERS
            }
            emerged = sum(
                1 for m in OWNERSHIP_MARKERS
                if m in game.language.vocabulary
            )
            recent = game.round_history[-50:]
            avg_conflict = np.mean([
                h['conflict_rate'] for h in recent
            ]) if recent else 0.0
            total_sharings = sum(h['sharings'] for h in recent)

            snapshot = {
                'round': r + 1,
                'emerged_markers': emerged,
                'marker_frequencies': vocab_markers,
                'avg_conflict_rate': round(float(avg_conflict), 4),
                'recent_sharings': total_sharings,
            }
            snapshots.append(snapshot)
            print(f"  Round {r+1}: markers={emerged}/{len(OWNERSHIP_MARKERS)}, "
                  f"conflict={avg_conflict:.3f}, sharings={total_sharings}")

    final_markers = {
        m: game.language.vocabulary.get(m, {}).get('frequency', 0)
        for m in OWNERSHIP_MARKERS
    }
    emerged_count = sum(1 for m in OWNERSHIP_MARKERS if m in game.language.vocabulary)

    total_conflicts = sum(h['conflicts'] for h in game.round_history)
    total_sharings = sum(h['sharings'] for h in game.round_history)

    # 冲突解决率（有语言协调时冲突后成功分配的比例）
    conflict_rounds = [h for h in game.round_history if h['conflicts'] > 0]
    if conflict_rounds:
        first_half = conflict_rounds[:len(conflict_rounds) // 2]
        second_half = conflict_rounds[len(conflict_rounds) // 2:]
        early_rate = np.mean([h['conflict_rate'] for h in first_half])
        late_rate = np.mean([h['conflict_rate'] for h in second_half])
    else:
        early_rate = 0.0
        late_rate = 0.0

    print(f"\n  最终所有权标记数: {emerged_count}/{len(OWNERSHIP_MARKERS)}")
    print(f"  标记频率: {final_markers}")
    print(f"  总冲突: {total_conflicts}, 总共享: {total_sharings}")
    print(f"  冲突率变化: {early_rate:.3f} -> {late_rate:.3f}")

    return {
        'final_marker_count': emerged_count,
        'marker_frequencies': final_markers,
        'total_conflicts': total_conflicts,
        'total_sharings': total_sharings,
        'conflict_rate_change': {
            'early': round(float(early_rate), 4),
            'late': round(float(late_rate), 4),
            'improvement': round(float(early_rate - late_rate), 4),
        },
        'snapshots': snapshots,
    }


def experiment_2_with_vs_without_ownership(num_rounds: int = 200,
                                            num_runs: int = 5) -> Dict:
    """
    实验 2：有/无所有权语言对比

    对比 OwnershipGame vs BaselineNoOwnershipGame。
    追踪冲突率和资源获取效率。
    """
    print("=" * 60)
    print("实验 2：有/无所有权语言对比")
    print("=" * 60)

    conditions = ['with_ownership', 'without_ownership']
    condition_results = {}

    for condition in conditions:
        run_results = []
        for run in range(num_runs):
            if condition == 'with_ownership':
                game = OwnershipGame(num_agents=3, num_resources=5)
            else:
                game = BaselineNoOwnershipGame(num_agents=3, num_resources=5)

            for _ in range(num_rounds):
                game.play_round()

            conflict_rates = [h['conflict_rate'] for h in game.round_history]
            sharings = sum(h['sharings'] for h in game.round_history)
            acquisitions = sum(h['acquisitions'] for h in game.round_history)

            # 资源获取效率 = 成功获取（无冲突）的比例
            efficiency = (acquisitions - sum(
                h['conflicts'] for h in game.round_history
            )) / max(1, acquisitions)

            run_results.append({
                'avg_conflict_rate': round(float(np.mean(conflict_rates)), 4),
                'total_sharings': sharings,
                'efficiency': round(float(efficiency), 4),
            })

        avg_cr = round(float(np.mean([r['avg_conflict_rate'] for r in run_results])), 4)
        avg_sh = round(float(np.mean([r['total_sharings'] for r in run_results])), 4)
        avg_eff = round(float(np.mean([r['efficiency'] for r in run_results])), 4)

        condition_results[condition] = {
            'avg_conflict_rate': avg_cr,
            'total_sharings': avg_sh,
            'efficiency': avg_eff,
            'runs': run_results,
        }
        print(f"  {condition}: conflict_rate={avg_cr:.3f}, "
              f"sharings={avg_sh:.0f}, efficiency={avg_eff:.3f}")

    with_cr = condition_results['with_ownership']['avg_conflict_rate']
    without_cr = condition_results['without_ownership']['avg_conflict_rate']
    conflict_reduction = without_cr - with_cr

    with_eff = condition_results['with_ownership']['efficiency']
    without_eff = condition_results['without_ownership']['efficiency']
    efficiency_gain = with_eff - without_eff

    print(f"\n  冲突率降低: {conflict_reduction:.3f} "
          f"({conflict_reduction / max(0.01, without_cr) * 100:.1f}%)")
    print(f"  效率提升: {efficiency_gain:.3f} "
          f"({efficiency_gain / max(0.01, abs(without_eff)) * 100:.1f}%)")

    return {
        'conditions': condition_results,
        'conflict_reduction': round(float(conflict_reduction), 4),
        'efficiency_gain': round(float(efficiency_gain), 4),
    }


def experiment_3_sharing_emergence(num_rounds: int = 300,
                                    scarcity_levels: Optional[List[float]] = None
                                    ) -> Dict:
    """
    实验 3：共享涌现与资源稀缺性

    改变资源稀缺程度，测量共享标记频率。
    预期：适度稀缺 → 共享最多；极高稀缺 → 共享最少。
    """
    if scarcity_levels is None:
        scarcity_levels = [0.3, 0.5, 0.7, 1.0]

    print("=" * 60)
    print("实验 3：共享涌现与资源稀缺性")
    print("=" * 60)

    scarcity_results = {}

    for scarcity in scarcity_levels:
        # 稀缺度决定资源数量：稀缺度高 → 资源少
        num_resources = max(2, int(8 * (1.0 - scarcity) + 2))
        game = OwnershipGame(num_agents=3, num_resources=num_resources)

        sharing_history = []
        marker_history = defaultdict(list)

        for r in range(num_rounds):
            result = game.play_round()
            sharing_history.append(result['sharings'])

            # 每 50 轮追踪共享标记频率
            if (r + 1) % 50 == 0:
                for m in ('share', 'ours'):
                    freq = game.language.vocabulary.get(m, {}).get('frequency', 0)
                    marker_history[m].append(freq)

        total_sharings = sum(sharing_history)
        avg_sharing_rate = total_sharings / num_rounds
        share_freq = game.language.vocabulary.get('share', {}).get('frequency', 0)
        ours_freq = game.language.vocabulary.get('ours', {}).get('frequency', 0)

        scarcity_results[str(scarcity)] = {
            'num_resources': num_resources,
            'total_sharings': total_sharings,
            'avg_sharing_rate': round(float(avg_sharing_rate), 4),
            'share_marker_freq': share_freq,
            'ours_marker_freq': ours_freq,
            'marker_history': {k: v for k, v in marker_history.items()},
        }
        print(f"  稀缺度={scarcity:.1f} (资源={num_resources}): "
              f"共享={total_sharings}, share频率={share_freq}, ours频率={ours_freq}")

    # 找出共享最多的稀缺度
    best_scarcity = max(
        scarcity_results.items(),
        key=lambda x: x[1]['total_sharings']
    )

    print(f"\n  共享最多的稀缺度: {best_scarcity[0]} "
          f"(共享={best_scarcity[1]['total_sharings']})")

    return {
        'scarcity_results': scarcity_results,
        'best_scarcity_for_sharing': best_scarcity[0],
    }


def experiment_4_ownership_transfer(num_rounds: int = 200) -> Dict:
    """
    实验 4：所有权转移标记

    测试 'give'/'take' 转移标记是否提高资源分配效率。
    对比有转移标记 vs 无转移标记。
    """
    print("=" * 60)
    print("实验 4：所有权转移标记")
    print("=" * 60)

    conditions = ['with_transfer', 'without_transfer']
    condition_results = {}

    for condition in conditions:
        language = EmergingLanguage()
        agents = {
            i: OwnershipAgent(i, language) for i in range(3)
        }
        transfer_count = 0
        efficiency_scores = []
        allocation_history = []

        for r in range(num_rounds):
            scenario = ResourceScenario.generate_scenario(
                num_resources=5, num_agents=3
            )

            # 初始随机分配
            for res in scenario.resources:
                res.owner = random.randint(0, 2)
                agents[res.owner].owned_items.append(res.item_id)

            # 计算当前分配效率（价值与 Agent 需求的匹配度）
            def compute_efficiency():
                values = defaultdict(float)
                for res in scenario.resources:
                    if res.owner is not None:
                        values[res.owner] += res.value
                total = sum(values.values())
                if total == 0:
                    return 1.0
                # 效率 = 1 - 标准差/均值（越均匀越好）
                vals = list(values.values())
                if not vals or np.mean(vals) == 0:
                    return 1.0
                return float(1.0 - np.std(vals) / np.mean(vals))

            pre_efficiency = compute_efficiency()

            if condition == 'with_transfer':
                # 允许转移：高价值 Agent 给低价值 Agent 转移资源
                values = defaultdict(float)
                for res in scenario.resources:
                    if res.owner is not None:
                        values[res.owner] += res.value

                for res in scenario.resources:
                    if res.owner is not None and random.random() < 0.3:
                        current_owner = res.owner
                        # 找价值最低的 Agent
                        min_agent = min(values, key=values.get)
                        if min_agent != current_owner:
                            # 执行转移
                            values[current_owner] -= res.value
                            values[min_agent] += res.value
                            res.owner = min_agent
                            transfer_count += 1

                            # 记录转移标记
                            marker = random.choice(['give', 'take'])
                            language.record_usage([marker], True)

            post_efficiency = compute_efficiency()
            efficiency_scores.append(post_efficiency)
            allocation_history.append({
                'pre': round(pre_efficiency, 4),
                'post': round(post_efficiency, 4),
                'improvement': round(post_efficiency - pre_efficiency, 4),
            })

        # 清理 Agent 状态
        for agent in agents.values():
            agent.owned_items = []
            agent.claimed_items = []

        avg_eff = round(float(np.mean(efficiency_scores)), 4)
        give_freq = language.vocabulary.get('give', {}).get('frequency', 0)
        take_freq = language.vocabulary.get('take', {}).get('frequency', 0)

        # 每 20 轮计算一次平均效率作为历史趋势
        eff_history = []
        step = 20
        for i in range(0, len(allocation_history), step):
            batch = allocation_history[i:i + step]
            if batch:
                eff_history.append(
                    round(float(np.mean([h['post'] for h in batch])), 4)
                )

        condition_results[condition] = {
            'avg_efficiency': avg_eff,
            'transfers': transfer_count,
            'give_frequency': give_freq,
            'take_frequency': take_freq,
            'efficiency_history': eff_history,
        }
        print(f"  {condition}: efficiency={avg_eff:.3f}, "
              f"transfers={transfer_count}, "
              f"give={give_freq}, take={take_freq}")

    # 计算效率差异
    with_eff = condition_results['with_transfer']['avg_efficiency']
    without_eff = condition_results['without_transfer']['avg_efficiency']
    improvement = with_eff - without_eff
    pct_improvement = improvement / max(0.01, abs(without_eff)) * 100

    print(f"\n  转移标记效率提升: {improvement:.3f} ({pct_improvement:.1f}%)")

    return {
        'conditions': condition_results,
        'efficiency_improvement': round(float(improvement), 4),
        'pct_improvement': round(float(pct_improvement), 2),
    }


if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_ownership_markers()
    results['experiment_2'] = experiment_2_with_vs_without_ownership()
    results['experiment_3'] = experiment_3_sharing_emergence()
    results['experiment_4'] = experiment_4_ownership_transfer()

    output_file = 'ownership_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {output_file}")
