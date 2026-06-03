"""
Phase 84: 方言分歧与语言接触（洋泾浜/克里奥尔语形成）

核心思想：
隔离群体发展出不同的词汇（方言分歧），
当群体恢复接触时，沟通困难 → 共享符号涌现（洋泾浜），
最终形成稳定的接触语言（克里奥尔化）。

涌现条件：
1. 隔离期：群体内部交流导致词汇差异化
2. 接触期：跨群体交流压力催生共享符号
3. 稳定期：共享符号体系固化为克里奥尔语

4 个实验：
1. 方言分歧 — 隔离导致词汇分化
2. 洋泾浜形成 — 接触后共享符号涌现
3. 克里奥尔化 — 从洋泾浜到稳定语言
4. 群体规模 vs 接触结果
"""

import sys
import io
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from language_emergence import (
    EmergingLanguage, COLORS, SHAPES, SIZES, MATERIALS,
    _symbol_category, generate_rich_scene,
)


# ============================================================
# 方言标记常量
# ============================================================

# 方言标记：用于区分不同方言社群的特有符号
DIALECT_PREFIX_A = ['aka', 'bik', 'cka', 'dka']  # 社群 A 偏好前缀
DIALECT_PREFIX_B = ['ola', 'pol', 'rol', 'sol']  # 社群 B 偏好前缀
DIALECT_PREFIX_C = ['enu', 'fun', 'gun', 'hun']  # 社群 C 偏好前缀

ALL_DIALECT_PREFIXES = [DIALECT_PREFIX_A, DIALECT_PREFIX_B, DIALECT_PREFIX_C]


# ============================================================
# DialectAgent
# ============================================================

class DialectAgent:
    """
    方言 Agent

    拥有独立的 EmergingLanguage，归属于一个社群。
    在社群内交流时使用社区方言词汇；
    接触外来符号时可选择适应。
    """

    def __init__(self, agent_id: int, community_id: int):
        self.agent_id = agent_id
        self.community_id = community_id
        self.language = EmergingLanguage()

        # 方言偏好：社区特有的符号→成功率映射
        self.dialect_preference: Dict[str, float] = {}

        # 接触适应记录
        self.foreign_symbols_adopted: Set[str] = set()

    def describe_dialect(self, target: Dict[str, str]) -> List[str]:
        """
        用社区方言描述目标物体

        优先选择社区特有的符号组合。
        """
        symbols = []

        # 从目标属性中提取符号，加入方言偏好
        for attr_key, attr_val in target.items():
            if attr_val in COLORS or attr_val in SHAPES or \
               attr_val in SIZES or attr_val in MATERIALS:
                symbols.append(attr_val)

        # 如果社区有方言偏好，以一定概率替换为方言符号
        dialect_syms = []
        for sym in symbols:
            if sym in self.dialect_preference and random.random() < 0.3:
                # 使用方言变体
                dialect_sym = f"{sym}_{self.community_id}"
                dialect_syms.append(dialect_sym)
                # 确保方言符号也在词汇表中
                if dialect_sym not in self.language.vocabulary:
                    self.language.vocabulary[dialect_sym] = {
                        'frequency': 0, 'successes': 0, 'success_rate': 0.0
                    }
            else:
                dialect_syms.append(sym)

        return dialect_syms if dialect_syms else symbols

    def interpret_dialect(self, utterance: List[str],
                          scene: List[Dict[str, str]]) -> int:
        """
        解释方言描述，在场景中选出目标

        优先匹配社区词汇。
        """
        best_idx = 0
        best_score = -1.0

        for idx, obj in enumerate(scene):
            score = 0.0
            for sym in utterance:
                # 去掉方言后缀进行匹配
                base_sym = sym.split('_')[0] if '_' in sym else sym

                # 检查物体属性是否匹配
                for attr_key, attr_val in obj.items():
                    if attr_val == base_sym or attr_val == sym:
                        score += 1.0
                    elif base_sym in attr_val or attr_val in base_sym:
                        score += 0.5

                # 社区词汇加分
                if sym in self.dialect_preference:
                    score += 0.3 * self.dialect_preference[sym]

                # 已知词汇加分
                if sym in self.language.vocabulary:
                    vocab_sr = self.language.vocabulary[sym].get('success_rate', 0)
                    score += 0.2 * vocab_sr

            # 加入随机性（模拟理解偏差）
            score += random.gauss(0, 0.1)

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def adapt_to_contact(self, foreign_utterance: List[str],
                         success: bool):
        """
        从跨群体接触中更新词汇

        成功沟通时，Agent 倾向于采纳外来符号。
        """
        for sym in foreign_utterance:
            base_sym = sym.split('_')[0] if '_' in sym else sym

            # 记录到语言中
            self.language.record_usage([sym], success)

            if success:
                # 成功时更可能采纳外来符号
                if base_sym not in self.dialect_preference:
                    self.dialect_preference[base_sym] = 0.1
                self.dialect_preference[base_sym] = min(
                    1.0, self.dialect_preference[base_sym] + 0.05
                )
                self.foreign_symbols_adopted.add(sym)
            else:
                # 失败时小幅降低偏好
                if base_sym in self.dialect_preference:
                    self.dialect_preference[base_sym] = max(
                        0.0, self.dialect_preference[base_sym] - 0.02
                    )


# ============================================================
# DialectCommunity
# ============================================================

class DialectCommunity:
    """
    方言社群

    一群共享语言变体的 Agent。
    在社群内部交流中发展出独特的词汇。
    """

    def __init__(self, community_id: int, num_agents: int = 10):
        self.community_id = community_id
        self.language = EmergingLanguage()
        self.agents = [
            DialectAgent(i, community_id)
            for i in range(num_agents)
        ]

        # 社群特有符号集合
        self.community_symbols: Set[str] = set()

    def develop_dialect(self, rounds: int = 100):
        """
        社群内部交流，发展独特词汇

        每轮：随机选两个 Agent 进行参照游戏。
        成功模式固化到社群语言中。
        """
        for r in range(rounds):
            scene = generate_rich_scene('medium')
            target_idx = random.randint(0, len(scene) - 1)
            target = scene[target_idx]

            # 随机选一对 Agent
            a, b = random.sample(self.agents, 2)
            speaker, listener = (a, b) if random.random() < 0.5 else (b, a)

            # 说话者描述
            utterance = speaker.describe_dialect(target)

            # 听话者解释
            chosen = listener.interpret_dialect(utterance, scene)
            success = (chosen == target_idx)

            # 更新双方语言
            for sym in utterance:
                speaker.language.record_usage([sym], success)
                listener.language.record_usage([sym], success)
                self.language.record_usage([sym], success)

            # 更新社群符号
            for sym in utterance:
                self.community_symbols.add(sym)

            # 社群内部也更新方言偏好
            if success:
                for sym in utterance:
                    base = sym.split('_')[0] if '_' in sym else sym
                    for agent in [speaker, listener]:
                        if base not in agent.dialect_preference:
                            agent.dialect_preference[base] = 0.1
                        agent.dialect_preference[base] = min(
                            1.0, agent.dialect_preference[base] + 0.02
                        )

    def vocabulary_overlap(self, other: 'DialectCommunity') -> float:
        """计算与另一个社群的词汇重叠率"""
        my_vocab = set(self.language.vocabulary.keys())
        other_vocab = set(other.language.vocabulary.keys())

        if not my_vocab and not other_vocab:
            return 1.0
        if not my_vocab or not other_vocab:
            return 0.0

        intersection = my_vocab & other_vocab
        union = my_vocab | other_vocab
        return len(intersection) / len(union)

    def distinctive_markers(self) -> List[str]:
        """返回本社群独有的符号"""
        return sorted(self.community_symbols)


# ============================================================
# PidginTracker
# ============================================================

class PidginTracker:
    """
    洋泾浜追踪器

    追踪接触期间涌现的共享符号。
    """

    def __init__(self):
        self.shared_vocabulary: Set[str] = set()
        self.emergence_order: List[Tuple[str, int]] = []  # (symbol, round)
        self.round_scheduled: Set[str] = set()

    def record_shared(self, symbol: str, round_num: int):
        """记录一个共享符号的出现"""
        if symbol not in self.round_scheduled:
            self.shared_vocabulary.add(symbol)
            self.emergence_order.append((symbol, round_num))
            self.round_scheduled.add(symbol)

    def measure_coverage(self, community_vocab: Set[str]) -> float:
        """衡量洋泾浜覆盖社群词汇的比例"""
        if not community_vocab:
            return 0.0
        return len(self.shared_vocabulary & community_vocab) / len(community_vocab)


# ============================================================
# ContactScenario
# ============================================================

class ContactScenario:
    """
    语言接触场景

    管理从隔离到接触再到洋泾浜/克里奥尔化的全过程。
    """

    def __init__(self, communities: List[DialectCommunity]):
        self.communities = communities
        self.phase = 'isolation'
        self.pidgin_tracker = PidginTracker()

        # 接触历史
        self.contact_history: List[Dict] = []

    def run_isolation(self, rounds: int = 100):
        """隔离阶段：各社群独立发展"""
        self.phase = 'isolation'
        print(f"  [隔离阶段] {len(self.communities)} 个社群，各 {rounds} 轮")

        for i, comm in enumerate(self.communities):
            comm.develop_dialect(rounds=rounds)
            vocab_size = len(comm.language.vocabulary)
            sr = 0.0
            if comm.language.total_games > 0:
                sr = comm.language.total_successes / comm.language.total_games
            print(f"    社群 {i}: 词汇={vocab_size}, 成功率={sr:.3f}")

    def run_contact(self, rounds: int = 100):
        """
        接触阶段：跨社群交流

        当不同社群的 Agent 交流时：
        - 初期成功率低（词汇不匹配）
        - 逐渐涌现共享符号（洋泾浜）
        - 最终形成稳定的接触语言（克里奥尔化）
        """
        self.phase = 'contact'
        print(f"  [接触阶段] {rounds} 轮跨社群交流")

        for r in range(rounds):
            # 选择来自不同社群的两个 Agent
            comm_indices = random.sample(range(len(self.communities)), 2)
            comm_a = self.communities[comm_indices[0]]
            comm_b = self.communities[comm_indices[1]]

            agent_a = random.choice(comm_a.agents)
            agent_b = random.choice(comm_b.agents)

            # 生成场景（使用说话者社群的场景偏好）
            scene = generate_rich_scene('medium')
            target_idx = random.randint(0, len(scene) - 1)
            target = scene[target_idx]

            # 说话者用自己方言描述
            speaker, listener = (agent_a, agent_b) if random.random() < 0.5 else (agent_b, agent_a)
            utterance = speaker.describe_dialect(target)

            # 听话者尝试理解
            chosen = listener.interpret_dialect(utterance, scene)
            success = (chosen == target_idx)

            # 更新双方
            for sym in utterance:
                speaker.language.record_usage([sym], success)
                listener.language.record_usage([sym], success)

            # 接触适应
            listener.adapt_to_contact(utterance, success)
            speaker.adapt_to_contact(utterance, success)

            # 追踪共享符号
            if success:
                for sym in utterance:
                    self.pidgin_tracker.record_shared(sym, r)

            # 记录接触历史（采样）
            if (r + 1) % 20 == 0 or r == 0:
                self.contact_history.append({
                    'round': r + 1,
                    'success': success,
                })

            # 定期报告
            if (r + 1) % 50 == 0:
                # 计算当前跨社群成功率
                window = self.contact_history[-50:] if len(self.contact_history) >= 50 else self.contact_history
                recent_sr = sum(1 for h in window if h['success']) / max(1, len(window))
                shared_count = len(self.pidgin_tracker.shared_vocabulary)
                print(f"    Round {r+1}: 共享符号={shared_count}, "
                      f"近期成功率={recent_sr:.3f}")

        # 进入洋泾浜/克里奥尔阶段判断
        if len(self.pidgin_tracker.shared_vocabulary) > 10:
            self.phase = 'pidgin'
        if len(self.pidgin_tracker.shared_vocabulary) > 30:
            self.phase = 'creole'

    def measure_divergence(self) -> Dict:
        """测量社群间的词汇分化程度"""
        result = {
            'pairwise_overlap': {},
            'avg_overlap': 0.0,
            'min_overlap': 1.0,
            'max_overlap': 0.0,
            'shared_symbols': len(self.pidgin_tracker.shared_vocabulary),
            'phase': self.phase,
        }

        overlaps = []
        for i in range(len(self.communities)):
            for j in range(i + 1, len(self.communities)):
                ov = self.communities[i].vocabulary_overlap(self.communities[j])
                overlaps.append(ov)
                result['pairwise_overlap'][f'{i}-{j}'] = round(ov, 4)

        if overlaps:
            result['avg_overlap'] = round(float(np.mean(overlaps)), 4)
            result['min_overlap'] = round(float(np.min(overlaps)), 4)
            result['max_overlap'] = round(float(np.max(overlaps)), 4)

        # 各社群词汇量
        result['community_vocab_sizes'] = [
            len(c.language.vocabulary) for c in self.communities
        ]

        # 社群内部成功率
        srs = []
        for c in self.communities:
            if c.language.total_games > 0:
                srs.append(c.language.total_successes / c.language.total_games)
            else:
                srs.append(0.0)
        result['community_success_rates'] = [round(s, 4) for s in srs]

        return result


# ============================================================
# 辅助函数
# ============================================================

def _compute_cross_success(communities: List[DialectCommunity],
                           trials: int = 100) -> float:
    """测量跨社群沟通成功率"""
    if len(communities) < 2:
        return 0.0

    successes = 0
    total = 0

    for _ in range(trials):
        comm_indices = random.sample(range(len(communities)), 2)
        agent_a = random.choice(communities[comm_indices[0]].agents)
        agent_b = random.choice(communities[comm_indices[1]].agents)

        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)
        target = scene[target_idx]

        utterance = agent_a.describe_dialect(target)
        chosen = agent_b.interpret_dialect(utterance, scene)

        if chosen == target_idx:
            successes += 1
        total += 1

    return successes / total if total > 0 else 0.0


# ============================================================
# 实验 1：方言分歧
# ============================================================

def experiment_1_divergence(num_communities: int = 3,
                             isolation_rounds: int = 150) -> Dict:
    """
    隔离导致词汇分化

    N 个社群独立发展，测量词汇重叠率变化。
    预期：重叠率从 100% 下降到 30-50%。
    """
    print("=" * 60)
    print("实验 1：方言分歧（隔离期词汇分化）")
    print("=" * 60)

    communities = [
        DialectCommunity(i, num_agents=10)
        for i in range(num_communities)
    ]

    # 初始重叠率（理论上 100%，因为所有词汇表为空）
    initial_overlap = 1.0

    snapshots = []
    step = 10
    for r in range(0, isolation_rounds, step):
        # 每步发展
        for comm in communities:
            comm.develop_dialect(rounds=step)

        # 测量分化
        overlaps = []
        for i in range(num_communities):
            for j in range(i + 1, num_communities):
                ov = communities[i].vocabulary_overlap(communities[j])
                overlaps.append(ov)

        avg_ov = float(np.mean(overlaps)) if overlaps else 0.0
        vocab_sizes = [len(c.language.vocabulary) for c in communities]

        snapshot = {
            'round': r + step,
            'avg_overlap': round(avg_ov, 4),
            'pairwise': [round(o, 4) for o in overlaps],
            'vocab_sizes': vocab_sizes,
        }
        snapshots.append(snapshot)

        print(f"  Round {r+step:4d}: 平均重叠={avg_ov:.3f}, "
              f"词汇量={[vs for vs in vocab_sizes]}")

    # 最终分析
    final_overlaps = []
    for i in range(num_communities):
        for j in range(i + 1, num_communities):
            ov = communities[i].vocabulary_overlap(communities[j])
            final_overlaps.append(ov)

    final_avg = float(np.mean(final_overlaps))
    print(f"\n  最终平均重叠率: {final_avg:.3f}")
    print(f"  初始重叠率: {initial_overlap:.3f}")
    print(f"  分化程度: {1 - final_avg:.3f}")

    # 各社群独有符号
    for i, comm in enumerate(communities):
        markers = comm.distinctive_markers()
        print(f"  社群 {i} 独有符号 ({len(markers)}): {markers[:10]}")

    return {
        'snapshots': snapshots,
        'initial_overlap': round(initial_overlap, 4),
        'final_avg_overlap': round(final_avg, 4),
        'divergence': round(1 - final_avg, 4),
        'num_communities': num_communities,
        'isolation_rounds': isolation_rounds,
    }


# ============================================================
# 实验 2：洋泾浜形成
# ============================================================

def experiment_2_pidgin_formation(num_communities: int = 2,
                                   isolation_rounds: int = 100,
                                   contact_rounds: int = 200) -> Dict:
    """
    洋泾浜形成：隔离 → 接触

    预期：共享符号从 ~30% 增长到 ~60-70%，成功率从 ~30% 恢复到 ~60%。
    """
    print("=" * 60)
    print("实验 2：洋泾浜形成")
    print("=" * 60)

    communities = [
        DialectCommunity(i, num_agents=10)
        for i in range(num_communities)
    ]
    scenario = ContactScenario(communities)

    # 阶段 1：隔离
    print("\n--- 阶段 1：隔离 ---")
    scenario.run_isolation(rounds=isolation_rounds)

    # 测量隔离后的跨社群沟通
    pre_contact_sr = _compute_cross_success(communities, trials=100)
    pre_divergence = scenario.measure_divergence()
    print(f"  隔离后跨社群成功率: {pre_contact_sr:.3f}")
    print(f"  隔离后词汇重叠: {pre_divergence['avg_overlap']:.3f}")

    # 阶段 2：接触
    print("\n--- 阶段 2：接触 ---")
    pidgin_snapshots = []

    for batch in range(0, contact_rounds, 50):
        scenario.run_contact(rounds=50)

        sr = _compute_cross_success(communities, trials=50)
        div = scenario.measure_divergence()
        shared = len(scenario.pidgin_tracker.shared_vocabulary)

        pidgin_snapshots.append({
            'round': batch + 50,
            'cross_sr': round(sr, 4),
            'avg_overlap': div['avg_overlap'],
            'shared_symbols': shared,
            'phase': scenario.phase,
        })

        print(f"  接触 Round {batch+50:4d}: 跨社群SR={sr:.3f}, "
              f"重叠={div['avg_overlap']:.3f}, 共享符号={shared}")

    # 最终分析
    post_divergence = scenario.measure_divergence()
    post_contact_sr = _compute_cross_success(communities, trials=100)

    print(f"\n  接触前 → 接触后:")
    print(f"    跨社群SR: {pre_contact_sr:.3f} → {post_contact_sr:.3f}")
    print(f"    词汇重叠: {pre_divergence['avg_overlap']:.3f} → {post_divergence['avg_overlap']:.3f}")
    print(f"    共享符号: 0 → {len(scenario.pidgin_tracker.shared_vocabulary)}")
    print(f"    最终阶段: {scenario.phase}")

    # 洋泾浜符号分析
    emergence = scenario.pidgin_tracker.emergence_order[:20]
    print(f"    最先涌现的共享符号: {emergence}")

    return {
        'pidgin_snapshots': pidgin_snapshots,
        'pre_contact_sr': round(pre_contact_sr, 4),
        'post_contact_sr': round(post_contact_sr, 4),
        'pre_overlap': pre_divergence['avg_overlap'],
        'post_overlap': post_divergence['avg_overlap'],
        'shared_symbols_count': len(scenario.pidgin_tracker.shared_vocabulary),
        'final_phase': scenario.phase,
        'emergence_order': emergence,
        'isolation_rounds': isolation_rounds,
        'contact_rounds': contact_rounds,
    }


# ============================================================
# 实验 3：克里奥尔化
# ============================================================

def experiment_3_creole_development(num_communities: int = 3,
                                     isolation: int = 100,
                                     contact: int = 300) -> Dict:
    """
    克里奥尔化：完整的 洋泾浜 → 克里奥尔 周期

    追踪：词汇稳定化、语法模式、新符号涌现。
    预期：克里奥尔语有 ~80%+ 共享词汇，新组合符号涌现。
    """
    print("=" * 60)
    print("实验 3：克里奥尔化（洋泾浜 → 克里奥尔）")
    print("=" * 60)

    communities = [
        DialectCommunity(i, num_agents=10)
        for i in range(num_communities)
    ]
    scenario = ContactScenario(communities)

    # 阶段 1：隔离
    print("\n--- 阶段 1：隔离 ---")
    scenario.run_isolation(rounds=isolation)

    # 阶段 2：接触（洋泾浜形成期）
    print("\n--- 阶段 2：接触/洋泾浜 ---")

    creole_snapshots = []
    novel_symbols_timeline: List[Tuple[int, int]] = []  # (round, count)

    # 记录接触前的词汇
    pre_contact_vocab = set()
    for comm in communities:
        pre_contact_vocab.update(comm.language.vocabulary.keys())

    step = 50
    for batch in range(0, contact, step):
        scenario.run_contact(rounds=step)

        # 测量各项指标
        sr = _compute_cross_success(communities, trials=50)
        div = scenario.measure_divergence()
        shared = len(scenario.pidgin_tracker.shared_vocabulary)

        # 新符号：接触后出现但不在隔离词汇中的符号
        current_vocab = set()
        for comm in communities:
            current_vocab.update(comm.language.vocabulary.keys())
        novel = current_vocab - pre_contact_vocab
        novel_symbols_timeline.append((batch + step, len(novel)))

        # 词汇稳定性：最后 50 轮新增词汇量
        prev_vocab = set()
        if creole_snapshots:
            prev_vocab_size = creole_snapshots[-1].get('total_vocab', 0)
        else:
            prev_vocab_size = len(pre_contact_vocab)

        vocab_growth = len(current_vocab) - prev_vocab_size

        # 社群内部 vs 跨社群 成功率
        intra_srs = []
        for comm in communities:
            if comm.language.total_games > 0:
                intra_srs.append(comm.language.total_successes / comm.language.total_games)
        avg_intra = float(np.mean(intra_srs)) if intra_srs else 0.0

        snapshot = {
            'round': batch + step,
            'cross_sr': round(sr, 4),
            'intra_sr': round(avg_intra, 4),
            'avg_overlap': div['avg_overlap'],
            'shared_symbols': shared,
            'novel_symbols': len(novel),
            'vocab_growth': vocab_growth,
            'total_vocab': len(current_vocab),
            'phase': scenario.phase,
        }
        creole_snapshots.append(snapshot)

        print(f"  Round {batch+step:4d}: 跨社群SR={sr:.3f}, "
              f"群内SR={avg_intra:.3f}, 重叠={div['avg_overlap']:.3f}, "
              f"共享={shared}, 新增={len(novel)}, 阶段={scenario.phase}")

    # 最终克里奥尔分析
    print("\n--- 克里奥尔语分析 ---")

    # 各社群最终词汇
    final_vocabs = []
    for i, comm in enumerate(communities):
        vocab = set(comm.language.vocabulary.keys())
        final_vocabs.append(vocab)
        print(f"  社群 {i}: 词汇量={len(vocab)}")

    # 全部共享符号
    if final_vocabs:
        common = set.intersection(*final_vocabs)
    else:
        common = set()
    print(f"  全部社群共享符号: {len(common)}")

    # 任意两社群共享
    pairwise_shared = []
    for i in range(num_communities):
        for j in range(i + 1, num_communities):
            shared_ij = final_vocabs[i] & final_vocabs[j]
            pairwise_shared.append(len(shared_ij))
    avg_pairwise = float(np.mean(pairwise_shared)) if pairwise_shared else 0
    print(f"  平均两两共享符号: {avg_pairwise:.1f}")

    # 克里奥尔化指标
    total_vocab = len(set.union(*final_vocabs)) if final_vocabs else 0
    creole_ratio = len(common) / total_vocab if total_vocab > 0 else 0
    print(f"  克里奥尔覆盖率: {creole_ratio:.3f}")

    # 新符号涌现分析
    novel_final = set.union(*final_vocabs) - pre_contact_vocab if final_vocabs else set()
    print(f"  接触期新涌现符号: {len(novel_final)}")
    print(f"    前 10 个: {sorted(list(novel_final))[:10]}")

    return {
        'creole_snapshots': creole_snapshots,
        'final_common_symbols': len(common),
        'creole_coverage': round(creole_ratio, 4),
        'novel_symbols_total': len(novel_final),
        'novel_timeline': novel_symbols_timeline,
        'pairwise_shared_avg': round(avg_pairwise, 1),
        'final_phase': scenario.phase,
        'isolation_rounds': isolation,
        'contact_rounds': contact,
    }


# ============================================================
# 实验 4：群体规模 vs 接触结果
# ============================================================

def experiment_4_dialect_contact_scaling(
        community_sizes: List[int] = None) -> Dict:
    """
    不同社群规模对接触结果的影响

    预期：更大社群 → 更慢收敛但更稳定的洋泾浜。
    """
    if community_sizes is None:
        community_sizes = [3, 5, 10, 20]

    print("=" * 60)
    print("实验 4：群体规模 vs 接触结果")
    print("=" * 60)

    isolation_rounds = 80
    contact_rounds = 150

    results = {}

    for size in community_sizes:
        print(f"\n--- 社群规模: {size} ---")

        communities = [
            DialectCommunity(i, num_agents=size)
            for i in range(2)
        ]
        scenario = ContactScenario(communities)

        # 隔离
        scenario.run_isolation(rounds=isolation_rounds)

        # 隔离后测量
        pre_sr = _compute_cross_success(communities, trials=80)
        pre_div = scenario.measure_divergence()

        # 接触（分步追踪）
        contact_log = []
        for batch in range(0, contact_rounds, 30):
            scenario.run_contact(rounds=30)

            sr = _compute_cross_success(communities, trials=40)
            div = scenario.measure_divergence()
            shared = len(scenario.pidgin_tracker.shared_vocabulary)

            contact_log.append({
                'round': batch + 30,
                'cross_sr': round(sr, 4),
                'avg_overlap': div['avg_overlap'],
                'shared_symbols': shared,
            })

        # 最终测量
        post_sr = _compute_cross_success(communities, trials=80)
        post_div = scenario.measure_divergence()

        # 社群内部沟通质量
        intra_srs = []
        for comm in communities:
            if comm.language.total_games > 0:
                intra_srs.append(comm.language.total_successes / comm.language.total_games)
        avg_intra = float(np.mean(intra_srs)) if intra_srs else 0.0

        # 稳定性：最后 30 轮成功率方差
        if len(contact_log) >= 2:
            last_srs = [c['cross_sr'] for c in contact_log[-3:]]
            stability = 1.0 - float(np.std(last_srs))
        else:
            stability = 0.0

        result = {
            'community_size': size,
            'pre_contact_sr': round(pre_sr, 4),
            'post_contact_sr': round(post_sr, 4),
            'sr_improvement': round(post_sr - pre_sr, 4),
            'pre_overlap': pre_div['avg_overlap'],
            'post_overlap': post_div['avg_overlap'],
            'shared_symbols': len(scenario.pidgin_tracker.shared_vocabulary),
            'intra_sr': round(avg_intra, 4),
            'stability': round(max(0.0, stability), 4),
            'contact_log': contact_log,
        }
        results[f'size_{size}'] = result

        print(f"  规模 {size:2d}: 跨社群SR {pre_sr:.3f} → {post_sr:.3f}, "
              f"重叠 {pre_div['avg_overlap']:.3f} → {post_div['avg_overlap']:.3f}, "
              f"共享={len(scenario.pidgin_tracker.shared_vocabulary)}, "
              f"稳定性={stability:.3f}")

    # 规模效应分析
    print("\n--- 规模效应分析 ---")
    sizes_list = []
    improvements = []
    stabilities = []

    for key, data in results.items():
        s = data['community_size']
        sizes_list.append(s)
        improvements.append(data['sr_improvement'])
        stabilities.append(data['stability'])
        print(f"  规模 {s:2d}: SR提升={data['sr_improvement']:+.3f}, "
              f"稳定性={data['stability']:.3f}")

    # 相关性
    if len(sizes_list) >= 3:
        corr_improve = float(np.corrcoef(sizes_list, improvements)[0, 1])
        corr_stable = float(np.corrcoef(sizes_list, stabilities)[0, 1])
        print(f"\n  规模-SR提升 相关系数: {corr_improve:.3f}")
        print(f"  规模-稳定性 相关系数: {corr_stable:.3f}")
    else:
        corr_improve = 0.0
        corr_stable = 0.0

    return {
        'by_size': results,
        'corr_size_improvement': round(corr_improve, 4),
        'corr_size_stability': round(corr_stable, 4),
    }


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    all_results = {}

    all_results['experiment_1_divergence'] = experiment_1_divergence(
        num_communities=3, isolation_rounds=150
    )
    all_results['experiment_2_pidgin'] = experiment_2_pidgin_formation(
        num_communities=2, isolation_rounds=100, contact_rounds=200
    )
    all_results['experiment_3_creole'] = experiment_3_creole_development(
        num_communities=3, isolation=100, contact=300
    )
    all_results['experiment_4_scaling'] = experiment_4_dialect_contact_scaling(
        community_sizes=[3, 5, 10, 20]
    )

    with open('dialect_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print("\n\n结果已保存到 dialect_results.json")
