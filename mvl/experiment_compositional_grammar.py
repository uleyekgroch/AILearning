"""
Phase 32 实验：组合语法涌现

4 个实验：
1. 词序涌现：100 Agent 交流，观察词序一致性是否收敛
2. 信息量排序效果：对比信息量排序 vs 随机排序 vs 类别偏好排序
3. 渐进式 vs 一次性匹配：对比两种 Listener 策略
4. 词序策略趋同：区域隔离 → 开放通信，观察策略融合
"""

import sys
import random
import numpy as np
import json
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from language_emergence import (
    LanguageAgent, Speaker, Listener, EmergingLanguage,
    cross_language_round, compute_language_similarity,
    _symbol_category,
)
from language_rich_scene import generate_rich_scene_v2, RICH_ATTRIBUTES, ALL_ATTRIBUTE_NAMES
from language_compositional import (
    CompositionalSpeaker, CompositionalListener, CompositionalLanguageAgent,
    compositional_cross_language_round, compute_compositional_similarity,
    compute_order_agreement,
)


# ============================================================
# 辅助：随机排序 Speaker（对照组）
# ============================================================

class RandomOrderSpeaker(CompositionalSpeaker):
    """随机排序 Speaker — shape 在前，其余随机（对照组）"""

    def _order_by_information(self, symbols, scene):
        symbols = list(symbols)
        shape_syms = [s for s in symbols if _symbol_category(s) == 'shape']
        other_syms = [s for s in symbols if _symbol_category(s) != 'shape']
        random.shuffle(other_syms)
        return shape_syms + other_syms


class CategoryOrderSpeaker(CompositionalSpeaker):
    """类别偏好排序 Speaker — shape 在前，其余按固定类别顺序"""

    CATEGORY_PRIORITY = {'size': 0, 'color': 1, 'material': 2, 'texture': 3}

    def _order_by_information(self, symbols, scene):
        shape_syms = [s for s in symbols if _symbol_category(s) == 'shape']
        other_syms = [s for s in symbols if _symbol_category(s) != 'shape']

        def priority(sym):
            cat = _symbol_category(sym) or 'other'
            return self.CATEGORY_PRIORITY.get(cat, 50)

        other_sorted = sorted(other_syms, key=priority)
        return shape_syms + other_sorted


class OneShotListener(Listener):
    """一次性匹配 Listener — 基类行为（对照组）"""
    pass


# ============================================================
# 实验 1：词序涌现
# ============================================================

def experiment_1_order_emergence(num_agents: int = 100,
                                 num_rounds: int = 2000,
                                 verbose: bool = True) -> Dict:
    """
    词序涌现实验

    100 Agent 使用 CompositionalSpeaker/Listener 交流 2000 轮。
    测量词序一致性随时间的变化。
    预期：从随机（~0.3）收敛到一致（>0.7）。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：词序涌现")
        print("=" * 60)

    agents = [CompositionalLanguageAgent(f"agent_{i}") for i in range(num_agents)]

    log = {
        'rounds': [],
        'order_agreement': [],
        'avg_order_consistency': [],
        'success_rate': [],
        'avg_similarity': [],
    }

    for r in range(num_rounds):
        # 随机选一对 agent 交流
        i, j = random.sample(range(num_agents), 2)
        scene = generate_rich_scene_v2(num_objects=8, num_attributes=4)
        target_idx = random.randint(0, len(scene) - 1)
        compositional_cross_language_round(agents[i], agents[j], scene, target_idx)

        if (r + 1) % 200 == 0:
            # 计算指标
            agreement = compute_order_agreement(agents, sample_size=200)
            consistencies = [a.get_order_consistency() for a in agents]
            avg_consistency = np.mean(consistencies)

            # 成功率（最近 200 轮）
            recent = []
            for a in agents:
                for log_entry in a.communication_log[-50:]:
                    recent.append(log_entry['success'])
            success_rate = np.mean(recent) if recent else 0.0

            # 相似度（采样）
            sims = []
            for _ in range(50):
                ai, aj = random.sample(range(num_agents), 2)
                sim = compute_compositional_similarity(agents[ai], agents[aj])
                sims.append(sim['overall_similarity'])

            log['rounds'].append(r + 1)
            log['order_agreement'].append(agreement)
            log['avg_order_consistency'].append(avg_consistency)
            log['success_rate'].append(success_rate)
            log['avg_similarity'].append(np.mean(sims))

            if verbose:
                print(f"  Round {r+1:5d}: agreement={agreement:.3f}, "
                      f"consistency={avg_consistency:.3f}, "
                      f"success={success_rate:.3f}, "
                      f"similarity={np.mean(sims):.3f}")

    # 最终分析
    final_agreement = compute_order_agreement(agents, sample_size=500)
    final_consistency = np.mean([a.get_order_consistency() for a in agents])

    # 找出最常见的类别顺序
    cat_orders = [tuple(a.get_category_order()) for a in agents]
    from collections import Counter
    order_counts = Counter(cat_orders)
    top_orders = order_counts.most_common(3)

    if verbose:
        print(f"\n词序涌现结果:")
        print(f"  最终词序一致性: {final_agreement:.3f}")
        print(f"  平均个体一致性: {final_consistency:.3f}")
        print(f"  最常见类别顺序:")
        for order, count in top_orders:
            print(f"    {order}: {count}/{num_agents} agents")

    return {
        'log': log,
        'final_agreement': final_agreement,
        'final_consistency': final_consistency,
        'top_orders': [(list(o), c) for o, c in top_orders],
    }


# ============================================================
# 实验 2：信息量排序效果
# ============================================================

def experiment_2_information_ordering(num_agents: int = 50,
                                      num_rounds: int = 2000,
                                      verbose: bool = True) -> Dict:
    """
    信息量排序效果对比

    三种 Speaker 排序策略：
    A: 信息量排序（CompositionalSpeaker — 默认）
    B: 随机排序（RandomOrderSpeaker）
    C: 类别偏好排序（CategoryOrderSpeaker）

    每种策略 50 Agent，2000 轮，对比通信成功率和平均话语长度。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：信息量排序效果对比")
        print("=" * 60)

    results = {}

    for name, speaker_class in [
        ('info_order', CompositionalSpeaker),
        ('random_order', RandomOrderSpeaker),
        ('category_order', CategoryOrderSpeaker),
    ]:
        if verbose:
            print(f"\n--- {name} ---")

        agents = []
        for i in range(num_agents):
            agent = CompositionalLanguageAgent(f"{name}_{i}")
            agent.speaker = speaker_class(agent.language)
            agents.append(agent)

        log = {'rounds': [], 'success_rate': [], 'avg_utterance_len': []}

        for r in range(num_rounds):
            i, j = random.sample(range(num_agents), 2)
            scene = generate_rich_scene_v2(num_objects=8, num_attributes=4)
            target_idx = random.randint(0, len(scene) - 1)
            compositional_cross_language_round(agents[i], agents[j], scene, target_idx)

            if (r + 1) % 200 == 0:
                recent_success = []
                recent_lens = []
                for a in agents:
                    for entry in a.communication_log[-30:]:
                        recent_success.append(entry['success'])
                        recent_lens.append(len(entry['utterance']))

                sr = np.mean(recent_success) if recent_success else 0.0
                al = np.mean(recent_lens) if recent_lens else 0.0

                log['rounds'].append(r + 1)
                log['success_rate'].append(sr)
                log['avg_utterance_len'].append(al)

                if verbose:
                    print(f"  Round {r+1:5d}: success={sr:.3f}, avg_len={al:.2f}")

        results[name] = log

    if verbose:
        print(f"\n排序策略对比结果:")
        for name, log in results.items():
            print(f"  {name:20s}: final_success={log['success_rate'][-1]:.3f}, "
                  f"final_avg_len={log['avg_utterance_len'][-1]:.2f}")

    return results


# ============================================================
# 实验 3：渐进式 vs 一次性匹配
# ============================================================

def experiment_3_progressive_vs_oneshot(num_rounds: int = 2000,
                                        verbose: bool = True) -> Dict:
    """
    渐进式 vs 一次性匹配对比

    相同场景、相同 Speaker，对比两种 Listener：
    A: 渐进式匹配（CompositionalListener）
    B: 一次性匹配（OneShotListener = 基类 Listener）

    使用共享语言（单 agent 模式）确保 Speaker 行为一致。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：渐进式 vs 一次性匹配")
        print("=" * 60)

    results = {}

    for name, listener_class in [
        ('progressive', CompositionalListener),
        ('oneshot', OneShotListener),
    ]:
        if verbose:
            print(f"\n--- {name} ---")

        language = EmergingLanguage()
        speaker = CompositionalSpeaker(language)
        listener = listener_class(language)

        log = {'rounds': [], 'success_rate': [], 'avg_candidates': []}
        total_success = 0
        total_games = 0

        for r in range(num_rounds):
            scene = generate_rich_scene_v2(num_objects=8, num_attributes=4)
            target_idx = random.randint(0, len(scene) - 1)
            target = scene[target_idx]

            utterance = speaker.describe(target, scene)
            if not utterance:
                continue

            chosen = listener.interpret(utterance, scene)
            success = (chosen == target_idx)

            total_games += 1
            if success:
                total_success += 1

            # 更新语言
            language.total_games += 1
            if success:
                language.total_successes += 1
            language.record_usage(utterance, success)
            if len(utterance) >= 2:
                for i in range(len(utterance) - 1):
                    language.record_collocation(utterance[i], utterance[i + 1], success)
                language.record_ngram(utterance, success)

            if (r + 1) % 200 == 0:
                sr = total_success / total_games if total_games > 0 else 0.0
                log['rounds'].append(r + 1)
                log['success_rate'].append(sr)
                log['avg_candidates'].append(0.0)  # placeholder

                if verbose:
                    print(f"  Round {r+1:5d}: success={sr:.3f}")

        results[name] = log

    if verbose:
        print(f"\n匹配策略对比结果:")
        for name, log in results.items():
            print(f"  {name:15s}: final_success={log['success_rate'][-1]:.3f}")

    return results


# ============================================================
# 实验 4：词序策略趋同
# ============================================================

def experiment_4_order_convergence(num_agents: int = 100,
                                   num_regions: int = 5,
                                   isolation_rounds: int = 1000,
                                   open_rounds: int = 2000,
                                   verbose: bool = True) -> Dict:
    """
    词序策略趋同实验

    前 isolation_rounds 轮：5 个区域各自发展词序策略（区域隔离）
    后 open_rounds 轮：开放跨区域通信，观察策略融合

    预期：隔离期区域间词序差异增大，开放后差异减小。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 4：词序策略趋同")
        print("=" * 60)

    agents = [CompositionalLanguageAgent(f"agent_{i}") for i in range(num_agents)]
    agent_regions = {i: i % num_regions for i in range(num_agents)}

    log = {
        'rounds': [],
        'intra_agreement': [],
        'inter_agreement': [],
        'overall_agreement': [],
        'success_rate': [],
    }

    def regional_step():
        """区域内交流"""
        region = random.randint(0, num_regions - 1)
        region_agents = [i for i in range(num_agents) if agent_regions[i] == region]
        if len(region_agents) < 2:
            return
        i, j = random.sample(region_agents, 2)
        scene = generate_rich_scene_v2(num_objects=8, num_attributes=4)
        target_idx = random.randint(0, len(scene) - 1)
        compositional_cross_language_round(agents[i], agents[j], scene, target_idx)

    def open_step():
        """跨区域交流"""
        i, j = random.sample(range(num_agents), 2)
        scene = generate_rich_scene_v2(num_objects=8, num_attributes=4)
        target_idx = random.randint(0, len(scene) - 1)
        compositional_cross_language_round(agents[i], agents[j], scene, target_idx)

    # 阶段 1：区域隔离
    if verbose:
        print(f"\n阶段 1：区域隔离（{isolation_rounds} 轮）")

    for r in range(isolation_rounds):
        regional_step()
        if (r + 1) % 200 == 0:
            # 区域内一致性
            intra_agrs = []
            for region in range(num_regions):
                region_agents = [agents[i] for i in range(num_agents)
                                if agent_regions[i] == region]
                if len(region_agents) >= 2:
                    intra_agrs.append(compute_order_agreement(region_agents, sample_size=50))

            # 区域间一致性（跨区域 agent 对）
            inter_agrs = []
            for _ in range(100):
                ri, rj = random.sample(range(num_regions), 2)
                ai = random.choice([i for i in range(num_agents) if agent_regions[i] == ri])
                aj = random.choice([i for i in range(num_agents) if agent_regions[i] == rj])
                cat_i = tuple(agents[ai].get_category_order())
                cat_j = tuple(agents[aj].get_category_order())
                if cat_i and cat_j:
                    inter_agrs.append(1.0 if cat_i == cat_j else 0.0)

            overall = compute_order_agreement(agents, sample_size=200)

            log['rounds'].append(r + 1)
            log['intra_agreement'].append(np.mean(intra_agrs) if intra_agrs else 0.0)
            log['inter_agreement'].append(np.mean(inter_agrs) if inter_agrs else 0.0)
            log['overall_agreement'].append(overall)
            log['success_rate'].append(0.0)

            if verbose:
                print(f"  Round {r+1:5d}: intra={np.mean(intra_agrs) if intra_agrs else 0:.3f}, "
                      f"inter={np.mean(inter_agrs) if inter_agrs else 0:.3f}, "
                      f"overall={overall:.3f}")

    # 阶段 2：开放通信
    if verbose:
        print(f"\n阶段 2：开放跨区域通信（{open_rounds} 轮）")

    for r in range(open_rounds):
        open_step()
        if (r + 1) % 200 == 0:
            total_round = isolation_rounds + r + 1
            overall = compute_order_agreement(agents, sample_size=200)

            # 最终类别顺序
            cat_orders = [tuple(a.get_category_order()) for a in agents]

            log['rounds'].append(total_round)
            log['intra_agreement'].append(overall)
            log['inter_agreement'].append(overall)
            log['overall_agreement'].append(overall)
            log['success_rate'].append(0.0)

            if verbose:
                print(f"  Round {total_round:5d}: overall_agreement={overall:.3f}")

    # 最终分析
    final_agreement = compute_order_agreement(agents, sample_size=500)
    cat_orders = [tuple(a.get_category_order()) for a in agents]
    from collections import Counter
    order_counts = Counter(cat_orders)
    top_orders = order_counts.most_common(3)

    if verbose:
        print(f"\n词序策略趋同结果:")
        print(f"  最终词序一致性: {final_agreement:.3f}")
        print(f"  最常见类别顺序:")
        for order, count in top_orders:
            print(f"    {order}: {count}/{num_agents} agents")

    return {
        'log': log,
        'final_agreement': final_agreement,
        'top_orders': [(list(o), c) for o, c in top_orders],
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 32: 组合语法涌现实验")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    # 实验 1：词序涌现
    results['exp1'] = experiment_1_order_emergence(
        num_agents=100, num_rounds=2000, verbose=True
    )

    # 实验 2：信息量排序效果
    results['exp2'] = experiment_2_information_ordering(
        num_agents=50, num_rounds=2000, verbose=True
    )

    # 实验 3：渐进式 vs 一次性匹配
    results['exp3'] = experiment_3_progressive_vs_oneshot(
        num_rounds=2000, verbose=True
    )

    # 实验 4：词序策略趋同
    results['exp4'] = experiment_4_order_convergence(
        num_agents=100, num_regions=5,
        isolation_rounds=1000, open_rounds=2000, verbose=True
    )

    # 保存结果
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, set):
            return sorted(list(obj))
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    with open('compositional_grammar_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 compositional_grammar_results.json")

    return results


if __name__ == '__main__':
    main()
