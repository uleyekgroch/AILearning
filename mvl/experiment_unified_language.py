"""
Phase 24 实验：统一语言系统

5 个实验验证多种符号系统能否组合使用。
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounding_unified_language import (
    UnifiedCommunicationGame, BaselineVisualGame,
    generate_unified_scenario,
)


def experiment_1_single_module(num_rounds=300, verbose=True):
    """实验 1：单模块基线 — 每种歧义类型独立测试"""
    print("\n" + "=" * 60)
    print("实验 1：单模块基线")
    print("=" * 60)

    ambiguity_configs = [
        ('visual_ambiguous', {'visual_ambiguous'}),
        ('crossmodal', {'crossmodal'}),
        ('subset', {'subset'}),
        ('causal', {'causal'}),
        ('confidence', {'confidence'}),
        ('tool', {'tool'}),
    ]

    results = {}
    for name, types in ambiguity_configs:
        game = UnifiedCommunicationGame()
        for _ in range(num_rounds):
            scene = generate_unified_scenario(ambiguity_types=types, num_objects=3)
            game.play_round(scene)

        stats = game.get_stats()
        results[name] = stats
        if verbose:
            print(f"  {name:20s}: 成功率={stats['success_rate']:.1%}, "
                  f"词汇量={stats['vocabulary_size']}, "
                  f"类别数={stats['num_categories']}")

    return results


def experiment_2_dual_combination(num_rounds=300, verbose=True):
    """实验 2：双模块组合 — 两种歧义叠加"""
    print("\n" + "=" * 60)
    print("实验 2：双模块组合")
    print("=" * 60)

    combos = [
        ('visual+crossmodal', {'visual_ambiguous', 'crossmodal'}),
        ('visual+subset', {'visual_ambiguous', 'subset'}),
        ('crossmodal+causal', {'crossmodal', 'causal'}),
        ('subset+confidence', {'subset', 'confidence'}),
        ('tool+crossmodal', {'tool', 'crossmodal'}),
    ]

    results = {}
    for name, types in combos:
        game = UnifiedCommunicationGame()
        for _ in range(num_rounds):
            scene = generate_unified_scenario(
                ambiguity_types=types, num_objects=3,
                speaker_confidence=random.uniform(0.3, 0.95),
            )
            game.play_round(scene)

        stats = game.get_stats()
        results[name] = stats
        if verbose:
            print(f"  {name:25s}: 成功率={stats['success_rate']:.1%}, "
                  f"词汇量={stats['vocabulary_size']}, "
                  f"类别数={stats['num_categories']}")

    return results


def experiment_3_full_combination(num_rounds=500, verbose=True):
    """实验 3：全模块组合 — 所有歧义同时存在"""
    print("\n" + "=" * 60)
    print("实验 3：全模块组合")
    print("=" * 60)

    all_types = {'visual_ambiguous', 'crossmodal', 'subset', 'causal', 'confidence', 'tool'}

    game = UnifiedCommunicationGame()
    for i in range(num_rounds):
        scene = generate_unified_scenario(
            ambiguity_types=all_types,
            num_objects=4,
            speaker_confidence=random.uniform(0.2, 0.95),
        )
        game.play_round(scene)

        if verbose and (i + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  轮次 {i+1}: 成功率={stats['success_rate']:.1%}, "
                  f"词汇量={stats['vocabulary_size']}, "
                  f"类别数={stats['num_categories']}")

    stats = game.get_stats()
    if verbose:
        print(f"\n  最终结果:")
        print(f"    成功率: {stats['success_rate']:.1%}")
        print(f"    词汇量: {stats['vocabulary_size']}")
        print(f"    策略使用: {stats['strategy_usage']}")
        print(f"    符号类别: {stats['symbol_categories_used']}")
        print(f"    类别数: {stats['num_categories']}")

    return stats


def experiment_4_comparison(num_rounds=300, verbose=True):
    """实验 4：组合 vs 纯视觉基线"""
    print("\n" + "=" * 60)
    print("实验 4：组合系统 vs 纯视觉基线")
    print("=" * 60)

    all_types = {'visual_ambiguous', 'crossmodal', 'subset', 'causal', 'confidence'}

    # 统一系统
    unified = UnifiedCommunicationGame()
    for _ in range(num_rounds):
        scene = generate_unified_scenario(
            ambiguity_types=all_types, num_objects=3,
            speaker_confidence=random.uniform(0.3, 0.95),
        )
        unified.play_round(scene)

    # 纯视觉基线
    baseline = BaselineVisualGame()
    for _ in range(num_rounds):
        scene = generate_unified_scenario(
            ambiguity_types=all_types, num_objects=3,
            speaker_confidence=random.uniform(0.3, 0.95),
        )
        baseline.play_round(scene)

    u_stats = unified.get_stats()
    b_stats = baseline.get_stats()
    improvement = u_stats['success_rate'] - b_stats['success_rate']

    if verbose:
        print(f"  统一系统: {u_stats['success_rate']:.1%} (词汇量={u_stats['vocabulary_size']}, 类别数={u_stats['num_categories']})")
        print(f"  纯视觉:  {b_stats['success_rate']:.1%} (词汇量={b_stats['vocabulary_size']})")
        print(f"  改善: {improvement:+.1%}")

    return {
        'unified': u_stats,
        'baseline': b_stats,
        'improvement': improvement,
    }


def experiment_5_stability(num_runs=5, num_rounds=300, verbose=True):
    """实验 5：稳定性验证 — 多次运行"""
    print("\n" + "=" * 60)
    print("实验 5：稳定性验证")
    print("=" * 60)

    all_types = {'visual_ambiguous', 'crossmodal', 'subset', 'causal', 'confidence'}

    success_rates = []
    category_counts = []
    vocabulary_sizes = []

    for run in range(num_runs):
        game = UnifiedCommunicationGame()
        for _ in range(num_rounds):
            scene = generate_unified_scenario(
                ambiguity_types=all_types, num_objects=3,
                speaker_confidence=random.uniform(0.3, 0.95),
            )
            game.play_round(scene)

        stats = game.get_stats()
        success_rates.append(stats['success_rate'])
        category_counts.append(stats['num_categories'])
        vocabulary_sizes.append(stats['vocabulary_size'])

        if verbose:
            print(f"  运行 {run+1}: 成功率={stats['success_rate']:.1%}, "
                  f"类别数={stats['num_categories']}, "
                  f"词汇量={stats['vocabulary_size']}")

    avg_success = sum(success_rates) / len(success_rates)
    avg_categories = sum(category_counts) / len(category_counts)
    avg_vocab = sum(vocabulary_sizes) / len(vocabulary_sizes)

    if verbose:
        print(f"\n  平均成功率: {avg_success:.1%}")
        print(f"  平均类别数: {avg_categories:.1f}")
        print(f"  平均词汇量: {avg_vocab:.1f}")

    return {
        'avg_success_rate': avg_success,
        'avg_categories': avg_categories,
        'avg_vocabulary': avg_vocab,
        'all_success_rates': success_rates,
    }


def run_all_experiments():
    """运行全部 5 个实验"""
    print("=" * 60)
    print("Phase 24: 统一语言系统实验")
    print("=" * 60)

    random.seed(42)

    r1 = experiment_1_single_module()
    r2 = experiment_2_dual_combination()
    r3 = experiment_3_full_combination()
    r4 = experiment_4_comparison()
    r5 = experiment_5_stability()

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)

    print("\n单模块基线:")
    for name, stats in r1.items():
        print(f"  {name:20s}: {stats['success_rate']:.1%}")

    print("\n双模块组合:")
    for name, stats in r2.items():
        print(f"  {name:25s}: {stats['success_rate']:.1%}")

    print(f"\n全模块组合: {r3['success_rate']:.1%} ({r3['num_categories']} 类符号)")
    print(f"组合 vs 基线: {r4['improvement']:+.1%}")
    print(f"稳定性: {r5['avg_success_rate']:.1%} ± {max(r5['all_success_rates']) - min(r5['all_success_rates']):.1%}")

    return {
        'single_module': r1,
        'dual_combination': r2,
        'full_combination': r3,
        'comparison': r4,
        'stability': r5,
    }


if __name__ == '__main__':
    run_all_experiments()
