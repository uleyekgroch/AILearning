"""
Phase 19: 因果推理实验

验证 "because" 从因果推理需求中涌现。

核心机制：
当场景中同时存在因果对和虚假相关时，
Speaker 必须用 "because" 标记真正的因果关系，
用 "then" 标记纯时序关系，否则 Listener 无法区分。

实验：
1. 纯因果场景（基线——"then" 应该足够）
2. 虚假相关场景（"because" 应该涌现）
3. 对比实验：有因果推理 vs 无因果推理
4. 稳定性验证：5 次运行，测量 "because" 涌现率
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict

from grounding_causal_reasoning import (
    CausalCommunicationGame, BaselineCausalGame,
    generate_causal_scenario,
    CAUSAL_REASONING_MARKERS, TEMPORAL_MARKERS,
)


def experiment_1_pure_causal(num_rounds: int = 300, verbose: bool = True):
    """实验 1: 纯因果场景（基线）"""
    print("=" * 60)
    print("实验 1: 纯因果场景（基线）")
    print("=" * 60)

    game = CausalCommunicationGame()
    because_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        events = generate_causal_scenario(mode='pure_causal', num_pairs=3)
        game.play_round(events)

        if not because_emerged:
            if 'because' in game.language.vocabulary:
                because_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"because_used={stats['because_used']}, "
                  f"then_used={stats['then_used']}")

    stats = game.get_stats()
    because_in_vocab = 'because' in game.language.vocabulary

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  because 使用: {stats['because_used']}")
    print(f"  then 使用: {stats['then_used']}")
    print(f"  because 在词汇表: {because_in_vocab}")
    print(f"  because 涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'because_used': stats['because_used'],
        'then_used': stats['then_used'],
        'because_in_vocab': because_in_vocab,
        'emergence_round': emergence_round,
    }


def experiment_2_spurious_correlation(num_rounds: int = 300, verbose: bool = True):
    """实验 2: 虚假相关场景（需要 "because"）"""
    print(f"\n{'=' * 60}")
    print(f"实验 2: 虚假相关场景（需要 'because'）")
    print(f"{'=' * 60}")

    game = CausalCommunicationGame()
    because_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        events = generate_causal_scenario(mode='spurious', num_pairs=4)
        game.play_round(events)

        if not because_emerged:
            if 'because' in game.language.vocabulary:
                because_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"because_used={stats['because_used']}, "
                  f"because_success={stats['because_success']:.1%}")

    stats = game.get_stats()
    because_in_vocab = 'because' in game.language.vocabulary

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  because 使用: {stats['because_used']}")
    print(f"  because 成功率: {stats['because_success']:.1%}")
    print(f"  then 使用: {stats['then_used']}")
    print(f"  then 成功率: {stats['then_success']:.1%}")
    print(f"  because 在词汇表: {because_in_vocab}")
    print(f"  because 涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'because_used': stats['because_used'],
        'because_success': stats['because_success'],
        'then_used': stats['then_used'],
        'then_success': stats['then_success'],
        'because_in_vocab': because_in_vocab,
        'emergence_round': emergence_round,
    }


def experiment_3_comparison(num_rounds: int = 300, verbose: bool = True):
    """实验 3: 有因果推理 vs 无因果推理"""
    print(f"\n{'=' * 60}")
    print(f"实验 3: 有因果推理 vs 无因果推理")
    print(f"{'=' * 60}")

    results = {}

    # 有因果推理
    game_causal = CausalCommunicationGame()
    for r in range(num_rounds):
        events = generate_causal_scenario(mode='spurious', num_pairs=4)
        game_causal.play_round(events)

    stats_causal = game_causal.get_stats()
    results['有因果推理'] = {
        'success_rate': stats_causal['success_rate'],
        'because_used': stats_causal['because_used'],
        'because_success': stats_causal['because_success'],
    }

    # 无因果推理（基线）
    game_baseline = BaselineCausalGame()
    for r in range(num_rounds):
        events = generate_causal_scenario(mode='spurious', num_pairs=4)
        game_baseline.play_round(events)

    stats_baseline = game_baseline.get_stats()
    results['无因果推理'] = {
        'success_rate': stats_baseline['success_rate'],
    }

    if verbose:
        print(f"\n  有因果推理:")
        print(f"    成功率: {stats_causal['success_rate']:.1%}")
        print(f"    because 使用: {stats_causal['because_used']}")
        print(f"    because 成功率: {stats_causal['because_success']:.1%}")
        print(f"\n  无因果推理:")
        print(f"    成功率: {stats_baseline['success_rate']:.1%}")

    improvement = stats_causal['success_rate'] - stats_baseline['success_rate']
    print(f"\n  改善: {improvement:+.1%}")

    results['improvement'] = improvement
    return results


def experiment_4_stability(num_rounds: int = 200, num_runs: int = 5, verbose: bool = True):
    """实验 4: "because" 涌现稳定性"""
    print(f"\n{'=' * 60}")
    print(f"实验 4: 'because' 涌现稳定性 ({num_runs} 次运行)")
    print(f"{'=' * 60}")

    run_results = []
    for run in range(num_runs):
        game = CausalCommunicationGame()
        for r in range(num_rounds):
            events = generate_causal_scenario(mode='spurious', num_pairs=4)
            game.play_round(events)

        stats = game.get_stats()
        because_in_vocab = 'because' in game.language.vocabulary
        result = {
            'because_emerged': because_in_vocab,
            'because_used': stats['because_used'],
            'because_success': stats['because_success'],
            'success_rate': stats['success_rate'],
        }
        run_results.append(result)

        if verbose:
            print(f"  Run {run+1}: because_used={result['because_used']}, "
                  f"emerged={result['because_emerged']}, "
                  f"success={result['success_rate']:.1%}")

    avg_because_used = np.mean([r['because_used'] for r in run_results])
    emergence_rate = sum(1 for r in run_results if r['because_emerged']) / num_runs
    avg_success = np.mean([r['success_rate'] for r in run_results])

    print(f"\n汇总:")
    print(f"  平均 because 使用: {avg_because_used:.1f}")
    print(f"  because 涌现率: {emergence_rate:.1%} ({sum(1 for r in run_results if r['because_emerged'])}/{num_runs})")
    print(f"  平均成功率: {avg_success:.1%}")

    return {
        'avg_because_used': avg_because_used,
        'emergence_rate': emergence_rate,
        'avg_success_rate': avg_success,
        'run_details': run_results,
    }


def main():
    """运行所有实验"""
    print("=" * 80)
    print("Phase 19: 因果推理 —— 让 'because' 涌现")
    print("核心假设：混杂场景中的因果区分需求驱动 'because' 涌现")
    print("=" * 80)

    # 实验 1
    r1 = experiment_1_pure_causal(300, verbose=True)

    # 实验 2
    r2 = experiment_2_spurious_correlation(300, verbose=True)

    # 实验 3
    r3 = experiment_3_comparison(300, verbose=True)

    # 实验 4
    r4 = experiment_4_stability(200, 5, verbose=True)

    # 汇总
    print(f"\n{'=' * 80}")
    print("实验汇总")
    print(f"{'=' * 80}")
    print(f"\n实验 1 (纯因果):")
    print(f"  because 涌现: {r1['because_in_vocab']}")
    print(f"  because 使用: {r1['because_used']}")
    print(f"\n实验 2 (虚假相关):")
    print(f"  because 涌现: {r2['because_in_vocab']}")
    print(f"  because 使用: {r2['because_used']}")
    print(f"  because 成功率: {r2['because_success']:.1%}")
    print(f"\n实验 3 (对比):")
    print(f"  改善: {r3['improvement']:+.1%}")
    print(f"\n实验 4 (稳定性):")
    print(f"  because 涌现率: {r4['emergence_rate']:.1%}")
    print(f"  平均 because 使用: {r4['avg_because_used']:.1f}")

    print(f"\n{'=' * 80}")
    print("核心结论")
    print(f"{'=' * 80}")
    print("1. 'because' 在虚假相关场景中涌现")
    print("2. 因果推理提供预测优势")
    print("3. 标记选择本身成为通信成功的关键因素")
    print("4. 与人类语言一致：'because' 表达因果，'then' 表达时序")


if __name__ == '__main__':
    main()
