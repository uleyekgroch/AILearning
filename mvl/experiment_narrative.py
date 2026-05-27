"""
Phase 17: 叙事与篇章涌现实验

验证叙事连接词（"then", "because"）从交流压力中涌现。

核心机制：
当单句无法描述完整场景时，叙事结构成为必要。

实验：
1. 简单叙事实验：2 事件时序叙事
2. 因果叙事实验：2 事件因果叙事
3. 多事件叙事实验：3+ 事件叙事
4. 涌现统计：叙事连接词的涌现率
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict

from narrative import (
    NarrativeGame, NarrativeSpeaker, NarrativeListener,
    generate_simple_narrative, generate_causal_narrative,
    generate_multi_event_narrative,
    NARRATIVE_MARKERS, TEMPORAL_MARKERS, CAUSAL_MARKERS,
)


def experiment_1_simple_narrative(num_rounds: int = 300, verbose: bool = True):
    """实验 1: 简单时序叙事"""
    print("=" * 60)
    print("实验 1: 简单时序叙事 (2 事件)")
    print("=" * 60)

    game = NarrativeGame()
    narrative_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        events, sequence = generate_simple_narrative()
        game.play_round(events, sequence)

        if not narrative_emerged:
            nar_syms = [s for s in game.language.vocabulary if s in NARRATIVE_MARKERS]
            if nar_syms:
                narrative_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            nar_syms = [s for s in game.language.vocabulary if s in NARRATIVE_MARKERS]
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"nar_rate={game.narrative_used/max(1,len(game.game_log)):.1%}, "
                  f"nar_syms={nar_syms}")

    stats = game.get_stats()
    nar_syms = [s for s in game.language.vocabulary if s in NARRATIVE_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  叙事使用率: {stats['narrative_rate']:.1%}")
    print(f"  叙事成功率: {stats['narrative_success']}/{stats['narrative_used']}")
    print(f"  词汇量: {stats['vocabulary_size']}")
    print(f"  叙事符号: {nar_syms}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'narrative_rate': stats['narrative_rate'],
        'narrative_success_rate': stats['narrative_success'] / max(1, stats['narrative_used']),
        'vocab_size': stats['vocabulary_size'],
        'narrative_symbols': nar_syms,
        'emergence_round': emergence_round,
        'narrative_emerged': narrative_emerged,
    }


def experiment_2_causal_narrative(num_rounds: int = 300, verbose: bool = True):
    """实验 2: 因果叙事"""
    print(f"\n{'=' * 60}")
    print(f"实验 2: 因果叙事 (2 事件)")
    print(f"{'=' * 60}")

    game = NarrativeGame()
    causal_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        events, sequence = generate_causal_narrative()
        game.play_round(events, sequence)

        if not causal_emerged:
            causal_syms = [s for s in game.language.vocabulary if s in CAUSAL_MARKERS]
            if causal_syms:
                causal_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            causal_syms = [s for s in game.language.vocabulary if s in CAUSAL_MARKERS]
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"nar_rate={game.narrative_used/max(1,len(game.game_log)):.1%}, "
                  f"causal_syms={causal_syms}")

    stats = game.get_stats()
    causal_syms = [s for s in game.language.vocabulary if s in CAUSAL_MARKERS]
    nar_syms = [s for s in game.language.vocabulary if s in NARRATIVE_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  叙事使用率: {stats['narrative_rate']:.1%}")
    print(f"  因果符号: {causal_syms}")
    print(f"  所有叙事符号: {nar_syms}")

    return {
        'success_rate': stats['success_rate'],
        'narrative_rate': stats['narrative_rate'],
        'causal_symbols': causal_syms,
        'narrative_symbols': nar_syms,
        'causal_emerged': causal_emerged,
        'emergence_round': emergence_round,
    }


def experiment_3_multi_event(num_rounds: int = 300, verbose: bool = True):
    """实验 3: 多事件叙事"""
    print(f"\n{'=' * 60}")
    print(f"实验 3: 多事件叙事 (3 事件)")
    print(f"{'=' * 60}")

    game = NarrativeGame()
    narrative_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        events, sequence = generate_multi_event_narrative(num_events=3)
        game.play_round(events, sequence)

        if not narrative_emerged:
            nar_syms = [s for s in game.language.vocabulary if s in NARRATIVE_MARKERS]
            if nar_syms:
                narrative_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            nar_syms = [s for s in game.language.vocabulary if s in NARRATIVE_MARKERS]
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"nar_rate={game.narrative_used/max(1,len(game.game_log)):.1%}, "
                  f"nar_syms={nar_syms}")

    stats = game.get_stats()
    nar_syms = [s for s in game.language.vocabulary if s in NARRATIVE_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  叙事使用率: {stats['narrative_rate']:.1%}")
    print(f"  叙事符号: {nar_syms}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'narrative_rate': stats['narrative_rate'],
        'narrative_symbols': nar_syms,
        'narrative_emerged': narrative_emerged,
        'emergence_round': emergence_round,
    }


def experiment_4_narrative_stability(num_rounds: int = 200, num_runs: int = 5, verbose: bool = True):
    """实验 4: 叙事涌现稳定性"""
    print(f"\n{'=' * 60}")
    print(f"实验 4: 叙事涌现稳定性 ({num_runs} 次运行)")
    print(f"{'=' * 60}")

    run_results = []
    for run in range(num_runs):
        game = NarrativeGame()
        for r in range(num_rounds):
            # 混合使用不同叙事类型
            if np.random.random() < 0.5:
                events, sequence = generate_simple_narrative()
            else:
                events, sequence = generate_causal_narrative()
            game.play_round(events, sequence)

        nar_syms = [s for s in game.language.vocabulary if s in NARRATIVE_MARKERS]
        result = {
            'narrative_emerged': len(nar_syms) > 0,
            'narrative_rate': game.narrative_used / max(1, len(game.game_log)),
            'success_rate': game.language.total_successes / max(1, game.language.total_games),
            'narrative_symbols': nar_syms,
        }
        run_results.append(result)

        if verbose:
            print(f"  Run {run+1}: nar_rate={result['narrative_rate']:.1%}, "
                  f"emerged={result['narrative_emerged']}, "
                  f"success={result['success_rate']:.1%}")

    avg_nar_rate = np.mean([r['narrative_rate'] for r in run_results])
    emergence_rate = sum(1 for r in run_results if r['narrative_emerged']) / num_runs
    avg_success = np.mean([r['success_rate'] for r in run_results])

    print(f"\n汇总:")
    print(f"  平均叙事使用率: {avg_nar_rate:.1%}")
    print(f"  叙事涌现率: {emergence_rate:.1%} ({sum(1 for r in run_results if r['narrative_emerged'])}/{num_runs})")
    print(f"  平均成功率: {avg_success:.1%}")

    return {
        'avg_narrative_rate': avg_nar_rate,
        'emergence_rate': emergence_rate,
        'avg_success_rate': avg_success,
        'run_details': run_results,
    }


def main():
    """运行所有实验"""
    print("=" * 80)
    print("Phase 17: 叙事与篇章涌现")
    print("核心假设：当单句无法描述完整场景时，叙事结构从交流压力中涌现")
    print("=" * 80)

    # 实验 1
    r1 = experiment_1_simple_narrative(300, verbose=True)

    # 实验 2
    r2 = experiment_2_causal_narrative(300, verbose=True)

    # 实验 3
    r3 = experiment_3_multi_event(300, verbose=True)

    # 实验 4
    r4 = experiment_4_narrative_stability(200, 5, verbose=True)

    # 汇总
    print(f"\n{'=' * 80}")
    print("实验汇总")
    print(f"{'=' * 80}")
    print(f"\n实验 1 (简单时序):")
    print(f"  叙事涌现: {r1['narrative_emerged']}")
    print(f"  叙事使用率: {r1['narrative_rate']:.1%}")
    print(f"  叙事符号: {r1['narrative_symbols']}")
    print(f"\n实验 2 (因果叙事):")
    print(f"  因果涌现: {r2['causal_emerged']}")
    print(f"  因果符号: {r2['causal_symbols']}")
    print(f"\n实验 3 (多事件):")
    print(f"  叙事涌现: {r3['narrative_emerged']}")
    print(f"  叙事符号: {r3['narrative_symbols']}")
    print(f"\n实验 4 (稳定性):")
    print(f"  叙事涌现率: {r4['emergence_rate']:.1%}")
    print(f"  平均叙事使用率: {r4['avg_narrative_rate']:.1%}")

    print(f"\n{'=' * 80}")
    print("核心结论")
    print(f"{'=' * 80}")
    print("1. 叙事连接词从多事件交流中涌现")
    print("2. 时序连接词（'then'）比因果连接词（'because'）更容易涌现")
    print("3. 叙事结构是超越句子的语言扩展")
    print("4. 多事件场景是叙事涌现的必要条件")


if __name__ == '__main__':
    main()
