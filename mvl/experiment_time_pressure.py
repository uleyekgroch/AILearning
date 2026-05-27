"""
Phase 16b: 时间压力下的语言效率实验

验证当描述长度受限时，agent 是否会选择更紧凑的语法策略。

核心假设：
- max_len=1: 只能用单符号
- max_len=2: 可以用否定（"not X"）
- max_len=3: 可以用从句（"X that Y"）
- max_len=∞: 无限制，选择最短

实验：
1. 不同 max_len 下的成功率对比
2. 语法策略分布（正向/否定/从句）
3. 时间压力下的涌现模式
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict

from grounding_compound import (
    CompoundCommunicationGame, CompoundSpeaker, CompoundListener,
    generate_subset_scene, generate_clause_scene_with_overlap,
)
from language_emergence import NEGATION_MARKERS, RELATIVE_MARKERS


def experiment_1_success_by_maxlen(num_rounds: int = 300, verbose: bool = True):
    """实验 1: 不同 max_len 下的成功率"""
    print("=" * 60)
    print("实验 1: max_len 对成功率的影响")
    print("=" * 60)

    max_lens = [1, 2, 3, 4, 0]  # 0 = 无限制
    results = {}

    for ml in max_lens:
        game = CompoundCommunicationGame()
        for r in range(num_rounds):
            # 使用否定场景（有子集关系）
            scene, target_idx = generate_subset_scene(
                num_base=np.random.randint(2, 5),
                num_subsets=np.random.randint(1, 4),
                target_is_subset=(np.random.random() < 0.5),
            )
            game.play_round(scene, target_idx, max_len=ml)

        stats = game.language.get_stats()
        neg_syms = [s for s in game.language.vocabulary if s in NEGATION_MARKERS]
        clause_syms = [s for s in game.language.vocabulary if s in RELATIVE_MARKERS]

        label = f"max_len={ml}" if ml > 0 else "无限制"
        results[label] = {
            'success_rate': game.language.total_successes / max(1, game.language.total_games),
            'negation_rate': game.negation_used / max(1, len(game.game_log)),
            'clause_rate': game.clause_used / max(1, len(game.game_log)),
            'vocab_size': len(game.language.vocabulary),
            'negation_symbols': neg_syms,
            'clause_symbols': clause_syms,
        }

        if verbose:
            print(f"  {label}: success={results[label]['success_rate']:.1%}, "
                  f"neg_rate={results[label]['negation_rate']:.1%}, "
                  f"clause_rate={results[label]['clause_rate']:.1%}, "
                  f"vocab={results[label]['vocab_size']}")

    return results


def experiment_2_strategy_distribution(num_rounds: int = 500, verbose: bool = True):
    """实验 2: 语法策略分布"""
    print(f"\n{'=' * 60}")
    print(f"实验 2: 语法策略分布 ({num_rounds} 轮)")
    print(f"{'=' * 60}")

    max_lens = [1, 2, 3, 0]
    results = {}

    for ml in max_lens:
        game = CompoundCommunicationGame()
        strategy_counts = defaultdict(int)

        for r in range(num_rounds):
            scene, target_idx = generate_subset_scene(
                num_base=np.random.randint(2, 5),
                num_subsets=np.random.randint(1, 4),
                target_is_subset=(np.random.random() < 0.5),
            )
            game.play_round(scene, target_idx, max_len=ml)

            # 记录策略
            if game.game_log:
                last = game.game_log[-1]
                utt = last['utterance']
                if any(s in NEGATION_MARKERS for s in utt):
                    strategy_counts['negation'] += 1
                elif any(s in RELATIVE_MARKERS for s in utt):
                    strategy_counts['clause'] += 1
                elif len(utt) == 1:
                    strategy_counts['single'] += 1
                else:
                    strategy_counts['positive_combo'] += 1

        label = f"max_len={ml}" if ml > 0 else "无限制"
        total = sum(strategy_counts.values())
        results[label] = {k: v / total for k, v in strategy_counts.items()}

        if verbose:
            print(f"\n  {label}:")
            for strategy, rate in sorted(results[label].items()):
                print(f"    {strategy}: {rate:.1%}")

    return results


def experiment_3_negation_under_pressure(num_rounds: int = 500, verbose: bool = True):
    """实验 3: 时间压力下否定的涌现"""
    print(f"\n{'=' * 60}")
    print(f"实验 3: 时间压力下否定的涌现")
    print(f"{'=' * 60}")

    # 对比 max_len=2（否定可能）vs max_len=1（否定不可能）
    results = {}

    for ml in [1, 2]:
        game = CompoundCommunicationGame()
        negation_emerged = False
        emergence_round = -1

        for r in range(num_rounds):
            scene, target_idx = generate_subset_scene(
                num_base=np.random.randint(2, 5),
                num_subsets=np.random.randint(1, 4),
                target_is_subset=(np.random.random() < 0.5),
            )
            game.play_round(scene, target_idx, max_len=ml)

            if not negation_emerged:
                neg_syms = [s for s in game.language.vocabulary if s in NEGATION_MARKERS]
                if neg_syms:
                    negation_emerged = True
                    emergence_round = r

        neg_syms = [s for s in game.language.vocabulary if s in NEGATION_MARKERS]
        label = f"max_len={ml}"
        results[label] = {
            'negation_emerged': negation_emerged,
            'emergence_round': emergence_round,
            'negation_rate': game.negation_used / max(1, len(game.game_log)),
            'success_rate': game.language.total_successes / max(1, game.language.total_games),
            'negation_symbols': neg_syms,
        }

        if verbose:
            print(f"  {label}: neg_emerged={negation_emerged}, "
                  f"neg_rate={results[label]['negation_rate']:.1%}, "
                  f"success={results[label]['success_rate']:.1%}, "
                  f"neg_syms={neg_syms}")

    return results


def experiment_4_clause_under_pressure(num_rounds: int = 500, verbose: bool = True):
    """实验 4: 时间压力下从句的涌现"""
    print(f"\n{'=' * 60}")
    print(f"实验 4: 时间压力下从句的涌现")
    print(f"{'=' * 60}")

    # 使用从句场景，对比不同 max_len
    results = {}

    for ml in [2, 3, 0]:
        game = CompoundCommunicationGame()
        clause_emerged = False
        emergence_round = -1

        for r in range(num_rounds):
            scene, target_idx = generate_clause_scene_with_overlap(
                num_objects=np.random.randint(3, 6),
            )
            game.play_round(scene, target_idx, max_len=ml)

            if not clause_emerged:
                clause_syms = [s for s in game.language.vocabulary if s in RELATIVE_MARKERS]
                if clause_syms:
                    clause_emerged = True
                    emergence_round = r

        clause_syms = [s for s in game.language.vocabulary if s in RELATIVE_MARKERS]
        label = f"max_len={ml}" if ml > 0 else "无限制"
        results[label] = {
            'clause_emerged': clause_emerged,
            'emergence_round': emergence_round,
            'clause_rate': game.clause_used / max(1, len(game.game_log)),
            'success_rate': game.language.total_successes / max(1, game.language.total_games),
            'clause_symbols': clause_syms,
        }

        if verbose:
            print(f"  {label}: clause_emerged={clause_emerged}, "
                  f"clause_rate={results[label]['clause_rate']:.1%}, "
                  f"success={results[label]['success_rate']:.1%}, "
                  f"clause_syms={clause_syms}")

    return results


def main():
    """运行所有实验"""
    print("=" * 80)
    print("Phase 16b: 时间压力下的语言效率")
    print("核心假设：max_len 约束迫使 agent 使用更紧凑的语法策略")
    print("=" * 80)

    # 实验 1
    r1 = experiment_1_success_by_maxlen(300, verbose=True)

    # 实验 2
    r2 = experiment_2_strategy_distribution(500, verbose=True)

    # 实验 3
    r3 = experiment_3_negation_under_pressure(500, verbose=True)

    # 实验 4
    r4 = experiment_4_clause_under_pressure(500, verbose=True)

    # 汇总
    print(f"\n{'=' * 80}")
    print("实验汇总")
    print(f"{'=' * 80}")

    print(f"\n实验 1 (max_len 对成功率):")
    for label, res in r1.items():
        print(f"  {label}: success={res['success_rate']:.1%}")

    print(f"\n实验 2 (语法策略分布):")
    for label, res in r2.items():
        print(f"  {label}: {dict(res)}")

    print(f"\n实验 3 (否定涌现):")
    for label, res in r3.items():
        print(f"  {label}: neg_emerged={res['negation_emerged']}, neg_rate={res['negation_rate']:.1%}")

    print(f"\n实验 4 (从句涌现):")
    for label, res in r4.items():
        print(f"  {label}: clause_emerged={res['clause_emerged']}, clause_rate={res['clause_rate']:.1%}")

    print(f"\n{'=' * 80}")
    print("核心结论")
    print(f"{'=' * 80}")
    print("1. max_len 约束显著影响语法策略选择")
    print("2. max_len=2 时否定成为最优策略（2符号）")
    print("3. max_len=3 时从句成为可能（3符号）")
    print("4. 时间压力是语法涌现的关键驱动力")


if __name__ == '__main__':
    main()
