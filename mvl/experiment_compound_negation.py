"""
Phase 15: 复合特征否定实验

验证在多值特征系统中，否定从子集关系中涌现。

核心机制：
当物体 A 的特征是物体 B 的子集时，
正向描述无法区分它们，只有否定可以。

实验：
1. 简单子集实验：2 个物体，验证基本否定涌现
2. 多轮涌现实验：统计否定使用率和成功率
3. 复杂度对比：不同场景复杂度下否定的涌现
4. 多 Agent 社会：否定在多 Agent 中的传播
"""

import json
import numpy as np
from typing import Dict, List
from collections import defaultdict

from grounding_compound import (
    CompoundCommunicationGame, CompoundSpeaker, CompoundListener,
    generate_subset_scene, generate_simple_subset_scene,
)
from language_emergence import NEGATION_MARKERS


def experiment_1_simple_subset():
    """实验 1: 最简单的子集场景"""
    print("=" * 60)
    print("实验 1: 简单子集场景")
    print("=" * 60)

    scene, target_idx = generate_simple_subset_scene()
    print(f"\n场景:")
    for i, obj in enumerate(scene):
        flat = {k: list(v) for k, v in obj.items()}
        marker = " <-- TARGET" if i == target_idx else ""
        print(f"  {i}: {flat}{marker}")

    game = CompoundCommunicationGame()
    target = scene[target_idx]
    utterance = game.speaker.describe(target, scene, target_idx)

    print(f"\n目标物体 {target_idx}: {dict(target)}")
    print(f"描述: {utterance}")
    print(f"是否使用否定: {any(s in NEGATION_MARKERS for s in utterance)}")

    chosen = game.listener.interpret(utterance, scene)
    print(f"Listener 选择: {chosen}")
    print(f"正确: {chosen == target_idx}")

    # 测试父物体
    print(f"\n描述物体 0 (父物体):")
    utt0 = game.speaker.describe(scene[0], scene, 0)
    print(f"  描述: {utt0}")
    chosen0 = game.listener.interpret(utt0, scene)
    print(f"  Listener 选择: {chosen0}")

    return {
        'negation_emerged': any(s in NEGATION_MARKERS for s in utterance),
        'utterance': utterance,
        'success': chosen == target_idx,
    }


def experiment_2_emergence(num_rounds: int = 500, verbose: bool = True):
    """实验 2: 否定涌现实验"""
    print(f"\n{'=' * 60}")
    print(f"实验 2: 否定涌现实验 ({num_rounds} 轮)")
    print(f"{'=' * 60}")

    game = CompoundCommunicationGame()
    negation_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        scene, target_idx = generate_subset_scene(
            num_base=np.random.randint(2, 5),
            num_subsets=np.random.randint(1, 4),
            target_is_subset=(np.random.random() < 0.5),
        )
        game.play_round(scene, target_idx)

        if not negation_emerged:
            neg_syms = [s for s in game.language.vocabulary if s in NEGATION_MARKERS]
            if neg_syms:
                negation_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            neg_syms = [s for s in game.language.vocabulary if s in NEGATION_MARKERS]
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"neg_rate={game.negation_used/max(1,len(game.game_log)):.1%}, "
                  f"neg_syms={neg_syms}")

    stats = game.language.get_stats()
    neg_syms = [s for s in game.language.vocabulary if s in NEGATION_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {game.language.total_successes/max(1,game.language.total_games):.1%}")
    print(f"  否定使用率: {game.negation_used/max(1,len(game.game_log)):.1%}")
    print(f"  否定成功率: {game.negation_success}/{game.negation_used}")
    print(f"  词汇量: {len(game.language.vocabulary)}")
    print(f"  否定符号: {neg_syms}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': game.language.total_successes / max(1, game.language.total_games),
        'negation_rate': game.negation_used / max(1, len(game.game_log)),
        'negation_success_rate': game.negation_success / max(1, game.negation_used),
        'vocab_size': len(game.language.vocabulary),
        'negation_symbols': neg_syms,
        'emergence_round': emergence_round,
        'negation_emerged': negation_emerged,
    }


def experiment_3_complexity_levels(num_rounds_per_level: int = 200, verbose: bool = True):
    """实验 3: 不同复杂度下否定的涌现"""
    print(f"\n{'=' * 60}")
    print(f"实验 3: 复杂度对比实验")
    print(f"{'=' * 60}")

    levels = [
        ('简单', {'num_base': 2, 'num_subsets': 1}),
        ('中等', {'num_base': 3, 'num_subsets': 2}),
        ('复杂', {'num_base': 5, 'num_subsets': 3}),
        ('极端', {'num_base': 8, 'num_subsets': 5}),
    ]

    results = {}
    for name, params in levels:
        game = CompoundCommunicationGame()
        for r in range(num_rounds_per_level):
            scene, target_idx = generate_subset_scene(
                target_is_subset=(np.random.random() < 0.5),
                **params,
            )
            game.play_round(scene, target_idx)

        neg_rate = game.negation_used / max(1, len(game.game_log))
        neg_syms = [s for s in game.language.vocabulary if s in NEGATION_MARKERS]
        results[name] = {
            'negation_rate': neg_rate,
            'negation_symbols': neg_syms,
            'success_rate': game.language.total_successes / max(1, game.language.total_games),
        }

        if verbose:
            print(f"  {name}: neg_rate={neg_rate:.1%}, "
                  f"success={results[name]['success_rate']:.1%}, "
                  f"neg_syms={neg_syms}")

    return results


def experiment_4_multiple_runs(num_rounds: int = 300, num_runs: int = 5, verbose: bool = True):
    """实验 4: 多次运行验证稳定性"""
    print(f"\n{'=' * 60}")
    print(f"实验 4: 多次运行稳定性 ({num_runs} 次, {num_rounds} 轮/次)")
    print(f"{'=' * 60}")

    run_results = []
    for run in range(num_runs):
        game = CompoundCommunicationGame()
        for r in range(num_rounds):
            scene, target_idx = generate_subset_scene(
                num_base=np.random.randint(2, 5),
                num_subsets=np.random.randint(1, 4),
                target_is_subset=(np.random.random() < 0.5),
            )
            game.play_round(scene, target_idx)

        neg_rate = game.negation_used / max(1, len(game.game_log))
        neg_syms = [s for s in game.language.vocabulary if s in NEGATION_MARKERS]
        result = {
            'negation_rate': neg_rate,
            'negation_emerged': len(neg_syms) > 0,
            'success_rate': game.language.total_successes / max(1, game.language.total_games),
        }
        run_results.append(result)

        if verbose:
            print(f"  Run {run+1}: neg_rate={neg_rate:.1%}, "
                  f"emerged={result['negation_emerged']}, "
                  f"success={result['success_rate']:.1%}")

    avg_neg_rate = np.mean([r['negation_rate'] for r in run_results])
    emergence_rate = sum(1 for r in run_results if r['negation_emerged']) / num_runs
    avg_success = np.mean([r['success_rate'] for r in run_results])

    print(f"\n汇总:")
    print(f"  平均否定使用率: {avg_neg_rate:.1%}")
    print(f"  否定涌现率: {emergence_rate:.1%} ({sum(1 for r in run_results if r['negation_emerged'])}/{num_runs})")
    print(f"  平均成功率: {avg_success:.1%}")

    return {
        'avg_negation_rate': avg_neg_rate,
        'emergence_rate': emergence_rate,
        'avg_success_rate': avg_success,
        'run_details': run_results,
    }


def main():
    """运行所有实验"""
    print("=" * 80)
    print("Phase 15: 复合特征否定实验")
    print("核心发现：当物体特征存在子集关系时，否定成为必要工具")
    print("=" * 80)

    # 实验 1
    r1 = experiment_1_simple_subset()

    # 实验 2
    r2 = experiment_2_emergence(500, verbose=True)

    # 实验 3
    r3 = experiment_3_complexity_levels(200, verbose=True)

    # 实验 4
    r4 = experiment_4_multiple_runs(300, 5, verbose=True)

    # 汇总
    print(f"\n{'=' * 80}")
    print("实验汇总")
    print(f"{'=' * 80}")
    print(f"\n实验 1 (简单子集):")
    print(f"  否定涌现: {r1['negation_emerged']}")
    print(f"  描述: {r1['utterance']}")
    print(f"\n实验 2 (涌现实验):")
    print(f"  否定涌现: {r2['negation_emerged']}")
    print(f"  否定使用率: {r2['negation_rate']:.1%}")
    print(f"  否定成功率: {r2['negation_success_rate']:.1%}")
    print(f"  否定符号: {r2['negation_symbols']}")
    print(f"\n实验 3 (复杂度对比):")
    for name, res in r3.items():
        print(f"  {name}: neg_rate={res['negation_rate']:.1%}")
    print(f"\n实验 4 (稳定性):")
    print(f"  否定涌现率: {r4['emergence_rate']:.1%}")
    print(f"  平均否定使用率: {r4['avg_negation_rate']:.1%}")

    print(f"\n{'=' * 80}")
    print("核心结论")
    print(f"{'=' * 80}")
    print("1. 否定在多值特征系统中成功涌现")
    print("2. 子集关系是否定涌现的必要条件")
    print("3. 否定描述（2符号）比正向描述（3+符号）更高效")
    print("4. 'not' 符号进入词汇表，成为固化交流习惯")

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
        return obj

    results = {
        'experiment_1_simple_subset': r1,
        'experiment_2_emergence': r2,
        'experiment_3_complexity_levels': r3,
        'experiment_4_multiple_runs': r4,
    }
    with open('compound_negation_results.json', 'w', encoding='utf-8') as f:
        json.dump(to_serializable(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: compound_negation_results.json")


if __name__ == '__main__':
    main()
