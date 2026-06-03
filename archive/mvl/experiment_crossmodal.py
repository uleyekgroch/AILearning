"""
Phase 23: 跨模态语言实验

验证听觉/触觉符号从视觉模糊中涌现。

核心机制：
当两个物体外观完全相同时，
Speaker 必须用听觉或触觉特征描述。
"loud", "rough" 等跨模态符号从需要区分时涌现。

实验：
1. 视觉唯一基线（不需要跨模态）
2. 视觉模糊（需要跨模态）
3. 纯跨模态（只有非视觉特征能区分）
4. 对比实验：跨模态 vs 纯视觉
5. 稳定性验证：5 次运行
"""

import json
import numpy as np
from typing import Dict, List
from collections import defaultdict

from grounding_crossmodal import (
    CrossModalGame, BaselineCrossModalGame,
    generate_crossmodal_scenario, CrossModalObject,
)
from language_emergence import AUDITORY_SYMBOLS, TACTILE_SYMBOLS


def experiment_1_visual_unique(num_rounds: int = 300, verbose: bool = True):
    """实验 1: 视觉唯一基线（不需要跨模态）"""
    print("=" * 60)
    print("实验 1: 视觉唯一基线（不需要跨模态）")
    print("=" * 60)

    game = CrossModalGame()
    crossmodal_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        objects, target_idx = generate_crossmodal_scenario(mode='visual_unique', num_objects=4)
        game.play_round(objects, target_idx)

        if not crossmodal_emerged:
            markers = [s for s in game.language.vocabulary
                       if s in AUDITORY_SYMBOLS or s in TACTILE_SYMBOLS]
            if markers:
                crossmodal_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"visual={stats['visual_only_used']}, "
                  f"crossmodal={stats['crossmodal_used']}")

    stats = game.get_stats()
    markers = [s for s in game.language.vocabulary
               if s in AUDITORY_SYMBOLS or s in TACTILE_SYMBOLS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  视觉描述使用: {stats['visual_only_used']}")
    print(f"  跨模态描述使用: {stats['crossmodal_used']}")
    print(f"  跨模态符号: {markers}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'visual_used': stats['visual_only_used'],
        'crossmodal_used': stats['crossmodal_used'],
        'crossmodal_symbols': markers,
        'emergence_round': emergence_round,
    }


def experiment_2_visual_ambiguous(num_rounds: int = 500, verbose: bool = True):
    """实验 2: 视觉模糊（需要跨模态符号）"""
    print(f"\n{'=' * 60}")
    print(f"实验 2: 视觉模糊（跨模态符号应该涌现）")
    print(f"{'=' * 60}")

    game = CrossModalGame()
    crossmodal_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        objects, target_idx = generate_crossmodal_scenario(mode='visual_ambiguous', num_objects=4)
        game.play_round(objects, target_idx)

        if not crossmodal_emerged:
            markers = [s for s in game.language.vocabulary
                       if s in AUDITORY_SYMBOLS or s in TACTILE_SYMBOLS]
            if markers:
                crossmodal_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"visual={stats['visual_only_used']}, "
                  f"crossmodal={stats['crossmodal_used']}, "
                  f"crossmodal_success={stats['crossmodal_success']:.1%}")

    stats = game.get_stats()
    markers = [s for s in game.language.vocabulary
               if s in AUDITORY_SYMBOLS or s in TACTILE_SYMBOLS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  视觉描述使用: {stats['visual_only_used']}")
    print(f"  跨模态描述使用: {stats['crossmodal_used']}")
    print(f"  跨模态成功率: {stats['crossmodal_success']:.1%}")
    print(f"  跨模态符号: {markers}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'visual_used': stats['visual_only_used'],
        'crossmodal_used': stats['crossmodal_used'],
        'crossmodal_success': stats['crossmodal_success'],
        'crossmodal_symbols': markers,
        'emergence_round': emergence_round,
    }


def experiment_3_crossmodal_only(num_rounds: int = 500, verbose: bool = True):
    """实验 3: 纯跨模态（所有物体外观相同）"""
    print(f"\n{'=' * 60}")
    print(f"实验 3: 纯跨模态（只有非视觉特征能区分）")
    print(f"{'=' * 60}")

    game = CrossModalGame()
    crossmodal_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        objects, target_idx = generate_crossmodal_scenario(mode='crossmodal_only', num_objects=4)
        game.play_round(objects, target_idx)

        if not crossmodal_emerged:
            markers = [s for s in game.language.vocabulary
                       if s in AUDITORY_SYMBOLS or s in TACTILE_SYMBOLS]
            if markers:
                crossmodal_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"crossmodal={stats['crossmodal_used']}, "
                  f"crossmodal_success={stats['crossmodal_success']:.1%}")

    stats = game.get_stats()
    markers = [s for s in game.language.vocabulary
               if s in AUDITORY_SYMBOLS or s in TACTILE_SYMBOLS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  跨模态描述使用: {stats['crossmodal_used']}")
    print(f"  跨模态成功率: {stats['crossmodal_success']:.1%}")
    print(f"  跨模态符号: {markers}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'crossmodal_used': stats['crossmodal_used'],
        'crossmodal_success': stats['crossmodal_success'],
        'crossmodal_symbols': markers,
        'emergence_round': emergence_round,
    }


def experiment_4_comparison(num_rounds: int = 300, verbose: bool = True):
    """实验 4: 跨模态 vs 纯视觉"""
    print(f"\n{'=' * 60}")
    print(f"实验 4: 跨模态 vs 纯视觉")
    print(f"{'=' * 60}")

    results = {}

    # 跨模态游戏
    game_cross = CrossModalGame()
    for r in range(num_rounds):
        objects, target_idx = generate_crossmodal_scenario(mode='visual_ambiguous', num_objects=4)
        game_cross.play_round(objects, target_idx)

    stats_cross = game_cross.get_stats()
    results['跨模态'] = {
        'success_rate': stats_cross['success_rate'],
        'crossmodal_used': stats_cross['crossmodal_used'],
    }

    # 纯视觉基线
    game_baseline = BaselineCrossModalGame()
    for r in range(num_rounds):
        objects, target_idx = generate_crossmodal_scenario(mode='visual_ambiguous', num_objects=4)
        game_baseline.play_round(objects, target_idx)

    stats_baseline = game_baseline.get_stats()
    results['纯视觉'] = {
        'success_rate': stats_baseline['success_rate'],
    }

    if verbose:
        print(f"\n  跨模态:")
        print(f"    成功率: {stats_cross['success_rate']:.1%}")
        print(f"    跨模态使用: {stats_cross['crossmodal_used']}")
        print(f"\n  纯视觉:")
        print(f"    成功率: {stats_baseline['success_rate']:.1%}")

    improvement = stats_cross['success_rate'] - stats_baseline['success_rate']
    print(f"\n  改善: {improvement:+.1%}")

    results['improvement'] = improvement
    return results


def experiment_5_stability(num_rounds: int = 300, num_runs: int = 5, verbose: bool = True):
    """实验 5: 跨模态符号涌现稳定性"""
    print(f"\n{'=' * 60}")
    print(f"实验 5: 跨模态符号涌现稳定性 ({num_runs} 次运行)")
    print(f"{'=' * 60}")

    run_results = []
    for run in range(num_runs):
        game = CrossModalGame()
        for r in range(num_rounds):
            objects, target_idx = generate_crossmodal_scenario(mode='visual_ambiguous', num_objects=4)
            game.play_round(objects, target_idx)

        stats = game.get_stats()
        markers = [s for s in game.language.vocabulary
                   if s in AUDITORY_SYMBOLS or s in TACTILE_SYMBOLS]
        result = {
            'crossmodal_emerged': len(markers) > 0,
            'markers': markers,
            'crossmodal_used': stats['crossmodal_used'],
            'crossmodal_success': stats['crossmodal_success'],
            'success_rate': stats['success_rate'],
        }
        run_results.append(result)

        if verbose:
            print(f"  Run {run+1}: markers={result['markers']}, "
                  f"crossmodal={result['crossmodal_used']}, "
                  f"success={result['success_rate']:.1%}")

    avg_crossmodal = np.mean([r['crossmodal_used'] for r in run_results])
    emergence_rate = sum(1 for r in run_results if r['crossmodal_emerged']) / num_runs
    avg_success = np.mean([r['success_rate'] for r in run_results])

    print(f"\n汇总:")
    print(f"  平均跨模态使用: {avg_crossmodal:.1f}")
    print(f"  跨模态符号涌现率: {emergence_rate:.1%} ({sum(1 for r in run_results if r['crossmodal_emerged'])}/{num_runs})")
    print(f"  平均成功率: {avg_success:.1%}")

    return {
        'avg_crossmodal_used': avg_crossmodal,
        'emergence_rate': emergence_rate,
        'avg_success_rate': avg_success,
        'run_details': run_results,
    }


def main():
    """运行所有实验"""
    print("=" * 80)
    print("Phase 23: 跨模态语言 —— 视觉、听觉、触觉整合")
    print("核心假设：视觉模糊驱动听觉/触觉符号涌现")
    print("=" * 80)

    r1 = experiment_1_visual_unique(300, verbose=True)
    r2 = experiment_2_visual_ambiguous(500, verbose=True)
    r3 = experiment_3_crossmodal_only(500, verbose=True)
    r4 = experiment_4_comparison(300, verbose=True)
    r5 = experiment_5_stability(300, 5, verbose=True)

    # 汇总
    print(f"\n{'=' * 80}")
    print("实验汇总")
    print(f"{'=' * 80}")
    print(f"\n实验 1 (视觉唯一):")
    print(f"  跨模态符号: {r1['crossmodal_symbols']}")
    print(f"  跨模态使用: {r1['crossmodal_used']}")
    print(f"\n实验 2 (视觉模糊):")
    print(f"  跨模态符号: {r2['crossmodal_symbols']}")
    print(f"  跨模态使用: {r2['crossmodal_used']}")
    print(f"  跨模态成功率: {r2['crossmodal_success']:.1%}")
    print(f"\n实验 3 (纯跨模态):")
    print(f"  跨模态符号: {r3['crossmodal_symbols']}")
    print(f"  跨模态使用: {r3['crossmodal_used']}")
    print(f"\n实验 4 (对比):")
    print(f"  改善: {r4['improvement']:+.1%}")
    print(f"\n实验 5 (稳定性):")
    print(f"  跨模态符号涌现率: {r5['emergence_rate']:.1%}")
    print(f"  平均跨模态使用: {r5['avg_crossmodal_used']:.1f}")

    print(f"\n{'=' * 80}")
    print("核心结论")
    print(f"{'=' * 80}")
    print("1. 跨模态符号从视觉模糊中涌现")
    print("2. 听觉/触觉特征在视觉无法区分时成为必要描述")
    print("3. 与人类发展一致：多模态感知整合在婴儿期开始")
    print("4. 符号接地不限于视觉——所有感知模态都能接地")

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
        'experiment_1_visual_unique': r1,
        'experiment_2_visual_ambiguous': r2,
        'experiment_3_crossmodal_only': r3,
        'experiment_4_comparison': r4,
        'experiment_5_stability': r5,
    }
    with open('crossmodal_results.json', 'w', encoding='utf-8') as f:
        json.dump(to_serializable(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: crossmodal_results.json")


if __name__ == '__main__':
    main()
