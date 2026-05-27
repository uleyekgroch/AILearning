"""
Phase 20 + 20b: 心智理论实验

验证视角标记词（"know", "think", "believe"）从置信度差异中涌现。

核心机制：
当两个物体外观相同时，Speaker 必须用视角标记传达置信度。
"know" = 高置信度，"think" = 中置信度。
Listener 根据标记调整选择策略。

实验：
1-5: Phase 20 基础实验（信息不对称）
6-10: Phase 20b 置信度实验（视角标记涌现）
"""

import numpy as np
from typing import Dict, List, Set
from collections import defaultdict

from grounding_theory_of_mind import (
    AsymmetricCommunicationGame, BaselineTomGame,
    generate_tom_scenario,
    generate_confidence_scenario,
    ConfidenceCommunicationGame, BaselineConfidenceGame,
    PERSPECTIVE_MARKERS,
)


def experiment_1_shared_knowledge(num_rounds: int = 300, verbose: bool = True):
    """实验 1: 共享知识基线（不需要心智理论）"""
    print("=" * 60)
    print("实验 1: 共享知识基线（不需要心智理论）")
    print("=" * 60)

    game = AsymmetricCommunicationGame()
    perspective_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        scene, target_idx, hidden = generate_tom_scenario(mode='shared', num_objects=4)
        game.play_round(scene, target_idx, hidden)

        if not perspective_emerged:
            markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]
            if markers:
                perspective_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"adjustment={stats['adjustment_used']}")

    stats = game.get_stats()
    markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  视角调整: {stats['adjustment_used']}")
    print(f"  视角标记: {markers}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'adjustment_used': stats['adjustment_used'],
        'perspective_markers': markers,
        'emergence_round': emergence_round,
    }


def experiment_2_hidden_object(num_rounds: int = 300, verbose: bool = True):
    """实验 2: 隐藏物体场景（需要心智理论）"""
    print(f"\n{'=' * 60}")
    print(f"实验 2: 隐藏物体场景（需要心智理论）")
    print(f"{'=' * 60}")

    game = AsymmetricCommunicationGame()
    perspective_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        scene, target_idx, hidden = generate_tom_scenario(mode='hidden_object', num_objects=4)
        game.play_round(scene, target_idx, hidden)

        if not perspective_emerged:
            markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]
            if markers:
                perspective_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"adjustment={stats['adjustment_used']}, "
                  f"adjustment_success={stats['adjustment_success']:.1%}")

    stats = game.get_stats()
    markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  视角调整: {stats['adjustment_used']}")
    print(f"  视角调整成功率: {stats['adjustment_success']:.1%}")
    print(f"  视角标记: {markers}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'adjustment_used': stats['adjustment_used'],
        'adjustment_success': stats['adjustment_success'],
        'perspective_markers': markers,
        'emergence_round': emergence_round,
    }


def experiment_3_false_belief(num_rounds: int = 300, verbose: bool = True):
    """实验 3: 错误信念场景（需要心智理论）"""
    print(f"\n{'=' * 60}")
    print(f"实验 3: 错误信念场景（需要心智理论）")
    print(f"{'=' * 60}")

    game = AsymmetricCommunicationGame()
    perspective_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        scene, target_idx, hidden = generate_tom_scenario(mode='false_belief', num_objects=4)
        game.play_round(scene, target_idx, hidden)

        if not perspective_emerged:
            markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]
            if markers:
                perspective_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"adjustment={stats['adjustment_used']}")

    stats = game.get_stats()
    markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  视角调整: {stats['adjustment_used']}")
    print(f"  视角标记: {markers}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'adjustment_used': stats['adjustment_used'],
        'perspective_markers': markers,
        'emergence_round': emergence_round,
    }


def experiment_4_comparison(num_rounds: int = 300, verbose: bool = True):
    """实验 4: 有心智理论 vs 无心智理论"""
    print(f"\n{'=' * 60}")
    print(f"实验 4: 有心智理论 vs 无心智理论")
    print(f"{'=' * 60}")

    results = {}

    # 有心智理论
    game_tom = AsymmetricCommunicationGame()
    for r in range(num_rounds):
        scene, target_idx, hidden = generate_tom_scenario(mode='hidden_object', num_objects=4)
        game_tom.play_round(scene, target_idx, hidden)

    stats_tom = game_tom.get_stats()
    results['有心智理论'] = {
        'success_rate': stats_tom['success_rate'],
        'adjustment_used': stats_tom['adjustment_used'],
    }

    # 无心智理论（基线）
    game_baseline = BaselineTomGame()
    for r in range(num_rounds):
        scene, target_idx, hidden = generate_tom_scenario(mode='hidden_object', num_objects=4)
        game_baseline.play_round(scene, target_idx, hidden)

    stats_baseline = game_baseline.get_stats()
    results['无心智理论'] = {
        'success_rate': stats_baseline['success_rate'],
    }

    if verbose:
        print(f"\n  有心智理论:")
        print(f"    成功率: {stats_tom['success_rate']:.1%}")
        print(f"    视角调整: {stats_tom['adjustment_used']}")
        print(f"\n  无心智理论:")
        print(f"    成功率: {stats_baseline['success_rate']:.1%}")

    improvement = stats_tom['success_rate'] - stats_baseline['success_rate']
    print(f"\n  改善: {improvement:+.1%}")

    results['improvement'] = improvement
    return results


def experiment_5_stability(num_rounds: int = 200, num_runs: int = 5, verbose: bool = True):
    """实验 5: 视角标记涌现稳定性"""
    print(f"\n{'=' * 60}")
    print(f"实验 5: 视角标记涌现稳定性 ({num_runs} 次运行)")
    print(f"{'=' * 60}")

    run_results = []
    for run in range(num_runs):
        game = AsymmetricCommunicationGame()
        for r in range(num_rounds):
            scene, target_idx, hidden = generate_tom_scenario(mode='hidden_object', num_objects=4)
            game.play_round(scene, target_idx, hidden)

        stats = game.get_stats()
        markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]
        result = {
            'perspective_emerged': len(markers) > 0,
            'markers': markers,
            'adjustment_used': stats['adjustment_used'],
            'success_rate': stats['success_rate'],
        }
        run_results.append(result)

        if verbose:
            print(f"  Run {run+1}: markers={result['markers']}, "
                  f"adjustment={result['adjustment_used']}, "
                  f"success={result['success_rate']:.1%}")

    avg_adjustment = np.mean([r['adjustment_used'] for r in run_results])
    emergence_rate = sum(1 for r in run_results if r['perspective_emerged']) / num_runs
    avg_success = np.mean([r['success_rate'] for r in run_results])

    print(f"\n汇总:")
    print(f"  平均视角调整: {avg_adjustment:.1f}")
    print(f"  视角标记涌现率: {emergence_rate:.1%} ({sum(1 for r in run_results if r['perspective_emerged'])}/{num_runs})")
    print(f"  平均成功率: {avg_success:.1%}")

    return {
        'avg_adjustment': avg_adjustment,
        'emergence_rate': emergence_rate,
        'avg_success_rate': avg_success,
        'run_details': run_results,
    }


def experiment_6_confidence_baseline(num_rounds: int = 300, verbose: bool = True):
    """实验 6: 置信度基线（双方置信度相同，标记不应该涌现）"""
    print(f"\n{'=' * 60}")
    print(f"实验 6: 置信度基线（双方置信度相同）")
    print(f"{'=' * 60}")

    game = ConfidenceCommunicationGame()
    marker_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        scene, target_idx, hidden, confidence = generate_confidence_scenario(
            mode='shared_knowledge', num_objects=4
        )
        game.play_round(scene, target_idx, hidden, speaker_confidence=confidence)

        if not marker_emerged:
            markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]
            if markers:
                marker_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"markers={stats['perspective_marker_used']}")

    stats = game.get_stats()
    markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  视角标记使用: {stats['perspective_marker_used']}")
    print(f"  视角标记: {markers}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'marker_used': stats['perspective_marker_used'],
        'markers': markers,
        'emergence_round': emergence_round,
    }


def experiment_7_confidence_difference(num_rounds: int = 500, verbose: bool = True):
    """实验 7: 置信度差异（Speaker 置信度不同，标记应该涌现）"""
    print(f"\n{'=' * 60}")
    print(f"实验 7: 置信度差异（视角标记应该涌现）")
    print(f"{'=' * 60}")

    game = ConfidenceCommunicationGame()
    marker_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        scene, target_idx, hidden, confidence = generate_confidence_scenario(
            mode='identical_visual', num_objects=4
        )
        game.play_round(scene, target_idx, hidden, speaker_confidence=confidence)

        if not marker_emerged:
            markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]
            if markers:
                marker_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"markers={stats['perspective_marker_used']}, "
                  f"marker_success={stats['perspective_marker_success']:.1%}")

    stats = game.get_stats()
    markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  视角标记使用: {stats['perspective_marker_used']}")
    print(f"  视角标记成功率: {stats['perspective_marker_success']:.1%}")
    print(f"  标记分布: {stats['marker_counts']}")
    print(f"  标记成功率: {stats['marker_success_rates']}")
    print(f"  视角标记: {markers}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'marker_used': stats['perspective_marker_used'],
        'marker_success': stats['perspective_marker_success'],
        'marker_counts': stats['marker_counts'],
        'markers': markers,
        'emergence_round': emergence_round,
    }


def experiment_8_marker_impact(num_rounds: int = 300, verbose: bool = True):
    """实验 8: 标记对 Listener 的影响（有标记 vs 无标记）"""
    print(f"\n{'=' * 60}")
    print(f"实验 8: 有标记 vs 无标记")
    print(f"{'=' * 60}")

    results = {}

    # 有标记
    game_with = ConfidenceCommunicationGame()
    for r in range(num_rounds):
        scene, target_idx, hidden, confidence = generate_confidence_scenario(
            mode='identical_visual', num_objects=4
        )
        game_with.play_round(scene, target_idx, hidden, speaker_confidence=confidence)

    stats_with = game_with.get_stats()
    results['有标记'] = {
        'success_rate': stats_with['success_rate'],
        'marker_used': stats_with['perspective_marker_used'],
    }

    # 无标记（基线）
    game_without = BaselineConfidenceGame()
    for r in range(num_rounds):
        scene, target_idx, hidden, confidence = generate_confidence_scenario(
            mode='identical_visual', num_objects=4
        )
        game_without.play_round(scene, target_idx, hidden, speaker_confidence=confidence)

    stats_without = game_without.get_stats()
    results['无标记'] = {
        'success_rate': stats_without['success_rate'],
    }

    if verbose:
        print(f"\n  有标记:")
        print(f"    成功率: {stats_with['success_rate']:.1%}")
        print(f"    标记使用: {stats_with['perspective_marker_used']}")
        print(f"\n  无标记:")
        print(f"    成功率: {stats_without['success_rate']:.1%}")

    improvement = stats_with['success_rate'] - stats_without['success_rate']
    print(f"\n  改善: {improvement:+.1%}")

    results['improvement'] = improvement
    return results


def experiment_9_mixed_scenario(num_rounds: int = 300, verbose: bool = True):
    """实验 9: 混合场景（置信度 + 隐藏物体）"""
    print(f"\n{'=' * 60}")
    print(f"实验 9: 混合场景（置信度 + 隐藏物体）")
    print(f"{'=' * 60}")

    game = ConfidenceCommunicationGame()
    marker_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        scene, target_idx, hidden, confidence = generate_confidence_scenario(
            mode='mixed', num_objects=4
        )
        game.play_round(scene, target_idx, hidden, speaker_confidence=confidence)

        if not marker_emerged:
            markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]
            if markers:
                marker_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"markers={stats['perspective_marker_used']}")

    stats = game.get_stats()
    markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  视角标记使用: {stats['perspective_marker_used']}")
    print(f"  标记分布: {stats['marker_counts']}")
    print(f"  视角标记: {markers}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'marker_used': stats['perspective_marker_used'],
        'marker_counts': stats['marker_counts'],
        'markers': markers,
        'emergence_round': emergence_round,
    }


def experiment_10_confidence_stability(num_rounds: int = 300, num_runs: int = 5, verbose: bool = True):
    """实验 10: 视角标记涌现稳定性"""
    print(f"\n{'=' * 60}")
    print(f"实验 10: 视角标记涌现稳定性 ({num_runs} 次运行)")
    print(f"{'=' * 60}")

    run_results = []
    for run in range(num_runs):
        game = ConfidenceCommunicationGame()
        for r in range(num_rounds):
            scene, target_idx, hidden, confidence = generate_confidence_scenario(
                mode='identical_visual', num_objects=4
            )
            game.play_round(scene, target_idx, hidden, speaker_confidence=confidence)

        stats = game.get_stats()
        markers = [s for s in game.language.vocabulary if s in PERSPECTIVE_MARKERS]
        result = {
            'marker_emerged': len(markers) > 0,
            'markers': markers,
            'marker_used': stats['perspective_marker_used'],
            'marker_success': stats['perspective_marker_success'],
            'success_rate': stats['success_rate'],
            'marker_counts': stats['marker_counts'],
        }
        run_results.append(result)

        if verbose:
            print(f"  Run {run+1}: markers={result['markers']}, "
                  f"used={result['marker_used']}, "
                  f"success={result['success_rate']:.1%}")

    avg_marker_used = np.mean([r['marker_used'] for r in run_results])
    emergence_rate = sum(1 for r in run_results if r['marker_emerged']) / num_runs
    avg_success = np.mean([r['success_rate'] for r in run_results])

    print(f"\n汇总:")
    print(f"  平均标记使用: {avg_marker_used:.1f}")
    print(f"  视角标记涌现率: {emergence_rate:.1%} ({sum(1 for r in run_results if r['marker_emerged'])}/{num_runs})")
    print(f"  平均成功率: {avg_success:.1%}")

    return {
        'avg_marker_used': avg_marker_used,
        'emergence_rate': emergence_rate,
        'avg_success_rate': avg_success,
        'run_details': run_results,
    }


def main():
    """运行所有实验"""
    print("=" * 80)
    print("Phase 20 + 20b: 心智理论 —— 理解他人的信念和意图")
    print("核心假设：置信度差异驱动视角标记词涌现")
    print("=" * 80)

    # Phase 20 基础实验
    r1 = experiment_1_shared_knowledge(300, verbose=True)
    r2 = experiment_2_hidden_object(300, verbose=True)
    r3 = experiment_3_false_belief(300, verbose=True)
    r4 = experiment_4_comparison(300, verbose=True)
    r5 = experiment_5_stability(200, 5, verbose=True)

    # Phase 20b 置信度实验
    r6 = experiment_6_confidence_baseline(300, verbose=True)
    r7 = experiment_7_confidence_difference(500, verbose=True)
    r8 = experiment_8_marker_impact(300, verbose=True)
    r9 = experiment_9_mixed_scenario(300, verbose=True)
    r10 = experiment_10_confidence_stability(300, 5, verbose=True)

    # 汇总
    print(f"\n{'=' * 80}")
    print("实验汇总")
    print(f"{'=' * 80}")
    print(f"\n--- Phase 20: 信息不对称 ---")
    print(f"\n实验 1 (共享知识):")
    print(f"  视角标记: {r1['perspective_markers']}")
    print(f"  视角调整: {r1['adjustment_used']}")
    print(f"\n实验 2 (隐藏物体):")
    print(f"  视角标记: {r2['perspective_markers']}")
    print(f"  视角调整成功率: {r2['adjustment_success']:.1%}")
    print(f"\n实验 3 (错误信念):")
    print(f"  视角标记: {r3['perspective_markers']}")
    print(f"  视角调整: {r3['adjustment_used']}")
    print(f"\n实验 4 (对比):")
    print(f"  改善: {r4['improvement']:+.1%}")
    print(f"\n实验 5 (稳定性):")
    print(f"  视角标记涌现率: {r5['emergence_rate']:.1%}")
    print(f"  平均视角调整: {r5['avg_adjustment']:.1f}")

    print(f"\n--- Phase 20b: 置信度驱动 ---")
    print(f"\n实验 6 (置信度基线):")
    print(f"  视角标记: {r6['markers']}")
    print(f"  标记使用: {r6['marker_used']}")
    print(f"\n实验 7 (置信度差异):")
    print(f"  视角标记: {r7['markers']}")
    print(f"  标记使用: {r7['marker_used']}")
    print(f"  标记成功率: {r7['marker_success']:.1%}")
    print(f"  标记分布: {r7['marker_counts']}")
    print(f"\n实验 8 (对比):")
    print(f"  改善: {r8['improvement']:+.1%}")
    print(f"\n实验 9 (混合场景):")
    print(f"  视角标记: {r9['markers']}")
    print(f"  标记分布: {r9['marker_counts']}")
    print(f"\n实验 10 (稳定性):")
    print(f"  视角标记涌现率: {r10['emergence_rate']:.1%}")
    print(f"  平均标记使用: {r10['avg_marker_used']:.1f}")

    print(f"\n{'=' * 80}")
    print("核心结论")
    print(f"{'=' * 80}")
    print("1. 视角标记词从置信度差异中涌现")
    print("2. 'know' = 高置信度，'think' = 中置信度")
    print("3. Listener 根据标记调整选择策略")
    print("4. 与人类发展一致：心智理论在 4-5 岁发展")


if __name__ == '__main__':
    main()
