"""
Phase 25 实验：自适应策略选择

5 个实验验证自适应系统是否优于固定策略系统。
"""

import json
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grounding_unified_language import (
    UnifiedCommunicationGame, generate_unified_scenario,
)
from adaptive_strategy import AdaptiveCommunicationGame
import numpy as np


def experiment_1_adaptive_vs_fixed():
    """实验 1：自适应 vs 固定策略"""
    print("=" * 60)
    print("实验 1: 自适应 vs 固定策略")
    print("=" * 60)

    results = {'fixed': [], 'adaptive': []}

    for run in range(5):
        seed = 42 + run
        scenes = []
        random.seed(seed)
        for _ in range(500):
            amb_types = random.choice([
                {'visual_ambiguous', 'crossmodal'},
                {'subset'},
                {'tool'},
                {'causal'},
                {'confidence'},
                {'visual_ambiguous', 'crossmodal', 'subset', 'causal', 'confidence'},
            ])
            scenes.append(generate_unified_scenario(
                ambiguity_types=amb_types,
                num_objects=3,
                speaker_confidence=random.uniform(0.2, 0.95),
            ))

        # 固定系统
        random.seed(seed)
        fixed_game = UnifiedCommunicationGame()
        for scene in scenes:
            fixed_game.play_round(scene)
        fixed_rate = fixed_game.get_stats()['success_rate']
        results['fixed'].append(fixed_rate)

        # 自适应系统
        random.seed(seed)
        adaptive_game = AdaptiveCommunicationGame()
        for scene in scenes:
            adaptive_game.play_round(scene)
        adaptive_rate = adaptive_game.get_stats()['success_rate']
        results['adaptive'].append(adaptive_rate)

        print(f"  运行 {run+1}: 固定={fixed_rate:.1%}, 自适应={adaptive_rate:.1%}")

    fixed_avg = sum(results['fixed']) / len(results['fixed'])
    adaptive_avg = sum(results['adaptive']) / len(results['adaptive'])
    print(f"\n  平均: 固定={fixed_avg:.1%}, 自适应={adaptive_avg:.1%}")
    print(f"  差异: {adaptive_avg - fixed_avg:+.1%}")
    return results


def experiment_2_convergence():
    """实验 2：策略收敛"""
    print("\n" + "=" * 60)
    print("实验 2: 策略权重收敛")
    print("=" * 60)

    random.seed(42)
    game = AdaptiveCommunicationGame()

    weight_history = []
    success_history = []

    for i in range(2000):
        amb_types = random.choice([
            {'visual_ambiguous', 'crossmodal'},
            {'subset'},
            {'tool'},
            {'causal'},
            {'confidence'},
            {'visual_ambiguous', 'crossmodal', 'subset', 'causal', 'confidence'},
        ])
        scene = generate_unified_scenario(
            ambiguity_types=amb_types,
            num_objects=3,
            speaker_confidence=random.uniform(0.2, 0.95),
        )
        game.play_round(scene)

        if (i + 1) % 100 == 0:
            stats = game.get_stats()
            weight_history.append((i + 1, stats['strategy_weights'].copy()))
            success_history.append((i + 1, stats['recent_success_rate']))

    print("\n  策略权重演化:")
    for round_num, weights in weight_history[::4]:  # 每400轮打印一次
        w = {k: f"{v:.1f}" for k, v in weights.items()}
        print(f"    轮次 {round_num}: {w}")

    print("\n  成功率演化:")
    for round_num, rate in success_history[::4]:
        print(f"    轮次 {round_num}: {rate:.1%}")

    # 收敛分析
    final_weights = weight_history[-1][1]
    print(f"\n  最终权重: {final_weights}")

    # 检查权重是否排序合理
    primary = ['visual', 'crossmodal', 'negation', 'tool']
    sorted_primary = sorted(primary, key=lambda s: final_weights[s], reverse=True)
    print(f"  策略优先级: {' > '.join(sorted_primary)}")

    return weight_history, success_history


def experiment_3_ambiguity_preference():
    """实验 3：歧义类型偏好"""
    print("\n" + "=" * 60)
    print("实验 3: 歧义类型-策略偏好")
    print("=" * 60)

    random.seed(42)
    game = AdaptiveCommunicationGame()

    # 分别测试每种歧义类型
    ambiguity_configs = [
        ('visual_ambiguous+crossmodal', {'visual_ambiguous', 'crossmodal'}),
        ('subset', {'subset'}),
        ('tool', {'tool'}),
        ('causal', {'causal'}),
        ('confidence', {'confidence'}),
    ]

    for name, amb_types in ambiguity_configs:
        random.seed(42)
        game_type = AdaptiveCommunicationGame()

        for _ in range(500):
            scene = generate_unified_scenario(
                ambiguity_types=amb_types,
                num_objects=3,
                speaker_confidence=random.uniform(0.2, 0.95),
            )
            game_type.play_round(scene)

        stats = game_type.get_stats()
        weights = stats['strategy_weights']
        usage = stats['strategy_usage']

        print(f"\n  场景: {name}")
        print(f"    成功率: {stats['success_rate']:.1%}")
        print(f"    策略权重: {weights}")
        print(f"    策略使用: {usage}")

        # 找出最优策略
        primary_strategies = ['visual', 'crossmodal', 'negation', 'tool']
        best = max(primary_strategies, key=lambda s: weights[s])
        print(f"    最优策略: {best}")

    return game


def experiment_4_listener_weights():
    """实验 4：Listener 权重演化"""
    print("\n" + "=" * 60)
    print("实验 4: Listener 评分权重演化")
    print("=" * 60)

    random.seed(42)
    game = AdaptiveCommunicationGame()

    weight_history = []

    for i in range(2000):
        amb_types = random.choice([
            {'visual_ambiguous', 'crossmodal'},
            {'subset'},
            {'tool'},
            {'causal'},
            {'confidence'},
            {'visual_ambiguous', 'crossmodal', 'subset', 'causal', 'confidence'},
        ])
        scene = generate_unified_scenario(
            ambiguity_types=amb_types,
            num_objects=3,
            speaker_confidence=random.uniform(0.2, 0.95),
        )
        game.play_round(scene)

        if (i + 1) % 500 == 0:
            weight_history.append((i + 1, game.listener.scoring_weights.copy()))

    print("\n  Listener 权重演化:")
    initial = {
        'visual_match': 2.0, 'auditory_match': 2.0, 'tactile_match': 2.0,
        'affordance_match': 2.0, 'negation_penalty': -10.0,
        'confidence_know': 1.5, 'confidence_think': 0.5,
    }
    print(f"    初始: {initial}")

    for round_num, weights in weight_history:
        print(f"    轮次 {round_num}: {weights}")

    # 变化分析
    final = weight_history[-1][1]
    print("\n  权重变化:")
    for key in initial:
        change = final[key] - initial[key]
        if abs(change) > 0.01:
            print(f"    {key}: {initial[key]:.2f} → {final[key]:.2f} ({change:+.2f})")
        else:
            print(f"    {key}: 无变化")

    return weight_history


def experiment_5_stability():
    """实验 5：稳定性验证"""
    print("\n" + "=" * 60)
    print("实验 5: 稳定性验证（5 次运行）")
    print("=" * 60)

    results = []
    for run in range(5):
        random.seed(42 + run)
        game = AdaptiveCommunicationGame()

        for _ in range(1000):
            amb_types = random.choice([
                {'visual_ambiguous', 'crossmodal'},
                {'subset'},
                {'tool'},
                {'causal'},
                {'confidence'},
                {'visual_ambiguous', 'crossmodal', 'subset', 'causal', 'confidence'},
            ])
            scene = generate_unified_scenario(
                ambiguity_types=amb_types,
                num_objects=3,
                speaker_confidence=random.uniform(0.2, 0.95),
            )
            game.play_round(scene)

        stats = game.get_stats()
        results.append(stats['success_rate'])
        print(f"  运行 {run+1}: 成功率={stats['success_rate']:.1%}")

    avg = sum(results) / len(results)
    std = (sum((r - avg) ** 2 for r in results) / len(results)) ** 0.5
    print(f"\n  平均: {avg:.1%} ± {std:.1%}")
    print(f"  范围: {min(results):.1%} - {max(results):.1%}")

    # 稳定性判断
    if std < 0.05:
        print("  判定: 稳定 (标准差 < 5%)")
    else:
        print(f"  判定: 波动较大 (标准差 = {std:.1%})")

    return results


if __name__ == '__main__':
    print("Phase 25: 自适应策略选择实验")
    print("=" * 60)

    r1 = experiment_1_adaptive_vs_fixed()
    r2_weights, r2_success = experiment_2_convergence()
    r3 = experiment_3_ambiguity_preference()
    r4 = experiment_4_listener_weights()
    r5 = experiment_5_stability()

    print("\n" + "=" * 60)
    print("所有实验完成")
    print("=" * 60)

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
            return [to_serializable(v) for v in obj]
        return obj

    results = {
        'experiment_1_adaptive_vs_fixed': r1,
        'experiment_2_convergence': {
            'weight_history': [(rn, w) for rn, w in r2_weights],
            'success_history': [(rn, s) for rn, s in r2_success],
        },
        'experiment_5_stability': r5,
    }
    with open('adaptive_strategy_results.json', 'w', encoding='utf-8') as f:
        json.dump(to_serializable(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: adaptive_strategy_results.json")
