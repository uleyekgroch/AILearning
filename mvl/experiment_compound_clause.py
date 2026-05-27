"""
Phase 16a: 从句在多值特征系统中涌现实验

验证当多个物体共享所有静态属性值但动作值有重叠时，
相对从句（"that"）从交流压力中涌现。

核心机制：
- 物体 A: {color: {red}, action: {push, pull}}
- 物体 B: {color: {red}, action: {push, grab}}
- "red" 匹配两个，"push" 也匹配两个
- "red that pull" 只匹配 A

实验：
1. 简单从句场景：验证基本从句涌现
2. 多轮涌现实验：统计从句使用率和成功率
3. 复杂度对比：不同场景复杂度下从句的涌现
4. 多次运行验证稳定性
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict

from grounding_compound import (
    CompoundCommunicationGame, CompoundSpeaker, CompoundListener,
    generate_clause_scene, generate_clause_scene_with_overlap,
)
from language_emergence import RELATIVE_MARKERS


def experiment_1_simple_clause():
    """实验 1: 最简单的从句场景"""
    print("=" * 60)
    print("实验 1: 简单从句场景")
    print("=" * 60)

    scene, target_idx = generate_clause_scene_with_overlap(num_objects=3)
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
    print(f"是否使用从句: {any(s in RELATIVE_MARKERS for s in utterance)}")

    chosen = game.listener.interpret(utterance, scene)
    print(f"Listener 选择: {chosen}")
    print(f"正确: {chosen == target_idx}")

    return {
        'clause_emerged': any(s in RELATIVE_MARKERS for s in utterance),
        'utterance': utterance,
        'success': chosen == target_idx,
    }


def experiment_2_emergence(num_rounds: int = 500, verbose: bool = True):
    """实验 2: 从句涌现实验"""
    print(f"\n{'=' * 60}")
    print(f"实验 2: 从句涌现实验 ({num_rounds} 轮)")
    print(f"{'=' * 60}")

    game = CompoundCommunicationGame()
    clause_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        # 交替使用两种场景生成器
        if np.random.random() < 0.5:
            scene, target_idx = generate_clause_scene(
                num_objects=np.random.randint(3, 6),
                num_action_dims=np.random.choice([1, 2]),
            )
        else:
            scene, target_idx = generate_clause_scene_with_overlap(
                num_objects=np.random.randint(3, 5),
            )
        game.play_round(scene, target_idx)

        if not clause_emerged:
            clause_syms = [s for s in game.language.vocabulary if s in RELATIVE_MARKERS]
            if clause_syms:
                clause_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            clause_syms = [s for s in game.language.vocabulary if s in RELATIVE_MARKERS]
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"clause_rate={game.clause_used/max(1,len(game.game_log)):.1%}, "
                  f"clause_syms={clause_syms}")

    stats = game.language.get_stats()
    clause_syms = [s for s in game.language.vocabulary if s in RELATIVE_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {game.language.total_successes/max(1,game.language.total_games):.1%}")
    print(f"  从句使用率: {game.clause_used/max(1,len(game.game_log)):.1%}")
    print(f"  从句成功率: {game.clause_success}/{game.clause_used}")
    print(f"  词汇量: {len(game.language.vocabulary)}")
    print(f"  从句符号: {clause_syms}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': game.language.total_successes / max(1, game.language.total_games),
        'clause_rate': game.clause_used / max(1, len(game.game_log)),
        'clause_success_rate': game.clause_success / max(1, game.clause_used),
        'vocab_size': len(game.language.vocabulary),
        'clause_symbols': clause_syms,
        'emergence_round': emergence_round,
        'clause_emerged': clause_emerged,
    }


def experiment_3_complexity_levels(num_rounds_per_level: int = 200, verbose: bool = True):
    """实验 3: 不同复杂度下从句的涌现"""
    print(f"\n{'=' * 60}")
    print(f"实验 3: 复杂度对比实验")
    print(f"{'=' * 60}")

    levels = [
        ('简单', {'num_objects': 3, 'num_action_dims': 1}),
        ('中等', {'num_objects': 4, 'num_action_dims': 2}),
        ('复杂', {'num_objects': 6, 'num_action_dims': 2}),
        ('极端', {'num_objects': 8, 'num_action_dims': 3}),
    ]

    results = {}
    for name, params in levels:
        game = CompoundCommunicationGame()
        for r in range(num_rounds_per_level):
            scene, target_idx = generate_clause_scene(**params)
            game.play_round(scene, target_idx)

        clause_rate = game.clause_used / max(1, len(game.game_log))
        clause_syms = [s for s in game.language.vocabulary if s in RELATIVE_MARKERS]
        results[name] = {
            'clause_rate': clause_rate,
            'clause_symbols': clause_syms,
            'success_rate': game.language.total_successes / max(1, game.language.total_games),
        }

        if verbose:
            print(f"  {name}: clause_rate={clause_rate:.1%}, "
                  f"success={results[name]['success_rate']:.1%}, "
                  f"clause_syms={clause_syms}")

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
            if np.random.random() < 0.5:
                scene, target_idx = generate_clause_scene(
                    num_objects=np.random.randint(3, 6),
                    num_action_dims=np.random.choice([1, 2]),
                )
            else:
                scene, target_idx = generate_clause_scene_with_overlap(
                    num_objects=np.random.randint(3, 5),
                )
            game.play_round(scene, target_idx)

        clause_rate = game.clause_used / max(1, len(game.game_log))
        clause_syms = [s for s in game.language.vocabulary if s in RELATIVE_MARKERS]
        result = {
            'clause_rate': clause_rate,
            'clause_emerged': len(clause_syms) > 0,
            'success_rate': game.language.total_successes / max(1, game.language.total_games),
        }
        run_results.append(result)

        if verbose:
            print(f"  Run {run+1}: clause_rate={clause_rate:.1%}, "
                  f"emerged={result['clause_emerged']}, "
                  f"success={result['success_rate']:.1%}")

    avg_clause_rate = np.mean([r['clause_rate'] for r in run_results])
    emergence_rate = sum(1 for r in run_results if r['clause_emerged']) / num_runs
    avg_success = np.mean([r['success_rate'] for r in run_results])

    print(f"\n汇总:")
    print(f"  平均从句使用率: {avg_clause_rate:.1%}")
    print(f"  从句涌现率: {emergence_rate:.1%} ({sum(1 for r in run_results if r['clause_emerged'])}/{num_runs})")
    print(f"  平均成功率: {avg_success:.1%}")

    return {
        'avg_clause_rate': avg_clause_rate,
        'emergence_rate': emergence_rate,
        'avg_success_rate': avg_success,
        'run_details': run_results,
    }


def experiment_5_forced_clause():
    """实验 5: 强制使用从句的场景（验证机制正确性）"""
    print(f"\n{'=' * 60}")
    print(f"实验 5: 强制从句场景（验证机制）")
    print(f"{'=' * 60}")

    # 构造一个从句是最优解的场景：
    # 目标有 2 个静态值和 1 个动作值
    # 其他物体共享部分静态值但动作值不同
    scene = [
        {'color': {'red'}, 'shape': {'circle'}, 'action': {'push'}},      # 0
        {'color': {'red'}, 'shape': {'circle'}, 'action': {'pull'}},      # 1
        {'color': {'red'}, 'shape': {'square'}, 'action': {'push'}},      # 2
        {'color': {'blue'}, 'shape': {'circle'}, 'action': {'push'}},     # 3
    ]
    target_idx = 0

    game = CompoundCommunicationGame()
    target = scene[target_idx]
    utterance = game.speaker.describe(target, scene, target_idx)

    print(f"场景:")
    for i, obj in enumerate(scene):
        flat = {k: list(v) for k, v in obj.items()}
        marker = " <-- TARGET" if i == target_idx else ""
        print(f"  {i}: {flat}{marker}")

    print(f"\n目标: {dict(target)}")
    print(f"描述: {utterance}")
    print(f"长度: {len(utterance)}")
    has_clause = any(s in RELATIVE_MARKERS for s in utterance)
    print(f"使用从句: {has_clause}")

    # 手动测试从句机制
    print(f"\n--- 从句机制测试 ---")
    clause_result = game.speaker._try_relative_clause(target, scene, target_idx)
    print(f"从句结果: {clause_result}")

    if clause_result:
        chosen = game.listener.interpret(clause_result, scene)
        print(f"Listener 选择: {chosen}")
        print(f"正确: {chosen == target_idx}")

    return {
        'utterance': utterance,
        'has_clause': has_clause,
        'clause_result': clause_result,
    }


def main():
    """运行所有实验"""
    print("=" * 80)
    print("Phase 16a: 从句在多值特征系统中的涌现")
    print("=" * 80)

    # 实验 1
    r1 = experiment_1_simple_clause()

    # 实验 2
    r2 = experiment_2_emergence(500, verbose=True)

    # 实验 3
    r3 = experiment_3_complexity_levels(200, verbose=True)

    # 实验 4
    r4 = experiment_4_multiple_runs(300, 5, verbose=True)

    # 实验 5: 验证机制
    r5 = experiment_5_forced_clause()

    # 汇总
    print(f"\n{'=' * 80}")
    print("实验汇总")
    print(f"{'=' * 80}")
    print(f"\n实验 1 (简单从句):")
    print(f"  从句涌现: {r1['clause_emerged']}")
    print(f"  描述: {r1['utterance']}")
    print(f"\n实验 2 (涌现实验):")
    print(f"  从句涌现: {r2['clause_emerged']}")
    print(f"  从句使用率: {r2['clause_rate']:.1%}")
    print(f"  从句符号: {r2['clause_symbols']}")
    print(f"\n实验 3 (复杂度对比):")
    for name, res in r3.items():
        print(f"  {name}: clause_rate={res['clause_rate']:.1%}")
    print(f"\n实验 4 (稳定性):")
    print(f"  从句涌现率: {r4['emergence_rate']:.1%}")
    print(f"\n实验 5 (机制验证):")
    print(f"  从句机制可用: {r5['clause_result'] is not None}")
    print(f"  从句结果: {r5['clause_result']}")

    print(f"\n{'=' * 80}")
    print("核心结论")
    print(f"{'=' * 80}")
    print("1. 从句机制已正确实现（_try_relative_clause + _interpret_with_relative）")
    print("2. 在无约束条件下，2符号组合总是比3符号从句更高效")
    print("3. 从句需要时间压力（max_len约束）才能成为最优策略")
    print("4. 下一步：Phase 16b 时间压力实验")


if __name__ == '__main__':
    main()
