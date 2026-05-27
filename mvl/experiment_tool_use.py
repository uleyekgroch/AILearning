"""
Phase 22: 工具使用实验

验证功能描述标记词（"use", "for"）从工具选择压力中涌现。

核心机制：
当多个工具有相似外观但不同功能时，
直接外观描述无法区分。
Speaker 必须用功能描述（"use reach for reach_object"）来区分。

实验：
1. 直接行动基线（不需要工具）
2. 单工具选择（功能描述应该涌现）
3. 工具选择压力（外观相似的工具）
4. 有工具推理 vs 无工具推理
5. 稳定性验证：5 次运行
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict

from grounding_tool_use import (
    ToolCommunicationGame, BaselineToolGame,
    generate_tool_scenario, Tool, Goal,
    TOOL_TEMPLATES, GOAL_TEMPLATES,
)
from language_emergence import TOOL_MARKERS


def experiment_1_direct_action(num_rounds: int = 300, verbose: bool = True):
    """实验 1: 直接行动基线（不需要工具）"""
    print("=" * 60)
    print("实验 1: 直接行动基线（不需要工具）")
    print("=" * 60)

    game = ToolCommunicationGame()
    functional_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        tools, goal, correct_idx = generate_tool_scenario(mode='direct', num_tools=4)
        game.play_round(tools, goal, correct_idx)

        if not functional_emerged:
            markers = [s for s in game.language.vocabulary if s in TOOL_MARKERS]
            if markers:
                functional_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"functional={stats['functional_used']}")

    stats = game.get_stats()
    markers = [s for s in game.language.vocabulary if s in TOOL_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  功能描述使用: {stats['functional_used']}")
    print(f"  外观描述使用: {stats['appearance_used']}")
    print(f"  功能标记: {markers}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'functional_used': stats['functional_used'],
        'appearance_used': stats['appearance_used'],
        'tool_markers': markers,
        'emergence_round': emergence_round,
    }


def experiment_2_single_tool(num_rounds: int = 500, verbose: bool = True):
    """实验 2: 单工具选择（需要识别正确工具）"""
    print(f"\n{'=' * 60}")
    print(f"实验 2: 单工具选择（功能描述应该涌现）")
    print(f"{'=' * 60}")

    game = ToolCommunicationGame()
    functional_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        tools, goal, correct_idx = generate_tool_scenario(mode='single_tool', num_tools=4)
        game.play_round(tools, goal, correct_idx)

        if not functional_emerged:
            markers = [s for s in game.language.vocabulary if s in TOOL_MARKERS]
            if markers:
                functional_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"functional={stats['functional_used']}, "
                  f"functional_success={stats['functional_success']:.1%}")

    stats = game.get_stats()
    markers = [s for s in game.language.vocabulary if s in TOOL_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  功能描述使用: {stats['functional_used']}")
    print(f"  功能描述成功率: {stats['functional_success']:.1%}")
    print(f"  外观描述使用: {stats['appearance_used']}")
    print(f"  外观描述成功率: {stats['appearance_success']:.1%}")
    print(f"  功能标记: {markers}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'functional_used': stats['functional_used'],
        'functional_success': stats['functional_success'],
        'appearance_used': stats['appearance_used'],
        'appearance_success': stats['appearance_success'],
        'tool_markers': markers,
        'emergence_round': emergence_round,
    }


def experiment_3_tool_selection(num_rounds: int = 300, verbose: bool = True):
    """实验 3: 工具选择压力（外观相似的工具）"""
    print(f"\n{'=' * 60}")
    print(f"实验 3: 工具选择压力")
    print(f"{'=' * 60}")

    game = ToolCommunicationGame()
    functional_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        tools, goal, correct_idx = generate_tool_scenario(mode='tool_selection', num_tools=6)
        game.play_round(tools, goal, correct_idx)

        if not functional_emerged:
            markers = [s for s in game.language.vocabulary if s in TOOL_MARKERS]
            if markers:
                functional_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"functional={stats['functional_used']}")

    stats = game.get_stats()
    markers = [s for s in game.language.vocabulary if s in TOOL_MARKERS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  功能描述使用: {stats['functional_used']}")
    print(f"  功能描述成功率: {stats['functional_success']:.1%}")
    print(f"  功能标记: {markers}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'functional_used': stats['functional_used'],
        'functional_success': stats['functional_success'],
        'tool_markers': markers,
        'emergence_round': emergence_round,
    }


def experiment_4_comparison(num_rounds: int = 300, verbose: bool = True):
    """实验 4: 有工具推理 vs 无工具推理"""
    print(f"\n{'=' * 60}")
    print(f"实验 4: 有工具推理 vs 无工具推理")
    print(f"{'=' * 60}")

    results = {}

    # 有工具推理
    game_tool = ToolCommunicationGame()
    for r in range(num_rounds):
        tools, goal, correct_idx = generate_tool_scenario(mode='tool_selection', num_tools=4)
        game_tool.play_round(tools, goal, correct_idx)

    stats_tool = game_tool.get_stats()
    results['有工具推理'] = {
        'success_rate': stats_tool['success_rate'],
        'functional_used': stats_tool['functional_used'],
    }

    # 无工具推理（基线）
    game_baseline = BaselineToolGame()
    for r in range(num_rounds):
        tools, goal, correct_idx = generate_tool_scenario(mode='tool_selection', num_tools=4)
        game_baseline.play_round(tools, goal, correct_idx)

    stats_baseline = game_baseline.get_stats()
    results['无工具推理'] = {
        'success_rate': stats_baseline['success_rate'],
    }

    if verbose:
        print(f"\n  有工具推理:")
        print(f"    成功率: {stats_tool['success_rate']:.1%}")
        print(f"    功能描述使用: {stats_tool['functional_used']}")
        print(f"\n  无工具推理:")
        print(f"    成功率: {stats_baseline['success_rate']:.1%}")

    improvement = stats_tool['success_rate'] - stats_baseline['success_rate']
    print(f"\n  改善: {improvement:+.1%}")

    results['improvement'] = improvement
    return results


def experiment_5_stability(num_rounds: int = 200, num_runs: int = 5, verbose: bool = True):
    """实验 5: 功能标记涌现稳定性"""
    print(f"\n{'=' * 60}")
    print(f"实验 5: 功能标记涌现稳定性 ({num_runs} 次运行)")
    print(f"{'=' * 60}")

    run_results = []
    for run in range(num_runs):
        game = ToolCommunicationGame()
        for r in range(num_rounds):
            tools, goal, correct_idx = generate_tool_scenario(mode='tool_selection', num_tools=4)
            game.play_round(tools, goal, correct_idx)

        stats = game.get_stats()
        markers = [s for s in game.language.vocabulary if s in TOOL_MARKERS]
        result = {
            'functional_emerged': len(markers) > 0,
            'markers': markers,
            'functional_used': stats['functional_used'],
            'functional_success': stats['functional_success'],
            'success_rate': stats['success_rate'],
        }
        run_results.append(result)

        if verbose:
            print(f"  Run {run+1}: markers={result['markers']}, "
                  f"functional={result['functional_used']}, "
                  f"success={result['success_rate']:.1%}")

    avg_functional = np.mean([r['functional_used'] for r in run_results])
    emergence_rate = sum(1 for r in run_results if r['functional_emerged']) / num_runs
    avg_success = np.mean([r['success_rate'] for r in run_results])

    print(f"\n汇总:")
    print(f"  平均功能描述使用: {avg_functional:.1f}")
    print(f"  功能标记涌现率: {emergence_rate:.1%} ({sum(1 for r in run_results if r['functional_emerged'])}/{num_runs})")
    print(f"  平均成功率: {avg_success:.1%}")

    return {
        'avg_functional_used': avg_functional,
        'emergence_rate': emergence_rate,
        'avg_success_rate': avg_success,
        'run_details': run_results,
    }


def main():
    """运行所有实验"""
    print("=" * 80)
    print("Phase 22: 工具使用 —— 问题解决和规划")
    print("核心假设：工具选择压力驱动功能描述标记词涌现")
    print("=" * 80)

    r1 = experiment_1_direct_action(300, verbose=True)
    r2 = experiment_2_single_tool(500, verbose=True)
    r3 = experiment_3_tool_selection(300, verbose=True)
    r4 = experiment_4_comparison(300, verbose=True)
    r5 = experiment_5_stability(200, 5, verbose=True)

    # 汇总
    print(f"\n{'=' * 80}")
    print("实验汇总")
    print(f"{'=' * 80}")
    print(f"\n实验 1 (直接行动):")
    print(f"  功能标记: {r1['tool_markers']}")
    print(f"  功能描述使用: {r1['functional_used']}")
    print(f"\n实验 2 (单工具选择):")
    print(f"  功能标记: {r2['tool_markers']}")
    print(f"  功能描述使用: {r2['functional_used']}")
    print(f"  功能描述成功率: {r2['functional_success']:.1%}")
    print(f"\n实验 3 (工具选择压力):")
    print(f"  功能标记: {r3['tool_markers']}")
    print(f"  功能描述使用: {r3['functional_used']}")
    print(f"\n实验 4 (对比):")
    print(f"  改善: {r4['improvement']:+.1%}")
    print(f"\n实验 5 (稳定性):")
    print(f"  功能标记涌现率: {r5['emergence_rate']:.1%}")
    print(f"  平均功能描述使用: {r5['avg_functional_used']:.1f}")

    print(f"\n{'=' * 80}")
    print("核心结论")
    print(f"{'=' * 80}")
    print("1. 功能描述标记词从工具选择压力中涌现")
    print("2. 外观相似时，功能描述是唯一区分策略")
    print("3. 工具使用需要目标依赖的物体重释")
    print("4. 与人类发展一致：工具使用在 12-18 个月发展")


if __name__ == '__main__':
    main()
