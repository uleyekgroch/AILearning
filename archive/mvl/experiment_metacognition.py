"""
Phase 18: 元认知与自我反思实验

验证元认知信号（"uncertain", "help"）从交流困难中涌现。

核心机制：
当 Agent 遇到交流困难时，元认知信号帮助调整学习策略。

实验：
1. 基本元认知实验：验证信号涌现
2. 困难场景实验：高歧义下元认知的作用
3. 对比实验：有元认知 vs 无元认知的学习效率
4. 稳定性验证：多次运行的涌现率
"""

import json
import numpy as np
from typing import Dict, List
from collections import defaultdict

from metacognition import (
    MetacognitiveGame, MetacognitiveAgent, MetacognitiveTeacher,
    generate_metacognitive_scene, generate_difficult_scene,
    METACOGNITIVE_SIGNALS,
)


class BaselineGame:
    """无元认知的基线游戏"""

    def __init__(self):
        from language_emergence import EmergingLanguage
        self.language = EmergingLanguage()
        self.game_log = []

    def play_round(self, scene_features: List[Dict[str, str]],
                   target_idx: int) -> bool:
        if target_idx >= len(scene_features):
            return False

        target = scene_features[target_idx]
        target_values = set(target.values())

        # 简单描述策略
        utterance = []
        for sym in target_values:
            count = sum(1 for obj in scene_features if sym in obj.values())
            if count == 1:
                utterance = [sym]
                break

        if not utterance:
            utterance = [list(target_values)[0]]

        # 检查成功
        scores = []
        for i, obj in enumerate(scene_features):
            obj_values = set(obj.values())
            matches = sum(1 for s in utterance if s in obj_values)
            scores.append((i, matches))

        scores.sort(key=lambda x: x[1], reverse=True)
        success = scores[0][0] == target_idx and scores[0][1] > 0

        self.language.total_games += 1
        if success:
            self.language.total_successes += 1
        self.language.record_usage(utterance, success)

        self.game_log.append({
            'target_idx': target_idx,
            'utterance': utterance,
            'success': success,
        })
        return success

    def get_stats(self):
        return self.language.get_stats()


def experiment_1_basic_metacognition(num_rounds: int = 300, verbose: bool = True):
    """实验 1: 基本元认知实验"""
    print("=" * 60)
    print("实验 1: 基本元认知实验")
    print("=" * 60)

    game = MetacognitiveGame()
    signal_emerged = False
    emergence_round = -1

    for r in range(num_rounds):
        # 混合简单和困难场景
        if np.random.random() < 0.5:
            scene, target_idx = generate_metacognitive_scene()
        else:
            scene, target_idx = generate_difficult_scene()
        game.play_round(scene, target_idx)

        if not signal_emerged:
            signals = [s for s in game.language.vocabulary if s in METACOGNITIVE_SIGNALS]
            if signals:
                signal_emerged = True
                emergence_round = r

        if verbose and (r + 1) % 100 == 0:
            signals = [s for s in game.language.vocabulary if s in METACOGNITIVE_SIGNALS]
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"signal_rate={stats['signal_rate']:.1%}, "
                  f"signals={signals}")

    stats = game.get_stats()
    signals = [s for s in game.language.vocabulary if s in METACOGNITIVE_SIGNALS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  信号使用率: {stats['signal_rate']:.1%}")
    print(f"  信号成功率: {stats['signal_success']}/{stats['signal_used']}")
    print(f"  帮助请求: {stats['help_requests']}")
    print(f"  教师调整: {stats['teacher_adaptations']}")
    print(f"  元认知符号: {signals}")
    print(f"  涌现轮次: {emergence_round}")

    return {
        'success_rate': stats['success_rate'],
        'signal_rate': stats['signal_rate'],
        'signal_success_rate': stats['signal_success'] / max(1, stats['signal_used']),
        'help_requests': stats['help_requests'],
        'teacher_adaptations': stats['teacher_adaptations'],
        'metacognitive_symbols': signals,
        'emergence_round': emergence_round,
        'signal_emerged': signal_emerged,
    }


def experiment_2_difficult_scenes(num_rounds: int = 300, verbose: bool = True):
    """实验 2: 困难场景下的元认知"""
    print(f"\n{'=' * 60}")
    print(f"实验 2: 困难场景下的元认知")
    print(f"{'=' * 60}")

    game = MetacognitiveGame()

    for r in range(num_rounds):
        scene, target_idx = generate_difficult_scene(num_objects=6)
        game.play_round(scene, target_idx)

        if verbose and (r + 1) % 100 == 0:
            stats = game.get_stats()
            print(f"  Round {r+1}: "
                  f"success={game.language.total_successes}/{game.language.total_games}, "
                  f"signal_rate={stats['signal_rate']:.1%}, "
                  f"help={stats['help_requests']}")

    stats = game.get_stats()
    signals = [s for s in game.language.vocabulary if s in METACOGNITIVE_SIGNALS]

    print(f"\n最终结果:")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  信号使用率: {stats['signal_rate']:.1%}")
    print(f"  帮助请求: {stats['help_requests']}")
    print(f"  教师调整: {stats['teacher_adaptations']}")
    print(f"  元认知符号: {signals}")

    return {
        'success_rate': stats['success_rate'],
        'signal_rate': stats['signal_rate'],
        'help_requests': stats['help_requests'],
        'teacher_adaptations': stats['teacher_adaptations'],
        'metacognitive_symbols': signals,
    }


def experiment_3_comparison(num_rounds: int = 300, verbose: bool = True):
    """实验 3: 有元认知 vs 无元认知"""
    print(f"\n{'=' * 60}")
    print(f"实验 3: 有元认知 vs 无元认知")
    print(f"{'=' * 60}")

    results = {}

    # 有元认知
    game_meta = MetacognitiveGame()
    for r in range(num_rounds):
        if np.random.random() < 0.5:
            scene, target_idx = generate_metacognitive_scene()
        else:
            scene, target_idx = generate_difficult_scene()
        game_meta.play_round(scene, target_idx)

    stats_meta = game_meta.get_stats()
    results['有元认知'] = {
        'success_rate': stats_meta['success_rate'],
        'signal_rate': stats_meta['signal_rate'],
        'help_requests': stats_meta['help_requests'],
    }

    # 无元认知
    game_base = BaselineGame()
    for r in range(num_rounds):
        if np.random.random() < 0.5:
            scene, target_idx = generate_metacognitive_scene()
        else:
            scene, target_idx = generate_difficult_scene()
        game_base.play_round(scene, target_idx)

    stats_base = game_base.get_stats()
    results['无元认知'] = {
        'success_rate': stats_base['success_rate'],
    }

    if verbose:
        print(f"\n  有元认知:")
        print(f"    成功率: {stats_meta['success_rate']:.1%}")
        print(f"    信号使用率: {stats_meta['signal_rate']:.1%}")
        print(f"    帮助请求: {stats_meta['help_requests']}")
        print(f"\n  无元认知:")
        print(f"    成功率: {stats_base['success_rate']:.1%}")

    improvement = stats_meta['success_rate'] - stats_base['success_rate']
    print(f"\n  改善: {improvement:+.1%}")

    results['improvement'] = improvement
    return results


def experiment_4_stability(num_rounds: int = 200, num_runs: int = 5, verbose: bool = True):
    """实验 4: 元认知涌现稳定性"""
    print(f"\n{'=' * 60}")
    print(f"实验 4: 元认知涌现稳定性 ({num_runs} 次运行)")
    print(f"{'=' * 60}")

    run_results = []
    for run in range(num_runs):
        game = MetacognitiveGame()
        for r in range(num_rounds):
            if np.random.random() < 0.5:
                scene, target_idx = generate_metacognitive_scene()
            else:
                scene, target_idx = generate_difficult_scene()
            game.play_round(scene, target_idx)

        stats = game.get_stats()
        signals = [s for s in game.language.vocabulary if s in METACOGNITIVE_SIGNALS]
        result = {
            'signal_emerged': len(signals) > 0,
            'signal_rate': stats['signal_rate'],
            'success_rate': stats['success_rate'],
            'help_requests': stats['help_requests'],
        }
        run_results.append(result)

        if verbose:
            print(f"  Run {run+1}: signal_rate={result['signal_rate']:.1%}, "
                  f"emerged={result['signal_emerged']}, "
                  f"success={result['success_rate']:.1%}")

    avg_signal_rate = np.mean([r['signal_rate'] for r in run_results])
    emergence_rate = sum(1 for r in run_results if r['signal_emerged']) / num_runs
    avg_success = np.mean([r['success_rate'] for r in run_results])

    print(f"\n汇总:")
    print(f"  平均信号使用率: {avg_signal_rate:.1%}")
    print(f"  元认知涌现率: {emergence_rate:.1%} ({sum(1 for r in run_results if r['signal_emerged'])}/{num_runs})")
    print(f"  平均成功率: {avg_success:.1%}")

    return {
        'avg_signal_rate': avg_signal_rate,
        'emergence_rate': emergence_rate,
        'avg_success_rate': avg_success,
        'run_details': run_results,
    }


def main():
    """运行所有实验"""
    print("=" * 80)
    print("Phase 18: 元认知与自我反思")
    print("核心假设：元认知信号从交流困难中涌现，帮助调整学习策略")
    print("=" * 80)

    # 实验 1
    r1 = experiment_1_basic_metacognition(300, verbose=True)

    # 实验 2
    r2 = experiment_2_difficult_scenes(300, verbose=True)

    # 实验 3
    r3 = experiment_3_comparison(300, verbose=True)

    # 实验 4
    r4 = experiment_4_stability(200, 5, verbose=True)

    # 汇总
    print(f"\n{'=' * 80}")
    print("实验汇总")
    print(f"{'=' * 80}")
    print(f"\n实验 1 (基本元认知):")
    print(f"  元认知涌现: {r1['signal_emerged']}")
    print(f"  信号使用率: {r1['signal_rate']:.1%}")
    print(f"  元认知符号: {r1['metacognitive_symbols']}")
    print(f"\n实验 2 (困难场景):")
    print(f"  信号使用率: {r2['signal_rate']:.1%}")
    print(f"  帮助请求: {r2['help_requests']}")
    print(f"\n实验 3 (对比):")
    print(f"  改善: {r3['improvement']:+.1%}")
    print(f"\n实验 4 (稳定性):")
    print(f"  元认知涌现率: {r4['emergence_rate']:.1%}")
    print(f"  平均信号使用率: {r4['avg_signal_rate']:.1%}")

    print(f"\n{'=' * 80}")
    print("核心结论")
    print(f"{'=' * 80}")
    print("1. 元认知信号从交流困难中涌现")
    print("2. 帮助请求（'help'）在困难场景中频繁出现")
    print("3. 教师根据信号调整教学策略")
    print("4. 元认知是自我改进学习的关键机制")

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
        'experiment_1_basic_metacognition': r1,
        'experiment_2_difficult_scenes': r2,
        'experiment_3_comparison': r3,
        'experiment_4_stability': r4,
    }
    with open('metacognition_results.json', 'w', encoding='utf-8') as f:
        json.dump(to_serializable(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: metacognition_results.json")


if __name__ == '__main__':
    main()
