"""
Phase 43 实验：内部语言——语言作为思维工具

3 个实验：
1. 内部语言涌现：追踪内部语言的使用和效果
2. 规划效果：有内部语言 vs 无内部语言的描述效率对比
3. 复杂场景优势：内部语言在不同场景复杂度下的边际效益
"""

import sys
import random
import numpy as np
import json
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from multi_agent_3d_env import MultiAgent3DEnv
from environment_3d import MATERIAL_NAMES
from language_emergence import (
    EmergingLanguage, CommunicationGame, Speaker, Listener
)


def _get_scene_features(env) -> List[Dict]:
    return [env.physics.get_object_features(obj.id) for obj in env.physics.objects]


def _train_with_inner_speech(num_rounds: int, bounds, shapes, materials,
                              num_objects: int = 12) -> EmergingLanguage:
    """使用内部语言训练语言系统"""
    game = CommunicationGame()
    for _ in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        game.play_round(scene, target_idx)
    return game.language


def _train_without_inner_speech(num_rounds: int, bounds, shapes, materials,
                                 num_objects: int = 12) -> EmergingLanguage:
    """不使用内部语言训练（对照组）"""
    game = CommunicationGame()
    # 临时禁用内部语言
    original_play = game.play_round

    def play_no_inner(scene_features, target_idx):
        if target_idx >= len(scene_features) or not scene_features:
            return False
        target = scene_features[target_idx]
        # 直接描述，不经过内部语言
        utterance = game.speaker.describe(target, scene_features)
        if not utterance:
            return False
        chosen_idx = game.listener.interpret(utterance, scene_features)
        success = (chosen_idx == target_idx)
        game.language.total_games += 1
        if success:
            game.language.total_successes += 1
        if len(utterance) > 1:
            game.language.multi_symbol_games += 1
        if len(utterance) >= 3:
            game.language.tri_symbol_games += 1
        game.language.record_usage(utterance, success)
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                game.language.record_collocation(
                    utterance[i], utterance[i + 1], success
                )
        if len(utterance) >= 2:
            game.language.record_ngram(utterance, success)
        game.language.record_compound_cooccurrence(utterance, success)
        game.language.check_compound_formation(utterance, success)
        if game.language.total_games % 50 == 0:
            game.language.prune_compounds(max_age=100)
        return success

    game.play_round = play_no_inner
    for _ in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        game.play_round(scene, target_idx)
    return game.language


def _measure_stats(language: EmergingLanguage, num_rounds: int,
                   bounds, shapes, materials,
                   num_objects: int = 12, use_inner: bool = True) -> Dict:
    """测量描述统计"""
    game = CommunicationGame()
    game.language = language
    game.speaker = Speaker(language)
    game.listener = Listener(language)

    desc_lengths = []
    successes = 0
    total = 0

    for _ in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        target = scene[target_idx]

        if use_inner:
            scene_model = game.speaker.inner_describe(scene)
            utterance = game.speaker.describe(target, scene,
                                              scene_model=scene_model)
        else:
            utterance = game.speaker.describe(target, scene)

        chosen_idx = game.listener.interpret(utterance, scene)
        success = (chosen_idx == target_idx)

        desc_lengths.append(len(utterance))
        if success:
            successes += 1
        total += 1

    return {
        'avg_length': np.mean(desc_lengths) if desc_lengths else 0,
        'success_rate': successes / total if total > 0 else 0,
    }


# ============================================================
# 实验 1：内部语言涌现
# ============================================================

def experiment_1_inner_speech_emergence(num_rounds: int = 500,
                                        verbose: bool = True) -> Dict:
    """
    追踪内部语言的涌现和效果

    预期：内部语言帮助 Speaker 选择更好的描述
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：内部语言涌现")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    game = CommunicationGame()
    snapshots = []

    for round_num in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)

        # 有内部语言的描述
        scene_model = game.speaker.inner_describe(scene)
        game.language.inner_speech_games += 1
        utterance_with = game.speaker.describe(
            scene[target_idx], scene, scene_model=scene_model
        )

        # 无内部语言的描述（对照）
        utterance_without = game.speaker.describe(scene[target_idx], scene)

        # 用有内部语言的描述进行游戏
        chosen_idx = game.listener.interpret(utterance_with, scene)
        success = (chosen_idx == target_idx)

        game.language.total_games += 1
        if success:
            game.language.total_successes += 1
        if len(utterance_with) > 1:
            game.language.multi_symbol_games += 1
        game.language.record_usage(utterance_with, success)
        if len(utterance_with) >= 2:
            for i in range(len(utterance_with) - 1):
                game.language.record_collocation(
                    utterance_with[i], utterance_with[i + 1], success
                )
        if len(utterance_with) >= 2:
            game.language.record_ngram(utterance_with, success)
        game.language.record_compound_cooccurrence(utterance_with, success)
        game.language.check_compound_formation(utterance_with, success)
        if game.language.total_games % 50 == 0:
            game.language.prune_compounds(max_age=100)

        if (round_num + 1) % 100 == 0:
            lang = game.language
            sr = lang.total_successes / lang.total_games if lang.total_games > 0 else 0
            # 比较有/无内部语言的描述长度差异
            len_diff = len(utterance_without) - len(utterance_with)
            snapshots.append({
                'round': round_num + 1,
                'success_rate': sr,
                'vocab_size': len(lang.vocabulary),
                'inner_speech_games': lang.inner_speech_games,
                'length_diff': len_diff,
            })
            if verbose:
                print(f"  轮次 {round_num+1}: 成功率={sr:.3f}, "
                      f"词汇={len(lang.vocabulary)}, "
                      f"长度差异={len_diff:+d}")

    return {
        'snapshots': snapshots,
        'total_games': game.language.total_games,
    }


# ============================================================
# 实验 2：规划效果对比
# ============================================================

def experiment_2_planning_effect(num_rounds: int = 500,
                                 verbose: bool = True) -> Dict:
    """
    有内部语言 vs 无内部语言的描述效率对比

    A: 有内部语言（默认）
    B: 无内部语言（直接描述）
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：规划效果对比")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    # A: 有内部语言
    if verbose:
        print(f"\n  A: 有内部语言（{num_rounds} 轮）...")
    lang_a = _train_with_inner_speech(num_rounds, bounds, shapes, materials, num_objects)
    stats_a = _measure_stats(lang_a, 100, bounds, shapes, materials, num_objects,
                             use_inner=True)
    if verbose:
        print(f"    成功率={stats_a['success_rate']:.3f}, "
              f"描述长度={stats_a['avg_length']:.1f}")

    # B: 无内部语言
    if verbose:
        print(f"\n  B: 无内部语言（{num_rounds} 轮）...")
    lang_b = _train_without_inner_speech(num_rounds, bounds, shapes, materials, num_objects)
    stats_b = _measure_stats(lang_b, 100, bounds, shapes, materials, num_objects,
                             use_inner=False)
    if verbose:
        print(f"    成功率={stats_b['success_rate']:.3f}, "
              f"描述长度={stats_b['avg_length']:.1f}")

    if verbose:
        print(f"\n对比:")
        print(f"  成功率: 有内部={stats_a['success_rate']:.3f}, "
              f"无内部={stats_b['success_rate']:.3f}, "
              f"差距={stats_a['success_rate'] - stats_b['success_rate']:+.3f}")
        print(f"  描述长度: 有内部={stats_a['avg_length']:.1f}, "
              f"无内部={stats_b['avg_length']:.1f}, "
              f"差距={stats_a['avg_length'] - stats_b['avg_length']:+.1f}")

    return {
        'with_inner': stats_a,
        'without_inner': stats_b,
        'success_rate_diff': stats_a['success_rate'] - stats_b['success_rate'],
        'length_diff': stats_a['avg_length'] - stats_b['avg_length'],
    }


# ============================================================
# 实验 3：复杂场景优势
# ============================================================

def experiment_3_complexity_advantage(verbose: bool = True) -> Dict:
    """
    内部语言在不同场景复杂度下的边际效益

    测试 8, 12, 20, 30 物体的场景
    预期：物体越多，内部语言的优势越明显
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：复杂场景优势")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_rounds = 300

    results = {}
    for num_objects in [8, 12, 20, 30]:
        if verbose:
            print(f"\n  {num_objects} 物体场景（{num_rounds} 轮）...")

        # 有内部语言
        lang_inner = _train_with_inner_speech(
            num_rounds, bounds, shapes, materials, num_objects
        )
        stats_inner = _measure_stats(
            lang_inner, 100, bounds, shapes, materials, num_objects, use_inner=True
        )

        # 无内部语言
        lang_no_inner = _train_without_inner_speech(
            num_rounds, bounds, shapes, materials, num_objects
        )
        stats_no_inner = _measure_stats(
            lang_no_inner, 100, bounds, shapes, materials, num_objects, use_inner=False
        )

        results[num_objects] = {
            'with_inner': stats_inner,
            'without_inner': stats_no_inner,
            'success_diff': stats_inner['success_rate'] - stats_no_inner['success_rate'],
            'length_diff': stats_inner['avg_length'] - stats_no_inner['avg_length'],
        }

        if verbose:
            print(f"    有内部: 成功率={stats_inner['success_rate']:.3f}, "
                  f"长度={stats_inner['avg_length']:.1f}")
            print(f"    无内部: 成功率={stats_no_inner['success_rate']:.3f}, "
                  f"长度={stats_no_inner['avg_length']:.1f}")
            print(f"    差距: 成功率={results[num_objects]['success_diff']:+.3f}, "
                  f"长度={results[num_objects]['length_diff']:+.1f}")

    return results


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 43: 内部语言——语言作为思维工具")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    results['exp1'] = experiment_1_inner_speech_emergence(
        num_rounds=500, verbose=True
    )

    results['exp2'] = experiment_2_planning_effect(
        num_rounds=500, verbose=True
    )

    results['exp3'] = experiment_3_complexity_advantage(verbose=True)

    # 保存结果
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open('inner_speech_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 inner_speech_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 43 总结")
    print("=" * 60)

    exp2 = results['exp2']
    print(f"\n  规划效果:")
    print(f"    成功率差距: {exp2['success_rate_diff']:+.3f}")
    print(f"    描述长度差距: {exp2['length_diff']:+.1f}")

    exp3 = results['exp3']
    print(f"\n  复杂场景优势:")
    for n_objs, data in exp3.items():
        print(f"    {n_objs} 物体: 成功率差距={data['success_diff']:+.3f}, "
              f"长度差距={data['length_diff']:+.1f}")

    return results


if __name__ == '__main__':
    main()
