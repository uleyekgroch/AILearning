"""
Phase 39 实验：词汇引导行为——从"记录"到"知识"

3 个实验：
1. 大概念空间：12 物体 × 7 材料，词汇引导 vs 随机选择
2. 迁移优势：训练→测试，迁移+学习 vs 从零学习
3. 学习效率：不同物体数量 (4/8/12/16)，词汇覆盖率变化
"""

import sys
import random
import numpy as np
import json
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from multi_agent_3d_env import MultiAgent3DEnv
from environment_3d import MATERIAL_NAMES
from language_emergence import EmergingLanguage, CommunicationGame, Speaker, Listener


def _get_scene_features(env) -> List[Dict]:
    """从 3D 环境获取物体特征列表"""
    return [env.physics.get_object_features(obj.id) for obj in env.physics.objects]


def _train_language(num_rounds: int, bounds, shapes, materials,
                    num_objects: int = 12) -> EmergingLanguage:
    """在指定环境中训练语言系统"""
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


def _test_learning_curve(num_rounds: int, bounds, shapes, materials,
                          num_objects: int = 12,
                          initial_language: EmergingLanguage = None,
                          window: int = 20) -> Dict:
    """测试学习曲线"""
    game = CommunicationGame()
    if initial_language is not None:
        game.language = initial_language
        game.speaker = Speaker(initial_language)
        game.listener = Listener(initial_language)

    results = []
    window_successes = 0
    window_symbols = 0
    window_count = 0

    for round_num in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue

        target_idx = random.randint(0, len(scene) - 1)
        target = scene[target_idx]
        utterance = game.speaker.describe(target, scene)
        success = game.play_round(scene, target_idx)

        window_successes += 1 if success else 0
        window_symbols += len(utterance)
        window_count += 1

        if (round_num + 1) % window == 0 and window_count > 0:
            results.append({
                'round': round_num + 1,
                'success_rate': window_successes / window_count,
                'avg_symbols': window_symbols / window_count,
            })
            window_successes = 0
            window_symbols = 0
            window_count = 0

    return {'curve': results, 'language': game.language}


def _measure_vocab_influence(language: EmergingLanguage,
                              num_rounds: int, bounds, shapes,
                              materials, num_objects: int) -> Dict:
    """
    测量词汇对描述生成的实际影响

    比较：有词汇 vs 无词汇，描述是否不同
    """
    # 有词汇的 Speaker
    speaker_with = Speaker(language)
    # 无词汇的 Speaker
    empty_lang = EmergingLanguage()
    speaker_without = Speaker(empty_lang)

    same_count = 0
    diff_count = 0
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

        desc_with = speaker_with.describe(target, scene)
        desc_without = speaker_without.describe(target, scene)

        total += 1
        if desc_with == desc_without:
            same_count += 1
        else:
            diff_count += 1

    return {
        'same': same_count,
        'different': diff_count,
        'total': total,
        'influence_rate': diff_count / total if total > 0 else 0,
    }


# ============================================================
# 实验 1：大概念空间
# ============================================================

def experiment_1_large_concept_space(num_rounds: int = 200,
                                      verbose: bool = True) -> Dict:
    """
    大概念空间实验

    12 物体 × 7 材料，迫使使用多符号组合。
    测量词汇对描述生成的实际影响。

    预期：词汇影响率 > 10%（因为有多候选情况）
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：大概念空间（12 物体 × 7 材料）")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES  # 全部 7 种

    # 训练
    if verbose:
        print(f"  训练：{num_rounds} 轮...")
    lang = _train_language(num_rounds, bounds, shapes, materials, num_objects=12)
    if verbose:
        sr = lang.total_successes / lang.total_games if lang.total_games > 0 else 0
        print(f"  训练完成：词汇={len(lang.vocabulary)}, 成功率={sr:.3f}")

    # 测量词汇影响
    if verbose:
        print(f"  测量词汇影响（100 轮）...")
    influence = _measure_vocab_influence(
        lang, 100, bounds, shapes, materials, num_objects=12
    )

    if verbose:
        print(f"\n大概念空间结果:")
        print(f"  词汇影响率: {influence['influence_rate']:.3f} "
              f"({influence['different']}/{influence['total']})")
        print(f"  词汇量: {len(lang.vocabulary)}")

    return {
        'influence_rate': influence['influence_rate'],
        'vocab_size': len(lang.vocabulary),
        'training_sr': lang.total_successes / lang.total_games if lang.total_games > 0 else 0,
    }


# ============================================================
# 实验 2：迁移优势
# ============================================================

def experiment_2_transfer_advantage(num_train: int = 300,
                                     num_test: int = 200,
                                     verbose: bool = True) -> Dict:
    """
    迁移优势实验

    训练环境 A：12 物体 × 7 材料
    测试环境 B：同配置，不同随机物体

    迁移+学习 vs 从零学习
    预期：迁移 Agent 前期优势 > 5%
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：迁移优势（大概念空间）")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES

    # 训练
    if verbose:
        print(f"  训练：{num_train} 轮...")
    trained_lang = _train_language(num_train, bounds, shapes, materials, num_objects=12)
    if verbose:
        sr = trained_lang.total_successes / trained_lang.total_games if trained_lang.total_games > 0 else 0
        print(f"  训练完成：词汇={len(trained_lang.vocabulary)}, 成功率={sr:.3f}")

    # 迁移 + 继续学习
    if verbose:
        print(f"  迁移测试：{num_test} 轮...")
    saved = trained_lang.save_state()
    transfer_lang = EmergingLanguage()
    transfer_lang.load_state(saved)
    transfer_result = _test_learning_curve(
        num_test, bounds, shapes, materials,
        num_objects=12, initial_language=transfer_lang
    )

    # 基线
    if verbose:
        print(f"  基线测试：{num_test} 轮...")
    baseline_result = _test_learning_curve(
        num_test, bounds, shapes, materials,
        num_objects=12, initial_language=None
    )

    t_curve = transfer_result['curve']
    b_curve = baseline_result['curve']

    if verbose:
        print(f"\n学习曲线:")
        print(f"  {'轮次':>6}  {'迁移':>8}  {'基线':>8}  {'差距':>8}")
        for t, b in zip(t_curve, b_curve):
            diff = t['success_rate'] - b['success_rate']
            print(f"  {t['round']:6d}  {t['success_rate']:8.3f}  "
                  f"{b['success_rate']:8.3f}  {diff:+8.3f}")

    early_t = np.mean([r['success_rate'] for r in t_curve[:3]])
    early_b = np.mean([r['success_rate'] for r in b_curve[:3]])
    late_t = np.mean([r['success_rate'] for r in t_curve[-3:]])
    late_b = np.mean([r['success_rate'] for r in b_curve[-3:]])

    if verbose:
        print(f"\n  前期: 迁移={early_t:.3f}, 基线={early_b:.3f}, "
              f"差距={early_t - early_b:+.3f}")
        print(f"  后期: 迁移={late_t:.3f}, 基线={late_b:.3f}, "
              f"差距={late_t - late_b:+.3f}")

    return {
        'early_advantage': early_t - early_b,
        'late_advantage': late_t - late_b,
        'transfer_curve': t_curve,
        'baseline_curve': b_curve,
        'vocab_size': len(trained_lang.vocabulary),
    }


# ============================================================
# 实验 3：学习效率
# ============================================================

def experiment_3_learning_efficiency(num_rounds: int = 200,
                                      verbose: bool = True) -> Dict:
    """
    学习效率实验

    不同物体数量 (4/8/12/16)，测量：
    - 词汇覆盖率（词汇中有多少特征值）
    - 成功率
    - 平均描述长度

    预期：物体越多，词汇覆盖率越高，描述越长
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：学习效率（不同复杂度）")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES

    results = {}

    for num_obj in [4, 8, 12, 16]:
        if verbose:
            print(f"\n  物体数量={num_obj}: 训练 {num_rounds} 轮...")

        lang = _train_language(num_rounds, bounds, shapes, materials,
                                num_objects=num_obj)

        # 测量词汇覆盖率
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_obj, shapes=shapes, materials=materials)
        scene = _get_scene_features(env)

        all_feature_values = set()
        for f in scene:
            for v in f.values():
                if isinstance(v, str):
                    all_feature_values.add(v)

        vocab = set(lang.vocabulary.keys())
        coverage = len(vocab & all_feature_values) / len(all_feature_values) if all_feature_values else 0

        # 测量平均描述长度
        game = CommunicationGame()
        game.language = lang
        game.speaker = Speaker(lang)
        desc_lengths = []
        for _ in range(50):
            env2 = MultiAgent3DEnv(bounds=bounds)
            env2.physics.add_random_objects(num_obj, shapes=shapes, materials=materials)
            scene2 = _get_scene_features(env2)
            if len(scene2) < 2:
                continue
            target_idx = random.randint(0, len(scene2) - 1)
            target = scene2[target_idx]
            utterance = game.speaker.describe(target, scene2)
            desc_lengths.append(len(utterance))

        sr = lang.total_successes / lang.total_games if lang.total_games > 0 else 0
        avg_desc = np.mean(desc_lengths) if desc_lengths else 0

        results[num_obj] = {
            'success_rate': sr,
            'vocab_size': len(lang.vocabulary),
            'coverage': coverage,
            'avg_desc_length': avg_desc,
        }

        if verbose:
            print(f"    成功率={sr:.3f}, 词汇={len(lang.vocabulary)}, "
                  f"覆盖率={coverage:.3f}, 平均描述={avg_desc:.1f}")

    return results


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 39: 词汇引导行为")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    results['exp1'] = experiment_1_large_concept_space(
        num_rounds=200, verbose=True
    )

    results['exp2'] = experiment_2_transfer_advantage(
        num_train=300, num_test=200, verbose=True
    )

    results['exp3'] = experiment_3_learning_efficiency(
        num_rounds=200, verbose=True
    )

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

    with open('vocabulary_guided_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 vocabulary_guided_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 39 总结")
    print("=" * 60)
    print(f"  词汇影响率: {results['exp1']['influence_rate']:.3f}")
    print(f"  迁移优势: 前期 {results['exp2']['early_advantage']:+.3f}, "
          f"后期 {results['exp2']['late_advantage']:+.3f}")
    print(f"  学习效率:")
    for n, r in results['exp3'].items():
        print(f"    {n} 物体: 成功率={r['success_rate']:.3f}, "
              f"覆盖率={r['coverage']:.3f}, 描述={r['avg_desc_length']:.1f}")

    return results


if __name__ == '__main__':
    main()
