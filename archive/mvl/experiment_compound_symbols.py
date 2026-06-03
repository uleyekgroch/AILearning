"""
Phase 41 实验：复合符号生成——从固定词汇到创造性语言

3 个实验：
1. 复合符号涌现：追踪复合符号的生成过程
2. 描述效率：复合符号启用 vs 禁用的描述效率对比
3. 复合符号迁移：训练环境的复合符号在新场景中的复用率
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
    EmergingLanguage, CommunicationGame, Speaker, Listener,
    _is_compound, _expand_compound
)


def _get_scene_features(env) -> List[Dict]:
    """从 3D 环境获取物体特征列表"""
    return [env.physics.get_object_features(obj.id) for obj in env.physics.objects]


def _train_language(num_rounds: int, bounds, shapes, materials,
                    num_objects: int = 12,
                    enable_compounds: bool = True) -> EmergingLanguage:
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


def _measure_desc_stats(language: EmergingLanguage, num_rounds: int,
                        bounds, shapes, materials,
                        num_objects: int = 12) -> Dict:
    """测量描述统计：平均长度、复合符号使用率、成功率"""
    game = CommunicationGame()
    game.language = language
    game.speaker = Speaker(language)
    game.listener = Listener(language)

    desc_lengths = []
    compound_uses = 0
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

        utterance = game.speaker.describe(target, scene)
        chosen_idx = game.listener.interpret(utterance, scene)
        success = (chosen_idx == target_idx)

        desc_lengths.append(len(utterance))
        if any(_is_compound(s) for s in utterance):
            compound_uses += 1
        if success:
            successes += 1
        total += 1

    return {
        'avg_length': np.mean(desc_lengths) if desc_lengths else 0,
        'compound_usage_rate': compound_uses / total if total > 0 else 0,
        'success_rate': successes / total if total > 0 else 0,
        'total': total,
    }


# ============================================================
# 实验 1：复合符号涌现
# ============================================================

def experiment_1_compound_emergence(num_rounds: int = 500,
                                    verbose: bool = True) -> Dict:
    """
    追踪复合符号的涌现过程

    每 50 轮记录：复合符号数量、词汇量、描述效率
    预期：复合符号在 100+ 轮后开始涌现，描述长度逐渐缩短
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：复合符号涌现")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12
    checkpoint_interval = 50

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
        game.play_round(scene, target_idx)

        if (round_num + 1) % checkpoint_interval == 0:
            lang = game.language
            sr = lang.total_successes / lang.total_games if lang.total_games > 0 else 0

            # 测量描述效率
            stats = _measure_desc_stats(lang, 50, bounds, shapes, materials, num_objects)

            snapshot = {
                'round': round_num + 1,
                'success_rate': sr,
                'vocab_size': len(lang.vocabulary),
                'compound_count': len(lang.compounds),
                'avg_desc_length': stats['avg_length'],
                'compound_usage_rate': stats['compound_usage_rate'],
            }
            snapshots.append(snapshot)

            if verbose:
                print(f"  轮次 {round_num+1}: 成功率={sr:.3f}, "
                      f"词汇={len(lang.vocabulary)}, "
                      f"复合符号={len(lang.compounds)}, "
                      f"描述长度={stats['avg_length']:.1f}, "
                      f"复合使用率={stats['compound_usage_rate']:.2f}")

    # 最终复合符号列表
    if verbose:
        print(f"\n最终复合符号 ({len(game.language.compounds)} 个):")
        for sym, data in sorted(game.language.compounds.items(),
                                 key=lambda x: -x[1]['success_rate']):
            print(f"  {sym}: 组件={data['components']}, "
                  f"成功率={data['success_rate']:.2f}, "
                  f"频率={data['frequency']}")

    return {
        'snapshots': snapshots,
        'final_compounds': {k: v for k, v in game.language.compounds.items()},
        'total_games': game.language.total_games,
    }


# ============================================================
# 实验 2：描述效率
# ============================================================

def experiment_2_description_efficiency(num_rounds: int = 300,
                                        verbose: bool = True) -> Dict:
    """
    复合符号启用 vs 禁用的描述效率对比

    A: 复合符号启用（默认）
    B: 复合符号禁用（不生成复合符号）

    测量：成功率、平均描述长度、复合符号使用率
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：描述效率对比")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    # A: 复合符号启用
    if verbose:
        print(f"\n  A: 复合符号启用（{num_rounds} 轮）...")
    lang_a = _train_language(num_rounds, bounds, shapes, materials, num_objects)
    stats_a = _measure_desc_stats(lang_a, 100, bounds, shapes, materials, num_objects)
    if verbose:
        print(f"    成功率={stats_a['success_rate']:.3f}, "
              f"描述长度={stats_a['avg_length']:.1f}, "
              f"复合使用率={stats_a['compound_usage_rate']:.2f}, "
              f"复合符号={len(lang_a.compounds)}")

    # B: 复合符号禁用（通过禁用 check_compound_formation）
    if verbose:
        print(f"\n  B: 复合符号禁用（{num_rounds} 轮）...")
    # 临时禁用复合符号生成
    from language_emergence import EmergingLanguage as EL
    original_check = EL.check_compound_formation
    EL.check_compound_formation = lambda self, symbols, success: None
    try:
        lang_b = _train_language(num_rounds, bounds, shapes, materials, num_objects)
    finally:
        EL.check_compound_formation = original_check
    stats_b = _measure_desc_stats(lang_b, 100, bounds, shapes, materials, num_objects)
    if verbose:
        print(f"    成功率={stats_b['success_rate']:.3f}, "
              f"描述长度={stats_b['avg_length']:.1f}, "
              f"复合使用率={stats_b['compound_usage_rate']:.2f}, "
              f"复合符号={len(lang_b.compounds)}")

    if verbose:
        print(f"\n对比:")
        print(f"  成功率: 复合={stats_a['success_rate']:.3f}, "
              f"普通={stats_b['success_rate']:.3f}, "
              f"差距={stats_a['success_rate'] - stats_b['success_rate']:+.3f}")
        print(f"  描述长度: 复合={stats_a['avg_length']:.1f}, "
              f"普通={stats_b['avg_length']:.1f}, "
              f"差距={stats_a['avg_length'] - stats_b['avg_length']:+.1f}")

    return {
        'with_compounds': stats_a,
        'without_compounds': stats_b,
        'compound_count': len(lang_a.compounds),
        'success_rate_diff': stats_a['success_rate'] - stats_b['success_rate'],
        'length_diff': stats_a['avg_length'] - stats_b['avg_length'],
    }


# ============================================================
# 实验 3：复合符号迁移
# ============================================================

def experiment_3_compound_transfer(num_train: int = 300,
                                   num_test: int = 200,
                                   verbose: bool = True) -> Dict:
    """
    复合符号迁移实验

    训练环境 A → 测试环境 B（同配置，不同随机物体）

    测量：复合符号在新场景中的复用率
    预期：复合符号（shape-material）在新场景中仍然有效
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：复合符号迁移")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    # 训练
    if verbose:
        print(f"  训练：{num_train} 轮...")
    trained_lang = _train_language(num_train, bounds, shapes, materials, num_objects)
    if verbose:
        sr = trained_lang.total_successes / trained_lang.total_games if trained_lang.total_games > 0 else 0
        print(f"  训练完成：词汇={len(trained_lang.vocabulary)}, "
              f"复合符号={len(trained_lang.compounds)}, 成功率={sr:.3f}")

    # 测试复合符号在新场景中的有效性
    if verbose:
        print(f"\n  测试复合符号有效性（{num_test} 轮）...")

    # 测量：携带复合符号 vs 不携带
    # A: 携带完整语言（含复合符号）
    saved = trained_lang.save_state()
    lang_transfer = EmergingLanguage()
    lang_transfer.load_state(saved)
    stats_transfer = _measure_desc_stats(
        lang_transfer, num_test, bounds, shapes, materials, num_objects
    )

    # B: 只携带基础词汇（无复合符号）
    lang_basic = EmergingLanguage()
    basic_state = saved.copy()
    basic_state['compounds'] = {}
    lang_basic.load_state(basic_state)
    stats_basic = _measure_desc_stats(
        lang_basic, num_test, bounds, shapes, materials, num_objects
    )

    # C: 从零学习
    lang_fresh = EmergingLanguage()
    stats_fresh = _measure_desc_stats(
        lang_fresh, num_test, bounds, shapes, materials, num_objects
    )

    if verbose:
        print(f"\n迁移效果:")
        print(f"  {'条件':<15} {'成功率':>8} {'描述长度':>10} {'复合使用率':>10}")
        print(f"  {'完整迁移':<15} {stats_transfer['success_rate']:8.3f} "
              f"{stats_transfer['avg_length']:10.1f} "
              f"{stats_transfer['compound_usage_rate']:10.2f}")
        print(f"  {'基础词汇':<15} {stats_basic['success_rate']:8.3f} "
              f"{stats_basic['avg_length']:10.1f} "
              f"{stats_basic['compound_usage_rate']:10.2f}")
        print(f"  {'从零学习':<15} {stats_fresh['success_rate']:8.3f} "
              f"{stats_fresh['avg_length']:10.1f} "
              f"{stats_fresh['compound_usage_rate']:10.2f}")

    return {
        'transfer': stats_transfer,
        'basic': stats_basic,
        'fresh': stats_fresh,
        'compound_count': len(trained_lang.compounds),
        'transfer_vs_fresh': stats_transfer['success_rate'] - stats_fresh['success_rate'],
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 41: 复合符号生成")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    results['exp1'] = experiment_1_compound_emergence(
        num_rounds=500, verbose=True
    )

    results['exp2'] = experiment_2_description_efficiency(
        num_rounds=300, verbose=True
    )

    results['exp3'] = experiment_3_compound_transfer(
        num_train=300, num_test=200, verbose=True
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

    with open('compound_symbols_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 compound_symbols_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 41 总结")
    print("=" * 60)

    print(f"\n  复合符号涌现:")
    exp1 = results['exp1']
    print(f"    最终复合符号数: {len(exp1['final_compounds'])}")
    if exp1['snapshots']:
        first = exp1['snapshots'][0]
        last = exp1['snapshots'][-1]
        print(f"    描述长度: {first['avg_desc_length']:.1f} → {last['avg_desc_length']:.1f}")

    print(f"\n  描述效率:")
    exp2 = results['exp2']
    print(f"    成功率差距: {exp2['success_rate_diff']:+.3f}")
    print(f"    描述长度差距: {exp2['length_diff']:+.1f}")

    print(f"\n  复合符号迁移:")
    exp3 = results['exp3']
    print(f"    完整迁移 vs 从零: {exp3['transfer_vs_fresh']:+.3f}")
    print(f"    复合符号数: {exp3['compound_count']}")

    return results


if __name__ == '__main__':
    main()
