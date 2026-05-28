"""
Phase 42 实验：3+ 复合符号 + 符号淘汰

3 个实验：
1. 3+ 复合符号涌现：追踪不同长度复合符号的生成
2. 符号淘汰效果：淘汰机制对词汇量和描述效率的影响
3. 淘汰+迁移：淘汰后的语言在新场景中的表现
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
    return [env.physics.get_object_features(obj.id) for obj in env.physics.objects]


def _train_language(num_rounds: int, bounds, shapes, materials,
                    num_objects: int = 12) -> EmergingLanguage:
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
    }


# ============================================================
# 实验 1：3+ 复合符号涌现
# ============================================================

def experiment_1_multi_compound(num_rounds: int = 500,
                                verbose: bool = True) -> Dict:
    """
    追踪不同长度复合符号的涌现

    预期：2 组件先涌现，3+ 组件后涌现（阈值更低）
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：3+ 复合符号涌现")
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
            # 按组件数分类
            by_len = {}
            for sym, data in lang.compounds.items():
                n = len(data['components'])
                by_len.setdefault(n, []).append(sym)

            snapshot = {
                'round': round_num + 1,
                'vocab_size': len(lang.vocabulary),
                'compound_count': len(lang.compounds),
                'by_length': {n: len(syms) for n, syms in by_len.items()},
            }
            snapshots.append(snapshot)

            if verbose:
                len_str = ', '.join(f"{n}组件={len(syms)}"
                                   for n, syms in sorted(by_len.items()))
                print(f"  轮次 {round_num+1}: 词汇={len(lang.vocabulary)}, "
                      f"复合符号={len(lang.compounds)} ({len_str})")

    # 最终复合符号列表（按组件数分组）
    if verbose:
        lang = game.language
        by_len = {}
        for sym, data in lang.compounds.items():
            n = len(data['components'])
            by_len.setdefault(n, []).append((sym, data))

        for n in sorted(by_len.keys()):
            print(f"\n{n} 组件复合符号 ({len(by_len[n])} 个):")
            for sym, data in sorted(by_len[n], key=lambda x: -x[1]['success_rate']):
                print(f"  {sym}: 组件={data['components']}, "
                      f"成功率={data['success_rate']:.2f}, "
                      f"频率={data['frequency']}")

    return {
        'snapshots': snapshots,
        'final_compounds': dict(game.language.compounds),
    }


# ============================================================
# 实验 2：符号淘汰效果
# ============================================================

def experiment_2_pruning(num_rounds: int = 500,
                         verbose: bool = True) -> Dict:
    """
    符号淘汰效果

    A: 有淘汰（每 50 轮淘汰 100 轮未使用的复合符号）
    B: 无淘汰（所有复合符号永久保留）

    测量：活跃复合符号数、描述效率
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：符号淘汰效果")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    # A: 有淘汰
    if verbose:
        print(f"\n  A: 有淘汰（{num_rounds} 轮）...")
    lang_a = _train_language(num_rounds, bounds, shapes, materials, num_objects)
    stats_a = _measure_desc_stats(lang_a, 100, bounds, shapes, materials, num_objects)

    # 统计活跃 vs 总复合符号
    active_a = sum(1 for d in lang_a.compounds.values()
                   if lang_a.total_games - d.get('last_used', 0) <= 100)
    if verbose:
        print(f"    复合符号: {len(lang_a.compounds)} (活跃={active_a})")
        print(f"    成功率={stats_a['success_rate']:.3f}, "
              f"描述长度={stats_a['avg_length']:.1f}")

    # B: 无淘汰（禁用 prune_compounds）
    if verbose:
        print(f"\n  B: 无淘汰（{num_rounds} 轮）...")
    original_prune = EmergingLanguage.prune_compounds
    EmergingLanguage.prune_compounds = lambda self, max_age=100: None
    try:
        lang_b = _train_language(num_rounds, bounds, shapes, materials, num_objects)
    finally:
        EmergingLanguage.prune_compounds = original_prune
    stats_b = _measure_desc_stats(lang_b, 100, bounds, shapes, materials, num_objects)

    active_b = len(lang_b.compounds)  # 无淘汰时全部活跃
    if verbose:
        print(f"    复合符号: {len(lang_b.compounds)} (全部活跃)")
        print(f"    成功率={stats_b['success_rate']:.3f}, "
              f"描述长度={stats_b['avg_length']:.1f}")

    if verbose:
        print(f"\n对比:")
        print(f"  复合符号数: 有淘汰={len(lang_a.compounds)}, "
              f"无淘汰={len(lang_b.compounds)}")
        print(f"  成功率: 有淘汰={stats_a['success_rate']:.3f}, "
              f"无淘汰={stats_b['success_rate']:.3f}")
        print(f"  描述长度: 有淘汰={stats_a['avg_length']:.1f}, "
              f"无淘汰={stats_b['avg_length']:.1f}")

    return {
        'with_pruning': {
            'compounds': len(lang_a.compounds),
            'active': active_a,
            **stats_a,
        },
        'without_pruning': {
            'compounds': len(lang_b.compounds),
            'active': active_b,
            **stats_b,
        },
    }


# ============================================================
# 实验 3：淘汰+迁移
# ============================================================

def experiment_3_pruned_transfer(num_train: int = 400,
                                 num_test: int = 200,
                                 verbose: bool = True) -> Dict:
    """
    淘汰后的语言迁移

    训练 400 轮（足够生成 3+ 复合符号），然后迁移测试

    A: 完整迁移（含所有复合符号）
    B: 只迁移活跃复合符号
    C: 从零学习
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：淘汰+迁移")
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

        # 按组件数统计
        by_len = {}
        for sym, data in trained_lang.compounds.items():
            n = len(data['components'])
            by_len.setdefault(n, []).append(sym)
        for n in sorted(by_len.keys()):
            print(f"    {n} 组件: {len(by_len[n])} 个")

    # A: 完整迁移
    if verbose:
        print(f"\n  A: 完整迁移（{num_test} 轮）...")
    state_full = trained_lang.save_state()
    lang_full = EmergingLanguage()
    lang_full.load_state(state_full)
    stats_a = _measure_desc_stats(lang_full, num_test, bounds, shapes, materials, num_objects)

    # B: 只迁移活跃复合符号（淘汰不活跃的）
    if verbose:
        print(f"  B: 只迁移活跃复合符号（{num_test} 轮）...")
    state_active = trained_lang.save_state()
    # 移除不活跃的复合符号
    active_compounds = {}
    for sym, data in trained_lang.compounds.items():
        if trained_lang.total_games - data.get('last_used', 0) <= 100:
            active_compounds[sym] = data
    state_active['compounds'] = active_compounds
    lang_active = EmergingLanguage()
    lang_active.load_state(state_active)
    stats_b = _measure_desc_stats(lang_active, num_test, bounds, shapes, materials, num_objects)

    # C: 从零学习
    if verbose:
        print(f"  C: 从零学习（{num_test} 轮）...")
    lang_fresh = EmergingLanguage()
    stats_c = _measure_desc_stats(lang_fresh, num_test, bounds, shapes, materials, num_objects)

    if verbose:
        print(f"\n迁移效果:")
        print(f"  {'条件':<20} {'成功率':>8} {'描述长度':>10} {'复合使用率':>10}")
        print(f"  {'完整迁移':<20} {stats_a['success_rate']:8.3f} "
              f"{stats_a['avg_length']:10.1f} "
              f"{stats_a['compound_usage_rate']:10.2f}")
        print(f"  {'活跃复合迁移':<20} {stats_b['success_rate']:8.3f} "
              f"{stats_b['avg_length']:10.1f} "
              f"{stats_b['compound_usage_rate']:10.2f}")
        print(f"  {'从零学习':<20} {stats_c['success_rate']:8.3f} "
              f"{stats_c['avg_length']:10.1f} "
              f"{stats_c['compound_usage_rate']:10.2f}")

    return {
        'full_transfer': stats_a,
        'active_transfer': stats_b,
        'from_scratch': stats_c,
        'total_compounds': len(trained_lang.compounds),
        'active_compounds': len(active_compounds),
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 42: 3+ 复合符号 + 符号淘汰")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    results['exp1'] = experiment_1_multi_compound(
        num_rounds=800, verbose=True
    )

    results['exp2'] = experiment_2_pruning(
        num_rounds=500, verbose=True
    )

    results['exp3'] = experiment_3_pruned_transfer(
        num_train=600, num_test=200, verbose=True
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

    with open('compound_pruning_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 compound_pruning_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 42 总结")
    print("=" * 60)

    exp1 = results['exp1']
    print(f"\n  3+ 复合符号涌现:")
    if exp1['snapshots']:
        last = exp1['snapshots'][-1]
        print(f"    最终复合符号: {last['compound_count']}")
        print(f"    按长度分布: {last['by_length']}")

    exp2 = results['exp2']
    print(f"\n  符号淘汰:")
    print(f"    有淘汰: {exp2['with_pruning']['compounds']} 个 "
          f"(活跃={exp2['with_pruning']['active']})")
    print(f"    无淘汰: {exp2['without_pruning']['compounds']} 个")

    exp3 = results['exp3']
    print(f"\n  淘汰+迁移:")
    print(f"    完整迁移: {exp3['full_transfer']['success_rate']:.3f}")
    print(f"    活跃迁移: {exp3['active_transfer']['success_rate']:.3f}")
    print(f"    从零学习: {exp3['from_scratch']['success_rate']:.3f}")

    return results


if __name__ == '__main__':
    main()
