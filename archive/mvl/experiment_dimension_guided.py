"""
Phase 40 实验：维度级词汇引导

核心问题：维度级知识（shape/material/... 的整体成功率）是否比符号级知识（单个符号的成功率）
更能引导描述生成？

3 个实验：
1. 维度偏好演化：追踪维度成功率如何随训练轮次变化
2. 三种引导对比：维度引导 vs 符号引导 vs 随机选择
3. 维度知识迁移：迁移维度统计 vs 迁移符号统计 vs 从零学习
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
    _symbol_category
)


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
    window_count = 0

    for round_num in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        success = game.play_round(scene, target_idx)

        window_successes += 1 if success else 0
        window_count += 1

        if (round_num + 1) % window == 0 and window_count > 0:
            results.append({
                'round': round_num + 1,
                'success_rate': window_successes / window_count,
            })
            window_successes = 0
            window_count = 0

    return {'curve': results, 'language': game.language}


# ============================================================
# 实验 1：维度偏好演化
# ============================================================

def experiment_1_dimension_evolution(num_rounds: int = 300,
                                     verbose: bool = True) -> Dict:
    """
    追踪维度成功率如何随训练轮次演化

    每 50 轮记录一次各维度的成功率，观察哪个维度先成熟。
    预期：material/shape 先成熟（区分度高），color 后成熟
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：维度偏好演化")
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
            snapshot = {
                'round': round_num + 1,
                'vocab_size': len(lang.vocabulary),
                'dimension_stats': dict(lang.dimension_stats),
            }
            snapshots.append(snapshot)
            if verbose:
                dims = lang.dimension_stats
                dim_str = ', '.join(
                    f"{d}={s['success_rate']:.2f}({s['frequency']})"
                    for d, s in sorted(dims.items(),
                                       key=lambda x: -x[1].get('success_rate', 0))
                )
                print(f"  轮次 {round_num+1}: 词汇={len(lang.vocabulary)}, "
                      f"维度=[{dim_str}]")

    # 最终维度排名
    final = game.language.dimension_stats
    ranking = sorted(final.items(),
                     key=lambda x: -x[1].get('success_rate', 0))

    if verbose:
        print(f"\n维度排名（最终）:")
        for dim, stats in ranking:
            print(f"  {dim}: 成功率={stats['success_rate']:.3f}, "
                  f"频率={stats['frequency']}")

    return {
        'snapshots': snapshots,
        'final_ranking': [(d, s['success_rate']) for d, s in ranking],
        'total_games': game.language.total_games,
    }


# ============================================================
# 实验 2：三种引导对比
# ============================================================

def _train_with_disabled_dimension_guidance(num_rounds: int, bounds,
                                             shapes, materials,
                                             num_objects: int) -> EmergingLanguage:
    """训练语言系统，但禁用维度引导（只用符号级分数）"""
    # 创建一个临时补丁：清空 dimension_stats 使 _candidate_score 退化为符号级
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
    # 清空维度统计，迫使 _candidate_score 只用符号级
    game.language.dimension_stats = {}
    return game.language


def _train_random_guidance(num_rounds: int, bounds, shapes, materials,
                           num_objects: int) -> EmergingLanguage:
    """训练语言系统，但随机选择（禁用所有词汇引导）"""
    # 临时禁用 _candidate_score：直接返回 0
    game = CommunicationGame()
    original_score = Speaker._candidate_score

    def zero_score(self, symbols):
        return 0.0

    Speaker._candidate_score = zero_score
    try:
        for _ in range(num_rounds):
            env = MultiAgent3DEnv(bounds=bounds)
            env.physics.add_random_objects(num_objects, shapes=shapes,
                                           materials=materials)
            scene = _get_scene_features(env)
            if len(scene) < 2:
                continue
            target_idx = random.randint(0, len(scene) - 1)
            game.play_round(scene, target_idx)
    finally:
        Speaker._candidate_score = original_score

    return game.language


def experiment_2_three_guidance_modes(num_rounds: int = 200,
                                      verbose: bool = True) -> Dict:
    """
    三种引导模式对比

    A: 维度引导（默认）— 维度级 + 符号级融合
    B: 符号引导 — 只用符号级成功率
    C: 随机选择 — 禁用所有引导

    测量最终成功率和描述效率
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：三种引导模式对比")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    # A: 维度引导（默认）
    if verbose:
        print(f"\n  A: 维度引导（{num_rounds} 轮）...")
    lang_a = _train_language(num_rounds, bounds, shapes, materials, num_objects)
    sr_a = lang_a.total_successes / lang_a.total_games if lang_a.total_games > 0 else 0
    if verbose:
        print(f"    成功率={sr_a:.3f}, 词汇={len(lang_a.vocabulary)}")
        print(f"    维度统计: {dict(lang_a.dimension_stats)}")

    # B: 符号引导
    if verbose:
        print(f"\n  B: 符号引导（{num_rounds} 轮）...")
    lang_b = _train_with_disabled_dimension_guidance(
        num_rounds, bounds, shapes, materials, num_objects
    )
    sr_b = lang_b.total_successes / lang_b.total_games if lang_b.total_games > 0 else 0
    if verbose:
        print(f"    成功率={sr_b:.3f}, 词汇={len(lang_b.vocabulary)}")

    # C: 随机选择
    if verbose:
        print(f"\n  C: 随机选择（{num_rounds} 轮）...")
    lang_c = _train_random_guidance(num_rounds, bounds, shapes, materials, num_objects)
    sr_c = lang_c.total_successes / lang_c.total_games if lang_c.total_games > 0 else 0
    if verbose:
        print(f"    成功率={sr_c:.3f}, 词汇={len(lang_c.vocabulary)}")

    if verbose:
        print(f"\n对比:")
        print(f"  维度引导: {sr_a:.3f}")
        print(f"  符号引导: {sr_b:.3f}")
        print(f"  随机选择: {sr_c:.3f}")
        print(f"  维度 vs 符号: {sr_a - sr_b:+.3f}")
        print(f"  维度 vs 随机: {sr_a - sr_c:+.3f}")

    return {
        'dimension_guided': sr_a,
        'symbol_guided': sr_b,
        'random': sr_c,
        'dim_vs_sym': sr_a - sr_b,
        'dim_vs_random': sr_a - sr_c,
        'vocab_a': len(lang_a.vocabulary),
        'vocab_b': len(lang_b.vocabulary),
        'vocab_c': len(lang_c.vocabulary),
    }


# ============================================================
# 实验 3：维度知识迁移
# ============================================================

def _build_dimension_only_state(lang: EmergingLanguage) -> dict:
    """只保留维度统计，清空符号统计"""
    state = lang.save_state()
    # 清空符号级统计
    state['vocabulary'] = {}
    state['collocations'] = {}
    state['ngram_patterns'] = {}
    state['grammar_rules'] = []
    # 保留维度级统计
    return state


def _build_symbol_only_state(lang: EmergingLanguage) -> dict:
    """只保留符号统计，清空维度统计"""
    state = lang.save_state()
    state['dimension_stats'] = {}
    return state


def experiment_3_dimension_transfer(num_train: int = 300,
                                    num_test: int = 200,
                                    verbose: bool = True) -> Dict:
    """
    维度知识迁移实验

    A: 完整迁移（符号 + 维度）
    B: 只迁移维度统计
    C: 只迁移符号统计
    D: 从零学习

    预期：维度迁移 > 符号迁移（维度是更高层次的抽象）
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：维度知识迁移")
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
              f"成功率={sr:.3f}")
        print(f"  维度统计: {dict(trained_lang.dimension_stats)}")

    # A: 完整迁移
    if verbose:
        print(f"\n  A: 完整迁移（{num_test} 轮）...")
    state_full = trained_lang.save_state()
    lang_full = EmergingLanguage()
    lang_full.load_state(state_full)
    result_a = _test_learning_curve(num_test, bounds, shapes, materials,
                                     num_objects, initial_language=lang_full)
    sr_a = np.mean([r['success_rate'] for r in result_a['curve']])

    # B: 只迁移维度
    if verbose:
        print(f"  B: 只迁移维度（{num_test} 轮）...")
    state_dim = _build_dimension_only_state(trained_lang)
    lang_dim = EmergingLanguage()
    lang_dim.load_state(state_dim)
    result_b = _test_learning_curve(num_test, bounds, shapes, materials,
                                     num_objects, initial_language=lang_dim)
    sr_b = np.mean([r['success_rate'] for r in result_b['curve']])

    # C: 只迁移符号
    if verbose:
        print(f"  C: 只迁移符号（{num_test} 轮）...")
    state_sym = _build_symbol_only_state(trained_lang)
    lang_sym = EmergingLanguage()
    lang_sym.load_state(state_sym)
    result_c = _test_learning_curve(num_test, bounds, shapes, materials,
                                     num_objects, initial_language=lang_sym)
    sr_c = np.mean([r['success_rate'] for r in result_c['curve']])

    # D: 从零学习
    if verbose:
        print(f"  D: 从零学习（{num_test} 轮）...")
    result_d = _test_learning_curve(num_test, bounds, shapes, materials,
                                     num_objects, initial_language=None)
    sr_d = np.mean([r['success_rate'] for r in result_d['curve']])

    if verbose:
        print(f"\n迁移效果对比:")
        print(f"  完整迁移:     {sr_a:.3f}")
        print(f"  只迁移维度:   {sr_b:.3f}")
        print(f"  只迁移符号:   {sr_c:.3f}")
        print(f"  从零学习:     {sr_d:.3f}")
        print(f"  维度 vs 从零: {sr_b - sr_d:+.3f}")
        print(f"  符号 vs 从零: {sr_c - sr_d:+.3f}")

    return {
        'full_transfer': sr_a,
        'dimension_only': sr_b,
        'symbol_only': sr_c,
        'from_scratch': sr_d,
        'dim_advantage': sr_b - sr_d,
        'sym_advantage': sr_c - sr_d,
        'full_curve': result_a['curve'],
        'dim_curve': result_b['curve'],
        'sym_curve': result_c['curve'],
        'scratch_curve': result_d['curve'],
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 40: 维度级词汇引导")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    results['exp1'] = experiment_1_dimension_evolution(
        num_rounds=300, verbose=True
    )

    results['exp2'] = experiment_2_three_guidance_modes(
        num_rounds=200, verbose=True
    )

    results['exp3'] = experiment_3_dimension_transfer(
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

    with open('dimension_guided_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 dimension_guided_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 40 总结")
    print("=" * 60)

    print(f"\n  维度演化:")
    for dim, rate in results['exp1']['final_ranking']:
        print(f"    {dim}: {rate:.3f}")

    print(f"\n  引导模式对比:")
    print(f"    维度引导: {results['exp2']['dimension_guided']:.3f}")
    print(f"    符号引导: {results['exp2']['symbol_guided']:.3f}")
    print(f"    随机选择: {results['exp2']['random']:.3f}")

    print(f"\n  迁移效果:")
    print(f"    完整迁移: {results['exp3']['full_transfer']:.3f}")
    print(f"    只迁移维度: {results['exp3']['dimension_only']:.3f}")
    print(f"    只迁移符号: {results['exp3']['symbol_only']:.3f}")
    print(f"    从零学习: {results['exp3']['from_scratch']:.3f}")

    return results


if __name__ == '__main__':
    main()
