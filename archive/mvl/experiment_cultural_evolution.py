"""
Phase 45 实验：文化演化——语言跨代变化

3 个实验：
1. 代际简化：10 代语言演化，追踪词汇量和描述长度变化
2. 借词：2 个隔离群体，第 5 代开始交流后词汇趋同
3. 语义漂移：单群体 10 代，追踪符号成功率变化轨迹
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


def _train_generation(num_rounds: int, bounds, shapes, materials,
                      num_objects: int = 12,
                      seed_lang: EmergingLanguage = None) -> EmergingLanguage:
    """训练一代语言"""
    game = CommunicationGame()
    if seed_lang:
        state = seed_lang.save_state()
        game.language.load_state(state)
        game.speaker = Speaker(game.language)
        game.listener = Listener(game.language)

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
                   num_objects: int = 12) -> Dict:
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


def _compute_language_overlap(lang_a: EmergingLanguage,
                               lang_b: EmergingLanguage) -> float:
    """计算两个语言的词汇重叠率"""
    vocab_a = set(lang_a.vocabulary.keys())
    vocab_b = set(lang_b.vocabulary.keys())
    if not vocab_a or not vocab_b:
        return 0.0
    intersection = vocab_a & vocab_b
    union = vocab_a | vocab_b
    return len(intersection) / len(union) if union else 0.0


# ============================================================
# 实验 1：代际简化
# ============================================================

def experiment_1_generational_simplification(num_generations: int = 10,
                                              rounds_per_gen: int = 300,
                                              verbose: bool = True) -> Dict:
    """
    10 代语言演化，每代继承上一代的语言并变异

    追踪：词汇量、描述长度、复合符号数
    预期：语言逐渐简化，描述长度缩短
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：代际简化")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    generations = []
    current_lang = None

    for gen in range(num_generations):
        # 训练这一代
        lang = _train_generation(rounds_per_gen, bounds, shapes, materials,
                                 num_objects, seed_lang=current_lang)

        # 测量
        stats = _measure_stats(lang, 100, bounds, shapes, materials, num_objects)
        sr = lang.total_successes / lang.total_games if lang.total_games > 0 else 0

        gen_data = {
            'generation': gen + 1,
            'vocab_size': len(lang.vocabulary),
            'compound_count': len(lang.compounds),
            'success_rate': sr,
            'avg_length': stats['avg_length'],
            'total_games': lang.total_games,
        }
        generations.append(gen_data)

        if verbose:
            print(f"  代 {gen+1}: 词汇={len(lang.vocabulary)}, "
                  f"复合={len(lang.compounds)}, "
                  f"成功率={sr:.3f}, 长度={stats['avg_length']:.1f}")

        # 变异后传给下一代
        lang.mutate(mutation_rate=0.05)
        current_lang = lang

    return {'generations': generations}


# ============================================================
# 实验 2：借词
# ============================================================

def experiment_2_borrowing(num_generations: int = 10,
                           rounds_per_gen: int = 300,
                           verbose: bool = True) -> Dict:
    """
    2 个隔离群体，第 5 代开始交流后词汇趋同

    A: 群体 1（独立演化）
    B: 群体 2（独立演化，第 5 代开始从 A 借词）

    测量：词汇重叠率
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：借词")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    lang_a = None
    lang_b = None
    overlaps = []

    for gen in range(num_generations):
        # 群体 A：独立演化
        lang_a = _train_generation(rounds_per_gen, bounds, shapes, materials,
                                   num_objects, seed_lang=lang_a)
        lang_a.mutate(mutation_rate=0.05)

        # 群体 B：独立演化，第 5 代开始借词
        borrow_from = lang_a if gen >= 4 else None
        lang_b = _train_generation(rounds_per_gen, bounds, shapes, materials,
                                   num_objects, seed_lang=lang_b)
        lang_b.mutate(mutation_rate=0.05, borrow_from=borrow_from)

        overlap = _compute_language_overlap(lang_a, lang_b)
        overlaps.append({
            'generation': gen + 1,
            'overlap': overlap,
            'vocab_a': len(lang_a.vocabulary),
            'vocab_b': len(lang_b.vocabulary),
        })

        if verbose:
            borrow_str = " (借词)" if gen >= 4 else ""
            print(f"  代 {gen+1}: 重叠={overlap:.3f}, "
                  f"词汇A={len(lang_a.vocabulary)}, "
                  f"词汇B={len(lang_b.vocabulary)}{borrow_str}")

    return {'overlaps': overlaps}


# ============================================================
# 实验 3：语义漂移
# ============================================================

def experiment_3_semantic_drift(num_generations: int = 10,
                                rounds_per_gen: int = 300,
                                verbose: bool = True) -> Dict:
    """
    单群体 10 代，追踪符号成功率变化轨迹

    预期：符号成功率在变异后会波动，但通过交流学习能恢复
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：语义漂移")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    lang = None
    drift_data = []

    for gen in range(num_generations):
        # 训练
        lang = _train_generation(rounds_per_gen, bounds, shapes, materials,
                                 num_objects, seed_lang=lang)

        # 记录变异前的状态
        pre_rates = {s: d['success_rate'] for s, d in lang.vocabulary.items()}

        # 变异
        lang.mutate(mutation_rate=0.05)

        # 记录变异后的状态
        post_rates = {s: d['success_rate'] for s, d in lang.vocabulary.items()}

        # 计算漂移幅度
        drifts = []
        for sym in pre_rates:
            if sym in post_rates:
                drifts.append(abs(post_rates[sym] - pre_rates[sym]))

        avg_drift = np.mean(drifts) if drifts else 0
        sr = lang.total_successes / lang.total_games if lang.total_games > 0 else 0

        drift_data.append({
            'generation': gen + 1,
            'avg_drift': avg_drift,
            'success_rate': sr,
            'vocab_size': len(lang.vocabulary),
        })

        if verbose:
            print(f"  代 {gen+1}: 漂移={avg_drift:.4f}, "
                  f"成功率={sr:.3f}, 词汇={len(lang.vocabulary)}")

    return {'drift': drift_data}


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 45: 文化演化")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    results['exp1'] = experiment_1_generational_simplification(
        num_generations=10, rounds_per_gen=300, verbose=True
    )

    results['exp2'] = experiment_2_borrowing(
        num_generations=10, rounds_per_gen=300, verbose=True
    )

    results['exp3'] = experiment_3_semantic_drift(
        num_generations=10, rounds_per_gen=300, verbose=True
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

    with open('cultural_evolution_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 cultural_evolution_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 45 总结")
    print("=" * 60)

    exp1 = results['exp1']['generations']
    first = exp1[0]
    last = exp1[-1]
    print(f"\n  代际简化:")
    print(f"    词汇量: {first['vocab_size']} → {last['vocab_size']}")
    print(f"    描述长度: {first['avg_length']:.1f} → {last['avg_length']:.1f}")
    print(f"    成功率: {first['success_rate']:.3f} → {last['success_rate']:.3f}")

    exp2 = results['exp2']['overlaps']
    pre_borrow = exp2[3]['overlap']  # 第 4 代（借词前）
    post_borrow = exp2[9]['overlap']  # 第 10 代（借词后）
    print(f"\n  借词效果:")
    print(f"    重叠率: {pre_borrow:.3f}（借词前）→ {post_borrow:.3f}（借词后）")

    return results


if __name__ == '__main__':
    main()
