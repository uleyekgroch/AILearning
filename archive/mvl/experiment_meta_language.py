"""
Phase 46 实验：元语言——语言谈论语言本身

3 个实验：
1. 元语言涌现：追踪元语言符号的使用
2. 纠错效果：有元语言纠错 vs 无纠错的词汇收敛速度
3. 语言协商：2 个不同方言的 Agent 通过元语言协商达成一致
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
# 实验 1：元语言涌现
# ============================================================

def experiment_1_meta_emergence(num_rounds: int = 500,
                                verbose: bool = True) -> Dict:
    """
    追踪元语言纠错的涌现和效果

    2 个 Agent 交替进行参照游戏 + 元语言回合
    测量：成功率变化、元语言使用频率
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：元语言涌现")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    game_a = CommunicationGame()
    game_b = CommunicationGame()

    snapshots = []
    meta_rounds = 0
    meta_successes = 0

    for r in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)

        # 普通回合
        game_a.play_round(scene, target_idx)
        game_b.play_round(scene, target_idx)

        # 每 10 轮尝试一次元语言回合
        if (r + 1) % 10 == 0:
            meta_rounds += 1
            # A 描述，B 解释，如果失败则元语言纠错
            target = scene[target_idx]
            utterance = game_a.speaker.describe(target, scene)
            chosen_idx = game_b.listener.interpret(utterance, scene)
            if chosen_idx != target_idx:
                # 失败了，进行元语言回合（传入失败的 utterance 和 chosen_idx）
                success = game_a.play_meta_round(
                    scene, target_idx, game_b,
                    failed_utterance=utterance,
                    failed_chosen_idx=chosen_idx
                )
                if success:
                    meta_successes += 1

        if (r + 1) % 100 == 0:
            lang_a = game_a.language
            sr_a = lang_a.total_successes / lang_a.total_games if lang_a.total_games > 0 else 0
            overlap = _compute_language_overlap(game_a.language, game_b.language)
            snapshots.append({
                'round': r + 1,
                'success_rate_a': sr_a,
                'vocab_a': len(lang_a.vocabulary),
                'overlap': overlap,
                'meta_rounds': meta_rounds,
                'meta_successes': meta_successes,
            })
            if verbose:
                meta_rate = meta_successes / meta_rounds if meta_rounds > 0 else 0
                print(f"  轮次 {r+1}: 成功率={sr_a:.3f}, "
                      f"词汇={len(lang_a.vocabulary)}, "
                      f"重叠={overlap:.3f}, "
                      f"元语言成功率={meta_rate:.3f}")

    return {
        'snapshots': snapshots,
        'meta_rounds': meta_rounds,
        'meta_successes': meta_successes,
    }


# ============================================================
# 实验 2：纠错效果
# ============================================================

def experiment_2_correction_effect(num_rounds: int = 500,
                                   verbose: bool = True) -> Dict:
    """
    有元语言纠错 vs 无纠错的词汇收敛速度

    A: 有元语言纠错（失败后进行元语言回合）
    B: 无纠错（只进行普通回合）

    测量：词汇收敛速度、最终成功率
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：纠错效果")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    # A: 有元语言纠错
    if verbose:
        print(f"\n  A: 有元语言纠错（{num_rounds} 轮）...")
    game_a1 = CommunicationGame()
    game_a2 = CommunicationGame()
    checkpoints_a = []

    for r in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)

        game_a1.play_round(scene, target_idx)

        # 每 5 轮尝试元语言纠错
        if (r + 1) % 5 == 0:
            target = scene[target_idx]
            utterance = game_a1.speaker.describe(target, scene)
            chosen_idx = game_a2.listener.interpret(utterance, scene)
            if chosen_idx != target_idx:
                game_a1.play_meta_round(
                    scene, target_idx, game_a2,
                    failed_utterance=utterance,
                    failed_chosen_idx=chosen_idx
                )

        if (r + 1) % 100 == 0:
            sr = game_a1.language.total_successes / game_a1.language.total_games
            overlap = _compute_language_overlap(game_a1.language, game_a2.language)
            checkpoints_a.append({
                'round': r + 1,
                'success_rate': sr,
                'vocab_size': len(game_a1.language.vocabulary),
                'overlap': overlap,
            })
            if verbose:
                print(f"    轮次 {r+1}: 成功率={sr:.3f}, "
                      f"词汇={len(game_a1.language.vocabulary)}, "
                      f"重叠={overlap:.3f}")

    # B: 无纠错
    if verbose:
        print(f"\n  B: 无纠错（{num_rounds} 轮）...")
    game_b1 = CommunicationGame()
    game_b2 = CommunicationGame()
    checkpoints_b = []

    for r in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)

        game_b1.play_round(scene, target_idx)
        game_b2.play_round(scene, target_idx)

        if (r + 1) % 100 == 0:
            sr = game_b1.language.total_successes / game_b1.language.total_games
            overlap = _compute_language_overlap(game_b1.language, game_b2.language)
            checkpoints_b.append({
                'round': r + 1,
                'success_rate': sr,
                'vocab_size': len(game_b1.language.vocabulary),
                'overlap': overlap,
            })
            if verbose:
                print(f"    轮次 {r+1}: 成功率={sr:.3f}, "
                      f"词汇={len(game_b1.language.vocabulary)}, "
                      f"重叠={overlap:.3f}")

    return {
        'with_meta': checkpoints_a,
        'without_meta': checkpoints_b,
    }


# ============================================================
# 实验 3：语言协商
# ============================================================

def experiment_3_negotiation(num_rounds: int = 500,
                             verbose: bool = True) -> Dict:
    """
    2 个不同方言的 Agent 通过元语言协商达成一致

    步骤：
    1. 两个 Agent 各自独立学习 200 轮（形成不同方言）
    2. 开始交流 + 元语言协商 300 轮
    3. 测量词汇一致率变化

    预期：元语言协商加速词汇趋同
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：语言协商")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    # Phase 1: 独立学习（形成方言）
    if verbose:
        print("  Phase 1: 独立学习 200 轮（形成方言）...")
    game_a = CommunicationGame()
    game_b = CommunicationGame()

    for _ in range(200):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        game_a.play_round(scene, target_idx)

        # B 用不同的随机种子学习（通过不同场景）
        env2 = MultiAgent3DEnv(bounds=bounds)
        env2.physics.add_random_objects(num_objects, shapes=shapes,
                                        materials=materials)
        scene2 = _get_scene_features(env2)
        if len(scene2) < 2:
            continue
        target_idx2 = random.randint(0, len(scene2) - 1)
        game_b.play_round(scene2, target_idx2)

    overlap_initial = _compute_language_overlap(game_a.language, game_b.language)
    if verbose:
        print(f"    初始重叠: {overlap_initial:.3f}")
        print(f"    A 词汇: {len(game_a.language.vocabulary)}")
        print(f"    B 词汇: {len(game_b.language.vocabulary)}")

    # Phase 2: 交流 + 元语言协商
    if verbose:
        print(f"\n  Phase 2: 交流 + 协商 {num_rounds} 轮...")

    overlaps = []
    for r in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)

        # A 描述，B 解释
        game_a.play_round(scene, target_idx)

        # 每 3 轮进行元语言协商
        if (r + 1) % 3 == 0:
            target = scene[target_idx]
            utterance = game_a.speaker.describe(target, scene)
            chosen_idx = game_b.listener.interpret(utterance, scene)
            if chosen_idx != target_idx:
                game_a.play_meta_round(
                    scene, target_idx, game_b,
                    failed_utterance=utterance,
                    failed_chosen_idx=chosen_idx
                )

        # B 也描述，A 解释
        game_b.play_round(scene, target_idx)
        if (r + 1) % 3 == 0:
            target = scene[target_idx]
            utterance = game_b.speaker.describe(target, scene)
            chosen_idx = game_a.listener.interpret(utterance, scene)
            if chosen_idx != target_idx:
                game_b.play_meta_round(
                    scene, target_idx, game_a,
                    failed_utterance=utterance,
                    failed_chosen_idx=chosen_idx
                )

        if (r + 1) % 50 == 0:
            overlap = _compute_language_overlap(game_a.language, game_b.language)
            overlaps.append({
                'round': r + 1,
                'overlap': overlap,
            })
            if verbose:
                print(f"    轮次 {r+1}: 重叠={overlap:.3f}")

    overlap_final = _compute_language_overlap(game_a.language, game_b.language)
    if verbose:
        print(f"\n  结果:")
        print(f"    重叠率: {overlap_initial:.3f} → {overlap_final:.3f}")

    return {
        'initial_overlap': overlap_initial,
        'final_overlap': overlap_final,
        'overlaps': overlaps,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 46: 元语言——语言谈论语言本身")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    results['exp1'] = experiment_1_meta_emergence(
        num_rounds=500, verbose=True
    )

    results['exp2'] = experiment_2_correction_effect(
        num_rounds=500, verbose=True
    )

    results['exp3'] = experiment_3_negotiation(
        num_rounds=300, verbose=True
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

    with open('meta_language_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 meta_language_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 46 总结")
    print("=" * 60)

    exp1 = results['exp1']
    print(f"\n  元语言涌现:")
    print(f"    元语言回合: {exp1['meta_rounds']}")
    print(f"    元语言成功: {exp1['meta_successes']}")
    if exp1['meta_rounds'] > 0:
        print(f"    元语言成功率: {exp1['meta_successes']/exp1['meta_rounds']:.3f}")

    exp3 = results['exp3']
    print(f"\n  语言协商:")
    print(f"    重叠率: {exp3['initial_overlap']:.3f} → {exp3['final_overlap']:.3f}")
    print(f"    变化: {exp3['final_overlap'] - exp3['initial_overlap']:+.3f}")

    return results


if __name__ == '__main__':
    main()
