"""
Phase 44 实验：主动教学——根据学习者调整描述

3 个实验：
1. 教学 vs 被动观察：教师用教学模式描述 vs 普通描述
2. 脚手架效果：有脚手架（限制新符号数）vs 无脚手架
3. 纠错效果：有纠错反馈 vs 无纠错
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


# ============================================================
# 实验 1：教学 vs 被动观察
# ============================================================

def experiment_1_teaching_vs_passive(num_rounds: int = 500,
                                     verbose: bool = True) -> Dict:
    """
    教师用教学模式描述 vs 普通描述

    A: 教学模式——教师优先使用学生已知的符号
    B: 被动模式——教师用自己的语言描述，学生自己学习

    测量：学生的词汇量、成功率
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：教学 vs 被动观察")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    # 教师先自己学习 200 轮，建立词汇表
    if verbose:
        print("  教师预训练 200 轮...")
    teacher_game = CommunicationGame()
    for _ in range(200):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        teacher_game.play_round(scene, target_idx)
    teacher_lang = teacher_game.language

    # A: 教学模式
    if verbose:
        print(f"\n  A: 教学模式（{num_rounds} 轮）...")
    student_a = CommunicationGame()
    teacher_a = CommunicationGame()
    teacher_a.language = teacher_lang
    teacher_a.speaker = Speaker(teacher_lang)

    successes_a = 0
    total_a = 0
    for _ in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        target = scene[target_idx]

        # 教师用教学模式描述（优先使用学生已知的符号）
        utterance = teacher_a.speaker.describe(
            target, scene,
            listener_vocab=student_a.language.vocabulary
        )

        # 学生解释
        chosen_idx = student_a.listener.interpret(utterance, scene)
        success = (chosen_idx == target_idx)

        # 学生更新语言
        student_a.language.total_games += 1
        if success:
            student_a.language.total_successes += 1
            successes_a += 1
        total_a += 1
        student_a.language.record_usage(utterance, success)
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                student_a.language.record_collocation(
                    utterance[i], utterance[i + 1], success
                )
        if len(utterance) >= 2:
            student_a.language.record_ngram(utterance, success)

    sr_a = successes_a / total_a if total_a > 0 else 0
    if verbose:
        print(f"    学生词汇={len(student_a.language.vocabulary)}, "
              f"成功率={sr_a:.3f}")

    # B: 被动模式（教师用自己的语言，不考虑学生）
    if verbose:
        print(f"\n  B: 被动模式（{num_rounds} 轮）...")
    student_b = CommunicationGame()
    teacher_b_game = CommunicationGame()
    teacher_b_game.language = teacher_lang
    teacher_b_game.speaker = Speaker(teacher_lang)

    successes_b = 0
    total_b = 0
    for _ in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        target = scene[target_idx]

        # 教师用自己的语言描述（不考虑学生）
        utterance = teacher_b_game.speaker.describe(target, scene)

        # 学生解释
        chosen_idx = student_b.listener.interpret(utterance, scene)
        success = (chosen_idx == target_idx)

        student_b.language.total_games += 1
        if success:
            student_b.language.total_successes += 1
            successes_b += 1
        total_b += 1
        student_b.language.record_usage(utterance, success)
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                student_b.language.record_collocation(
                    utterance[i], utterance[i + 1], success
                )
        if len(utterance) >= 2:
            student_b.language.record_ngram(utterance, success)

    sr_b = successes_b / total_b if total_b > 0 else 0
    if verbose:
        print(f"    学生词汇={len(student_b.language.vocabulary)}, "
              f"成功率={sr_b:.3f}")

    if verbose:
        print(f"\n对比:")
        print(f"  词汇量: 教学={len(student_a.language.vocabulary)}, "
              f"被动={len(student_b.language.vocabulary)}")
        print(f"  成功率: 教学={sr_a:.3f}, 被动={sr_b:.3f}, "
              f"差距={sr_a - sr_b:+.3f}")

    return {
        'teaching': {
            'vocab_size': len(student_a.language.vocabulary),
            'success_rate': sr_a,
        },
        'passive': {
            'vocab_size': len(student_b.language.vocabulary),
            'success_rate': sr_b,
        },
        'success_rate_diff': sr_a - sr_b,
        'vocab_diff': len(student_a.language.vocabulary) - len(student_b.language.vocabulary),
    }


# ============================================================
# 实验 2：脚手架效果
# ============================================================

def experiment_2_scaffolding(num_rounds: int = 500,
                             verbose: bool = True) -> Dict:
    """
    有脚手架 vs 无脚手架

    A: 有脚手架——每次最多引入 1 个新符号
    B: 无脚手架——可以引入任意数量新符号（但仍优先已知符号）

    测量：学生学习速度
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：脚手架效果")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    # 教师预训练
    teacher_game = CommunicationGame()
    for _ in range(200):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        teacher_game.play_round(scene, target_idx)
    teacher_lang = teacher_game.language

    # A: 有脚手架（describe with listener_vocab）
    if verbose:
        print(f"\n  A: 有脚手架（{num_rounds} 轮）...")
    student_a = CommunicationGame()
    teacher_a = CommunicationGame()
    teacher_a.language = teacher_lang
    teacher_a.speaker = Speaker(teacher_lang)

    checkpoints_a = []
    for r in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        target = scene[target_idx]

        utterance = teacher_a.speaker.describe(
            target, scene,
            listener_vocab=student_a.language.vocabulary
        )
        chosen_idx = student_a.listener.interpret(utterance, scene)
        success = (chosen_idx == target_idx)

        student_a.language.total_games += 1
        if success:
            student_a.language.total_successes += 1
        student_a.language.record_usage(utterance, success)
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                student_a.language.record_collocation(
                    utterance[i], utterance[i + 1], success
                )
        if len(utterance) >= 2:
            student_a.language.record_ngram(utterance, success)

        if (r + 1) % 100 == 0:
            sr = student_a.language.total_successes / student_a.language.total_games
            checkpoints_a.append({
                'round': r + 1,
                'vocab_size': len(student_a.language.vocabulary),
                'success_rate': sr,
            })
            if verbose:
                print(f"    轮次 {r+1}: 词汇={len(student_a.language.vocabulary)}, "
                      f"成功率={sr:.3f}")

    # B: 无脚手架（不传 listener_vocab）
    if verbose:
        print(f"\n  B: 无脚手架（{num_rounds} 轮）...")
    student_b = CommunicationGame()
    teacher_b = CommunicationGame()
    teacher_b.language = teacher_lang
    teacher_b.speaker = Speaker(teacher_lang)

    checkpoints_b = []
    for r in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        target = scene[target_idx]

        utterance = teacher_b.speaker.describe(target, scene)
        chosen_idx = student_b.listener.interpret(utterance, scene)
        success = (chosen_idx == target_idx)

        student_b.language.total_games += 1
        if success:
            student_b.language.total_successes += 1
        student_b.language.record_usage(utterance, success)
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                student_b.language.record_collocation(
                    utterance[i], utterance[i + 1], success
                )
        if len(utterance) >= 2:
            student_b.language.record_ngram(utterance, success)

        if (r + 1) % 100 == 0:
            sr = student_b.language.total_successes / student_b.language.total_games
            checkpoints_b.append({
                'round': r + 1,
                'vocab_size': len(student_b.language.vocabulary),
                'success_rate': sr,
            })
            if verbose:
                print(f"    轮次 {r+1}: 词汇={len(student_b.language.vocabulary)}, "
                      f"成功率={sr:.3f}")

    return {
        'scaffolded': checkpoints_a,
        'unscaffolded': checkpoints_b,
    }


# ============================================================
# 实验 3：纠错效果
# ============================================================

def experiment_3_correction(num_rounds: int = 500,
                            verbose: bool = True) -> Dict:
    """
    有纠错 vs 无纠错

    纠错机制：当学生理解错误时，教师提供纠正信号
    - 纠正 = 用更简单的描述重新描述目标

    测量：学生错误率下降速度
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：纠错效果")
        print("=" * 60)

    bounds = (14.0, 14.0, 5.0)
    shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES
    num_objects = 12

    # 教师预训练
    teacher_game = CommunicationGame()
    for _ in range(200):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        teacher_game.play_round(scene, target_idx)
    teacher_lang = teacher_game.language

    # A: 有纠错
    if verbose:
        print(f"\n  A: 有纠错（{num_rounds} 轮）...")
    student_a = CommunicationGame()
    teacher_a = CommunicationGame()
    teacher_a.language = teacher_lang
    teacher_a.speaker = Speaker(teacher_lang)

    errors_a = []
    for r in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        target = scene[target_idx]

        # 教师描述
        utterance = teacher_a.speaker.describe(
            target, scene,
            listener_vocab=student_a.language.vocabulary
        )
        chosen_idx = student_a.listener.interpret(utterance, scene)
        success = (chosen_idx == target_idx)

        if not success:
            # 纠错：用更简单的描述（只用单符号）重试
            simple_utterance = teacher_a.speaker.describe(
                target, scene,
                listener_vocab=student_a.language.vocabulary
            )
            # 如果简单描述不同，让学生再试一次
            if simple_utterance != utterance:
                chosen_idx2 = student_a.listener.interpret(simple_utterance, scene)
                success2 = (chosen_idx2 == target_idx)
                # 用成功的描述更新学生
                if success2:
                    student_a.language.record_usage(simple_utterance, True)
                    success = True
                    utterance = simple_utterance

        student_a.language.total_games += 1
        if success:
            student_a.language.total_successes += 1
        else:
            errors_a.append(r)
        student_a.language.record_usage(utterance, success)
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                student_a.language.record_collocation(
                    utterance[i], utterance[i + 1], success
                )
        if len(utterance) >= 2:
            student_a.language.record_ngram(utterance, success)

    sr_a = student_a.language.total_successes / student_a.language.total_games
    if verbose:
        print(f"    成功率={sr_a:.3f}, 错误次数={len(errors_a)}")

    # B: 无纠错
    if verbose:
        print(f"\n  B: 无纠错（{num_rounds} 轮）...")
    student_b = CommunicationGame()
    teacher_b = CommunicationGame()
    teacher_b.language = teacher_lang
    teacher_b.speaker = Speaker(teacher_lang)

    errors_b = []
    for r in range(num_rounds):
        env = MultiAgent3DEnv(bounds=bounds)
        env.physics.add_random_objects(num_objects, shapes=shapes,
                                       materials=materials)
        scene = _get_scene_features(env)
        if len(scene) < 2:
            continue
        target_idx = random.randint(0, len(scene) - 1)
        target = scene[target_idx]

        utterance = teacher_b.speaker.describe(target, scene)
        chosen_idx = student_b.listener.interpret(utterance, scene)
        success = (chosen_idx == target_idx)

        student_b.language.total_games += 1
        if success:
            student_b.language.total_successes += 1
        else:
            errors_b.append(r)
        student_b.language.record_usage(utterance, success)
        if len(utterance) >= 2:
            for i in range(len(utterance) - 1):
                student_b.language.record_collocation(
                    utterance[i], utterance[i + 1], success
                )
        if len(utterance) >= 2:
            student_b.language.record_ngram(utterance, success)

    sr_b = student_b.language.total_successes / student_b.language.total_games
    if verbose:
        print(f"    成功率={sr_b:.3f}, 错误次数={len(errors_b)}")

    if verbose:
        print(f"\n对比:")
        print(f"  成功率: 纠错={sr_a:.3f}, 无纠错={sr_b:.3f}, "
              f"差距={sr_a - sr_b:+.3f}")
        print(f"  错误次数: 纠错={len(errors_a)}, 无纠错={len(errors_b)}")

    return {
        'with_correction': {
            'success_rate': sr_a,
            'error_count': len(errors_a),
        },
        'without_correction': {
            'success_rate': sr_b,
            'error_count': len(errors_b),
        },
        'success_rate_diff': sr_a - sr_b,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 44: 主动教学")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    results['exp1'] = experiment_1_teaching_vs_passive(
        num_rounds=500, verbose=True
    )

    results['exp2'] = experiment_2_scaffolding(
        num_rounds=500, verbose=True
    )

    results['exp3'] = experiment_3_correction(
        num_rounds=500, verbose=True
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

    with open('teaching_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 teaching_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 44 总结")
    print("=" * 60)

    exp1 = results['exp1']
    print(f"\n  教学 vs 被动:")
    print(f"    成功率差距: {exp1['success_rate_diff']:+.3f}")
    print(f"    词汇量差距: {exp1['vocab_diff']:+d}")

    exp3 = results['exp3']
    print(f"\n  纠错效果:")
    print(f"    成功率差距: {exp3['success_rate_diff']:+.3f}")

    return results


if __name__ == '__main__':
    main()
