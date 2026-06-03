"""
Phase 38 实验：迁移学习——在一个环境中学到的表示迁移到新环境

3 个实验：
1. 符号迁移：同词汇、不同物体，测试描述效率
2. 新概念适应：遇到未见过的 shape，测试学习速度
3. 跨域迁移：不同 bounds、不同物体数量，测试策略适应
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
    """从 3D 环境获取物体特征列表（只含字符串字段）"""
    features = []
    for obj in env.physics.objects:
        f = env.physics.get_object_features(obj.id)
        features.append(f)
    return features


def _train_language(num_rounds: int, bounds, shapes, materials,
                    num_objects: int = 4) -> EmergingLanguage:
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
                          num_objects: int = 4,
                          initial_language: EmergingLanguage = None,
                          frozen: bool = False,
                          window: int = 20) -> Dict:
    """
    测试学习曲线：每 window 轮记录一次成功率

    Args:
        initial_language: 初始语言（迁移测试），None 表示从零开始
        frozen: 如果为 True，不更新语言统计（纯迁移测试）
        window: 统计窗口大小
    """
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

        if frozen:
            success = game.play_round_frozen(scene, target_idx)
        else:
            success = game.play_round(scene, target_idx)

        window_successes += 1 if success else 0
        window_symbols += len(utterance)
        window_count += 1

        if (round_num + 1) % window == 0 and window_count > 0:
            sr = window_successes / window_count
            avg_sym = window_symbols / window_count
            results.append({
                'round': round_num + 1,
                'success_rate': sr,
                'avg_symbols': avg_sym,
            })
            window_successes = 0
            window_symbols = 0
            window_count = 0

    return {'curve': results, 'language': game.language}


# ============================================================
# 实验 1：符号迁移
# ============================================================

def experiment_1_symbol_transfer(num_train: int = 300,
                                  num_test: int = 200,
                                  verbose: bool = True) -> Dict:
    """
    符号迁移实验

    训练环境 A → 测试环境 B（同配置，不同随机物体）

    迁移 Agent：携带训练词汇，冻结测试
    基线 Agent：从零开始，学习测试

    测量：学习曲线（每 20 轮的成功率）
    预期：迁移 Agent 前期优势明显
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：符号迁移（同词汇、不同物体）")
        print("=" * 60)

    materials = MATERIAL_NAMES[:4]
    shapes = ['sphere', 'cube', 'cylinder']
    bounds = (10.0, 10.0, 5.0)

    # 训练
    if verbose:
        print(f"  训练：{num_train} 轮...")
    trained_lang = _train_language(num_train, bounds, shapes, materials)
    if verbose:
        sr = trained_lang.total_successes / trained_lang.total_games if trained_lang.total_games > 0 else 0
        print(f"  训练完成：词汇={len(trained_lang.vocabulary)}, 成功率={sr:.3f}")

    # 3 种测试条件
    # A: 迁移 + 冻结（纯迁移效果）
    # B: 迁移 + 继续学习（预训练 + 适应）
    # C: 从零学习（基线）

    # A: 迁移 + 冻结
    if verbose:
        print(f"  条件 A：迁移 + 冻结（{num_test} 轮）...")
    saved = trained_lang.save_state()
    frozen_lang = EmergingLanguage()
    frozen_lang.load_state(saved)
    frozen_result = _test_learning_curve(
        num_test, bounds, shapes, materials,
        initial_language=frozen_lang, frozen=True, window=20
    )

    # B: 迁移 + 继续学习
    if verbose:
        print(f"  条件 B：迁移 + 继续学习（{num_test} 轮）...")
    saved2 = trained_lang.save_state()
    transfer_lang = EmergingLanguage()
    transfer_lang.load_state(saved2)
    transfer_result = _test_learning_curve(
        num_test, bounds, shapes, materials,
        initial_language=transfer_lang, frozen=False, window=20
    )

    # C: 从零学习
    if verbose:
        print(f"  条件 C：从零学习（{num_test} 轮）...")
    baseline_result = _test_learning_curve(
        num_test, bounds, shapes, materials,
        initial_language=None, frozen=False, window=20
    )

    frozen_curve = frozen_result['curve']
    transfer_curve = transfer_result['curve']
    baseline_curve = baseline_result['curve']

    if verbose:
        print(f"\n学习曲线对比:")
        print(f"  {'轮次':>6}  {'冻结':>8}  {'迁移+学习':>10}  {'基线':>8}")
        for f, t, b in zip(frozen_curve, transfer_curve, baseline_curve):
            print(f"  {f['round']:6d}  {f['success_rate']:8.3f}  "
                  f"{t['success_rate']:10.3f}  {b['success_rate']:8.3f}")

    # 前期/后期对比
    early_f = np.mean([r['success_rate'] for r in frozen_curve[:3]])
    early_t = np.mean([r['success_rate'] for r in transfer_curve[:3]])
    early_b = np.mean([r['success_rate'] for r in baseline_curve[:3]])
    late_f = np.mean([r['success_rate'] for r in frozen_curve[-3:]])
    late_t = np.mean([r['success_rate'] for r in transfer_curve[-3:]])
    late_b = np.mean([r['success_rate'] for r in baseline_curve[-3:]])

    if verbose:
        print(f"\n  前期: 冻结={early_f:.3f}, 迁移+学习={early_t:.3f}, "
              f"基线={early_b:.3f}")
        print(f"  后期: 冻结={late_f:.3f}, 迁移+学习={late_t:.3f}, "
              f"基线={late_b:.3f}")
        print(f"  迁移+学习 vs 基线: 前期 {early_t - early_b:+.3f}, "
              f"后期 {late_t - late_b:+.3f}")

    return {
        'frozen_curve': frozen_curve,
        'transfer_curve': transfer_curve,
        'baseline_curve': baseline_curve,
        'early_advantage': early_t - early_b,
        'late_advantage': late_t - late_b,
        'frozen_early': early_f,
        'frozen_late': late_f,
        'vocab_size': len(trained_lang.vocabulary),
    }


# ============================================================
# 实验 2：新概念适应
# ============================================================

def experiment_2_new_concept(num_train: int = 300,
                              num_test: int = 200,
                              verbose: bool = True) -> Dict:
    """
    新概念适应实验

    训练：shapes=sphere,cube（只见过两种形状）
    测试：shapes=sphere,cube,cylinder（加入新形状）

    迁移 Agent：携带训练词汇，遇到 cylinder 时复用已有词汇
    基线 Agent：从零学习

    测量：学习曲线 + cylinder 相关词汇的出现时间
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：新概念适应（未见过的 shape）")
        print("=" * 60)

    train_shapes = ['sphere', 'cube']
    test_shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES[:4]
    bounds = (10.0, 10.0, 5.0)

    # 训练
    if verbose:
        print(f"  训练：{num_train} 轮（shapes={train_shapes}）...")
    trained_lang = _train_language(num_train, bounds, train_shapes, materials)
    has_cylinder = 'cylinder' in trained_lang.vocabulary
    if verbose:
        print(f"  训练完成：词汇={len(trained_lang.vocabulary)}, "
              f"有'cylinder': {has_cylinder}")

    # 迁移 + 继续学习
    if verbose:
        print(f"  迁移测试：{num_test} 轮（继续学习）...")
    saved = trained_lang.save_state()
    transfer_lang = EmergingLanguage()
    transfer_lang.load_state(saved)
    transfer_result = _test_learning_curve(
        num_test, bounds, test_shapes, materials,
        initial_language=transfer_lang, frozen=False, window=20
    )

    # 基线
    if verbose:
        print(f"  基线测试：{num_test} 轮...")
    baseline_result = _test_learning_curve(
        num_test, bounds, test_shapes, materials,
        initial_language=None, frozen=False, window=20
    )

    # 检查迁移后是否学到了 cylinder
    has_cylinder_after = 'cylinder' in transfer_result['language'].vocabulary

    transfer_curve = transfer_result['curve']
    baseline_curve = baseline_result['curve']

    if verbose:
        print(f"\n学习曲线对比:")
        print(f"  {'轮次':>6}  {'迁移':>8}  {'基线':>8}  {'差距':>8}")
        for t, b in zip(transfer_curve, baseline_curve):
            diff = t['success_rate'] - b['success_rate']
            print(f"  {t['round']:6d}  {t['success_rate']:8.3f}  "
                  f"{b['success_rate']:8.3f}  {diff:+8.3f}")

    early_t = np.mean([r['success_rate'] for r in transfer_curve[:3]])
    early_b = np.mean([r['success_rate'] for r in baseline_curve[:3]])
    late_t = np.mean([r['success_rate'] for r in transfer_curve[-3:]])
    late_b = np.mean([r['success_rate'] for r in baseline_curve[-3:]])

    if verbose:
        print(f"\n  前期: 迁移={early_t:.3f}, 基线={early_b:.3f}, "
              f"差距={early_t - early_b:+.3f}")
        print(f"  后期: 迁移={late_t:.3f}, 基线={late_b:.3f}, "
              f"差距={late_t - late_b:+.3f}")
        print(f"  测试后有'cylinder': {has_cylinder_after}")

    return {
        'transfer_curve': transfer_curve,
        'baseline_curve': baseline_curve,
        'early_advantage': early_t - early_b,
        'late_advantage': late_t - late_b,
        'had_cylinder_before': has_cylinder,
        'has_cylinder_after': has_cylinder_after,
    }


# ============================================================
# 实验 3：跨域迁移
# ============================================================

def experiment_3_cross_domain(num_train: int = 300,
                               num_test: int = 200,
                               verbose: bool = True) -> Dict:
    """
    跨域迁移实验

    训练：小空间 (8,8,5)，3 物体，只用 2 种 shape
    测试：大空间 (14,14,5)，8 物体，用全部 3 种 shape

    测试描述策略是否能从简单环境迁移到复杂环境
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：跨域迁移（简单→复杂）")
        print("=" * 60)

    train_bounds = (8.0, 8.0, 5.0)
    test_bounds = (14.0, 14.0, 5.0)
    train_shapes = ['sphere', 'cube']
    test_shapes = ['sphere', 'cube', 'cylinder']
    materials = MATERIAL_NAMES[:4]

    # 训练
    if verbose:
        print(f"  训练：{num_train} 轮（bounds={train_bounds}, "
              f"shapes={train_shapes}, 3 物体）...")
    trained_lang = _train_language(num_train, train_bounds, train_shapes,
                                    materials, num_objects=3)
    if verbose:
        sr = trained_lang.total_successes / trained_lang.total_games if trained_lang.total_games > 0 else 0
        print(f"  训练完成：词汇={len(trained_lang.vocabulary)}, 成功率={sr:.3f}")

    # 迁移 + 继续学习
    if verbose:
        print(f"  迁移测试：{num_test} 轮（继续学习）...")
    saved = trained_lang.save_state()
    transfer_lang = EmergingLanguage()
    transfer_lang.load_state(saved)
    transfer_result = _test_learning_curve(
        num_test, test_bounds, test_shapes, materials,
        initial_language=transfer_lang, frozen=False, window=20
    )

    # 基线
    if verbose:
        print(f"  基线测试：{num_test} 轮...")
    baseline_result = _test_learning_curve(
        num_test, test_bounds, test_shapes, materials,
        initial_language=None, window=20
    )

    transfer_curve = transfer_result['curve']
    baseline_curve = baseline_result['curve']

    if verbose:
        print(f"\n学习曲线对比:")
        print(f"  {'轮次':>6}  {'迁移':>8}  {'基线':>8}  {'差距':>8}")
        for t, b in zip(transfer_curve, baseline_curve):
            diff = t['success_rate'] - b['success_rate']
            print(f"  {t['round']:6d}  {t['success_rate']:8.3f}  "
                  f"{b['success_rate']:8.3f}  {diff:+8.3f}")

    early_t = np.mean([r['success_rate'] for r in transfer_curve[:3]])
    early_b = np.mean([r['success_rate'] for r in baseline_curve[:3]])
    late_t = np.mean([r['success_rate'] for r in transfer_curve[-3:]])
    late_b = np.mean([r['success_rate'] for r in baseline_curve[-3:]])

    if verbose:
        print(f"\n  前期: 迁移={early_t:.3f}, 基线={early_b:.3f}, "
              f"差距={early_t - early_b:+.3f}")
        print(f"  后期: 迁移={late_t:.3f}, 基线={late_b:.3f}, "
              f"差距={late_t - late_b:+.3f}")

    return {
        'transfer_curve': transfer_curve,
        'baseline_curve': baseline_curve,
        'early_advantage': early_t - early_b,
        'late_advantage': late_t - late_b,
        'vocab_size': len(trained_lang.vocabulary),
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 38: 迁移学习")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    results['exp1'] = experiment_1_symbol_transfer(
        num_train=300, num_test=200, verbose=True
    )

    results['exp2'] = experiment_2_new_concept(
        num_train=300, num_test=200, verbose=True
    )

    results['exp3'] = experiment_3_cross_domain(
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

    with open('transfer_learning_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 transfer_learning_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 38 总结")
    print("=" * 60)
    for name, label in [('exp1', '符号迁移'), ('exp2', '新概念适应'),
                         ('exp3', '跨域迁移')]:
        r = results[name]
        print(f"  {label}: 前期 {r['early_advantage']:+.3f}, "
              f"后期 {r['late_advantage']:+.3f}")

    return results


if __name__ == '__main__':
    main()
