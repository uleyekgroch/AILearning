"""
Phase 26 实验：语言与环境探索整合

4 个实验验证语言是否从真实环境探索中涌现。
"""

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import Object, SimpleGridWorld, create_simple_world
from environment_language_bridge import (
    EnvironmentLanguageBridge, ExperienceDrivenScenarioGenerator,
)
from grounding_unified_language import UnifiedObject, UnifiedScene, generate_unified_scenario
from adaptive_strategy import AdaptiveCommunicationGame


def experiment_1_feature_derivation():
    """实验 1：特征推导验证"""
    print("=" * 60)
    print("实验 1: 特征推导验证")
    print("=" * 60)

    bridge = EnvironmentLanguageBridge()

    # 测试不同材质的物体
    test_objects = [
        Object(0, 0, 0, 'red', 'circle', 1.5, material='metal'),
        Object(1, 0, 0, 'blue', 'square', 0.5, material='fabric'),
        Object(2, 0, 0, 'green', 'triangle', 1.0, material='wood'),
        Object(3, 0, 0, 'yellow', 'circle', 1.8, material='stone'),
        Object(4, 0, 0, 'red', 'square', 0.3, material='glass'),
    ]

    for obj in test_objects:
        obj.enrich_features()
        unified = bridge._object_to_unified(obj)
        print(f"\n  Object(id={obj.id}, color={obj.color}, shape={obj.shape}, "
              f"weight={obj.weight}, material={obj.material})")
        print(f"    -> sound={unified.auditory['sound']}, "
              f"texture={unified.tactile['texture']}, "
              f"affordances={unified.affordances}")
        print(f"    -> visual={unified.visual}")

    # 验证推导逻辑
    print("\n  推导逻辑验证:")
    metal_obj = test_objects[0]
    assert metal_obj.sound == 'loud', f"重物应为 loud, 实际 {metal_obj.sound}"
    assert metal_obj.texture == 'hard', f"金属应为 hard, 实际 {metal_obj.texture}"
    print("    [OK] 重物->loud, 金属->hard")

    fabric_obj = test_objects[1]
    assert fabric_obj.sound == 'quiet', f"轻物应为 quiet, 实际 {fabric_obj.sound}"
    assert fabric_obj.texture == 'soft_tactile', f"织物应为 soft_tactile, 实际 {fabric_obj.texture}"
    print("    [OK] 轻物->quiet, 织物->soft_tactile")

    glass_obj = test_objects[4]
    assert 'contain' in glass_obj.affordances, f"玻璃应有 contain, 实际 {glass_obj.affordances}"
    print("    [OK] 玻璃->contain")

    print("\n  判定: 所有推导正确")
    return test_objects


def experiment_2_bridge_conversion():
    """实验 2：桥梁转换验证"""
    print("\n" + "=" * 60)
    print("实验 2: 桥梁转换验证")
    print("=" * 60)

    bridge = EnvironmentLanguageBridge()

    # 创建环境，物体放在 agent 附近
    env = SimpleGridWorld(10, 10)
    env.add_object(Object(0, 4, 4, 'red', 'circle', 1.0, material='metal'))
    env.add_object(Object(1, 5, 5, 'blue', 'square', 1.5, material='stone'))
    env.add_object(Object(2, 6, 5, 'green', 'triangle', 0.8, material='fabric'))
    for obj in env.objects:
        obj.enrich_features()
    obs = env.get_observation()

    print(f"\n  环境观测: {len(obs['visible_objects'])} 个可见物体")

    # 转换为统一场景
    scene = bridge.observation_to_scene(obs, target_idx=0)

    if scene is None:
        print("  错误: 转换返回 None（可见物体不足 2 个）")
        return None

    print(f"  统一场景: {len(scene.objects)} 个物体, "
          f"目标={scene.target_idx}, 歧义={scene.ambiguity_types}")

    for i, obj in enumerate(scene.objects):
        marker = " <- target" if i == scene.target_idx else ""
        print(f"    [{i}] visual={dict(obj.visual)}, "
              f"sound={obj.auditory['sound']}, "
              f"texture={obj.tactile['texture']}{marker}")

    # 测试歧义检测
    print(f"\n  歧义类型检测: {scene.ambiguity_types}")

    # 创建有明确歧义的场景
    # 两个视觉相同但声音不同的物体
    obj1 = Object(0, 0, 0, 'red', 'circle', 1.5, material='metal')
    obj2 = Object(1, 1, 0, 'red', 'circle', 0.5, material='fabric')
    obj1.enrich_features()
    obj2.enrich_features()

    obs_ambiguous = {
        'visible_objects': [
            {'object': obj1, 'relative_x': 0, 'relative_y': 0, 'distance': 0},
            {'object': obj2, 'relative_x': 1, 'relative_y': 0, 'distance': 1},
        ]
    }
    scene_amb = bridge.observation_to_scene(obs_ambiguous, target_idx=0)
    if scene_amb:
        print(f"  歧义场景: visual_ambiguous={('visual_ambiguous' in scene_amb.ambiguity_types)}, "
              f"crossmodal={('crossmodal' in scene_amb.ambiguity_types)}")

    return scene


def experiment_3_exploration_communication():
    """实验 3：探索-通信整合"""
    print("\n" + "=" * 60)
    print("实验 3: 探索-通信整合（500 步探索）")
    print("=" * 60)

    bridge = EnvironmentLanguageBridge()
    scenario_gen = ExperienceDrivenScenarioGenerator(bridge)
    game = AdaptiveCommunicationGame()

    # 创建丰富环境（多材质物体）
    env = SimpleGridWorld(10, 10)
    objects_data = [
        (0, 2, 2, 'red', 'circle', 1.5, 'metal'),
        (1, 5, 5, 'blue', 'square', 0.5, 'fabric'),
        (2, 7, 3, 'green', 'triangle', 1.0, 'wood'),
        (3, 3, 7, 'yellow', 'circle', 1.8, 'stone'),
        (4, 6, 1, 'red', 'square', 0.8, 'plastic'),
        (5, 8, 6, 'blue', 'circle', 1.2, 'glass'),
        (6, 1, 4, 'green', 'square', 0.6, 'fabric'),
        (7, 4, 8, 'yellow', 'triangle', 1.4, 'metal'),
    ]
    for oid, x, y, color, shape, weight, material in objects_data:
        obj = Object(oid, x, y, color, shape, weight, material=material)
        obj.enrich_features()
        env.add_object(obj)

    success_history = []

    for step in range(500):
        # 探索
        action = random.randint(0, 4)
        obs = env.step(action)[0]

        # 记录经验
        scenario_gen.record_observation(obs)

        # 视野中有 2+ 物体时通信
        visible = obs.get('visible_objects', [])
        if len(visible) >= 2:
            scene = bridge.observation_to_scene(obs)
            if scene:
                success = game.play_round(scene)
                if (step + 1) % 100 == 0:
                    stats = game.get_stats()
                    success_history.append((step + 1, stats['success_rate']))

    print(f"\n  经验缓冲: {scenario_gen.experience_count} 条观测")
    stats = game.get_stats()
    print(f"  通信统计: {stats['total_games']} 轮, "
          f"成功率 {stats['success_rate']:.1%}")

    if success_history:
        print("\n  成功率演化:")
        for step, rate in success_history:
            print(f"    步骤 {step}: {rate:.1%}")

    return game, scenario_gen


def experiment_4_comparison():
    """实验 4：对比实验（探索驱动 vs 随机场景 vs 固定策略）"""
    print("\n" + "=" * 60)
    print("实验 4: 对比实验")
    print("=" * 60)

    bridge = EnvironmentLanguageBridge()
    results = {'exploration': [], 'random': [], 'fixed': []}

    for run in range(5):
        seed = 42 + run

        # --- 探索驱动 ---
        random.seed(seed)
        env = SimpleGridWorld(10, 10)
        for oid, x, y, color, shape, weight, material in [
            (0, 2, 2, 'red', 'circle', 1.5, 'metal'),
            (1, 5, 5, 'blue', 'square', 0.5, 'fabric'),
            (2, 7, 3, 'green', 'triangle', 1.0, 'wood'),
            (3, 3, 7, 'yellow', 'circle', 1.8, 'stone'),
            (4, 6, 1, 'red', 'square', 0.8, 'plastic'),
            (5, 8, 6, 'blue', 'circle', 1.2, 'glass'),
        ]:
            obj = Object(oid, x, y, color, shape, weight, material=material)
            obj.enrich_features()
            env.add_object(obj)

        game_exp = AdaptiveCommunicationGame()
        for step in range(500):
            action = random.randint(0, 4)
            obs = env.step(action)[0]
            visible = obs.get('visible_objects', [])
            if len(visible) >= 2:
                scene = bridge.observation_to_scene(obs)
                if scene:
                    game_exp.play_round(scene)
        results['exploration'].append(game_exp.get_stats()['success_rate'])

        # --- 随机场景 ---
        random.seed(seed)
        game_rand = AdaptiveCommunicationGame()
        for _ in range(500):
            scene = generate_unified_scenario(
                ambiguity_types=random.choice([
                    {'visual_ambiguous', 'crossmodal'},
                    {'subset'},
                    {'tool'},
                    {'causal'},
                    {'confidence'},
                ]),
                num_objects=3,
            )
            game_rand.play_round(scene)
        results['random'].append(game_rand.get_stats()['success_rate'])

        # --- 固定策略（无自适应） ---
        random.seed(seed)
        from grounding_unified_language import UnifiedCommunicationGame
        game_fixed = UnifiedCommunicationGame()
        for _ in range(500):
            scene = generate_unified_scenario(
                ambiguity_types=random.choice([
                    {'visual_ambiguous', 'crossmodal'},
                    {'subset'},
                    {'tool'},
                    {'causal'},
                    {'confidence'},
                ]),
                num_objects=3,
            )
            game_fixed.play_round(scene)
        results['fixed'].append(game_fixed.get_stats()['success_rate'])

        print(f"  运行 {run+1}: 探索={results['exploration'][-1]:.1%}, "
              f"随机={results['random'][-1]:.1%}, "
              f"固定={results['fixed'][-1]:.1%}")

    # 汇总
    for key in results:
        avg = sum(results[key]) / len(results[key])
        std = (sum((r - avg) ** 2 for r in results[key]) / len(results[key])) ** 0.5
        print(f"\n  {key}: {avg:.1%} ± {std:.1%}")

    return results


if __name__ == '__main__':
    print("Phase 26: 语言与环境探索整合实验")
    print("=" * 60)

    experiment_1_feature_derivation()
    experiment_2_bridge_conversion()
    experiment_3_exploration_communication()
    experiment_4_comparison()

    print("\n" + "=" * 60)
    print("所有实验完成")
    print("=" * 60)
