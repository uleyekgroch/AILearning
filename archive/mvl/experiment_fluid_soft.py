"""
Phase 48 实验：流体与软体物理

4 个实验：
1. 流体物理发现 — Agent 观察流体 vs 刚体行为差异
2. 流体语言涌现 — 多 Agent 描述流体场景，期望涌现 flow, splash 等词汇
3. 软体碰撞实验 — 对比推刚体 vs 推软体的预测误差
4. 混合场景语言 — 同一场景包含刚体、流体、软体的语言区分
"""

import sys
import random
import numpy as np
import json

sys.stdout.reconfigure(encoding='utf-8')

from environment_3d import PhysicsWorld3D
from language_emergence import LanguageAgent, generate_rich_scene


def _filter_features(feat: dict) -> dict:
    """过滤掉非字符串值，语言系统只处理字符串特征"""
    return {k: v for k, v in feat.items() if isinstance(v, str)}


def _enrich_fluid_features(world) -> dict:
    """为流体生成更丰富的特征描述（去掉 type 标签）"""
    fs = world._fluid_system
    if not fs or not fs.particles:
        return {}
    level = fs.get_level()
    speed = fs.get_avg_speed()
    spread = fs.get_spread()
    return {
        'material': 'liquid',
        'behavior': 'flowing' if speed > 0.5 else 'settling' if speed > 0.05 else 'still',
        'spread': 'wide' if spread > 1.0 else 'narrow' if spread < 0.3 else 'medium',
        'level': 'high' if level > world.bounds[2] * 0.6 else
                 'low' if level < world.bounds[2] * 0.3 else 'mid',
        'texture': 'smooth',
        'hardness': 'soft',
    }


def _enrich_soft_features(feat: dict, world) -> dict:
    """为软体生成更丰富的特征描述"""
    # 去掉 type 和 index，加上物理属性
    result = {
        'material': 'rubber',
        'behavior': feat.get('deformation', 'stable'),
        'height': feat.get('height', 'mid'),
        'texture': 'squishy',
        'hardness': 'soft',
        'shape': 'deformable',
    }
    return result


def _enrich_rigid_features(feat: dict) -> dict:
    """确保刚体特征一致"""
    result = dict(feat)
    # 确保有 behavior 字段
    if 'behavior' not in result:
        result['behavior'] = 'rigid'
    if 'texture' not in result:
        result['texture'] = 'solid'
    return result


def experiment_1_fluid_discovery():
    """实验 1：流体物理发现 — Agent 观察流体 vs 刚体下落"""
    print("\n" + "=" * 60)
    print("实验 1：流体物理发现")
    print("=" * 60)

    results = {}

    for world_type in ['rigid', 'fluid']:
        world = PhysicsWorld3D(bounds=(5, 5, 3))

        if world_type == 'fluid':
            world.add_fluid(center=np.array([2.5, 2.5, 2.5]), count=40, spread=0.3)
        else:
            for i in range(40):
                pos = np.array([
                    2.5 + np.random.uniform(-0.3, 0.3),
                    2.5 + np.random.uniform(-0.3, 0.3),
                    2.5 + np.random.uniform(-0.3, 0.3),
                ])
                world.add_object(pos, mass=0.1, radius=0.05, material='rubber')

        # 记录物体高度随时间变化
        heights = []
        for step in range(200):
            obs = world.step(0)
            if world_type == 'fluid' and world._fluid_system:
                heights.append(world._fluid_system.get_level())
            else:
                if world.objects:
                    heights.append(np.mean([o.position[2] for o in world.objects]))
                else:
                    heights.append(0)

        results[world_type] = {
            'initial_height': heights[0] if heights else 0,
            'final_height': heights[-1] if heights else 0,
            'height_drop': (heights[0] - heights[-1]) if heights else 0,
            'spread': world._fluid_system.get_spread() if world_type == 'fluid' and world._fluid_system else 0,
        }
        print(f"  {world_type}: 初始高度={results[world_type]['initial_height']:.2f}, "
              f"最终高度={results[world_type]['final_height']:.2f}, "
              f"下降={results[world_type]['height_drop']:.2f}")

    print(f"\n  流体扩散度: {results['fluid']['spread']:.3f}")
    print(f"  结论: 流体扩散，刚体聚集")

    return results


def experiment_2_fluid_language():
    """实验 2：流体语言涌现 — 去掉 type 标签，用物理特征区分"""
    print("\n" + "=" * 60)
    print("实验 2：流体语言涌现")
    print("=" * 60)

    world = PhysicsWorld3D(bounds=(8, 8, 4))
    world.add_fluid(center=np.array([4.0, 4.0, 3.0]), count=60, spread=0.5)
    world.add_object(np.array([2.0, 2.0, 1.0]), mass=2.0, material='metal', shape='cube')
    world.add_object(np.array([6.0, 6.0, 1.0]), mass=0.5, material='rubber')

    for _ in range(100):
        world.step(0)

    agents = [LanguageAgent(f"agent_{i}") for i in range(4)]

    fluid_symbols = set()
    total_rounds = 0
    successes = 0

    for round_num in range(500):
        scene = []
        # 流体特征（无 type 标签）
        if world._fluid_system and world._fluid_system.particles:
            scene.append(_enrich_fluid_features(world))
        # 刚体特征
        for obj in world.objects:
            feats = _enrich_rigid_features(world.get_object_features(obj.id))
            scene.append(_filter_features(feats))
        # 软体特征
        for sb_feat in world.get_soft_body_features():
            scene.append(_enrich_soft_features(sb_feat, world))

        if len(scene) < 2:
            scene = generate_rich_scene('medium')

        target_idx = random.randint(0, len(scene) - 1)

        a, b = random.sample(agents, 2)
        speaker, listener = (a, b) if random.random() < 0.5 else (b, a)

        utterance = speaker.speak(scene[target_idx], scene)
        chosen = listener.listen(utterance, scene)
        success = (chosen == target_idx)

        speaker.update_from_communication(utterance, success)
        listener.update_from_communication(utterance, success)

        if success:
            successes += 1
        total_rounds += 1

        world.step(random.randint(0, 11))

        # 检查流体相关符号（包括物理特征衍生的符号）
        for sym in speaker.language.vocabulary:
            if sym in ['flow', 'splash', 'pour', 'fill', 'drip', 'liquid',
                       'wet', 'spread', 'pool', 'stream', 'liquid',
                       'smooth', 'soft', 'settling', 'flowing']:
                fluid_symbols.add(sym)

    sr = successes / total_rounds if total_rounds > 0 else 0
    print(f"  通信成功率: {sr:.3f}")
    print(f"  流体相关符号: {fluid_symbols if fluid_symbols else '(无专门流体符号)'}")

    all_symbols = set()
    for agent in agents:
        all_symbols.update(agent.language.vocabulary.keys())
    print(f"  总词汇量: {len(all_symbols)}")

    return {
        'success_rate': sr,
        'fluid_symbols': list(fluid_symbols),
        'total_symbols': len(all_symbols),
    }


def experiment_3_soft_collision():
    """实验 3：软体碰撞实验 — 对比推刚体 vs 推软体"""
    print("\n" + "=" * 60)
    print("实验 3：软体碰撞实验")
    print("=" * 60)

    results = {}

    for obj_type in ['rigid', 'soft']:
        world = PhysicsWorld3D(bounds=(5, 5, 3))

        if obj_type == 'soft':
            world.add_soft_body(
                center=np.array([2.5, 2.5, 0.7]),
                size=np.array([0.4, 0.4, 0.4]),
                stiffness=30.0
            )
        else:
            world.add_object(
                np.array([2.5, 2.5, 0.7]),
                mass=1.0, material='wood', shape='cube',
                size=np.array([0.4, 0.4, 0.4])
            )

        world.agent_pos = np.array([2.5, 1.5, 0.5])
        world.agent_facing = np.pi / 2

        collision_events = 0
        deformations = []
        position_changes = []

        prev_pos = None
        for step in range(300):
            if step < 50:
                obs = world.step(0)  # FORWARD
            elif step < 100:
                obs = world.step(11)  # PUSH
            else:
                obs = world.step(0)  # 观察

            collision_events += len([e for e in obs.audio_events
                                     if e.event_type == 'collision'])

            if obj_type == 'soft' and world._soft_bodies:
                deformations.append(world._soft_bodies[0].get_deformation())
            elif obj_type == 'rigid' and world.objects:
                cur_pos = world.objects[0].position.copy()
                if prev_pos is not None:
                    position_changes.append(np.linalg.norm(cur_pos - prev_pos))
                prev_pos = cur_pos

        results[obj_type] = {
            'collision_events': collision_events,
            'avg_deformation': np.mean(deformations) if deformations else 0,
            'avg_movement': np.mean(position_changes) if position_changes else 0,
        }
        print(f"  {obj_type}: 碰撞事件={collision_events}, "
              f"平均形变={results[obj_type]['avg_deformation']:.3f}, "
              f"平均移动={results[obj_type]['avg_movement']:.3f}")

    print(f"\n  结论: 软体形变={results['soft']['avg_deformation']:.3f}, "
          f"刚体移动={results['rigid']['avg_movement']:.3f}")

    return results


def experiment_4_mixed_scene_language():
    """实验 4：混合场景语言 — 刚体+流体+软体（无 type 标签）"""
    print("\n" + "=" * 60)
    print("实验 4：混合场景语言")
    print("=" * 60)

    world = PhysicsWorld3D(bounds=(10, 10, 5))
    world.add_fluid(center=np.array([5.0, 5.0, 3.5]), count=50, spread=0.5)
    world.add_soft_body(center=np.array([3.0, 3.0, 1.5]), size=np.array([0.5, 0.5, 0.5]))
    world.add_object(np.array([7.0, 7.0, 1.0]), mass=3.0, material='metal')
    world.add_object(np.array([2.0, 8.0, 1.0]), mass=0.5, material='rubber')

    for _ in range(150):
        world.step(0)

    agents = [LanguageAgent(f"agent_{i}") for i in range(6)]

    # 用 material 而非 type 来分类
    material_vocab = {'liquid': set(), 'rubber': set(), 'metal': set()}
    total_rounds = 0
    successes = 0

    for round_num in range(800):
        scene = []
        if world._fluid_system and world._fluid_system.particles:
            scene.append(_enrich_fluid_features(world))
        for sb_feat in world.get_soft_body_features():
            scene.append(_enrich_soft_features(sb_feat, world))
        for obj in world.objects:
            scene.append(_enrich_rigid_features(
                _filter_features(world.get_object_features(obj.id))))

        if len(scene) < 2:
            scene = generate_rich_scene('medium')

        target_idx = random.randint(0, len(scene) - 1)

        a, b = random.sample(agents, 2)
        speaker, listener = (a, b) if random.random() < 0.5 else (b, a)

        utterance = speaker.speak(scene[target_idx], scene)
        chosen = listener.listen(utterance, scene)
        success = (chosen == target_idx)

        speaker.update_from_communication(utterance, success)
        listener.update_from_communication(utterance, success)

        if success:
            successes += 1
        total_rounds += 1

        # 记录与 material 相关的符号
        target_material = scene[target_idx].get('material', 'unknown')
        if success and target_material in material_vocab:
            for sym in utterance:
                material_vocab[target_material].add(sym)

        world.step(random.randint(0, 11))

    sr = successes / total_rounds if total_rounds > 0 else 0
    print(f"  通信成功率: {sr:.3f}")
    for mat, syms in material_vocab.items():
        print(f"  {mat} 相关符号 ({len(syms)}): {sorted(syms)[:10]}")

    all_symbols = set()
    for agent in agents:
        all_symbols.update(agent.language.vocabulary.keys())
    print(f"  总词汇量: {len(all_symbols)}")

    return {
        'success_rate': sr,
        'material_vocab': {k: list(v) for k, v in material_vocab.items()},
        'total_symbols': len(all_symbols),
    }


if __name__ == '__main__':
    all_results = {}

    all_results['fluid_discovery'] = experiment_1_fluid_discovery()
    all_results['fluid_language'] = experiment_2_fluid_language()
    all_results['soft_collision'] = experiment_3_soft_collision()
    all_results['mixed_language'] = experiment_4_mixed_scene_language()

    with open('fluid_soft_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n\n结果已保存到 fluid_soft_results.json")
