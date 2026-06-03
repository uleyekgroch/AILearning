"""
Phase 34 实验：复杂感官环境

4 个实验：
1. 形状识别：不同形状物体，Agent 学习形状差异
2. 连续控制精度：离散动作 vs 连续动作的命中率
3. 材质属性学习：7 种材质，Agent 通过交互学习材质属性
4. 复杂场景理解：混合形状+材质的综合描述
"""

import sys
import random
import numpy as np
import json
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from environment_3d import PhysicsWorld3D, MATERIAL_NAMES, MATERIALS
from agent_3d import Agent3D


# ============================================================
# 实验 1：形状识别
# ============================================================

def experiment_1_shape_recognition(num_agents: int = 8,
                                    num_rounds: int = 2000,
                                    verbose: bool = True) -> Dict:
    """
    形状识别实验

    不同形状的物体（sphere/cube/cylinder），Agent 通过交互学习形状差异。
    测量：形状相关符号涌现率。

    预期："round"/"flat"/"tall" 等符号从交互中涌现。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：形状识别")
        print("=" * 60)

    # 创建 agent
    agents = []
    for i in range(num_agents):
        physics = PhysicsWorld3D(bounds=(10.0, 10.0, 5.0))
        # 添加不同形状的物体
        physics.add_random_objects(6, shapes=['sphere', 'cube', 'cylinder'],
                                   materials=['wood', 'plastic', 'metal'])
        agent = Agent3D(physics, use_language=True)
        agents.append(agent)

    shape_symbols = set()
    log = {'rounds': [], 'shape_symbols': [], 'vocab_size': [], 'success_rate': []}

    for r in range(num_rounds):
        i, j = random.sample(range(num_agents), 2)
        agent_a = agents[i]
        agent_b = agents[j]

        visible = agent_a.physics.get_visible_objects()
        if not visible:
            continue

        target = random.choice(visible)
        target_id = target['id']

        # Agent A 描述物体
        symbols = agent_a.describe_object(target_id)
        if not symbols:
            continue

        # 记录形状相关符号
        for sym in symbols:
            if sym in ('sphere', 'cube', 'cylinder'):
                shape_symbols.add(sym)

        # Agent B 尝试匹配
        visible_b = agent_b.physics.get_visible_objects()
        success = False

        if visible_b and symbols:
            best_score = -1
            for obj in visible_b:
                obj_features = agent_b.physics.get_object_features(obj['id'])
                score = sum(1 for s in symbols
                           for k, v in obj_features.items()
                           if k not in ('id', 'distance') and s == v)
                if score > best_score:
                    best_score = score
            success = best_score > 0

        if agent_a.language:
            agent_a.language.total_games += 1
            if success:
                agent_a.language.total_successes += 1
            agent_a.language.record_usage(symbols, success)

        if (r + 1) % 200 == 0:
            all_vocab = set()
            for agent in agents:
                if agent.language:
                    all_vocab.update(agent.language.vocabulary.keys())

            sr = agent_a.language.total_successes / agent_a.language.total_games \
                if agent_a.language.total_games > 0 else 0.0

            log['rounds'].append(r + 1)
            log['shape_symbols'].append(len(shape_symbols))
            log['vocab_size'].append(len(all_vocab))
            log['success_rate'].append(sr)

            if verbose:
                print(f"  Round {r+1:5d}: shape_syms={len(shape_symbols)}, "
                      f"vocab={len(all_vocab)}, success={sr:.3f}")

    if verbose:
        print(f"\n形状识别结果:")
        print(f"  形状符号: {shape_symbols}")
        print(f"  形状符号数: {len(shape_symbols)}")
        print(f"  最终成功率: {log['success_rate'][-1]:.3f}" if log['success_rate'] else "  无数据")

    return {
        'log': log,
        'shape_symbols': sorted(list(shape_symbols)),
        'num_shape_symbols': len(shape_symbols),
    }


# ============================================================
# 实验 2：连续控制精度
# ============================================================

def experiment_2_continuous_control(num_episodes: int = 80,
                                     verbose: bool = True) -> Dict:
    """
    连续控制精度实验

    目标：将物体扔到指定位置。
    对比：最大力度扔 vs 根据距离精确控制力度。

    测量：命中率。
    预期：精确力度控制 > 暴力最大力度。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：连续控制精度")
        print("=" * 60)

    coarse_hits = 0
    precise_hits = 0

    for ep in range(num_episodes):
        for mode in ['coarse', 'precise']:
            physics = PhysicsWorld3D(bounds=(12.0, 12.0, 5.0))

            # 目标位置（中等距离）
            target_pos = np.array([
                np.random.uniform(5.0, 8.0),
                np.random.uniform(5.0, 8.0),
                0.3,
            ])

            # 物体放在 agent 附近
            agent_start = np.array([3.0, 3.0, 0.5])
            physics.agent_pos = agent_start.copy()
            obj_id = physics.add_object(
                np.array([3.5, 3.0, 0.5]), mass=0.5, radius=0.2, material='rubber'
            )

            agent = Agent3D(physics, use_language=False)

            # 先面朝物体并抓取
            obj = physics._get_object(obj_id)
            physics.agent_facing = np.arctan2(
                obj.position[1] - physics.agent_pos[1],
                obj.position[0] - physics.agent_pos[0],
            )
            grab_action = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            agent.step(grab_action)

            # 转向目标
            physics.agent_facing = np.arctan2(
                target_pos[1] - physics.agent_pos[1],
                target_pos[0] - physics.agent_pos[0],
            )

            if mode == 'coarse':
                # 粗暴：最大力度扔（总是 1.0）
                throw_action = np.array([0.0, 0.0, 0.3, 0.0, 0.8, 1.0])
                agent.step(throw_action)
                # 等待物体飞行（50 步 = 1 秒）
                for _ in range(50):
                    agent.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            else:
                # 精确：根据距离调整力度
                dist = np.linalg.norm(target_pos - physics.agent_pos)
                throw_speed = np.clip(dist / 10.0, 0.3, 0.9)
                throw_action = np.array([0.0, 0.0, 0.3, 0.0, 0.8, throw_speed])
                agent.step(throw_action)
                for _ in range(50):
                    agent.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

            # 检查物体是否接近目标
            obj = physics._get_object(obj_id)
            if obj:
                dist_to_target = np.linalg.norm(obj.position[:2] - target_pos[:2])
                if dist_to_target < 2.5:
                    if mode == 'coarse':
                        coarse_hits += 1
                    else:
                        precise_hits += 1

        if verbose and (ep + 1) % 20 == 0:
            c_rate = coarse_hits / (ep + 1) if ep > 0 else 0
            p_rate = precise_hits / (ep + 1) if ep > 0 else 0
            print(f"  Episode {ep+1:3d}: coarse={c_rate:.3f}, precise={p_rate:.3f}")

    coarse_rate = coarse_hits / num_episodes if num_episodes > 0 else 0
    precise_rate = precise_hits / num_episodes if num_episodes > 0 else 0

    if verbose:
        print(f"\n连续控制结果:")
        print(f"  粗暴命中率: {coarse_rate:.3f}")
        print(f"  精确命中率: {precise_rate:.3f}")
        print(f"  提升: {(precise_rate - coarse_rate):.3f}")

    return {
        'coarse_rate': coarse_rate,
        'precise_rate': precise_rate,
        'improvement': precise_rate - coarse_rate,
    }


# ============================================================
# 实验 3：材质属性学习
# ============================================================

def experiment_3_material_learning(num_agents: int = 8,
                                    num_rounds: int = 2000,
                                    verbose: bool = True) -> Dict:
    """
    材质属性学习实验

    7 种材质的物体，Agent 通过推/扔/碰撞学习材质属性。
    测量：材质相关符号涌现。

    预期：material 符号从碰撞体验中涌现。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：材质属性学习")
        print("=" * 60)

    agents = []
    for i in range(num_agents):
        physics = PhysicsWorld3D(bounds=(10.0, 10.0, 5.0))
        # 使用全部 7 种材质
        physics.add_random_objects(7, shapes=['sphere'],
                                   materials=MATERIAL_NAMES)
        agent = Agent3D(physics, use_language=True)
        agents.append(agent)

    material_symbols = set()
    log = {'rounds': [], 'material_symbols': [], 'vocab_size': [], 'success_rate': []}

    for r in range(num_rounds):
        i, j = random.sample(range(num_agents), 2)
        agent_a = agents[i]
        agent_b = agents[j]

        # 让 agent 主动物体交互（推物体产生碰撞音频）
        if random.random() < 0.3:
            action = agent_a.choose_action(epsilon=0.2)
            agent_a.step(action)

        visible = agent_a.physics.get_visible_objects()
        if not visible:
            continue

        target = random.choice(visible)
        symbols = agent_a.describe_object(target['id'])
        if not symbols:
            continue

        # 记录材质相关符号
        for sym in symbols:
            if sym in MATERIAL_NAMES or sym in ('hard', 'soft', 'elastic'):
                material_symbols.add(sym)

        # Agent B 匹配
        visible_b = agent_b.physics.get_visible_objects()
        success = False

        if visible_b and symbols:
            best_score = -1
            for obj in visible_b:
                obj_features = agent_b.physics.get_object_features(obj['id'])
                score = sum(1 for s in symbols
                           for k, v in obj_features.items()
                           if k not in ('id', 'distance') and s == v)
                if score > best_score:
                    best_score = score
            success = best_score > 0

        if agent_a.language:
            agent_a.language.total_games += 1
            if success:
                agent_a.language.total_successes += 1
            agent_a.language.record_usage(symbols, success)

        if (r + 1) % 200 == 0:
            all_vocab = set()
            for agent in agents:
                if agent.language:
                    all_vocab.update(agent.language.vocabulary.keys())

            sr = agent_a.language.total_successes / agent_a.language.total_games \
                if agent_a.language.total_games > 0 else 0.0

            log['rounds'].append(r + 1)
            log['material_symbols'].append(len(material_symbols))
            log['vocab_size'].append(len(all_vocab))
            log['success_rate'].append(sr)

            if verbose:
                print(f"  Round {r+1:5d}: material_syms={len(material_symbols)}, "
                      f"vocab={len(all_vocab)}, success={sr:.3f}")

    if verbose:
        print(f"\n材质属性学习结果:")
        print(f"  材质符号: {material_symbols}")
        print(f"  材质符号数: {len(material_symbols)}")
        print(f"  最终成功率: {log['success_rate'][-1]:.3f}" if log['success_rate'] else "  无数据")

    return {
        'log': log,
        'material_symbols': sorted(list(material_symbols)),
        'num_material_symbols': len(material_symbols),
    }


# ============================================================
# 实验 4：复杂场景理解
# ============================================================

def experiment_4_complex_scene(num_agents: int = 8,
                                num_rounds: int = 2000,
                                verbose: bool = True) -> Dict:
    """
    复杂场景理解实验

    混合形状 + 混合材质的场景。
    Agent 需要综合形状、材质、物理规律来完成任务。

    测量：综合描述成功率 vs 单属性描述。
    预期：多属性描述成功率 > 单属性描述。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 4：复杂场景理解")
        print("=" * 60)

    agents = []
    for i in range(num_agents):
        physics = PhysicsWorld3D(bounds=(10.0, 10.0, 5.0))
        # 混合形状 + 混合材质
        physics.add_random_objects(8,
                                   shapes=['sphere', 'cube', 'cylinder'],
                                   materials=MATERIAL_NAMES)
        agent = Agent3D(physics, use_language=True)
        agents.append(agent)

    all_symbols = set()
    multi_attr_count = 0
    single_attr_count = 0
    multi_success = 0
    single_success = 0

    log = {'rounds': [], 'all_symbols': [], 'multi_rate': [], 'single_rate': []}

    for r in range(num_rounds):
        i, j = random.sample(range(num_agents), 2)
        agent_a = agents[i]
        agent_b = agents[j]

        # 交互
        if random.random() < 0.3:
            action = agent_a.choose_action(epsilon=0.2)
            agent_a.step(action)

        visible = agent_a.physics.get_visible_objects()
        if not visible:
            continue

        target = random.choice(visible)
        symbols = agent_a.describe_object(target['id'])
        if not symbols:
            continue

        for sym in symbols:
            all_symbols.add(sym)

        # 判断是多属性还是单属性描述
        is_multi = len(symbols) >= 3
        if is_multi:
            multi_attr_count += 1
        else:
            single_attr_count += 1

        # Agent B 匹配
        visible_b = agent_b.physics.get_visible_objects()
        success = False

        if visible_b and symbols:
            best_score = -1
            for obj in visible_b:
                obj_features = agent_b.physics.get_object_features(obj['id'])
                score = sum(1 for s in symbols
                           for k, v in obj_features.items()
                           if k not in ('id', 'distance') and s == v)
                if score > best_score:
                    best_score = score
            success = best_score > 0

        if success:
            if is_multi:
                multi_success += 1
            else:
                single_success += 1

        if agent_a.language:
            agent_a.language.total_games += 1
            if success:
                agent_a.language.total_successes += 1
            agent_a.language.record_usage(symbols, success)

        if (r + 1) % 200 == 0:
            multi_rate = multi_success / multi_attr_count if multi_attr_count > 0 else 0
            single_rate = single_success / single_attr_count if single_attr_count > 0 else 0

            log['rounds'].append(r + 1)
            log['all_symbols'].append(len(all_symbols))
            log['multi_rate'].append(multi_rate)
            log['single_rate'].append(single_rate)

            if verbose:
                print(f"  Round {r+1:5d}: syms={len(all_symbols)}, "
                      f"multi={multi_rate:.3f}, single={single_rate:.3f}")

    multi_rate = multi_success / multi_attr_count if multi_attr_count > 0 else 0
    single_rate = single_success / single_attr_count if single_attr_count > 0 else 0

    if verbose:
        print(f"\n复杂场景理解结果:")
        print(f"  总符号数: {len(all_symbols)}")
        print(f"  多属性成功率: {multi_rate:.3f}")
        print(f"  单属性成功率: {single_rate:.3f}")
        print(f"  差异: {(multi_rate - single_rate):.3f}")

    return {
        'log': log,
        'all_symbols': sorted(list(all_symbols)),
        'num_symbols': len(all_symbols),
        'multi_attr_rate': multi_rate,
        'single_attr_rate': single_rate,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 34: 复杂感官环境实验")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    # 实验 1：形状识别
    results['exp1'] = experiment_1_shape_recognition(
        num_agents=8, num_rounds=2000, verbose=True
    )

    # 实验 2：连续控制精度
    results['exp2'] = experiment_2_continuous_control(
        num_episodes=80, verbose=True
    )

    # 实验 3：材质属性学习
    results['exp3'] = experiment_3_material_learning(
        num_agents=8, num_rounds=2000, verbose=True
    )

    # 实验 4：复杂场景理解
    results['exp4'] = experiment_4_complex_scene(
        num_agents=8, num_rounds=2000, verbose=True
    )

    # 保存结果
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, set):
            return sorted(list(obj))
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    with open('rich_sensory_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 rich_sensory_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 34 总结")
    print("=" * 60)
    print(f"  形状符号数: {results['exp1']['num_shape_symbols']}")
    print(f"  连续控制提升: {results['exp2']['improvement']:.3f}")
    print(f"  材质符号数: {results['exp3']['num_material_symbols']}")
    print(f"  总符号数: {results['exp4']['num_symbols']}")

    return results


if __name__ == '__main__':
    main()
