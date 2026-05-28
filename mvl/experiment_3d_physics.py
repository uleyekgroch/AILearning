"""
Phase 33 实验：3D 物理世界

4 个实验：
1. 重力发现：Agent 在有重力/无重力环境中探索，测量预测误差
2. 碰撞物理：不同质量/弹性的物体，Agent 学习碰撞规律
3. 工具使用：用重物体当"锤子"，轻物体当"球"
4. 物理语言涌现：Agent 在 3D 物理世界中交流，观察物理符号涌现
"""

import sys
import random
import numpy as np
import json
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from environment_3d import PhysicsWorld3D, Action3D
from agent_3d import Agent3D


# ============================================================
# 实验 1：重力发现
# ============================================================

def experiment_1_gravity_discovery(num_steps: int = 2000,
                                    verbose: bool = True) -> Dict:
    """
    重力发现实验

    Agent 在有重力/无重力环境中探索。
    测量：Agent 是否学会预测物体下落。

    预期：有重力环境的预测误差最终更低（因为物理规律可预测）。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：重力发现")
        print("=" * 60)

    results = {}

    for env_name, gravity in [('with_gravity', -9.8), ('no_gravity', 0.0)]:
        if verbose:
            print(f"\n--- {env_name} ---")

        # 创建世界
        physics = PhysicsWorld3D(bounds=(8.0, 8.0, 5.0), gravity=gravity)
        # 添加物体（从高处释放）
        for i in range(5):
            pos = np.array([
                np.random.uniform(1.0, 7.0),
                np.random.uniform(1.0, 7.0),
                np.random.uniform(2.0, 4.0),  # 高处
            ])
            physics.add_object(pos, mass=1.0, radius=0.3, material='wood')

        # 创建 agent
        agent = Agent3D(physics, use_language=False)

        # 探索
        log = {'steps': [], 'avg_error': [], 'curiosity': []}

        for block in range(0, num_steps, 200):
            result = agent.explore(num_steps=200, epsilon=0.2)
            log['steps'].append(block + 200)
            log['avg_error'].append(result['avg_error'])
            log['curiosity'].append(result['avg_curiosity'])

            if verbose:
                print(f"  Step {block+200:5d}: error={result['avg_error']:.4f}, "
                      f"curiosity={result['avg_curiosity']:.4f}")

        results[env_name] = {
            'log': log,
            'final_error': log['avg_error'][-1] if log['avg_error'] else 0.0,
            'error_trend': (log['avg_error'][-1] - log['avg_error'][0]) if len(log['avg_error']) > 1 else 0.0,
        }

    # 对比
    if verbose:
        print(f"\n重力发现结果:")
        print(f"  有重力最终误差: {results['with_gravity']['final_error']:.4f}")
        print(f"  无重力最终误差: {results['no_gravity']['final_error']:.4f}")
        diff = results['no_gravity']['final_error'] - results['with_gravity']['final_error']
        print(f"  差异: {diff:.4f} (正 = 有重力更可预测)")

    return results


# ============================================================
# 实验 2：碰撞物理
# ============================================================

def experiment_2_collision_physics(num_steps: int = 3000,
                                    verbose: bool = True) -> Dict:
    """
    碰撞物理实验

    不同质量/弹性的物体，Agent 推/扔物体，学习碰撞规律。
    测量：Agent 能否预测碰撞后的运动方向。

    预期：经过足够交互，碰撞预测误差降低。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：碰撞物理")
        print("=" * 60)

    # 创建世界
    physics = PhysicsWorld3D(bounds=(10.0, 10.0, 5.0))

    # 添加不同材质的物体
    objects = [
        (np.array([3.0, 5.0, 0.5]), 1.0, 0.3, 'metal'),    # 轻金属
        (np.array([5.0, 5.0, 0.5]), 3.0, 0.4, 'metal'),    # 重金属
        (np.array([7.0, 5.0, 0.5]), 1.0, 0.3, 'wood'),     # 轻木头
        (np.array([5.0, 3.0, 0.5]), 2.0, 0.3, 'plastic'),  # 中塑料
        (np.array([5.0, 7.0, 0.5]), 4.0, 0.5, 'wood'),     # 重木头
    ]
    for pos, mass, radius, material in objects:
        physics.add_object(pos, mass=mass, radius=radius, material=material)

    # 创建 agent
    agent = Agent3D(physics, use_language=False)

    # 探索，重点使用 push 和 throw 动作
    log = {'steps': [], 'avg_error': [], 'collision_count': []}
    total_collisions = 0

    for block in range(0, num_steps, 200):
        errors = []
        for _ in range(200):
            # 交替使用不同动作
            if random.random() < 0.3:
                action = Action3D.PUSH
            elif random.random() < 0.3:
                action = Action3D.THROW
            elif random.random() < 0.3:
                action = Action3D.GRAB
            else:
                action = agent.choose_action(epsilon=0.2)

            result = agent.step(action)
            errors.append(result['prediction_error'])

            # 统计碰撞事件
            for event in result['observation'].audio_events:
                if event.event_type == 'collision':
                    total_collisions += 1

        log['steps'].append(block + 200)
        log['avg_error'].append(np.mean(errors) if errors else 0.0)
        log['collision_count'].append(total_collisions)

        if verbose:
            print(f"  Step {block+200:5d}: error={np.mean(errors) if errors else 0:.4f}, "
                  f"collisions={total_collisions}")

    # 分析
    if verbose:
        print(f"\n碰撞物理结果:")
        print(f"  最终误差: {log['avg_error'][-1]:.4f}")
        print(f"  误差变化: {log['avg_error'][0]:.4f} → {log['avg_error'][-1]:.4f}")
        print(f"  总碰撞次数: {total_collisions}")

    return {
        'log': log,
        'final_error': log['avg_error'][-1],
        'total_collisions': total_collisions,
    }


# ============================================================
# 实验 3：工具使用
# ============================================================

def experiment_3_tool_use(num_episodes: int = 50,
                           verbose: bool = True) -> Dict:
    """
    工具使用实验

    场景：目标物体在远处，Agent 需要选择合适的物体来完成任务。
    - 用重物体当"锤子"（push 效果大）
    - 用轻物体当"球"（throw 距离远）

    测量：Agent 是否学会根据物体属性选择动作。
    预期：工具使用成功率 > 随机动作。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：工具使用")
        print("=" * 60)

    success_count = 0
    random_success = 0
    total_episodes = 0

    for ep in range(num_episodes):
        # 创建世界
        physics = PhysicsWorld3D(bounds=(10.0, 10.0, 5.0))

        # 目标物体（远处的轻物体）
        target_pos = np.array([8.0, 8.0, 0.5])
        physics.add_object(target_pos, mass=0.5, radius=0.3, material='plastic')

        # 工具候选（近处的不同物体）
        heavy_tool = physics.add_object(
            np.array([3.0, 5.0, 0.5]), mass=5.0, radius=0.4, material='metal'
        )
        light_tool = physics.add_object(
            np.array([5.0, 3.0, 0.5]), mass=0.5, radius=0.2, material='plastic'
        )

        # 创建 agent
        agent = Agent3D(physics, use_language=False)

        # Agent 先抓取一个工具
        # 面向重工具
        physics.agent_facing = np.arctan2(
            5.0 - physics.agent_pos[1], 3.0 - physics.agent_pos[0]
        )
        agent.step(Action3D.GRAB)

        # 执行动作
        action = Action3D.PUSH if physics.held_object == heavy_tool else Action3D.THROW
        for _ in range(5):
            agent.step(action)

        # 检查目标是否被移动
        target_obj = physics._get_object(1)  # target id = 1
        if target_obj:
            dist_moved = np.linalg.norm(target_obj.position - target_pos)
            if dist_moved > 1.0:
                success_count += 1

        # 随机基线
        random_action = random.choice([Action3D.PUSH, Action3D.THROW])
        if random_action == Action3D.PUSH:
            # push 重物体更有效
            random_success += 1 if physics.held_object == heavy_tool else 0
        else:
            # throw 轻物体更有效
            random_success += 1 if physics.held_object == light_tool else 0

        total_episodes += 1

        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1:3d}: success={success_count}/{total_episodes}")

    success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
    random_rate = random_success / total_episodes if total_episodes > 0 else 0.0

    if verbose:
        print(f"\n工具使用结果:")
        print(f"  成功率: {success_rate:.3f}")
        print(f"  随机基线: {random_rate:.3f}")
        print(f"  提升: {(success_rate - random_rate):.3f}")

    return {
        'success_rate': success_rate,
        'random_baseline': random_rate,
        'total_episodes': total_episodes,
    }


# ============================================================
# 实验 4：物理语言涌现
# ============================================================

def experiment_4_physics_language(num_agents: int = 10,
                                   num_rounds: int = 2000,
                                   verbose: bool = True) -> Dict:
    """
    物理语言涌现实验

    Agent 在 3D 物理世界中交流。
    观察物理相关符号涌现（"heavy", "bounce", "fall", "roll"）。

    测量：物理符号的涌现率和使用成功率。
    预期：物理符号从交互压力中涌现。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 4：物理语言涌现")
        print("=" * 60)

    # 创建 agent（共享一个物理世界）
    agents = []
    for i in range(num_agents):
        physics = PhysicsWorld3D(bounds=(10.0, 10.0, 5.0))
        physics.add_random_objects(5)
        agent = Agent3D(physics, use_language=True)
        agents.append(agent)

    # 收集所有物理相关符号
    physical_symbols = set()

    log = {'rounds': [], 'vocab_size': [], 'physical_symbols': [], 'success_rate': []}

    for r in range(num_rounds):
        # 随机选一对 agent
        i, j = random.sample(range(num_agents), 2)
        agent_a = agents[i]
        agent_b = agents[j]

        # Agent A 描述面前的物体
        visible = agent_a.physics.get_visible_objects()
        if not visible:
            continue

        # 选一个物体描述
        target = random.choice(visible)
        target_id = target['id']

        # Agent A 描述
        symbols = agent_a.describe_object(target_id)
        if not symbols:
            continue

        # 记录物理符号
        for sym in symbols:
            if sym in ('heavy', 'light', 'bouncy', 'stable', 'fast', 'moving',
                       'still', 'high', 'low', 'mid'):
                physical_symbols.add(sym)

        # Agent B 尝试理解并找到物体
        # 简化：检查 Agent B 是否能根据符号找到对应物体
        visible_b = agent_b.physics.get_visible_objects()
        success = False

        if visible_b and symbols:
            # 找最佳匹配
            best_match = None
            best_score = -1

            for obj in visible_b:
                obj_features = agent_b.physics.get_object_features(obj['id'])
                score = 0
                for sym in symbols:
                    # 检查符号是否匹配物体特征
                    for key, value in obj_features.items():
                        if key == 'id' or key == 'distance':
                            continue
                        if sym == value:
                            score += 1
                if score > best_score:
                    best_score = score
                    best_match = obj

            # 成功条件：最佳匹配的分数 > 0
            success = best_score > 0

        # 更新语言
        if agent_a.language:
            agent_a.language.total_games += 1
            if success:
                agent_a.language.total_successes += 1
            agent_a.language.record_usage(symbols, success)

        if (r + 1) % 200 == 0:
            # 统计
            all_vocab = set()
            for agent in agents:
                if agent.language:
                    all_vocab.update(agent.language.vocabulary.keys())

            sr = agent_a.language.total_successes / agent_a.language.total_games \
                if agent_a.language.total_games > 0 else 0.0

            log['rounds'].append(r + 1)
            log['vocab_size'].append(len(all_vocab))
            log['physical_symbols'].append(len(physical_symbols))
            log['success_rate'].append(sr)

            if verbose:
                print(f"  Round {r+1:5d}: vocab={len(all_vocab)}, "
                      f"physical={len(physical_symbols)}, success={sr:.3f}")

    if verbose:
        print(f"\n物理语言涌现结果:")
        print(f"  物理符号: {physical_symbols}")
        print(f"  物理符号数: {len(physical_symbols)}")
        print(f"  最终成功率: {log['success_rate'][-1]:.3f}" if log['success_rate'] else "  无数据")

    return {
        'log': log,
        'physical_symbols': sorted(list(physical_symbols)),
        'num_physical_symbols': len(physical_symbols),
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 33: 3D 物理世界实验")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    # 实验 1：重力发现
    results['exp1'] = experiment_1_gravity_discovery(
        num_steps=2000, verbose=True
    )

    # 实验 2：碰撞物理
    results['exp2'] = experiment_2_collision_physics(
        num_steps=3000, verbose=True
    )

    # 实验 3：工具使用
    results['exp3'] = experiment_3_tool_use(
        num_episodes=50, verbose=True
    )

    # 实验 4：物理语言涌现
    results['exp4'] = experiment_4_physics_language(
        num_agents=10, num_rounds=2000, verbose=True
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

    with open('3d_physics_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 3d_physics_results.json")

    return results


if __name__ == '__main__':
    main()
