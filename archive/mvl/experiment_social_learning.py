"""
Phase 35 实验：社会学习

4 个实验：
1. 空间邻近通信：Agent 在共享世界中移动，近距离通信
2. 协作搬运：重物体需要两个 Agent 合力移动
3. 观察学习：观察他人动作 vs 独立学习
4. 社会语言涌现：多 Agent 共享世界中的语言演化
"""

import sys
import random
import numpy as np
import json
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from multi_agent_3d_env import MultiAgent3DEnv
from agent_social import SocialAgent3D
from environment_3d import MATERIAL_NAMES


# ============================================================
# 实验 1：空间邻近通信
# ============================================================

def experiment_1_proximity_communication(num_agents: int = 5,
                                          num_steps: int = 1000,
                                          verbose: bool = True) -> Dict:
    """
    空间邻近通信实验

    5 个 Agent 在共享世界中自由移动。
    当两个 Agent 靠近时，自动尝试通信。

    测量：通信成功率 vs 距离
    预期：近距离通信成功率 > 远距离
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：空间邻近通信")
        print("=" * 60)

    # 创建共享环境
    env = MultiAgent3DEnv(bounds=(12.0, 12.0, 5.0), comm_range=3.0)
    env.physics.add_random_objects(6, shapes=['sphere', 'cube', 'cylinder'],
                                    materials=MATERIAL_NAMES[:4])

    # 创建 agent
    agents = {}
    for i in range(num_agents):
        agent_id = env.add_agent()
        agents[agent_id] = SocialAgent3D(agent_id, use_language=True)

    # 统计
    close_comm = 0      # 距离 < 2 的通信
    close_success = 0
    medium_comm = 0     # 距离 2-3
    medium_success = 0

    log = {'steps': [], 'total_comm': [], 'success_rate': []}
    total_comm = 0
    total_success = 0

    for step in range(num_steps):
        # 每个 agent 随机移动
        actions = {}
        for agent_id in agents:
            actions[agent_id] = agents[agent_id]._random_continuous_action()

        # 执行
        observations = env.step(actions)

        # 更新 agent 学习
        for agent_id, obs in observations.items():
            agents[agent_id].step_with_observation(obs, actions[agent_id])

        # 检查邻近通信
        pairs = env.get_nearby_pairs()
        for id_a, id_b in pairs:
            agent_a = agents[id_a]
            agent_b = agents[id_b]

            # 获取可见物体
            visible = env.physics.get_nearby_objects(env.agents[id_a].pos, radius=3.0)
            if not visible:
                continue

            # 选一个物体描述
            target = random.choice(visible)
            dist = np.linalg.norm(env.agents[id_a].pos - env.agents[id_b].pos)

            # 通信
            success = agent_a.try_communicate(agent_b, target)

            total_comm += 1
            if success:
                total_success += 1

            if dist < 2.0:
                close_comm += 1
                if success:
                    close_success += 1
            else:
                medium_comm += 1
                if success:
                    medium_success += 1

        if (step + 1) % 200 == 0:
            sr = total_success / total_comm if total_comm > 0 else 0
            log['steps'].append(step + 1)
            log['total_comm'].append(total_comm)
            log['success_rate'].append(sr)

            if verbose:
                print(f"  Step {step+1:5d}: comm={total_comm}, success={sr:.3f}")

    # 分析
    close_sr = close_success / close_comm if close_comm > 0 else 0
    medium_sr = medium_success / medium_comm if medium_comm > 0 else 0

    if verbose:
        print(f"\n空间邻近通信结果:")
        print(f"  近距离 (<2) 通信: {close_comm} 次, 成功率 {close_sr:.3f}")
        print(f"  中距离 (2-3) 通信: {medium_comm} 次, 成功率 {medium_sr:.3f}")
        print(f"  总通信: {total_comm} 次, 总成功率 {total_success/total_comm:.3f}" if total_comm > 0 else "")

    return {
        'log': log,
        'close_distance_rate': close_sr,
        'medium_distance_rate': medium_sr,
        'total_comm': total_comm,
        'total_success_rate': total_success / total_comm if total_comm > 0 else 0,
    }


# ============================================================
# 实验 2：协作搬运
# ============================================================

def experiment_2_cooperative_carry(num_episodes: int = 30,
                                    verbose: bool = True) -> Dict:
    """
    协作搬运实验

    一个重物体，两个 Agent 同时推 vs 一个 Agent 单独推。
    测量物体移动距离。

    预期：两人合力 > 一人
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：协作搬运")
        print("=" * 60)

    coop_distances = []
    solo_distances = []

    for ep in range(num_episodes):
        # --- 协作：两人推（使用离散动作 PUSH=11） ---
        env = MultiAgent3DEnv(bounds=(10.0, 10.0, 5.0))
        heavy_pos = np.array([5.0, 5.0, 0.5])
        heavy_id = env.physics.add_object(heavy_pos, mass=3.0, radius=0.4, material='stone')

        # 两个 agent 在物体左侧，面向物体
        a_id = env.add_agent(np.array([3.5, 4.5, 0.5]))
        b_id = env.add_agent(np.array([3.5, 5.5, 0.5]))
        env.agents[a_id].facing = 0.0  # 面右（朝物体）
        env.agents[b_id].facing = 0.0

        # 先前进靠近物体，再推
        for _ in range(5):
            env.step({a_id: 0, b_id: 0})  # FORWARD
        for _ in range(15):
            env.step({a_id: 11, b_id: 11})  # PUSH

        obj = env.physics._get_object(heavy_id)
        coop_dist = np.linalg.norm(obj.position[:2] - heavy_pos[:2]) if obj else 0
        coop_distances.append(coop_dist)

        # --- 单人：一人推 ---
        env2 = MultiAgent3DEnv(bounds=(10.0, 10.0, 5.0))
        heavy_pos2 = np.array([5.0, 5.0, 0.5])
        heavy_id2 = env2.physics.add_object(heavy_pos2, mass=3.0, radius=0.4, material='stone')

        solo_id = env2.add_agent(np.array([3.5, 5.0, 0.5]))
        env2.agents[solo_id].facing = 0.0

        for _ in range(5):
            env2.step({solo_id: 0})  # FORWARD
        for _ in range(15):
            env2.step({solo_id: 11})  # PUSH

        obj2 = env2.physics._get_object(heavy_id2)
        solo_dist = np.linalg.norm(obj2.position[:2] - heavy_pos2[:2]) if obj2 else 0
        solo_distances.append(solo_dist)

        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1:3d}: coop={np.mean(coop_distances):.3f}, "
                  f"solo={np.mean(solo_distances):.3f}")

    coop_mean = np.mean(coop_distances)
    solo_mean = np.mean(solo_distances)

    if verbose:
        print(f"\n协作搬运结果:")
        print(f"  协作平均移动: {coop_mean:.3f}")
        print(f"  单人平均移动: {solo_mean:.3f}")
        print(f"  协作优势: {(coop_mean - solo_mean):.3f}")

    return {
        'coop_distance': coop_mean,
        'solo_distance': solo_mean,
        'improvement': coop_mean - solo_mean,
    }


# ============================================================
# 实验 3：观察学习
# ============================================================

def experiment_3_observational_learning(num_trials: int = 20,
                                         verbose: bool = True) -> Dict:
    """
    模仿学习实验

    对比两种学习方式：
    - 社会学习：新手 agent 与有经验 agent 在同一世界，以 30% 概率模仿对方动作
    - 独立学习：新手 agent 独自学习

    测量：100 步后的预测误差
    预期：社会学习 < 独立学习（模仿有经验者加速学习）
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：模仿学习")
        print("=" * 60)

    social_errors = []
    independent_errors = []

    for trial in range(num_trials):
        # --- 社会学习：新手 + 有经验 agent 在同一世界 ---
        env_social = MultiAgent3DEnv(bounds=(10.0, 10.0, 5.0))
        env_social.physics.add_random_objects(4)

        mentor_id = env_social.add_agent(np.array([3.0, 5.0, 0.5]))
        novice_id = env_social.add_agent(np.array([7.0, 5.0, 0.5]))

        mentor = SocialAgent3D(mentor_id, use_language=False)
        novice = SocialAgent3D(novice_id, use_language=False)

        # 导师先学习 50 步（积累经验）
        for _ in range(50):
            action = mentor._random_continuous_action()
            obs = env_social.step({mentor_id: action})
            mentor.step_with_observation(obs[mentor_id], action)

        # 新手学习 100 步，30% 概率模仿导师动作
        for _ in range(100):
            mentor_action = mentor._random_continuous_action()

            if np.random.random() < 0.3:
                # 模仿：新手使用导师的动作
                novice_action = mentor_action.copy()
            else:
                novice_action = novice._random_continuous_action()

            obs = env_social.step({mentor_id: mentor_action, novice_id: novice_action})
            novice.step_with_observation(obs[novice_id], novice_action)
            mentor.step_with_observation(obs[mentor_id], mentor_action)

        # 评估新手
        eval_errors = []
        for _ in range(20):
            action = novice._random_continuous_action()
            obs = env_social.step({novice_id: action})
            result = novice.step_with_observation(obs[novice_id], action)
            eval_errors.append(result['prediction_error'])
        social_errors.append(np.mean(eval_errors))

        # --- 独立学习：单个 agent ---
        env_ind = MultiAgent3DEnv(bounds=(10.0, 10.0, 5.0))
        env_ind.physics.add_random_objects(4)

        ind_id = env_ind.add_agent(np.array([5.0, 5.0, 0.5]))
        independent = SocialAgent3D(ind_id, use_language=False)

        # 同样 100 步
        for _ in range(100):
            action = independent._random_continuous_action()
            obs = env_ind.step({ind_id: action})
            independent.step_with_observation(obs[ind_id], action)

        # 评估
        eval_errors_ind = []
        for _ in range(20):
            action = independent._random_continuous_action()
            obs = env_ind.step({ind_id: action})
            result = independent.step_with_observation(obs[ind_id], action)
            eval_errors_ind.append(result['prediction_error'])
        independent_errors.append(np.mean(eval_errors_ind))

        if verbose and (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1:3d}: social={np.mean(social_errors):.4f}, "
                  f"independent={np.mean(independent_errors):.4f}")

    social_mean = np.mean(social_errors)
    ind_mean = np.mean(independent_errors)

    if verbose:
        print(f"\n模仿学习结果:")
        print(f"  社会学习误差: {social_mean:.4f}")
        print(f"  独立学习误差: {ind_mean:.4f}")
        print(f"  社会学习优势: {(ind_mean - social_mean):.4f}")

    return {
        'social_error': social_mean,
        'independent_error': ind_mean,
        'advantage': ind_mean - social_mean,
    }


# ============================================================
# 实验 4：社会语言涌现
# ============================================================

def experiment_4_social_language(num_agents: int = 6,
                                  num_steps: int = 1000,
                                  verbose: bool = True) -> Dict:
    """
    社会语言涌现实验

    多 Agent 在共享世界中自由移动和通信。
    观察语言从社会交互中涌现。

    测量：词汇量、通信成功率
    预期：共享世界的语言符号 > 独立世界
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 4：社会语言涌现")
        print("=" * 60)

    # 创建共享环境
    env = MultiAgent3DEnv(bounds=(14.0, 14.0, 5.0), comm_range=3.5)
    env.physics.add_random_objects(8,
                                    shapes=['sphere', 'cube', 'cylinder'],
                                    materials=MATERIAL_NAMES)

    # 创建 agent
    agents = {}
    for i in range(num_agents):
        agent_id = env.add_agent()
        agents[agent_id] = SocialAgent3D(agent_id, use_language=True)

    log = {'steps': [], 'vocab_size': [], 'success_rate': [], 'comm_count': []}
    all_symbols = set()
    total_comm = 0
    total_success = 0

    for step in range(num_steps):
        # 每个 agent 选择动作
        actions = {}
        for agent_id, agent in agents.items():
            # 获取附近 agent 的动作（用于社会参考）
            nearby = env.get_nearby_agents(agent_id)
            nearby_actions = []
            for nid in nearby:
                if nid in agents and agents[nid]._last_action is not None:
                    nearby_actions.append(agents[nid]._last_action)

            actions[agent_id] = agent.choose_action_social(
                epsilon=0.15, nearby_actions=nearby_actions if nearby_actions else None
            )

        # 执行
        observations = env.step(actions)

        # 更新 agent 学习
        for agent_id, obs in observations.items():
            agents[agent_id].step_with_observation(obs, actions[agent_id])

        # 近距离 agent 之间通信
        pairs = env.get_nearby_pairs()
        for id_a, id_b in pairs:
            visible = env.physics.get_nearby_objects(env.agents[id_a].pos, radius=3.0)
            if not visible:
                continue

            target = random.choice(visible)
            success = agents[id_a].try_communicate(agents[id_b], target)

            total_comm += 1
            if success:
                total_success += 1

            # 记录符号
            symbols = agents[id_a].describe_object(target)
            for s in symbols:
                all_symbols.add(s)

        # 观察学习：agent 观察附近 agent 的动作
        for agent_id, agent in agents.items():
            nearby = env.get_nearby_agents(agent_id)
            for nid in nearby:
                if nid in observations and agent._last_encoded is not None:
                    other_encoded = agents[nid].encoder.encode(observations[nid])
                    agent.observe_other(
                        actions[nid], agent._last_encoded, other_encoded
                    )

        if (step + 1) % 200 == 0:
            sr = total_success / total_comm if total_comm > 0 else 0
            log['steps'].append(step + 1)
            log['vocab_size'].append(len(all_symbols))
            log['success_rate'].append(sr)
            log['comm_count'].append(total_comm)

            if verbose:
                print(f"  Step {step+1:5d}: vocab={len(all_symbols)}, "
                      f"comm={total_comm}, success={sr:.3f}")

    sr = total_success / total_comm if total_comm > 0 else 0

    if verbose:
        print(f"\n社会语言涌现结果:")
        print(f"  总符号数: {len(all_symbols)}")
        print(f"  符号: {sorted(list(all_symbols))}")
        print(f"  总通信次数: {total_comm}")
        print(f"  通信成功率: {sr:.3f}")

    return {
        'log': log,
        'all_symbols': sorted(list(all_symbols)),
        'num_symbols': len(all_symbols),
        'total_comm': total_comm,
        'success_rate': sr,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 35: 社会学习实验")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    # 实验 1：空间邻近通信
    results['exp1'] = experiment_1_proximity_communication(
        num_agents=5, num_steps=1000, verbose=True
    )

    # 实验 2：协作搬运
    results['exp2'] = experiment_2_cooperative_carry(
        num_episodes=30, verbose=True
    )

    # 实验 3：观察学习
    results['exp3'] = experiment_3_observational_learning(
        num_trials=20, verbose=True
    )

    # 实验 4：社会语言涌现
    results['exp4'] = experiment_4_social_language(
        num_agents=6, num_steps=1000, verbose=True
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
        return obj

    with open('social_learning_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 social_learning_results.json")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 35 总结")
    print("=" * 60)
    print(f"  近距离通信成功率: {results['exp1']['close_distance_rate']:.3f}")
    print(f"  协作搬运优势: {results['exp2']['improvement']:.3f}")
    print(f"  共享学习优势: {results['exp3']['advantage']:.4f}")
    print(f"  社会语言符号数: {results['exp4']['num_symbols']}")

    return results


if __name__ == '__main__':
    main()
