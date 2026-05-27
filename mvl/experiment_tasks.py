"""
复杂物理任务对比实验

三条件对比：
A. 无任务（纯好奇心）- 基线
B. 任务+好奇心（混合奖励）- 设计方案
C. 纯任务（外在奖励，无好奇心）- 对照

验证：任务驱动学习 vs 纯好奇心学习的差异
"""

import sys
import numpy as np
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from agent_3d import LearningAgent3D
from task_environments import (
    create_push_to_goal_env,
    create_push_to_goal_env_easy,
    create_no_task_env,
)


def run_condition_a(num_steps: int = 500):
    """条件A：无任务（纯好奇心）"""
    print("\n--- 条件A：无任务（纯好奇心）---")
    env = create_no_task_env()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': [],
        'task_rewards': [],
    }

    for step in range(num_steps):
        obs = env.get_observation()
        action = agent.act(obs)
        next_obs, reward, done = env.step(action)

        # 无任务，外在奖励为 0
        error = agent.learn_from_experience(obs, action, next_obs, extrinsic_reward=0.0)

        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())
        trajectory['task_rewards'].append(0.0)

        if done:
            env.reset()

    return trajectory


def run_condition_b(num_steps: int = 500):
    """条件B：任务+好奇心（混合奖励）"""
    print("\n--- 条件B：任务+好奇心（混合奖励）---")
    env, task = create_push_to_goal_env_easy()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    # 解锁所有动作（包括推/拉）
    agent._get_available_actions = lambda: list(range(8))

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': [],
        'task_rewards': [],
        'success_count': 0,
        'total_episodes': 0,
    }

    for step in range(num_steps):
        obs = env.get_observation()
        action = agent.act(obs)
        next_obs, reward, done = env.step(action)

        # 任务奖励作为外在奖励
        error = agent.learn_from_experience(obs, action, next_obs, extrinsic_reward=reward)

        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())
        trajectory['task_rewards'].append(reward)

        if done:
            trajectory['total_episodes'] += 1
            if task.is_success:
                trajectory['success_count'] += 1
            env.reset()

    return trajectory


def run_condition_c(num_steps: int = 500):
    """条件C：纯任务（外在奖励，无好奇心）"""
    print("\n--- 条件C：纯任务（外在奖励，无好奇心）---")
    env, task = create_push_to_goal_env_easy()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    # 解锁所有动作
    agent._available_actions = list(range(8))
    # 禁用好奇心
    agent.curiosity.alpha = 0.0
    agent.curiosity.beta = 0.0

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': [],
        'task_rewards': [],
        'success_count': 0,
        'total_episodes': 0,
    }

    for step in range(num_steps):
        obs = env.get_observation()
        action = agent.act(obs)
        next_obs, reward, done = env.step(action)

        # 任务奖励作为外在奖励
        error = agent.learn_from_experience(obs, action, next_obs, extrinsic_reward=reward)

        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())
        trajectory['task_rewards'].append(reward)

        if done:
            trajectory['total_episodes'] += 1
            if task.is_success:
                trajectory['success_count'] += 1
            env.reset()

    return trajectory


def test_physics_generalization(agent_a, agent_b, agent_c) -> Dict:
    """
    物理理解泛化测试

    在新场景（新材质、新位置）上测试预测准确率。
    真正理解物理的 agent 应该在新场景上也有较低的预测误差。
    """
    from environment_physics import PhysicsEnvironment
    from physics_rigid import RigidBody, MaterialType

    print("\n--- 物理理解泛化测试 ---")

    # 创建新场景：不同位置、不同材质
    test_env = PhysicsEnvironment(10, 10, 10)
    test_env.add_rigid_body(RigidBody(5, 5, 5, radius=0.3, mass=5.0, material=MaterialType.STONE))

    errors = {'agent_a': [], 'agent_b': [], 'agent_c': []}
    agents = {'agent_a': agent_a, 'agent_b': agent_b, 'agent_c': agent_c}

    for name, agent in agents.items():
        test_env_copy = PhysicsEnvironment(10, 10, 10)
        test_env_copy.add_rigid_body(RigidBody(5, 5, 5, radius=0.3, mass=5.0, material=MaterialType.STONE))

        for _ in range(50):
            obs = test_env_copy.get_observation()
            action = agent.act(obs)
            next_obs, _, done = test_env_copy.step(action)

            # 只测试预测，不学习
            obs_vec = agent.perceive(obs)
            next_obs_vec = agent.perceive(next_obs)
            predicted = agent.predictive_model.predict(obs_vec, action)
            error = np.mean((predicted - next_obs_vec)**2)
            errors[name].append(error)

            if done:
                test_env_copy.reset()

    return {
        'agent_a_mean_error': np.mean(errors['agent_a']),
        'agent_b_mean_error': np.mean(errors['agent_b']),
        'agent_c_mean_error': np.mean(errors['agent_c']),
    }


def analyze_results(results: Dict):
    """分析实验结果"""
    print("\n" + "=" * 60)
    print("复杂物理任务对比实验结果")
    print("=" * 60)

    traj_a = results['condition_a']
    traj_b = results['condition_b']
    traj_c = results['condition_c']

    # 最终预测误差
    final_error_a = np.mean(traj_a['prediction_errors'][-50:])
    final_error_b = np.mean(traj_b['prediction_errors'][-50:])
    final_error_c = np.mean(traj_c['prediction_errors'][-50:])

    print(f"\n1. 最终预测误差（越低越好）:")
    print(f"   条件A（纯好奇心）: {final_error_a:.4f}")
    print(f"   条件B（任务+好奇心）: {final_error_b:.4f}")
    print(f"   条件C（纯任务）: {final_error_c:.4f}")

    # 学习进度
    final_progress_a = traj_a['learning_progress'][-1]
    final_progress_b = traj_b['learning_progress'][-1]
    final_progress_c = traj_c['learning_progress'][-1]

    print(f"\n2. 学习进度:")
    print(f"   条件A（纯好奇心）: {final_progress_a:.2%}")
    print(f"   条件B（任务+好奇心）: {final_progress_b:.2%}")
    print(f"   条件C（纯任务）: {final_progress_c:.2%}")

    # 符号学习
    final_symbols_a = traj_a['symbols'][-1]
    final_symbols_b = traj_b['symbols'][-1]
    final_symbols_c = traj_c['symbols'][-1]

    print(f"\n3. 符号学习:")
    print(f"   条件A（纯好奇心）: {final_symbols_a} 个")
    print(f"   条件B（任务+好奇心）: {final_symbols_b} 个")
    print(f"   条件C（纯任务）: {final_symbols_c} 个")

    # 任务成功率
    if 'success_count' in traj_b:
        success_rate_b = traj_b['success_count'] / max(traj_b['total_episodes'], 1)
        print(f"\n4. 任务成功率:")
        print(f"   条件B（任务+好奇心）: {success_rate_b:.1%} ({traj_b['success_count']}/{traj_b['total_episodes']})")

    if 'success_count' in traj_c:
        success_rate_c = traj_c['success_count'] / max(traj_c['total_episodes'], 1)
        print(f"   条件C（纯任务）: {success_rate_c:.1%} ({traj_c['success_count']}/{traj_c['total_episodes']})")

    # 任务奖励
    avg_reward_b = np.mean(traj_b['task_rewards'][-100:])
    avg_reward_c = np.mean(traj_c['task_rewards'][-100:])

    print(f"\n5. 平均任务奖励（最后100步）:")
    print(f"   条件B（任务+好奇心）: {avg_reward_b:.4f}")
    print(f"   条件C（纯任务）: {avg_reward_c:.4f}")

    # 泛化测试
    if 'generalization' in results:
        gen = results['generalization']
        print(f"\n6. 物理理解泛化测试（新场景预测误差）:")
        print(f"   条件A（纯好奇心）: {gen['agent_a_mean_error']:.4f}")
        print(f"   条件B（任务+好奇心）: {gen['agent_b_mean_error']:.4f}")
        print(f"   条件C（纯任务）: {gen['agent_c_mean_error']:.4f}")


def main():
    """主函数"""
    print("开始复杂物理任务对比实验...")

    num_steps = 500

    # 运行三个条件
    traj_a = run_condition_a(num_steps)
    traj_b = run_condition_b(num_steps)
    traj_c = run_condition_c(num_steps)

    # 泛化测试
    # 需要重新创建 agent 来测试
    env_a = create_no_task_env()
    agent_a = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    # 快速训练 agent_a
    for _ in range(200):
        obs = env_a.get_observation()
        action = agent_a.act(obs)
        next_obs, _, done = env_a.step(action)
        agent_a.learn_from_experience(obs, action, next_obs)
        if done:
            env_a.reset()

    env_b, _ = create_push_to_goal_env_easy()
    agent_b = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    agent_b._available_actions = list(range(8))
    for _ in range(200):
        obs = env_b.get_observation()
        action = agent_b.act(obs)
        next_obs, reward, done = env_b.step(action)
        agent_b.learn_from_experience(obs, action, next_obs, extrinsic_reward=reward)
        if done:
            env_b.reset()

    env_c, _ = create_push_to_goal_env_easy()
    agent_c = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    agent_c._available_actions = list(range(8))
    agent_c.curiosity.alpha = 0.0
    agent_c.curiosity.beta = 0.0
    for _ in range(200):
        obs = env_c.get_observation()
        action = agent_c.act(obs)
        next_obs, reward, done = env_c.step(action)
        agent_c.learn_from_experience(obs, action, next_obs, extrinsic_reward=reward)
        if done:
            env_c.reset()

    generalization = test_physics_generalization(agent_a, agent_b, agent_c)

    # 分析结果
    results = {
        'condition_a': traj_a,
        'condition_b': traj_b,
        'condition_c': traj_c,
        'generalization': generalization,
    }
    analyze_results(results)

    # 保存结果
    import json
    output = {
        'condition_a': {
            'final_error': float(np.mean(traj_a['prediction_errors'][-50:])),
            'final_progress': float(traj_a['learning_progress'][-1]),
            'final_symbols': int(traj_a['symbols'][-1]),
        },
        'condition_b': {
            'final_error': float(np.mean(traj_b['prediction_errors'][-50:])),
            'final_progress': float(traj_b['learning_progress'][-1]),
            'final_symbols': int(traj_b['symbols'][-1]),
            'success_rate': float(traj_b['success_count'] / max(traj_b['total_episodes'], 1)),
        },
        'condition_c': {
            'final_error': float(np.mean(traj_c['prediction_errors'][-50:])),
            'final_progress': float(traj_c['learning_progress'][-1]),
            'final_symbols': int(traj_c['symbols'][-1]),
            'success_rate': float(traj_c['success_count'] / max(traj_c['total_episodes'], 1)),
        },
        'generalization': generalization,
    }

    with open('D:/mayAi/AILearning_v0527/mvl/tasks_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n结果已保存到: tasks_comparison.json")


if __name__ == '__main__':
    main()
