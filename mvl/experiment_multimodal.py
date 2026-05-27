"""
多模态感知对比实验

三种感知模式对比：
A. 结构化特征（agent_3d 的 perceive）—— 基线
B. 纯视觉（视觉场，无听觉）
C. 视觉+听觉（完整多模态）

验证：原始感官输入 vs 预处理特征的学习效果差异
"""

import sys
import numpy as np
from typing import Dict

sys.stdout.reconfigure(encoding='utf-8')

from environment_physics import PhysicsEnvironment, create_rich_physics_world
from agent_3d import LearningAgent3D
from agent_multimodal import MultimodalAgent


def run_structured_features(num_steps: int = 500):
    """条件A：结构化特征（baseline）"""
    print("\n--- 条件A：结构化特征（agent_3d perceive）---")
    env = create_rich_physics_world()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'learning_progress': [],
        'symbols': [],
    }

    for step in range(num_steps):
        obs = env.get_observation()
        action = agent.act(obs)
        next_obs, _, done = env.step(action)

        error = agent.learn_from_experience(obs, action, next_obs)

        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))

        if done:
            env.reset()

    return trajectory


def run_visual_only(num_steps: int = 500):
    """条件B：纯视觉（视觉场，无听觉）"""
    print("\n--- 条件B：纯视觉（视觉场）---")
    env = create_rich_physics_world()
    agent = MultimodalAgent(action_dim=8, fusion_dim=64)

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'learning_progress': [],
        'symbols': [],
    }

    for step in range(num_steps):
        obs = env.get_observation()
        # 移除听觉信息
        obs_mod = obs.copy()
        obs_mod['audio_events'] = np.zeros(56)

        action = agent.act(obs_mod)
        next_obs, _, done = env.step(action)
        next_obs_mod = next_obs.copy()
        next_obs_mod['audio_events'] = np.zeros(56)

        error = agent.learn_from_experience(obs_mod, action, next_obs_mod)

        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))

        if done:
            env.reset()

    return trajectory


def run_visual_plus_audio(num_steps: int = 500):
    """条件C：视觉+听觉（完整多模态）"""
    print("\n--- 条件C：视觉+听觉（完整多模态）---")
    env = create_rich_physics_world()
    agent = MultimodalAgent(action_dim=8, fusion_dim=64)

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'learning_progress': [],
        'symbols': [],
    }

    for step in range(num_steps):
        obs = env.get_observation()
        action = agent.act(obs)
        next_obs, _, done = env.step(action)

        error = agent.learn_from_experience(obs, action, next_obs)

        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))

        if done:
            env.reset()

    return trajectory


def test_generalization(agent_structured, agent_visual, agent_multimodal) -> Dict:
    """
    泛化测试：在新场景上测试预测准确率

    真正理解物理的 agent 应该在新场景上也有较低的预测误差。
    """
    from physics_rigid import RigidBody, MaterialType
    from physics_fluid import FluidRegion

    print("\n--- 泛化测试 ---")

    errors = {'structured': [], 'visual': [], 'multimodal': []}
    agents = {
        'structured': agent_structured,
        'visual': agent_visual,
        'multimodal': agent_multimodal,
    }

    for name, agent in agents.items():
        # 创建新场景
        test_env = PhysicsEnvironment(10, 10, 10)
        test_env.add_rigid_body(RigidBody(3, 3, 5, radius=0.4, mass=3.0, material=MaterialType.STONE))
        test_env.add_rigid_body(RigidBody(7, 7, 3, radius=0.6, mass=1.0, material=MaterialType.RUBBER))
        test_env.add_fluid_region(FluidRegion('water',
                                               x_min=0, y_min=0, z_min=0,
                                               x_max=10, y_max=10, z_max=3))

        for _ in range(50):
            obs = test_env.get_observation()
            action = agent.act(obs)
            next_obs, _, done = test_env.step(action)

            # 只测试预测，不学习
            obs_encoded = agent.perceive(obs)
            next_obs_encoded = agent.perceive(next_obs)
            predicted = agent.predictive_model.predict(obs_encoded, action)
            error = np.mean((predicted - next_obs_encoded) ** 2)
            errors[name].append(error)

            if done:
                test_env.reset()

    return {
        'structured_mean_error': np.mean(errors['structured']),
        'visual_mean_error': np.mean(errors['visual']),
        'multimodal_mean_error': np.mean(errors['multimodal']),
    }


def analyze_results(results: Dict):
    """分析实验结果"""
    print("\n" + "=" * 60)
    print("多模态感知对比实验结果")
    print("=" * 60)

    traj_a = results['structured']
    traj_b = results['visual']
    traj_c = results['multimodal']

    # 最终预测误差
    final_a = np.mean(traj_a['prediction_errors'][-50:])
    final_b = np.mean(traj_b['prediction_errors'][-50:])
    final_c = np.mean(traj_c['prediction_errors'][-50:])

    print(f"\n1. 最终预测误差（最后50步均值）:")
    print(f"   条件A（结构化特征）: {final_a:.4f}")
    print(f"   条件B（纯视觉）:     {final_b:.4f}")
    print(f"   条件C（视觉+听觉）:  {final_c:.4f}")

    # 学习进度
    prog_a = traj_a['learning_progress'][-1]
    prog_b = traj_b['learning_progress'][-1]
    prog_c = traj_c['learning_progress'][-1]

    print(f"\n2. 学习进度:")
    print(f"   条件A（结构化特征）: {prog_a:.2%}")
    print(f"   条件B（纯视觉）:     {prog_b:.2%}")
    print(f"   条件C（视觉+听觉）:  {prog_c:.2%}")

    # 符号数量
    sym_a = traj_a['symbols'][-1]
    sym_b = traj_b['symbols'][-1]
    sym_c = traj_c['symbols'][-1]

    print(f"\n3. 符号学习:")
    print(f"   条件A（结构化特征）: {sym_a} 个")
    print(f"   条件B（纯视觉）:     {sym_b} 个")
    print(f"   条件C（视觉+听觉）:  {sym_c} 个")

    # 泛化测试
    if 'generalization' in results:
        gen = results['generalization']
        print(f"\n4. 泛化测试（新场景预测误差）:")
        print(f"   条件A（结构化特征）: {gen['structured_mean_error']:.4f}")
        print(f"   条件B（纯视觉）:     {gen['visual_mean_error']:.4f}")
        print(f"   条件C（视觉+听觉）:  {gen['multimodal_mean_error']:.4f}")


def main():
    """主函数"""
    print("开始多模态感知对比实验...")

    num_steps = 500

    # 运行三个条件
    traj_a = run_structured_features(num_steps)
    traj_b = run_visual_only(num_steps)
    traj_c = run_visual_plus_audio(num_steps)

    # 泛化测试：快速训练 agent 并测试
    print("\n训练泛化测试用 agent...")

    env_a = create_rich_physics_world()
    agent_a = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    for _ in range(200):
        obs = env_a.get_observation()
        action = agent_a.act(obs)
        next_obs, _, done = env_a.step(action)
        agent_a.learn_from_experience(obs, action, next_obs)
        if done:
            env_a.reset()

    env_b = create_rich_physics_world()
    agent_b = MultimodalAgent(action_dim=8, fusion_dim=64)
    for _ in range(200):
        obs = env_b.get_observation()
        obs_mod = obs.copy()
        obs_mod['audio_events'] = np.zeros(56)
        action = agent_b.act(obs_mod)
        next_obs, _, done = env_b.step(action)
        next_obs_mod = next_obs.copy()
        next_obs_mod['audio_events'] = np.zeros(56)
        agent_b.learn_from_experience(obs_mod, action, next_obs_mod)
        if done:
            env_b.reset()

    env_c = create_rich_physics_world()
    agent_c = MultimodalAgent(action_dim=8, fusion_dim=64)
    for _ in range(200):
        obs = env_c.get_observation()
        action = agent_c.act(obs)
        next_obs, _, done = env_c.step(action)
        agent_c.learn_from_experience(obs, action, next_obs)
        if done:
            env_c.reset()

    generalization = test_generalization(agent_a, agent_b, agent_c)

    # 分析结果
    results = {
        'structured': traj_a,
        'visual': traj_b,
        'multimodal': traj_c,
        'generalization': generalization,
    }
    analyze_results(results)

    # 保存结果
    import json
    output = {
        'structured': {
            'final_error': float(np.mean(traj_a['prediction_errors'][-50:])),
            'final_progress': float(traj_a['learning_progress'][-1]),
            'final_symbols': int(traj_a['symbols'][-1]),
        },
        'visual_only': {
            'final_error': float(np.mean(traj_b['prediction_errors'][-50:])),
            'final_progress': float(traj_b['learning_progress'][-1]),
            'final_symbols': int(traj_b['symbols'][-1]),
        },
        'visual_plus_audio': {
            'final_error': float(np.mean(traj_c['prediction_errors'][-50:])),
            'final_progress': float(traj_c['learning_progress'][-1]),
            'final_symbols': int(traj_c['symbols'][-1]),
        },
        'generalization': generalization,
    }

    with open('D:/mayAi/AILearning_v0527/mvl/multimodal_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n结果已保存到: multimodal_comparison.json")


if __name__ == '__main__':
    main()
