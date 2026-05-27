"""
物理效果对比实验

比较简单物理和真实物理的学习效果。

实验设计：
1. 简单物理：基本的重力和碰撞
2. 真实物理：刚体、流体、软体

验证：真实物理是否提供更丰富的学习机会
"""

import sys
import numpy as np
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from environment_3d import create_rich_3d_world, GridWorld3D
from environment_physics import create_rich_physics_world, PhysicsEnvironment
from agent_3d import LearningAgent3D


def run_simple_physics_experiment(num_steps: int = 300):
    """运行简单物理实验"""
    print("\n--- 简单物理实验 ---")
    env = create_rich_3d_world()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': []
    }

    for step in range(num_steps):
        obs = env.get_observation()

        # 学习体行动
        action = agent.act(obs)
        next_obs, _, done = env.step(action)
        error = agent.learn_from_experience(obs, action, next_obs)

        # 记录
        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())

        if done:
            env.reset()

    return trajectory


def run_real_physics_experiment(num_steps: int = 300):
    """运行真实物理实验"""
    print("\n--- 真实物理实验 ---")
    env = create_rich_physics_world()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': []
    }

    for step in range(num_steps):
        obs = env.get_observation()

        # 学习体行动
        action = agent.act(obs)
        next_obs, _, done = env.step(action)
        error = agent.learn_from_experience(obs, action, next_obs)

        # 记录
        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())

        if done:
            env.reset()

    return trajectory


def analyze_results(results: Dict):
    """分析实验结果"""
    print("\n" + "=" * 60)
    print("简单物理 vs 真实物理对比分析")
    print("=" * 60)

    traj_simple = results['simple']
    traj_real = results['real']

    # 最终结果
    final_error_simple = np.mean(traj_simple['prediction_errors'][-50:])
    final_error_real = np.mean(traj_real['prediction_errors'][-50:])
    final_symbols_simple = traj_simple['symbols'][-1]
    final_symbols_real = traj_real['symbols'][-1]
    final_progress_simple = traj_simple['learning_progress'][-1]
    final_progress_real = traj_real['learning_progress'][-1]

    print(f"\n1. 最终预测误差:")
    print(f"   简单物理: {final_error_simple:.4f}")
    print(f"   真实物理: {final_error_real:.4f}")

    print(f"\n2. 符号学习:")
    print(f"   简单物理: {final_symbols_simple} 个符号")
    print(f"   真实物理: {final_symbols_real} 个符号")

    print(f"\n3. 学习进度:")
    print(f"   简单物理: {final_progress_simple:.2%}")
    print(f"   真实物理: {final_progress_real:.2%}")

    # 学习曲线分析
    print(f"\n4. 学习曲线:")
    if len(traj_simple['prediction_errors']) > 50:
        early_simple = np.mean(traj_simple['prediction_errors'][:50])
        late_simple = np.mean(traj_simple['prediction_errors'][-50:])
        if early_simple > 0:
            reduction_simple = (early_simple - late_simple) / early_simple
            print(f"   简单物理: 误差减少 {reduction_simple:.1%}")

    if len(traj_real['prediction_errors']) > 50:
        early_real = np.mean(traj_real['prediction_errors'][:50])
        late_real = np.mean(traj_real['prediction_errors'][-50:])
        if early_real > 0:
            reduction_real = (early_real - late_real) / early_real
            print(f"   真实物理: 误差减少 {reduction_real:.1%}")

    # 物理效果分析
    print(f"\n5. 物理效果分析:")
    print(f"   简单物理: 基本重力和碰撞")
    print(f"   真实物理: 刚体动力学 + 流体力学 + 软体物理")
    print(f"   真实物理提供了更丰富的物理交互机会")


def main():
    """主函数"""
    print("开始物理效果对比实验...")

    # 运行实验
    traj_simple = run_simple_physics_experiment(num_steps=300)
    traj_real = run_real_physics_experiment(num_steps=300)

    # 分析结果
    results = {
        'simple': traj_simple,
        'real': traj_real
    }
    analyze_results(results)

    # 保存结果
    import json
    output = {
        'simple': {
            'final_error': float(np.mean(traj_simple['prediction_errors'][-50:])),
            'final_symbols': int(traj_simple['symbols'][-1]),
            'final_progress': float(traj_simple['learning_progress'][-1])
        },
        'real': {
            'final_error': float(np.mean(traj_real['prediction_errors'][-50:])),
            'final_symbols': int(traj_real['symbols'][-1]),
            'final_progress': float(traj_real['learning_progress'][-1])
        }
    }

    with open('D:/mayAi/AILearning_v0527/mvl/physics_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n结果已保存到: physics_comparison.json")


if __name__ == '__main__':
    main()
