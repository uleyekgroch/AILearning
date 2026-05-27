"""
3D环境对比实验

比较2D和3D环境的学习效果。

实验设计：
1. 2D环境：原始网格世界
2. 3D环境：新的3D物理世界

验证：3D环境是否提供更丰富的学习机会
"""

import sys
import numpy as np
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from environment import create_simple_world
from environment_rich import create_rich_world
from environment_3d import create_simple_3d_world, create_rich_3d_world
from agent import LearningAgent
from agent_3d import LearningAgent3D
from teacher import SimpleTeacher


def run_2d_experiment(num_steps: int = 300):
    """运行2D环境实验"""
    print("\n--- 2D环境实验 ---")
    env = create_rich_world()
    agent = LearningAgent(obs_dim=12, action_dim=5, model_type='neural_network')
    teacher = SimpleTeacher()

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': []
    }

    for step in range(num_steps):
        obs = env.get_observation()

        # 教师教学
        target = teacher.observe(obs)
        if target:
            teaching_action = teacher.decide_teaching_action(
                agent.get_stats(), target
            )
            if teaching_action:
                teacher.execute_teaching(teaching_action, target, agent)

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


def run_3d_experiment(num_steps: int = 300):
    """运行3D环境实验"""
    print("\n--- 3D环境实验 ---")
    env = create_rich_3d_world()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    teacher = SimpleTeacher()

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': []
    }

    for step in range(num_steps):
        obs = env.get_observation()

        # 教师教学（简化版，因为3D环境的观测格式不同）
        # 这里可以扩展教师模块以支持3D环境

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
    print("2D vs 3D环境对比分析")
    print("=" * 60)

    traj_2d = results['2d']
    traj_3d = results['3d']

    # 最终结果
    final_error_2d = np.mean(traj_2d['prediction_errors'][-50:])
    final_error_3d = np.mean(traj_3d['prediction_errors'][-50:])
    final_symbols_2d = traj_2d['symbols'][-1]
    final_symbols_3d = traj_3d['symbols'][-1]
    final_progress_2d = traj_2d['learning_progress'][-1]
    final_progress_3d = traj_3d['learning_progress'][-1]

    print(f"\n1. 最终预测误差:")
    print(f"   2D环境: {final_error_2d:.4f}")
    print(f"   3D环境: {final_error_3d:.4f}")

    print(f"\n2. 符号学习:")
    print(f"   2D环境: {final_symbols_2d} 个符号")
    print(f"   3D环境: {final_symbols_3d} 个符号")

    print(f"\n3. 学习进度:")
    print(f"   2D环境: {final_progress_2d:.2%}")
    print(f"   3D环境: {final_progress_3d:.2%}")

    # 学习曲线分析
    print(f"\n4. 学习曲线:")
    if len(traj_2d['prediction_errors']) > 50:
        early_2d = np.mean(traj_2d['prediction_errors'][:50])
        late_2d = np.mean(traj_2d['prediction_errors'][-50:])
        if early_2d > 0:
            reduction_2d = (early_2d - late_2d) / early_2d
            print(f"   2D环境: 误差减少 {reduction_2d:.1%}")

    if len(traj_3d['prediction_errors']) > 50:
        early_3d = np.mean(traj_3d['prediction_errors'][:50])
        late_3d = np.mean(traj_3d['prediction_errors'][-50:])
        if early_3d > 0:
            reduction_3d = (early_3d - late_3d) / early_3d
            print(f"   3D环境: 误差减少 {reduction_3d:.1%}")


def main():
    """主函数"""
    print("开始2D vs 3D环境对比实验...")

    # 运行实验
    traj_2d = run_2d_experiment(num_steps=300)
    traj_3d = run_3d_experiment(num_steps=300)

    # 分析结果
    results = {
        '2d': traj_2d,
        '3d': traj_3d
    }
    analyze_results(results)

    # 保存结果
    import json
    output = {
        '2d': {
            'final_error': float(np.mean(traj_2d['prediction_errors'][-50:])),
            'final_symbols': int(traj_2d['symbols'][-1]),
            'final_progress': float(traj_2d['learning_progress'][-1])
        },
        '3d': {
            'final_error': float(np.mean(traj_3d['prediction_errors'][-50:])),
            'final_symbols': int(traj_3d['symbols'][-1]),
            'final_progress': float(traj_3d['learning_progress'][-1])
        }
    }

    with open('D:/mayAi/AILearning_v0527/mvl/2d_vs_3d_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n结果已保存到: 2d_vs_3d_comparison.json")


if __name__ == '__main__':
    main()
