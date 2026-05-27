"""
神经网络 vs 线性模型对比实验

比较两种预测模型的学习效果：
1. 线性模型：原始版本
2. 神经网络：升级版本

验证：神经网络是否能更好地学习世界动力学
"""

import sys
import numpy as np
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from environment import create_simple_world
from environment_rich import create_rich_world
from agent import LearningAgent
from teacher import SimpleTeacher


def run_model_comparison(num_steps: int = 300, use_rich_env: bool = True):
    """
    运行模型对比实验

    比较线性模型和神经网络模型的学习效果。
    """
    print("=" * 60)
    print("模型对比实验：线性 vs 神经网络")
    print("=" * 60)

    configs = [
        {'name': '线性模型', 'model_type': 'linear'},
        {'name': '神经网络', 'model_type': 'neural_network'},
        {'name': '自适应NN', 'model_type': 'adaptive_nn'},
    ]

    results = {}

    for config in configs:
        print(f"\n--- {config['name']} ---")

        # 创建环境
        if use_rich_env:
            env = create_rich_world()
        else:
            env = create_simple_world()

        # 创建学习体
        agent = LearningAgent(
            obs_dim=12,
            action_dim=5,
            model_type=config['model_type']
        )
        teacher = SimpleTeacher()

        # 记录轨迹
        trajectory = {
            'steps': [],
            'prediction_errors': [],
            'symbols': [],
            'learning_progress': []
        }

        # 运行实验
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

        # 计算统计
        final_error = np.mean(trajectory['prediction_errors'][-50:])
        final_symbols = trajectory['symbols'][-1]
        final_progress = trajectory['learning_progress'][-1]

        results[config['name']] = {
            'final_error': final_error,
            'final_symbols': final_symbols,
            'final_progress': final_progress,
            'trajectory': trajectory
        }

        print(f"  最终预测误差: {final_error:.4f}")
        print(f"  符号数量: {final_symbols}")
        print(f"  学习进度: {final_progress:.2%}")

    return results


def analyze_results(results: Dict):
    """分析对比结果"""
    print("\n" + "=" * 60)
    print("对比分析")
    print("=" * 60)

    # 找出最佳模型
    best_model = min(results.items(), key=lambda x: x[1]['final_error'])
    worst_model = max(results.items(), key=lambda x: x[1]['final_error'])

    print(f"\n最佳模型: {best_model[0]}")
    print(f"  预测误差: {best_model[1]['final_error']:.4f}")
    print(f"  符号数量: {best_model[1]['final_symbols']}")

    print(f"\n最差模型: {worst_model[0]}")
    print(f"  预测误差: {worst_model[1]['final_error']:.4f}")
    print(f"  符号数量: {worst_model[1]['final_symbols']}")

    # 计算改进
    if worst_model[1]['final_error'] > 0:
        improvement = (worst_model[1]['final_error'] - best_model[1]['final_error']) / worst_model[1]['final_error']
        print(f"\n改进幅度: {improvement:.1%}")

    # 学习曲线分析
    print("\n学习曲线分析:")
    for name, data in results.items():
        errors = data['trajectory']['prediction_errors']
        # 计算误差减少率
        if len(errors) > 50:
            early_error = np.mean(errors[:50])
            late_error = np.mean(errors[-50:])
            if early_error > 0:
                reduction = (early_error - late_error) / early_error
                print(f"  {name}: 误差减少 {reduction:.1%}")


def main():
    """主函数"""
    print("开始神经网络 vs 线性模型对比实验...")

    # 运行对比实验
    results = run_model_comparison(num_steps=300, use_rich_env=True)

    # 分析结果
    analyze_results(results)

    # 保存结果
    import json
    output = {}
    for name, data in results.items():
        output[name] = {
            'final_error': float(data['final_error']),
            'final_symbols': int(data['final_symbols']),
            'final_progress': float(data['final_progress'])
        }

    with open('D:/mayAi/AILearning_v0527/mvl/model_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n结果已保存到: model_comparison.json")


if __name__ == '__main__':
    main()
