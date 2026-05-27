"""
自适应模型选择对比实验

比较固定模型和自适应模型的学习效果。

实验设计：
1. 固定线性模型
2. 固定神经网络
3. 自适应模型选择

验证：自适应模型是否能根据环境复杂度选择合适的模型
"""

import sys
import numpy as np
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from environment import create_simple_world
from environment_rich import create_rich_world
from agent import LearningAgent
from teacher import SimpleTeacher


def run_adaptive_experiment(num_steps: int = 300):
    """
    运行自适应模型选择实验

    比较固定模型和自适应模型的学习效果。
    """
    print("=" * 60)
    print("自适应模型选择实验")
    print("=" * 60)

    configs = [
        {'name': '固定线性', 'model_type': 'linear'},
        {'name': '固定神经网络', 'model_type': 'neural_network'},
        {'name': '自适应选择', 'model_type': 'adaptive'},
    ]

    results = {}

    for config in configs:
        print(f"\n--- {config['name']} ---")

        # 创建环境
        env = create_rich_world()

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
            'learning_progress': [],
            'model_types': []
        }

        # 运行实验
        for step in range(num_steps):
            obs = env.get_observation()

            # 自适应模型切换
            if config['model_type'] == 'adaptive':
                # 获取环境中的物体
                objects = []
                for obj in env.objects:
                    objects.append({
                        'color': obj.color,
                        'shape': obj.shape,
                        'weight': obj.weight,
                        'x': obj.x,
                        'y': obj.y
                    })
                agent.adapt_model_to_environment(objects, (env.width, env.height))

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
            trajectory['model_types'].append(type(agent.predictive_model).__name__)

            if done:
                env.reset()

        # 计算统计
        final_error = np.mean(trajectory['prediction_errors'][-50:])
        final_symbols = trajectory['symbols'][-1]

        results[config['name']] = {
            'final_error': final_error,
            'final_symbols': final_symbols,
            'trajectory': trajectory
        }

        print(f"  最终预测误差: {final_error:.4f}")
        print(f"  符号数量: {final_symbols}")

        # 显示模型切换历史
        if config['model_type'] == 'adaptive':
            model_changes = []
            for i in range(1, len(trajectory['model_types'])):
                if trajectory['model_types'][i] != trajectory['model_types'][i-1]:
                    model_changes.append((i, trajectory['model_types'][i-1], trajectory['model_types'][i]))
            if model_changes:
                print(f"  模型切换次数: {len(model_changes)}")
                for step, old, new in model_changes:
                    print(f"    步骤 {step}: {old} → {new}")

    return results


def analyze_adaptive_results(results: Dict):
    """分析自适应实验结果"""
    print("\n" + "=" * 60)
    print("自适应实验分析")
    print("=" * 60)

    # 找出最佳模型
    best_model = min(results.items(), key=lambda x: x[1]['final_error'])

    print(f"\n最佳模型: {best_model[0]}")
    print(f"  预测误差: {best_model[1]['final_error']:.4f}")
    print(f"  符号数量: {best_model[1]['final_symbols']}")

    # 计算自适应模型的优势
    if '自适应选择' in results and '固定线性' in results:
        adaptive_error = results['自适应选择']['final_error']
        linear_error = results['固定线性']['final_error']
        if linear_error > 0:
            improvement = (linear_error - adaptive_error) / linear_error
            print(f"\n自适应 vs 线性改进: {improvement:.1%}")

    if '自适应选择' in results and '固定神经网络' in results:
        adaptive_error = results['自适应选择']['final_error']
        nn_error = results['固定神经网络']['final_error']
        if nn_error > 0:
            improvement = (nn_error - adaptive_error) / nn_error
            print(f"自适应 vs 神经网络改进: {improvement:.1%}")


def main():
    """主函数"""
    print("开始自适应模型选择实验...")

    # 运行实验
    results = run_adaptive_experiment(num_steps=300)

    # 分析结果
    analyze_adaptive_results(results)

    # 保存结果
    import json
    output = {}
    for name, data in results.items():
        output[name] = {
            'final_error': float(data['final_error']),
            'final_symbols': int(data['final_symbols'])
        }

    with open('D:/mayAi/AILearning_v0527/mvl/adaptive_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n结果已保存到: adaptive_comparison.json")


if __name__ == '__main__':
    main()
