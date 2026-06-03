"""
自适应模型选择可视化

展示模型切换过程和学习曲线。
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

sys.stdout.reconfigure(encoding='utf-8')

from environment import create_simple_world
from environment_rich import create_rich_world
from agent import LearningAgent
from teacher import SimpleTeacher


def collect_adaptive_trajectory(num_steps: int = 300) -> Dict:
    """收集自适应模型的学习轨迹"""
    env = create_rich_world()
    agent = LearningAgent(obs_dim=12, action_dim=5, model_type='adaptive')
    teacher = SimpleTeacher()

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'model_types': [],
        'complexity': []
    }

    for step in range(num_steps):
        obs = env.get_observation()

        # 获取环境复杂度
        objects = []
        for obj in env.objects:
            objects.append({
                'color': obj.color,
                'shape': obj.shape,
                'weight': obj.weight,
                'x': obj.x,
                'y': obj.y
            })

        # 估计复杂度
        complexity = agent.model_selector.estimator.estimate(objects, (env.width, env.height))
        trajectory['complexity'].append(complexity)

        # 自适应模型切换
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
        trajectory['model_types'].append(type(agent.predictive_model).__name__)

        if done:
            env.reset()

    return trajectory


def plot_adaptive_trajectory(trajectory: Dict, save_path: str = 'adaptive_trajectory.png'):
    """绘制自适应模型的学习轨迹"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('自适应模型选择学习轨迹', fontsize=16)

    steps = trajectory['steps']
    window = 20

    # 1. 预测误差
    ax1 = axes[0, 0]
    errors = trajectory['prediction_errors']
    if len(errors) > window:
        moving_avg = np.convolve(errors, np.ones(window)/window, mode='valid')
        ax1.plot(steps[window-1:], moving_avg, color='blue', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Prediction Error')
    ax1.set_title('Prediction Error')
    ax1.grid(True, alpha=0.3)

    # 2. 符号学习
    ax2 = axes[0, 1]
    ax2.plot(steps, trajectory['symbols'], color='green', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Symbols Learned')
    ax2.set_title('Symbol Learning')
    ax2.grid(True, alpha=0.3)

    # 3. 环境复杂度
    ax3 = axes[1, 0]
    ax3.plot(steps, trajectory['complexity'], color='red', linewidth=2)
    ax3.axhline(y=0.3, color='orange', linestyle='--', label='Low Threshold')
    ax3.axhline(y=0.7, color='purple', linestyle='--', label='High Threshold')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Complexity')
    ax3.set_title('Environment Complexity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 模型类型
    ax4 = axes[1, 1]
    model_map = {'PredictiveModel': 0, 'NeuralNetworkPredictor': 1, 'AdaptiveLearningRatePredictor': 2}
    model_values = [model_map.get(m, 0) for m in trajectory['model_types']]
    ax4.plot(steps, model_values, color='purple', linewidth=2)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Model Type')
    ax4.set_title('Model Selection')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Linear', 'Neural Net', 'Adaptive NN'])
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")

    return fig


def main():
    """主函数"""
    print("=" * 60)
    print("自适应模型选择可视化")
    print("=" * 60)

    # 收集轨迹
    print("\n收集自适应模型学习轨迹...")
    trajectory = collect_adaptive_trajectory(num_steps=300)

    # 绘制图表
    print("\n绘制学习轨迹...")
    plot_adaptive_trajectory(trajectory, 'D:/mayAi/AILearning_v0527/mvl/adaptive_trajectory.png')

    print("\n可视化完成！")


if __name__ == '__main__':
    main()
