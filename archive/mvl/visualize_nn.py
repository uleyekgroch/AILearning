"""
神经网络对比实验可视化

生成图表展示线性模型和神经网络模型的学习差异。
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


def collect_trajectory(model_type: str, num_steps: int = 300, use_rich_env: bool = True) -> Dict:
    """收集学习轨迹"""
    if use_rich_env:
        env = create_rich_world()
    else:
        env = create_simple_world()

    agent = LearningAgent(obs_dim=12, action_dim=5, model_type=model_type)
    teacher = SimpleTeacher()

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': [],
        'curiosity_rewards': []
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
        trajectory['curiosity_rewards'].append(
            np.mean(list(agent.curiosity.reward_history)) if agent.curiosity.reward_history else 0
        )

        if done:
            env.reset()

    return trajectory


def plot_model_comparison(save_path: str = 'model_comparison.png'):
    """绘制模型对比图"""
    print("收集线性模型轨迹...")
    linear_traj = collect_trajectory('linear', num_steps=300)

    print("收集神经网络轨迹...")
    nn_traj = collect_trajectory('neural_network', num_steps=300)

    print("收集自适应NN轨迹...")
    adaptive_traj = collect_trajectory('adaptive_nn', num_steps=300)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('线性模型 vs 神经网络 对比', fontsize=16)

    window = 20

    # 1. 预测误差
    ax1 = axes[0, 0]
    for traj, name, color in [
        (linear_traj, '线性模型', 'blue'),
        (nn_traj, '神经网络', 'red'),
        (adaptive_traj, '自适应NN', 'green')
    ]:
        errors = traj['prediction_errors']
        if len(errors) > window:
            moving_avg = np.convolve(errors, np.ones(window)/window, mode='valid')
            ax1.plot(traj['steps'][window-1:], moving_avg, label=name, color=color, linewidth=2)

    ax1.set_xlabel('步骤')
    ax1.set_ylabel('预测误差')
    ax1.set_title('预测误差变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 符号学习
    ax2 = axes[0, 1]
    for traj, name, color in [
        (linear_traj, '线性模型', 'blue'),
        (nn_traj, '神经网络', 'red'),
        (adaptive_traj, '自适应NN', 'green')
    ]:
        ax2.plot(traj['steps'], traj['symbols'], label=name, color=color, linewidth=2)

    ax2.set_xlabel('步骤')
    ax2.set_ylabel('已学符号数')
    ax2.set_title('符号学习曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 学习进度
    ax3 = axes[1, 0]
    for traj, name, color in [
        (linear_traj, '线性模型', 'blue'),
        (nn_traj, '神经网络', 'red'),
        (adaptive_traj, '自适应NN', 'green')
    ]:
        progress = traj['learning_progress']
        if len(progress) > window:
            moving_avg = np.convolve(progress, np.ones(window)/window, mode='valid')
            ax3.plot(traj['steps'][window-1:], moving_avg, label=name, color=color, linewidth=2)

    ax3.set_xlabel('步骤')
    ax3.set_ylabel('学习进度')
    ax3.set_title('学习进度变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 好奇心奖励
    ax4 = axes[1, 1]
    for traj, name, color in [
        (linear_traj, '线性模型', 'blue'),
        (nn_traj, '神经网络', 'red'),
        (adaptive_traj, '自适应NN', 'green')
    ]:
        curiosity = traj['curiosity_rewards']
        if len(curiosity) > window:
            moving_avg = np.convolve(curiosity, np.ones(window)/window, mode='valid')
            ax4.plot(traj['steps'][window-1:], moving_avg, label=name, color=color, linewidth=2)

    ax4.set_xlabel('步骤')
    ax4.set_ylabel('好奇心奖励')
    ax4.set_title('好奇心变化')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")

    return fig


def main():
    """主函数"""
    print("=" * 60)
    print("神经网络对比实验可视化")
    print("=" * 60)

    plot_model_comparison('D:/mayAi/AILearning_v0527/mvl/model_comparison.png')

    print("\n可视化完成！")


if __name__ == '__main__':
    main()
