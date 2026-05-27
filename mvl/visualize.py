"""
可视化脚本：展示学习过程

生成学习轨迹的可视化图表。
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from environment import create_simple_world
from environment_rich import create_rich_world
from agent import LearningAgent
from teacher import SimpleTeacher


def collect_trajectory(num_steps: int = 300, use_rich_env: bool = True) -> Dict:
    """收集学习轨迹数据"""
    if use_rich_env:
        env = create_rich_world()
    else:
        env = create_simple_world()

    agent = LearningAgent(obs_dim=12, action_dim=5)
    teacher = SimpleTeacher()

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'stages': [],
        'symbols': [],
        'curiosity_rewards': [],
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
        trajectory['stages'].append(agent.development.current_stage)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['curiosity_rewards'].append(
            np.mean(list(agent.curiosity.reward_history)) if agent.curiosity.reward_history else 0
        )
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())

        if done:
            env.reset()

    return trajectory


def plot_trajectory(trajectory: Dict, save_path: str = 'learning_trajectory.png'):
    """绘制学习轨迹"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('学习体发展轨迹', fontsize=16)

    steps = trajectory['steps']

    # 1. 预测误差
    ax1 = axes[0, 0]
    ax1.plot(steps, trajectory['prediction_errors'], alpha=0.3, color='blue')
    # 移动平均
    window = 20
    if len(trajectory['prediction_errors']) > window:
        moving_avg = np.convolve(trajectory['prediction_errors'],
                                  np.ones(window)/window, mode='valid')
        ax1.plot(steps[window-1:], moving_avg, color='blue', linewidth=2)
    ax1.set_xlabel('步骤')
    ax1.set_ylabel('预测误差')
    ax1.set_title('预测误差变化')
    ax1.grid(True, alpha=0.3)

    # 2. 符号学习
    ax2 = axes[0, 1]
    ax2.plot(steps, trajectory['symbols'], color='green', linewidth=2)
    ax2.set_xlabel('步骤')
    ax2.set_ylabel('已学符号数')
    ax2.set_title('符号学习曲线')
    ax2.grid(True, alpha=0.3)

    # 3. 好奇心奖励
    ax3 = axes[1, 0]
    ax3.plot(steps, trajectory['curiosity_rewards'], alpha=0.3, color='red')
    if len(trajectory['curiosity_rewards']) > window:
        moving_avg = np.convolve(trajectory['curiosity_rewards'],
                                  np.ones(window)/window, mode='valid')
        ax3.plot(steps[window-1:], moving_avg, color='red', linewidth=2)
    ax3.set_xlabel('步骤')
    ax3.set_ylabel('好奇心奖励')
    ax3.set_title('好奇心变化')
    ax3.grid(True, alpha=0.3)

    # 4. 发展阶段
    ax4 = axes[1, 1]
    stage_map = {'sensorimotor': 0, 'pre_operational': 1, 'concrete_operational': 2}
    stage_values = [stage_map.get(s, 0) for s in trajectory['stages']]
    ax4.plot(steps, stage_values, color='purple', linewidth=2)
    ax4.set_xlabel('步骤')
    ax4.set_ylabel('发展阶段')
    ax4.set_title('发展阶段变化')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['感知运动', '前运算', '具体运算'])
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")

    return fig


def plot_comparison(save_path: str = 'comparison.png'):
    """绘制对比实验结果"""
    configs = [
        {'name': '好奇心驱动', 'curiosity': True, 'teacher': True},
        {'name': '无好奇心', 'curiosity': False, 'teacher': True},
        {'name': '无教师', 'curiosity': True, 'teacher': False},
        {'name': '随机基线', 'curiosity': False, 'teacher': False},
    ]

    results = []
    for config in configs:
        env = create_simple_world()
        agent = LearningAgent(obs_dim=12, action_dim=5)
        teacher = SimpleTeacher()

        if not config['curiosity']:
            agent.curiosity.compute_intrinsic_reward = lambda x, y: 0.0

        errors = []
        for step in range(200):
            obs = env.get_observation()
            if config['teacher']:
                target = teacher.observe(obs)
                if target:
                    teaching_action = teacher.decide_teaching_action(
                        agent.get_stats(), target
                    )
                    if teaching_action:
                        teacher.execute_teaching(teaching_action, target, agent)

            if config['curiosity']:
                action = agent.act(obs)
            else:
                action = np.random.randint(0, 5)

            next_obs, _, done = env.step(action)
            error = agent.learn_from_experience(obs, action, next_obs)
            errors.append(error)

            if done:
                env.reset()

        results.append({
            'name': config['name'],
            'errors': errors,
            'final_error': np.mean(errors[-50:]),
            'symbols': len(agent.grounding.get_grounded_symbols())
        })

    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('不同配置的学习效果对比', fontsize=14)

    # 学习曲线对比
    colors = ['blue', 'orange', 'green', 'red']
    for i, result in enumerate(results):
        window = 20
        if len(result['errors']) > window:
            moving_avg = np.convolve(result['errors'],
                                      np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(result['errors'])),
                    moving_avg, label=result['name'], color=colors[i], linewidth=2)

    ax1.set_xlabel('步骤')
    ax1.set_ylabel('预测误差')
    ax1.set_title('学习曲线对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 最终结果对比
    names = [r['name'] for r in results]
    final_errors = [r['final_error'] for r in results]
    symbols = [r['symbols'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax2.bar(x - width/2, final_errors, width, label='最终误差', color='skyblue')
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, symbols, width, label='符号数', color='lightgreen')

    ax2.set_xlabel('配置')
    ax2.set_ylabel('预测误差')
    ax2_twin.set_ylabel('符号数')
    ax2.set_title('最终结果对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")

    return fig


def main():
    """主函数"""
    print("=" * 60)
    print("学习体可视化")
    print("=" * 60)

    # 收集轨迹
    print("\n正在收集学习轨迹（300步）...")
    trajectory = collect_trajectory(num_steps=300, use_rich_env=True)

    # 绘制轨迹
    print("\n正在绘制学习轨迹...")
    plot_trajectory(trajectory, 'D:/mayAi/AILearning_v0527/mvl/learning_trajectory.png')

    # 绘制对比
    print("\n正在绘制对比实验...")
    plot_comparison('D:/mayAi/AILearning_v0527/mvl/comparison.png')

    print("\n可视化完成！")


if __name__ == '__main__':
    main()
