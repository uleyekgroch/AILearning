"""
主程序：最小可行学习体 (Minimum Viable Learner)

从学习本源出发的AI系统演示。

运行方式：
    cd mvl
    python main.py

你将看到：
1. 一个学习体在2D网格世界中探索
2. 它通过好奇心驱动学习（不是外在奖励）
3. 它经历发展阶段（感知运动→前运算）
4. 它与教师交互，学习符号（不是预训练）
5. 它从感知经验中涌现概念（不是预定义）

这展示了学习的本质：
- 不是"更大的模型"
- 而是"不同的学习范式"
"""

import numpy as np
import time
import sys
from typing import Dict

from environment import SimpleGridWorld, Object, create_simple_world
from agent import LearningAgent
from teacher import SimpleTeacher


def print_header():
    """打印程序头部"""
    print("""
==========================================================
          最小可行学习体 (Minimum Viable Learner)

  从学习本源出发的AI系统原型

  核心理念：
  - 学习信号 = 预测误差（不是标签）
  - 驱动力 = 好奇心（不是损失函数）
  - 数据来源 = 主动探索（不是被动接收）
  - 发展 = 阶段性（不是一次性训练）
==========================================================
    """)


def print_step_info(step: int, env: SimpleGridWorld, agent: LearningAgent,
                     teacher: SimpleTeacher, action: int, prediction_error: float):
    """打印每步信息"""
    action_names = {0: '↑上', 1: '↓下', 2: '←左', 3: '→右', 4: '⊕推'}
    action_name = action_names.get(action, '?')

    stage_info = agent.development.get_stage_info()

    print(f"\n步骤 {step}:")
    print(f"  动作: {action_name}")
    print(f"  位置: ({env.agent_x}, {env.agent_y})")
    print(f"  预测误差: {prediction_error:.4f}")
    print(f"  发展阶段: {stage_info['name']}")
    print(f"  已学符号: {len(agent.grounding.get_grounded_symbols())}")

    # 显示环境
    print(f"\n环境状态:")
    print(env.render())
    print(f"A=Agent, R=红球, B=蓝方块, G=绿三角, Y=黄球")


def print_learning_progress(agent: LearningAgent, step: int):
    """打印学习进度"""
    stats = agent.get_stats()

    print(f"\n{'='*50}")
    print(f"学习进度报告 (步骤 {step})")
    print(f"{'='*50}")
    print(f"发展阶段: {stats['current_stage']}")
    print(f"平均预测误差: {stats['avg_prediction_error']:.4f}")
    print(f"学习进度: {stats['learning_progress']:.2%}")
    print(f"已接地符号: {stats['grounded_symbols']}")
    print(f"发展晋升: {stats['stage_changes']}次")

    # 显示已学符号
    if agent.grounding.symbol_mappings:
        print(f"\n已学习的符号:")
        for symbol, meaning in agent.grounding.symbol_mappings.items():
            print(f"  • {symbol}: 置信度 {meaning['confidence']:.2f}")

    print(f"{'='*50}")


def run_simulation(num_steps: int = 200, verbose: bool = True):
    """
    运行学习模拟

    这是核心演示——
    看一个学习体如何从零开始学习。
    """
    print_header()

    # 创建环境
    print("正在创建环境...")
    env = create_simple_world()

    # 创建学习体
    print("正在初始化学习体...")
    agent = LearningAgent(obs_dim=12, action_dim=5)

    # 创建教师
    print("正在初始化教师...")
    teacher = SimpleTeacher()

    print(f"\n开始学习模拟 ({num_steps}步)...")
    print(f"{'='*50}")

    # 记录学习曲线
    learning_curve = []

    for step in range(num_steps):
        # 1. 获取当前观测
        obs = env.get_observation()

        # 2. 教师观察并决定是否教学
        target = teacher.observe(obs)
        if target:
            teaching_action = teacher.decide_teaching_action(
                agent.get_stats(), target
            )
            if teaching_action:
                episode = teacher.execute_teaching(
                    teaching_action, target, agent
                )
                if verbose and episode.success:
                    print(f"\n[教师] {teaching_action}: {episode.learner_response}")

        # 3. 学习体选择动作
        action = agent.act(obs)

        # 4. 执行动作
        next_obs, reward, done = env.step(action)

        # 5. 学习体从经验中学习
        prediction_error = agent.learn_from_experience(obs, action, next_obs)

        # 记录学习曲线
        learning_curve.append({
            'step': step,
            'prediction_error': prediction_error,
            'stage': agent.development.current_stage,
            'symbols': len(agent.grounding.get_grounded_symbols())
        })

        # 打印信息
        if verbose:
            print_step_info(step, env, agent, teacher, action, prediction_error)

        # 每50步打印学习进度
        if (step + 1) % 50 == 0:
            print_learning_progress(agent, step + 1)

        # 检查是否结束
        if done:
            print(f"\n环境重置（达到最大步数）")
            env.reset()

        # 短暂延迟以便观察
        if verbose:
            time.sleep(0.1)

    # 最终报告
    print("\n" + "="*60)
    print("学习模拟完成！")
    print("="*60)
    print(agent.report())

    # 打印教学统计
    teacher_stats = teacher.get_teaching_stats()
    print(f"\n教师教学统计:")
    print(f"  总教学次数: {teacher_stats['total_episodes']}")
    print(f"  成功率: {teacher_stats['success_rate']:.2%}")
    print(f"  教学类型分布: {teacher_stats['action_distribution']}")

    return learning_curve


def run_experiment():
    """
    运行对比实验

    比较：
    1. 好奇心驱动的学习（本系统）
    2. 随机探索（基线）
    """
    print("\n" + "="*60)
    print("对比实验：好奇心驱动 vs 随机探索")
    print("="*60)

    # 实验1：好奇心驱动
    print("\n[实验1] 好奇心驱动的学习")
    env1 = create_simple_world()
    agent1 = LearningAgent(obs_dim=12, action_dim=5)

    errors1 = []
    for step in range(100):
        obs = env1.get_observation()
        action = agent1.act(obs)
        next_obs, _, _ = env1.step(action)
        error = agent1.learn_from_experience(obs, action, next_obs)
        errors1.append(error)
        if (step + 1) % 20 == 0:
            print(f"  步骤 {step+1}: 平均误差 {np.mean(errors1[-20:]):.4f}")

    # 实验2：随机探索
    print("\n[实验2] 随机探索（基线）")
    env2 = create_simple_world()
    agent2 = LearningAgent(obs_dim=12, action_dim=5)

    errors2 = []
    for step in range(100):
        obs = env2.get_observation()
        action = np.random.randint(0, 5)  # 随机动作
        next_obs, _, _ = env2.step(action)
        error = agent2.learn_from_experience(obs, action, next_obs)
        errors2.append(error)
        if (step + 1) % 20 == 0:
            print(f"  步骤 {step+1}: 平均误差 {np.mean(errors2[-20:]):.4f}")

    # 比较结果
    print("\n" + "-"*40)
    print("实验结果:")
    print(f"  好奇心驱动 - 最终平均误差: {np.mean(errors1[-20:]):.4f}")
    print(f"  随机探索   - 最终平均误差: {np.mean(errors2[-20:]):.4f}")

    improvement = (np.mean(errors2[-20:]) - np.mean(errors1[-20:])) / np.mean(errors2[-20:]) * 100
    print(f"  好奇心驱动比随机探索好: {improvement:.1f}%")

    print("\n结论:")
    if improvement > 0:
        print("  好奇心驱动的学习更高效！")
        print("  这验证了Schmidhuber的理论：")
        print("  好奇心 = 对学习进度的优化")
    else:
        print("  需要更多实验或调整参数")


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == '--experiment':
        run_experiment()
    else:
        # 默认运行演示模式（较少步骤，详细输出）
        run_simulation(num_steps=100, verbose=True)


if __name__ == '__main__':
    main()
