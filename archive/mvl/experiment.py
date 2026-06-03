"""
实验脚本：更深入的探索

运行更长的实验，观察：
1. 发展阶段转换
2. 符号涌现
3. 好奇心驱动 vs 随机探索的长期差异
"""

import sys
import numpy as np
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from environment import SimpleGridWorld, Object, create_simple_world
from environment_rich import create_rich_world
from agent import LearningAgent
from teacher import SimpleTeacher


def run_extended_experiment(num_steps: int = 500, use_rich_env: bool = True):
    """
    运行扩展实验

    观察学习体的长期发展轨迹。
    """
    print("=" * 60)
    print("扩展实验：长期学习轨迹观察")
    print("=" * 60)

    # 创建环境
    if use_rich_env:
        print("\n使用丰富环境（8个物体）")
        env = create_rich_world()
    else:
        print("\n使用简单环境（4个物体）")
        env = create_simple_world()

    # 创建学习体和教师
    agent = LearningAgent(obs_dim=12, action_dim=5)
    teacher = SimpleTeacher()

    # 记录学习轨迹
    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'stages': [],
        'symbols': [],
        'curiosity_rewards': [],
        'learning_progress': []
    }

    print(f"\n开始 {num_steps} 步的学习...")
    print("-" * 60)

    for step in range(num_steps):
        # 获取观测
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

        # 学习
        error = agent.learn_from_experience(obs, action, next_obs)

        # 记录轨迹
        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['stages'].append(agent.development.current_stage)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['curiosity_rewards'].append(
            np.mean(list(agent.curiosity.reward_history)) if agent.curiosity.reward_history else 0
        )
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())

        # 每100步打印报告
        if (step + 1) % 100 == 0:
            print(f"\n步骤 {step + 1}:")
            print(f"  阶段: {agent.development.current_stage}")
            print(f"  符号: {len(agent.grounding.get_grounded_symbols())}")
            print(f"  预测误差: {error:.4f}")
            print(f"  学习进度: {agent.predictive_model.get_learning_progress():.2%}")

            # 显示已学符号
            if agent.grounding.symbol_mappings:
                print(f"  已学符号:")
                for symbol, meaning in agent.grounding.symbol_mappings.items():
                    print(f"    - {symbol}")

        # 环境重置
        if done:
            env.reset()

    # 最终报告
    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print(agent.report())

    # 分析学习轨迹
    analyze_trajectory(trajectory)

    return trajectory


def analyze_trajectory(trajectory: Dict):
    """分析学习轨迹"""
    print("\n" + "=" * 60)
    print("学习轨迹分析")
    print("=" * 60)

    steps = trajectory['steps']
    errors = trajectory['prediction_errors']
    stages = trajectory['stages']
    symbols = trajectory['symbols']

    # 预测误差趋势
    print("\n1. 预测误差趋势:")
    chunks = [errors[i:i+50] for i in range(0, len(errors), 50)]
    for i, chunk in enumerate(chunks):
        avg_error = np.mean(chunk)
        print(f"   步骤 {i*50}-{(i+1)*50}: {avg_error:.4f}")

    # 发展阶段变化
    print("\n2. 发展阶段变化:")
    stage_changes = []
    for i in range(1, len(stages)):
        if stages[i] != stages[i-1]:
            stage_changes.append((i, stages[i-1], stages[i]))

    if stage_changes:
        for step, old_stage, new_stage in stage_changes:
            print(f"   步骤 {step}: {old_stage} → {new_stage}")
    else:
        print(f"   未发生阶段转换（全程: {stages[0]}）")

    # 符号学习曲线
    print("\n3. 符号学习曲线:")
    symbol_counts = list(set(symbols))
    symbol_counts.sort()
    for count in symbol_counts:
        first_occurrence = symbols.index(count)
        print(f"   学到 {count} 个符号时: 步骤 {first_occurrence}")

    # 好奇心奖励趋势
    print("\n4. 好奇心奖励趋势:")
    curiosity = trajectory['curiosity_rewards']
    chunks = [curiosity[i:i+50] for i in range(0, len(curiosity), 50)]
    for i, chunk in enumerate(chunks):
        avg_curiosity = np.mean(chunk)
        print(f"   步骤 {i*50}-{(i+1)*50}: {avg_curiosity:.4f}")


def run_comparison_experiment():
    """
    运行对比实验

    比较不同配置的学习效果。
    """
    print("\n" + "=" * 60)
    print("对比实验：不同配置的学习效果")
    print("=" * 60)

    configs = [
        {'name': '好奇心驱动', 'curiosity': True, 'teacher': True},
        {'name': '无好奇心', 'curiosity': False, 'teacher': True},
        {'name': '无教师', 'curiosity': True, 'teacher': False},
        {'name': '随机基线', 'curiosity': False, 'teacher': False},
    ]

    results = {}

    for config in configs:
        print(f"\n--- {config['name']} ---")

        env = create_simple_world()
        agent = LearningAgent(obs_dim=12, action_dim=5)
        teacher = SimpleTeacher()

        # 禁用好奇心（如果配置要求）
        if not config['curiosity']:
            agent.curiosity.compute_intrinsic_reward = lambda x, y: 0.0

        errors = []
        for step in range(200):
            obs = env.get_observation()

            # 教师教学
            if config['teacher']:
                target = teacher.observe(obs)
                if target:
                    teaching_action = teacher.decide_teaching_action(
                        agent.get_stats(), target
                    )
                    if teaching_action:
                        teacher.execute_teaching(teaching_action, target, agent)

            # 行动
            if config['curiosity']:
                action = agent.act(obs)
            else:
                action = np.random.randint(0, 5)

            next_obs, _, done = env.step(action)
            error = agent.learn_from_experience(obs, action, next_obs)
            errors.append(error)

            if done:
                env.reset()

        results[config['name']] = {
            'final_error': np.mean(errors[-50:]),
            'symbols': len(agent.grounding.get_grounded_symbols()),
            'stage': agent.development.current_stage
        }

    # 打印比较结果
    print("\n" + "=" * 60)
    print("比较结果")
    print("=" * 60)
    print(f"{'配置':<15} {'最终误差':<12} {'符号数':<8} {'发展阶段':<15}")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name:<15} {result['final_error']:<12.4f} {result['symbols']:<8} {result['stage']:<15}")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--extended':
            run_extended_experiment(num_steps=500, use_rich_env=True)
        elif sys.argv[1] == '--comparison':
            run_comparison_experiment()
        elif sys.argv[1] == '--both':
            run_extended_experiment(num_steps=300, use_rich_env=True)
            run_comparison_experiment()
        else:
            print("用法:")
            print("  python experiment.py --extended    # 扩展实验")
            print("  python experiment.py --comparison  # 对比实验")
            print("  python experiment.py --both        # 两个都跑")
    else:
        # 默认运行扩展实验
        run_extended_experiment(num_steps=500, use_rich_env=True)


if __name__ == '__main__':
    main()
