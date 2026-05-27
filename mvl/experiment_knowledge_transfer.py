"""
知识迁移对比实验

比较有无知识迁移的模型切换效果。

实验设计：
1. 无知识迁移：直接切换模型
2. 有知识迁移：渐进切换 + 知识注入

验证：知识迁移是否减少遗忘
"""

import sys
import numpy as np
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from environment import create_simple_world
from environment_rich import create_rich_world
from agent import LearningAgent
from teacher import SimpleTeacher
from knowledge_transfer import KnowledgeExtractor, KnowledgeInjector
from predictive_nn import NeuralNetworkPredictor


def run_experiment_without_transfer(num_steps: int = 300):
    """
    运行无知识迁移实验

    直接切换模型，不保留知识。
    """
    print("\n--- 实验1：无知识迁移 ---")
    env = create_rich_world()
    agent = LearningAgent(obs_dim=12, action_dim=5, model_type='linear')
    teacher = SimpleTeacher()

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'model_types': []
    }

    switch_point = None

    for step in range(num_steps):
        obs = env.get_observation()

        # 在步骤100时切换到神经网络（无知识迁移）
        if step == 100:
            print(f"\n步骤 {step}: 直接切换到神经网络（无知识迁移）")
            agent.predictive_model = NeuralNetworkPredictor(agent.obs_dim, agent.action_dim)
            switch_point = step

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

    return trajectory, switch_point


def run_experiment_with_transfer(num_steps: int = 300):
    """
    运行有知识迁移实验

    渐进切换 + 知识注入。
    """
    print("\n--- 实验2：有知识迁移 ---")
    env = create_rich_world()
    agent = LearningAgent(obs_dim=12, action_dim=5, model_type='linear')
    teacher = SimpleTeacher()

    extractor = KnowledgeExtractor()
    injector = KnowledgeInjector()

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'model_types': []
    }

    switch_point = None
    transition_phase = False
    transition_step = 0
    transition_steps = 50
    old_model = None
    new_model = None
    old_knowledge = None

    for step in range(num_steps):
        obs = env.get_observation()

        # 在步骤100时开始渐进切换
        if step == 100:
            print(f"\n步骤 {step}: 开始渐进切换到神经网络（有知识迁移）")
            switch_point = step

            # 提取旧模型知识
            old_knowledge = {
                'predictive_model': extractor.extract_from_predictive_model(agent.predictive_model),
                'grounding': extractor.extract_from_grounding(agent.grounding),
                'curiosity': extractor.extract_from_curiosity(agent.curiosity)
            }

            # 创建新模型
            from predictive_nn import NeuralNetworkPredictor
            new_model = NeuralNetworkPredictor(agent.obs_dim, agent.action_dim)

            # 注入知识
            injector.inject_to_predictive_model(new_model, old_knowledge['predictive_model'], injection_rate=0.5)

            # 保存旧模型
            old_model = agent.predictive_model

            # 开始过渡
            transition_phase = True
            transition_step = 0

        # 过渡阶段：混合预测
        if transition_phase and transition_step < transition_steps:
            # 计算混合权重
            weight = transition_step / transition_steps

            # 混合预测（在learn_from_experience中使用）
            old_pred = old_model.predict(agent.perceive(obs), agent.act(obs))
            new_pred = new_model.predict(agent.perceive(obs), agent.act(obs))

            # 推进过渡
            transition_step += 1

            # 如果过渡完成，切换到新模型
            if transition_step >= transition_steps:
                print(f"\n步骤 {step}: 过渡完成，切换到新模型")
                agent.predictive_model = new_model
                transition_phase = False

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

    return trajectory, switch_point


def analyze_results(results: Dict):
    """分析实验结果"""
    print("\n" + "=" * 60)
    print("知识迁移实验分析")
    print("=" * 60)

    traj_no = results['no_transfer']['trajectory']
    traj_with = results['with_transfer']['trajectory']
    switch_no = results['no_transfer']['switch_point']
    switch_with = results['with_transfer']['switch_point']

    print(f"\n1. 模型切换点:")
    print(f"   无知识迁移: 步骤 {switch_no}")
    print(f"   有知识迁移: 步骤 {switch_with}")

    # 计算切换前后的误差
    if switch_no and switch_no > 10:
        error_before_no = np.mean(traj_no['prediction_errors'][max(0, switch_no-10):switch_no])
        error_after_no = np.mean(traj_no['prediction_errors'][switch_no:min(len(traj_no['prediction_errors']), switch_no+10)])
        print(f"\n2. 无知识迁移 - 切换前后误差:")
        print(f"   切换前: {error_before_no:.4f}")
        print(f"   切换后: {error_after_no:.4f}")
        if error_before_no > 0:
            print(f"   误差增加: {(error_after_no - error_before_no) / error_before_no:.1%}")

    if switch_with and switch_with > 10:
        error_before_with = np.mean(traj_with['prediction_errors'][max(0, switch_with-10):switch_with])
        error_after_with = np.mean(traj_with['prediction_errors'][switch_with:min(len(traj_with['prediction_errors']), switch_with+10)])
        print(f"\n3. 有知识迁移 - 切换前后误差:")
        print(f"   切换前: {error_before_with:.4f}")
        print(f"   切换后: {error_after_with:.4f}")
        if error_before_with > 0:
            print(f"   误差增加: {(error_after_with - error_before_with) / error_before_with:.1%}")

    # 最终结果
    final_error_no = np.mean(traj_no['prediction_errors'][-50:])
    final_error_with = np.mean(traj_with['prediction_errors'][-50:])
    final_symbols_no = traj_no['symbols'][-1]
    final_symbols_with = traj_with['symbols'][-1]

    print(f"\n4. 最终结果:")
    print(f"   无知识迁移 - 误差: {final_error_no:.4f}, 符号: {final_symbols_no}")
    print(f"   有知识迁移 - 误差: {final_error_with:.4f}, 符号: {final_symbols_with}")

    if final_error_no > 0:
        improvement = (final_error_no - final_error_with) / final_error_no
        print(f"\n5. 知识迁移改进: {improvement:.1%}")


def main():
    """主函数"""
    print("开始知识迁移实验...")

    # 运行实验
    traj_no, switch_no = run_experiment_without_transfer(num_steps=300)
    traj_with, switch_with = run_experiment_with_transfer(num_steps=300)

    # 分析结果
    results = {
        'no_transfer': {'trajectory': traj_no, 'switch_point': switch_no},
        'with_transfer': {'trajectory': traj_with, 'switch_point': switch_with}
    }
    analyze_results(results)

    # 保存结果
    import json
    output = {
        'no_transfer': {
            'final_error': float(np.mean(traj_no['prediction_errors'][-50:])),
            'final_symbols': int(traj_no['symbols'][-1]),
            'switch_point': switch_no
        },
        'with_transfer': {
            'final_error': float(np.mean(traj_with['prediction_errors'][-50:])),
            'final_symbols': int(traj_with['symbols'][-1]),
            'switch_point': switch_with
        }
    }

    with open('D:/mayAi/AILearning_v0527/mvl/knowledge_transfer_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n结果已保存到: knowledge_transfer_comparison.json")


if __name__ == '__main__':
    main()
