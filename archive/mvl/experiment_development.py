"""
发展阶段对比实验

验证完整的 Piaget 八阶段发展轨迹：
sensorimotor → early_preoperational → late_preoperational → early_concrete
→ late_concrete → early_formal → late_formal → adolescent

对比：
A. 有阶段限制（渐进解锁能力）
B. 无阶段限制（一开始就全部解锁）

指标：阶段转换时间、最终阶段、泛化能力
"""

import sys
import numpy as np
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from environment_physics import PhysicsEnvironment, create_rich_physics_world
from physics_rigid import RigidBody, MaterialType
from agent_3d import LearningAgent3D
from teacher import SimpleTeacher


def run_with_stages(num_steps: int = 1000):
    """条件A：有阶段限制（渐进解锁）"""
    print("\n--- 条件A：有阶段限制（渐进解锁）---")
    env = create_rich_physics_world()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    teacher = SimpleTeacher()

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': [],
        'stages': [],
        'stage_changes': [],
        'social_interactions': [],
        'abstract_reasoning': [],
        'hypothesis_confirmed': [],
        'counterfactual_diversity': [],
        'meta_cognition': [],
    }

    stage_transitions = []

    for step in range(num_steps):
        obs = env.get_observation()
        action = agent.act(obs)
        next_obs, _, done = env.step(action)

        error = agent.learn_from_experience(obs, action, next_obs)

        # 教师教学
        target = teacher.observe(obs)
        teach_action = teacher.decide_teaching_action(agent.get_stats(), target)
        if teach_action and target:
            teacher.execute_teaching(teach_action, target, agent)

        # 记录阶段转换
        current_stage = agent.development.current_stage
        if not stage_transitions or stage_transitions[-1][1] != current_stage:
            stage_transitions.append((step, current_stage))

        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())
        trajectory['stages'].append(current_stage)
        trajectory['stage_changes'].append(agent.stats.get('stage_changes', 0))
        trajectory['social_interactions'].append(agent.stats.get('social_interactions', 0))
        trajectory['abstract_reasoning'].append(agent._compute_abstract_reasoning())
        trajectory['hypothesis_confirmed'].append(agent._compute_hypothesis_confirmed())
        trajectory['counterfactual_diversity'].append(agent._compute_counterfactual_diversity())
        trajectory['meta_cognition'].append(agent._compute_meta_cognition())

        if done:
            env.reset()

        if step % 400 == 0:
            print(f"  步骤 {step}: 阶段={current_stage}, "
                  f"误差={error:.4f}, 符号={len(agent.grounding.get_grounded_symbols())}, "
                  f"抽象推理={agent._compute_abstract_reasoning():.3f}")

    return trajectory, agent, stage_transitions


def run_without_stages(num_steps: int = 1000):
    """条件B：无阶段限制（全部解锁）"""
    print("\n--- 条件B：无阶段限制（全部解锁）---")
    env = create_rich_physics_world()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    # 绕过阶段限制，解锁全部动作
    agent._get_available_actions = lambda: list(range(8))
    teacher = SimpleTeacher()

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': [],
        'social_interactions': [],
    }

    for step in range(num_steps):
        obs = env.get_observation()
        action = agent.act(obs)
        next_obs, _, done = env.step(action)

        error = agent.learn_from_experience(obs, action, next_obs)

        # 教师教学
        target = teacher.observe(obs)
        teach_action = teacher.decide_teaching_action(agent.get_stats(), target)
        if teach_action and target:
            teacher.execute_teaching(teach_action, target, agent)

        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())
        trajectory['social_interactions'].append(agent.stats.get('social_interactions', 0))

        if done:
            env.reset()

        if step % 400 == 0:
            print(f"  步骤 {step}: 误差={error:.4f}, "
                  f"符号={len(agent.grounding.get_grounded_symbols())}")

    return trajectory, agent


def test_generalization(agents: Dict) -> Dict:
    """泛化测试"""
    print("\n--- 泛化测试 ---")

    errors = {}
    for name, agent in agents.items():
        test_env = PhysicsEnvironment(10, 10, 10)
        test_env.add_rigid_body(RigidBody(3, 3, 5, radius=0.4, mass=3.0, material=MaterialType.STONE))
        test_env.add_rigid_body(RigidBody(7, 7, 3, radius=0.6, mass=1.0, material=MaterialType.RUBBER))
        test_env.add_rigid_body(RigidBody(5, 5, 7, radius=0.3, mass=0.5, material=MaterialType.WOOD))

        agent_errors = []
        for _ in range(50):
            obs = test_env.get_observation()
            action = agent.act(obs)
            next_obs, _, done = test_env.step(action)

            obs_vec = agent.perceive(obs)
            next_obs_vec = agent.perceive(next_obs)
            predicted = agent.predictive_model.predict(obs_vec, action)
            error = np.mean((predicted - next_obs_vec) ** 2)
            agent_errors.append(error)

            if done:
                test_env.reset()

        errors[name] = np.mean(agent_errors)

    return errors


def analyze_results(trajectory_a, agent_a, stage_transitions,
                    trajectory_b, agent_b, generalization):
    """分析实验结果"""
    print("\n" + "=" * 60)
    print("发展阶段对比实验结果")
    print("=" * 60)

    # 阶段转换时间线
    print(f"\n1. 阶段转换时间线（条件A）:")
    for step, stage in stage_transitions:
        stage_names = {
            'sensorimotor': '感知运动',
            'early_preoperational': '前运算早期',
            'late_preoperational': '前运算晚期',
            'early_concrete': '具体运算早期',
            'late_concrete': '具体运算晚期',
            'early_formal': '形式运算早期',
            'late_formal': '形式运算晚期',
            'adolescent': '青少年期',
        }
        print(f"   步骤 {step:>4d}: → {stage_names.get(stage, stage)}")

    # 最终阶段
    print(f"\n2. 最终发展阶段:")
    print(f"   条件A（有阶段限制）: {trajectory_a['stages'][-1]}")
    print(f"   条件B（无阶段限制）: 始终全部解锁")

    # 最终预测误差
    final_error_a = np.mean(trajectory_a['prediction_errors'][-50:])
    final_error_b = np.mean(trajectory_b['prediction_errors'][-50:])
    print(f"\n3. 最终预测误差（最后50步均值）:")
    print(f"   条件A: {final_error_a:.4f}")
    print(f"   条件B: {final_error_b:.4f}")

    # 符号学习
    print(f"\n4. 符号学习数量:")
    print(f"   条件A: {trajectory_a['symbols'][-1]} 个")
    print(f"   条件B: {trajectory_b['symbols'][-1]} 个")

    # 形式运算能力指标（条件A）
    print(f"\n5. 形式运算能力指标（条件A最终值）:")
    print(f"   抽象推理分数: {trajectory_a['abstract_reasoning'][-1]:.4f}")
    print(f"   假设验证正确率: {trajectory_a['hypothesis_confirmed'][-1]:.4f}")
    print(f"   反事实多样性: {trajectory_a['counterfactual_diversity'][-1]:.4f}")
    print(f"   元认知分数: {trajectory_a['meta_cognition'][-1]:.4f}")

    # 泛化测试
    print(f"\n6. 泛化测试（新场景预测误差）:")
    for name, error in generalization.items():
        label = '条件A（有阶段限制）' if name == 'a' else '条件B（无阶段限制）'
        print(f"   {label}: {error:.4f}")

    # 泛化能力对比
    if 'a' in generalization and 'b' in generalization:
        improvement = (generalization['b'] - generalization['a']) / generalization['b'] * 100
        print(f"\n   → 有阶段限制比无阶段限制泛化误差{'降低' if improvement > 0 else '增加'} "
              f"{abs(improvement):.1f}%")


def main():
    """主函数"""
    print("开始发展阶段对比实验...")
    print("目标：验证完整的 Piaget 八阶段发展轨迹\n")

    num_steps = 2000

    # 运行两个条件
    traj_a, agent_a, transitions = run_with_stages(num_steps)
    traj_b, agent_b = run_without_stages(num_steps)

    # 泛化测试
    generalization = test_generalization({
        'a': agent_a,
        'b': agent_b,
    })

    # 分析结果
    analyze_results(traj_a, agent_a, transitions, traj_b, agent_b, generalization)

    # 保存结果
    import json
    output = {
        'condition_a': {
            'stage_transitions': [(int(s), st) for s, st in transitions],
            'final_stage': traj_a['stages'][-1],
            'final_error': float(np.mean(traj_a['prediction_errors'][-50:])),
            'final_symbols': int(traj_a['symbols'][-1]),
            'final_abstract_reasoning': float(traj_a['abstract_reasoning'][-1]),
            'final_hypothesis_confirmed': float(traj_a['hypothesis_confirmed'][-1]),
            'final_counterfactual_diversity': float(traj_a['counterfactual_diversity'][-1]),
            'final_meta_cognition': float(traj_a['meta_cognition'][-1]),
        },
        'condition_b': {
            'final_error': float(np.mean(traj_b['prediction_errors'][-50:])),
            'final_symbols': int(traj_b['symbols'][-1]),
        },
        'generalization': generalization,
    }

    with open('D:/mayAi/AILearning_v0527/mvl/development_comparison.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到: development_comparison.json")


if __name__ == '__main__':
    main()
