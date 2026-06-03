"""
社会交互对比实验

四条件对比：
A. 无社会交互（纯好奇心）—— 基线
B. 有教师（命名+示范，静态脚手架）
C. 有教师 + 脚手架渐退
D. 有教师 + 脚手架渐退 + 模仿学习

验证：社会交互对学习的价值
"""

import sys
import numpy as np
from typing import Dict

sys.stdout.reconfigure(encoding='utf-8')

from environment_physics import PhysicsEnvironment, create_rich_physics_world
from physics_rigid import RigidBody, MaterialType
from agent_3d import LearningAgent3D
from teacher import SimpleTeacher


def run_condition_a(num_steps: int = 500):
    """条件A：无社会交互（纯好奇心）"""
    print("\n--- 条件A：无社会交互（纯好奇心）---")
    env = create_rich_physics_world()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': [],
        'stage_changes': [],
        'social_interactions': [],
    }

    for step in range(num_steps):
        obs = env.get_observation()
        action = agent.act(obs)
        next_obs, _, done = env.step(action)

        error = agent.learn_from_experience(obs, action, next_obs)

        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())
        trajectory['stage_changes'].append(agent.stats.get('stage_changes', 0))
        trajectory['social_interactions'].append(agent.stats.get('social_interactions', 0))

        if done:
            env.reset()

    return trajectory, agent


def run_condition_b(num_steps: int = 500):
    """条件B：有教师（静态脚手架）"""
    print("\n--- 条件B：有教师（静态脚手架）---")
    env = create_rich_physics_world()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    teacher = SimpleTeacher()

    # 禁用脚手架渐退（保持固定概率）
    teacher.update_scaffold_level = lambda x: None  # 不更新

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': [],
        'stage_changes': [],
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
        trajectory['stage_changes'].append(agent.stats.get('stage_changes', 0))
        trajectory['social_interactions'].append(agent.stats.get('social_interactions', 0))

        if done:
            env.reset()

    return trajectory, agent


def run_condition_c(num_steps: int = 500):
    """条件C：有教师 + 脚手架渐退"""
    print("\n--- 条件C：有教师 + 脚手架渐退 ---")
    env = create_rich_physics_world()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    teacher = SimpleTeacher()

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': [],
        'stage_changes': [],
        'social_interactions': [],
        'scaffold_levels': [],
    }

    for step in range(num_steps):
        obs = env.get_observation()
        action = agent.act(obs)
        next_obs, _, done = env.step(action)

        error = agent.learn_from_experience(obs, action, next_obs)

        # 教师教学（带脚手架渐退）
        target = teacher.observe(obs)
        teach_action = teacher.decide_teaching_action(agent.get_stats(), target)
        if teach_action and target:
            teacher.execute_teaching(teach_action, target, agent)

        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())
        trajectory['stage_changes'].append(agent.stats.get('stage_changes', 0))
        trajectory['social_interactions'].append(agent.stats.get('social_interactions', 0))
        trajectory['scaffold_levels'].append(teacher.scaffold_level)

        if done:
            env.reset()

    return trajectory, agent


def run_condition_d(num_steps: int = 500):
    """条件D：有教师 + 脚手架渐退 + 模仿学习"""
    print("\n--- 条件D：有教师 + 脚手架渐退 + 模仿学习 ---")
    env = create_rich_physics_world()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    # 解锁推动/拉动动作
    agent._get_available_actions = lambda: list(range(8))
    teacher = SimpleTeacher()

    trajectory = {
        'steps': [],
        'prediction_errors': [],
        'symbols': [],
        'learning_progress': [],
        'stage_changes': [],
        'social_interactions': [],
        'scaffold_levels': [],
    }

    for step in range(num_steps):
        obs = env.get_observation()
        action = agent.act(obs)
        next_obs, _, done = env.step(action)

        error = agent.learn_from_experience(obs, action, next_obs)

        # 教师教学（带脚手架渐退 + 示范）
        target = teacher.observe(obs)
        teach_action = teacher.decide_teaching_action(agent.get_stats(), target)
        if teach_action and target:
            teacher.execute_teaching(teach_action, target, agent)

        trajectory['steps'].append(step)
        trajectory['prediction_errors'].append(error)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())
        trajectory['stage_changes'].append(agent.stats.get('stage_changes', 0))
        trajectory['social_interactions'].append(agent.stats.get('social_interactions', 0))
        trajectory['scaffold_levels'].append(teacher.scaffold_level)

        if done:
            env.reset()

    return trajectory, agent


def test_generalization(agents: Dict) -> Dict:
    """泛化测试"""
    print("\n--- 泛化测试 ---")

    errors = {}
    for name, agent in agents.items():
        test_env = PhysicsEnvironment(10, 10, 10)
        test_env.add_rigid_body(RigidBody(3, 3, 5, radius=0.4, mass=3.0, material=MaterialType.STONE))
        test_env.add_rigid_body(RigidBody(7, 7, 3, radius=0.6, mass=1.0, material=MaterialType.RUBBER))

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


def analyze_results(results: Dict):
    """分析实验结果"""
    print("\n" + "=" * 60)
    print("社会交互对比实验结果")
    print("=" * 60)

    names = {
        'a': '条件A（无社会交互）',
        'b': '条件B（静态脚手架）',
        'c': '条件C（脚手架渐退）',
        'd': '条件D（渐退+模仿学习）',
    }

    # 最终预测误差
    print(f"\n1. 最终预测误差（最后50步均值）:")
    for key, name in names.items():
        traj = results[key]
        final_error = np.mean(traj['prediction_errors'][-50:])
        print(f"   {name}: {final_error:.4f}")

    # 符号学习
    print(f"\n2. 符号学习数量:")
    for key, name in names.items():
        traj = results[key]
        print(f"   {name}: {traj['symbols'][-1]} 个")

    # 社会交互次数
    print(f"\n3. 社会交互次数:")
    for key, name in names.items():
        traj = results[key]
        print(f"   {name}: {traj['social_interactions'][-1]} 次")

    # 发展晋升
    print(f"\n4. 发展晋升次数:")
    for key, name in names.items():
        traj = results[key]
        print(f"   {name}: {traj['stage_changes'][-1]} 次")

    # 脚手架级别（条件C和D）
    if 'scaffold_levels' in results.get('c', {}):
        print(f"\n5. 脚手架级别（最终值）:")
        print(f"   条件C: {results['c']['scaffold_levels'][-1]:.2f}")
        print(f"   条件D: {results['d']['scaffold_levels'][-1]:.2f}")

    # 泛化测试
    if 'generalization' in results:
        print(f"\n6. 泛化测试（新场景预测误差）:")
        for name, error in results['generalization'].items():
            print(f"   {names.get(name, name)}: {error:.4f}")


def main():
    """主函数"""
    print("开始社会交互对比实验...")

    num_steps = 500

    # 运行四个条件
    traj_a, agent_a = run_condition_a(num_steps)
    traj_b, agent_b = run_condition_b(num_steps)
    traj_c, agent_c = run_condition_c(num_steps)
    traj_d, agent_d = run_condition_d(num_steps)

    # 泛化测试
    generalization = test_generalization({
        'a': agent_a,
        'b': agent_b,
        'c': agent_c,
        'd': agent_d,
    })

    # 分析结果
    results = {
        'a': traj_a,
        'b': traj_b,
        'c': traj_c,
        'd': traj_d,
        'generalization': generalization,
    }
    analyze_results(results)

    # 保存结果
    import json
    output = {}
    for key in ['a', 'b', 'c', 'd']:
        traj = results[key]
        output[key] = {
            'final_error': float(np.mean(traj['prediction_errors'][-50:])),
            'final_symbols': int(traj['symbols'][-1]),
            'social_interactions': int(traj['social_interactions'][-1]),
            'stage_changes': int(traj['stage_changes'][-1]),
        }
        if 'scaffold_levels' in traj:
            output[key]['final_scaffold'] = float(traj['scaffold_levels'][-1])

    output['generalization'] = generalization

    with open('D:/mayAi/AILearning_v0527/mvl/social_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n结果已保存到: social_comparison.json")


if __name__ == '__main__':
    main()
