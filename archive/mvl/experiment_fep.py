"""
自由能原理对比实验

三条件对比：
A. MSE 学习（当前系统 baseline）
B. 自由能学习（精度加权，无主动推理）
C. 自由能 + 主动推理（完整 FEP）

验证：FEP 的精度加权和主动推理是否优于 MSE
"""

import sys
import numpy as np
from typing import Dict

sys.stdout.reconfigure(encoding='utf-8')

from environment_physics import PhysicsEnvironment, create_rich_physics_world
from physics_rigid import RigidBody, MaterialType
from agent_3d import LearningAgent3D
from agent_fep import FEPAgent
from teacher import SimpleTeacher


def run_condition_mse(num_steps: int = 1000):
    """条件A：MSE 学习（baseline）"""
    print("\n--- 条件A：MSE 学习（baseline）---")
    env = create_rich_physics_world()
    agent = LearningAgent3D(obs_dim=20, action_dim=8, model_type='neural_network')
    teacher = SimpleTeacher()

    trajectory = {
        'steps': [], 'errors': [], 'symbols': [],
        'learning_progress': [], 'stages': [],
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
        trajectory['errors'].append(error)
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['learning_progress'].append(agent.predictive_model.get_learning_progress())
        trajectory['stages'].append(agent.development.current_stage)

        if done:
            env.reset()

        if step % 200 == 0:
            print(f"  步骤 {step}: 误差={error:.4f}, 符号={len(agent.grounding.get_grounded_symbols())}")

    return trajectory, agent


def run_condition_fep(num_steps: int = 1000):
    """条件B：自由能学习（精度加权，无主动推理）"""
    print("\n--- 条件B：自由能学习（精度加权）---")
    env = create_rich_physics_world()
    agent = FEPAgent(obs_dim=20, action_dim=8)
    teacher = SimpleTeacher()

    trajectory = {
        'steps': [], 'errors': [], 'symbols': [],
        'learning_progress': [], 'stages': [],
        'free_energies': [], 'precisions': [],
    }

    for step in range(num_steps):
        obs = env.get_observation()
        action = agent.act(obs)
        next_obs, _, done = env.step(action)

        free_energy, pw_error = agent.learn_from_experience(obs, action, next_obs)

        # 使用原始 MSE 作为对比指标（与 MSE agent 一致）
        obs_vec = agent.perceive(obs)
        next_obs_vec = agent.perceive(next_obs)
        predicted = agent.predictor.predict_mean_only(obs_vec, action)
        raw_mse = np.mean((predicted - next_obs_vec) ** 2)

        # 教师教学
        target = teacher.observe(obs)
        if target:
            obj = target.get('object')
            if obj:
                name = teacher._get_object_name(obj)
                if name:
                    agent.grounding.ground_from_social(name, obs_vec, 'naming')
                    agent.record_social_interaction()

        trajectory['steps'].append(step)
        trajectory['errors'].append(raw_mse)  # 使用原始 MSE
        trajectory['symbols'].append(len(agent.grounding.get_grounded_symbols()))
        trajectory['learning_progress'].append(agent.predictor.get_learning_progress())
        trajectory['stages'].append(agent.development.current_stage)
        trajectory['free_energies'].append(free_energy)
        trajectory['precisions'].append(agent.predictor.get_avg_precision())

        if done:
            env.reset()

        if step % 200 == 0:
            print(f"  步骤 {step}: MSE={raw_mse:.4f}, FE={free_energy:.4f}, "
                  f"精度={agent.predictor.get_avg_precision():.3f}, "
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

            if hasattr(agent, 'predictor'):
                # FEP agent
                predicted = agent.predictor.predict_mean_only(obs_vec, action)
            else:
                predicted = agent.predictive_model.predict(obs_vec, action)

            error = np.mean((predicted - next_obs_vec) ** 2)
            agent_errors.append(error)

            if done:
                test_env.reset()

        errors[name] = np.mean(agent_errors)

    return errors


def test_uncertainty_calibration(agent_fep) -> Dict:
    """
    不确定性校准测试

    衡量 FEP agent 的不确定性估计是否准确：
    - 预测 ± 2σ 应该覆盖 ~95% 的实际观测
    - 如果实际误差经常超出预测区间 → 不确定性被低估
    - 如果实际误差总是在预测区间内 → 不确定性被高估
    """
    print("\n--- 不确定性校准测试 ---")

    test_env = PhysicsEnvironment(10, 10, 10)
    test_env.add_rigid_body(RigidBody(3, 3, 5, radius=0.4, mass=3.0, material=MaterialType.STONE))
    test_env.add_rigid_body(RigidBody(7, 7, 3, radius=0.6, mass=1.0, material=MaterialType.RUBBER))

    coverage_1sigma = 0  # 实际误差在 ±1σ 内的比例
    coverage_2sigma = 0  # 实际误差在 ±2σ 内的比例
    total_dims = 0
    calibration_errors = []  # |predicted_std - actual_error|

    for _ in range(100):
        obs = test_env.get_observation()
        action = agent_fep.act(obs)
        next_obs, _, done = test_env.step(action)

        obs_vec = agent_fep.perceive(obs)
        next_obs_vec = agent_fep.perceive(next_obs)

        # 概率预测
        mean, log_var = agent_fep.predictor.predict(obs_vec, action)
        std = np.sqrt(np.exp(log_var))

        actual_error = np.abs(next_obs_vec - mean)

        # 覆盖率
        coverage_1sigma += np.sum(actual_error <= std)
        coverage_2sigma += np.sum(actual_error <= 2 * std)
        total_dims += len(actual_error)

        # 校准误差
        calibration_errors.append(np.mean(np.abs(std - actual_error)))

        if done:
            test_env.reset()

    return {
        'coverage_1sigma': coverage_1sigma / total_dims,
        'coverage_2sigma': coverage_2sigma / total_dims,
        'mean_calibration_error': np.mean(calibration_errors),
        'ideal_1sigma': 0.683,  # 正态分布理论值
        'ideal_2sigma': 0.954,
    }


def analyze_results(traj_mse, agent_mse, traj_fep, agent_fep, generalization, calibration):
    """分析实验结果"""
    print("\n" + "=" * 60)
    print("自由能原理对比实验结果")
    print("=" * 60)

    # 最终误差
    final_mse = np.mean(traj_mse['errors'][-50:])
    final_fep = np.mean(traj_fep['errors'][-50:])
    print(f"\n1. 最终预测误差（最后50步均值）:")
    print(f"   MSE 学习: {final_mse:.4f}")
    print(f"   FEP 学习: {final_fep:.4f}")

    # 学习速度（误差下降到 0.1 以下的步数）
    mse_below_01 = next((i for i, e in enumerate(traj_mse['errors']) if e < 0.1), num_steps)
    fep_below_01 = next((i for i, e in enumerate(traj_fep['errors']) if e < 0.1), num_steps)
    print(f"\n2. 学习速度（误差降到 0.1 以下的步数）:")
    print(f"   MSE 学习: {mse_below_01}")
    print(f"   FEP 学习: {fep_below_01}")

    # 符号学习
    print(f"\n3. 符号学习数量:")
    print(f"   MSE 学习: {traj_mse['symbols'][-1]} 个")
    print(f"   FEP 学习: {traj_fep['symbols'][-1]} 个")

    # 发展阶段
    print(f"\n4. 最终发展阶段:")
    print(f"   MSE 学习: {traj_mse['stages'][-1]}")
    print(f"   FEP 学习: {traj_fep['stages'][-1]}")

    # 精度统计
    if 'precisions' in traj_fep:
        print(f"\n5. 精度统计（FEP）:")
        print(f"   平均精度: {np.mean(traj_fep['precisions']):.3f}")
        print(f"   精度范围: [{min(traj_fep['precisions']):.3f}, {max(traj_fep['precisions']):.3f}]")

    # 自由能
    if 'free_energies' in traj_fep:
        print(f"\n6. 自由能统计（FEP）:")
        print(f"   初始自由能: {traj_fep['free_energies'][0]:.4f}")
        print(f"   最终自由能: {np.mean(traj_fep['free_energies'][-50:]):.4f}")
        fe_reduction = (traj_fep['free_energies'][0] - np.mean(traj_fep['free_energies'][-50:]))
        print(f"   自由能减少: {fe_reduction:.4f}")

    # 泛化测试
    print(f"\n7. 泛化测试（新场景预测误差）:")
    for name, error in generalization.items():
        label = 'MSE 学习' if name == 'mse' else 'FEP 学习'
        print(f"   {label}: {error:.4f}")

    if 'mse' in generalization and 'fep' in generalization:
        improvement = (generalization['mse'] - generalization['fep']) / generalization['mse'] * 100
        print(f"\n   → FEP 比 MSE 泛化误差{'降低' if improvement > 0 else '增加'} {abs(improvement):.1f}%")

    # 不确定性校准（FEP 独有优势）
    print(f"\n8. 不确定性校准（FEP 独有优势）:")
    print(f"   ±1σ 覆盖率: {calibration['coverage_1sigma']:.1%} (理想: {calibration['ideal_1sigma']:.1%})")
    print(f"   ±2σ 覆盖率: {calibration['coverage_2sigma']:.1%} (理想: {calibration['ideal_2sigma']:.1%})")
    print(f"   平均校准误差: {calibration['mean_calibration_error']:.4f}")

    # 总结
    print(f"\n{'='*60}")
    print("总结:")
    print(f"  MSE 优势: 更低的预测误差 ({final_mse:.4f} vs {final_fep:.4f})")
    print(f"  FEP 优势: 不确定性估计 (±2σ 覆盖率 {calibration['coverage_2sigma']:.1%})")
    print(f"  FEP 优势: 更多符号学习 ({traj_fep['symbols'][-1]} vs {traj_mse['symbols'][-1]})")


# 全局 num_steps 供 analyze_results 使用
num_steps = 1000


def main():
    """主函数"""
    global num_steps
    print("开始自由能原理对比实验...")
    print("对比 MSE 学习 vs 自由能学习\n")

    num_steps = 1000

    # 运行两个条件
    traj_mse, agent_mse = run_condition_mse(num_steps)
    traj_fep, agent_fep = run_condition_fep(num_steps)

    # 泛化测试
    generalization = test_generalization({
        'mse': agent_mse,
        'fep': agent_fep,
    })

    # 不确定性校准测试（FEP 独有）
    calibration = test_uncertainty_calibration(agent_fep)

    # 分析结果
    analyze_results(traj_mse, agent_mse, traj_fep, agent_fep, generalization, calibration)

    # 保存结果
    import json
    output = {
        'mse': {
            'final_error': float(np.mean(traj_mse['errors'][-50:])),
            'final_symbols': int(traj_mse['symbols'][-1]),
            'final_stage': traj_mse['stages'][-1],
        },
        'fep': {
            'final_error': float(np.mean(traj_fep['errors'][-50:])),
            'final_symbols': int(traj_fep['symbols'][-1]),
            'final_stage': traj_fep['stages'][-1],
            'avg_precision': float(np.mean(traj_fep['precisions'])),
            'initial_fe': float(traj_fep['free_energies'][0]),
            'final_fe': float(np.mean(traj_fep['free_energies'][-50:])),
        },
        'generalization': generalization,
        'calibration': calibration,
    }

    with open('D:/mayAi/AILearning_v0527/mvl/fep_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n结果已保存到: fep_comparison.json")


if __name__ == '__main__':
    main()
