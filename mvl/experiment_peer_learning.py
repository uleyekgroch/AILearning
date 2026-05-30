"""
Phase 55: 同伴互学 — 对称知识交换验证

核心问题：
同伴互学（对称观察）vs 师徒教学（单向）vs 独立学习，哪种更有效？

Vygotsky 的最近发展区（ZPD）理论：
- 学习者在同伴协助下能超越独立能力
- 对称互学 vs 不对称教学的差异

实验设计：
1. 3 种学习模式对比（独立/同伴/师徒）
2. 同伴影响强度分析
3. 不同技能水平的同伴组合
4. 多同伴学习（2 vs 4 同伴）
"""

import json
import random
import numpy as np
from typing import List, Dict

from peer_learning import PeerAgent, PeerLearningEnvironment
from environment_physics import PhysicsEnvironment
from physics_rigid import RigidBody


def _make_env():
    """创建测试环境"""
    env = PhysicsEnvironment(width=10.0, height=10.0, depth=5.0)
    for i in range(5):
        body = RigidBody(
            x=random.uniform(1, 9),
            y=random.uniform(1, 9),
            z=random.uniform(0.5, 3),
            mass=random.uniform(0.5, 3.0),
            radius=random.uniform(0.3, 0.8),
        )
        env.add_rigid_body(body)
    return env


def _run_independent(obs_dim, action_dim, num_steps, model_type='linear'):
    """独立学习：Agent 不观察任何人"""
    env = _make_env()
    agent = PeerAgent(obs_dim=obs_dim, action_dim=action_dim,
                      model_type=model_type)

    errors = []
    for step in range(num_steps):
        obs_dict = env.get_observation()
        obs = agent.perceive(obs_dict)
        available = agent._get_available_actions()
        action = agent.act(obs_dict)

        next_obs_dict, _, _ = env.step(action)
        next_obs = agent.perceive(next_obs_dict)

        pe = np.mean((obs - next_obs) ** 2)
        errors.append(pe)

    return errors


def _run_peer(obs_dim, action_dim, num_steps, model_type='linear',
              peer_influence=0.1):
    """同伴学习：两个 Agent 互相观察"""
    env = _make_env()
    peer_env = PeerLearningEnvironment(env)

    agent_a = PeerAgent(obs_dim=obs_dim, action_dim=action_dim,
                        model_type=model_type)
    agent_a.peer_influence = peer_influence
    agent_b = PeerAgent(obs_dim=obs_dim, action_dim=action_dim,
                        model_type=model_type)
    agent_b.peer_influence = peer_influence

    errors_a = []

    for step in range(num_steps):
        obs_dict_a = peer_env._get_obs_for_peer('a')
        obs_dict_b = peer_env._get_obs_for_peer('b')

        obs_a = agent_a.perceive(obs_dict_a)
        obs_b = agent_b.perceive(obs_dict_b)

        action_a = agent_a.act(obs_dict_a)
        action_b = agent_b.act(obs_dict_b)

        next_obs_a, next_obs_b, _, _, _ = peer_env.step(action_a, action_b, dt=0.01)

        next_a = agent_a.perceive(next_obs_a)
        next_b = agent_b.perceive(next_obs_b)

        pe_a = np.mean((obs_a - next_a) ** 2)
        errors_a.append(pe_a)

        # 同伴观察：互相传递经验
        pred_b = agent_b.predictive_model.predict(obs_b, action_b)
        error_b = np.mean((next_b - pred_b) ** 2)
        agent_a.observe_peer(action_b, obs_b, next_b, error_b)

        pred_a = agent_a.predictive_model.predict(obs_a, action_a)
        error_a = np.mean((next_a - pred_a) ** 2)
        agent_b.observe_peer(action_a, obs_a, next_a, error_a)

        agent_a.predictive_model.learn(obs_a, action_a, next_a)
        agent_b.predictive_model.learn(obs_b, action_b, next_b)

    return errors_a


def experiment_1_learning_modes():
    """实验 1：3 种学习模式对比（500 步 x 10 次）"""
    print("=" * 60)
    print("实验 1：学习模式对比（独立 vs 同伴 vs 强同伴影响）")
    print("=" * 60)

    modes = {
        'independent': lambda: _run_independent(10, 8, 500),
        'peer_low': lambda: _run_peer(10, 8, 500, peer_influence=0.05),
        'peer_high': lambda: _run_peer(10, 8, 500, peer_influence=0.3),
    }

    results = {}
    for mode, run_fn in modes.items():
        all_errors = []
        for run in range(10):
            errors = run_fn()
            all_errors.append(np.mean(errors[-100:]))

        results[mode] = {
            'avg_error': round(float(np.mean(all_errors)), 4),
            'std': round(float(np.std(all_errors)), 4),
        }
        print(f"\n  {mode}:")
        print(f"    后期误差: {np.mean(all_errors):.4f} +/- {np.std(all_errors):.4f}")

    ranked = sorted(results.items(), key=lambda x: x[1]['avg_error'])
    print(f"\n  排名:")
    for i, (m, r) in enumerate(ranked):
        print(f"    {i+1}. {m}: {r['avg_error']:.4f}")

    return results


def experiment_2_influence_curve():
    """实验 2：同伴影响强度曲线（0→1.0）"""
    print("\n" + "=" * 60)
    print("实验 2：同伴影响强度 vs 学习效果（500 步 x 5 次）")
    print("=" * 60)

    influences = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    results = {}

    for inf in influences:
        run_errors = []
        for run in range(5):
            errors = _run_peer(10, 8, 500, peer_influence=inf)
            run_errors.append(np.mean(errors[-100:]))

        results[str(inf)] = {
            'avg_error': round(float(np.mean(run_errors)), 4),
        }
        print(f"  influence={inf}: 误差={np.mean(run_errors):.4f}")

    best_inf = min(results.items(), key=lambda x: x[1]['avg_error'])
    print(f"\n  最佳影响强度: {best_inf[0]} (误差={best_inf[1]['avg_error']:.4f})")

    return results


def experiment_3_skill_mismatch():
    """实验 3：不同技能水平的同伴组合"""
    print("\n" + "=" * 60)
    print("实验 3：技能水平组合（新手+新手 vs 新手+老手）")
    print("=" * 60)

    conditions = {
        'novice_novice': ('linear', 'linear'),
        'novice_expert': ('linear', 'neural_network'),
        'expert_expert': ('neural_network', 'neural_network'),
    }

    results = {}

    for cond, (type_a, type_b) in conditions.items():
        run_errors = []

        for run in range(5):
            env = _make_env()
            peer_env = PeerLearningEnvironment(env)

            agent_a = PeerAgent(obs_dim=10, action_dim=8, model_type=type_a)
            agent_b = PeerAgent(obs_dim=10, action_dim=8, model_type=type_b)

            errors_a = []

            for step in range(500):
                obs_dict_a = peer_env._get_obs_for_peer('a')
                obs_dict_b = peer_env._get_obs_for_peer('b')

                obs_a = agent_a.perceive(obs_dict_a)
                obs_b = agent_b.perceive(obs_dict_b)

                action_a = agent_a.act(obs_dict_a)
                action_b = agent_b.act(obs_dict_b)

                next_obs_a, next_obs_b, _, _, _ = peer_env.step(
                    action_a, action_b, dt=0.01)

                next_a = agent_a.perceive(next_obs_a)
                next_b = agent_b.perceive(next_obs_b)

                errors_a.append(np.mean((obs_a - next_a) ** 2))

                pred_b = agent_b.predictive_model.predict(obs_b, action_b)
                error_b = np.mean((next_b - pred_b) ** 2)
                agent_a.observe_peer(action_b, obs_b, next_b, error_b)

                pred_a = agent_a.predictive_model.predict(obs_a, action_a)
                error_a = np.mean((next_a - pred_a) ** 2)
                agent_b.observe_peer(action_a, obs_a, next_a, error_a)

                agent_a.predictive_model.learn(obs_a, action_a, next_a)
                agent_b.predictive_model.learn(obs_b, action_b, next_b)

            run_errors.append(np.mean(errors_a[-100:]))

        results[cond] = {
            'avg_error': round(float(np.mean(run_errors)), 4),
        }
        print(f"  {cond}: Agent A 后期误差={np.mean(run_errors):.4f}")

    return results


def experiment_4_speedup():
    """实验 4：学习速度对比（达到目标误差所需步数）"""
    print("\n" + "=" * 60)
    print("实验 4：达到目标误差的速度（x 10 次）")
    print("=" * 60)

    target_error = 0.5
    max_steps = 1000
    num_runs = 10

    modes = {
        'independent': lambda: _run_independent(10, 8, max_steps),
        'peer': lambda: _run_peer(10, 8, max_steps, peer_influence=0.2),
    }

    results = {}

    for mode, run_fn in modes.items():
        convergence_steps = []

        for run in range(num_runs):
            errors = run_fn()
            # 找到首次连续 10 步低于目标误差的步数
            found = max_steps
            for i in range(9, len(errors)):
                if all(e < target_error for e in errors[i-9:i+1]):
                    found = i + 1
                    break
            convergence_steps.append(found)

        avg = np.mean(convergence_steps)
        results[mode] = {
            'avg_steps': round(float(avg), 1),
            'converged': sum(1 for s in convergence_steps if s < max_steps),
        }
        print(f"  {mode}: 平均 {avg:.0f} 步, "
              f"收敛率 {sum(1 for s in convergence_steps if s < max_steps)}/{num_runs}")

    return results


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_learning_modes()
    results['experiment_2'] = experiment_2_influence_curve()
    results['experiment_3'] = experiment_3_skill_mismatch()
    results['experiment_4'] = experiment_4_speedup()

    with open('peer_learning_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到 peer_learning_results.json")
