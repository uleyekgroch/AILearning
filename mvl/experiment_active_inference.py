"""
Phase 54: 主动推理验证 — Expected Free Energy 动作选择

核心问题：
主动推理（最小化期望自由能）vs 随机 vs 纯好奇 vs 纯目标，哪种策略最优？

期望自由能 G = -信息增益 - 工具价值 + 风险惩罚
- 信息增益（探索）：选择能减少不确定性的动作
- 工具价值（利用）：选择能到达偏好状态的动作
- 主动推理统一两者

实验设计：
1. 4 种动作选择策略对比（500 步 x 10 次）
2. 探索-利用权衡分析
3. 风险敏感度效果
4. 迁移到新目标
"""

import json
import random
import numpy as np
from typing import List
from collections import deque

from active_inference import ActiveInferenceModule
from environment_physics import PhysicsEnvironment
from physics_rigid import RigidBody


def _make_env() -> PhysicsEnvironment:
    """创建带刚体的测试环境"""
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


class SimplePredictor:
    """简单线性预测模型——用于主动推理实验"""

    def __init__(self, obs_dim: int = 20, action_dim: int = 8):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.W = np.random.randn(obs_dim, obs_dim) * 0.1
        self.W_a = np.random.randn(action_dim, obs_dim) * 0.1
        self.b = np.zeros(obs_dim)
        self.log_var = np.zeros(obs_dim)  # 不确定性
        self._last_log_var = np.zeros(obs_dim)
        self.lr = 0.01

    def predict(self, obs: np.ndarray, action: int):
        """预测下一个观测"""
        h = obs @ self.W + self.W_a[action] + self.b
        self._last_log_var = self.log_var.copy()
        return h.copy(), self.log_var.copy()

    def update(self, obs: np.ndarray, action: int, next_obs: np.ndarray):
        """更新预测模型"""
        pred, _ = self.predict(obs, action)
        error = next_obs - pred
        # 梯度裁剪防止发散
        error_clipped = np.clip(error, -5.0, 5.0)
        self.W += self.lr * np.outer(np.clip(obs, -5, 5), error_clipped)
        self.W_a[action] += self.lr * error_clipped
        self.b += self.lr * error_clipped * 0.1
        # 权重正则化
        self.W = np.clip(self.W, -3, 3)
        self.W_a = np.clip(self.W_a, -3, 3)
        # 更新不确定性
        self.log_var = 0.95 * self.log_var + 0.05 * np.clip(
            np.log(error_clipped ** 2 + 1e-4), -5, 2)

    def get_last_log_var(self):
        return self._last_log_var


class SimpleBelief:
    """简单信念状态"""
    def __init__(self, obs_dim: int = 20):
        self.log_var = np.zeros(obs_dim)

    def update(self, obs, pred):
        error = obs - pred
        self.log_var = 0.9 * self.log_var + 0.1 * np.log(error ** 2 + 1e-8)


def _obs_from_env(env: PhysicsEnvironment) -> np.ndarray:
    """从物理环境提取观测向量"""
    obs_dict = env.get_observation()
    parts = [np.array([env.agent_x, env.agent_y, env.agent_z])]
    for body in env.rigid_engine.bodies:
        parts.append(np.array([body.x, body.y, body.z, body.mass, body.radius]))
    # padding to obs_dim=20, clipped for stability
    flat = np.concatenate(parts) if parts else np.zeros(20)
    if len(flat) < 20:
        flat = np.concatenate([flat, np.zeros(20 - len(flat))])
    return np.clip(flat[:20], -5, 5)


ACTIONS = list(range(8))  # 0-7: 前后左右上下推拉


def _select_curiosity(predictor, obs, available):
    """纯好奇心：选最不确定的动作"""
    best_a = available[0]
    best_u = -float('inf')
    for a in available:
        _, log_var = predictor.predict(obs, a)
        u = np.sum(np.exp(log_var))
        if u > best_u:
            best_u = u
            best_a = a
    return best_a


def _select_goal(predictor, obs, available, pref_mu):
    """纯目标导向：只看工具价值"""
    best_a = available[0]
    best_v = -float('inf')
    for a in available:
        pred, _ = predictor.predict(obs, a)
        err = pred - pref_mu[:len(pred)]
        v = -0.5 * np.sum(err ** 2) + np.random.normal(0, 0.1)
        if v > best_v:
            best_v = v
            best_a = a
    return best_a


def experiment_1_strategy_comparison():
    """实验 1：4 种策略对比（500 步 x 10 次）"""
    print("=" * 60)
    print("实验 1：动作选择策略对比（500 步 x 10 次）")
    print("=" * 60)

    strategies = ['random', 'curiosity', 'goal_directed', 'active_inference']
    all_results = {}

    for strategy in strategies:
        run_errors = []
        run_coverages = []

        for run in range(10):
            env = _make_env()
            predictor = SimplePredictor()
            belief = SimpleBelief()
            ai_module = ActiveInferenceModule(action_dim=8)
            target = np.random.randn(20) * 0.3
            if strategy in ('goal_directed', 'active_inference'):
                ai_module.set_preference(target, precision=2.0)

            errors = []
            unique_actions = set()

            for step in range(500):
                obs = _obs_from_env(env)
                pred, _ = predictor.predict(obs, 0)
                belief.update(obs, pred)

                if strategy == 'random':
                    action = random.choice(ACTIONS)
                elif strategy == 'curiosity':
                    action = _select_curiosity(predictor, obs, ACTIONS)
                elif strategy == 'goal_directed':
                    action = _select_goal(predictor, obs, ACTIONS, target)
                else:
                    action = ai_module.select_action(
                        obs, predictor, belief, ACTIONS)

                unique_actions.add(action)
                next_obs_dict, _, _ = env.step(action)
                next_obs = _obs_from_env(env)

                pred, _ = predictor.predict(obs, action)
                pe = np.mean((next_obs - pred) ** 2)
                errors.append(pe)
                predictor.update(obs, action, next_obs)

            run_errors.append(np.mean(errors[-100:]))
            run_coverages.append(len(unique_actions) / 8)

        all_results[strategy] = {
            'avg_error': round(float(np.mean(run_errors)), 4),
            'std_error': round(float(np.std(run_errors)), 4),
            'action_coverage': round(float(np.mean(run_coverages)), 4),
        }
        print(f"\n  {strategy}:")
        print(f"    后期误差: {np.mean(run_errors):.4f} +/- {np.std(run_errors):.4f}")
        print(f"    动作覆盖: {np.mean(run_coverages):.1%}")

    ranked = sorted(all_results.items(), key=lambda x: x[1]['avg_error'])
    print(f"\n  排名（误差从低到高）:")
    for i, (s, r) in enumerate(ranked):
        print(f"    {i+1}. {s}: {r['avg_error']:.4f}")

    return all_results


def experiment_2_exploration_exploitation():
    """实验 2：探索-利用动态平衡"""
    print("\n" + "=" * 60)
    print("实验 2：探索-利用动态平衡（500 步）")
    print("=" * 60)

    env = _make_env()
    predictor = SimplePredictor()
    belief = SimpleBelief()
    ai_module = ActiveInferenceModule(action_dim=8)
    target = np.random.randn(20) * 0.3
    ai_module.set_preference(target, precision=2.0)

    phases = {'early': [], 'mid': [], 'late': []}

    for step in range(500):
        obs = _obs_from_env(env)
        pred, _ = predictor.predict(obs, 0)
        belief.update(obs, pred)

        # 记录各动作的信息增益和工具价值
        info_gains = []
        pragmatics = []
        for a in ACTIONS:
            p, lv = predictor.predict(obs, a)
            var = np.exp(lv)
            cur_var = np.exp(belief.log_var)
            ig = np.sum(np.log(cur_var + 1e-8) - np.log(var + 1e-8))
            pe = p - target[:len(p)]
            pr = -0.5 * 2.0 * np.sum(pe ** 2)
            info_gains.append(ig)
            pragmatics.append(pr)

        action = ai_module.select_action(obs, predictor, belief, ACTIONS)

        next_obs_dict, _, _ = env.step(action)
        next_obs = _obs_from_env(env)
        predictor.update(obs, action, next_obs)

        phase = 'early' if step < 100 else ('mid' if step < 300 else 'late')
        phases[phase].append({
            'info_gain': info_gains[action],
            'pragmatic': pragmatics[action],
        })

    results = {}
    for phase, records in phases.items():
        avg_ig = np.mean([r['info_gain'] for r in records])
        avg_pr = np.mean([r['pragmatic'] for r in records])
        total = abs(avg_ig) + abs(avg_pr) + 1e-8
        explore_ratio = abs(avg_ig) / total

        results[phase] = {
            'avg_info_gain': round(float(avg_ig), 4),
            'avg_pragmatic': round(float(avg_pr), 4),
            'explore_ratio': round(float(explore_ratio), 4),
        }
        print(f"  {phase}: 信息增益={avg_ig:.4f}, 工具价值={avg_pr:.4f}, "
              f"探索占比={explore_ratio:.2%}")

    return results


def experiment_3_risk_sensitivity():
    """实验 3：风险敏感度效果"""
    print("\n" + "=" * 60)
    print("实验 3：风险敏感度（500 步 x 5 次）")
    print("=" * 60)

    weights = [0.0, 0.1, 0.5, 1.0, 2.0]
    results = {}

    for w in weights:
        run_gef = []
        run_safe = []

        for run in range(5):
            env = _make_env()
            predictor = SimplePredictor()
            belief = SimpleBelief()
            ai = ActiveInferenceModule(action_dim=8)
            ai.set_preference(np.random.randn(20) * 0.3, precision=2.0)
            ai.set_risk_sensitivity(w)

            safe = 0
            for step in range(500):
                obs = _obs_from_env(env)
                pred, _ = predictor.predict(obs, 0)
                belief.update(obs, pred)

                action_vars = []
                for a in ACTIONS:
                    _, lv = predictor.predict(obs, a)
                    action_vars.append(np.sum(np.exp(lv)))

                action = ai.select_action(obs, predictor, belief, ACTIONS)

                median_var = np.median(action_vars)
                if action_vars[action] <= median_var:
                    safe += 1

                next_obs_dict, _, _ = env.step(action)
                next_obs = _obs_from_env(env)
                predictor.update(obs, action, next_obs)

            run_gef.append(ai.get_avg_gef())
            run_safe.append(safe / 500)

        results[str(w)] = {
            'avg_gef': round(float(np.mean(run_gef)), 4),
            'safe_ratio': round(float(np.mean(run_safe)), 4),
        }
        print(f"  weight={w}: GEF={np.mean(run_gef):.4f}, "
              f"安全动作={np.mean(run_safe):.1%}")

    return results


def experiment_4_transfer():
    """实验 4：迁移到新目标（预训练 200 + 新目标 300 步 x 5 次）"""
    print("\n" + "=" * 60)
    print("实验 4：目标迁移效果（预训练 200 + 新目标 300 步 x 5 次）")
    print("=" * 60)

    strategies = ['random', 'curiosity', 'active_inference']
    results = {}

    for strategy in strategies:
        run_errors = []

        for run in range(5):
            env = _make_env()
            predictor = SimplePredictor()
            belief = SimpleBelief()
            ai = ActiveInferenceModule(action_dim=8)
            old_target = np.random.randn(20) * 0.3
            ai.set_preference(old_target, precision=2.0)

            # 预训练
            for step in range(200):
                obs = _obs_from_env(env)
                pred, _ = predictor.predict(obs, 0)
                belief.update(obs, pred)

                if strategy == 'random':
                    action = random.choice(ACTIONS)
                elif strategy == 'curiosity':
                    action = _select_curiosity(predictor, obs, ACTIONS)
                else:
                    action = ai.select_action(obs, predictor, belief, ACTIONS)

                next_obs_dict, _, _ = env.step(action)
                next_obs = _obs_from_env(env)
                predictor.update(obs, action, next_obs)

            # 新目标
            new_target = np.random.randn(20) * 0.5
            ai.set_preference(new_target, precision=2.0)

            new_errors = []
            for step in range(300):
                obs = _obs_from_env(env)
                pred, _ = predictor.predict(obs, 0)
                belief.update(obs, pred)

                if strategy == 'random':
                    action = random.choice(ACTIONS)
                elif strategy == 'curiosity':
                    action = _select_curiosity(predictor, obs, ACTIONS)
                else:
                    action = ai.select_action(obs, predictor, belief, ACTIONS)

                next_obs_dict, _, _ = env.step(action)
                next_obs = _obs_from_env(env)

                pred, _ = predictor.predict(obs, action)
                new_errors.append(np.mean((next_obs - pred) ** 2))
                predictor.update(obs, action, next_obs)

            run_errors.append(np.mean(new_errors[-100:]))

        results[strategy] = {
            'new_task_error': round(float(np.mean(run_errors)), 4),
        }
        print(f"  {strategy}: 新任务后期误差={np.mean(run_errors):.4f}")

    return results


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_strategy_comparison()
    results['experiment_2'] = experiment_2_exploration_exploitation()
    results['experiment_3'] = experiment_3_risk_sensitivity()
    results['experiment_4'] = experiment_4_transfer()

    with open('active_inference_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到 active_inference_results.json")
