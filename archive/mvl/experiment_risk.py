"""
不确定性感知决策实验

核心对比：FEP Risk-Aware vs FEP Risk-Unaware
同一 agent，同一学习机制，唯一变量：是否启用精度加权风险惩罚。

为什么不用 MSE 对比？
MSE agent 的动作选择由好奇心驱动，不使用任务奖励。
对比 MSE vs FEP 混淆了两个因素：(1) 不确定性估计 (2) 目标导向行为。
FEP ON vs OFF 直接隔离"精度驱动风险规避"的效果。

三个风险敏感任务：
1. 悬崖导航 — 掉落=失败
2. 危险探索 — 进入=失败
3. 风险-收益权衡 — 安全长路 vs 危险短路

验证假设：在"做错有代价"的环境中，
精度加权风险惩罚能降低失败率、提高风险调整收益。
"""

import sys
import numpy as np
from typing import Dict

sys.stdout.reconfigure(encoding='utf-8')

from agent_fep import FEPAgent
from task_environments import (
    create_cliff_navigation_env,
    create_hazardous_exploration_env,
    create_risk_reward_tradeoff_env,
)


def run_episode(agent, env, task, risk_weight: float = 0.0) -> Dict:
    """运行一个 episode，返回指标"""
    agent.set_risk_sensitive(risk_weight)

    obs = env.get_observation()
    total_reward = 0.0

    for step in range(task.max_steps):
        action = agent.act(obs)
        next_obs, reward, done = env.step(action)
        agent.learn_from_experience(obs, action, next_obs, reward)

        total_reward += reward
        obs = next_obs

        if done:
            break

    return {
        'success': task.is_success,
        'failure': task.is_failure,
        'steps': task.step_count,
        'total_reward': total_reward,
        'hazard_steps': getattr(task, 'hazard_steps', 0),
        'near_hazard_steps': getattr(task, 'near_hazard_steps', 0),
    }


def run_task_comparison(task_name: str, create_env_fn, num_episodes: int = 50,
                        num_steps_train: int = 1000,
                        risk_weight: float = 0.5) -> Dict:
    """在单个任务上对比 FEP Risk-OFF vs FEP Risk-ON"""
    results = {'risk_off': [], 'risk_on': []}

    for condition in ['risk_off', 'risk_on']:
        agent = FEPAgent(obs_dim=20, action_dim=8)
        rw = 0.0 if condition == 'risk_off' else risk_weight

        # 预训练阶段（学习物理规律，无风险惩罚）
        train_env, train_task = create_env_fn()
        agent.set_risk_sensitive(0.0)  # 预训练时关闭风险惩罚
        for _ in range(num_steps_train):
            obs = train_env.get_observation()
            action = agent.act(obs)
            next_obs, reward, done = train_env.step(action)
            agent.learn_from_experience(obs, action, next_obs, reward)
            if done:
                train_env.reset()

        # 评估阶段
        for ep in range(num_episodes):
            env, task = create_env_fn()
            metrics = run_episode(agent, env, task, rw)
            results[condition].append(metrics)
            env.reset()

    return results


def analyze_task_results(results: Dict, task_name: str) -> Dict:
    """分析并返回汇总统计"""
    summary = {}
    for condition in ['risk_off', 'risk_on']:
        episodes = results[condition]
        successes = sum(1 for e in episodes if e['success'])
        failures = sum(1 for e in episodes if e['failure'])
        total = len(episodes)

        rewards = [e['total_reward'] for e in episodes]
        avg_reward = np.mean(rewards)
        failure_rate = failures / total

        # 成功率加权收益：成功 episode 的平均收益 × 成功率
        success_rewards = [e['total_reward'] for e in episodes if e['success']]
        success_avg = np.mean(success_rewards) if success_rewards else 0.0
        weighted_reward = success_avg * (successes / total)

        summary[condition] = {
            'success_rate': successes / total,
            'failure_rate': failure_rate,
            'avg_steps': float(np.mean([e['steps'] for e in episodes])),
            'avg_reward': float(avg_reward),
            'avg_success_reward': float(success_avg),
            'weighted_reward': float(weighted_reward),
            'avg_hazard_steps': float(np.mean([e['hazard_steps'] for e in episodes])),
            'avg_near_hazard_steps': float(np.mean([e['near_hazard_steps'] for e in episodes])),
        }

    return summary


def print_task_results(summary: Dict, task_name: str):
    """打印任务结果"""
    print(f"\n{'='*70}")
    print(f"任务: {task_name}")
    print(f"{'='*70}")
    print(f"{'条件':>12} | {'成功率':>8} | {'失败率':>8} | {'平均步数':>8} | {'加权收益':>10}")
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    for condition in ['risk_off', 'risk_on']:
        s = summary[condition]
        label = 'FEP-RiskOFF' if condition == 'risk_off' else 'FEP-RiskON'
        print(f"{label:>12} | {s['success_rate']:>7.1%} | {s['failure_rate']:>7.1%} | "
              f"{s['avg_steps']:>8.1f} | {s['weighted_reward']:>10.3f}")


def main():
    """主函数"""
    print("不确定性感知决策实验")
    print("FEP Risk-OFF vs Risk-ON 风险权重扫描\n")

    tasks = {
        '悬崖导航': create_cliff_navigation_env,
        '危险探索': create_hazardous_exploration_env,
        '风险-收益权衡': create_risk_reward_tradeoff_env,
    }

    risk_weights = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    all_results = {}

    for task_name, create_fn in tasks.items():
        print(f"\n{'='*70}")
        print(f"任务: {task_name}")
        print(f"{'='*70}")
        print(f"{'权重':>6} | {'成功率':>8} | {'失败率':>8} | {'加权收益':>10}")
        print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

        task_results = {}
        for rw in risk_weights:
            if rw == 0.0:
                # Risk-OFF baseline
                results = run_task_comparison(task_name, create_fn, num_episodes=30,
                                              num_steps_train=500, risk_weight=0.0)
                summary = analyze_task_results(results, task_name)
                off_summary = summary['risk_off']
                task_results[0.0] = off_summary
                print(f"{'OFF':>6} | {off_summary['success_rate']:>7.1%} | "
                      f"{off_summary['failure_rate']:>7.1%} | {off_summary['weighted_reward']:>10.3f}")
            else:
                results = run_task_comparison(task_name, create_fn, num_episodes=30,
                                              num_steps_train=500, risk_weight=rw)
                summary = analyze_task_results(results, task_name)
                on_summary = summary['risk_on']
                task_results[rw] = on_summary
                print(f"{rw:>6.1f} | {on_summary['success_rate']:>7.1%} | "
                      f"{on_summary['failure_rate']:>7.1%} | {on_summary['weighted_reward']:>10.3f}")

        all_results[task_name] = task_results

    # 总结：每个任务的最优权重
    print(f"\n{'='*70}")
    print("最优风险权重")
    print(f"{'='*70}")

    for task_name, task_results in all_results.items():
        best_rw = max(task_results.keys(),
                      key=lambda rw: task_results[rw]['weighted_reward'])
        best = task_results[best_rw]
        baseline = task_results[0.0]

        print(f"\n{task_name}:")
        print(f"  最优权重: {best_rw}")
        print(f"  成功率: {baseline['success_rate']:.1%} → {best['success_rate']:.1%}")
        print(f"  失败率: {baseline['failure_rate']:.1%} → {best['failure_rate']:.1%}")
        print(f"  加权收益: {baseline['weighted_reward']:.3f} → {best['weighted_reward']:.3f}")

    # 验证假设
    print(f"\n{'='*70}")
    print("假设验证")
    print(f"{'='*70}")

    tasks_improved = 0
    for task_name, task_results in all_results.items():
        baseline = task_results[0.0]
        best_rw = max(task_results.keys(),
                      key=lambda rw: task_results[rw]['weighted_reward'])
        best = task_results[best_rw]

        improved = (best['failure_rate'] < baseline['failure_rate'] or
                    best['success_rate'] > baseline['success_rate'])
        status = "✓ 有效" if improved else "✗ 无效"
        print(f"  {task_name}: {status} (最优权重={best_rw})")
        if improved:
            tasks_improved += 1

    print(f"\n  {tasks_improved}/{len(tasks)} 个任务显示风险惩罚有效")

    # 保存结果
    import json
    output = {}
    for task_name, task_results in all_results.items():
        output[task_name] = {str(k): v for k, v in task_results.items()}

    with open('D:/mayAi/AILearning_v0527/mvl/risk_comparison.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n结果已保存到: risk_comparison.json")


if __name__ == '__main__':
    main()
