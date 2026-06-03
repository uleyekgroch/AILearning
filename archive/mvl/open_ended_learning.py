"""
Phase 56: 开放式学习 — 纯内驱力探索

核心问题：
没有外部任务，Agent 只有内驱力（好奇/新奇/掌控感），能否自发发现环境结构？

内驱力理论：
1. 好奇心（curiosity）：预测误差 × 可学习性（Schmidhuber）
2. 新奇性（novelty）：从未见过的状态（Bellemare et al., 2016）
3. 掌控感（empowerment）：能控制环境变化的程度（Klyubin et al., 2005）
4. 信息增益（information gain）：能减少多少不确定性（Active Inference）

与现有系统的区别：
- 无外部奖励、无通信任务、无目标
- Agent 纯粹被内驱力驱动
- 测量：探索覆盖率、发现的结构、学习进度

三阶段实验：
1. 内驱力对比（好奇/新奇/掌控/信息增益/随机）
2. 环境结构发现率
3. 内驱力驱动的语言涌现（无通信压力下符号是否涌现？）
"""

import json
import math
import random
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import deque

from language_emergence import LanguageAgent, cross_language_round, generate_rich_scene
from language_rich_scene import generate_rich_scene_v2, ALL_ATTRIBUTE_NAMES


class IntrinsicMotivation:
    """内驱力模块"""

    def __init__(self, obs_dim: int = 20, action_dim: int = 8):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 预测模型（简化版）
        self.W = np.random.randn(obs_dim, obs_dim) * 0.1
        self.W_a = np.random.randn(action_dim, obs_dim) * 0.1
        self.lr = 0.01

        # 新奇性追踪
        self.visited_states = deque(maxlen=200)
        self.novelty_count = {}

        # 掌控感追踪
        self.action_effects = {a: [] for a in range(action_dim)}

        # 学习进度追踪
        self.error_history = deque(maxlen=50)

    def predict(self, obs: np.ndarray, action: int):
        h = obs @ self.W + self.W_a[action]
        return h.copy()

    def update(self, obs, action, next_obs):
        pred = self.predict(obs, action)
        error = next_obs - pred
        self.W += self.lr * np.outer(obs, error)
        self.W_a[action] += self.lr * error
        pe = float(np.mean(error ** 2))
        self.error_history.append(pe)

        # 记录动作效果
        effect = float(np.linalg.norm(next_obs - obs))
        self.action_effects[action].append(effect)

        # 记录状态（离散化）
        state_key = tuple((obs * 3).astype(int))
        self.visited_states.append(state_key)

        return pe

    def curiosity(self, obs, action):
        """好奇心 = 预测误差 × 可学习性"""
        pred = self.predict(obs, action)
        # 估计可学习性：近期误差下降速度
        if len(self.error_history) < 5:
            learnability = 1.0
        else:
            recent = list(self.error_history)[-5:]
            learnability = max(0, recent[0] - recent[-1]) / (recent[0] + 1e-8)
        uncertainty = float(np.std(pred))
        return uncertainty * (1 + learnability)

    def novelty(self, obs, action):
        """新奇性 = 状态越罕见越有吸引力"""
        pred = self.predict(obs, action)
        state_key = tuple((pred * 3).astype(int))
        count = sum(1 for s in self.visited_states if s == state_key)
        return 1.0 / (1.0 + count)

    def empowerment(self, obs, action):
        """掌控感 = 该动作过去产生过多少变化"""
        effects = self.action_effects[action]
        if len(effects) < 3:
            return 0.5  # 未知动作的默认值
        return float(np.mean(effects[-10:]))

    def information_gain(self, obs, action):
        """信息增益 = 不确定性减少的期望"""
        pred = self.predict(obs, action)
        if len(self.error_history) < 3:
            return float(np.std(pred))
        recent_err = np.mean(list(self.error_history)[-3:])
        return recent_err * float(np.std(pred))


class OpenEndedAgent:
    """开放式学习 Agent"""

    def __init__(self, obs_dim: int = 20, action_dim: int = 8,
                 drive: str = 'curiosity'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.drive = drive
        self.motivation = IntrinsicMotivation(obs_dim, action_dim)
        self.discovered_structures = set()
        self.state_coverage = set()

    def act(self, obs: np.ndarray) -> int:
        """根据内驱力选择动作"""
        scores = []
        for a in range(self.action_dim):
            if self.drive == 'curiosity':
                score = self.motivation.curiosity(obs, a)
            elif self.drive == 'novelty':
                score = self.motivation.novelty(obs, a)
            elif self.drive == 'empowerment':
                score = self.motivation.empowerment(obs, a)
            elif self.drive == 'information_gain':
                score = self.motivation.information_gain(obs, a)
            else:  # random
                score = random.random()
            score += np.random.normal(0, 0.01)
            scores.append(score)

        return int(np.argmax(scores))

    def learn(self, obs, action, next_obs):
        """从经验中学习"""
        pe = self.motivation.update(obs, action, next_obs)

        # 发现结构：状态变化超过阈值
        delta = float(np.linalg.norm(next_obs - obs))
        if delta > 0.5:
            structure_key = (int(delta * 10), action)
            self.discovered_structures.add(structure_key)

        # 覆盖率追踪
        state_key = tuple((next_obs * 2).astype(int))
        self.state_coverage.add(state_key)

        return pe


def _make_scene_obs():
    """从语言场景生成观测向量"""
    scene = generate_rich_scene()
    # 将场景对象转换为观测向量
    obs_parts = []
    for obj in scene:
        for v in obj.values():
            if isinstance(v, (int, float)):
                obs_parts.append(float(v))
            elif isinstance(v, str):
                obs_parts.append(float(hash(v) % 100) / 100.0)
    flat = np.array(obs_parts) if obs_parts else np.zeros(20)
    if len(flat) < 20:
        flat = np.concatenate([flat, np.zeros(20 - len(flat))])
    return flat[:20]


# ============================================================
# 实验
# ============================================================

def experiment_1_drive_comparison():
    """实验 1：5 种内驱力对比（500 步 x 10 次）"""
    print("=" * 60)
    print("实验 1：内驱力对比（好奇/新奇/掌控/信息增益/随机）")
    print("=" * 60)

    drives = ['curiosity', 'novelty', 'empowerment', 'information_gain', 'random']
    results = {}

    for drive in drives:
        run_pes = []
        run_coverage = []
        run_structures = []

        for run in range(10):
            agent = OpenEndedAgent(obs_dim=20, action_dim=8, drive=drive)
            pes = []

            for step in range(500):
                obs = _make_scene_obs()
                action = agent.act(obs)

                # 模拟动作效果
                effect = np.random.randn(20) * 0.1
                effect[action % 20] += 0.3
                next_obs = obs + effect

                pe = agent.learn(obs, action, next_obs)
                pes.append(pe)

            run_pes.append(np.mean(pes[-100:]))
            run_coverage.append(len(agent.state_coverage))
            run_structures.append(len(agent.discovered_structures))

        results[drive] = {
            'avg_error': round(float(np.mean(run_pes)), 4),
            'avg_coverage': round(float(np.mean(run_coverage)), 1),
            'avg_structures': round(float(np.mean(run_structures)), 1),
        }
        print(f"\n  {drive}:")
        print(f"    误差: {np.mean(run_pes):.4f}")
        print(f"    状态覆盖: {np.mean(run_coverage):.0f}")
        print(f"    发现结构: {np.mean(run_structures):.0f}")

    return results


def experiment_2_discovery_rate():
    """实验 2：环境结构发现率（随时间变化）"""
    print("\n" + "=" * 60)
    print("实验 2：环境结构发现率（1000 步 x 3 次）")
    print("=" * 60)

    drives = ['curiosity', 'novelty', 'random']
    results = {}

    for drive in drives:
        all_trajectories = []

        for run in range(3):
            agent = OpenEndedAgent(obs_dim=20, action_dim=8, drive=drive)
            trajectory = []

            for step in range(1000):
                obs = _make_scene_obs()
                action = agent.act(obs)
                effect = np.random.randn(20) * 0.1
                effect[action % 20] += 0.3
                next_obs = obs + effect
                agent.learn(obs, action, next_obs)

                if (step + 1) % 100 == 0:
                    trajectory.append({
                        'step': step + 1,
                        'coverage': len(agent.state_coverage),
                        'structures': len(agent.discovered_structures),
                        'error': np.mean(list(agent.motivation.error_history)[-20:])
                            if agent.motivation.error_history else 0,
                    })

            all_trajectories.append(trajectory)

        # 平均
        avg_traj = []
        for i in range(len(all_trajectories[0])):
            avg_traj.append({
                'step': all_trajectories[0][i]['step'],
                'coverage': round(float(np.mean([t[i]['coverage'] for t in all_trajectories])), 1),
                'structures': round(float(np.mean([t[i]['structures'] for t in all_trajectories])), 1),
            })

        results[drive] = avg_traj
        final = avg_traj[-1]
        print(f"  {drive}: 最终覆盖={final['coverage']}, "
              f"结构={final['structures']}")

    return results


def experiment_3_language_emergence_no_pressure():
    """实验 3：无通信压力下的语言涌现（500 轮）"""
    print("\n" + "=" * 60)
    print("实验 3：无通信压力下的语言涌现")
    print("=" * 60)

    # 两种条件：有通信压力 vs 无通信压力
    conditions = {
        'with_pressure': True,
        'no_pressure': False,
    }
    results = {}

    for cond, has_pressure in conditions.items():
        run_stats = []

        for run in range(5):
            speaker = LanguageAgent(f'sp_{run}')
            listener = LanguageAgent(f'li_{run}')

            for r in range(500):
                scene = generate_rich_scene()
                target = random.randint(0, len(scene) - 1)

                if has_pressure:
                    # 正常通信游戏
                    cross_language_round(speaker, listener, scene, target)
                else:
                    # 无通信压力：只有 speaker 观察 scene，更新词汇
                    # 不执行 listener 匹配——词汇从观察中被动学习
                    for obj in scene:
                        for dim, val in obj.items():
                            sym = f"{dim}_{val}"
                            if sym not in speaker.language.vocabulary:
                                speaker.language.vocabulary[sym] = {
                                    'frequency': 1,
                                    'success_rate': 0.5,
                                    'contexts': [],
                                }
                            else:
                                speaker.language.vocabulary[sym]['frequency'] += 1

            stats = speaker.language.get_stats()
            run_stats.append(stats)

        results[cond] = {
            'vocabulary_size': round(float(np.mean([s['vocabulary_size'] for s in run_stats])), 1),
            'success_rate': round(float(np.mean([s['success_rate'] for s in run_stats])), 4),
            'combination_rate': round(float(np.mean([s['combination_rate'] for s in run_stats])), 4),
        }
        print(f"  {cond}: 词汇={results[cond]['vocabulary_size']}, "
              f"成功率={results[cond]['success_rate']:.1%}, "
              f"组合率={results[cond]['combination_rate']:.4f}")

    return results


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_drive_comparison()
    results['experiment_2'] = experiment_2_discovery_rate()
    results['experiment_3'] = experiment_3_language_emergence_no_pressure()

    with open('open_ended_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到 open_ended_results.json")
