"""
Phase 59: 关键期关闭机制 — 生物学启发的突触可塑性衰减

核心问题：
人类婴儿有语言学习的关键期（ puberty 前后关闭），
关键期关闭后学习效率显著下降。
能否用生物学启发的机制模拟关键期？

生物学基础：
1. 突触修剪（synaptic pruning）：青春期大量突触被修剪
2. 神经可塑性下降：GABA 抑制增强，LTP 阈值提高
3. 感觉剥夺实验（Hubel & Wiesel, 1970）：幼猫视觉剥夺后永久失明

实现方式：
- plasticity(epoch) = base_rate × decay(epoch)
- decay(epoch) 从 1.0 指数衰减到 floor
- 关键期窗口：高可塑性 → 逐渐关闭 → 低可塑性

测试：
1. 不同衰减速率对比（无衰减/慢衰减/快衰减/阶梯衰减）
2. 关键期关闭后学习新语言的能力
3. 关键期关闭后适应新环境的能力
4. 关键期重新打开（青春期后学习能力恢复实验）
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from language_emergence import LanguageAgent, cross_language_round, generate_rich_scene
from language_rich_scene import generate_rich_scene_v2, ALL_ATTRIBUTE_NAMES


# ============================================================
# 关键期可塑性衰减函数
# ============================================================

def plasticity_exponential(epoch: int, total_epochs: int,
                           base_rate: float = 1.0,
                           decay_rate: float = 0.01,
                           floor: float = 0.1) -> float:
    """指数衰减：平滑下降"""
    return max(floor, base_rate * np.exp(-decay_rate * epoch))


def plasticity_sigmoid(epoch: int, total_epochs: int,
                       midpoint: float = 0.5,
                       steepness: float = 10.0,
                       floor: float = 0.1) -> float:
    """Sigmoid 衰减：关键期突然关闭（模拟青春期）"""
    t = epoch / max(total_epochs, 1)
    return max(floor, 1.0 - 1.0 / (1.0 + np.exp(-steepness * (t - midpoint))))


def plasticity_linear(epoch: int, total_epochs: int,
                      floor: float = 0.1) -> float:
    """线性衰减"""
    t = epoch / max(total_epochs, 1)
    return max(floor, 1.0 - t * (1.0 - floor))


def plasticity_step(epoch: int, total_epochs: int,
                    close_at: float = 0.6,
                    floor: float = 0.1) -> float:
    """阶梯衰减：在某个时间点突然关闭"""
    t = epoch / max(total_epochs, 1)
    return 1.0 if t < close_at else floor


def plasticity_none(epoch: int, total_epochs: int) -> float:
    """无衰减（对照）"""
    return 1.0


PLASTICITY_SCHEDULES = {
    'none': plasticity_none,
    'exponential': plasticity_exponential,
    'sigmoid': plasticity_sigmoid,
    'linear': plasticity_linear,
    'step': plasticity_step,
}


# ============================================================
# 关键期学习 Agent
# ============================================================

class CriticalPeriodAgent(LanguageAgent):
    """带关键期衰减的语言学习 Agent"""

    def __init__(self, agent_id: str,
                 plasticity_fn=None,
                 floor: float = 0.1):
        super().__init__(agent_id)
        self.plasticity_fn = plasticity_fn or plasticity_none
        self.floor = floor
        self.epoch = 0
        self.total_epochs = 1000
        self.plasticity_history = []

    def get_plasticity(self) -> float:
        """获取当前可塑性"""
        p = self.plasticity_fn(self.epoch, self.total_epochs)
        p = max(self.floor, p)
        self.plasticity_history.append(p)
        return p

    def update_from_communication(self, utterance, success):
        """
        覆盖更新方法：按可塑性缩放学习量

        可塑性高 → 符号频率更新大
        可塑性低 → 符号频率更新小
        """
        p = self.get_plasticity()

        # 只以 plasticity 概率真正学习
        if random.random() > p:
            return  # 跳过此次学习

        super().update_from_communication(utterance, success)
        self.epoch += 1


# ============================================================
# 实验
# ============================================================

def experiment_1_schedule_comparison():
    """实验 1：5 种可塑性衰减策略对比（500 轮 x 10 次）"""
    print("=" * 60)
    print("实验 1：可塑性衰减策略对比（500 轮 x 10 次）")
    print("=" * 60)

    results = {}

    for schedule_name, schedule_fn in PLASTICITY_SCHEDULES.items():
        run_stats = []

        for run in range(10):
            speaker = CriticalPeriodAgent(
                f'sp_{run}',
                plasticity_fn=schedule_fn,
                floor=0.1
            )
            speaker.total_epochs = 500
            listener = CriticalPeriodAgent(
                f'li_{run}',
                plasticity_fn=schedule_fn,
                floor=0.1
            )
            listener.total_epochs = 500

            for r in range(500):
                scene = generate_rich_scene()
                target = random.randint(0, len(scene) - 1)
                cross_language_round(speaker, listener, scene, target)

            stats = speaker.language.get_stats()
            stats['final_plasticity'] = speaker.get_plasticity()
            run_stats.append(stats)

        results[schedule_name] = {
            'avg_sr': round(float(np.mean([s['success_rate'] for s in run_stats])), 4),
            'avg_vocab': round(float(np.mean([s['vocabulary_size'] for s in run_stats])), 1),
            'avg_combo': round(float(np.mean([s['combination_rate'] for s in run_stats])), 4),
            'avg_grammar': round(float(np.mean([s['grammar_rules'] for s in run_stats])), 1),
        }
        print(f"\n  {schedule_name}:")
        print(f"    成功率={results[schedule_name]['avg_sr']:.1%}, "
              f"词汇={results[schedule_name]['avg_vocab']:.0f}, "
              f"组合={results[schedule_name]['avg_combo']:.2%}, "
              f"语法={results[schedule_name]['avg_grammar']:.1f}")

    ranked = sorted(results.items(), key=lambda x: x[1]['avg_sr'], reverse=True)
    print(f"\n  排名:")
    for i, (s, r) in enumerate(ranked):
        print(f"    {i+1}. {s}: {r['avg_sr']:.1%}")

    return results


def experiment_2_floor_effect():
    """实验 2：最低可塑性（floor）对后期学习的影响"""
    print("\n" + "=" * 60)
    print("实验 2：最低可塑性 floor 效果（500 轮 x 5 次）")
    print("=" * 60)

    floors = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    results = {}

    for floor in floors:
        run_stats = []

        for run in range(5):
            speaker = CriticalPeriodAgent(
                f'sp_{run}',
                plasticity_fn=plasticity_sigmoid,
                floor=floor
            )
            speaker.total_epochs = 500
            listener = CriticalPeriodAgent(
                f'li_{run}',
                plasticity_fn=plasticity_sigmoid,
                floor=floor
            )
            listener.total_epochs = 500

            for r in range(500):
                scene = generate_rich_scene()
                target = random.randint(0, len(scene) - 1)
                cross_language_round(speaker, listener, scene, target)

            stats = speaker.language.get_stats()
            run_stats.append(stats)

        results[str(floor)] = {
            'avg_sr': round(float(np.mean([s['success_rate'] for s in run_stats])), 4),
            'avg_vocab': round(float(np.mean([s['vocabulary_size'] for s in run_stats])), 1),
            'avg_combo': round(float(np.mean([s['combination_rate'] for s in run_stats])), 4),
        }
        print(f"  floor={floor}: SR={np.mean([s['success_rate'] for s in run_stats]):.1%}, "
              f"词汇={np.mean([s['vocabulary_size'] for s in run_stats]):.0f}")

    return results


def experiment_3_new_language_after_close():
    """实验 3：关键期关闭后学习新语言（预训练 300 + 新环境 200 轮）"""
    print("\n" + "=" * 60)
    print("实验 3：关键期关闭后学习新语言（300+200 轮 x 5 次）")
    print("=" * 60)

    conditions = {
        'no_critical_period': plasticity_none,
        'sigmoid_close': plasticity_sigmoid,
        'step_close': plasticity_step,
    }

    results = {}

    for cond, schedule_fn in conditions.items():
        run_stats = []

        for run in range(5):
            agent = CriticalPeriodAgent(f'sp_{run}', plasticity_fn=schedule_fn, floor=0.1)
            agent.total_epochs = 500
            partner = CriticalPeriodAgent(f'li_{run}', plasticity_fn=schedule_fn, floor=0.1)
            partner.total_epochs = 500

            # 阶段 1：基础语言学习（300 轮）
            for r in range(300):
                scene = generate_rich_scene()
                target = random.randint(0, len(scene) - 1)
                cross_language_round(agent, partner, scene, target)

            stats_before = agent.language.get_stats()

            # 阶段 2：新环境（200 轮，用不同的属性组合）
            for r in range(200):
                attrs = random.sample(ALL_ATTRIBUTE_NAMES, 4)
                scene = generate_rich_scene_v2(num_objects=random.randint(4, 6),
                                               attribute_names=attrs)
                target = random.randint(0, len(scene) - 1)
                cross_language_round(agent, partner, scene, target)

            stats_after = agent.language.get_stats()
            run_stats.append({
                'before_sr': stats_before['success_rate'],
                'before_vocab': stats_before['vocabulary_size'],
                'after_sr': stats_after['success_rate'],
                'after_vocab': stats_after['vocabulary_size'],
                'vocab_gain': stats_after['vocabulary_size'] - stats_before['vocabulary_size'],
            })

        avg_gain = np.mean([s['vocab_gain'] for s in run_stats])
        results[cond] = {
            'before_sr': round(float(np.mean([s['before_sr'] for s in run_stats])), 4),
            'after_sr': round(float(np.mean([s['after_sr'] for s in run_stats])), 4),
            'vocab_gain': round(float(avg_gain), 1),
        }
        print(f"  {cond}: SR {results[cond]['before_sr']:.1%}→"
              f"{results[cond]['after_sr']:.1%}, "
              f"词汇增长={avg_gain:.1f}")

    return results


def experiment_4_reopening():
    """实验 4：关键期重新打开（500 轮学习 + 200 轮关闭 + 300 轮重新打开）"""
    print("\n" + "=" * 60)
    print("实验 4：关键期重新打开（500+200+300 轮 x 3 次）")
    print("=" * 60)

    run_stats = []

    for run in range(3):
        agent = CriticalPeriodAgent(f'sp_{run}', plasticity_fn=plasticity_sigmoid, floor=0.05)
        agent.total_epochs = 1000
        partner = CriticalPeriodAgent(f'li_{run}', plasticity_fn=plasticity_sigmoid, floor=0.05)
        partner.total_epochs = 1000

        # 阶段 1：关键期开放（500 轮）
        for r in range(500):
            scene = generate_rich_scene()
            target = random.randint(0, len(scene) - 1)
            cross_language_round(agent, partner, scene, target)
        stats_phase1 = agent.language.get_stats()

        # 阶段 2：关键期关闭（200 轮，floor=0.05）
        agent.floor = 0.05
        partner.floor = 0.05
        for r in range(200):
            attrs = random.sample(ALL_ATTRIBUTE_NAMES, 4)
            scene = generate_rich_scene_v2(num_objects=random.randint(4, 6),
                                           attribute_names=attrs)
            target = random.randint(0, len(scene) - 1)
            cross_language_round(agent, partner, scene, target)
        stats_phase2 = agent.language.get_stats()

        # 阶段 3：关键期重新打开（300 轮，floor=1.0）
        agent.floor = 1.0
        partner.floor = 1.0
        for r in range(300):
            attrs = random.sample(ALL_ATTRIBUTE_NAMES, 4)
            scene = generate_rich_scene_v2(num_objects=random.randint(4, 6),
                                           attribute_names=attrs)
            target = random.randint(0, len(scene) - 1)
            cross_language_round(agent, partner, scene, target)
        stats_phase3 = agent.language.get_stats()

        run_stats.append({
            'phase1_sr': stats_phase1['success_rate'],
            'phase1_vocab': stats_phase1['vocabulary_size'],
            'phase2_sr': stats_phase2['success_rate'],
            'phase2_vocab': stats_phase2['vocabulary_size'],
            'phase3_sr': stats_phase3['success_rate'],
            'phase3_vocab': stats_phase3['vocabulary_size'],
        })

    results = {
        'phase1_open': {
            'sr': round(float(np.mean([s['phase1_sr'] for s in run_stats])), 4),
            'vocab': round(float(np.mean([s['phase1_vocab'] for s in run_stats])), 1),
        },
        'phase2_closed': {
            'sr': round(float(np.mean([s['phase2_sr'] for s in run_stats])), 4),
            'vocab': round(float(np.mean([s['phase2_vocab'] for s in run_stats])), 1),
        },
        'phase3_reopened': {
            'sr': round(float(np.mean([s['phase3_sr'] for s in run_stats])), 4),
            'vocab': round(float(np.mean([s['phase3_vocab'] for s in run_stats])), 1),
        },
    }

    print(f"  阶段 1 (开放): SR={results['phase1_open']['sr']:.1%}, "
          f"词汇={results['phase1_open']['vocab']:.0f}")
    print(f"  阶段 2 (关闭): SR={results['phase2_closed']['sr']:.1%}, "
          f"词汇={results['phase2_closed']['vocab']:.0f}")
    print(f"  阶段 3 (重新打开): SR={results['phase3_reopened']['sr']:.1%}, "
          f"词汇={results['phase3_reopened']['vocab']:.0f}")

    # 重新打开后的学习能力恢复
    vocab_gain_closed = results['phase2_closed']['vocab'] - results['phase1_open']['vocab']
    vocab_gain_reopened = results['phase3_reopened']['vocab'] - results['phase2_closed']['vocab']
    print(f"\n  关闭期间词汇增长: {vocab_gain_closed:.1f}")
    print(f"  重新打开后词汇增长: {vocab_gain_reopened:.1f}")
    print(f"  恢复倍率: {vocab_gain_reopened / max(vocab_gain_closed, 0.1):.1f}x")

    return results


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_schedule_comparison()
    results['experiment_2'] = experiment_2_floor_effect()
    results['experiment_3'] = experiment_3_new_language_after_close()
    results['experiment_4'] = experiment_4_reopening()

    with open('critical_period_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到 critical_period_results.json")
