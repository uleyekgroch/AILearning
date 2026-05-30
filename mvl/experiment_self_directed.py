"""
Phase 51 实验：自主目标设定（Agent 自己决定学什么）

3 个实验：
1. 自主 vs 随机 vs 均匀学习（300 轮，5 对 Agent）
2. 知识差距恢复（200 轮）
3. 目标适应性（300 轮，追踪目标选择变化）
"""

import sys
import time
import random
import numpy as np
import json

sys.stdout.reconfigure(encoding='utf-8')

from language_emergence import LanguageAgent, cross_language_round
from language_rich_scene import (
    generate_rich_scene_v2, ALL_ATTRIBUTE_NAMES, RICH_ATTRIBUTES
)
from self_directed_learning import (
    KnowledgeAssessor, GoalSelector, SelfDirectedLearner,
    run_random_learning, run_uniform_learning
)


# ============================================================
# 实验 1：自主 vs 随机 vs 均匀
# ============================================================

def experiment_1_comparison():
    """实验 1：三种学习策略对比"""
    print("\n" + "=" * 60)
    print("实验 1：自主 vs 随机 vs 均匀学习（300 轮 x 5 对）")
    print("=" * 60)

    num_pairs = 5
    num_rounds = 300
    strategies = ['self_directed', 'random', 'uniform']
    all_results = {s: [] for s in strategies}

    for pair_idx in range(num_pairs):
        print(f"\n--- Agent 对 {pair_idx + 1}/{num_pairs} ---")

        for strategy in strategies:
            # 每个策略用新的 Agent 对（公平比较）
            speaker = LanguageAgent(f"{strategy}_s_{pair_idx}")
            listener = LanguageAgent(f"{strategy}_l_{pair_idx}")

            if strategy == 'self_directed':
                learner = SelfDirectedLearner(speaker, listener)
                result = learner.run_session(num_rounds, log_interval=50)
            elif strategy == 'random':
                result = run_random_learning(speaker, listener, num_rounds,
                                             log_interval=50)
            else:  # uniform
                result = run_uniform_learning(speaker, listener, num_rounds,
                                              log_interval=50)

            all_results[strategy].append(result)
            print(f"  {strategy:15s}: 成功率={result['success_rate']:.3f}, "
                  f"词汇量={result['vocabulary_size']}")

    # 汇总统计
    print("\n  汇总:")
    for strategy in strategies:
        rates = [r['success_rate'] for r in all_results[strategy]]
        vocabs = [r['vocabulary_size'] for r in all_results[strategy]]
        print(f"  {strategy:15s}: 成功率={np.mean(rates):.3f} ± {np.std(rates):.3f}, "
              f"词汇量={np.mean(vocabs):.1f} ± {np.std(vocabs):.1f}")

    # 维度覆盖率对比
    print("\n  维度覆盖率（self_directed 最后一个 Agent 的知识状态）:")
    last_sd = all_results['self_directed'][-1]
    if 'knowledge_snapshots' in last_sd:
        last_snap = last_sd['knowledge_snapshots']
        if last_snap:
            last_round = max(last_snap.keys(), key=lambda x: int(x))
            summary = last_snap[last_round]
            for attr, info in summary.items():
                print(f"    {attr:12s}: {info['status']:8s} "
                      f"(freq={info['frequency']}, sr={info['success_rate']})")

    return all_results


# ============================================================
# 实验 2：知识差距恢复
# ============================================================

def experiment_2_gap_recovery():
    """实验 2：知识差距恢复"""
    print("\n" + "=" * 60)
    print("实验 2：知识差距恢复（200 轮）")
    print("=" * 60)

    num_pairs = 5
    biased_attrs = ['color', 'shape', 'size']  # 只用 3 个属性
    missing_attrs = [a for a in ALL_ATTRIBUTE_NAMES if a not in biased_attrs]

    results = {'self_directed': [], 'random': []}

    for pair_idx in range(num_pairs):
        print(f"\n--- Agent 对 {pair_idx + 1}/{num_pairs} ---")

        for strategy in ['self_directed', 'random']:
            # 阶段 1：偏置训练（100 轮，只用 3 个属性）
            speaker = LanguageAgent(f"{strategy}_gap_s_{pair_idx}")
            listener = LanguageAgent(f"{strategy}_gap_l_{pair_idx}")

            for r in range(100):
                scene = generate_rich_scene_v2(
                    num_objects=8, attribute_names=biased_attrs)
                target_idx = random.randint(0, len(scene) - 1)
                cross_language_round(speaker, listener, scene, target_idx)

            # 记录偏置训练后的知识状态
            assessor_before = KnowledgeAssessor(speaker.language)
            before_summary = assessor_before.get_knowledge_summary()
            before_covered = sum(1 for a in ALL_ATTRIBUTE_NAMES
                                 if before_summary[a]['status'] != 'unseen')

            # 阶段 2：恢复训练（100 轮）
            if strategy == 'self_directed':
                learner = SelfDirectedLearner(speaker, listener)
                phase2 = learner.run_session(100, log_interval=50)
            else:
                phase2 = run_random_learning(speaker, listener, 100,
                                             log_interval=50)

            # 记录恢复后的知识状态
            assessor_after = KnowledgeAssessor(speaker.language)
            after_summary = assessor_after.get_knowledge_summary()
            after_covered = sum(1 for a in ALL_ATTRIBUTE_NAMES
                                if after_summary[a]['status'] != 'unseen')

            results[strategy].append({
                'before_covered': before_covered,
                'after_covered': after_covered,
                'recovery': after_covered - before_covered,
                'phase2_success': phase2['success_rate'],
            })

            print(f"  {strategy:15s}: 覆盖 {before_covered}/10 → {after_covered}/10 "
                  f"(+{after_covered - before_covered}), "
                  f"阶段2成功率={phase2['success_rate']:.3f}")

    # 汇总
    print("\n  汇总:")
    for strategy in ['self_directed', 'random']:
        recoveries = [r['recovery'] for r in results[strategy]]
        print(f"  {strategy:15s}: 平均恢复 +{np.mean(recoveries):.1f} 维度 "
              f"(±{np.std(recoveries):.1f})")

    return results


# ============================================================
# 实验 3：目标适应性
# ============================================================

def experiment_3_adaptive_goals():
    """实验 3：目标选择随时间变化"""
    print("\n" + "=" * 60)
    print("实验 3：目标适应性（300 轮）")
    print("=" * 60)

    speaker = LanguageAgent("adaptive_s")
    listener = LanguageAgent("adaptive_l")
    learner = SelfDirectedLearner(speaker, listener, epsilon=0.05)

    # 分阶段运行并记录目标
    goal_snapshots = {}
    knowledge_snapshots = {}

    for checkpoint in [50, 100, 150, 200, 250, 300]:
        # 运行到下一个检查点
        rounds_to_run = checkpoint - (checkpoint - 50 if checkpoint > 50 else 0)
        for r in range(50):
            learner.run_round()

        # 记录目标选择
        recent_goals = learner.goal_selector.goal_history[-50:]
        goal_counts = {}
        for goals in recent_goals:
            for g in goals:
                goal_counts[g] = goal_counts.get(g, 0) + 1

        goal_snapshots[checkpoint] = goal_counts

        # 记录知识状态
        assessor = KnowledgeAssessor(speaker.language)
        knowledge_snapshots[checkpoint] = assessor.get_knowledge_summary()

        print(f"\n  Round {checkpoint}:")
        print(f"    Top 3 目标: {sorted(goal_counts.items(), key=lambda x: x[1], reverse=True)[:3]}")
        print(f"    词汇量: {len(speaker.language.vocabulary)}")

        # 显示维度状态
        summary = knowledge_snapshots[checkpoint]
        statuses = {'mastered': [], 'learning': [], 'weak': [], 'unseen': []}
        for attr, info in summary.items():
            statuses[info['status']].append(attr)
        for status, attrs in statuses.items():
            if attrs:
                print(f"    {status}: {attrs}")

    # 分析目标转移
    print("\n  目标转移分析:")
    print(f"    Round 50 的 Top 3 目标: "
          f"{sorted(goal_snapshots[50].items(), key=lambda x: x[1], reverse=True)[:3]}")
    print(f"    Round 300 的 Top 3 目标: "
          f"{sorted(goal_snapshots[300].items(), key=lambda x: x[1], reverse=True)[:3]}")

    # 计算目标变化度
    early_goals = set(g for goals in learner.goal_selector.goal_history[:50] for g in goals)
    late_goals = set(g for goals in learner.goal_selector.goal_history[-50:] for g in goals)
    overlap = early_goals & late_goals
    print(f"    早期目标: {sorted(early_goals)}")
    print(f"    后期目标: {sorted(late_goals)}")
    print(f"    重叠: {sorted(overlap)}")

    return {
        'goal_snapshots': {str(k): v for k, v in goal_snapshots.items()},
        'knowledge_snapshots': {str(k): v for k, v in knowledge_snapshots.items()},
    }


if __name__ == '__main__':
    all_results = {}

    all_results['comparison'] = experiment_1_comparison()
    all_results['gap_recovery'] = experiment_2_gap_recovery()
    all_results['adaptive_goals'] = experiment_3_adaptive_goals()

    with open('self_directed_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print("\n\n结果已保存到 self_directed_results.json")
