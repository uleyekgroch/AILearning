"""
Phase 52: 课程涌现 — 从低歧义到高歧义的自发课程

实验设计：
1. 5 种策略对比（歧义度控制 vs 随机/固定）
2. 自主节奏轨迹分析
3. 前置知识：先学低歧义场景是否有益？
4. 长期收敛（1000 轮）
"""

import json
import random
import numpy as np
from language_emergence import LanguageAgent, cross_language_round
from curriculum_learning import (
    run_curriculum, generate_ambiguous_scene, compute_ambiguity,
    count_unique_objects, AMBIGUITY_LEVELS
)


def experiment_1_comparison():
    """实验 1：5 种策略对比（500 轮 x 5 对 Agent）"""
    print("=" * 60)
    print("实验 1：歧义度策略对比（500 轮 x 5 对 Agent）")
    print("=" * 60)

    # 先展示歧义等级参数
    print("\n歧义等级定义:")
    for lv, params in AMBIGUITY_LEVELS.items():
        unique = params['total_attrs'] - params['num_shared']
        print(f"  等级 {lv}: {params['num_objects']}物体, "
              f"{params['total_attrs']}属性, "
              f"{params['num_shared']}共享 → "
              f"{unique} 有效区分维度")

    strategies = ['random', 'easy', 'hard', 'progressive', 'self_paced']
    all_results = {}

    for strategy in strategies:
        run_results = []
        for run in range(5):
            speaker = LanguageAgent(f'sp_{run}')
            listener = LanguageAgent(f'li_{run}')
            result = run_curriculum(speaker, listener, 500, strategy)
            run_results.append(result)

        avg_sr = np.mean([r['success_rate'] for r in run_results])
        std_sr = np.std([r['success_rate'] for r in run_results])
        avg_vocab = np.mean([r['vocabulary_size'] for r in run_results])
        avg_amb = np.mean([r['avg_ambiguity'] for r in run_results])

        all_results[strategy] = {
            'success_rate': round(float(avg_sr), 4),
            'std': round(float(std_sr), 4),
            'vocabulary_size': round(float(avg_vocab), 1),
            'avg_ambiguity': round(float(avg_amb), 4),
        }

        print(f"\n{strategy}:")
        print(f"  成功率: {avg_sr:.1%} +/- {std_sr:.1%}")
        print(f"  词汇量: {avg_vocab:.0f}")
        print(f"  平均歧义度: {avg_amb:.4f}")

    ranked = sorted(all_results.items(), key=lambda x: x[1]['success_rate'],
                    reverse=True)
    print(f"\n排名:")
    for i, (s, r) in enumerate(ranked):
        print(f"  {i+1}. {s}: {r['success_rate']:.1%}")

    return all_results


def experiment_2_self_paced():
    """实验 2：自主节奏的歧义度轨迹（500 轮 x 3 次）"""
    print("\n" + "=" * 60)
    print("实验 2：自主节奏轨迹（500 轮 x 3 次）")
    print("=" * 60)

    all_rates = []
    all_final_levels = []

    for run in range(3):
        speaker = LanguageAgent(f'sp_{run}')
        listener = LanguageAgent(f'li_{run}')
        result = run_curriculum(speaker, listener, 500, 'self_paced')

        all_rates.append(result['success_rate'])
        all_final_levels.append(result.get('avg_level', 1.0))

        print(f"\n运行 {run + 1}:")
        print(f"  成功率: {result['success_rate']:.1%}")
        print(f"  平均等级: {result.get('avg_level', 0):.1f}")
        print(f"  词汇量: {result['vocabulary_size']}")

    print(f"\n汇总:")
    print(f"  平均成功率: {np.mean(all_rates):.1%}")
    print(f"  平均最终等级: {np.mean(all_final_levels):.1f}")

    return {
        'avg_success_rate': round(float(np.mean(all_rates)), 4),
        'avg_final_level': round(float(np.mean(all_final_levels)), 2),
    }


def experiment_3_ambiguity_effect():
    """实验 3：歧义度对成功率的影响（每个等级 200 轮）"""
    print("\n" + "=" * 60)
    print("实验 3：歧义度对成功率的影响（每等级 200 轮 x 3 次）")
    print("=" * 60)

    level_results = {}

    for level in range(1, 8):
        run_rates = []
        run_unique = []
        for run in range(3):
            speaker = LanguageAgent(f'sp_{run}')
            listener = LanguageAgent(f'li_{run}')

            successes = 0
            for r in range(200):
                scene = generate_ambiguous_scene(
                    **AMBIGUITY_LEVELS[level]
                )
                target_idx = random.randint(0, len(scene) - 1)
                success = cross_language_round(
                    speaker, listener, scene, target_idx
                )
                if success:
                    successes += 1

            run_rates.append(successes / 200)

            # 测量场景的实际歧义度和唯一物体数
            sample_scene = generate_ambiguous_scene(
                **AMBIGUITY_LEVELS[level]
            )
            run_unique.append(count_unique_objects(sample_scene))

        avg_rate = np.mean(run_rates)
        avg_unique = np.mean(run_unique)
        params = AMBIGUITY_LEVELS[level]
        unique_attrs = params['total_attrs'] - params['num_shared']

        level_results[str(level)] = {
            'success_rate': round(float(avg_rate), 4),
            'unique_attrs': unique_attrs,
            'num_objects': params['num_objects'],
            'avg_unique_objects': round(float(avg_unique), 1),
        }

        print(f"\n等级 {level} ({params['num_objects']}物体, "
              f"{unique_attrs}有效维度):")
        print(f"  成功率: {avg_rate:.1%}")
        print(f"  场景唯一物体: {avg_unique:.1f}/{params['num_objects']}")

    return level_results


def experiment_4_curriculum_effect():
    """实验 4：课程效果——先低歧义后高歧义 vs 直接高歧义"""
    print("\n" + "=" * 60)
    print("实验 4：课程效果（低→高 vs 直接高 vs 高→低）")
    print("=" * 60)

    conditions = {
        'low_then_high': (list(range(1, 4)), list(range(4, 8))),
        'high_direct': (list(range(1, 8)), list(range(1, 8))),
        'high_then_low': (list(range(4, 8)), list(range(1, 4))),
        'random': None,
    }

    all_results = {}

    for cond, levels in conditions.items():
        run_rates = []
        for run in range(5):
            speaker = LanguageAgent(f'sp_{run}')
            listener = LanguageAgent(f'li_{run}')

            successes = 0
            total = 500

            for r in range(total):
                if cond == 'random':
                    level = random.randint(1, 7)
                elif r < 250:
                    level = random.choice(levels[0])
                else:
                    level = random.choice(levels[1])

                scene = generate_ambiguous_scene(
                    **AMBIGUITY_LEVELS[level]
                )
                target_idx = random.randint(0, len(scene) - 1)
                success = cross_language_round(
                    speaker, listener, scene, target_idx
                )
                if success:
                    successes += 1

            run_rates.append(successes / total)

        avg = np.mean(run_rates)
        std = np.std(run_rates)

        all_results[cond] = {
            'success_rate': round(float(avg), 4),
            'std': round(float(std), 4),
        }

        print(f"\n{cond}:")
        print(f"  成功率: {avg:.1%} +/- {std:.1%}")

    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]['success_rate'], reverse=True)
    print(f"\n排名:")
    for i, (s, r) in enumerate(ranked):
        print(f"  {i+1}. {s}: {r['success_rate']:.1%}")

    return all_results


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_comparison()
    results['experiment_2'] = experiment_2_self_paced()
    results['experiment_3'] = experiment_3_ambiguity_effect()
    results['experiment_4'] = experiment_4_curriculum_effect()

    with open('curriculum_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到 curriculum_results.json")
