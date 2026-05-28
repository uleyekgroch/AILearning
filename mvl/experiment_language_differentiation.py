"""
Phase 31 实验：大概念空间语言分化

5 个实验：
1. 概念空间对比：72 对象 vs 10,368 对象
2. 噪声效果对比：0%/10%/20%/30% 噪声
3. 区域化方言涌现：5 区域隔离 → 开放通信
4. 通用语涌现：区域化 + 小世界网络
5. 语言灭绝与复兴：区域化 + 无标度网络
"""

import sys
import random
import numpy as np
import json
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from language_emergence import (
    LanguageAgent, cross_language_round, compute_language_similarity,
    generate_rich_scene, EmergingLanguage
)
from language_rich_scene import (
    generate_rich_scene_v2, RegionConfig, RICH_ATTRIBUTES, ALL_ATTRIBUTE_NAMES
)
from language_noisy import NoisyLanguageAgent, noisy_cross_language_round
from language_regional import RegionalLanguageSociety
from language_society_large import LargeScaleLanguageSociety


# ============================================================
# 实验 1：概念空间对比
# ============================================================

def experiment_1_concept_space(num_agents: int = 100,
                               num_rounds: int = 2000,
                               verbose: bool = True) -> Dict:
    """
    概念空间对比：72 对象（当前）vs 10,368 对象（新）

    100 Agent，全连接，2000 轮
    预期：大空间收敛更慢，最终相似度更低
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：概念空间对比（72 vs 10,368 对象）")
        print("=" * 60)

    results = {}

    # A: 小概念空间（72 对象，当前系统）
    if verbose:
        print("\n--- A: 小概念空间（72 对象）---")
    society_a = LargeScaleLanguageSociety(num_agents, topology='full')
    log_a = {'rounds': [], 'success_rate': [], 'similarity': []}
    for r in range(num_rounds):
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)
        society_a.step(scene, target_idx)
        if (r + 1) % 200 == 0:
            sr = society_a.get_communication_success_rate(200)
            metrics = society_a.get_global_metrics(sample_size=100)
            log_a['rounds'].append(r + 1)
            log_a['success_rate'].append(sr)
            log_a['similarity'].append(metrics['avg_overall_similarity'])
            if verbose:
                print(f"  Round {r+1:5d}: success={sr:.3f}, "
                      f"similarity={metrics['avg_overall_similarity']:.3f}")
    results['small_space'] = log_a

    # B: 大概念空间（10,368 对象，4 属性）
    if verbose:
        print("\n--- B: 大概念空间（10,368 对象，4 属性）---")
    society_b = LargeScaleLanguageSociety(num_agents, topology='full')
    log_b = {'rounds': [], 'success_rate': [], 'similarity': []}
    for r in range(num_rounds):
        scene = generate_rich_scene_v2(num_objects=8, num_attributes=4)
        target_idx = random.randint(0, len(scene) - 1)
        society_b.step(scene, target_idx)
        if (r + 1) % 200 == 0:
            sr = society_b.get_communication_success_rate(200)
            metrics = society_b.get_global_metrics(sample_size=100)
            log_b['rounds'].append(r + 1)
            log_b['success_rate'].append(sr)
            log_b['similarity'].append(metrics['avg_overall_similarity'])
            if verbose:
                print(f"  Round {r+1:5d}: success={sr:.3f}, "
                      f"similarity={metrics['avg_overall_similarity']:.3f}")
    results['large_space'] = log_b

    # C: 大概念空间（6 属性）
    if verbose:
        print("\n--- C: 大概念空间（10,368 对象，6 属性）---")
    society_c = LargeScaleLanguageSociety(num_agents, topology='full')
    log_c = {'rounds': [], 'success_rate': [], 'similarity': []}
    for r in range(num_rounds):
        scene = generate_rich_scene_v2(num_objects=8, num_attributes=6)
        target_idx = random.randint(0, len(scene) - 1)
        society_c.step(scene, target_idx)
        if (r + 1) % 200 == 0:
            sr = society_c.get_communication_success_rate(200)
            metrics = society_c.get_global_metrics(sample_size=100)
            log_c['rounds'].append(r + 1)
            log_c['success_rate'].append(sr)
            log_c['similarity'].append(metrics['avg_overall_similarity'])
            if verbose:
                print(f"  Round {r+1:5d}: success={sr:.3f}, "
                      f"similarity={metrics['avg_overall_similarity']:.3f}")
    results['large_space_6attr'] = log_c

    if verbose:
        print(f"\n概念空间对比结果:")
        for name, log in results.items():
            print(f"  {name:20s}: final_similarity={log['similarity'][-1]:.3f}, "
                  f"final_success={log['success_rate'][-1]:.3f}")

    return results


# ============================================================
# 实验 2：噪声效果对比
# ============================================================

def experiment_2_noise_effect(num_agents: int = 100,
                              num_rounds: int = 2000,
                              verbose: bool = True) -> Dict:
    """
    噪声效果对比：0%/10%/20%/30% 噪声

    100 Agent，小世界网络，大概念空间，2000 轮
    预期：噪声越大，收敛越慢，方言越多样
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：噪声效果对比")
        print("=" * 60)

    results = {}

    for noise_rate in [0.0, 0.1, 0.2, 0.3]:
        if verbose:
            print(f"\n--- 噪声率 {noise_rate:.0%} ---")

        society = RegionalLanguageSociety(
            num_agents=num_agents,
            num_regions=1,  # 单区域，纯测噪声效果
            topology='small_world',
            noise_rate=noise_rate,
            num_attributes=4,
        )

        log = {'rounds': [], 'success_rate': [], 'similarity': []}
        for r in range(num_rounds):
            society.step()
            if (r + 1) % 200 == 0:
                sr = society.get_communication_success_rate(200)
                metrics = society.get_global_metrics(sample_size=100)
                log['rounds'].append(r + 1)
                log['success_rate'].append(sr)
                log['similarity'].append(metrics['avg_overall_similarity'])
                if verbose:
                    print(f"  Round {r+1:5d}: success={sr:.3f}, "
                          f"similarity={metrics['avg_overall_similarity']:.3f}")

        results[f'noise_{int(noise_rate*100)}'] = log

    if verbose:
        print(f"\n噪声效果对比结果:")
        for name, log in results.items():
            print(f"  {name:12s}: final_similarity={log['similarity'][-1]:.3f}, "
                  f"final_success={log['success_rate'][-1]:.3f}")

    return results


# ============================================================
# 实验 3：区域化方言涌现
# ============================================================

def experiment_3_regional_dialect(num_agents: int = 100,
                                  num_regions: int = 5,
                                  isolation_rounds: int = 1000,
                                  open_rounds: int = 2000,
                                  verbose: bool = True) -> Dict:
    """
    区域化方言涌现实验

    前 isolation_rounds 轮：区域隔离通信（不同环境分布）
    后 open_rounds 轮：开放跨区域通信
    观察方言分化和融合
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：区域化方言涌现")
        print("=" * 60)

    society = RegionalLanguageSociety(
        num_agents=num_agents,
        num_regions=num_regions,
        topology='groups',  # 区域隔离
        noise_rate=0.1,
        num_attributes=4,
    )

    log = {
        'rounds': [],
        'num_families': [],
        'dialect_divergence': [],
        'intra_similarity': [],
        'inter_similarity': [],
        'success_rate': [],
    }

    # 阶段 1：区域隔离
    if verbose:
        print(f"\n阶段 1：区域隔离（{isolation_rounds} 轮）")

    for r in range(isolation_rounds):
        society.step()
        if (r + 1) % 200 == 0:
            sr = society.get_communication_success_rate(200)
            div = society.get_dialect_divergence(sample_size=100)
            regional = society.get_regional_metrics(sample_size=50)
            cross = society.get_cross_region_metrics(sample_size=100)
            families = society.detect_language_families(threshold=0.5, sample_size=200)

            avg_intra = np.mean([v['avg_intra_similarity'] for v in regional.values()])

            log['rounds'].append(r + 1)
            log['num_families'].append(len(families))
            log['dialect_divergence'].append(div)
            log['intra_similarity'].append(avg_intra)
            log['inter_similarity'].append(cross['avg_cross_region_similarity'])
            log['success_rate'].append(sr)

            if verbose:
                print(f"  Round {r+1:5d}: families={len(families)}, "
                      f"divergence={div:.3f}, "
                      f"intra={avg_intra:.3f}, "
                      f"inter={cross['avg_cross_region_similarity']:.3f}")

    # 阶段 2：开放通信
    if verbose:
        print(f"\n阶段 2：开放跨区域通信（{open_rounds} 轮）")

    society.merge_regions()

    for r in range(open_rounds):
        society.step()
        if (r + 1) % 200 == 0:
            total_round = isolation_rounds + r + 1
            sr = society.get_communication_success_rate(200)
            metrics = society.get_global_metrics(sample_size=200)
            families = society.detect_language_families(threshold=0.5, sample_size=200)

            log['rounds'].append(total_round)
            log['num_families'].append(len(families))
            log['dialect_divergence'].append(0.0)
            log['intra_similarity'].append(metrics['avg_overall_similarity'])
            log['inter_similarity'].append(metrics['avg_overall_similarity'])
            log['success_rate'].append(sr)

            if verbose:
                print(f"  Round {total_round:5d}: families={len(families)}, "
                      f"similarity={metrics['avg_overall_similarity']:.3f}")

    final_families = society.detect_language_families(threshold=0.5, sample_size=200)
    vocab_div = society.get_vocabulary_diversity()

    if verbose:
        print(f"\n方言涌现结果:")
        print(f"  初始区域数: {num_regions}")
        print(f"  最终语言家族数: {len(final_families)}")
        for i, family in enumerate(final_families):
            print(f"  家族 {i}: {len(family)} 个 agent")
        print(f"\n各区域词汇多样性:")
        for region, info in vocab_div.items():
            print(f"  {region}: {info['vocabulary_size']} 个符号")

    return {
        'log': log,
        'final_families': final_families,
        'vocabulary_diversity': vocab_div,
    }


# ============================================================
# 实验 4：通用语涌现
# ============================================================

def experiment_4_lingua_franca(num_agents: int = 100,
                               num_regions: int = 5,
                               num_rounds: int = 5000,
                               check_interval: int = 500,
                               verbose: bool = True) -> Dict:
    """
    通用语涌现实验

    区域化 + 小世界网络，5000 轮
    每 500 轮检测通用语
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 4：通用语涌现")
        print("=" * 60)

    society = RegionalLanguageSociety(
        num_agents=num_agents,
        num_regions=num_regions,
        topology='small_world',
        noise_rate=0.15,
        num_attributes=4,
    )

    log = {
        'rounds': [],
        'success_rate': [],
        'avg_similarity': [],
        'dialect_divergence': [],
        'lingua_franca': [],
        'num_families': [],
    }

    for r in range(num_rounds):
        society.step()
        if (r + 1) % check_interval == 0:
            sr = society.get_communication_success_rate(check_interval)
            metrics = society.get_global_metrics(sample_size=200)
            div = society.get_dialect_divergence(sample_size=100)
            lf = society.detect_lingua_franca(cross_group_threshold=0.5)
            families = society.detect_language_families(threshold=0.5, sample_size=200)

            log['rounds'].append(r + 1)
            log['success_rate'].append(sr)
            log['avg_similarity'].append(metrics['avg_overall_similarity'])
            log['dialect_divergence'].append(div)
            log['lingua_franca'].append(lf)
            log['num_families'].append(len(families))

            if verbose:
                lf_str = lf if lf else "None"
                print(f"  Round {r+1:5d}: success={sr:.3f}, "
                      f"similarity={metrics['avg_overall_similarity']:.3f}, "
                      f"divergence={div:.3f}, "
                      f"lingua_franca={lf_str}, "
                      f"families={len(families)}")

    final_lf = society.detect_lingua_franca(cross_group_threshold=0.4)
    final_families = society.detect_language_families(threshold=0.5, sample_size=200)

    if verbose:
        print(f"\n通用语结果:")
        print(f"  最终通用语: {final_lf if final_lf else 'None'}")
        print(f"  最终语言家族数: {len(final_families)}")
        lf_appearances = sum(1 for lf in log['lingua_franca'] if lf is not None)
        print(f"  通用语出现次数: {lf_appearances}/{len(log['lingua_franca'])}")

    return {
        'log': log,
        'final_lingua_franca': final_lf,
        'final_families': final_families,
    }


# ============================================================
# 实验 5：语言灭绝与复兴
# ============================================================

def experiment_5_extinction_revival(num_agents: int = 100,
                                    num_regions: int = 5,
                                    num_rounds: int = 5000,
                                    check_interval: int = 500,
                                    verbose: bool = True) -> Dict:
    """
    语言灭绝与复兴实验

    区域化 + 无标度网络，5000 轮
    每 500 轮：移除最小语言家族的 10% agent
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 5：语言灭绝与复兴")
        print("=" * 60)

    society = RegionalLanguageSociety(
        num_agents=num_agents,
        num_regions=num_regions,
        topology='scale_free',
        noise_rate=0.15,
        num_attributes=4,
    )

    log = {
        'rounds': [],
        'success_rate': [],
        'avg_similarity': [],
        'diversity_index': [],
        'extinct_count': [],
        'num_families': [],
        'total_replaced': [],
    }

    total_replaced = 0

    for r in range(num_rounds):
        society.step()
        if (r + 1) % check_interval == 0:
            sr = society.get_communication_success_rate(check_interval)
            metrics = society.get_global_metrics(sample_size=200)
            diversity = society.get_language_diversity_index()
            extinct = society.find_extinct_candidates(threshold=0.25)
            families = society.detect_language_families(threshold=0.5, sample_size=200)

            log['rounds'].append(r + 1)
            log['success_rate'].append(sr)
            log['avg_similarity'].append(metrics['avg_overall_similarity'])
            log['diversity_index'].append(diversity)
            log['extinct_count'].append(len(extinct))
            log['num_families'].append(len(families))
            log['total_replaced'].append(total_replaced)

            if verbose:
                print(f"  Round {r+1:5d}: success={sr:.3f}, "
                      f"similarity={metrics['avg_overall_similarity']:.3f}, "
                      f"diversity={diversity:.3f}, "
                      f"extinct={len(extinct)}, "
                      f"families={len(families)}, "
                      f"replaced={total_replaced}")

            # 移除最弱的 agent，补充新 agent
            if extinct:
                num_replace = max(1, len(extinct) // 3)
                to_replace = random.sample(extinct, min(num_replace, len(extinct)))
                society.replace_agents(to_replace)
                total_replaced += len(to_replace)

                if verbose:
                    print(f"    → 替换 {len(to_replace)} 个 agent (累计: {total_replaced})")

    final_diversity = society.get_language_diversity_index()
    final_families = society.detect_language_families(threshold=0.5, sample_size=200)

    if verbose:
        print(f"\n灭绝与复兴结果:")
        print(f"  总替换 agent 数: {total_replaced}")
        print(f"  最终多样性指数: {final_diversity:.3f}")
        print(f"  最终语言家族数: {len(final_families)}")

    return {
        'log': log,
        'total_replaced': total_replaced,
        'final_diversity': final_diversity,
        'final_families': final_families,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Phase 31: 大概念空间语言分化实验")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    results = {}

    # 实验 1：概念空间对比
    results['exp1'] = experiment_1_concept_space(
        num_agents=100, num_rounds=2000, verbose=True
    )

    # 实验 2：噪声效果对比
    results['exp2'] = experiment_2_noise_effect(
        num_agents=100, num_rounds=2000, verbose=True
    )

    # 实验 3：区域化方言涌现
    results['exp3'] = experiment_3_regional_dialect(
        num_agents=100, num_regions=5,
        isolation_rounds=1000, open_rounds=2000, verbose=True
    )

    # 实验 4：通用语涌现
    results['exp4'] = experiment_4_lingua_franca(
        num_agents=100, num_regions=5,
        num_rounds=5000, verbose=True
    )

    # 实验 5：语言灭绝与复兴
    results['exp5'] = experiment_5_extinction_revival(
        num_agents=100, num_regions=5,
        num_rounds=5000, verbose=True
    )

    # 保存结果
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, set):
            return sorted(list(obj))
        return obj

    with open('language_differentiation_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 language_differentiation_results.json")

    return results


if __name__ == '__main__':
    main()
