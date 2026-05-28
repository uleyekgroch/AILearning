"""
Phase 30 实验：大规模社会语言演化

5 个实验：
1. 规模对比：10/50/100/200 Agent 的语言演化差异
2. 拓扑对比：全连接/小世界/无标度/线/星形
3. 语言家族涌现：隔离组 → 开放通信
4. 通用语涌现：随机拓扑下通用语的出现
5. 语言灭绝与复兴：移除弱势语言，观察新语言涌现
"""

import sys
import random
import numpy as np
import json
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from language_emergence import generate_rich_scene
from language_society_large import LargeScaleLanguageSociety


def run_society(society: LargeScaleLanguageSociety,
                num_rounds: int = 1000,
                complexity: str = 'medium',
                log_interval: int = 100,
                verbose: bool = True) -> Dict:
    """运行社会交流实验，记录指标"""
    log = {
        'rounds': [],
        'success_rate': [],
        'avg_similarity': [],
    }

    for r in range(num_rounds):
        scene = generate_rich_scene(complexity)
        target_idx = random.randint(0, len(scene) - 1)
        society.step(scene, target_idx)

        if (r + 1) % log_interval == 0:
            sr = society.get_communication_success_rate(log_interval)
            metrics = society.get_global_metrics(sample_size=100)
            log['rounds'].append(r + 1)
            log['success_rate'].append(sr)
            log['avg_similarity'].append(metrics['avg_overall_similarity'])

            if verbose:
                print(f"  Round {r+1:5d}: success={sr:.3f}, "
                      f"similarity={metrics['avg_overall_similarity']:.3f}")

    return log


# ============================================================
# 实验 1：规模对比（10/50/100/200 Agent）
# ============================================================

def experiment_1_scale_comparison(num_rounds: int = 1000,
                                  verbose: bool = True) -> Dict:
    """不同规模 Agent 群体的语言演化对比"""
    if verbose:
        print("\n" + "=" * 60)
        print("实验 1：规模对比（10/50/100/200 Agent）")
        print("=" * 60)

    results = {}
    for n in [10, 50, 100, 200]:
        if verbose:
            print(f"\n--- {n} Agents (全连接) ---")
        society = LargeScaleLanguageSociety(n, topology='full')
        log = run_society(society, num_rounds, verbose=verbose)
        final_metrics = society.get_global_metrics(sample_size=200)
        results[n] = {
            'log': log,
            'final_metrics': final_metrics,
            'topology_stats': society.get_topology_stats(),
        }

    if verbose:
        print(f"\n规模对比结果:")
        for n, r in results.items():
            fm = r['final_metrics']
            print(f"  {n:3d} Agent: similarity={fm['avg_overall_similarity']:.3f}, "
                  f"success={r['log']['success_rate'][-1]:.3f}")

    return results


# ============================================================
# 实验 2：拓扑对比（全连接/小世界/无标度/线/星）
# ============================================================

def experiment_2_topology_comparison(num_agents: int = 100,
                                     num_rounds: int = 2000,
                                     verbose: bool = True) -> Dict:
    """不同网络拓扑的语言演化对比"""
    if verbose:
        print("\n" + "=" * 60)
        print("实验 2：拓扑对比（100 Agent）")
        print("=" * 60)

    results = {}
    for topo in ['full', 'small_world', 'scale_free', 'line', 'star']:
        if verbose:
            print(f"\n--- {topo} topology ---")
        society = LargeScaleLanguageSociety(num_agents, topology=topo)
        log = run_society(society, num_rounds, verbose=verbose)
        final_metrics = society.get_global_metrics(sample_size=200)
        topo_stats = society.get_topology_stats()
        results[topo] = {
            'log': log,
            'final_metrics': final_metrics,
            'topology_stats': topo_stats,
        }

    if verbose:
        print(f"\n拓扑对比结果:")
        for topo, r in results.items():
            fm = r['final_metrics']
            ts = r['topology_stats']
            print(f"  {topo:12s}: similarity={fm['avg_overall_similarity']:.3f}, "
                  f"avg_degree={ts['avg_degree']:.1f}")

    return results


# ============================================================
# 实验 3：语言家族涌现
# ============================================================

def experiment_3_language_families(num_agents: int = 100,
                                   num_groups: int = 5,
                                   isolation_rounds: int = 1000,
                                   open_rounds: int = 2000,
                                   verbose: bool = True) -> Dict:
    """
    语言家族涌现实验

    前 isolation_rounds 轮：组内隔离通信，每组面对不同的环境（不同随机种子）
    后 open_rounds 轮：开放跨组通信
    观察语言家族数量变化
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 3：语言家族涌现")
        print("=" * 60)

    # 分组
    group_size = num_agents // num_groups
    groups = []
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size if g < num_groups - 1 else num_agents
        groups.append(list(range(start, end)))

    society = LargeScaleLanguageSociety(
        num_agents, topology='groups', groups=groups
    )

    log = {
        'rounds': [],
        'num_families': [],
        'intra_similarity': [],
        'inter_similarity': [],
        'dialect_divergence': [],
    }

    # 阶段 1：隔离通信（每组面对不同环境，模拟地理隔离）
    if verbose:
        print(f"\n阶段 1：隔离通信（{isolation_rounds} 轮，不同环境）")

    base_seed = 42
    for r in range(isolation_rounds):
        # 每组用不同的随机种子生成场景，模拟地理隔离
        for group_idx, group in enumerate(groups):
            group_seed = base_seed + r * 100 + group_idx * 17
            np.random.seed(group_seed)
            scene = generate_rich_scene('medium')
            target_idx = np.random.randint(0, len(scene))

            # 组内随机配对交流
            agents = [society.agents[i] for i in group]
            a, b = random.sample(agents, 2)
            if random.random() < 0.5:
                speaker, listener = a, b
            else:
                speaker, listener = b, a

            from language_emergence import cross_language_round
            cross_language_round(speaker, listener, scene, target_idx)

        # 恢复全局种子
        np.random.seed(base_seed + r)

        if (r + 1) % 200 == 0:
            families = society.detect_language_families(threshold=0.5)
            dialect = society.get_dialect_metrics(sample_size=200)
            log['rounds'].append(r + 1)
            log['num_families'].append(len(families))
            log['intra_similarity'].append(dialect.get('intra_group_similarity', 0))
            log['inter_similarity'].append(dialect.get('inter_group_similarity', 0))
            log['dialect_divergence'].append(dialect.get('dialect_divergence', 0))

            if verbose:
                print(f"  Round {r+1:5d}: families={len(families)}, "
                      f"intra={dialect.get('intra_group_similarity', 0):.3f}, "
                      f"inter={dialect.get('inter_group_similarity', 0):.3f}")

    # 阶段 2：开放通信
    if verbose:
        print(f"\n阶段 2：开放跨组通信（{open_rounds} 轮）")

    society.merge_groups()

    for r in range(open_rounds):
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)
        society.step(scene, target_idx)

        if (r + 1) % 200 == 0:
            total_round = isolation_rounds + r + 1
            families = society.detect_language_families(threshold=0.5)
            metrics = society.get_global_metrics(sample_size=200)
            log['rounds'].append(total_round)
            log['num_families'].append(len(families))
            log['intra_similarity'].append(metrics['avg_overall_similarity'])
            log['inter_similarity'].append(metrics['avg_overall_similarity'])
            log['dialect_divergence'].append(0.0)

            if verbose:
                print(f"  Round {total_round:5d}: families={len(families)}, "
                      f"similarity={metrics['avg_overall_similarity']:.3f}")

    final_families = society.detect_language_families(threshold=0.5)

    if verbose:
        print(f"\n语言家族结果:")
        print(f"  初始组数: {num_groups}")
        print(f"  最终家族数: {len(final_families)}")
        for i, family in enumerate(final_families):
            print(f"  家族 {i}: {len(family)} 个 agent")

    return {
        'log': log,
        'final_families': final_families,
        'initial_groups': num_groups,
    }


# ============================================================
# 实验 4：通用语涌现
# ============================================================

def experiment_4_lingua_franca(num_agents: int = 100,
                               num_rounds: int = 5000,
                               check_interval: int = 500,
                               verbose: bool = True) -> Dict:
    """
    通用语涌现实验

    随机拓扑，持续通信。
    每 check_interval 轮检测是否有通用语出现。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 4：通用语涌现")
        print("=" * 60)

    society = LargeScaleLanguageSociety(num_agents, topology='small_world')

    log = {
        'rounds': [],
        'success_rate': [],
        'avg_similarity': [],
        'lingua_franca': [],
        'num_families': [],
    }

    for r in range(num_rounds):
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)
        society.step(scene, target_idx)

        if (r + 1) % check_interval == 0:
            sr = society.get_communication_success_rate(check_interval)
            metrics = society.get_global_metrics(sample_size=200)
            lf = society.detect_lingua_franca(cross_group_threshold=0.6)
            families = society.detect_language_families(threshold=0.5)

            log['rounds'].append(r + 1)
            log['success_rate'].append(sr)
            log['avg_similarity'].append(metrics['avg_overall_similarity'])
            log['lingua_franca'].append(lf)
            log['num_families'].append(len(families))

            if verbose:
                lf_str = lf if lf else "None"
                print(f"  Round {r+1:5d}: success={sr:.3f}, "
                      f"similarity={metrics['avg_overall_similarity']:.3f}, "
                      f"lingua_franca={lf_str}, families={len(families)}")

    # 最终检测
    final_lf = society.detect_lingua_franca(cross_group_threshold=0.5)
    final_families = society.detect_language_families(threshold=0.5)

    if verbose:
        print(f"\n通用语结果:")
        print(f"  最终通用语: {final_lf if final_lf else 'None'}")
        print(f"  最终家族数: {len(final_families)}")
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
                                    num_rounds: int = 5000,
                                    check_interval: int = 500,
                                    replace_fraction: float = 0.1,
                                    verbose: bool = True) -> Dict:
    """
    语言灭绝与复兴实验

    每 check_interval 轮：移除使用最少语言的 10% Agent，补充新 Agent。
    观察语言灭绝率、新语言涌现率、语言多样性指数变化。
    """
    if verbose:
        print("\n" + "=" * 60)
        print("实验 5：语言灭绝与复兴")
        print("=" * 60)

    society = LargeScaleLanguageSociety(num_agents, topology='scale_free')

    log = {
        'rounds': [],
        'success_rate': [],
        'avg_similarity': [],
        'diversity_index': [],
        'extinct_count': [],
        'num_families': [],
    }

    total_replaced = 0

    for r in range(num_rounds):
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)
        society.step(scene, target_idx)

        if (r + 1) % check_interval == 0:
            sr = society.get_communication_success_rate(check_interval)
            metrics = society.get_global_metrics(sample_size=200)
            diversity = society.get_language_diversity_index()
            extinct = society.find_extinct_candidates(threshold=0.3)
            families = society.detect_language_families(threshold=0.5)

            log['rounds'].append(r + 1)
            log['success_rate'].append(sr)
            log['avg_similarity'].append(metrics['avg_overall_similarity'])
            log['diversity_index'].append(diversity)
            log['extinct_count'].append(len(extinct))
            log['num_families'].append(len(families))

            if verbose:
                print(f"  Round {r+1:5d}: success={sr:.3f}, "
                      f"diversity={diversity:.3f}, "
                      f"extinct_candidates={len(extinct)}, "
                      f"families={len(families)}")

            # 移除最弱的 agent，补充新 agent
            if extinct:
                num_replace = max(1, int(len(extinct) * replace_fraction))
                to_replace = random.sample(extinct, min(num_replace, len(extinct)))
                society.replace_agents(to_replace)
                total_replaced += len(to_replace)

                if verbose:
                    print(f"    → 替换 {len(to_replace)} 个 agent "
                          f"(累计: {total_replaced})")

    final_diversity = society.get_language_diversity_index()
    final_families = society.detect_language_families(threshold=0.5)

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
    print("Phase 30: 大规模社会语言演化实验")
    print("=" * 60)

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    results = {}

    # 实验 1：规模对比
    results['exp1'] = experiment_1_scale_comparison(
        num_rounds=1000, verbose=True
    )

    # 实验 2：拓扑对比
    results['exp2'] = experiment_2_topology_comparison(
        num_agents=100, num_rounds=2000, verbose=True
    )

    # 实验 3：语言家族涌现
    results['exp3'] = experiment_3_language_families(
        num_agents=100, num_groups=5,
        isolation_rounds=1000, open_rounds=2000, verbose=True
    )

    # 实验 4：通用语涌现
    results['exp4'] = experiment_4_lingua_franca(
        num_agents=100, num_rounds=5000, verbose=True
    )

    # 实验 5：语言灭绝与复兴
    results['exp5'] = experiment_5_extinction_revival(
        num_agents=100, num_rounds=5000, verbose=True
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

    with open('large_society_results.json', 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n结果已保存到 large_society_results.json")

    return results


if __name__ == '__main__':
    main()
