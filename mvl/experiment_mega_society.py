"""
Phase 49 实验：2000-5000 Agent 超大规模社会

4 个实验：
1. 规模梯度 — 2000/3000/5000 Agent 的语言演化
2. 拓扑对比 — small_world vs scale_free vs line（2000 Agent）
3. 语言家族演化 — 5000 Agent 家族数随时间变化
4. 枢纽 Agent 分析 — scale_free 网络中高度节点的语言特征
"""

import sys
import time
import random
import numpy as np
import json

sys.stdout.reconfigure(encoding='utf-8')

from language_emergence import generate_rich_scene
from language_society_large import LargeScaleLanguageSociety


def run_mega_society(society: LargeScaleLanguageSociety,
                     num_rounds: int = 3000,
                     complexity: str = 'medium',
                     log_interval: int = 500,
                     verbose: bool = True) -> dict:
    """运行超大规模社会实验"""
    log = {
        'rounds': [],
        'success_rate': [],
        'avg_similarity': [],
        'num_families': [],
        'timing': {},
    }

    t_start = time.time()
    total_comm = 0
    total_metric = 0
    cumulative_success = 0
    cumulative_total = 0

    for r in range(0, num_rounds, 100):
        batch_size = min(100, num_rounds - r)

        # 通信（每轮 100 对 Agent 并行）
        t0 = time.time()
        scene = generate_rich_scene(complexity)
        target_idx = random.randint(0, len(scene) - 1)
        for _ in range(batch_size):
            rate = society.batch_step_parallel(scene, target_idx, num_pairs=50)
            cumulative_success += int(rate * 50)
            cumulative_total += 50
        t_comm = time.time() - t0
        total_comm += t_comm

        if (r + batch_size) % log_interval == 0 or (r + batch_size) >= num_rounds:
            # 指标计算
            t0 = time.time()
            metrics = society.get_global_metrics_fast(sample_size=200)
            t_metric = time.time() - t0
            total_metric += t_metric

            sr = cumulative_success / cumulative_total if cumulative_total > 0 else 0
            sim = metrics['avg_vocab_similarity']

            # 采样家族检测（避免全量矩阵）
            t0 = time.time()
            families = society.detect_language_families_sampled(sample_size=300, threshold=0.5)
            t_family = time.time() - t0

            log['rounds'].append(r + batch_size)
            log['success_rate'].append(sr)
            log['avg_similarity'].append(sim)
            log['num_families'].append(len(families))

            if verbose:
                elapsed = time.time() - t_start
                print(f"  Round {r+batch_size:5d} | "
                      f"成功率={sr:.3f} | 相似度={sim:.3f} | "
                      f"家族={len(families):3d} | "
                      f"耗时={elapsed:.1f}s")

    total_time = time.time() - t_start
    log['timing'] = {
        'total_seconds': round(total_time, 2),
        'comm_seconds': round(total_comm, 2),
        'metric_seconds': round(total_metric, 2),
        'rounds_per_second': round(num_rounds / total_time, 1) if total_time > 0 else 0,
    }

    return log


# ============================================================
# 实验 1：规模梯度
# ============================================================

def experiment_1_scale_gradient():
    """实验 1：2000/3000/5000 Agent 规模梯度"""
    print("\n" + "=" * 60)
    print("实验 1：规模梯度（2000/3000/5000 Agent）")
    print("=" * 60)

    results = {}
    scales = [2000, 3000, 5000]

    for n in scales:
        print(f"\n--- {n} Agent (small_world) ---")
        society = LargeScaleLanguageSociety(n, topology='small_world')
        log = run_mega_society(society, num_rounds=3000, log_interval=500)
        results[n] = log

        final_sim = log['avg_similarity'][-1] if log['avg_similarity'] else 0
        final_families = log['num_families'][-1] if log['num_families'] else 0
        print(f"  最终相似度={final_sim:.3f}, 家族数={final_families}, "
              f"速度={log['timing']['rounds_per_second']:.0f} rounds/s")

    return results


# ============================================================
# 实验 2：拓扑对比
# ============================================================

def experiment_2_topology_comparison():
    """实验 2：2000 Agent 不同拓扑对比"""
    print("\n" + "=" * 60)
    print("实验 2：拓扑对比（2000 Agent）")
    print("=" * 60)

    topologies = ['small_world', 'scale_free', 'line']
    results = {}

    for topo in topologies:
        print(f"\n--- {topo} ---")
        society = LargeScaleLanguageSociety(2000, topology=topo)
        log = run_mega_society(society, num_rounds=2000, log_interval=500)
        results[topo] = log

        final_sim = log['avg_similarity'][-1] if log['avg_similarity'] else 0
        final_families = log['num_families'][-1] if log['num_families'] else 0
        print(f"  最终相似度={final_sim:.3f}, 家族数={final_families}")

    return results


# ============================================================
# 实验 3：语言家族演化
# ============================================================

def experiment_3_family_evolution():
    """实验 3：5000 Agent 语言家族演化"""
    print("\n" + "=" * 60)
    print("实验 3：语言家族演化（5000 Agent）")
    print("=" * 60)

    society = LargeScaleLanguageSociety(5000, topology='small_world')

    family_history = []
    t_start = time.time()

    for r in range(0, 3000, 100):
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)
        for _ in range(100):
            society.batch_step_parallel(scene, target_idx, num_pairs=50)

        # 每 500 轮检测家族
        if (r + 100) % 500 == 0:
            families = society.detect_language_families_sampled(sample_size=300, threshold=0.5)
            family_sizes = sorted([len(f) for f in families], reverse=True)
            elapsed = time.time() - t_start
            family_history.append({
                'round': r + 100,
                'num_families': len(families),
                'largest_family': family_sizes[0] if family_sizes else 0,
                'top_5_sizes': family_sizes[:5],
            })
            print(f"  Round {r+100:5d} | 家族={len(families):3d} | "
                  f"最大家族={family_sizes[0] if family_sizes else 0:5d} | "
                  f"Top5={family_sizes[:5]} | {elapsed:.1f}s")

    return {'family_history': family_history}


# ============================================================
# 实验 4：枢纽 Agent 分析
# ============================================================

def experiment_4_hub_analysis():
    """实验 4：scale_free 网络枢纽 Agent 分析"""
    print("\n" + "=" * 60)
    print("实验 4：枢纽 Agent 分析（3000 Agent scale_free）")
    print("=" * 60)

    society = LargeScaleLanguageSociety(3000, topology='scale_free')

    # 找出度最高的枢纽节点
    degrees = {aid: len(nbs) for aid, nbs in society.adjacency.items()}
    sorted_by_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    hub_ids = [aid for aid, _ in sorted_by_degree[:10]]
    hub_degrees = [d for _, d in sorted_by_degree[:10]]

    print(f"  枢纽 Agent 度数: {hub_degrees}")
    print(f"  平均度数: {np.mean(list(degrees.values())):.1f}")

    # 运行社会交流
    print("\n  运行 2000 轮通信...")
    t_start = time.time()
    for r in range(0, 2000, 100):
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)
        for _ in range(100):
            society.batch_step_parallel(scene, target_idx, num_pairs=50)
    elapsed = time.time() - t_start
    print(f"  通信完成，耗时 {elapsed:.1f}s")

    # 分析枢纽 vs 普通 Agent
    hub_agents = [society.agent_map[aid] for aid in hub_ids]
    hub_vocab_sizes = [len(a.language.vocabulary) for a in hub_agents]

    # 随机采样 100 个普通 Agent
    non_hub_ids = [aid for aid in degrees if aid not in set(hub_ids)]
    sample_ids = random.sample(non_hub_ids, min(100, len(non_hub_ids)))
    sample_agents = [society.agent_map[aid] for aid in sample_ids]
    sample_vocab_sizes = [len(a.language.vocabulary) for a in sample_agents]

    # 枢纽 Agent 与其他 Agent 的平均相似度
    hub_sims = []
    for hub_agent in hub_agents:
        others = random.sample(range(society.num_agents), min(50, society.num_agents))
        for idx in others:
            other = society.agents[idx]
            if other.id != hub_agent.id:
                from language_emergence import compute_language_similarity
                sim = compute_language_similarity(hub_agent.language, other.language)['overall_similarity']
                hub_sims.append(sim)

    # 普通 Agent 与其他 Agent 的平均相似度
    normal_sims = []
    for agent in random.sample(sample_agents, min(10, len(sample_agents))):
        others = random.sample(range(society.num_agents), min(50, society.num_agents))
        for idx in others:
            other = society.agents[idx]
            if other.id != agent.id:
                from language_emergence import compute_language_similarity
                sim = compute_language_similarity(agent.language, other.language)['overall_similarity']
                normal_sims.append(sim)

    results = {
        'hub_degrees': hub_degrees,
        'avg_degree': float(np.mean(list(degrees.values()))),
        'hub_vocab_mean': float(np.mean(hub_vocab_sizes)),
        'hub_vocab_std': float(np.std(hub_vocab_sizes)),
        'normal_vocab_mean': float(np.mean(sample_vocab_sizes)),
        'normal_vocab_std': float(np.std(sample_vocab_sizes)),
        'hub_avg_similarity': float(np.mean(hub_sims)) if hub_sims else 0,
        'normal_avg_similarity': float(np.mean(normal_sims)) if normal_sims else 0,
    }

    print(f"\n  枢纽 Agent 词汇量: {results['hub_vocab_mean']:.1f} ± {results['hub_vocab_std']:.1f}")
    print(f"  普通 Agent 词汇量: {results['normal_vocab_mean']:.1f} ± {results['normal_vocab_std']:.1f}")
    print(f"  枢纽平均相似度: {results['hub_avg_similarity']:.3f}")
    print(f"  普通平均相似度: {results['normal_avg_similarity']:.3f}")

    return results


if __name__ == '__main__':
    all_results = {}

    all_results['scale_gradient'] = experiment_1_scale_gradient()
    all_results['topology_comparison'] = experiment_2_topology_comparison()
    all_results['family_evolution'] = experiment_3_family_evolution()
    all_results['hub_analysis'] = experiment_4_hub_analysis()

    with open('mega_society_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n\n结果已保存到 mega_society_results.json")
