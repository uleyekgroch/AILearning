"""
Phase 47 实验：CUDA 加速的 1000+ Agent 大社会

3 个实验：
1. 规模对比：100/500/1000 Agent 的语言演化（CUDA 加速）
2. 语言家族检测：1000 Agent 小世界网络下的家族涌现
3. 方言分化：分组隔离 → 语言分化 → 合并后通用语涌现

使用 GPU 加速的批量相似度计算和家族检测。
"""

import sys
import random
import time
import numpy as np
import json

sys.stdout.reconfigure(encoding='utf-8')

from language_emergence import generate_rich_scene
from language_society_large import LargeScaleLanguageSociety


def run_society_cuda(society: LargeScaleLanguageSociety,
                     num_rounds: int = 2000,
                     complexity: str = 'medium',
                     log_interval: int = 200,
                     verbose: bool = True) -> dict:
    """运行社会交流实验，使用 CUDA 加速指标计算"""
    log = {
        'rounds': [],
        'success_rate': [],
        'avg_similarity': [],
        'timing': {},
    }

    t_start = time.time()
    comm_time = 0
    metric_time = 0

    for r in range(num_rounds):
        # 通信
        t0 = time.time()
        scene = generate_rich_scene(complexity)
        target_idx = random.randint(0, len(scene) - 1)
        society.step(scene, target_idx)
        comm_time += time.time() - t0

        if (r + 1) % log_interval == 0:
            # 指标计算（CUDA 加速）
            t0 = time.time()
            sr = society.get_communication_success_rate(log_interval)
            metrics = society.get_global_metrics_fast(sample_size=200)
            metric_time += time.time() - t0

            log['rounds'].append(r + 1)
            log['success_rate'].append(sr)
            log['avg_similarity'].append(metrics['avg_vocab_similarity'])

            if verbose:
                elapsed = time.time() - t_start
                print(f"  Round {r+1:5d}: success={sr:.3f}, "
                      f"similarity={metrics['avg_vocab_similarity']:.3f}, "
                      f"elapsed={elapsed:.1f}s")

    log['timing'] = {
        'total': time.time() - t_start,
        'communication': comm_time,
        'metrics': metric_time,
    }

    return log


def experiment_1_scale_comparison():
    """实验 1：规模对比 — 100/500/1000 Agent"""
    print("\n" + "=" * 60)
    print("实验 1：规模对比（CUDA 加速）")
    print("=" * 60)

    results = {}
    for n_agents in [100, 500, 1000]:
        print(f"\n--- {n_agents} Agents ---")
        society = LargeScaleLanguageSociety(
            num_agents=n_agents, topology='small_world'
        )

        t0 = time.time()
        log = run_society_cuda(
            society, num_rounds=2000, log_interval=200, verbose=True
        )
        elapsed = time.time() - t0

        final_sr = log['success_rate'][-1] if log['success_rate'] else 0
        final_sim = log['avg_similarity'][-1] if log['avg_similarity'] else 0

        results[n_agents] = {
            'final_success_rate': final_sr,
            'final_similarity': final_sim,
            'total_time': elapsed,
            'rounds_per_sec': 2000 / elapsed,
        }

        print(f"  结果: success={final_sr:.3f}, similarity={final_sim:.3f}")
        print(f"  耗时: {elapsed:.1f}s ({2000/elapsed:.0f} rounds/s)")

    return results


def experiment_2_family_detection():
    """实验 2：1000 Agent 语言家族检测"""
    print("\n" + "=" * 60)
    print("实验 2：1000 Agent 语言家族检测")
    print("=" * 60)

    society = LargeScaleLanguageSociety(
        num_agents=1000, topology='small_world'
    )

    # 先运行一些通信轮次
    print("\nPhase 1: 建立语言...")
    for r in range(3000):
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)
        society.step(scene, target_idx)
        if (r + 1) % 500 == 0:
            sr = society.get_communication_success_rate(500)
            print(f"  Round {r+1}: success_rate={sr:.3f}")

    # 家族检测
    print("\nPhase 2: 家族检测...")
    t0 = time.time()
    families_fast = society.detect_language_families_fast(threshold=0.5)
    t_fast = time.time() - t0

    t0 = time.time()
    families_orig = society.detect_language_families(threshold=0.5, sample_size=1000)
    t_orig = time.time() - t0

    print(f"  Fast: {len(families_fast)} families in {t_fast:.3f}s")
    print(f"  Original: {len(families_orig)} families in {t_orig:.3f}s")
    print(f"  Speedup: {t_orig/max(0.001, t_fast):.1f}x")

    # 打印最大的几个家族
    families_fast.sort(key=len, reverse=True)
    print(f"\n  Top 5 families:")
    for i, f in enumerate(families_fast[:5]):
        print(f"    Family {i+1}: {len(f)} agents")

    return {
        'num_families_fast': len(families_fast),
        'num_families_orig': len(families_orig),
        'time_fast': t_fast,
        'time_orig': t_orig,
        'speedup': t_orig / max(0.001, t_fast),
        'top_5_sizes': [len(f) for f in families_fast[:5]],
    }


def experiment_3_dialect_divergence():
    """实验 3：方言分化 — 隔离组 → 语言分化"""
    print("\n" + "=" * 60)
    print("实验 3：方言分化（500 Agent, 5 组）")
    print("=" * 60)

    n_agents = 500
    n_groups = 5
    group_size = n_agents // n_groups
    groups = [
        list(range(i * group_size, (i + 1) * group_size))
        for i in range(n_groups)
    ]

    # Phase 1: 隔离通信（组内）
    print("\nPhase 1: 组内隔离通信...")
    society = LargeScaleLanguageSociety(
        num_agents=n_agents, topology='groups', groups=groups
    )

    for r in range(2000):
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)
        society.step(scene, target_idx)
        if (r + 1) % 500 == 0:
            sr = society.get_communication_success_rate(500)
            print(f"  Round {r+1}: success_rate={sr:.3f}")

    # 测量方言分化
    sim_matrix = society.batch_cosine_similarity()
    intra_sims = []
    inter_sims = []
    for gi in range(n_groups):
        for gj in range(n_groups):
            group_i = groups[gi]
            group_j = groups[gj]
            # 采样
            pairs = min(100, len(group_i) * len(group_j))
            for _ in range(pairs):
                a = random.choice(group_i)
                b = random.choice(group_j)
                s = sim_matrix[a, b]
                if gi == gj:
                    intra_sims.append(s)
                else:
                    inter_sims.append(s)

    avg_intra = np.mean(intra_sims) if intra_sims else 0
    avg_inter = np.mean(inter_sims) if inter_sims else 0
    print(f"\n  组内相似度: {avg_intra:.3f}")
    print(f"  组间相似度: {avg_inter:.3f}")
    print(f"  方言分化度: {1 - avg_inter/max(0.001, avg_intra):.3f}")

    # Phase 2: 合并后通信
    print("\nPhase 2: 合并后全网通信...")
    society.merge_groups()

    for r in range(2000):
        scene = generate_rich_scene('medium')
        target_idx = random.randint(0, len(scene) - 1)
        society.step(scene, target_idx)
        if (r + 1) % 500 == 0:
            sr = society.get_communication_success_rate(500)
            metrics = society.get_global_metrics_fast(sample_size=200)
            print(f"  Round {r+1}: success={sr:.3f}, "
                  f"similarity={metrics['avg_vocab_similarity']:.3f}")

    return {
        'intra_similarity': float(avg_intra),
        'inter_similarity': float(avg_inter),
        'dialect_divergence': float(1 - avg_inter / max(0.001, avg_intra)),
    }


if __name__ == '__main__':
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {}

    all_results['scale_comparison'] = experiment_1_scale_comparison()
    all_results['family_detection'] = experiment_2_family_detection()
    all_results['dialect_divergence'] = experiment_3_dialect_divergence()

    with open('cuda_society_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n\n结果已保存到 cuda_society_results.json")
