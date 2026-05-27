"""
多 Agent 社会语言实验：方言分化与语言融合

4 个实验：
1. 方言分化：隔离群体是否发展出不同方言？
2. 语言融合：打破隔离后，语言趋同还是保持差异？
3. 网络拓扑：全连接/星形/线形如何影响语言演化？
4. 人口规模：agent 数量如何影响趋同速度？
"""

import sys
import numpy as np
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from language_emergence import (
    LanguageAgent, cross_language_round, compute_language_similarity,
    generate_rich_scene, EmergingLanguage
)
from language_society import LanguageSociety


def run_isolation_experiment(num_agents: int = 6, num_groups: int = 2,
                             num_rounds: int = 1000,
                             complexity: str = 'medium',
                             seed: int = 42) -> Dict:
    """
    实验 1：方言分化

    将 agent 分成隔离的群体，组内交流但组间不交流。
    不同群体面对不同的场景采样（不同随机种子），增加环境差异。
    观察是否产生方言分化。
    """
    np.random.seed(seed)

    # 分组
    group_size = num_agents // num_groups
    groups = []
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size if g < num_groups - 1 else num_agents
        groups.append(list(range(start, end)))

    society = LanguageSociety(num_agents, topology='groups', groups=groups)

    # 记录
    log = {
        'successes': [],
        'intra_similarity': [],
        'inter_similarity': [],
        'dialect_divergence': [],
    }

    for round_idx in range(num_rounds):
        # 每轮为每个组生成不同的场景（不同随机种子）
        # 这模拟了不同群体面对不同环境的现实
        for group_idx, group in enumerate(groups):
            np.random.seed(seed + round_idx * 100 + group_idx * 10)
            scene = generate_rich_scene(complexity)
            target_idx = np.random.randint(0, len(scene))

            # 组内随机配对交流
            agents = [society.agents[i] for i in group]
            a, b = agents[np.random.randint(len(agents))], agents[np.random.randint(len(agents))]
            while a.id == b.id and len(agents) > 1:
                b = agents[np.random.randint(len(agents))]

            if np.random.random() < 0.5:
                speaker, listener = a, b
            else:
                speaker, listener = b, a

            cross_language_round(speaker, listener, scene, target_idx)

        # 恢复全局随机种子
        np.random.seed(seed + round_idx)

        if (round_idx + 1) % 100 == 0:
            metrics = society.get_dialect_metrics()
            log['successes'].append(society.get_communication_success_rate(100))
            log['intra_similarity'].append(metrics['intra_group_similarity'])
            log['inter_similarity'].append(metrics['inter_group_similarity'])
            log['dialect_divergence'].append(metrics['dialect_divergence'])

            print(f"  轮次 {round_idx+1:4d}: "
                  f"组内相似={metrics['intra_group_similarity']:.3f} "
                  f"组间相似={metrics['inter_group_similarity']:.3f} "
                  f"方言分化={metrics['dialect_divergence']:.3f}")

    final_metrics = society.get_dialect_metrics()
    return {
        'log': log,
        'final_metrics': final_metrics,
        'society': society,
    }


def run_fusion_experiment(isolation_rounds: int = 1000,
                          fusion_rounds: int = 500,
                          complexity: str = 'medium',
                          seed: int = 42) -> Dict:
    """
    实验 2：语言融合

    先隔离（产生方言），再融合（打破隔离），观察语言趋同。
    """
    np.random.seed(seed)

    # 阶段 1：隔离
    print("  阶段 1：隔离期（1000 轮）")
    groups = [[0, 1, 2], [3, 4, 5]]
    society = LanguageSociety(6, topology='groups', groups=groups)

    log = {
        'phase': [],
        'successes': [],
        'intra_similarity': [],
        'inter_similarity': [],
        'cross_group_success': [],
    }

    for round_idx in range(isolation_rounds):
        scene = generate_rich_scene(complexity)
        target_idx = np.random.randint(0, len(scene))
        society.step(scene, target_idx)

        if (round_idx + 1) % 200 == 0:
            metrics = society.get_dialect_metrics()
            log['phase'].append('isolation')
            log['successes'].append(society.get_communication_success_rate(100))
            log['intra_similarity'].append(metrics['intra_group_similarity'])
            log['inter_similarity'].append(metrics['inter_group_similarity'])
            log['cross_group_success'].append(0.0)
            print(f"    隔离 {round_idx+1}: "
                  f"组内相似={metrics['intra_group_similarity']:.3f} "
                  f"组间相似={metrics['inter_group_similarity']:.3f}")

    # 阶段 2：融合
    print("  阶段 2：融合期（500 轮）")
    society.merge_groups()

    for round_idx in range(fusion_rounds):
        scene = generate_rich_scene(complexity)
        target_idx = np.random.randint(0, len(scene))
        society.step(scene, target_idx)

        if (round_idx + 1) % 100 == 0:
            metrics = society.get_dialect_metrics()
            log['phase'].append('fusion')
            log['successes'].append(society.get_communication_success_rate(100))
            log['intra_similarity'].append(metrics.get('avg_overall_similarity', 0))
            log['inter_similarity'].append(metrics.get('avg_overall_similarity', 0))
            log['cross_group_success'].append(
                society.get_communication_success_rate(100)
            )
            print(f"    融合 {round_idx+1}: "
                  f"成功率={log['successes'][-1]:.1%} "
                  f"综合相似={metrics.get('avg_overall_similarity', 0):.3f}")

    return {
        'log': log,
        'final_metrics': society.get_dialect_metrics(),
        'society': society,
    }


def run_topology_experiment(num_agents: int = 8,
                            num_rounds: int = 1000,
                            complexity: str = 'medium',
                            seed: int = 42) -> Dict:
    """
    实验 3：网络拓扑对比

    对比全连接、星形、线形三种拓扑下的语言演化。
    """
    topologies = ['full', 'star', 'line']
    results = {}

    for topo in topologies:
        np.random.seed(seed)
        print(f"\n  拓扑: {topo}")
        society = LanguageSociety(num_agents, topology=topo)

        log = {'successes': [], 'overall_similarity': [], 'order_match': []}

        for round_idx in range(num_rounds):
            scene = generate_rich_scene(complexity)
            target_idx = np.random.randint(0, len(scene))
            society.step(scene, target_idx)

            if (round_idx + 1) % 100 == 0:
                metrics = society.get_dialect_metrics()
                log['successes'].append(society.get_communication_success_rate(100))
                log['overall_similarity'].append(metrics.get('avg_overall_similarity', 0))
                log['order_match'].append(metrics.get('avg_order_match', 0))

                print(f"    轮次 {round_idx+1}: "
                      f"成功率={log['successes'][-1]:.1%} "
                      f"综合相似={metrics.get('avg_overall_similarity', 0):.3f} "
                      f"词序一致={metrics.get('avg_order_match', 0):.3f}")

        results[topo] = {
            'log': log,
            'final_metrics': society.get_dialect_metrics(),
            'agent_stats': society.get_all_agent_stats(),
        }

    return results


def run_population_experiment(populations: List[int] = None,
                              num_rounds: int = 1000,
                              complexity: str = 'medium',
                              seed: int = 42) -> Dict:
    """
    实验 4：人口规模影响

    对比不同人口规模下的语言趋同速度。
    """
    if populations is None:
        populations = [2, 4, 8, 16]

    results = {}

    for pop in populations:
        np.random.seed(seed)
        print(f"\n  人口: {pop}")
        society = LanguageSociety(pop, topology='full')

        log = {'successes': [], 'overall_similarity': []}

        for round_idx in range(num_rounds):
            scene = generate_rich_scene(complexity)
            target_idx = np.random.randint(0, len(scene))
            society.step(scene, target_idx)

            if (round_idx + 1) % 200 == 0:
                metrics = society.get_dialect_metrics()
                log['successes'].append(society.get_communication_success_rate(100))
                log['overall_similarity'].append(metrics.get('avg_overall_similarity', 0))

                print(f"    轮次 {round_idx+1}: "
                      f"成功率={log['successes'][-1]:.1%} "
                      f"综合相似={metrics.get('avg_overall_similarity', 0):.3f}")

        results[pop] = {
            'log': log,
            'final_metrics': society.get_dialect_metrics(),
            'agent_stats': society.get_all_agent_stats(),
        }

    return results


def main():
    print("多 Agent 社会语言实验：方言分化与语言融合")
    print("=" * 70)

    # === 实验 1：方言分化 ===
    print("\n实验 1：方言分化（6 agent，2 组，2000 轮，extreme 复杂度）")
    print("-" * 70)
    iso_result = run_isolation_experiment(
        num_agents=6, num_groups=2, num_rounds=2000,
        complexity='extreme', seed=42
    )
    final = iso_result['final_metrics']
    print(f"\n  最终方言分化指标:")
    print(f"    组内相似度: {final['intra_group_similarity']:.3f}")
    print(f"    组间相似度: {final['inter_group_similarity']:.3f}")
    print(f"    方言分化度: {final['dialect_divergence']:.3f}")

    # 显示各组的词汇
    society = iso_result['society']
    for gi, group in enumerate(society.groups):
        agents = [society.agents[i] for i in group]
        vocabs = [set(a.language.vocabulary.keys()) for a in agents]
        shared = vocabs[0]
        for v in vocabs[1:]:
            shared = shared & v
        print(f"    组 {gi}: 共享词汇 = {shared}")

    # === 实验 2：语言融合 ===
    print("\n" + "=" * 70)
    print("实验 2：语言融合（先隔离 1000 轮，再融合 500 轮）")
    print("-" * 70)
    fusion_result = run_fusion_experiment(
        isolation_rounds=1000, fusion_rounds=500, seed=42
    )

    # === 实验 3：网络拓扑 ===
    print("\n" + "=" * 70)
    print("实验 3：网络拓扑对比（8 agent，1000 轮）")
    print("-" * 70)
    topo_result = run_topology_experiment(
        num_agents=8, num_rounds=1000, seed=42
    )

    # === 实验 4：人口规模 ===
    print("\n" + "=" * 70)
    print("实验 4：人口规模影响（1000 轮）")
    print("-" * 70)
    pop_result = run_population_experiment(
        populations=[2, 4, 8, 16], num_rounds=1000, seed=42
    )

    # === 汇总 ===
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)

    # 实验 1 汇总
    print("\n  实验 1 - 方言分化:")
    print(f"    组内相似度: {final['intra_group_similarity']:.3f}")
    print(f"    组间相似度: {final['inter_group_similarity']:.3f}")
    if final['intra_group_similarity'] > final['inter_group_similarity']:
        print(f"    ✓ 方言分化成功：组内相似度 > 组间相似度")
    else:
        print(f"    △ 未观察到明显方言分化")

    # 实验 3 汇总
    print("\n  实验 3 - 拓扑对比:")
    for topo, data in topo_result.items():
        fm = data['final_metrics']
        print(f"    {topo:>6s}: 综合相似={fm.get('avg_overall_similarity', 0):.3f} "
              f"词序一致={fm.get('avg_order_match', 0):.3f}")

    # 实验 4 汇总
    print("\n  实验 4 - 人口规模:")
    for pop, data in pop_result.items():
        fm = data['final_metrics']
        print(f"    {pop:2d} agent: 综合相似={fm.get('avg_overall_similarity', 0):.3f}")

    # === 保存结果 ===
    import json

    def to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        'isolation': {
            'final_metrics': {k: to_serializable(v) for k, v in final.items()},
        },
        'topology': {
            topo: {
                'final_metrics': {k: to_serializable(v)
                                  for k, v in data['final_metrics'].items()},
            }
            for topo, data in topo_result.items()
        },
        'population': {
            str(pop): {
                'final_metrics': {k: to_serializable(v)
                                  for k, v in data['final_metrics'].items()},
            }
            for pop, data in pop_result.items()
        },
    }

    with open('D:/mayAi/AILearning_v0527/mvl/language_society_results.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存到: language_society_results.json")


if __name__ == '__main__':
    main()
