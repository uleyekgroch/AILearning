"""
Phase 27 实验：多 Agent 共同探索与语言通信

5 个实验验证多 Agent 共同探索中的语言涌现。
"""

import json
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_env import MultiAgentGridWorld, create_multi_agent_world
from exploring_agent import ExploringAgent, CoExplorationGame, SoloExplorationGame
from language_emergence import compute_language_similarity
import numpy as np


def experiment_1_individual_view():
    """实验 1：个体视野验证"""
    print("=" * 60)
    print("实验 1: 个体视野验证")
    print("=" * 60)

    env = create_multi_agent_world(num_agents=2)

    # Agent 0 在 (1,1)，Agent 1 在 (10,10)
    obs0 = env.get_agent_observation(0)
    obs1 = env.get_agent_observation(1)

    ids0 = {v['object'].id for v in obs0['visible_objects']}
    ids1 = {v['object'].id for v in obs1['visible_objects']}

    print(f"\n  Agent 0 位置: ({env.agents[0].x}, {env.agents[0].y})")
    print(f"  Agent 0 可见: {len(obs0['visible_objects'])} 个物体, IDs={sorted(ids0)}")
    for vis in obs0['visible_objects']:
        obj = vis['object']
        print(f"    [{obj.id}] {obj.color} {obj.shape} (dist={vis['distance']})")

    print(f"\n  Agent 1 位置: ({env.agents[1].x}, {env.agents[1].y})")
    print(f"  Agent 1 可见: {len(obs1['visible_objects'])} 个物体, IDs={sorted(ids1)}")
    for vis in obs1['visible_objects']:
        obj = vis['object']
        print(f"    [{obj.id}] {obj.color} {obj.shape} (dist={vis['distance']})")

    overlap = ids0 & ids1
    only0 = ids0 - ids1
    only1 = ids1 - ids0
    print(f"\n  重叠: {len(overlap)} 个物体")
    print(f"  仅 Agent 0 可见: {sorted(only0)}")
    print(f"  仅 Agent 1 可见: {sorted(only1)}")

    if only0 or only1:
        print("  判定: 个体视野差异有效")
    else:
        print("  判定: 视野无差异（可能需要更大的环境）")

    return ids0, ids1


def experiment_2_language_emergence():
    """实验 2：语言涌现验证"""
    print("\n" + "=" * 60)
    print("实验 2: 语言涌现验证（500 步共同探索）")
    print("=" * 60)

    random.seed(42)
    game = CoExplorationGame(num_agents=2, world_size=12)
    game.run(total_steps=500, comm_prob=0.4)

    stats = game.get_stats()

    print(f"\n  总通信: {stats['total_comm']} 轮")
    print(f"  成功通信: {stats['success_comm']} 轮")
    print(f"  通信成功率: {stats['comm_success_rate']:.1%}")
    print(f"  物体发现: {stats['total_discovered']}/{stats['total_objects']}")

    for aid, count in stats['discoveries'].items():
        print(f"    Agent {aid}: 发现 {count} 个物体")

    # 语言统计
    for agent in game.agents:
        lang = agent.lang_agent.language
        vocab = lang.get_stats()['vocabulary_size']
        print(f"\n  Agent {agent.agent_id} 语言:")
        print(f"    词汇量: {vocab}")
        print(f"    词汇: {list(lang.vocabulary.keys())[:10]}")

    # 语言相似度
    if stats['language_similarity']:
        for pair, sim in stats['language_similarity'].items():
            overall = sim.get('overall_similarity', 0)
            print(f"\n  语言相似度 ({pair}): {overall:.3f}")

    return game


def experiment_3_communication_benefit():
    """实验 3：通信改善探索"""
    print("\n" + "=" * 60)
    print("实验 3: 通信 vs 无通信")
    print("=" * 60)

    results = {'with_comm': [], 'without_comm': [], 'solo': []}

    for run in range(5):
        seed = 42 + run

        # 有通信的共同探索
        random.seed(seed)
        game_comm = CoExplorationGame(num_agents=2, world_size=12)
        game_comm.run(total_steps=500, comm_prob=0.4)
        results['with_comm'].append(game_comm.get_stats()['total_discovered'])

        # 无通信的共同探索（comm_prob=0）
        random.seed(seed)
        game_no_comm = CoExplorationGame(num_agents=2, world_size=12)
        game_no_comm.run(total_steps=500, comm_prob=0.0)
        results['without_comm'].append(game_no_comm.get_stats()['total_discovered'])

        # 单 Agent 独自探索
        random.seed(seed)
        game_solo = SoloExplorationGame(world_size=12)
        game_solo.run(total_steps=500)
        results['solo'].append(game_solo.get_stats()['total_discovered'])

        print(f"  运行 {run+1}: 有通信={results['with_comm'][-1]}, "
              f"无通信={results['without_comm'][-1]}, "
              f"独自={results['solo'][-1]}")

    for key in results:
        avg = sum(results[key]) / len(results[key])
        print(f"\n  {key}: {avg:.1f}")

    return results


def experiment_4_language_convergence():
    """实验 4：语言趋同/分化"""
    print("\n" + "=" * 60)
    print("实验 4: 语言趋同/分化（多次运行）")
    print("=" * 60)

    similarities = []

    for run in range(5):
        random.seed(42 + run)
        game = CoExplorationGame(num_agents=2, world_size=12)
        game.run(total_steps=1000, comm_prob=0.5)

        stats = game.get_stats()
        if stats['language_similarity']:
            for pair, sim in stats['language_similarity'].items():
                overall = sim.get('overall_similarity', 0)
                similarities.append(overall)
                print(f"  运行 {run+1}: 相似度={overall:.3f}, "
                      f"通信={stats['total_comm']}, "
                      f"成功率={stats['comm_success_rate']:.1%}")

    if similarities:
        avg = sum(similarities) / len(similarities)
        std = (sum((s - avg) ** 2 for s in similarities) / len(similarities)) ** 0.5
        print(f"\n  平均相似度: {avg:.3f} +/- {std:.3f}")

        if avg > 0.7:
            print("  判定: 语言趋同")
        elif avg < 0.3:
            print("  判定: 语言分化")
        else:
            print("  判定: 部分趋同")

    return similarities


def experiment_5_scaling():
    """实验 5：Agent 数量扩展"""
    print("\n" + "=" * 60)
    print("实验 5: Agent 数量扩展（2/4/8）")
    print("=" * 60)

    for num_agents in [2, 4, 8]:
        discoveries = []
        comm_rates = []

        for run in range(3):
            random.seed(42 + run)
            game = CoExplorationGame(num_agents=num_agents, world_size=12)
            game.run(total_steps=500, comm_prob=0.3)
            stats = game.get_stats()
            discoveries.append(stats['total_discovered'])
            comm_rates.append(stats['comm_success_rate'])

        avg_disc = sum(discoveries) / len(discoveries)
        avg_comm = sum(comm_rates) / len(comm_rates)
        print(f"\n  {num_agents} Agents:")
        print(f"    平均发现: {avg_disc:.1f}/{game.get_stats()['total_objects']}")
        print(f"    平均通信成功率: {avg_comm:.1%}")

    return


if __name__ == '__main__':
    print("Phase 27: 多 Agent 共同探索与语言通信实验")
    print("=" * 60)

    r1_ids0, r1_ids1 = experiment_1_individual_view()
    r2 = experiment_2_language_emergence()
    r3 = experiment_3_communication_benefit()
    r4 = experiment_4_language_convergence()
    experiment_5_scaling()

    print("\n" + "=" * 60)
    print("所有实验完成")
    print("=" * 60)

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

    results = {
        'experiment_1_individual_view': {
            'agent0_visible': sorted(r1_ids0),
            'agent1_visible': sorted(r1_ids1),
        },
        'experiment_3_communication_benefit': r3,
        'experiment_4_language_convergence': r4,
    }
    with open('coexploration_results.json', 'w', encoding='utf-8') as f:
        json.dump(to_serializable(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: coexploration_results.json")
