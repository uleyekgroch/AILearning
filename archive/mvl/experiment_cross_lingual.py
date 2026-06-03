"""
Phase 50 实验：跨语言迁移（不同环境的语言互译）

4 个实验：
1. 跨环境基线 — 隔离后直接交流
2. 桥接 Agent 翻译 — 双语 Agent 作为翻译桥梁
3. 环境差异度 vs 翻译难度
4. 通用语涌现 — 5 个不同环境群体的融合
"""

import sys
import time
import random
import numpy as np
import json

sys.stdout.reconfigure(encoding='utf-8')

from language_rich_scene import RegionConfig, generate_regional_scene, ALL_ATTRIBUTE_NAMES
from language_translator import (
    CrossLingualAgent, CrossLingualSociety, measure_cross_lingual_metrics
)
from language_emergence import compute_language_similarity


# ============================================================
# 预定义区域
# ============================================================

TROPICAL = RegionConfig(0, "热带", preferred_attributes=['color', 'size', 'temperature', 'origin'])
ARCTIC = RegionConfig(1, "极地", preferred_attributes=['color', 'size', 'temperature', 'brightness'])
INDUSTRIAL = RegionConfig(2, "工业", preferred_attributes=['material', 'origin', 'texture', 'weight'])
FOREST = RegionConfig(3, "森林", preferred_attributes=['material', 'origin', 'pattern', 'texture'])
MAGICAL = RegionConfig(4, "魔法", preferred_attributes=['origin', 'brightness', 'pattern', 'temperature'])


# ============================================================
# 辅助函数
# ============================================================

def _run_isolation_contact(society: CrossLingualSociety,
                           iso_rounds: int, contact_rounds: int,
                           contact_mode: str = 'random',
                           num_bridges: int = 10,
                           verbose: bool = True) -> dict:
    """运行隔离期 + 接触期的通用流程"""
    # 隔离期
    iso_results = society.phase_isolation(num_rounds=iso_rounds, verbose=verbose)

    # 接触前测量
    before = society.measure_all(num_trials=100)
    if verbose:
        print(f"  接触前 — 跨环境基线: {before['cross_lingual']['cross_success']:.3f}")
        print(f"  接触前 — 符号重叠: {before['cross_lingual']['vocab_overlap']:.3f}")
        print(f"  接触前 — 语言相似度: {before['cross_lingual']['avg_similarity']:.3f}")

    # 接触期
    contact_results = society.phase_contact(
        num_rounds=contact_rounds,
        contact_mode=contact_mode,
        num_bridges=num_bridges,
        verbose=verbose
    )

    # 接触后测量
    after = society.measure_all(num_trials=100)
    if verbose:
        print(f"  接触后 — 跨环境基线: {after['cross_lingual']['cross_success']:.3f}")
        print(f"  接触后 — 符号重叠: {after['cross_lingual']['vocab_overlap']:.3f}")
        print(f"  接触后 — 语言相似度: {after['cross_lingual']['avg_similarity']:.3f}")

    return {
        'isolation': iso_results,
        'before': before,
        'contact': contact_results,
        'after': after,
    }


# ============================================================
# 实验 1：跨环境基线（隔离后直接交流）
# ============================================================

def experiment_1_cross_env_baseline():
    """实验 1：热带 vs 极地 — 隔离后直接交流"""
    print("\n" + "=" * 60)
    print("实验 1：跨环境基线（热带 vs 极地）")
    print("=" * 60)

    society = CrossLingualSociety(TROPICAL, ARCTIC, agents_per_region=200)

    results = _run_isolation_contact(
        society,
        iso_rounds=1000,
        contact_rounds=500,
        contact_mode='random',
        verbose=True
    )

    # 语言差异分析
    print("\n  语言差异分析:")
    vocab_a = set()
    for a in society.agents_a:
        vocab_a.update(a.language.vocabulary.keys())
    vocab_b = set()
    for b in society.agents_b:
        vocab_b.update(b.language.vocabulary.keys())

    only_a = vocab_a - vocab_b
    only_b = vocab_b - vocab_a
    shared = vocab_a & vocab_b
    print(f"    群体 A 独有符号: {sorted(only_a)}")
    print(f"    群体 B 独有符号: {sorted(only_b)}")
    print(f"    共享符号: {sorted(shared)}")

    return results


# ============================================================
# 实验 2：桥接 Agent 翻译
# ============================================================

def experiment_2_bridge_translation():
    """实验 2：桥接 Agent 翻译效果"""
    print("\n" + "=" * 60)
    print("实验 2：桥接 Agent 翻译（热带 vs 极地）")
    print("=" * 60)

    # 无桥接对照
    print("\n--- 对照组：无桥接 ---")
    society_no_bridge = CrossLingualSociety(TROPICAL, ARCTIC, agents_per_region=200)
    results_no_bridge = _run_isolation_contact(
        society_no_bridge,
        iso_rounds=1000,
        contact_rounds=500,
        contact_mode='random',
        verbose=True
    )

    # 有桥接
    print("\n--- 实验组：有桥接 Agent ---")
    society_bridge = CrossLingualSociety(TROPICAL, ARCTIC, agents_per_region=200)
    results_bridge = _run_isolation_contact(
        society_bridge,
        iso_rounds=1000,
        contact_rounds=500,
        contact_mode='bridge',
        num_bridges=10,
        verbose=True
    )

    # 双语者模式
    print("\n--- 实验组：双语者模式 ---")
    society_bilingual = CrossLingualSociety(TROPICAL, ARCTIC, agents_per_region=200)
    results_bilingual = _run_isolation_contact(
        society_bilingual,
        iso_rounds=1000,
        contact_rounds=500,
        contact_mode='bilingual',
        num_bridges=10,
        verbose=True
    )

    comparison = {
        'no_bridge': {
            'cross_before': results_no_bridge['before']['cross_lingual']['cross_success'],
            'cross_after': results_no_bridge['after']['cross_lingual']['cross_success'],
            'contact_success': results_no_bridge['contact']['cross_success'],
        },
        'bridge': {
            'cross_before': results_bridge['before']['cross_lingual']['cross_success'],
            'cross_after': results_bridge['after']['cross_lingual']['cross_success'],
            'contact_success': results_bridge['contact']['cross_success'],
        },
        'bilingual': {
            'cross_before': results_bilingual['before']['cross_lingual']['cross_success'],
            'cross_after': results_bilingual['after']['cross_lingual']['cross_success'],
            'contact_success': results_bilingual['contact']['cross_success'],
        },
    }

    print("\n  对比结果:")
    for mode, data in comparison.items():
        print(f"    {mode}: 接触前={data['cross_before']:.3f} → 接触后={data['cross_after']:.3f}")

    return comparison


# ============================================================
# 实验 3：环境差异度 vs 翻译难度
# ============================================================

def experiment_3_env_difference():
    """实验 3：环境差异度与翻译难度的关系"""
    print("\n" + "=" * 60)
    print("实验 3：环境差异度 vs 翻译难度")
    print("=" * 60)

    # 不同环境组合
    pairs = [
        ("热带 vs 极地", TROPICAL, ARCTIC, "高度不同"),
        ("热带 vs 工业", TROPICAL, INDUSTRIAL, "中度不同"),
        ("热带 vs 森林", TROPICAL, FOREST, "中度不同"),
        ("极地 vs 工业", ARCTIC, INDUSTRIAL, "中度不同"),
        ("工业 vs 魔法", INDUSTRIAL, MAGICAL, "中度不同"),
        ("热带 vs 热带", TROPICAL, RegionConfig(5, "热带2",
            preferred_attributes=['color', 'size', 'temperature', 'origin']),
         "对照组"),
    ]

    results = {}
    for label, region_a, region_b, diff_desc in pairs:
        print(f"\n--- {label} ({diff_desc}) ---")
        society = CrossLingualSociety(region_a, region_b, agents_per_region=150)

        iso = society.phase_isolation(num_rounds=800, verbose=False)
        before = society.measure_all(num_trials=80)

        contact = society.phase_contact(num_rounds=300, contact_mode='random', verbose=False)
        after = society.measure_all(num_trials=80)

        results[label] = {
            'iso_a': iso['A'],
            'iso_b': iso['B'],
            'cross_before': before['cross_lingual']['cross_success'],
            'cross_after': after['cross_lingual']['cross_success'],
            'vocab_overlap_before': before['cross_lingual']['vocab_overlap'],
            'vocab_overlap_after': after['cross_lingual']['vocab_overlap'],
            'sim_before': before['cross_lingual']['avg_similarity'],
            'sim_after': after['cross_lingual']['avg_similarity'],
        }

        print(f"  群体内成功率: A={iso['A']:.3f}, B={iso['B']:.3f}")
        print(f"  跨环境: {results[label]['cross_before']:.3f} → {results[label]['cross_after']:.3f}")
        print(f"  符号重叠: {results[label]['vocab_overlap_before']:.3f} → {results[label]['vocab_overlap_after']:.3f}")

    # 排序
    print("\n  环境差异 vs 翻译成功率（接触后）:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['cross_after'])
    for label, data in sorted_results:
        print(f"    {label}: 跨环境成功率={data['cross_after']:.3f}, "
              f"符号重叠={data['vocab_overlap_after']:.3f}")

    return results


# ============================================================
# 实验 4：通用语涌现（5 个环境群体）
# ============================================================

def experiment_4_lingua_franca():
    """实验 4：5 个不同环境群体融合后是否涌现通用语"""
    print("\n" + "=" * 60)
    print("实验 4：通用语涌现（5 个环境群体）")
    print("=" * 60)

    regions = [TROPICAL, ARCTIC, INDUSTRIAL, FOREST, MAGICAL]
    region_names = [r.name for r in regions]
    agents_per_region = 100

    # 创建所有 Agent
    all_agents = []
    region_agents = {}
    for region in regions:
        agents = [CrossLingualAgent(f"{region.name}_{i}", region.name)
                  for i in range(agents_per_region)]
        all_agents.extend(agents)
        region_agents[region.name] = agents

    print(f"  总 Agent 数: {len(all_agents)}")

    # 阶段 1：隔离期（各群体独立发展语言）
    print("\n  阶段 1：隔离期（800 轮）")
    for region in regions:
        agents = region_agents[region.name]
        adj = _build_small_world(agents)
        successes = 0
        for r in range(800):
            scene = generate_regional_scene(
                region.biases, num_objects=8,
                attribute_names=region.preferred_attributes)
            target_idx = random.randint(0, len(scene) - 1)
            a, b = _random_pair(agents, adj)
            speaker, listener = (a, b) if random.random() < 0.5 else (b, a)
            utt = speaker.speak(scene[target_idx], scene)
            chosen = listener.listen(utt, scene)
            success = (chosen == target_idx)
            speaker.update_from_communication(utt, success)
            listener.update_from_communication(utt, success)
            if success:
                successes += 1
        print(f"    {region.name}: 成功率={successes/800:.3f}")

    # 阶段 2：跨区域接触期
    print("\n  阶段 2：跨区域接触期（2000 轮）")
    contact_log = {'rounds': [], 'cross_success': [], 'intra_success': []}

    for r in range(0, 2000, 100):
        cross_success = 0
        cross_total = 0
        intra_success = 0
        intra_total = 0

        for _ in range(100):
            if random.random() < 0.5:
                # 群体内交流
                region = random.choice(regions)
                agents = region_agents[region.name]
                scene = generate_regional_scene(
                    region.biases, num_objects=8,
                    attribute_names=region.preferred_attributes)
                target_idx = random.randint(0, len(scene) - 1)
                a, b = _random_pair(agents, _build_small_world(agents))
                speaker, listener = (a, b) if random.random() < 0.5 else (b, a)
                utt = speaker.speak(scene[target_idx], scene)
                chosen = listener.listen(utt, scene)
                success = (chosen == target_idx)
                speaker.update_from_communication(utt, success)
                listener.update_from_communication(utt, success)
                if success:
                    intra_success += 1
                intra_total += 1
            else:
                # 跨区域交流
                region_a, region_b = random.sample(regions, 2)
                a = random.choice(region_agents[region_a.name])
                b = random.choice(region_agents[region_b.name])
                scene = generate_regional_scene(
                    region_a.biases, num_objects=8,
                    attribute_names=region_a.preferred_attributes)
                target_idx = random.randint(0, len(scene) - 1)
                utt = a.speak(scene[target_idx], scene)
                chosen = b.listen(utt, scene)
                success = (chosen == target_idx)
                a.update_from_communication(utt, success)
                b.update_from_communication(utt, success)
                if success:
                    cross_success += 1
                cross_total += 1

        sr_cross = cross_success / cross_total if cross_total > 0 else 0
        sr_intra = intra_success / intra_total if intra_total > 0 else 0
        contact_log['rounds'].append(r + 100)
        contact_log['cross_success'].append(sr_cross)
        contact_log['intra_success'].append(sr_intra)

        if (r + 100) % 500 == 0:
            print(f"    Round {r+100:5d}: 跨区域={sr_cross:.3f}, 群体内={sr_intra:.3f}")

    # 测量最终语言结构
    print("\n  最终语言结构分析:")
    region_vocabs = {}
    for region in regions:
        vocab = set()
        for a in region_agents[region.name]:
            vocab.update(a.language.vocabulary.keys())
        region_vocabs[region.name] = vocab

    # 两两比较
    for i, r1 in enumerate(region_names):
        for r2 in region_names[i+1:]:
            v1, v2 = region_vocabs[r1], region_vocabs[r2]
            overlap = len(v1 & v2)
            union = len(v1 | v2)
            ratio = overlap / union if union > 0 else 0
            print(f"    {r1} vs {r2}: 符号重叠={overlap}/{union}={ratio:.3f}")

    # 跨群体相似度
    print("\n  跨群体语言相似度:")
    for i, r1 in enumerate(region_names):
        for r2 in region_names[i+1:]:
            sims = []
            for _ in range(20):
                a = random.choice(region_agents[r1])
                b = random.choice(region_agents[r2])
                sim = compute_language_similarity(a.language, b.language)['overall_similarity']
                sims.append(sim)
            print(f"    {r1} vs {r2}: 相似度={np.mean(sims):.3f}")

    return contact_log


def _build_small_world(agents, k=4, p=0.1):
    """小世界网络"""
    ids = [a.id for a in agents]
    n = len(ids)
    adj_set = {aid: set() for aid in ids}
    half_k = k // 2
    for i in range(n):
        for d in range(1, half_k + 1):
            left = (i - d) % n
            right = (i + d) % n
            adj_set[ids[i]].add(ids[left])
            adj_set[ids[i]].add(ids[right])
            adj_set[ids[left]].add(ids[i])
            adj_set[ids[right]].add(ids[i])

    all_ids_set = set(ids)
    for i in range(n):
        neighbors = list(adj_set[ids[i]])
        for nb in neighbors:
            if random.random() < p:
                adj_set[ids[i]].discard(nb)
                adj_set[nb].discard(ids[i])
                candidates = all_ids_set - adj_set[ids[i]] - {ids[i]}
                if candidates:
                    new_id = random.choice(list(candidates))
                    adj_set[ids[i]].add(new_id)
                    adj_set[new_id].add(ids[i])

    return {aid: list(nbs) for aid, nbs in adj_set.items()}


def _random_pair(agents, adj):
    """随机选一对相邻 Agent"""
    a = random.choice(agents)
    neighbors = adj.get(a.id, [])
    if neighbors:
        b_id = random.choice(neighbors)
        b = next((x for x in agents if x.id == b_id), random.choice(agents))
    else:
        b = random.choice([x for x in agents if x.id != a.id])
    return a, b


if __name__ == '__main__':
    all_results = {}

    all_results['baseline'] = experiment_1_cross_env_baseline()
    all_results['bridge'] = experiment_2_bridge_translation()
    all_results['env_difference'] = experiment_3_env_difference()
    all_results['lingua_franca'] = experiment_4_lingua_franca()

    with open('cross_lingual_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print("\n\n结果已保存到 cross_lingual_results.json")
