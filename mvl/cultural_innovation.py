"""
Phase 57: 累积文化创新 — 踩在巨人肩膀上

核心问题：
文化演化能否产生累积创新？下一代是否能在上一代的基础上超越？

人类文化的独特性不是"有文化"（黑猩猩也有），
而是"累积文化"——每一代都在上一代基础上创新。
站在巨人的肩膀上，而不是重新发明轮子。

Tennie et al. (2009) 的 ratchet effect（棘轮效应）：
文化知识像棘轮一样只进不退，每代累积。

实验设计：
1. 棘轮效应：N 代传递，每代面对更难任务，测量累积进步
2. 有传承 vs 无传承对比
3. 创新涌现：新一代是否能发现上一代未发现的概念？
4. 文化复杂度增长：语言的描述能力是否代际增长？
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple

from language_emergence import (
    LanguageAgent, cross_language_round, generate_rich_scene
)
from language_rich_scene import (
    generate_rich_scene_v2, ALL_ATTRIBUTE_NAMES
)


def _make_task_difficulty(generation: int, max_gen: int = 10):
    """
    根据代际生成场景难度

    代际 1: 4 物体, 2 属性 → 容易
    代际 5: 6 物体, 4 属性 → 中等
    代际 10: 8 物体, 6 属性 → 困难
    """
    num_objects = min(4 + generation, 10)
    num_attrs = min(2 + generation // 2, len(ALL_ATTRIBUTE_NAMES))
    attrs = random.sample(ALL_ATTRIBUTE_NAMES, num_attrs)
    return {
        'num_objects': num_objects,
        'attribute_names': attrs,
    }


def run_generation(speaker: LanguageAgent, listener: LanguageAgent,
                   generation: int, num_rounds: int = 200) -> dict:
    """运行一代的学习"""
    successes = 0
    descriptions_lengths = []
    vocab_snapshots = []

    for r in range(num_rounds):
        params = _make_task_difficulty(generation)
        scene = generate_rich_scene_v2(**params)
        target = random.randint(0, len(scene) - 1)

        success = cross_language_round(speaker, listener, scene, target)
        if success:
            successes += 1

    stats = speaker.language.get_stats()
    stats['generation'] = generation
    stats['success_rate_raw'] = successes / num_rounds
    stats['difficulty'] = params['num_objects']

    return stats


def transfer_knowledge(teacher: LanguageAgent, student: LanguageAgent,
                       transfer_ratio: float = 0.5):
    """
    知识传递：teacher 的部分词汇传递给 student

    模拟人类文化的"部分传承"——
    不是100%复制，而是传递核心知识。
    """
    teacher_lang = teacher.language
    student_lang = student.language

    for dim, stats in teacher_lang.dimension_stats.items():
        if random.random() < transfer_ratio:
            if dim not in student_lang.dimension_stats:
                student_lang.dimension_stats[dim] = {
                    'frequency': max(1, int(stats['frequency'] * 0.5)),
                    'successes': max(1, int(stats.get('successes', stats['frequency'] * 0.5))),
                    'success_rate': stats['success_rate'],
                }

    # 传递部分符号
    for sym, info in teacher_lang.vocabulary.items():
        if random.random() < transfer_ratio:
            if sym not in student_lang.vocabulary:
                student_lang.vocabulary[sym] = {
                    'frequency': max(1, info['frequency'] // 2),
                    'successes': max(1, info.get('successes', info['frequency'] // 2)),
                    'success_rate': info['success_rate'],
                }


# ============================================================
# 实验
# ============================================================

def experiment_1_ratchet_effect():
    """实验 1：棘轮效应（10 代 x 3 次，每代更难）"""
    print("=" * 60)
    print("实验 1：棘轮效应（10 代，难度递增 x 3 次）")
    print("=" * 60)

    num_gens = 10
    num_runs = 3
    results = {'with_transfer': [], 'without_transfer': []}

    for run in range(num_runs):
        # === 有传承 ===
        gen_results_transfer = []
        prev_speaker = None

        for gen in range(num_gens):
            speaker = LanguageAgent(f'sp_gen{gen}_r{run}')
            listener = LanguageAgent(f'li_gen{gen}_r{run}')

            if prev_speaker is not None:
                transfer_knowledge(prev_speaker, speaker, transfer_ratio=0.5)

            stats = run_generation(speaker, listener, gen, 200)
            gen_results_transfer.append(stats)
            prev_speaker = speaker

        results['with_transfer'].append(gen_results_transfer)

        # === 无传承 ===
        gen_results_no_transfer = []

        for gen in range(num_gens):
            speaker = LanguageAgent(f'sp_nt_gen{gen}_r{run}')
            listener = LanguageAgent(f'li_nt_gen{gen}_r{run}')
            # 不传承——每代从零开始

            stats = run_generation(speaker, listener, gen, 200)
            gen_results_no_transfer.append(stats)

        results['without_transfer'].append(gen_results_no_transfer)

    # 平均各代
    avg_transfer = []
    avg_no_transfer = []

    for gen in range(num_gens):
        t_sr = np.mean([r[gen]['success_rate'] for r in results['with_transfer']])
        t_vocab = np.mean([r[gen]['vocabulary_size'] for r in results['with_transfer']])
        t_diff = results['with_transfer'][0][gen]['difficulty']

        nt_sr = np.mean([r[gen]['success_rate'] for r in results['without_transfer']])
        nt_vocab = np.mean([r[gen]['vocabulary_size'] for r in results['without_transfer']])

        avg_transfer.append({
            'generation': gen,
            'success_rate': round(float(t_sr), 4),
            'vocabulary_size': round(float(t_vocab), 1),
            'difficulty': t_diff,
        })
        avg_no_transfer.append({
            'generation': gen,
            'success_rate': round(float(nt_sr), 4),
            'vocabulary_size': round(float(nt_vocab), 1),
        })

    print(f"\n  {'代际':>4} {'难度':>4} | {'传承SR':>8} {'传承V':>6} | {'无传承SR':>8} {'无传承V':>6}")
    print(f"  {'─' * 50}")
    for i in range(num_gens):
        t = avg_transfer[i]
        nt = avg_no_transfer[i]
        print(f"  {i:>4} {t['difficulty']:>4} | "
              f"{t['success_rate']:>8.1%} {t['vocabulary_size']:>6.0f} | "
              f"{nt['success_rate']:>8.1%} {nt['vocabulary_size']:>6.0f}")

    return {
        'with_transfer': avg_transfer,
        'without_transfer': avg_no_transfer,
    }


def experiment_2_innovation_discovery():
    """实验 2：创新涌现（下一代是否能发现新概念？）"""
    print("\n" + "=" * 60)
    print("实验 2：代际创新涌现（10 代 x 5 次）")
    print("=" * 60)

    num_gens = 10
    num_runs = 5
    innovation_rates = []

    for run in range(num_runs):
        prev_symbols = set()
        run_innovations = []

        prev_speaker = None

        for gen in range(num_gens):
            speaker = LanguageAgent(f'sp_inn_gen{gen}_r{run}')
            listener = LanguageAgent(f'li_inn_gen{gen}_r{run}')

            if prev_speaker is not None:
                transfer_knowledge(prev_speaker, speaker, transfer_ratio=0.5)

            stats = run_generation(speaker, listener, gen, 200)

            new_symbols = set(speaker.language.vocabulary.keys()) - prev_symbols
            innovation_rate = len(new_symbols) / max(1, len(speaker.language.vocabulary))

            run_innovations.append({
                'generation': gen,
                'total_symbols': len(speaker.language.vocabulary),
                'new_symbols': len(new_symbols),
                'innovation_rate': round(innovation_rate, 4),
            })

            prev_symbols = set(speaker.language.vocabulary.keys())
            prev_speaker = speaker

        innovation_rates.append(run_innovations)

    # 平均
    avg = []
    for gen in range(num_gens):
        total = np.mean([r[gen]['total_symbols'] for r in innovation_rates])
        new = np.mean([r[gen]['new_symbols'] for r in innovation_rates])
        rate = np.mean([r[gen]['innovation_rate'] for r in innovation_rates])
        avg.append({
            'generation': gen,
            'total_symbols': round(float(total), 1),
            'new_symbols': round(float(new), 1),
            'innovation_rate': round(float(rate), 4),
        })
        print(f"  Gen {gen}: 总符号={total:.0f}, 新增={new:.1f}, "
              f"创新率={rate:.1%}")

    return avg


def experiment_3_transfer_ratio():
    """实验 3：传承比例效果（0%→100%）"""
    print("\n" + "=" * 60)
    print("实验 3：传承比例效果（5 代 x 5 次）")
    print("=" * 60)

    ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}

    for ratio in ratios:
        run_sr = []
        run_vocab = []

        for run in range(5):
            prev_speaker = None

            for gen in range(5):
                speaker = LanguageAgent(f'sp_tr_gen{gen}_r{run}')
                listener = LanguageAgent(f'li_tr_gen{gen}_r{run}')

                if prev_speaker is not None:
                    transfer_knowledge(prev_speaker, speaker,
                                      transfer_ratio=ratio)

                stats = run_generation(speaker, listener, gen, 200)
                prev_speaker = speaker

            # 最后一代的性能
            run_sr.append(stats['success_rate'])
            run_vocab.append(stats['vocabulary_size'])

        results[str(ratio)] = {
            'final_sr': round(float(np.mean(run_sr)), 4),
            'final_vocab': round(float(np.mean(run_vocab)), 1),
        }
        print(f"  ratio={ratio}: 最终成功率={np.mean(run_sr):.1%}, "
              f"词汇量={np.mean(run_vocab):.0f}")

    return results


def experiment_4_complexity_growth():
    """实验 4：文化复杂度增长（组合率/语法/三符号）"""
    print("\n" + "=" * 60)
    print("实验 4：文化复杂度增长（10 代 x 3 次）")
    print("=" * 60)

    num_gens = 10
    num_runs = 3
    all_gen_data = []

    for run in range(num_runs):
        prev_speaker = None
        run_data = []

        for gen in range(num_gens):
            speaker = LanguageAgent(f'sp_cx_gen{gen}_r{run}')
            listener = LanguageAgent(f'li_cx_gen{gen}_r{run}')

            if prev_speaker is not None:
                transfer_knowledge(prev_speaker, speaker, transfer_ratio=0.5)

            stats = run_generation(speaker, listener, gen, 200)
            run_data.append(stats)
            prev_speaker = speaker

        all_gen_data.append(run_data)

    # 平均
    avg = []
    for gen in range(num_gens):
        combo = np.mean([r[gen]['combination_rate'] for r in all_gen_data])
        tri = np.mean([r[gen]['tri_symbol_rate'] for r in all_gen_data])
        grammar = np.mean([r[gen]['grammar_rules'] for r in all_gen_data])
        order = np.mean([r[gen]['order_consistency'] for r in all_gen_data])
        avg.append({
            'generation': gen,
            'combination_rate': round(float(combo), 4),
            'tri_symbol_rate': round(float(tri), 4),
            'grammar_rules': round(float(grammar), 1),
            'order_consistency': round(float(order), 4),
        })
        print(f"  Gen {gen}: 组合={combo:.2%}, 三符号={tri:.2%}, "
              f"语法={grammar:.1f}, 词序={order:.2%}")

    return avg


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_ratchet_effect()
    results['experiment_2'] = experiment_2_innovation_discovery()
    results['experiment_3'] = experiment_3_transfer_ratio()
    results['experiment_4'] = experiment_4_complexity_growth()

    with open('cultural_innovation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到 cultural_innovation_results.json")
