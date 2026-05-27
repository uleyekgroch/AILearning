"""
跨代知识传递实验

4 个实验：
1. 单代传递效果：有教学 vs 无教学
2. 多代积累：4 代传递，观察知识积累
3. 传递机制对比：直接交流 vs 知识播种 vs 对照
4. 教学时长影响：100/300/500/1000 轮
"""

import sys
import numpy as np
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

from generational_transfer import (
    GenerationalChain, compare_transfer_modes, vary_teaching_duration
)


def run_single_gen_experiment(seed: int = 42) -> Dict:
    """
    实验 1：单代传递效果

    对照组：3 个新 agent，从零学习 2 轮
    实验组：3 个老 agent（500轮经验）教 3 个新 agent 2 轮
    """
    print("  对照组：从零学习 2 轮")
    chain_none = GenerationalChain(
        population_size=3,
        rounds_per_generation=2,
        teaching_rounds=0,
        transfer_mode='none',
        complexity='extreme',
    )
    # 第1代：从零学习
    chain_none.run_single_generation(0, seed=seed)
    # 第2代：也从零学习（对照）
    result_none = chain_none.run_single_generation(1, seed=seed + 100)

    print("  实验组：从老 agent 学习 2 轮")
    chain_direct = GenerationalChain(
        population_size=3,
        rounds_per_generation=500,  # 老 agent 学习 500 轮
        teaching_rounds=2,
        transfer_mode='direct',
        complexity='extreme',
    )
    # 第1代：学习 500 轮
    chain_direct.run_single_generation(0, seed=seed)
    # 第2代：从第1代学习 2 轮
    result_direct = chain_direct.run_single_generation(1, seed=seed + 100)

    return {
        'control': {
            'eval': result_none['evaluation'],
            'agents': result_none['agents'],
        },
        'experiment': {
            'eval': result_direct['evaluation'],
            'teaching': result_direct['teaching'],
            'agents': result_direct['agents'],
        },
    }


def run_accumulation_experiment(num_generations: int = 4,
                                seed: int = 42) -> Dict:
    """
    实验 2：多代积累

    第1代：从零学习 2 轮
    第2代：从第1代学习 2 轮
    第3代：从第2代学习 2 轮
    第4代：从第3代学习 2 轮
    """
    chain = GenerationalChain(
        population_size=3,
        rounds_per_generation=2,
        teaching_rounds=2,
        transfer_mode='direct',
        complexity='extreme',
    )

    results = chain.run_chain(num_generations, seed)
    return results


def run_mechanism_comparison_experiment(seed: int = 42) -> Dict:
    """
    实验 3：传递机制对比

    条件 A：直接交流（老 agent 教新 agent）
    条件 B：知识播种（复制语言统计）
    条件 C：对照（新 agent 从零学习）
    """
    results = compare_transfer_modes(
        num_generations=4,
        population_size=3,
        rounds_per_generation=2,
        teaching_rounds=2,
        complexity='extreme',
        seed=seed,
    )
    return results


def run_duration_experiment(seed: int = 42) -> Dict:
    """
    实验 4：教学时长的影响

    教学 1 轮 vs 2 轮 vs 5 轮 vs 10 轮
    """
    results = vary_teaching_duration(
        teaching_rounds_list=[1, 2, 5, 10],
        num_generations=3,
        population_size=3,
        rounds_per_generation=2,
        complexity='extreme',
        seed=seed,
    )
    return results


def print_generation_stats(generation: Dict, label: str = ""):
    """打印一代的统计信息"""
    eval_r = generation['evaluation']
    speed = eval_r.get('learning_speed', {})
    thresholds = speed.get('rounds_to_threshold', {})
    prefix = f"  {label}" if label else "  "
    t80 = thresholds.get(0.8, 'N/A')
    t90 = thresholds.get(0.9, 'N/A')
    print(f"{prefix}最终={eval_r['success_rate']:.1%} "
          f"词汇={eval_r['vocabulary_size']} "
          f"达80%={t80}轮 达90%={t90}轮")


def main():
    print("跨代知识传递实验：文化的积累与传承")
    print("=" * 70)

    # === 实验 1：单代传递效果 ===
    print("\n实验 1：单代传递效果（有教学 vs 无教学）")
    print("-" * 70)
    single_result = run_single_gen_experiment(seed=42)

    ctrl = single_result['control']['eval']
    exp = single_result['experiment']['eval']
    ctrl_speed = ctrl.get('learning_speed', {})
    exp_speed = exp.get('learning_speed', {})
    ctrl_t = ctrl_speed.get('rounds_to_threshold', {})
    exp_t = exp_speed.get('rounds_to_threshold', {})

    print(f"\n  对照组（无教学）:")
    print(f"    最终成功率: {ctrl['success_rate']:.1%}")
    print(f"    词汇量: {ctrl['vocabulary_size']}")
    print(f"    达80%所需轮次: {ctrl_t.get(0.8, 'N/A')}")
    print(f"    达90%所需轮次: {ctrl_t.get(0.9, 'N/A')}")

    print(f"\n  实验组（有教学）:")
    print(f"    最终成功率: {exp['success_rate']:.1%}")
    print(f"    词汇量: {exp['vocabulary_size']}")
    print(f"    达80%所需轮次: {exp_t.get(0.8, 'N/A')}")
    print(f"    达90%所需轮次: {exp_t.get(0.9, 'N/A')}")

    # 学习速度改善
    if ctrl_t.get(0.8) and exp_t.get(0.8):
        speedup = (ctrl_t[0.8] - exp_t[0.8]) / ctrl_t[0.8]
        print(f"\n  学习速度改善（达80%）: {speedup:+.1%}")

    # === 实验 2：多代积累 ===
    print("\n" + "=" * 70)
    print("实验 2：多代积累（4 代，直接交流）")
    print("-" * 70)
    accum_results = run_accumulation_experiment(num_generations=4, seed=42)

    print("\n  代际对比:")
    for i, gen in enumerate(accum_results):
        eval_r = gen['evaluation']
        speed = eval_r.get('learning_speed', {})
        thresholds = speed.get('rounds_to_threshold', {})
        t80 = thresholds.get(0.8, 'N/A')
        t90 = thresholds.get(0.9, 'N/A')
        print(f"    第{i}代: 最终={eval_r['success_rate']:.1%} "
              f"词汇={eval_r['vocabulary_size']} "
              f"达80%={t80}轮 达90%={t90}轮")

    # 计算积累效果
    if len(accum_results) >= 2:
        first_speed = accum_results[0]['evaluation'].get('learning_speed', {})
        last_speed = accum_results[-1]['evaluation'].get('learning_speed', {})
        first_t80 = first_speed.get('rounds_to_threshold', {}).get(0.8)
        last_t80 = last_speed.get('rounds_to_threshold', {}).get(0.8)
        print(f"\n  积累效果（第0代 → 第{len(accum_results)-1}代）:")
        print(f"    达80%所需轮次: {first_t80} → {last_t80}")
        if first_t80 and last_t80:
            improvement = (first_t80 - last_t80) / first_t80
            print(f"    学习速度改善: {improvement:+.1%}")

    # === 实验 3：传递机制对比 ===
    print("\n" + "=" * 70)
    print("实验 3：传递机制对比（4 代）")
    print("-" * 70)
    mechanism_results = run_mechanism_comparison_experiment(seed=42)

    print("\n  各机制代际对比:")
    for mode in ['direct', 'seed', 'none']:
        print(f"\n  {mode}:")
        for i, gen in enumerate(mechanism_results[mode]):
            eval_r = gen['evaluation']
            speed = eval_r.get('learning_speed', {})
            thresholds = speed.get('rounds_to_threshold', {})
            t80 = thresholds.get(0.8, 'N/A')
            t90 = thresholds.get(0.9, 'N/A')
            print(f"    第{i}代: 最终={eval_r['success_rate']:.1%} "
                  f"词汇={eval_r['vocabulary_size']} "
                  f"达80%={t80}轮 达90%={t90}轮")

    # === 实验 4：教学时长影响 ===
    print("\n" + "=" * 70)
    print("实验 4：教学时长影响（3 代）")
    print("-" * 70)
    duration_results = run_duration_experiment(seed=42)

    print("\n  各时长最终代对比:")
    for duration in [1, 2, 5, 10]:
        last_gen = duration_results[duration][-1]
        eval_r = last_gen['evaluation']
        print(f"    {duration:4d}轮: 最终={eval_r['success_rate']:.1%} "
              f"词汇={eval_r['vocabulary_size']}")

    # === 汇总 ===
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)

    # 实验 1 汇总
    print("\n  实验 1 - 单代传递:")
    ctrl = single_result['control']['eval']
    exp = single_result['experiment']['eval']
    ctrl_speed = ctrl.get('learning_speed', {})
    exp_speed = exp.get('learning_speed', {})
    ctrl_t80 = ctrl_speed.get('rounds_to_threshold', {}).get(0.8)
    exp_t80 = exp_speed.get('rounds_to_threshold', {}).get(0.8)
    if ctrl_t80 and exp_t80 and exp_t80 < ctrl_t80:
        print(f"    ✓ 有教学加速学习：达80%需 {exp_t80} 轮 vs {ctrl_t80} 轮")
    else:
        print(f"    △ 传递效果不明显")

    # 实验 2 汇总
    print("\n  实验 2 - 多代积累:")
    if len(accum_results) >= 2:
        first_speed = accum_results[0]['evaluation'].get('learning_speed', {})
        last_speed = accum_results[-1]['evaluation'].get('learning_speed', {})
        first_t80 = first_speed.get('rounds_to_threshold', {}).get(0.8)
        last_t80 = last_speed.get('rounds_to_threshold', {}).get(0.8)
        if first_t80 and last_t80 and last_t80 < first_t80:
            print(f"    ✓ 学习速度逐代提升：达80% 第0代={first_t80}轮 → 第{len(accum_results)-1}代={last_t80}轮")
        elif first_t80 and last_t80:
            print(f"    △ 学习速度无明显变化")
        else:
            print(f"    △ 数据不足")

    # 实验 3 汇总
    print("\n  实验 3 - 传递机制:")
    mechanism_t80 = {}
    for mode in ['direct', 'seed', 'none']:
        final = mechanism_results[mode][-1]['evaluation']
        speed = final.get('learning_speed', {})
        t80 = speed.get('rounds_to_threshold', {}).get(0.8)
        mechanism_t80[mode] = t80
    valid_modes = {k: v for k, v in mechanism_t80.items() if v is not None}
    if valid_modes:
        best_mode = min(valid_modes, key=valid_modes.get)
        print(f"    最佳机制: {best_mode} (达80%需 {valid_modes[best_mode]} 轮)")
    else:
        print(f"    △ 数据不足")

    # 实验 4 汇总
    print("\n  实验 4 - 教学时长:")
    duration_vocab = {}
    for duration in [1, 2, 5, 10]:
        last_gen = duration_results[duration][-1]
        duration_vocab[duration] = last_gen['evaluation']['vocabulary_size']
    best_duration = max(duration_vocab, key=duration_vocab.get)
    print(f"    词汇量随教学时长增长: {duration_vocab}")
    print(f"    最佳时长: {best_duration}轮 (词汇={duration_vocab[best_duration]})")

    # === 保存结果 ===
    import json

    def to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    output = {
        'single_gen': {
            'control': to_serializable(single_result['control']['eval']),
            'experiment': to_serializable(single_result['experiment']['eval']),
        },
        'accumulation': [
            {
                'generation': gen['generation'],
                'evaluation': to_serializable(gen['evaluation']),
            }
            for gen in accum_results
        ],
        'mechanism': {
            mode: [
                {
                    'generation': gen['generation'],
                    'evaluation': to_serializable(gen['evaluation']),
                }
                for gen in gens
            ]
            for mode, gens in mechanism_results.items()
        },
        'duration': {
            str(dur): [
                {
                    'generation': gen['generation'],
                    'evaluation': to_serializable(gen['evaluation']),
                }
                for gen in gens
            ]
            for dur, gens in duration_results.items()
        },
    }

    with open('D:/mayAi/AILearning_v0527/mvl/generational_results.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存到: generational_results.json")


if __name__ == '__main__':
    main()
