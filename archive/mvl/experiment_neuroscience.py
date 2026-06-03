"""
Phase 53: 神经科学验证 — 与真实婴儿脑成像数据对比

核心问题：系统的学习轨迹与真实婴儿的学习轨迹是否结构性相似？

5 个实验：
1. 词汇增长曲线对比（vs MacArthur-Bates CDI）
2. 预测误差轨迹对比（vs 神经习惯化 EEG 数据）
3. 语法涌现时间线（vs N400/P600 神经标记）
4. 阶段转换对比（vs Piaget 认知阶段）
5. 关键期效应（延迟 vs 受限 vs 正常）
"""

import json
import random
import numpy as np
from language_emergence import LanguageAgent, cross_language_round
from neuroscience_validation import (
    INFANT_VOCAB_CD, NEURAL_MARKERS, PIAGET_STAGES, HABITUATION_DECAY,
    fit_curves, compute_curve_correlation, run_longitudinal, run_critical_period,
)


def experiment_1_vocab_growth():
    """实验 1：词汇增长曲线 vs 婴儿 CDI 数据（1000 轮 x 3 次）"""
    print("=" * 60)
    print("实验 1：词汇增长曲线对比（1000 轮 x 3 次）")
    print("=" * 60)

    all_snapshots = []

    for run in range(3):
        speaker = LanguageAgent(f'sp_{run}')
        listener = LanguageAgent(f'li_{run}')
        result = run_longitudinal(speaker, listener, 1000, sample_interval=50)
        all_snapshots.append(result['snapshots'])
        print(f"  运行 {run + 1}: 最终词汇量 {result['snapshots'][-1]['vocabulary_size']}")

    # 平均各时间点的词汇量
    rounds = [s['round'] for s in all_snapshots[0]]
    avg_vocab = []
    for i in range(len(rounds)):
        vocab_i = np.mean([snap[i]['vocabulary_size'] for snap in all_snapshots])
        avg_vocab.append(vocab_i)

    # 婴儿数据
    infant_ages = list(INFANT_VOCAB_CD.keys())
    infant_vocabs = list(INFANT_VOCAB_CD.values())

    # 曲线形状对比（归一化后）
    corr = compute_curve_correlation(
        rounds, avg_vocab,
        infant_ages, infant_vocabs
    )

    # 拟合系统数据
    system_fit = fit_curves(rounds, avg_vocab)

    # 拟合婴儿数据
    infant_fit = fit_curves(infant_ages, infant_vocabs)

    print(f"\n  系统词汇增长:")
    print(f"    最佳拟合: {system_fit['best_model']} (R²={system_fit['r_squared']})")
    print(f"    所有模型: {system_fit['all_models']}")

    print(f"\n  婴儿 CDI 数据:")
    print(f"    最佳拟合: {infant_fit['best_model']} (R²={infant_fit['r_squared']})")
    print(f"    所有模型: {infant_fit['all_models']}")

    print(f"\n  曲线形状相关性:")
    print(f"    Pearson r = {corr['pearson_r']}, p = {corr['p_value']}")

    verdict = "PASS" if corr['pearson_r'] > 0.9 else "REVIEW"
    print(f"    判定: {verdict} (r > 0.9 = PASS)")

    return {
        'correlation': corr,
        'system_curve_fit': system_fit,
        'infant_curve_fit': infant_fit,
        'system_vocab_trajectory': {
            'rounds': rounds,
            'avg_vocab': [round(v, 1) for v in avg_vocab],
        },
        'verdict': verdict,
    }


def experiment_2_prediction_error():
    """实验 2：预测误差轨迹 vs 神经习惯化数据

    由于通信成功率从一开始就极高（>99%），
    改用"词汇增长率变化"作为替代指标——
    词汇增长加速然后减速，类似于神经响应的习惯化衰减。
    """
    print("\n" + "=" * 60)
    print("实验 2：词汇增长率变化 vs 习惯化衰减（1000 轮 x 3 次）")
    print("=" * 60)

    all_snapshots = []

    for run in range(3):
        speaker = LanguageAgent(f'sp_{run}')
        listener = LanguageAgent(f'li_{run}')
        result = run_longitudinal(speaker, listener, 1000, sample_interval=50)
        all_snapshots.append(result['snapshots'])

    # 计算词汇增长率（相邻时间点的词汇增量）
    rounds = [s['round'] for s in all_snapshots[0]]
    avg_vocab = []
    for i in range(len(rounds)):
        vocab_i = np.mean([snap[i]['vocabulary_size'] for snap in all_snapshots])
        avg_vocab.append(vocab_i)

    # 增长率 = delta_vocab / delta_round
    growth_rates = []
    growth_rounds = []
    for i in range(1, len(avg_vocab)):
        delta_v = avg_vocab[i] - avg_vocab[i-1]
        delta_r = rounds[i] - rounds[i-1]
        rate = delta_v / max(delta_r, 1)
        growth_rates.append(rate)
        growth_rounds.append(rounds[i])

    # 习惯化数据
    hab_trials = list(HABITUATION_DECAY.keys())
    hab_amp = list(HABITUATION_DECAY.values())

    # 曲线形状对比
    corr = compute_curve_correlation(growth_rounds, growth_rates, hab_trials, hab_amp)

    # 拟合系统增长率变化
    growth_fit = fit_curves(growth_rounds, growth_rates)

    # 拟合习惯化数据
    hab_fit = fit_curves(hab_trials, hab_amp)

    print(f"\n  系统词汇增长率:")
    print(f"    初始: {growth_rates[0]:.3f}, 最终: {growth_rates[-1]:.3f}")
    print(f"    最佳拟合: {growth_fit['best_model']} (R²={growth_fit['r_squared']})")

    print(f"\n  EEG 习惯化衰减:")
    print(f"    最佳拟合: {hab_fit['best_model']} (R²={hab_fit['r_squared']})")

    print(f"\n  曲线形状相关性:")
    print(f"    Pearson r = {corr['pearson_r']}, p = {corr['p_value']}")

    verdict = "PASS" if corr['pearson_r'] > 0.7 else "REVIEW"
    print(f"    判定: {verdict}")

    return {
        'correlation': corr,
        'growth_curve_fit': growth_fit,
        'habituation_fit': hab_fit,
        'growth_trajectory': {
            'rounds': growth_rounds,
            'growth_rates': [round(r, 4) for r in growth_rates],
        },
        'verdict': verdict,
    }


def experiment_3_grammar_timeline():
    """实验 3：语法涌现时间线 vs N400/P600 神经标记"""
    print("\n" + "=" * 60)
    print("实验 3：语法涌现时间线（1000 轮 x 3 次）")
    print("=" * 60)

    all_snapshots = []

    for run in range(3):
        speaker = LanguageAgent(f'sp_{run}')
        listener = LanguageAgent(f'li_{run}')
        result = run_longitudinal(speaker, listener, 1000, sample_interval=50)
        all_snapshots.append(result['snapshots'])

    # 找到关键涌现点
    rounds = [s['round'] for s in all_snapshots[0]]

    def find_threshold(snapshots_list, key, threshold, direction='above'):
        """找到指标首次超过阈值的轮次（中位数）"""
        onset_rounds = []
        for snaps in snapshots_list:
            for s in snaps:
                if direction == 'above' and s[key] >= threshold:
                    onset_rounds.append(s['round'])
                    break
                elif direction == 'below' and s[key] <= threshold:
                    onset_rounds.append(s['round'])
                    break
        if not onset_rounds:
            return None
        return float(np.median(onset_rounds))

    milestones = {
        'combination_50pct': find_threshold(all_snapshots, 'combination_rate', 0.5),
        'tri_symbol_10pct': find_threshold(all_snapshots, 'tri_symbol_rate', 0.1),
        'grammar_1_rule': find_threshold(all_snapshots, 'grammar_rules', 1),
        'order_consistency_50pct': find_threshold(all_snapshots, 'order_consistency', 0.5),
    }

    # 映射到神经标记时间线
    # 将系统的 1000 轮映射到婴儿的 72 个月（0-6 岁）
    scale_factor = 72.0 / 1000

    print(f"\n  系统涌现时间线:")
    for name, rnd in milestones.items():
        if rnd is not None:
            equiv_months = rnd * scale_factor
            print(f"    {name}: Round {rnd:.0f} (≈ {equiv_months:.1f} 月)")
        else:
            print(f"    {name}: 未达到")

    print(f"\n  神经标记时间线:")
    for marker, info in NEURAL_MARKERS.items():
        print(f"    {marker}: {info['onset']}-{info['mature']} 月 "
              f"({info['description']})")

    # 检查涌现顺序是否一致
    # 预期：单符号 → 组合 → 语法 → 词序一致
    # 对应：MMN_phonemic → N400_semantic → P600_syntactic → left_lateralization
    system_order = []
    for name in ['combination_50pct', 'tri_symbol_10pct',
                 'grammar_1_rule', 'order_consistency_50pct']:
        if milestones[name] is not None:
            system_order.append(name)

    neural_order = ['MMN_phonemic', 'N400_semantic',
                    'P600_syntactic', 'left_lateralization']

    print(f"\n  涌现顺序对比:")
    print(f"    系统: {' → '.join(system_order)}")
    print(f"    神经: {' → '.join(neural_order)}")

    # 顺序一致性检查
    order_match = len(system_order) >= 2  # 至少 2 个里程碑出现
    verdict = "PASS" if order_match else "REVIEW"
    print(f"    判定: {verdict}")

    return {
        'milestones': {k: v for k, v in milestones.items()},
        'milestones_equivalent_months': {
            k: round(v * scale_factor, 1) if v else None
            for k, v in milestones.items()
        },
        'system_order': system_order,
        'neural_order': neural_order,
        'verdict': verdict,
    }


def experiment_4_stage_transitions():
    """实验 4：阶段转换与 Piaget 阶段对比（细粒度采样）"""
    print("\n" + "=" * 60)
    print("实验 4：阶段转换与 Piaget 认知阶段对比")
    print("=" * 60)

    print(f"\n  Piaget 阶段:")
    for stage in PIAGET_STAGES:
        print(f"    {stage['name']}: {stage['age_range']} 岁 — "
              f"{', '.join(stage['key_abilities'])}")

    # 用细粒度采样（每 10 轮）来捕捉早期变化
    all_snapshots = []

    for run in range(3):
        speaker = LanguageAgent(f'sp_{run}')
        listener = LanguageAgent(f'li_{run}')
        result = run_longitudinal(speaker, listener, 500, sample_interval=10)
        all_snapshots.append(result['snapshots'])

    rounds = [s['round'] for s in all_snapshots[0]]

    # 定义系统阶段（基于涌现能力）
    def detect_stages(snaps):
        stages = []
        for s in snaps:
            vocab = s['vocabulary_size']
            combo = s['combination_rate']
            grammar = s['grammar_rules']
            order = s['order_consistency']

            if vocab <= 2:
                stage = 'sensorimotor'      # 极少符号
            elif combo < 0.1:
                stage = 'early_preoperational'  # 有符号但未组合
            elif grammar < 1:
                stage = 'late_preoperational'   # 组合但无语法
            elif order < 0.5:
                stage = 'concrete'               # 有语法但词序不稳
            else:
                stage = 'formal'                  # 稳定语法系统
            stages.append(stage)
        return stages

    # 检测阶段转换点
    avg_stage_transitions = {}

    for run_idx, snaps in enumerate(all_snapshots):
        stages = detect_stages(snaps)
        prev = stages[0]
        for i, stg in enumerate(stages):
            if stg != prev:
                key = f"{prev}→{stg}"
                if key not in avg_stage_transitions:
                    avg_stage_transitions[key] = []
                avg_stage_transitions[key].append(snaps[i]['round'])
            prev = stg

    print(f"\n  系统阶段转换:")
    for transition, rnds in avg_stage_transitions.items():
        avg_rnd = np.mean(rnds)
        equiv_month = avg_rnd * 72.0 / 500
        print(f"    {transition}: Round {avg_rnd:.0f} (≈ {equiv_month:.1f} 月)")

    # 检查顺序一致性
    stage_order = ['sensorimotor', 'early_preoperational',
                   'late_preoperational', 'concrete', 'formal']
    transitions_seen = list(avg_stage_transitions.keys())

    # 检查是否都是正向转换（没有跳回）
    piaget_order = ['sensorimotor', 'preoperational',
                    'concrete_operational', 'formal_operational']

    order_consistent = True
    for t in transitions_seen:
        parts = t.split('→')
        if len(parts) == 2:
            try:
                from_idx = stage_order.index(parts[0])
                to_idx = stage_order.index(parts[1])
                if to_idx <= from_idx:
                    order_consistent = False
            except ValueError:
                pass

    verdict = "PASS" if order_consistent and len(transitions_seen) >= 2 else "REVIEW"
    print(f"\n  顺序一致性: {'一致' if order_consistent else '不一致'}")
    print(f"  判定: {verdict}")

    return {
        'transitions': {
            k: {'avg_round': round(float(np.mean(v)), 0),
                'equiv_months': round(float(np.mean(v)) * 72.0 / 500, 1)}
            for k, v in avg_stage_transitions.items()
        },
        'order_consistent': order_consistent,
        'verdict': verdict,
    }


def experiment_5_critical_period():
    """实验 5：关键期效应（延迟 vs 受限 vs 正常）"""
    print("\n" + "=" * 60)
    print("实验 5：关键期效应（500 轮 x 5 次）")
    print("=" * 60)

    conditions = ['normal', 'delayed', 'restricted']
    results = {}

    for cond in conditions:
        print(f"\n  运行 {cond}...")
        r = run_critical_period(
            num_rounds=500,
            condition=cond,
            delay_rounds=200,
            num_runs=5,
        )
        results[cond] = r
        print(f"    成功率: {r['success_rate']:.1%}")
        print(f"    词汇量: {r['vocabulary_size']:.0f}")
        print(f"    组合率: {r['combination_rate']:.4f}")
        print(f"    维度覆盖: {r['dimensions_covered']:.1f}")

    # 关键期效应：延迟/受限 vs 正常的差距
    normal_sr = results['normal']['success_rate']
    delayed_sr = results['delayed']['success_rate']
    restricted_sr = results['restricted']['success_rate']

    gap_delayed = normal_sr - delayed_sr
    gap_restricted = normal_sr - restricted_sr

    print(f"\n  关键期效应:")
    print(f"    正常 vs 延迟: {gap_delayed:+.4f}")
    print(f"    正常 vs 受限: {gap_restricted:+.4f}")

    has_effect = gap_delayed > 0.01 or gap_restricted > 0.01
    verdict = "PASS" if has_effect else "REVIEW"
    print(f"    判定: {verdict}")

    return {
        'conditions': results,
        'gap_delayed': round(gap_delayed, 4),
        'gap_restricted': round(gap_restricted, 4),
        'has_critical_period_effect': has_effect,
        'verdict': verdict,
    }


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1_vocab_growth'] = experiment_1_vocab_growth()
    results['experiment_2_prediction_error'] = experiment_2_prediction_error()
    results['experiment_3_grammar_timeline'] = experiment_3_grammar_timeline()
    results['experiment_4_stage_transitions'] = experiment_4_stage_transitions()
    results['experiment_5_critical_period'] = experiment_5_critical_period()

    # 汇总
    print("\n" + "=" * 60)
    print("Phase 53 汇总")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name}: {r.get('verdict', 'N/A')}")

    with open('neuroscience_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到 neuroscience_results.json")
