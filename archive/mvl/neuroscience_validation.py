"""
神经科学验证 — 与真实婴儿脑成像数据对比

核心问题：
系统的学习轨迹与真实婴儿的学习轨迹是否结构性相似？
如果系统真正从"学习本源"出发，即使时间尺度不同，
学习的拓扑结构应该一致——曲线形状、阶段转换模式、关键期效应。

方法：
- 归一化两条曲线到 [0, 1]
- 计算 Pearson 相关系数（曲线形状相似性）
- 拟合曲线类型（线性、指数、sigmoid、幂律）
- 报告相关性，不做因果声明

数据来源：
- 词汇量：MacArthur-Bates CDI (Fenson et al., 1994)
- 神经标记：EEG/NIRS 综述 (Kuhl, 2010; Friederici, 2011)
- 发展阶段：Piaget (1952)
- 关键期：Hartshorne et al. (2018, Cognition)
"""

import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from language_emergence import (
    LanguageAgent, EmergingLanguage, cross_language_round,
    generate_rich_scene
)
from language_rich_scene import (
    generate_rich_scene_v2, ALL_ATTRIBUTE_NAMES
)


# ============================================================
# 文献数据（来自已发表的表格和综述）
# ============================================================

# 产出性词汇量（MacArthur-Bates CDI, 英语, 50 百分位）
# Fenson et al. (1994), 也见于 Hart & Risley (1995)
INFANT_VOCAB_CD = {
    # age_months: production_vocab_count
    8: 0,
    10: 1,
    12: 5,
    14: 15,
    16: 40,
    18: 50,      # 词汇爆发开始
    20: 120,
    22: 200,
    24: 250,
    27: 400,
    30: 500,
    33: 650,
    36: 800,
    42: 1100,
    48: 1500,
    54: 2200,
    60: 3000,
    66: 4000,
    72: 5000,
}

# 神经标记涌现时间线（月）
# Kushnerenko et al. (2002), Friedrich & Friederici (2005), Kuhl (2010)
NEURAL_MARKERS = {
    'MMN_acoustic': {'onset': 0, 'mature': 1,
                     'analogue': 'prediction_error',
                     'description': '声音变化检测'},
    'MMN_phonemic': {'onset': 3, 'mature': 6,
                     'analogue': 'single_symbol_match',
                     'description': '母语音素辨别'},
    'N400_semantic': {'onset': 12, 'mature': 18,
                      'analogue': 'combination_rate',
                      'description': '语义处理（词汇组合）'},
    'P600_syntactic': {'onset': 24, 'mature': 30,
                       'analogue': 'grammar_rules',
                       'description': '句法处理（语法规则）'},
    'theta_word_learning': {'onset': 12, 'mature': 24,
                            'analogue': 'vocabulary_growth_rate',
                            'description': '词汇学习加速'},
    'left_lateralization': {'onset': 24, 'mature': 36,
                            'analogue': 'order_consistency',
                            'description': '语言左半球偏侧化'},
}

# Piaget 认知发展阶段
PIAGET_STAGES = [
    {'name': 'sensorimotor', 'age_range': (0, 2),
     'key_abilities': ['客体永久性', '因果推理', '模仿']},
    {'name': 'preoperational', 'age_range': (2, 7),
     'key_abilities': ['符号思维', '自我中心', '分类']},
    {'name': 'concrete_operational', 'age_range': (7, 11),
     'key_abilities': ['守恒', '序列化', '传递性推理']},
    {'name': 'formal_operational', 'age_range': (11, 17),
     'key_abilities': ['假设演绎', '抽象推理', '元认知']},
]

# 神经习惯化数据（EEG 响应振幅衰减）
# 重复刺激后响应振幅的归一化衰减（近似值）
# Dehaene-Lambertz & Dehaene (1994), Gervain et al. (2008)
HABITUATION_DECAY = {
    # trial_number: normalized_response_amplitude
    1: 1.0,
    2: 0.85,
    3: 0.72,
    4: 0.65,
    5: 0.55,
    6: 0.50,
    7: 0.46,
    8: 0.43,
    9: 0.40,
    10: 0.38,
}


# ============================================================
# 曲线拟合工具
# ============================================================

def _sigmoid(x, L, k, x0, b):
    """Sigmoid 函数: L / (1 + exp(-k * (x - x0))) + b"""
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def _exponential(x, a, b, c):
    """指数衰减: a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c


def _power_law(x, a, b, c):
    """幂律: a * x ** (-b) + c"""
    return a * np.power(x + 1e-10, -b) + c


def _linear(x, a, b):
    """线性: a * x + b"""
    return a * x + b


def fit_curves(x_data, y_data):
    """
    拟合 4 种曲线模型，返回最佳拟合

    返回: {'best_model': str, 'r_squared': float, 'params': dict}
    """
    # 归一化
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)

    if len(x) < 4:
        return {'best_model': 'insufficient_data', 'r_squared': 0}

    results = {}
    models = {
        'sigmoid': (_sigmoid, [1.0, 0.1, np.median(x), 0.0],
                    {'maxfev': 5000}),
        'exponential': (_exponential, [1.0, 0.1, 0.0],
                        {'maxfev': 5000}),
        'power_law': (_power_law, [1.0, 0.5, 0.0],
                      {'maxfev': 5000}),
        'linear': (_linear, [0.1, 0.0], {}),
    }

    for name, (func, p0, kwargs) in models.items():
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, **kwargs)
            y_pred = func(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            results[name] = {'r_squared': r2, 'params': popt.tolist()}
        except (RuntimeError, ValueError):
            results[name] = {'r_squared': -1, 'params': None}

    best = max(results.items(), key=lambda kv: kv[1]['r_squared'])
    return {
        'best_model': best[0],
        'r_squared': round(best[1]['r_squared'], 4),
        'all_models': {k: round(v['r_squared'], 4) for k, v in results.items()
                       if v['r_squared'] >= 0},
    }


def compute_curve_correlation(x_a, y_a, x_b, y_b):
    """
    比较两条曲线的形状相似性

    在 x_a 和 x_b 的共同范围内插值，归一化后计算 Pearson r。
    """
    # 共同范围
    x_min = max(min(x_a), min(x_b))
    x_max = min(max(x_a), max(x_b))

    if x_max <= x_min:
        return {'pearson_r': 0, 'p_value': 1.0, 'note': 'no_overlap'}

    # 插值
    n_points = min(20, len(x_a), len(x_b))
    x_common = np.linspace(x_min, x_max, n_points)

    ya_interp = np.interp(x_common, x_a, y_a)
    yb_interp = np.interp(x_common, x_b, y_b)

    # 归一化到 [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    ya_norm = normalize(ya_interp)
    yb_norm = normalize(yb_interp)

    r, p = pearsonr(ya_norm, yb_norm)
    return {'pearson_r': round(float(r), 4), 'p_value': round(float(p), 6)}


# ============================================================
# 长期学习追踪器
# ============================================================

def run_longitudinal(speaker: LanguageAgent, listener: LanguageAgent,
                     num_rounds: int, sample_interval: int = 50) -> dict:
    """
    运行长期学习实验，定期采样所有指标

    返回时间序列: vocab_sizes, success_rates, combination_rates,
                 tri_rates, grammar_counts, order_consistencies
    """
    snapshots = []

    for r in range(num_rounds):
        # 混合场景（基础 + 丰富）
        if random.random() < 0.5:
            scene = generate_rich_scene()
        else:
            attrs = random.sample(ALL_ATTRIBUTE_NAMES,
                                  min(4, len(ALL_ATTRIBUTE_NAMES)))
            scene = generate_rich_scene_v2(
                num_objects=random.randint(4, 8),
                attribute_names=attrs
            )

        target_idx = random.randint(0, len(scene) - 1)
        cross_language_round(speaker, listener, scene, target_idx)

        if (r + 1) % sample_interval == 0:
            stats = speaker.language.get_stats()
            snapshots.append({
                'round': r + 1,
                'vocabulary_size': stats['vocabulary_size'],
                'success_rate': stats['success_rate'],
                'combination_rate': stats['combination_rate'],
                'tri_symbol_rate': stats['tri_symbol_rate'],
                'grammar_rules': stats['grammar_rules'],
                'order_consistency': stats['order_consistency'],
                'ngram_patterns': stats['ngram_patterns'],
            })

    return {'snapshots': snapshots}


# ============================================================
# 关键期测试
# ============================================================

def run_critical_period(num_rounds: int = 500, condition: str = 'normal',
                        delay_rounds: int = 200,
                        num_runs: int = 5) -> dict:
    """
    关键期效应测试

    条件：
    - 'normal': 从第 0 轮开始学习
    - 'delayed': 前 delay_rounds 轮无语言暴露（只用空场景）
    - 'restricted': 前 delay_rounds 轮只暴露于 color+shape
    """
    run_results = []

    for run in range(num_runs):
        speaker = LanguageAgent(f'sp_{run}')
        listener = LanguageAgent(f'li_{run}')

        for r in range(num_rounds):
            if condition == 'delayed' and r < delay_rounds:
                # 无语言暴露：跳过通信游戏
                continue
            elif condition == 'restricted' and r < delay_rounds:
                # 受限环境：只有 color + shape
                scene = generate_rich_scene_v2(
                    num_objects=4, attribute_names=['color', 'shape']
                )
            else:
                # 正常：混合场景
                if random.random() < 0.5:
                    scene = generate_rich_scene()
                else:
                    attrs = random.sample(ALL_ATTRIBUTE_NAMES, 4)
                    scene = generate_rich_scene_v2(
                        num_objects=random.randint(4, 8),
                        attribute_names=attrs
                    )

            target_idx = random.randint(0, len(scene) - 1)
            cross_language_round(speaker, listener, scene, target_idx)

        final_stats = speaker.language.get_stats()
        dims = len(speaker.language.dimension_stats)
        run_results.append({
            'success_rate': final_stats['success_rate'],
            'vocabulary_size': final_stats['vocabulary_size'],
            'combination_rate': final_stats['combination_rate'],
            'dimensions_covered': dims,
        })

    return {
        'condition': condition,
        'success_rate': round(float(np.mean([r['success_rate'] for r in run_results])), 4),
        'vocabulary_size': round(float(np.mean([r['vocabulary_size'] for r in run_results])), 1),
        'combination_rate': round(float(np.mean([r['combination_rate'] for r in run_results])), 4),
        'dimensions_covered': round(float(np.mean([r['dimensions_covered'] for r in run_results])), 1),
        'num_runs': num_runs,
    }
