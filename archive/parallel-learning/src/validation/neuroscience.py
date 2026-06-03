"""神经科学验证 — 与真实婴儿发展数据对比

对比维度：
1. MacArthur-Bates CDI 词汇增长曲线
2. EEG 神经标记时间线 (MMN, N400, P600...)
3. Piaget 认知阶段转换
4. 曲线拟合 + Pearson 相关性分析
"""

import math
from typing import Dict, List, Optional, Tuple

try:
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# 真实婴儿数据（来自已发表文献）
# ============================================================

# MacArthur-Bates CDI 产出性词汇量（月龄 → 平均词汇数）
INFANT_VOCAB_CDI = {
    8: 0, 10: 1, 12: 5, 14: 15, 16: 50,
    18: 100, 20: 200, 22: 300, 24: 450,
    26: 550, 28: 650, 30: 750, 32: 850,
    36: 1000, 42: 1500, 48: 2000, 60: 3000,
    72: 4000,
}

# 神经标记时间线
NEURAL_MARKERS = [
    {
        'name': 'MMN_acoustic',
        'onset_months': 0,
        'mature_months': 6,
        'mvl_analogue': 'prediction_accuracy',
        'description': '失配负波 — 声学变化检测',
    },
    {
        'name': 'MMN_phonemic',
        'onset_months': 6,
        'mature_months': 12,
        'mvl_analogue': 'symbol_grounding_rate',
        'description': '失配负波 — 音位区分',
    },
    {
        'name': 'N400_semantic',
        'onset_months': 12,
        'mature_months': 24,
        'mvl_analogue': 'vocabulary_size',
        'description': 'N400 — 语义处理',
    },
    {
        'name': 'P600_syntactic',
        'onset_months': 18,
        'mature_months': 36,
        'mvl_analogue': 'grammar_complexity',
        'description': 'P600 — 句法处理',
    },
    {
        'name': 'theta_word_learning',
        'onset_months': 10,
        'mature_months': 20,
        'mvl_analogue': 'learning_progress',
        'description': 'Theta 振荡 — 词汇学习',
    },
    {
        'name': 'left_lateralization',
        'onset_months': 24,
        'mature_months': 48,
        'mvl_analogue': 'composition_rate',
        'description': '左半球偏侧化 — 语言特化',
    },
]

# Piaget 认知阶段
PIAGET_STAGES = [
    {'name': '感知运动期', 'age_range': (0, 24), 'key_abilities': ['object_permanence', 'goal_directed']},
    {'name': '前运算期', 'age_range': (24, 72), 'key_abilities': ['symbolic_play', 'language']},
    {'name': '具体运算期', 'age_range': (72, 132), 'key_abilities': ['conservation', 'classification']},
    {'name': '形式运算期', 'age_range': (132, 192), 'key_abilities': ['abstract_reasoning', 'hypothesis']},
]

# 习惯化衰减曲线（EEG 试验 → 振幅）
HABITUATION_DECAY = {
    1: 1.0, 2: 0.9, 3: 0.78, 4: 0.65, 5: 0.55,
    6: 0.48, 7: 0.42, 8: 0.38, 9: 0.35, 10: 0.33,
}


# ============================================================
# 曲线拟合
# ============================================================

def _sigmoid(x: float, L: float, k: float, x0: float, b: float) -> float:
    return L / (1.0 + math.exp(-k * (x - x0))) + b


def _exponential(x: float, a: float, b: float, c: float) -> float:
    return a * math.exp(b * x) + c


def _power_law(x: float, a: float, b: float, c: float) -> float:
    return a * (x ** b) + c


def _linear(x: float, a: float, b: float) -> float:
    return a * x + b


def fit_curves(x_data: List[float], y_data: List[float]) -> Dict:
    """拟合 4 种曲线模型，返回最优模型

    需要 scipy。返回 {'best_model': name, 'params': [...], 'r_squared': float, 'all_models': {...}}
    """
    if not HAS_SCIPY:
        return {'best_model': 'none', 'error': 'scipy not available'}

    models = {
        'sigmoid': (_sigmoid, [1000.0, 0.1, 24.0, 0.0]),
        'exponential': (_exponential, [1.0, 0.1, 0.0]),
        'power_law': (_power_law, [1.0, 1.5, 0.0]),
        'linear': (_linear, [50.0, -100.0]),
    }

    results = {}
    best_name = None
    best_r2 = -float('inf')

    for name, (func, p0) in models.items():
        try:
            params, _ = curve_fit(func, x_data, y_data, p0=p0, maxfev=5000)
            y_pred = [func(x, *params) for x in x_data]
            ss_res = sum((y - yp) ** 2 for y, yp in zip(y_data, y_pred))
            ss_tot = sum((y - sum(y_data) / len(y_data)) ** 2 for y in y_data)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            results[name] = {'params': list(params), 'r_squared': r2}
            if r2 > best_r2:
                best_r2 = r2
                best_name = name
        except Exception:
            results[name] = {'params': None, 'r_squared': -1.0}

    return {
        'best_model': best_name,
        'params': results.get(best_name, {}).get('params'),
        'r_squared': best_r2,
        'all_models': results,
    }


def compute_correlation(system_data: Dict[float, float],
                        infant_data: Dict[float, float]) -> Dict:
    """计算系统学习曲线与婴儿数据的 Pearson 相关性

    在公共 x 范围内插值后计算。
    """
    common_x = sorted(set(system_data.keys()) & set(infant_data.keys()))
    if len(common_x) < 3:
        return {'correlation': 0.0, 'p_value': 1.0, 'n_points': len(common_x)}

    sys_y = [system_data[x] for x in common_x]
    inf_y = [infant_data[x] for x in common_x]

    # 归一化到 [0, 1]
    def normalize(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn) if mx > mn else 0.5 for v in vals]

    sys_norm = normalize(sys_y)
    inf_norm = normalize(inf_y)

    if HAS_SCIPY:
        r, p = pearsonr(sys_norm, inf_norm)
        return {'correlation': r, 'p_value': p, 'n_points': len(common_x)}

    # 手动 Pearson
    n = len(sys_norm)
    mean_a = sum(sys_norm) / n
    mean_b = sum(inf_norm) / n
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(sys_norm, inf_norm))
    std_a = math.sqrt(sum((a - mean_a) ** 2 for a in sys_norm))
    std_b = math.sqrt(sum((b - mean_b) ** 2 for b in inf_norm))
    r = cov / (std_a * std_b) if std_a > 0 and std_b > 0 else 0.0
    return {'correlation': r, 'p_value': None, 'n_points': n}


# ============================================================
# 纵向对比
# ============================================================

def run_longitudinal(learner, num_rounds: int = 500,
                     sample_interval: int = 50) -> Dict:
    """运行纵向学习对比

    周期性采样系统指标，与 CDI/神经标记对比。
    """
    snapshots = []
    for step in range(num_rounds):
        if step % sample_interval == 0:
            stats = learner.get_stats()
            month_equiv = step / num_rounds * 72  # 映射到 0-72 月
            snapshots.append({
                'step': step,
                'month_equiv': round(month_equiv, 1),
                'vocabulary_size': stats.get('vocabulary_size', 0),
                'prediction_accuracy': max(0, 1.0 - stats.get('avg_error', 1.0)),
                'comm_success_rate': stats.get('comm_success_rate', 0.0),
                'stage': stats.get('stage', 'sensorimotor'),
            })

    # 与 CDI 对比
    sys_vocab = {s['month_equiv']: s['vocabulary_size'] for s in snapshots}
    cdv_result = compute_correlation(sys_vocab, INFANT_VOCAB_CDI)

    # 曲线拟合
    x_data = [s['month_equiv'] for s in snapshots]
    y_data = [s['vocabulary_size'] for s in snapshots]
    curve_result = fit_curves(x_data, y_data)

    # 神经标记检查
    marker_status = []
    for marker in NEURAL_MARKERS:
        analogue = marker['mvl_analogue']
        relevant = [s for s in snapshots if s['month_equiv'] >= marker['onset_months']]
        if relevant:
            vals = [s.get(analogue, 0) for s in relevant]
            avg = sum(v for v in vals if isinstance(v, (int, float))) / max(len(vals), 1)
            marker_status.append({
                'marker': marker['name'],
                'expected_onset': marker['onset_months'],
                'system_value': avg,
                'emerged': avg > 0.3 if isinstance(avg, float) else False,
            })

    return {
        'snapshots': snapshots,
        'cdi_correlation': cdv_result,
        'curve_fit': curve_result,
        'neural_markers': marker_status,
        'total_rounds': num_rounds,
    }
