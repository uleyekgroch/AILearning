"""
具身隐喻追踪器 — Lakoff-Johnson 跨域隐喻映射

基于具身认知理论，追踪物理域符号（温度、重量、亮度等）
与抽象域符号（情感、重要性、智慧等）的共现关系，
检测隐喻是否从感知经验中涌现。
纯 PyTorch 实现，零 numpy 依赖。
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from src.core.device import get_device


# =====================================================================
# 5 种 Lakoff-Johnson 隐喻映射
# =====================================================================

METAPHOR_MAPPINGS: Dict[str, Dict] = {
    'temperature_emotion': {
        'source': 'temperature',
        'target': 'emotion',
        'mapping': {'hot': 'angry', 'warm': 'friendly', 'cold': 'unfriendly'},
        'description': '温度→情感（warm person = 友善的）',
    },
    'weight_importance': {
        'source': 'weight',
        'target': 'importance',
        'mapping': {'heavy': 'important', 'light': 'trivial'},
        'description': '重量→重要性（heavy responsibility = 重要的）',
    },
    'brightness_intelligence': {
        'source': 'brightness',
        'target': 'intelligence',
        'mapping': {'bright': 'smart', 'dark': 'confused'},
        'description': '亮度→智慧（bright idea = 聪明的）',
    },
    'height_power': {
        'source': 'height',
        'target': 'power',
        'mapping': {'high': 'powerful', 'low': 'weak'},
        'description': '高度→权力（high status = 有权力的）',
    },
    'cleanliness_morality': {
        'source': 'texture',
        'target': 'morality',
        'mapping': {'smooth': 'honest', 'rough': 'suspicious'},
        'description': '质地→道德（clean conscience = 无辜的）',
    },
}


# =====================================================================
# 场景属性域提取
# =====================================================================

def _extract_domain_values(scene: Dict, domain: str) -> Dict[str, float]:
    """
    从场景中提取指定域的属性值。

    Args:
        scene: 场景字典
        domain: 属性域名（temperature/weight/brightness/height/texture/emotion/...）

    Returns:
        {属性值名: 出现次数或强度}
    """
    values: Dict[str, float] = {}
    features = scene.get('features', [])

    for obj in features:
        if not isinstance(obj, dict):
            continue
        val = obj.get(domain)
        if val is not None:
            values[str(val)] = values.get(str(val), 0.0) + 1.0

    # 也检查顶层场景属性
    top_val = scene.get(domain)
    if top_val is not None:
        values[str(top_val)] = values.get(str(top_val), 0.0) + 1.0

    return values


# =====================================================================
# MetaphorTracker 主类
# =====================================================================

class MetaphorTracker:
    """
    具身隐喻追踪器 — 检测跨域符号共现

    当 Agent 在包含 '温度=hot' 的场景中使用与 'angry' 关联的符号时，
    记录 temperature→emotion 的 co-occurrence。当共现频率超过随机预期时，
    判定隐喻映射已涌现。
    """

    def __init__(self, device: str = 'auto'):
        """
        初始化隐喻追踪器。

        Args:
            device: 'auto' | 'cpu' | 'cuda'
        """
        self.device = get_device(device)

        # 源域属性值使用计数: domain -> value -> count
        self._source_usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # 目标域属性值使用计数: domain -> value -> count
        self._target_usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # 跨域共现计数: mapping_key -> source_val -> target_val -> count
        self._cross_domain: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        # 总记录数
        self._record_count: int = 0

    # ------------------------------------------------------------------
    # 记录符号-属性共现
    # ------------------------------------------------------------------

    def record(self, symbols: List[str], scene: Dict) -> None:
        """
        从场景中提取源域和目标域属性值，
        并记录符号与域值之间的共现关系。

        对于每个隐喻映射：
        1. 检查场景是否包含该映射的源域属性
        2. 检查符号是否包含该映射的目标域概念
        3. 若两者同时存在，记录跨域共现

        Args:
            symbols: Agent 使用的符号列表
            scene: 场景描述字典，可包含 'features' 列表和顶层属性
        """
        self._record_count += 1
        symbol_set = set(symbols)

        for key, meta in METAPHOR_MAPPINGS.items():
            source_domain = meta['source']
            target_domain = meta['target']
            mapping = meta['mapping']

            # 提取源域值
            source_vals = _extract_domain_values(scene, source_domain)
            if not source_vals:
                continue

            # 提取目标域值（从符号中检测）
            target_vals_in_symbols: Dict[str, int] = {}
            for src_val, tgt_val in mapping.items():
                if tgt_val in symbol_set:
                    target_vals_in_symbols[tgt_val] = target_vals_in_symbols.get(tgt_val, 0) + 1

            # 记录源域使用
            for sv, count in source_vals.items():
                self._source_usage[source_domain][sv] += int(count)

            # 记录目标域使用
            for tv, count in target_vals_in_symbols.items():
                self._target_usage[target_domain][tv] += count

            # 记录跨域共现
            for sv in source_vals:
                for tv in target_vals_in_symbols:
                    self._cross_domain[key][sv][tv] += 1

    # ------------------------------------------------------------------
    # 隐喻分数计算
    # ------------------------------------------------------------------

    def get_metaphor_score(self, mapping_key: str,
                           source_val: str, target_val: str) -> float:
        """
        计算特定源-目标值对的隐喻分数。

        公式: score = co_occurrence / max(freq_source, freq_target, 1)

        防止低频偶然共现导致虚高分数：
        - 若总记录数 < 10，返回 0.0

        Args:
            mapping_key: 隐喻映射键（如 'temperature_emotion'）
            source_val: 源域属性值（如 'hot'）
            target_val: 目标域属性值（如 'angry'）

        Returns:
            隐喻分数 (0.0 ~ 1.0)
        """
        if self._record_count < 10:
            return 0.0

        meta = METAPHOR_MAPPINGS.get(mapping_key)
        if meta is None:
            return 0.0

        co = self._cross_domain.get(mapping_key, {}).get(source_val, {}).get(target_val, 0)
        src_freq = self._source_usage.get(meta['source'], {}).get(source_val, 0)
        tgt_freq = self._target_usage.get(meta['target'], {}).get(target_val, 0)
        denominator = max(src_freq, tgt_freq, 1)
        return round(co / denominator, 4)

    # ------------------------------------------------------------------
    # 检测涌现隐喻
    # ------------------------------------------------------------------

    def detect_metaphors(self) -> Dict[str, Dict]:
        """
        扫描全部 5 个映射，检测涌现的跨域关联。

        对每个映射中的每个 source_val -> target_val 对，
        计算隐喻分数。仅返回分数 > 0 的结果。

        Returns:
            {
                mapping_key: {
                    'source_val_target_val': score,
                    ...
                },
                ...
            }
        """
        results: Dict[str, Dict] = {}

        for key, meta in METAPHOR_MAPPINGS.items():
            mapping = meta['mapping']
            pairs: Dict[str, float] = {}

            for src_val, tgt_val in mapping.items():
                score = self.get_metaphor_score(key, src_val, tgt_val)
                if score > 0.0:
                    pair_label = f'{src_val}_{tgt_val}'
                    pairs[pair_label] = score

            # 检测意外涌现：源域和目标域值共现但不在预定义映射中
            cross = self._cross_domain.get(key, {})
            for sv, tgt_dict in cross.items():
                for tv, count in tgt_dict.items():
                    # 跳过预定义映射中已有的对
                    if sv in mapping and mapping[sv] == tv:
                        continue
                    if count >= 3:
                        pair_label = f'{sv}_{tv}'
                        score = self.get_metaphor_score(key, sv, tv)
                        if score > 0.05 and pair_label not in pairs:
                            pairs[pair_label] = score

            if pairs:
                results[key] = pairs

        return results

    # ------------------------------------------------------------------
    # 场景生成
    # ------------------------------------------------------------------

    def generate_metaphor_scene(self, mapping_key: str,
                                 ambiguous: bool = False) -> Tuple[List[Dict], int]:
        """
        生成混合源域和目标域属性的场景。

        场景包含同时具有物理属性和抽象属性的对象，
        以测试 Agent 是否能捕捉跨域关联。

        Args:
            mapping_key: 隐喻映射键
            ambiguous: 若为 True，添加干扰属性使映射更难检测

        Returns:
            (scene_features, target_idx) — 对象列表和目标索引
        """
        meta = METAPHOR_MAPPINGS.get(mapping_key)
        if meta is None:
            # 回退通用场景
            return self._generic_scene(4)

        source_domain = meta['source']
        target_domain = meta['target']
        mapping = meta['mapping']

        objects: List[Dict] = []
        target_idx = 0

        # 为映射中每对 source->target 生成对象
        for i, (src_val, tgt_val) in enumerate(mapping.items()):
            obj = {
                'id': i,
                source_domain: src_val,
                target_domain: tgt_val,
                'primary_domain': source_domain,
            }

            if ambiguous:
                # 添加一个不匹配的干扰目标值
                other_tgts = [v for v in mapping.values() if v != tgt_val]
                if other_tgts:
                    noise_key = f'{target_domain}_noise'
                    obj[noise_key] = other_tgts[0]

            objects.append(obj)
            # 选择第一个对象作为目标
            if i == 0:
                target_idx = i

        # 添加额外的纯源域对象（无目标域属性）
        for j in range(len(mapping), len(mapping) + 2):
            src_vals = list(mapping.keys())
            obj = {
                'id': j,
                source_domain: src_vals[j % len(src_vals)],
                'primary_domain': source_domain,
            }
            objects.append(obj)

        # 构建完整场景
        scene = {
            'features': objects,
            'metaphor_type': mapping_key,
            'source_domain': source_domain,
            'target_domain': target_domain,
            'is_ambiguous': ambiguous,
            'num_source_objects': len(objects),
        }

        return objects, target_idx

    def _generic_scene(self, n: int) -> Tuple[List[Dict], int]:
        """生成通用回退场景"""
        objects = []
        for i in range(n):
            objects.append({
                'id': i,
                'features': torch.rand(8, device=self.device),
            })
        return objects, 0

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------

    def get_mapping_summary(self) -> Dict[str, Dict]:
        """
        返回所有映射的摘要统计。

        Returns:
            {
                mapping_key: {
                    'description': str,
                    'source_domain': str,
                    'target_domain': str,
                    'predefined_scores': {pair: score},
                    'emergent_pairs': {pair: score},
                    'source_coverage': float,  # 源域值覆盖率
                    'target_coverage': float,  # 目标域值覆盖率
                }
            }
        """
        summary: Dict[str, Dict] = {}

        for key, meta in METAPHOR_MAPPINGS.items():
            mapping = meta['mapping']
            source_domain = meta['source']
            target_domain = meta['target']

            # 预定义映射分数
            predefined: Dict[str, float] = {}
            for sv, tv in mapping.items():
                score = self.get_metaphor_score(key, sv, tv)
                predefined[f'{sv}->{tv}'] = score

            # 涌现的额外关联
            detected = self.detect_metaphors()
            emergent = detected.get(key, {})

            # 覆盖率计算
            src_observed = len(self._source_usage.get(source_domain, {}))
            src_total = len(set(mapping.keys()))
            src_coverage = min(src_observed / max(src_total, 1), 1.0)

            tgt_observed = len(self._target_usage.get(target_domain, {}))
            tgt_total = len(set(mapping.values()))
            tgt_coverage = min(tgt_observed / max(tgt_total, 1), 1.0)

            summary[key] = {
                'description': meta['description'],
                'source_domain': source_domain,
                'target_domain': target_domain,
                'predefined_scores': predefined,
                'emergent_pairs': emergent,
                'source_coverage': round(src_coverage, 3),
                'target_coverage': round(tgt_coverage, 3),
            }

        return summary

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        """
        将追踪器状态序列化为字典。

        Returns:
            包含所有内部状态的字典，可被 load_state 恢复。
        """
        source_plain: Dict[str, Dict[str, int]] = {}
        for domain, inner in self._source_usage.items():
            source_plain[domain] = dict(inner)

        target_plain: Dict[str, Dict[str, int]] = {}
        for domain, inner in self._target_usage.items():
            target_plain[domain] = dict(inner)

        cross_plain: Dict[str, Dict[str, Dict[str, int]]] = {}
        for key, sv_dict in self._cross_domain.items():
            cross_plain[key] = {}
            for sv, tv_dict in sv_dict.items():
                cross_plain[key][sv] = dict(tv_dict)

        return {
            'source_usage': source_plain,
            'target_usage': target_plain,
            'cross_domain': cross_plain,
            'record_count': self._record_count,
        }

    def load_state(self, state: dict) -> None:
        """
        从字典恢复追踪器状态。

        Args:
            state: save_state 返回的字典
        """
        self._source_usage = defaultdict(lambda: defaultdict(int))
        for domain, inner in state.get('source_usage', {}).items():
            for val, count in inner.items():
                self._source_usage[domain][val] = count

        self._target_usage = defaultdict(lambda: defaultdict(int))
        for domain, inner in state.get('target_usage', {}).items():
            for val, count in inner.items():
                self._target_usage[domain][val] = count

        self._cross_domain = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for key, sv_dict in state.get('cross_domain', {}).items():
            for sv, tv_dict in sv_dict.items():
                for tv, count in tv_dict.items():
                    self._cross_domain[key][sv][tv] = count

        self._record_count = state.get('record_count', 0)
