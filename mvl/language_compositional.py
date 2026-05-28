"""
组合性语言系统：从无序符号到结构化表达

核心思想：
当前语言系统的词序是装饰性的——Listener 把话语当作无序符号集合。
本模块实现两个关键改变：
1. CompositionalSpeaker: 按信息量（消歧能力）排序符号
2. CompositionalListener: 渐进式匹配（从左到右逐步过滤候选集）

词序的价值：
- 不在于选不同对象（相同符号集合总是匹配相同对象集）
- 而在于更高效地缩小候选集 + 在边界情况下决定胜负
- "最有信息量优先"是自然涌现的词序策略
"""

import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, LanguageAgent,
    _symbol_category, _detect_order_for_agent, _record_modifier_order_for_agent,
    NEGATION_MARKERS, RELATIVE_MARKERS,
)
from language_rich_scene import generate_rich_scene_v2


class CompositionalSpeaker:
    """
    组合性 Speaker — 按信息量排序符号

    与基类 Speaker 的差异：
    - 基类：按类别偏好排序（color→shape→material），排序与消歧能力无关
    - 本类：按信息量降序排序（最能缩小候选集的符号排最前）

    信息量 = 1 - (匹配该符号的物体数 / 场景总物体数)
    信息量越高 → 该符号越能区分目标
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language

    def describe(self, target_features: Dict[str, str],
                 scene_features: List[Dict[str, str]],
                 target_idx: int = -1) -> List[str]:
        """描述目标物体，符号按信息量降序排列"""
        target_symbols = [v for v in target_features.values() if v]
        if not target_symbols:
            return []

        # 尝试单符号
        for sym in target_symbols:
            if sum(1 for obj in scene_features if sym in obj.values()) == 1:
                return [sym]

        # 找最小唯一组合（复用基类逻辑）
        best = target_symbols[:2]
        best_matches = self._count_matches(best, scene_features)

        for n in range(2, len(target_symbols) + 1):
            from itertools import combinations
            for combo in combinations(target_symbols, n):
                combo = list(combo)
                matches = self._count_matches(combo, scene_features)
                if matches < best_matches:
                    best = combo
                    best_matches = matches
                if matches <= 1:
                    break
            if best_matches <= 1:
                break

        # 按信息量排序
        return self._order_by_information(list(best), scene_features)

    def _count_matches(self, combo: List[str], scene: List[Dict[str, str]]) -> int:
        return sum(1 for obj in scene if all(sym in obj.values() for sym in combo))

    def _order_by_information(self, symbols: List[str],
                               scene: List[Dict[str, str]]) -> List[str]:
        """头词优先 + 修饰语按信息量降序排列

        顺序：shape（头词/类别）→ 信息量最高的修饰语 → ... → 信息量最低的修饰语
        这形成了 "名词 + 形容词" 的自然语言结构。
        """
        if len(symbols) <= 1:
            return symbols

        shape_syms = [s for s in symbols if _symbol_category(s) == 'shape']
        other_syms = [s for s in symbols if _symbol_category(s) != 'shape']

        # 修饰语按信息量降序
        other_sorted = sorted(other_syms,
                              key=lambda s: self._info_score(s, scene),
                              reverse=True)

        # 头词在前，修饰语在后
        return shape_syms + other_sorted

    def _info_score(self, symbol: str, scene: List[Dict[str, str]]) -> float:
        """信息量 = 1 - 匹配比例（越稀有的符号信息量越高）"""
        if not scene:
            return 0.0
        matches = sum(1 for obj in scene if symbol in obj.values())
        return 1.0 - matches / len(scene)


class CompositionalListener:
    """
    组合性 Listener — 头词优先 + 修饰语消歧

    与基类 Listener 的核心差异：
    - 基类：一次性 bag-of-symbols 匹配，词序不影响结果
    - 本类：shape 作为头词（主过滤器），其他符号作为修饰语（辅助评分）

    为什么这使得词序有意义：
    - "circle red" → shape=circle 是主类别，color=red 是修饰语
    - "red circle" → 同上（shape 符号被识别为头词，与位置无关）
    - 但当多个 shape 匹配时，修饰语的顺序决定优先级
    - 第 1 个修饰语是主要消歧器，第 2 个是备用

    关键机制：shape 作为"名词"确定类别，其他属性作为"形容词"确定实例。
    这是人类语言名词短语的基本结构。
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language

    def interpret(self, utterance: List[str],
                  scene_features: List[Dict[str, str]]) -> Optional[int]:
        """头词优先匹配：shape 过滤 → 修饰语评分"""
        if not utterance or not scene_features:
            return None

        # 分离头词（shape）和修饰语（其他）
        shape_syms = [s for s in utterance if _symbol_category(s) == 'shape']
        modifier_syms = [s for s in utterance
                         if _symbol_category(s) != 'shape' and s not in NEGATION_MARKERS]

        # 处理否定
        negated_syms = set()
        for i, s in enumerate(utterance):
            if s in NEGATION_MARKERS and i + 1 < len(utterance):
                negated_syms.add(utterance[i + 1])

        # 阶段 1：用 shape 过滤（头词 = 类别）
        if shape_syms:
            candidates = [i for i, obj in enumerate(scene_features)
                         if any(s in obj.values() for s in shape_syms)]
        else:
            candidates = list(range(len(scene_features)))

        if not candidates:
            candidates = list(range(len(scene_features)))

        # 阶段 2：用修饰语评分（在候选集中消歧）
        # 词序有意义：靠前的修饰语权重更高
        scores = []
        for i in candidates:
            obj_values = set(scene_features[i].values())
            score = 0.0
            for pos, sym in enumerate(modifier_syms):
                if sym in negated_syms:
                    continue
                # 位置权重：第 1 个修饰语权重最高，依次递减
                weight = 1.0 / (pos + 1)
                if sym in obj_values:
                    score += weight
            # 否定惩罚
            for neg_sym in negated_syms:
                if neg_sym in obj_values:
                    score -= 2.0
            scores.append((i, score))

        if not scores:
            return 0

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]


class CompositionalLanguageAgent(LanguageAgent):
    """
    组合性语言 Agent

    在基类基础上增加：
    - 词序偏好追踪：每个符号的平均位置
    - 类别位置追踪：每个类别在第几位出现
    """

    def __init__(self, agent_id: str):
        # 不调用 super().__init__，因为要替换 Speaker/Listener
        self.id = agent_id
        self.language = EmergingLanguage()
        self.speaker = CompositionalSpeaker(self.language)
        self.listener = CompositionalListener(self.language)
        self.communication_log = []

        # 词序偏好：{symbol: [位置0次数, 位置1次数, ...]}
        self.symbol_order_preferences = defaultdict(lambda: [0] * 10)
        self.total_order_games = 0

    def update_from_communication(self, utterance: List[str], success: bool):
        """更新语言统计 + 词序偏好"""
        super().update_from_communication(utterance, success)

        # 记录词序偏好
        if len(utterance) >= 2:
            self.total_order_games += 1
            for pos, sym in enumerate(utterance):
                if pos < 10:
                    self.symbol_order_preferences[sym][pos] += 1

    def get_symbol_avg_position(self, symbol: str) -> float:
        """获取符号的平均位置（越小越靠前）"""
        counts = self.symbol_order_preferences[symbol]
        total = sum(counts)
        if total == 0:
            return -1.0
        return sum(i * c for i, c in enumerate(counts)) / total

    def get_category_order(self) -> List[str]:
        """获取类别偏好顺序（按平均位置排序）"""
        cat_positions = defaultdict(list)
        for sym, counts in self.symbol_order_preferences.items():
            total = sum(counts)
            if total > 0:
                avg_pos = sum(i * c for i, c in enumerate(counts)) / total
                cat = _symbol_category(sym)
                if cat:
                    cat_positions[cat].append(avg_pos)

        cat_avg = {cat: np.mean(positions) for cat, positions in cat_positions.items()}
        return sorted(cat_avg.keys(), key=lambda c: cat_avg[c])

    def get_order_consistency(self) -> float:
        """词序一致性：符号位置的确定性（0=完全随机，1=完全固定）"""
        if not self.symbol_order_preferences:
            return 0.0

        consistencies = []
        for sym, counts in self.symbol_order_preferences.items():
            total = sum(counts)
            if total < 3:
                continue
            probs = np.array(counts[:5], dtype=float)  # 只看前5个位置
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs = probs / probs_sum
                # 熵越低 → 一致性越高
                entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
                max_entropy = np.log2(len(probs))
                if max_entropy > 0:
                    consistencies.append(1.0 - entropy / max_entropy)

        return np.mean(consistencies) if consistencies else 0.0


def compositional_cross_language_round(
    speaker: CompositionalLanguageAgent,
    listener: CompositionalLanguageAgent,
    scene_features: List[Dict[str, str]],
    target_idx: int,
) -> bool:
    """
    组合性跨语言交流一轮

    与基类 cross_language_round 的差异：
    - 使用 CompositionalSpeaker（信息量排序）
    - 使用 CompositionalListener（渐进式匹配）
    - 记录词序偏好
    """
    if target_idx >= len(scene_features) or not scene_features:
        return False

    target = scene_features[target_idx]

    # speaker 描述（按信息量排序）
    utterance = speaker.speaker.describe(target, scene_features)
    if not utterance:
        return False

    # listener 解释（渐进式匹配）
    chosen_idx = listener.listener.interpret(utterance, scene_features)
    success = (chosen_idx == target_idx)

    # 更新两个 agent
    speaker.update_from_communication(utterance, success)
    listener.update_from_communication(utterance, success)

    # 记录词序和形容词层级
    order = _detect_order_for_agent(utterance, target)
    if order:
        speaker.language.record_word_order(order, success)
        listener.language.record_word_order(order, success)

    _record_modifier_order_for_agent(speaker, utterance, success)
    _record_modifier_order_for_agent(listener, utterance, success)

    # 记录日志
    log_entry = {
        'speaker': speaker.id,
        'listener': listener.id,
        'target': target,
        'utterance': utterance,
        'chosen': chosen_idx,
        'success': success,
    }
    speaker.communication_log.append(log_entry)
    listener.communication_log.append(log_entry)

    return success


def compute_compositional_similarity(
    agent_a: CompositionalLanguageAgent,
    agent_b: CompositionalLanguageAgent,
) -> Dict:
    """
    计算两个组合性语言的相似度

    在基类 compute_language_similarity 基础上增加：
    - symbol_position_similarity: 符号位置偏好的相似度
    - order_consistency_a/b: 各自的词序一致性
    """
    lang_a = agent_a.language
    lang_b = agent_b.language

    # 词汇频率相似度
    all_symbols = set(lang_a.vocabulary.keys()) | set(lang_b.vocabulary.keys())
    if all_symbols:
        freq_a = np.array([lang_a.vocabulary.get(s, {}).get('frequency', 0) for s in all_symbols])
        freq_b = np.array([lang_b.vocabulary.get(s, {}).get('frequency', 0) for s in all_symbols])
        norm_a, norm_b = np.linalg.norm(freq_a), np.linalg.norm(freq_b)
        vocab_sim = float(np.dot(freq_a, freq_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
    else:
        vocab_sim = 0.0

    # 组合频率相似度
    all_collocations = set(lang_a.collocations.keys()) | set(lang_b.collocations.keys())
    if all_collocations:
        coll_a = np.array([lang_a.collocations.get(c, {}).get('count', 0) for c in all_collocations])
        coll_b = np.array([lang_b.collocations.get(c, {}).get('count', 0) for c in all_collocations])
        norm_ca, norm_cb = np.linalg.norm(coll_a), np.linalg.norm(coll_b)
        coll_sim = float(np.dot(coll_a, coll_b) / (norm_ca * norm_cb)) if norm_ca > 0 and norm_cb > 0 else 0.0
    else:
        coll_sim = 0.0

    # 词序一致性
    order_a = lang_a.get_preferred_order()
    order_b = lang_b.get_preferred_order()
    order_match = 1.0 if order_a == order_b else 0.0

    # 形容词层级一致性
    mod_a = tuple(lang_a.get_preferred_modifier_order())
    mod_b = tuple(lang_b.get_preferred_modifier_order())
    mod_match = 1.0 if mod_a == mod_b else 0.0

    # 符号位置相似度（核心新增维度）
    all_syms = set(agent_a.symbol_order_preferences.keys()) | set(agent_b.symbol_order_preferences.keys())
    position_sims = []
    for sym in all_syms:
        pos_a = agent_a.get_symbol_avg_position(sym)
        pos_b = agent_b.get_symbol_avg_position(sym)
        if pos_a >= 0 and pos_b >= 0:
            # 位置差异越小越相似（归一化到 [0,1]）
            pos_sim = max(0, 1.0 - abs(pos_a - pos_b) / 5.0)
            position_sims.append(pos_sim)
    symbol_pos_sim = np.mean(position_sims) if position_sims else 0.5

    # 类别顺序一致性
    cat_a = tuple(agent_a.get_category_order())
    cat_b = tuple(agent_b.get_category_order())
    cat_order_match = 1.0 if cat_a == cat_b else 0.0

    # 综合相似度
    overall = (
        0.30 * vocab_sim +
        0.20 * coll_sim +
        0.10 * order_match +
        0.05 * mod_match +
        0.25 * symbol_pos_sim +
        0.10 * cat_order_match
    )

    return {
        'vocab_freq_similarity': vocab_sim,
        'collocation_freq_similarity': coll_sim,
        'order_match': order_match,
        'modifier_order_match': mod_match,
        'symbol_position_similarity': symbol_pos_sim,
        'category_order_match': cat_order_match,
        'overall_similarity': overall,
        'order_consistency_a': agent_a.get_order_consistency(),
        'order_consistency_b': agent_b.get_order_consistency(),
    }


def compute_order_agreement(agents: List[CompositionalLanguageAgent],
                            sample_size: int = 200) -> float:
    """
    计算一群 agent 的词序一致性

    随机采样 agent 对，比较类别顺序。
    返回 [0,1]：1 = 所有 agent 顺序相同，0 = 完全不同。
    """
    if len(agents) < 2:
        return 1.0

    agreements = 0
    total = 0

    for _ in range(sample_size):
        i, j = random.sample(range(len(agents)), 2)
        cat_i = tuple(agents[i].get_category_order())
        cat_j = tuple(agents[j].get_category_order())
        if cat_i and cat_j:
            total += 1
            if cat_i == cat_j:
                agreements += 1

    return agreements / total if total > 0 else 0.0
