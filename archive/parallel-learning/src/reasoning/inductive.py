"""归纳推理 — 从观察序列中发现模式和预测"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from src.knowledge.graph import KnowledgeGraph
from src.reasoning.abstraction import AbstractRule


class InductiveReasoner:
    """归纳推理器：从具体观察中提炼一般性规律，并据此预测。"""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        # 缓存：属性值 -> 出现次数
        self._observation_counts: Dict[str, Counter] = defaultdict(Counter)

    def observe_pattern(self, observations: List[Dict]) -> Optional[AbstractRule]:
        """从观察序列中找不变的模式。

        每个 observation 是一个 dict，代表一次观察。
        算法：
        1. 统计每个键值对在所有观察中出现的频率
        2. 提取高频不变的键值对作为规则条件
        3. 找到与条件关联的稳定结论
        4. 返回 AbstractRule
        """
        if not observations:
            return None

        # 第一步：统计每个键值对的频率
        pair_counts: Counter = Counter()
        for obs in observations:
            for key, value in obs.items():
                pair_counts[(key, str(value))] += 1

        total = len(observations)

        # 第二步：找出在所有观察中都出现的键值对（不变量）
        invariant_pairs = {
            pair: count for pair, count in pair_counts.items()
            if count == total
        }

        if not invariant_pairs:
            # 退而求其次：找出现频率 >= 80% 的键值对
            threshold = max(1, int(total * 0.8))
            invariant_pairs = {
                pair: count for pair, count in pair_counts.items()
                if count >= threshold
            }

        if not invariant_pairs:
            return None

        # 分离条件和可能的变化部分
        # 条件 = 高频不变的键值对
        condition = {pair[0]: pair[1] for pair in invariant_pairs.keys()}

        # 找出变化的部分作为结论候选
        # 即：条件确定后，其他属性的值分布
        outcome_candidates: Dict[str, Counter] = defaultdict(Counter)
        for obs in observations:
            # 检查该观察是否匹配条件
            matches = all(
                str(obs.get(pair[0])) == pair[1]
                for pair in invariant_pairs.keys()
                if pair[0] in obs
            )
            if matches:
                # 收集非条件部分的属性
                for key, value in obs.items():
                    if (key, str(value)) not in invariant_pairs:
                        outcome_candidates[key].update([str(value)])

        # 选择最稳定的结论属性
        best_conclusion_key = None
        best_conclusion_value = None
        best_stability = 0.0

        for key, counter in outcome_candidates.items():
            most_common = counter.most_common(1)
            if most_common:
                value, count = most_common[0]
                stability = count / total
                if stability > best_stability:
                    best_stability = stability
                    best_conclusion_key = key
                    best_conclusion_value = value

        conclusion = {}
        if best_conclusion_key is not None:
            conclusion[best_conclusion_key] = best_conclusion_value

        if not conclusion:
            return None

        # 统计例外
        exception_count = 0
        for obs in observations:
            matches_condition = all(
                str(obs.get(k)) == v for k, v in condition.items()
                if k in obs
            )
            matches_conclusion = all(
                str(obs.get(k)) == v for k, v in conclusion.items()
                if k in obs
            )
            if matches_condition and not matches_conclusion:
                exception_count += 1

        confidence = (total - exception_count) / total if total > 0 else 0.0

        rule = AbstractRule(
            condition=condition,
            conclusion=conclusion,
            support=total,
            confidence=confidence,
            exceptions=[],
        )
        return rule

    def predict_next(self, sequence: List[Dict]) -> Optional[Dict]:
        """基于简单频率预测序列中的下一个观察。

        算法：
        1. 统计序列中每个键值对的转移频率
        2. 使用最后几个观察的属性值作为上下文
        3. 预测每个属性最可能的下一个值（基于频率）
        4. 如果序列有明显的周期性，利用周期预测
        """
        if not sequence:
            return None

        if len(sequence) == 1:
            # 只有一个观察，直接返回它（最好猜测就是重复）
            return dict(sequence[0])

        # 第一步：检测周期性
        period = self._detect_period(sequence)
        if period > 0 and len(sequence) >= 2 * period:
            # 利用周期性预测
            return self._predict_by_period(sequence, period)

        # 第二步：基于属性值频率预测
        # 统计每个属性的最后值 -> 下一个值的转移频率
        transition_counts: Dict[str, Dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )

        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_obs = sequence[i + 1]
            for key, value in current.items():
                if key in next_obs:
                    transition_counts[key][str(value)].update([str(next_obs[key])])

        # 使用最后观察的属性值来预测下一个值
        last_obs = sequence[-1]
        prediction = {}

        for key, value in last_obs.items():
            str_val = str(value)
            if key in transition_counts and str_val in transition_counts[key]:
                # 选择频率最高的转移目标
                counter = transition_counts[key][str_val]
                most_common = counter.most_common(1)
                if most_common:
                    predicted_val = most_common[0][0]
                    # 尝试还原原始类型
                    prediction[key] = self._restore_type(predicted_val, value)

        # 对于没有转移统计的属性，使用全局频率最高的值
        if not prediction:
            # 退回到全局频率统计
            global_counts: Dict[str, Counter] = defaultdict(Counter)
            for obs in sequence:
                for key, value in obs.items():
                    global_counts[key].update([str(value)])

            for key, counter in global_counts.items():
                most_common = counter.most_common(1)
                if most_common:
                    prediction[key] = most_common[0][0]

        return prediction if prediction else None

    # ── 内部方法 ──────────────────────────────────────────────

    def _detect_period(self, sequence: List[Dict]) -> int:
        """检测序列是否有周期性，返回周期长度（0 表示无周期）"""
        n = len(sequence)
        if n < 4:
            return 0

        # 尝试不同的周期长度
        for period in range(1, n // 2 + 1):
            matches = 0
            total = 0
            for i in range(n - period):
                total += 1
                if self._observations_similar(sequence[i], sequence[i + period]):
                    matches += 1

            # 匹配率超过 80% 认为有周期性
            if total > 0 and matches / total >= 0.8:
                return period

        return 0

    def _predict_by_period(self, sequence: List[Dict], period: int) -> Dict:
        """基于周期性预测下一个观察"""
        # 预测位置对应周期中的哪个位置
        target_index = len(sequence) % period
        # 收集所有对应位置的观察
        candidates = [sequence[i] for i in range(target_index, len(sequence), period)]

        if not candidates:
            return dict(sequence[-period])

        # 合并候选：取每个属性频率最高的值
        prediction = {}
        attr_counts: Dict[str, Counter] = defaultdict(Counter)

        for obs in candidates:
            for key, value in obs.items():
                attr_counts[key].update([str(value)])

        for key, counter in attr_counts.items():
            most_common = counter.most_common(1)
            if most_common:
                prediction[key] = most_common[0][0]

        return prediction

    def _observations_similar(self, a: Dict, b: Dict) -> bool:
        """检查两个观察是否相似（共享键值对比例 >= 50%）"""
        if not a or not b:
            return False

        shared_keys = set(a.keys()) & set(b.keys())
        if not shared_keys:
            return False

        matching = sum(1 for k in shared_keys if str(a[k]) == str(b[k]))
        similarity = matching / len(shared_keys)

        return similarity >= 0.5

    def _restore_type(self, str_value: str, original_value) -> Any:
        """尝试将字符串还原为原始类型"""
        if isinstance(original_value, bool):
            return str_value.lower() == 'true'
        if isinstance(original_value, int):
            try:
                return int(str_value)
            except ValueError:
                return str_value
        if isinstance(original_value, float):
            try:
                return float(str_value)
            except ValueError:
                return str_value
        return str_value
