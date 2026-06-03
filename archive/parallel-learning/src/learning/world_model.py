"""世界模型 — 从交互中发现因果规律

核心思想：不是存数据，而是从交互中发现规律。
像婴儿一样：观察事物→发现规律→预测结果→验证规律。

与现有 LLM 的区别：
- LLM：统计关联（"天空"和"蓝色"经常一起出现）
- 世界模型：因果推理（"因为瑞利散射，所以天空是蓝色的"）

关键能力：
1. 观察：记录状态转移 (state, action) → next_state
2. 发现：从历史中提取因果规则
3. 预测：基于规则预测未来状态
4. 验证：用新观察验证/修正规则
5. 惊讶：计算预测与实际的差距
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import math


@dataclass
class StateSnapshot:
    """状态快照"""
    features: Dict[str, float]   # 特征向量
    timestamp: float = 0.0

    def similarity(self, other: 'StateSnapshot') -> float:
        """计算两个状态的相似度"""
        common_keys = set(self.features.keys()) & set(other.features.keys())
        if not common_keys:
            return 0.0
        dot = sum(self.features[k] * other.features[k] for k in common_keys)
        norm1 = math.sqrt(sum(v**2 for v in self.features.values()))
        norm2 = math.sqrt(sum(v**2 for v in other.features.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


@dataclass
class CausalRule:
    """因果规则：trigger → effect"""
    id: str
    trigger: Dict[str, Any]      # 触发条件
    effect: Dict[str, Any]       # 预期效果
    confidence: float = 0.5      # 置信度 [0, 1]
    evidence_count: int = 0      # 支持证据数
    exception_count: int = 0     # 反例数
    exceptions: List[Dict] = field(default_factory=list)
    created_at: float = 0.0
    last_verified: float = 0.0

    @property
    def reliability(self) -> float:
        """可靠性 = 置信度 * (1 - 例外率)"""
        total = self.evidence_count + self.exception_count
        if total == 0:
            return 0.0
        exception_rate = self.exception_count / total
        return self.confidence * (1.0 - exception_rate)

    def matches(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        """计算规则与当前状态+动作的匹配度"""
        combined = {**state, **action}
        if not self.trigger:
            return 0.0
        matches = 0
        for key, value in self.trigger.items():
            if key in combined:
                if isinstance(value, (int, float)) and isinstance(combined[key], (int, float)):
                    # 数值匹配：误差在10%内
                    if abs(value) < 1e-6:
                        matches += 1.0 if abs(combined[key]) < 0.1 else 0.0
                    else:
                        diff = abs(combined[key] - value) / abs(value)
                        matches += max(0, 1.0 - diff)
                else:
                    matches += 1.0 if combined[key] == value else 0.0
        return matches / len(self.trigger) if self.trigger else 0.0


@dataclass
class Prediction:
    """预测结果"""
    predicted_state: Dict[str, float]
    confidence: float
    matched_rules: List[str]     # 匹配的规则ID
    surprise: float = 0.0        # 惊讶度


class WorldModel:
    """世界模型 — 从交互中发现因果规律

    核心循环：
    1. 观察 (state, action) → next_state
    2. 尝试用已有规则预测
    3. 计算惊讶度
    4. 如果惊讶度高 → 更新/创建规则
    5. 定期发现新的因果规律
    """

    def __init__(self):
        # 因果规则库
        self.rules: Dict[str, CausalRule] = {}
        self._rule_counter = 0

        # 状态历史
        self.history: List[Tuple[StateSnapshot, Dict, StateSnapshot]] = []
        self.max_history = 10000

        # 统计
        self.total_observations = 0
        self.total_predictions = 0
        self.total_surprises = 0
        self.avg_surprise = 0.0

        # 特征重要性（从历史中学习哪些特征对预测最有用）
        self.feature_importance: Dict[str, float] = defaultdict(lambda: 1.0)

    def observe(self, state: Dict[str, float], action: Dict[str, Any],
                next_state: Dict[str, float]) -> float:
        """观察一次交互，更新世界模型

        Args:
            state: 当前状态特征
            action: 执行的动作
            next_state: 结果状态特征

        Returns:
            惊讶度 [0, 1]
        """
        self.total_observations += 1

        # 1. 记录历史
        state_snap = StateSnapshot(features=state, timestamp=time.time())
        next_snap = StateSnapshot(features=next_state, timestamp=time.time())
        self.history.append((state_snap, action, next_snap))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # 2. 尝试预测
        prediction = self.predict(state, action)
        surprise = self.compute_surprise(prediction.predicted_state, next_state)

        # 3. 更新惊讶度统计
        self.total_surprises += 1
        self.avg_surprise = (
            self.avg_surprise * (self.total_surprises - 1) + surprise
        ) / self.total_surprises

        # 4. 如果惊讶度高，更新规则
        if surprise > 0.3:
            self._update_rules(state, action, next_state, surprise)

        # 5. 验证已有规则
        self._verify_rules(state, action, next_state)

        # 6. 更新特征重要性
        self._update_feature_importance(state, action, next_state)

        return surprise

    def predict(self, state: Dict[str, float],
                action: Dict[str, Any]) -> Prediction:
        """基于世界模型预测结果"""
        self.total_predictions += 1

        # 匹配所有相关规则
        matched = []
        for rule_id, rule in self.rules.items():
            match_score = rule.matches(state, action)
            if match_score > 0.3:
                matched.append((rule, match_score))

        if not matched:
            # 没有匹配的规则，返回当前状态作为预测（无变化）
            return Prediction(
                predicted_state=dict(state),
                confidence=0.0,
                matched_rules=[],
            )

        # 按匹配度 * 可靠性排序
        matched.sort(key=lambda x: x[1] * x[0].reliability, reverse=True)

        # 综合前3条规则的预测
        predicted = dict(state)
        total_weight = 0.0
        matched_ids = []

        for rule, score in matched[:3]:
            weight = score * rule.reliability
            if weight > 0:
                for key, value in rule.effect.items():
                    if isinstance(value, (int, float)):
                        if key in predicted:
                            predicted[key] = predicted[key] * (1 - weight) + value * weight
                        else:
                            predicted[key] = value
                total_weight += weight
                matched_ids.append(rule.id)

        confidence = min(1.0, total_weight / max(len(matched), 1))

        return Prediction(
            predicted_state=predicted,
            confidence=confidence,
            matched_rules=matched_ids,
        )

    def compute_surprise(self, predicted: Dict[str, float],
                         actual: Dict[str, float]) -> float:
        """计算惊讶度 = 预测与实际的差距"""
        if not predicted or not actual:
            return 0.0

        common_keys = set(predicted.keys()) & set(actual.keys())
        if not common_keys:
            return 1.0

        total_error = 0.0
        total_weight = 0.0

        for key in common_keys:
            p = predicted[key]
            a = actual[key]
            importance = self.feature_importance.get(key, 1.0)

            if isinstance(p, (int, float)) and isinstance(a, (int, float)):
                # 归一化误差
                scale = max(abs(p), abs(a), 1.0)
                error = abs(p - a) / scale
                total_error += error * importance
                total_weight += importance

        if total_weight == 0:
            return 0.0

        return min(1.0, total_error / total_weight)

    def discover_laws(self) -> List[CausalRule]:
        """从历史中发现新的因果规律"""
        if len(self.history) < 10:
            return []

        new_rules = []

        # 方法1：频繁模式挖掘
        # 找到经常出现的 (state_feature, action) → next_state_feature 模式
        patterns = self._extract_patterns()
        for pattern in patterns:
            if pattern['support'] >= 3 and pattern['confidence'] >= 0.7:
                rule = self._create_rule_from_pattern(pattern)
                if rule and rule.id not in self.rules:
                    self.rules[rule.id] = rule
                    new_rules.append(rule)

        # 方法2：特征关联分析
        # 找到 state 中哪些特征变化与 action 相关
        correlations = self._find_correlations()
        for corr in correlations:
            if corr['strength'] >= 0.6:
                rule = self._create_rule_from_correlation(corr)
                if rule and rule.id not in self.rules:
                    self.rules[rule.id] = rule
                    new_rules.append(rule)

        return new_rules

    def get_knowledge_summary(self) -> Dict:
        """获取知识摘要"""
        return {
            'total_observations': self.total_observations,
            'total_rules': len(self.rules),
            'reliable_rules': sum(1 for r in self.rules.values() if r.reliability > 0.5),
            'avg_surprise': round(self.avg_surprise, 4),
            'total_predictions': self.total_predictions,
            'top_rules': self._get_top_rules(5),
            'feature_importance': dict(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1], reverse=True
            )[:10]),
        }

    # ── 内部方法 ──────────────────────────────────────────────

    def _update_rules(self, state: Dict, action: Dict,
                      next_state: Dict, surprise: float):
        """惊讶度高时，更新或创建规则"""
        # 找到最相关的已有规则
        best_rule = None
        best_score = 0.0

        for rule in self.rules.values():
            score = rule.matches(state, action)
            if score > best_score:
                best_score = score
                best_rule = rule

        if best_rule and best_score > 0.5:
            # 更新已有规则
            self._refine_rule(best_rule, state, action, next_state)
        else:
            # 创建新规则
            self._create_rule(state, action, next_state)

    def _create_rule(self, state: Dict, action: Dict, next_state: Dict):
        """从观察中创建新规则"""
        self._rule_counter += 1
        rule_id = f"rule_{self._rule_counter}"

        # 提取触发条件（state + action 中变化最大的特征）
        trigger = {}
        for key, value in state.items():
            if isinstance(value, (int, float)) and abs(value) > 0.1:
                trigger[key] = round(value, 2)
        for key, value in action.items():
            trigger[f"action_{key}"] = value

        # 提取效果（next_state 中与 state 不同的部分）
        effect = {}
        for key, value in next_state.items():
            if isinstance(value, (int, float)):
                old_value = state.get(key, 0.0)
                if abs(value - old_value) > 0.05:
                    effect[key] = round(value, 2)

        if not effect:
            return

        rule = CausalRule(
            id=rule_id,
            trigger=trigger,
            effect=effect,
            confidence=0.5,
            evidence_count=1,
            created_at=time.time(),
        )
        self.rules[rule_id] = rule

    def _refine_rule(self, rule: CausalRule, state: Dict,
                     action: Dict, next_state: Dict):
        """用新观察精炼已有规则"""
        rule.evidence_count += 1
        rule.last_verified = time.time()

        # 更新效果的滑动平均
        for key, value in next_state.items():
            if key in rule.effect and isinstance(value, (int, float)):
                old = rule.effect[key]
                n = rule.evidence_count
                rule.effect[key] = round(old * (n-1)/n + value / n, 3)

        # 增加置信度
        rule.confidence = min(1.0, rule.confidence + 0.05)

    def _verify_rules(self, state: Dict, action: Dict, next_state: Dict):
        """用新观察验证已有规则"""
        for rule in self.rules.values():
            match_score = rule.matches(state, action)
            if match_score > 0.5:
                # 检查预测是否准确
                is_correct = True
                for key, expected in rule.effect.items():
                    actual = next_state.get(key)
                    if actual is not None and isinstance(expected, (int, float)):
                        if abs(expected - actual) > 0.2 * max(abs(expected), 1.0):
                            is_correct = False
                            break

                if is_correct:
                    rule.confidence = min(1.0, rule.confidence + 0.02)
                    rule.evidence_count += 1
                else:
                    rule.exception_count += 1
                    rule.confidence = max(0.1, rule.confidence - 0.05)
                    # 记录例外
                    if len(rule.exceptions) < 5:
                        rule.exceptions.append({
                            'state': dict(state),
                            'action': dict(action),
                            'expected': dict(rule.effect),
                            'actual': {k: next_state.get(k) for k in rule.effect},
                        })

    def _update_feature_importance(self, state: Dict, action: Dict,
                                   next_state: Dict):
        """更新特征重要性"""
        for key in state:
            if key in next_state:
                change = abs(next_state[key] - state[key]) if isinstance(state[key], (int, float)) else 0
                if change > 0.1:
                    self.feature_importance[key] = (
                        self.feature_importance.get(key, 1.0) * 0.95 + change * 0.05
                    )

    def _extract_patterns(self) -> List[Dict]:
        """从历史中提取频繁模式"""
        patterns = defaultdict(lambda: {'support': 0, 'successes': 0})

        for state_snap, action, next_snap in self.history[-1000:]:
            state = state_snap.features
            next_state = next_snap.features
            action_key = str(sorted(action.items()))
            for key, value in next_state.items():
                if isinstance(value, (int, float)):
                    old = state.get(key, 0.0)
                    if abs(value - old) > 0.1:
                        pattern_key = (action_key, key)
                        patterns[pattern_key]['support'] += 1
                        patterns[pattern_key]['action'] = action
                        patterns[pattern_key]['feature'] = key
                        patterns[pattern_key]['old_value'] = old
                        patterns[pattern_key]['new_value'] = value
                        if (value > old and patterns[pattern_key].get('direction', 0) >= 0) or \
                           (value < old and patterns[pattern_key].get('direction', 0) <= 0):
                            patterns[pattern_key]['successes'] += 1
                        patterns[pattern_key]['direction'] = 1 if value > old else -1

        result = []
        for key, data in patterns.items():
            if data['support'] >= 3:
                data['confidence'] = data['successes'] / data['support']
                data['pattern_key'] = key
                result.append(data)

        return sorted(result, key=lambda x: x['support'], reverse=True)[:20]

    def _find_correlations(self) -> List[Dict]:
        """找到特征之间的关联"""
        correlations = []

        if len(self.history) < 20:
            return correlations

        # 收集所有特征的变化
        changes = defaultdict(list)
        for state_snap, action, next_snap in self.history[-500:]:
            state = state_snap.features
            next_state = next_snap.features
            for key in next_state:
                if key in state and isinstance(state[key], (int, float)):
                    change = next_state[key] - state[key]
                    changes[key].append(change)

        # 找到变化模式稳定的特征
        for key, values in changes.items():
            if len(values) >= 10:
                mean_change = sum(values) / len(values)
                std = math.sqrt(sum((v - mean_change)**2 for v in values) / len(values))
                if std < abs(mean_change) * 0.5 and abs(mean_change) > 0.05:
                    correlations.append({
                        'feature': key,
                        'mean_change': mean_change,
                        'std': std,
                        'strength': 1.0 - std / max(abs(mean_change), 0.01),
                        'count': len(values),
                    })

        return sorted(correlations, key=lambda x: x['strength'], reverse=True)[:10]

    def _create_rule_from_pattern(self, pattern: Dict) -> Optional[CausalRule]:
        """从频繁模式创建规则"""
        self._rule_counter += 1
        return CausalRule(
            id=f"pattern_{self._rule_counter}",
            trigger=pattern.get('action', {}),
            effect={pattern['feature']: pattern.get('new_value', 0)},
            confidence=pattern.get('confidence', 0.5),
            evidence_count=pattern.get('support', 1),
            created_at=time.time(),
        )

    def _create_rule_from_correlation(self, corr: Dict) -> Optional[CausalRule]:
        """从关联创建规则"""
        self._rule_counter += 1
        return CausalRule(
            id=f"corr_{self._rule_counter}",
            trigger={'feature': corr['feature']},
            effect={'change': corr['mean_change']},
            confidence=corr['strength'],
            evidence_count=corr['count'],
            created_at=time.time(),
        )

    def _get_top_rules(self, n: int) -> List[Dict]:
        """获取最可靠的规则"""
        sorted_rules = sorted(
            self.rules.values(),
            key=lambda r: r.reliability,
            reverse=True
        )
        return [{
            'id': r.id,
            'trigger': r.trigger,
            'effect': r.effect,
            'confidence': round(r.confidence, 3),
            'reliability': round(r.reliability, 3),
            'evidence': r.evidence_count,
        } for r in sorted_rules[:n]]
