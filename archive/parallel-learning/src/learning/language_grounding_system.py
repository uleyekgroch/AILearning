"""语言接地系统 — 词汇关联世界模型

核心思想：词汇不是符号，而是指向世界模型中的预测。
"热" → 关联到触摸→烫手的因果规则
"球" → 关联到扔→弹回的因果规则

语言理解 = 世界模型预测的激活

与现有系统的区别：
- 现有：词汇是 hash 编码的符号
- 新系统：词汇是世界模型中因果规则的指针
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GroundedWord:
    """接地词汇"""
    word: str
    causal_rules: List[str] = field(default_factory=list)  # 关联的因果规则ID
    perceptual_concepts: List[str] = field(default_factory=list)  # 关联的感知概念
    usage_count: int = 0
    confidence: float = 0.0

    def predict(self, world_model, context: Dict = None) -> Dict:
        """基于关联的因果规则预测结果"""
        predictions = []
        for rule_id in self.causal_rules:
            if rule_id in world_model.rules:
                rule = world_model.rules[rule_id]
                predictions.append({
                    'effect': rule.effect,
                    'confidence': rule.reliability,
                })
        return predictions


class LanguageGroundingSystem:
    """语言接地系统 — 词汇↔世界模型"""

    def __init__(self):
        # 词汇→世界模型映射
        self.grounded_words: Dict[str, GroundedWord] = {}

        # 统计
        self.total_groundings = 0
        self.total_predictions = 0

    def ground_word(self, word: str, world_model,
                    state: Dict, action: Dict, result: Dict):
        """将词汇接地到世界模型

        当系统听到"热"的同时触摸了热的东西，
        "热"就关联到"触摸→烫手"这条因果规则。
        """
        if word not in self.grounded_words:
            self.grounded_words[word] = GroundedWord(word=word)

        gw = self.grounded_words[word]
        gw.usage_count += 1

        # 找到相关的因果规则
        for rule_id, rule in world_model.rules.items():
            match_score = rule.matches(state, action)
            if match_score > 0.3 and rule_id not in gw.causal_rules:
                gw.causal_rules.append(rule_id)

        # 更新置信度
        gw.confidence = min(1.0, gw.confidence + 0.1)
        self.total_groundings += 1

    def understand_sentence(self, sentence: str, world_model) -> Dict:
        """通过世界模型理解句子

        将句子中的每个词映射到世界模型预测，
        综合预测形成对句子的理解。
        """
        words = sentence.lower().split()
        predictions = []

        for word in words:
            if word in self.grounded_words:
                gw = self.grounded_words[word]
                word_pred = gw.predict(world_model)
                predictions.extend(word_pred)

        # 综合所有词的预测
        if predictions:
            combined = {}
            total_conf = 0
            for pred in predictions:
                conf = pred['confidence']
                for key, value in pred['effect'].items():
                    if isinstance(value, (int, float)):
                        if key in combined:
                            combined[key] = combined[key] * (1-conf) + value * conf
                        else:
                            combined[key] = value
                total_conf += conf

            return {
                'understood': True,
                'prediction': combined,
                'confidence': total_conf / len(predictions),
                'words_grounded': sum(1 for w in words if w in self.grounded_words),
                'total_words': len(words),
            }

        return {
            'understood': False,
            'prediction': {},
            'confidence': 0.0,
            'words_grounded': 0,
            'total_words': len(words),
        }

    def predict_from_language(self, sentence: str, world_model) -> Dict:
        """从语言预测世界状态

        "把球扔到地上" → 预测球会弹起来
        """
        return self.understand_sentence(sentence, world_model)

    def get_grounded_vocabulary(self) -> Dict[str, float]:
        """获取已接地词汇及其置信度"""
        return {w: gw.confidence for w, gw in self.grounded_words.items()}

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'grounded_words': len(self.grounded_words),
            'total_groundings': self.total_groundings,
            'total_predictions': self.total_predictions,
            'avg_confidence': sum(gw.confidence for gw in self.grounded_words.values()) / max(len(self.grounded_words), 1),
            'avg_rules_per_word': sum(len(gw.causal_rules) for gw in self.grounded_words.values()) / max(len(self.grounded_words), 1),
        }
