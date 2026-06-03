"""好奇心驱动提问 — 从预测违例中生成 'why'/'what'/'how' 标记

当 Agent 的预测被实际观测违例时，产生 surprise（惊讶）信号。
惊讶驱动知识寻求，自然涌现疑问标记 (question markers)。

核心组件:
  QuestioningModule — 基于预测违例的提问生成

提问类型映射:
  prediction_error  → 'why'   (为什么预测错了?)
  novel_feature     → 'what'  (这是什么?)
  unexpected_change → 'how'   (怎么发生的?)
  missing_entity    → 'where' (在哪里?)
  timing_violation  → 'when'  (什么时候?)
  unknown_agent     → 'who'   (谁做的?)

所有数值计算使用 torch.Tensor，零 numpy 依赖。
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

from src.core.device import get_device

# 疑问标记词汇表
QUESTION_MARKERS = {'why', 'what', 'how', 'where', 'when', 'who'}


class QuestioningModule:
    """好奇心驱动的提问生成模块

    当预测与实际观测的差异超过阈值时，生成疑问。
    记录提问-回答历史，追踪各标记的使用频率与有效性。

    Args:
        surprise_threshold: 触发提问的惊讶阈值 [0, 1]，默认 0.5
    """

    def __init__(self, surprise_threshold: float = 0.5):
        self._device = get_device()
        self.surprise_threshold = surprise_threshold

        # 提问历史: [(marker, topic, answer, helpful), ...]
        self._qa_history: List[Tuple[str, str, str, bool]] = []

        # 各标记的使用频率
        self._marker_counts: Dict[str, int] = {m: 0 for m in QUESTION_MARKERS}

        # 各标记的有效性: 帮助性的滑动平均
        self._marker_helpfulness: Dict[str, float] = {m: 0.5 for m in QUESTION_MARKERS}

        # 惊讶历史（用于自适应阈值）
        self._surprise_history: List[float] = []

    def compute_surprise(self, prediction: Dict, actual: Dict) -> float:
        """计算预测与实际之间的惊讶度

        惊讶度 = 1 - similarity(prediction, actual)
        相似度使用特征重合率计算。

        Args:
            prediction: 预测场景 {feature: expected_value}
            actual: 实际场景 {feature: actual_value}

        Returns:
            惊讶度 [0, 1]
        """
        if not prediction and not actual:
            return 0.0

        # 计算特征值差异
        all_keys = set(prediction.keys()) | set(actual.keys())
        if not all_keys:
            return 0.0

        pred_tensor = torch.zeros(len(all_keys), dtype=torch.float32,
                                  device=self._device)
        actual_tensor = torch.zeros(len(all_keys), dtype=torch.float32,
                                    device=self._device)

        for i, key in enumerate(sorted(all_keys)):
            pred_val = prediction.get(key, 0.0)
            actual_val = actual.get(key, 0.0)

            # 尝试转换为数值
            try:
                pred_tensor[i] = float(pred_val)
            except (ValueError, TypeError):
                pred_tensor[i] = 0.0
            try:
                actual_tensor[i] = float(actual_val)
            except (ValueError, TypeError):
                actual_tensor[i] = 0.0

        # 余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(
            pred_tensor.unsqueeze(0), actual_tensor.unsqueeze(0)
        ).item()

        # 映射到 [0, 1] 惊讶度
        similarity = (cos_sim + 1.0) / 2.0  # [-1,1] → [0,1]
        surprise = 1.0 - similarity

        self._surprise_history.append(surprise)
        if len(self._surprise_history) > 500:
            self._surprise_history = self._surprise_history[-500:]

        return min(max(surprise, 0.0), 1.0)

    def generate_question(self, observation: Dict, prediction: Dict,
                          actual: Dict) -> Optional[Tuple[str, str]]:
        """从预测违例生成提问

        步骤:
        1. 计算惊讶度
        2. 如果超过阈值，分析违例类型
        3. 选择对应的疑问标记
        4. 提取违例主题

        Args:
            observation: 当前观测
            prediction: Agent 的预测
            actual: 实际结果

        Returns:
            (marker, topic) 元组，或 None（惊讶度不足时）
        """
        surprise = self.compute_surprise(prediction, actual)

        if surprise < self.surprise_threshold:
            return None

        # 分析违例类型
        surprise_type = self._classify_surprise(prediction, actual)

        # 选择疑问标记
        marker = self.select_question_type(surprise_type)

        # 提取主题: 找到差异最大的特征
        topic = self._extract_topic(prediction, actual)

        # 记录提问
        self._marker_counts[marker] = self._marker_counts.get(marker, 0) + 1

        return (marker, topic)

    def _classify_surprise(self, prediction: Dict, actual: Dict) -> str:
        """对惊讶类型进行分类

        Returns:
            惊讶类型字符串
        """
        pred_keys = set(prediction.keys())
        actual_keys = set(actual.keys())

        # 新出现的特征 → novel_feature
        new_features = actual_keys - pred_keys
        if new_features:
            return 'novel_feature'

        # 缺失的特征 → missing_entity
        missing = pred_keys - actual_keys
        if missing:
            return 'missing_entity'

        # 值变化 → 根据 key 语义判断类型
        changed_keys = []
        for key in pred_keys & actual_keys:
            try:
                if abs(float(prediction[key]) - float(actual[key])) > 0.1:
                    changed_keys.append(key)
            except (ValueError, TypeError):
                if str(prediction[key]) != str(actual[key]):
                    changed_keys.append(key)

        if not changed_keys:
            return 'prediction_error'

        # 根据变化的 key 判断类型
        for key in changed_keys:
            key_lower = key.lower()
            if 'time' in key_lower or 'step' in key_lower:
                return 'timing_violation'
            if 'agent' in key_lower or 'who' in key_lower:
                return 'unknown_agent'
            if 'change' in key_lower or 'delta' in key_lower:
                return 'unexpected_change'

        return 'prediction_error'

    def _extract_topic(self, prediction: Dict, actual: Dict) -> str:
        """提取违例主题（差异最大的特征键名）

        Args:
            prediction: 预测值
            actual: 实际值

        Returns:
            主题字符串
        """
        max_diff = 0.0
        topic = 'unknown'

        all_keys = set(prediction.keys()) | set(actual.keys())
        for key in all_keys:
            pred_val = prediction.get(key, 0.0)
            actual_val = actual.get(key, 0.0)

            try:
                diff = abs(float(pred_val) - float(actual_val))
            except (ValueError, TypeError):
                diff = 1.0 if str(pred_val) != str(actual_val) else 0.0

            if diff > max_diff:
                max_diff = diff
                topic = key

        return topic

    def select_question_type(self, surprise_type: str) -> str:
        """根据惊讶类型选择疑问标记

        Args:
            surprise_type: 惊讶分类

        Returns:
            疑问标记
        """
        type_to_marker = {
            'prediction_error': 'why',
            'novel_feature': 'what',
            'unexpected_change': 'how',
            'missing_entity': 'where',
            'timing_violation': 'when',
            'unknown_agent': 'who',
        }
        return type_to_marker.get(surprise_type, 'why')

    def record_answer(self, question_marker: str, topic: str,
                      answer: str, helpful: bool) -> None:
        """记录提问的回答，更新标记有效性

        Args:
            question_marker: 疑问标记
            topic: 提问主题
            answer: 回答内容
            helpful: 回答是否有帮助
        """
        self._qa_history.append((question_marker, topic, answer, helpful))

        # 限制历史长度
        if len(self._qa_history) > 500:
            self._qa_history = self._qa_history[-500:]

        # 更新标记有效性（指数移动平均）
        alpha = 0.2
        old_val = self._marker_helpfulness.get(question_marker, 0.5)
        new_val = 1.0 if helpful else 0.0
        self._marker_helpfulness[question_marker] = (
            (1 - alpha) * old_val + alpha * new_val
        )

    def get_question_patterns(self) -> Dict[str, float]:
        """获取各疑问标记的使用频率与有效性

        Returns:
            {marker: effectiveness} 字典
            effectiveness = frequency × helpfulness
        """
        total_questions = sum(self._marker_counts.values())
        if total_questions == 0:
            return {m: 0.0 for m in QUESTION_MARKERS}

        patterns: Dict[str, float] = {}
        for marker in QUESTION_MARKERS:
            frequency = self._marker_counts.get(marker, 0) / total_questions
            helpfulness = self._marker_helpfulness.get(marker, 0.5)
            patterns[marker] = frequency * helpfulness

        return patterns

    def adapt_threshold(self) -> None:
        """根据提问历史自适应调整惊讶阈值

        如果提问过多且大多数无帮助 → 提高阈值（减少提问）
        如果提问过少 → 降低阈值（增加提问）
        """
        if len(self._qa_history) < 20:
            return

        # 最近 20 条的帮助率
        recent = self._qa_history[-20:]
        help_rate = sum(1 for _, _, _, h in recent if h) / len(recent)

        if help_rate < 0.3:
            # 帮助率低: 提高阈值
            self.surprise_threshold = min(0.9, self.surprise_threshold + 0.05)
        elif help_rate > 0.7:
            # 帮助率高: 降低阈值以获得更多提问
            self.surprise_threshold = max(0.1, self.surprise_threshold - 0.05)

    # ── 持久化 ────────────────────────────────────────────────

    def save_state(self) -> dict:
        return {
            'surprise_threshold': self.surprise_threshold,
            'qa_history': [
                {'marker': m, 'topic': t, 'answer': a, 'helpful': h}
                for m, t, a, h in self._qa_history
            ],
            'marker_counts': dict(self._marker_counts),
            'marker_helpfulness': dict(self._marker_helpfulness),
            'surprise_history': list(self._surprise_history),
        }

    def load_state(self, state: dict) -> None:
        self.surprise_threshold = state.get('surprise_threshold', 0.5)
        self._qa_history = [
            (item['marker'], item['topic'], item['answer'], item['helpful'])
            for item in state.get('qa_history', [])
        ]
        self._marker_counts = state.get('marker_counts', {m: 0 for m in QUESTION_MARKERS})
        self._marker_helpfulness = state.get(
            'marker_helpfulness', {m: 0.5 for m in QUESTION_MARKERS}
        )
        self._surprise_history = state.get('surprise_history', [])
