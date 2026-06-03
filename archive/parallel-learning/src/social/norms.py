"""
统一学习系统 — 社会规范

学习和评估社会行为规范。
"""

from typing import Dict, Tuple


class SocialNorms:
    """社会规范系统"""

    def __init__(self):
        self.norms: Dict[str, str] = {}  # situation -> expected_behavior
        self.violation_history: list = []

    def learn_norm(self, situation: str, expected_behavior: str) -> None:
        """学习一条社会规范"""
        self.norms[situation] = expected_behavior

    def evaluate(self, behavior: str, situation: str) -> float:
        """
        评估行为是否符合社会规范。
        返回符合度得分 [0, 1]。
        """
        if situation not in self.norms:
            return 0.5  # 未知情境，中性评分

        expected = self.norms[situation]
        if behavior == expected:
            return 1.0

        # 部分匹配：行为字符串与期望有重叠
        behavior_words = set(behavior.lower().split())
        expected_words = set(expected.lower().split())
        overlap = behavior_words & expected_words

        if not expected_words:
            return 0.5

        score = len(overlap) / len(expected_words)
        self.violation_history.append({
            'situation': situation,
            'expected': expected,
            'actual': behavior,
            'score': score,
        })
        return score

    def get_norms(self) -> Dict[str, str]:
        """获取所有已学习的规范"""
        return dict(self.norms)

    def compliance_rate(self) -> float:
        """返回整体合规率"""
        if not self.violation_history:
            return 1.0
        scores = [v['score'] for v in self.violation_history]
        return sum(scores) / len(scores)
