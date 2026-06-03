"""
统一学习系统 — 能力评估器

评估学习者在各维度上的能力指标，供课程调度器判断是否可晋升阶段。
"""

import torch
from typing import Dict


class CapabilityEvaluator:
    """能力评估器"""

    def evaluate(self, learner) -> Dict[str, float]:
        """评估各项能力指标，返回指标名→值的字典

        输出键名与 learner._check_promotion() 的晋升条件键名对齐。
        """
        return {
            'prediction_accuracy': self._evaluate_prediction_accuracy(learner),
            'vocabulary_size': float(self._evaluate_vocabulary_size(learner)),
            'communication_success': self._evaluate_communication_success(learner),
            'composition_rate': self._evaluate_composition_rate(learner),
            'grammar_complexity': self._evaluate_grammar_complexity(learner),
            'exploration_diversity': self._evaluate_exploration_diversity(learner),
            'total_steps': float(self._get_total_steps(learner)),
            # 多模态技能
            'reading_accuracy': self._evaluate_skill_history(learner, 'reading_history'),
            'writing_accuracy': self._evaluate_skill_history(learner, 'writing_history'),
            'listening_accuracy': self._evaluate_skill_history(learner, 'listening_history'),
            'grammar_accuracy': self._evaluate_skill_history(learner, 'grammar_history'),
        }

    # ── 各维度评估 ────────────────────────────────────────────────────

    def _evaluate_prediction_accuracy(self, learner) -> float:
        """预测准确率：取最近预测误差的互补"""
        # 优先用 _error_history（Learner 的实际属性）
        if hasattr(learner, '_error_history') and learner._error_history:
            errors = list(learner._error_history)[-20:]
            avg_err = sum(errors) / len(errors)
            return max(0.0, 1.0 - avg_err)
        # 兼容旧接口
        if hasattr(learner, 'recent_errors') and learner.recent_errors:
            errors = learner.recent_errors[-20:]
            avg_err = sum(errors) / len(errors)
            return max(0.0, 1.0 - avg_err)
        return 0.0

    def _evaluate_vocabulary_size(self, learner) -> int:
        """已掌握的符号数量"""
        if hasattr(learner, 'get_vocabulary'):
            return len(learner.get_vocabulary())
        if hasattr(learner, 'vocabulary'):
            vocab = learner.vocabulary
            return len(vocabulary) if not isinstance(vocab, dict) else len(vocab)
        if hasattr(learner, 'communication') and hasattr(learner.communication, 'get_vocabulary'):
            return len(learner.communication.get_vocabulary())
        return 0

    def _evaluate_communication_success(self, learner) -> float:
        """沟通成功率"""
        if hasattr(learner, 'comm_success_rate'):
            return learner.comm_success_rate
        if hasattr(learner, 'communication_history'):
            history = learner.communication_history
            if not history:
                return 0.0
            return sum(history) / len(history)
        return 0.0

    def _evaluate_exploration_diversity(self, learner) -> float:
        """探索多样性"""
        if hasattr(learner, 'visited_cells'):
            return len(learner.visited_cells) / 100.0  # 归一化
        if hasattr(learner, 'env') and hasattr(learner.env, 'visited_cells'):
            return len(learner.env.visited_cells) / 100.0
        return 0.0

    def _evaluate_composition_rate(self, learner) -> float:
        """组合使用率：多符号表达占全部沟通的比例"""
        lang = self._get_language(learner)
        if lang is None or lang.total_games == 0:
            return 0.0
        rate = lang.multi_symbol_games / lang.total_games
        # 有复合符号说明组合能力已经涌现
        if hasattr(lang, 'compounds') and lang.compounds:
            rate = min(1.0, rate + 0.1)
        return rate

    def _evaluate_grammar_complexity(self, learner) -> float:
        """语法复杂度：基于学到的语法规则和 n-gram 模式"""
        complexity = 0.0
        lang = self._get_language(learner)

        if lang is not None:
            # n-gram 模式贡献
            if hasattr(lang, 'ngram_patterns'):
                complexity += len(lang.ngram_patterns) * 0.05
            # 修饰词层级贡献
            if hasattr(lang, 'modifier_order'):
                complexity += len(lang.modifier_order) * 0.05

        # 语法规则贡献
        if hasattr(learner, 'communication') and hasattr(learner.communication, 'grammar'):
            rules = learner.communication.grammar.get_rules()
            complexity += len(rules) * 0.1

        return min(1.0, complexity)

    def _get_language(self, learner):
        """获取 EmergingLanguage 对象"""
        if hasattr(learner, 'communication') and hasattr(learner.communication, 'language'):
            return learner.communication.language
        if hasattr(learner, 'language'):
            return learner.language
        return None

    def _get_total_steps(self, learner) -> int:
        """总步数"""
        # 优先用 _total_steps（Learner 的实际属性）
        if hasattr(learner, '_total_steps'):
            return learner._total_steps
        if hasattr(learner, 'total_steps'):
            return learner.total_steps
        if hasattr(learner, 'step_count'):
            return learner.step_count
        return 0

    def _evaluate_skill_history(self, learner, attr_name: str) -> float:
        """评估技能历史准确率（reading/writing/listening/grammar）"""
        history = getattr(learner, attr_name, [])
        if not history:
            return 0.0
        recent = list(history)[-50:]
        return sum(recent) / len(recent)
