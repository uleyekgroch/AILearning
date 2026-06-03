"""自主学习系统 — 编排器

协调世界模型、自主探索、概念形成、语言接地、自我迭代，
实现完整的自主学习循环。

核心循环（像人类婴儿一样）：
1. 感知：观察世界
2. 预测：基于世界模型预测结果
3. 行动：执行动作
4. 学习：从预测误差中学习
5. 概念化：从经验中形成概念
6. 语言化：将概念与语言关联
7. 反思：评估学习效果，改进策略

与现有系统的区别：
- 现有：加载数据→遍历→掌握度更新
- 新系统：交互→发现→预测→验证→概念化→语言化→自我改进
"""

import time
from typing import Dict, List, Optional, Any

from src.learning.world_model import WorldModel
from src.learning.autonomous_explorer import AutonomousExplorer
from src.learning.concept_former import ConceptFormer
from src.learning.language_grounding_system import LanguageGroundingSystem
from src.learning.self_iterator import SelfIterator


class AutonomousLearningSystem:
    """自主学习系统 — 像人类一样学习

    不喂数据，不预定义知识。
    通过与世界交互，自主发现规律，形成概念，学习语言。
    """

    def __init__(self):
        # 核心组件
        self.world_model = WorldModel()
        self.explorer = AutonomousExplorer()
        self.concept_former = ConceptFormer()
        self.language_system = LanguageGroundingSystem()
        self.self_iterator = SelfIterator()

        # 学习状态
        self.total_steps = 0
        self.total_interactions = 0
        self.learning_history: List[Dict] = []

    def step(self, environment) -> Dict:
        """执行一步自主学习

        Args:
            environment: 环境接口，需要提供:
                - get_state() -> Dict[str, float]
                - get_available_actions() -> List[Dict]
                - step(action) -> Dict (包含 next_state)
                - get_context() -> Dict (可选，上下文信息)

        Returns:
            学习报告
        """
        self.total_steps += 1

        # 1. 感知：获取当前状态
        state = environment.get_state() if hasattr(environment, 'get_state') else {}
        actions = environment.get_available_actions() if hasattr(environment, 'get_available_actions') else [{}]

        # 2. 选择探索目标
        target = self.explorer.select_exploration_target(
            self.world_model, None, actions
        )

        # 3. 执行探索
        result = self.explorer.explore(
            target, environment, self.world_model, None
        )

        # 4. 世界模型学习
        surprise = self.world_model.observe(
            state, result.action_taken, result.outcome
        )

        # 5. 概念形成
        if result.outcome:
            concept_id = self.concept_former.observe(result.outcome)

        # 6. 语言接地
        context = environment.get_context() if hasattr(environment, 'get_context') else {}
        if context.get('word'):
            self.language_system.ground_word(
                context['word'], self.world_model,
                state, result.action_taken, result.outcome
            )

        # 7. 定期发现因果规律
        if self.total_steps % 50 == 0:
            new_laws = self.world_model.discover_laws()

        # 8. 定期自我评估和改进
        if self.total_steps % 100 == 0:
            report = self.self_iterator.evaluate_learning(
                self.world_model, self.concept_former,
                self.explorer, self.language_system
            )
            improvements = self.self_iterator.improve_strategy(
                report, self.world_model, self.explorer
            )

        self.total_interactions += 1

        return {
            'step': self.total_steps,
            'surprise': surprise,
            'target_category': target.category,
            'new_concepts': len(self.concept_former.concepts),
            'world_model_rules': len(self.world_model.rules),
            'grounded_words': len(self.language_system.grounded_words),
        }

    def run_learning_cycles(self, environment, n_cycles: int = 100,
                            log_interval: int = 20) -> Dict:
        """运行多个学习周期

        Args:
            environment: 环境
            n_cycles: 周期数
            log_interval: 日志间隔

        Returns:
            学习报告
        """
        start_time = time.time()
        history = []

        for cycle in range(1, n_cycles + 1):
            result = self.step(environment)
            history.append(result)

            if cycle % log_interval == 0:
                self._log_progress(cycle, n_cycles, result)

        elapsed = time.time() - start_time

        # 最终评估
        final_report = self.self_iterator.evaluate_learning(
            self.world_model, self.concept_former,
            self.explorer, self.language_system
        )

        return {
            'cycles': n_cycles,
            'elapsed': round(elapsed, 1),
            'history': history,
            'final_report': {
                'prediction_accuracy': final_report.prediction_accuracy,
                'knowledge_coverage': final_report.knowledge_coverage,
                'concept_quality': final_report.concept_quality,
                'learning_efficiency': final_report.learning_efficiency,
                'surprise_trend': final_report.surprise_trend,
                'weaknesses': final_report.weaknesses,
                'recommendations': final_report.recommendations,
            },
            'world_model_summary': self.world_model.get_knowledge_summary(),
            'exploration_stats': self.explorer.get_exploration_stats(),
            'concept_stats': self.concept_former.get_stats(),
            'language_stats': self.language_system.get_stats(),
            'iteration_stats': self.self_iterator.get_iteration_stats(),
        }

    def understand_language(self, sentence: str) -> Dict:
        """理解语言（通过世界模型）"""
        return self.language_system.understand_sentence(sentence, self.world_model)

    def get_system_state(self) -> Dict:
        """获取系统完整状态"""
        return {
            'total_steps': self.total_steps,
            'total_interactions': self.total_interactions,
            'world_model': self.world_model.get_knowledge_summary(),
            'explorer': self.explorer.get_exploration_stats(),
            'concepts': self.concept_former.get_stats(),
            'language': self.language_system.get_stats(),
            'self_iteration': self.self_iterator.get_iteration_stats(),
        }

    def _log_progress(self, current: int, total: int, result: Dict):
        """打印进度"""
        print(f"  [Step {current}/{total}] "
              f"surprise={result['surprise']:.3f} "
              f"rules={result['world_model_rules']} "
              f"concepts={result['new_concepts']} "
              f"words={result['grounded_words']}")
