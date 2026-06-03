"""
统一学习系统 — 课程调度器

实现 ICurriculum 接口，管理阶段晋升与场景生成。
"""

import random
from typing import Dict, List

from src.core.interfaces import ICurriculum
from .stages import DEVELOPMENT_STAGES, get_stage_names, next_stage, get_stage


class CurriculumScheduler(ICurriculum):
    """课程调度器"""

    def __init__(self, strategy: str = 'self_paced'):
        self.strategy = strategy  # 'self_paced' | 'progressive' | 'fixed'
        self.current_stage = 'sensorimotor'
        self.stage_history: List[tuple] = [('sensorimotor', 0)]
        self.evaluator = None  # 延迟导入避免循环

    def _get_evaluator(self):
        if self.evaluator is None:
            from .evaluator import CapabilityEvaluator
            self.evaluator = CapabilityEvaluator()
        return self.evaluator

    # ── ICurriculum 接口 ──────────────────────────────────────────────

    def get_current_stage(self) -> str:
        return self.current_stage

    def evaluate(self, learner) -> Dict[str, float]:
        """评估学习者各项能力指标"""
        evaluator = self._get_evaluator()
        return evaluator.evaluate(learner)

    def should_advance(self, evaluation: Dict[str, float]) -> bool:
        """判断是否满足晋升条件"""
        stage_def = get_stage(self.current_stage)
        criteria = stage_def.promotion_criteria

        if not criteria:
            return False  # 最终阶段，无需晋升

        for key, threshold in criteria.items():
            val = evaluation.get(key)
            if val is None:
                return False
            # 布尔型标准
            if isinstance(threshold, bool):
                if not val:
                    return False
            # 数值型标准
            elif val < threshold:
                return False

        return True

    # ── 晋升 ──────────────────────────────────────────────────────────

    def advance(self) -> bool:
        """晋升到下一阶段，返回是否成功"""
        nxt = next_stage(self.current_stage)
        if nxt is None:
            return False
        self.stage_history.append((nxt, len(self.stage_history)))
        self.current_stage = nxt
        return True

    # ── 场景生成 ──────────────────────────────────────────────────────

    def generate_scene(self, stage: str) -> List[Dict]:
        """
        按阶段生成场景特征列表，供语言游戏使用。
        场景复杂度随阶段递增。
        """
        stage_def = get_stage(stage)

        # 各阶段的物体数量
        stage_objects = {
            'sensorimotor': 3,
            'early_preoperational': 5,
            'late_preoperational': 7,
            'early_concrete': 8,
            'late_concrete': 10,
            'early_formal': 10,
            'late_formal': 12,
            'adolescent': 15,
        }

        n = stage_objects.get(stage, 3)
        shapes = ['circle', 'square', 'triangle']
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'white', 'black']
        materials = ['wood', 'plastic', 'metal', 'glass', 'rubber']
        sizes = ['tiny', 'small', 'medium', 'big', 'huge']

        # 低阶段限制多样性
        if stage in ('sensorimotor',):
            shapes = shapes[:2]
            colors = colors[:3]
            materials = materials[:1]  # ['wood']
            sizes = sizes[1:3]         # ['small', 'medium']

        scene = []
        for i in range(n):
            scene.append({
                'id': i,
                'color': random.choice(colors),
                'shape': random.choice(shapes),
                'material': random.choice(materials),
                'size': random.choice(sizes),
            })
        return scene
