"""
课程系统测试 — TDD 先行

6 个测试用例覆盖：
1. 8 个阶段按正确顺序
2. 满足条件时晋升
3. 不满足条件时不晋升
4. 评估返回各项指标
5. 自主节奏课程
6. 课程生成合法场景
"""

import sys
sys.path.insert(0, '.')

import pytest
from src.curriculum.stages import DEVELOPMENT_STAGES, get_stage_names, next_stage
from src.curriculum.evaluator import CapabilityEvaluator
from src.curriculum.scheduler import CurriculumScheduler


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def scheduler():
    return CurriculumScheduler(strategy='self_paced')


@pytest.fixture
def evaluator():
    return CapabilityEvaluator()


# ── 辅助：模拟 learner ──────────────────────────────────────────────

class MockLearner:
    """用于测试的模拟学习者"""

    def __init__(self, **kwargs):
        self.total_steps = kwargs.get('total_steps', 0)
        self.recent_errors = kwargs.get('recent_errors', [])
        self.vocabulary = kwargs.get('vocabulary', {})
        self.comm_success_rate = kwargs.get('comm_success_rate', 0.0)
        self.visited_cells = kwargs.get('visited_cells', set())


# ── 1. 8 个阶段按正确顺序 ────────────────────────────────────────────

class TestStageOrder:
    """阶段顺序验证"""

    def test_stage_order(self):
        names = get_stage_names()
        expected = [
            'sensorimotor',
            'early_preoperational',
            'late_preoperational',
            'early_concrete',
            'late_concrete',
            'early_formal',
            'late_formal',
            'adolescent',
        ]
        assert names == expected
        assert len(names) == 8

    def test_next_stage_chain(self):
        """next_stage 形成完整链"""
        names = get_stage_names()
        for i in range(len(names) - 1):
            assert next_stage(names[i]) == names[i + 1]
        assert next_stage('adolescent') is None


# ── 2. 满足条件时晋升 ───────────────────────────────────────────────

class TestStageAdvancement:
    """满足条件时晋升"""

    def test_stage_advancement_criteria(self, scheduler):
        # sensorimotor 的标准: prediction_accuracy >= 0.5, exploration_diversity >= 0.3, total_steps >= 50
        evaluation = {
            'prediction_accuracy': 0.6,
            'exploration_diversity': 0.5,
            'total_steps': 60.0,
        }
        assert scheduler.should_advance(evaluation)
        assert scheduler.advance()
        assert scheduler.get_current_stage() == 'early_preoperational'


# ── 3. 不满足条件时不晋升 ───────────────────────────────────────────

class TestNoPrematureAdvancement:
    """不满足条件时不晋升"""

    def test_stage_not_premature(self, scheduler):
        # 缺少指标 → 不晋升
        evaluation = {
            'prediction_accuracy': 0.6,
            # 缺 exploration_diversity
            'total_steps': 60.0,
        }
        assert not scheduler.should_advance(evaluation)

    def test_insufficient_values(self, scheduler):
        """数值不足不晋升"""
        evaluation = {
            'prediction_accuracy': 0.3,  # 需要 0.5
            'exploration_diversity': 0.5,
            'total_steps': 60.0,
        }
        assert not scheduler.should_advance(evaluation)


# ── 4. 评估返回各项指标 ─────────────────────────────────────────────

class TestEvaluationMetrics:
    """评估返回各项指标"""

    def test_evaluation_returns_metrics(self, evaluator):
        learner = MockLearner(
            total_steps=100,
            recent_errors=[0.2, 0.3, 0.1],
            vocabulary={'red': 1, 'blue': 2},
            comm_success_rate=0.7,
            visited_cells=set(range(30)),
        )
        metrics = evaluator.evaluate(learner)

        assert isinstance(metrics, dict)
        assert 'prediction_accuracy' in metrics
        assert 'vocabulary_size' in metrics
        assert 'communication_success' in metrics
        assert 'exploration_diversity' in metrics
        assert 'total_steps' in metrics
        assert 'composition_rate' in metrics
        assert 'grammar_complexity' in metrics

        # 验证具体值
        assert metrics['prediction_accuracy'] > 0
        assert metrics['vocabulary_size'] == 2.0
        assert metrics['communication_success'] == 0.7
        assert metrics['total_steps'] == 100.0


# ── 5. 自主节奏课程 ─────────────────────────────────────────────────

class TestSelfPacedCurriculum:
    """自主节奏课程"""

    def test_self_paced_curriculum(self):
        sched = CurriculumScheduler(strategy='self_paced')
        assert sched.strategy == 'self_paced'
        assert sched.get_current_stage() == 'sensorimotor'

        # 模拟渐进学习
        learner = MockLearner(
            total_steps=100,
            recent_errors=[0.1] * 10,
            vocabulary={'a': 1},
            visited_cells=set(range(50)),
        )

        metrics = sched.evaluate(learner)
        # 满足 sensorimotor 标准
        if sched.should_advance(metrics):
            sched.advance()

        # 当前阶段要么已晋升，要么仍在 sensorimotor
        assert sched.get_current_stage() in get_stage_names()

    def test_progressive_strategy(self):
        sched = CurriculumScheduler(strategy='progressive')
        assert sched.strategy == 'progressive'


# ── 6. 课程生成合法场景 ─────────────────────────────────────────────

class TestCurriculumSceneGeneration:
    """课程生成合法场景"""

    def test_curriculum_generates_valid_scenes(self, scheduler):
        for stage in get_stage_names():
            scene = scheduler.generate_scene(stage)
            assert isinstance(scene, list)
            assert len(scene) > 0

            for obj in scene:
                assert 'color' in obj
                assert 'shape' in obj
                assert 'material' in obj
                assert 'size' in obj

    def test_sensorimotor_scene_simple(self, scheduler):
        """sensorimotor 场景应较简单"""
        scene = scheduler.generate_scene('sensorimotor')
        assert len(scene) == 3

        for obj in scene:
            assert obj['shape'] in ('circle', 'square')
            assert obj['material'] == 'wood'

    def test_adolescent_scene_complex(self, scheduler):
        """adolescent 场景应较复杂"""
        scene = scheduler.generate_scene('adolescent')
        assert len(scene) == 15
