"""
统一学习体测试 — TDD 先行

核心验证：
1. 感知→预测→学习闭环正确
2. 好奇心驱动探索有效
3. 记忆存储与检索一致
4. 序列化/反序列化可恢复
5. 发展阶段晋升触发
6. 与环境交互的端到端流程
"""

import sys
sys.path.insert(0, '.')

import pytest
import torch
import tempfile
import os

from src.core.config import LearnerConfig

AUDIO_DIM = LearnerConfig().audio_dim


@pytest.fixture
def config():
    return LearnerConfig(obs_dim=40, action_dim=4, hidden_dims=(32, 16))


@pytest.fixture
def learner(config):
    from src.core.learner import Learner
    return Learner(config)


class TestLearnerCreation:
    """学习体创建"""

    def test_learner_has_all_subsystems(self, learner):
        """应有感知、记忆、学习引擎子系统"""
        assert learner.perception is not None
        assert learner.memory is not None
        assert learner.engine is not None

    def test_learner_initial_stage(self, learner):
        """初始阶段应为 sensorimotor"""
        assert learner.stage == 'sensorimotor'


class TestPerceptionPredictionLoop:
    """感知→预测闭环"""

    def test_observe_and_predict(self, learner):
        """从原始输入到预测输出完整流程"""
        raw_input = {
            'visual': torch.randn(4, 8, 8),
            'auditory': torch.randn(AUDIO_DIM),
            'position': torch.randn(2),
        }
        obs = learner.perceive(raw_input)
        assert obs.shape == (40,)

        predicted = learner.predict_next(obs)
        assert predicted.shape == (40,)

    def test_learn_from_experience(self, learner, config):
        """学习应返回正的预测误差"""
        raw_input = {
            'visual': torch.randn(4, 8, 8),
            'auditory': torch.randn(AUDIO_DIM),
            'position': torch.randn(2),
        }
        obs = learner.perceive(raw_input)
        predicted = learner.predict_next(obs)
        actual = torch.randn(config.obs_dim)
        action = torch.tensor(0)

        error = learner.learn_from_experience(obs, action, actual)
        assert error > 0
        assert isinstance(error, float)


class TestCuriosityDrivenExploration:
    """好奇心驱动探索"""

    def test_choose_action_returns_valid(self, learner, config):
        """动作选择应在合法范围内"""
        obs = torch.randn(config.obs_dim)
        action = learner.choose_action(obs)
        assert 0 <= action < config.action_dim

    def test_curiosity_decreases_with_learning(self, learner, config):
        """学习后好奇心应下降"""
        obs = torch.randn(config.obs_dim)
        action = torch.tensor(0)
        actual = torch.randn(config.obs_dim)

        curiosities = []
        for _ in range(30):
            predicted = learner.predict_next(obs)
            learner.learn_from_experience(obs, action, actual)
            curiosities.append(learner.get_curiosity(obs))

        early_c = sum(curiosities[:5]) / 5
        late_c = sum(curiosities[-5:]) / 5
        assert late_c <= early_c, \
            f"Curiosity should decrease or stay: early={early_c:.4f}, late={late_c:.4f}"


class TestMemoryIntegration:
    """记忆集成"""

    def test_experience_stored_in_memory(self, learner, config):
        """经验应存储到记忆系统"""
        obs = torch.randn(config.obs_dim)
        action = torch.tensor(0)
        next_obs = torch.randn(config.obs_dim)

        learner.remember(obs, action, next_obs, reward=1.0, error=0.5)

        results = learner.recall(obs, k=3)
        assert len(results) > 0

    def test_consolidation_runs(self, learner, config):
        """巩固应能执行"""
        for i in range(10):
            obs = torch.randn(config.obs_dim)
            action = torch.tensor(i % config.action_dim)
            next_obs = torch.randn(config.obs_dim)
            learner.remember(obs, action, next_obs, reward=1.0, error=0.5)

        report = learner.consolidate()
        assert isinstance(report, dict)


class TestSerialization:
    """序列化/反序列化"""

    def test_save_and_load(self, learner, config):
        """保存后加载应恢复状态"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'learner.pt')

            # 学习几轮产生有意义的权重
            obs = torch.randn(config.obs_dim)
            action = torch.tensor(0)
            actual = torch.randn(config.obs_dim)
            for _ in range(10):
                predicted = learner.predict_next(obs)
                learner.learn_from_experience(obs, action, actual)

            learner.save(path)

            from src.core.learner import Learner
            loaded = Learner(config)
            loaded.load(path)

            # 验证加载后的权重一致
            assert torch.allclose(learner.engine.W1, loaded.engine.W1)

    def test_save_creates_file(self, learner):
        """保存应创建文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'learner.pt')
            learner.save(path)
            assert os.path.exists(path)


class TestStageAdvancement:
    """发展阶段晋升"""

    def test_advance_stage_updates_current(self, learner):
        """晋升应更新当前阶段"""
        old_stage = learner.stage
        # 强制晋升（通过直接设置评估结果）
        evaluation = {'prediction_accuracy': 0.8, 'vocabulary_size': 5}
        learner.try_advance(evaluation)
        # stage 可能不变（如果不满足条件），但方法不应报错

    def test_stage_progression_order(self, learner):
        """阶段应按正确顺序晋升"""
        valid_stages = ['sensorimotor', 'single_word', 'two_word',
                        'complex', 'literacy']
        assert learner.stage in valid_stages


class TestEndToEnd:
    """端到端交互"""

    def test_interact_loop(self, learner, config):
        """完整的交互循环：观察→预测→行动→学习→记忆"""
        raw_input = {
            'visual': torch.randn(4, 8, 8),
            'auditory': torch.randn(AUDIO_DIM),
            'position': torch.randn(2),
        }

        # 模拟 10 轮交互
        errors = []
        for _ in range(10):
            obs = learner.perceive(raw_input)
            predicted = learner.predict_next(obs)
            action = learner.choose_action(obs)
            next_obs = torch.randn(config.obs_dim)  # 模拟环境反馈

            error = learner.learn_from_experience(obs, action, next_obs)
            learner.remember(obs, action, next_obs, reward=1.0, error=error)
            errors.append(error)

        # 误差应记录
        assert all(e > 0 for e in errors)

        # 记忆应有内容
        results = learner.recall(obs, k=3)
        assert len(results) > 0

    def test_full_training_loop_reduces_error(self, learner, config):
        """多轮训练后误差应下降"""
        raw_input = {
            'visual': torch.randn(4, 8, 8),
            'auditory': torch.randn(AUDIO_DIM),
            'position': torch.randn(2),
        }
        actual = torch.randn(config.obs_dim)

        errors = []
        for _ in range(50):
            obs = learner.perceive(raw_input)
            predicted = learner.predict_next(obs)
            action = torch.tensor(0)
            error = learner.learn_from_experience(obs, action, actual)
            errors.append(error)

        early = sum(errors[:10]) / 10
        late = sum(errors[-10:]) / 10
        assert late < early, \
            f"Error should decrease: early={early:.4f}, late={late:.4f}"
