"""
学习引擎测试 — TDD 先行

核心验证：
1. 预测编码算法正确性（形状、收敛、误差下降）
2. 好奇心驱动机制（新奇高、熟悉低）
3. CUDA 加速兼容性
4. 局部 Hebbian 更新特性
"""

import sys
sys.path.insert(0, '.')

import pytest
import torch
from src.core.config import LearnerConfig


@pytest.fixture
def config():
    return LearnerConfig(obs_dim=20, action_dim=4, hidden_dims=(32, 16))


@pytest.fixture
def engine(config):
    from src.core.learning_engine import PredictiveCodingEngine
    return PredictiveCodingEngine(config)


class TestPredictShape:
    """预测输出形状正确"""

    def test_predict_returns_correct_shape(self, engine, config):
        obs = torch.randn(config.obs_dim)
        action = torch.tensor(2)  # 离散动作
        output = engine.predict(obs, action)
        assert output.shape == (config.obs_dim,), f"Expected ({config.obs_dim},), got {output.shape}"

    def test_predict_continuous_action(self, engine, config):
        obs = torch.randn(config.obs_dim)
        action = torch.randn(config.action_dim)
        output = engine.predict(obs, action)
        assert output.shape == (config.obs_dim,), f"Expected ({config.obs_dim},), got {output.shape}"


class TestLearningConvergence:
    """学习收敛性"""

    def test_learn_reduces_error_over_time(self, engine, config):
        """100 轮学习后误差应低于初始误差的 50%"""
        obs = torch.randn(config.obs_dim)
        action = torch.tensor(1)
        actual = torch.randn(config.obs_dim)

        errors = []
        for _ in range(100):
            predicted = engine.predict(obs, action)
            error = engine.learn(predicted, actual, obs, action)
            errors.append(error)

        # 后 10 轮平均误差应低于前 10 轮的 50%
        early_avg = sum(errors[:10]) / 10
        late_avg = sum(errors[-10:]) / 10
        assert late_avg < early_avg * 0.5, \
            f"Error not reduced enough: early={early_avg:.4f}, late={late_avg:.4f}"

    def test_learn_returns_positive_error(self, engine, config):
        obs = torch.randn(config.obs_dim)
        action = torch.tensor(0)
        predicted = engine.predict(obs, action)
        actual = torch.randn(config.obs_dim)
        error = engine.learn(predicted, actual, obs, action)
        assert error > 0, "Error should be positive MSE"


class TestGradientFlow:
    """梯度流"""

    def test_learn_with_gradient_returns_input_grad(self, engine, config):
        obs = torch.randn(config.obs_dim)
        action = torch.tensor(1)
        actual = torch.randn(config.obs_dim)
        error, grad = engine.learn_with_input_gradient(obs, action, actual)
        assert grad.shape == (config.obs_dim,), \
            f"Gradient shape should be ({config.obs_dim},), got {grad.shape}"

    def test_gradient_nonzero(self, engine, config):
        obs = torch.randn(config.obs_dim)
        action = torch.tensor(1)
        actual = torch.randn(config.obs_dim)
        error, grad = engine.learn_with_input_gradient(obs, action, actual)
        assert grad.abs().sum() > 0, "Gradient should be non-zero for random input"


class TestCuriosity:
    """好奇心机制"""

    def test_curiosity_high_for_novel_input(self, engine, config):
        novel_input = torch.randn(config.obs_dim)
        curiosity = engine.get_curiosity(novel_input)
        assert curiosity > 0, "Curiosity should be positive for novel input"

    def test_curiosity_decreases_with_repetition(self, engine, config):
        obs = torch.randn(config.obs_dim)
        action = torch.tensor(0)

        curiosities = []
        for _ in range(50):
            predicted = engine.predict(obs, action)
            actual = obs + torch.randn(config.obs_dim) * 0.1  # 小噪声
            engine.learn(predicted, actual, obs, action)
            curiosities.append(engine.get_curiosity(obs))

        # 后期好奇心应低于前期
        early_c = sum(curiosities[:10]) / 10
        late_c = sum(curiosities[-10:]) / 10
        assert late_c < early_c, \
            f"Curiosity should decrease: early={early_c:.4f}, late={late_c:.4f}"


class TestLearningProgress:
    """学习进度"""

    def test_learning_progress_positive(self, engine, config):
        obs = torch.randn(config.obs_dim)
        action = torch.tensor(0)
        actual = torch.randn(config.obs_dim)

        for _ in range(50):
            predicted = engine.predict(obs, action)
            engine.learn(predicted, actual, obs, action)

        progress = engine.get_learning_progress()
        assert progress >= 0, "Learning progress should be non-negative"


class TestCUDA:
    """CUDA 兼容性"""

    def test_cuda_acceleration_works(self, config):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from src.core.learning_engine import PredictiveCodingEngine
        cuda_config = LearnerConfig(
            obs_dim=config.obs_dim, action_dim=config.action_dim,
            hidden_dims=config.hidden_dims, device='cuda'
        )
        engine = PredictiveCodingEngine(cuda_config)

        obs = torch.randn(config.obs_dim).cuda()
        action = torch.tensor(0).cuda()
        actual = torch.randn(config.obs_dim).cuda()

        predicted = engine.predict(obs, action)
        error = engine.learn(predicted, actual, obs, action)

        assert isinstance(error, float)
        assert error > 0


class TestLocalUpdates:
    """局部 Hebbian 更新验证"""

    def test_weights_change_after_learning(self, engine, config):
        obs = torch.randn(config.obs_dim)
        action = torch.tensor(0)

        w1_before = engine.W1.clone()
        w2_before = engine.W2.clone()
        w3_before = engine.W3.clone()

        predicted = engine.predict(obs, action)
        actual = torch.randn(config.obs_dim)
        engine.learn(predicted, actual, obs, action)

        # 权重应该变化
        assert not torch.allclose(engine.W1, w1_before), "W1 should change after learning"
        assert not torch.allclose(engine.W3, w3_before), "W3 should change after learning"
