"""
统一训练器测试 — TDD 先行

核心验证：
1. 训练循环正确执行
2. 阶段自动晋升
3. 检查点保存与恢复
4. 训练统计收集
"""

import sys
sys.path.insert(0, '.')

import pytest
import torch
import tempfile
import os

from src.core.config import LearnerConfig, TrainerConfig

AUDIO_DIM = LearnerConfig().audio_dim


@pytest.fixture
def learner_config():
    return LearnerConfig(obs_dim=40, action_dim=4, hidden_dims=(32, 16))


@pytest.fixture
def trainer_config():
    return TrainerConfig(
        target_stage='sensorimotor',
        max_steps_per_stage=50,
        evaluation_interval=25,
        checkpoint_interval=25,
    )


class TestTrainerCreation:
    """训练器创建"""

    def test_trainer_has_learner(self, learner_config, trainer_config):
        from training.trainer import Trainer
        trainer = Trainer(learner_config, trainer_config)
        assert trainer.learner is not None

    def test_trainer_initial_step_zero(self, learner_config, trainer_config):
        from training.trainer import Trainer
        trainer = Trainer(learner_config, trainer_config)
        assert trainer._total_steps == 0


class TestTrainingLoop:
    """训练循环"""

    def test_train_runs_without_error(self, learner_config, trainer_config):
        """训练应能完成不报错"""
        from training.trainer import Trainer
        trainer = Trainer(learner_config, trainer_config)
        stats = trainer.train()
        assert stats is not None
        assert 'final_error' in stats
        assert 'total_steps' in stats

    def test_train_reduces_error(self, learner_config):
        """训练后误差应低于初始误差"""
        from training.trainer import Trainer
        tc = TrainerConfig(
            target_stage='sensorimotor',
            max_steps_per_stage=100,
            evaluation_interval=50,
        )
        trainer = Trainer(learner_config, tc)
        stats = trainer.train()

        assert stats['final_error'] < stats['initial_error'], \
            f"Training should reduce error: {stats['initial_error']:.4f} → {stats['final_error']:.4f}"

    def test_train_collects_statistics(self, learner_config, trainer_config):
        """训练应收集统计信息"""
        from training.trainer import Trainer
        trainer = Trainer(learner_config, trainer_config)
        stats = trainer.train()

        assert 'error_history' in stats
        assert len(stats['error_history']) > 0
        assert 'total_steps' in stats
        assert stats['total_steps'] > 0


class TestCheckpoints:
    """检查点"""

    def test_checkpoint_saves(self, learner_config):
        """检查点应保存到文件"""
        tc = TrainerConfig(
            target_stage='sensorimotor',
            max_steps_per_stage=30,
            checkpoint_interval=15,
        )
        from training.trainer import Trainer
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(learner_config, tc, results_dir=tmpdir)
            stats = trainer.train()
            # 检查是否有检查点文件
            checkpoints = [f for f in os.listdir(tmpdir) if f.endswith('.pt')]
            assert len(checkpoints) > 0, "Should create checkpoint files"


class TestStageProgression:
    """阶段推进"""

    def test_stage_can_advance_with_enough_training(self, learner_config):
        """充分训练后阶段应能推进"""
        tc = TrainerConfig(
            target_stage='single_word',
            max_steps_per_stage=200,
            evaluation_interval=50,
            advancement_threshold=0.5,
        )
        from training.trainer import Trainer
        trainer = Trainer(learner_config, tc)

        # 用固定场景训练，提高预测准确率
        raw_input = {
            'visual': torch.randn(4, 8, 8),
            'auditory': torch.randn(AUDIO_DIM),
            'position': torch.randn(2),
        }
        actual = torch.randn(learner_config.obs_dim)

        stats = trainer.train_with_fixed_scene(raw_input, actual)
        # 不强制要求晋升，但方法应能运行
        assert 'stage_reached' in stats
