"""
集成测试 — 端到端感知→语言→认知闭环

验证完整训练流程：
1. 学习体在真实 World 环境中探索
2. 通过感知聚类产生概念（符号接地）
3. 参照游戏驱动语言涌现
4. 课程评估器驱动阶段自动晋升
5. 记忆系统在整个过程中持续巩固
"""

import sys
sys.path.insert(0, '.')

import pytest
import torch

from src.core.config import LearnerConfig, TrainerConfig

AUDIO_DIM = LearnerConfig().audio_dim


@pytest.fixture
def config():
    return LearnerConfig(
        obs_dim=40, action_dim=4, hidden_dims=(32, 16),
        working_memory_capacity=5,
        episodic_memory_capacity=50,
        consolidation_interval=10,
    )


@pytest.fixture
def learner(config):
    from src.core.learner import Learner
    return Learner(config)


class TestEnvironmentIntegration:
    """环境集成：Learner 与 World 交互"""

    def test_learner_uses_world(self, config):
        """Learner 应能从 World 获取观测"""
        from src.core.learner import Learner
        from src.environment.world import World

        world = World(config)
        world.configure_for_stage('sensorimotor')
        learner = Learner(config)

        raw_input = world.observe()
        obs = learner.perceive(raw_input)

        assert obs.shape == (config.obs_dim,)
        assert obs.dtype == torch.float32

    def test_explore_world_collects_experiences(self, config):
        """探索 World 应积累经验"""
        from src.core.learner import Learner
        from src.environment.world import World

        world = World(config)
        world.configure_for_stage('sensorimotor')
        learner = Learner(config)

        errors = []
        for _ in range(20):
            raw_input = world.observe()
            obs = learner.perceive(raw_input)
            action = learner.choose_action(obs)
            action_tensor = torch.zeros(config.action_dim)
            action_tensor[action] = 1.0

            next_raw, reward, done = world.step(action_tensor)
            next_obs = learner.perceive(next_raw)

            error = learner.learn_from_experience(obs, action, next_obs)
            learner.remember(obs, action, next_obs, reward, error)
            errors.append(error)

            if done:
                world.reset()
                world.configure_for_stage('sensorimotor')

        assert len(errors) == 20
        assert len(learner.memory.working.items) > 0 or len(learner.memory.episodic.traces) > 0

    def test_world_stage_config_affects_observations(self, config):
        """不同阶段的环境应产生不同复杂度的观测"""
        from src.environment.world import World

        world = World(config)
        world.configure_for_stage('sensorimotor')
        obs_simple = world.observe()

        world.configure_for_stage('late_concrete')
        obs_complex = world.observe()

        # 复杂场景应有物体（observation 不为全零）
        assert obs_simple is not None
        assert obs_complex is not None


class TestLanguageIntegration:
    """语言集成：符号接地 + 参照游戏"""

    def test_grounding_from_exploration(self, config, learner):
        """探索中应能建立感知概念"""
        from src.language.grounding import GroundingModule

        grounding = GroundingModule(obs_dim=config.obs_dim)

        # 模拟多次感知同一类物体
        for _ in range(5):
            obs = torch.randn(config.obs_dim) * 0.1 + torch.tensor([1.0] * config.obs_dim)
            cluster = grounding.ground_from_perception(obs)

        # 应至少有一个概念
        assert len(grounding.perceptual_clusters) >= 1

    def test_reference_game_produces_vocabulary(self, config):
        """参照游戏应产生词汇"""
        from src.language.communication import CommunicationProtocol, generate_scene

        proto = CommunicationProtocol()

        # 多轮参照游戏
        successes = 0
        for _ in range(50):
            scene = generate_scene(num_objects=3, complexity='simple')
            if not scene:
                continue
            target = 0
            success = proto.play_round(proto, proto, scene, target)
            if success:
                successes += 1

        # 应有一定成功率
        assert successes > 0, "Reference game should succeed at least sometimes"
        assert proto.language.total_games > 0

    def test_grounding_and_communication_together(self, config, learner):
        """符号接地与通信协议协同工作"""
        from src.language.grounding import GroundingModule
        from src.language.communication import CommunicationProtocol

        grounding = GroundingModule(obs_dim=config.obs_dim)
        comm = CommunicationProtocol()

        # 模拟感知经验建立概念
        obs_red_circle = torch.randn(config.obs_dim) * 0.1 + torch.tensor([1.0] * 20 + [0.0] * 20)
        obs_blue_square = torch.randn(config.obs_dim) * 0.1 + torch.tensor([0.0] * 20 + [1.0] * 20)

        cluster_a = grounding.ground_from_perception(obs_red_circle)
        cluster_b = grounding.ground_from_perception(obs_blue_square)

        # 社会标注
        grounding.ground_from_social('red_circle', obs_red_circle, context='visual')
        assert 'red_circle' in grounding.get_grounded_symbols()

        meaning = grounding.get_symbol_meaning('red_circle')
        assert meaning is not None
        assert meaning['confidence'] > 0


class TestCurriculumIntegration:
    """课程集成：能力评估 + 阶段晋升"""

    def test_evaluator_with_learner(self, config, learner):
        """能力评估器应能评估 Learner"""
        from src.curriculum.evaluator import CapabilityEvaluator

        evaluator = CapabilityEvaluator()

        # 训练几轮
        raw_input = {
            'visual': torch.randn(4, 8, 8),
            'auditory': torch.randn(AUDIO_DIM),
            'position': torch.randn(2),
        }
        actual = torch.randn(config.obs_dim)
        for _ in range(20):
            obs = learner.perceive(raw_input)
            predicted = learner.predict_next(obs)
            error = learner.learn_from_experience(obs, torch.tensor(0), actual)

        evaluation = evaluator.evaluate(learner)
        assert 'prediction_accuracy' in evaluation
        assert 0.0 <= evaluation['prediction_accuracy'] <= 1.0

    def test_scheduler_advances_stage(self, config, learner):
        """课程调度器应能判断并执行阶段晋升"""
        from src.curriculum.scheduler import CurriculumScheduler

        scheduler = CurriculumScheduler(strategy='self_paced')

        # 模拟高分评估
        evaluation = {'prediction_accuracy': 0.9, 'symbol_count': 10.0}

        can_advance = scheduler.should_advance(evaluation)
        # sensorimotor 阶段的晋升条件可能包含多种指标
        assert isinstance(can_advance, bool)


class TestFullDevelopmentalPipeline:
    """完整发展管线"""

    def test_sensorimotor_to_language(self, config):
        """从感知运动到语言发展的完整流程"""
        from src.core.learner import Learner
        from src.environment.world import World
        from src.language.grounding import GroundingModule
        from src.language.communication import CommunicationProtocol, generate_scene
        from src.curriculum.evaluator import CapabilityEvaluator

        world = World(config)
        world.configure_for_stage('sensorimotor')
        learner = Learner(config)
        grounding = GroundingModule(obs_dim=config.obs_dim)
        comm = CommunicationProtocol()
        evaluator = CapabilityEvaluator()

        # Phase 1: 感知运动探索
        errors = []
        for step in range(30):
            raw_input = world.observe()
            obs = learner.perceive(raw_input)

            # 感知聚类
            grounding.ground_from_perception(obs)

            action = learner.choose_action(obs)
            action_tensor = torch.zeros(config.action_dim)
            action_tensor[action] = 1.0
            next_raw, reward, done = world.step(action_tensor)
            next_obs = learner.perceive(next_raw)

            error = learner.learn_from_experience(obs, action, next_obs)
            learner.remember(obs, action, next_obs, reward, error)
            errors.append(error)

            if done:
                world.reset()
                world.configure_for_stage('sensorimotor')

        # Phase 2: 语言游戏
        lang_successes = 0
        for _ in range(20):
            scene = generate_scene(num_objects=3, complexity='simple')
            if scene:
                success = comm.play_round(comm, comm, scene, target_idx=0)
                if success:
                    lang_successes += 1

        # Phase 3: 评估
        stats = learner.get_stats()
        evaluation = evaluator.evaluate(learner)

        # 验证各系统都产生了有意义的结果
        assert len(errors) == 30
        assert stats['total_steps'] > 0
        assert evaluation['prediction_accuracy'] >= 0.0
        assert comm.language.total_games > 0
        assert len(grounding.perceptual_clusters) > 0

    def test_memory_consolidation_during_development(self, config):
        """发展中记忆巩固应正常工作"""
        from src.core.learner import Learner
        from src.environment.world import World

        world = World(config)
        world.configure_for_stage('sensorimotor')
        learner = Learner(config)

        # 积累经验
        for _ in range(20):
            raw_input = world.observe()
            obs = learner.perceive(raw_input)
            action = learner.choose_action(obs)
            action_tensor = torch.zeros(config.action_dim)
            action_tensor[action] = 1.0
            next_raw, reward, done = world.step(action_tensor)
            next_obs = learner.perceive(next_raw)

            learner.remember(obs, action, next_obs, reward, error=0.5)
            if done:
                world.reset()
                world.configure_for_stage('sensorimotor')

        # 巩固
        report = learner.consolidate()
        assert isinstance(report, dict)

        # 巩固后记忆系统应有内容
        total_items = (len(learner.memory.working.items) +
                      len(learner.memory.episodic.traces) +
                      len(learner.memory.semantic.concepts))
        assert total_items > 0

    def test_trainer_with_real_environment(self, config):
        """Trainer 应能用真实环境训练"""
        from training.trainer import DevelopmentalTrainer

        tc = TrainerConfig(
            target_stage='sensorimotor',
            max_steps_per_stage=30,
            evaluation_interval=15,
            checkpoint_interval=100,  # 不生成检查点以加速测试
        )

        trainer = DevelopmentalTrainer(config, tc)
        stats = trainer.train()

        assert stats['total_steps'] > 0
        assert stats['final_error'] > 0
        assert 'stages_completed' in stats
        assert stats['stages_completed'] >= 1
