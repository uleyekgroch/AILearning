"""
统一训练编排器 — 发展阶段自动推进

实现"像小孩一样学习"的核心训练循环：
  sensorimotor → single_word → two_word → complex → literacy

训练流程：
  1. 按阶段配置环境复杂度
  2. 好奇心驱动探索
  3. 定期评估能力
  4. 满足条件自动晋升
  5. 定期保存检查点
"""

import torch
import os
import json
from typing import Dict, Optional
from datetime import datetime

from src.core.config import LearnerConfig, TrainerConfig
from src.core.learner import Learner, STAGE_ORDER


class Trainer:
    """统一训练编排器"""

    def __init__(self, learner_config: LearnerConfig,
                 trainer_config: TrainerConfig,
                 results_dir: str = None):
        self.learner_config = learner_config
        self.config = trainer_config
        self.results_dir = results_dir or trainer_config.results_dir

        self.learner = Learner(learner_config)
        self._total_steps = 0

        os.makedirs(self.results_dir, exist_ok=True)

    def train(self) -> Dict:
        """运行完整训练流程（到目标阶段）

        自动生成环境场景，驱动好奇心探索。
        """
        stats = {
            'initial_error': None,
            'final_error': None,
            'error_history': [],
            'total_steps': 0,
            'stage_reached': self.learner.stage,
            'stage_history': [],
        }

        # 训练到目标阶段
        target_idx = STAGE_ORDER.index(self.config.target_stage)

        for stage_idx in range(target_idx + 1):
            stage_name = STAGE_ORDER[stage_idx]
            max_steps = self.config.max_steps_per_stage

            stage_errors = []
            for step in range(max_steps):
                # 生成随机场景
                raw_input = self._generate_scene(stage_name)
                obs = self.learner.perceive(raw_input)

                # 预测 + 行动
                action = self.learner.choose_action(obs)
                next_input = self._generate_scene(stage_name)
                next_obs = self.learner.perceive(next_input)

                # 学习
                error = self.learner.learn_from_experience(obs, action, next_obs)
                self.learner.remember(obs, action, next_obs, reward=1.0, error=error)

                stage_errors.append(error)
                self._total_steps += 1

                # 记录初始误差
                if stats['initial_error'] is None:
                    stats['initial_error'] = error

                # 定期检查点
                if self._total_steps % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()

                # 词汇剪枝（防止长训练内存膨胀）
                if step > 0 and step % 500 == 0:
                    lang = getattr(self.learner, 'communication', None)
                    lang = getattr(lang, 'language', None) if lang else None
                    if lang is not None:
                        pruned = lang.prune_all()
                        if any(v > 0 for v in pruned.values()):
                            print(f"  [Step {step}] 词汇剪枝: {pruned}")

            # 阶段统计
            stats['stage_history'].append({
                'stage': stage_name,
                'steps': max_steps,
                'avg_error': sum(stage_errors) / len(stage_errors),
            })

            # 评估 + 晋升
            if step >= self.config.evaluation_interval:
                evaluation = self._evaluate()
                self.learner.try_advance(evaluation)

            stats['stage_reached'] = self.learner.stage

        stats['final_error'] = stage_errors[-1] if stage_errors else 0.0
        stats['error_history'] = stage_errors
        stats['total_steps'] = self._total_steps

        # 保存最终检查点
        self._save_checkpoint()

        # 保存结果
        self._save_results(stats)

        return stats

    def train_with_fixed_scene(self, raw_input: Dict[str, torch.Tensor],
                                actual: torch.Tensor) -> Dict:
        """用固定场景训练（加速收敛，用于测试）

        模拟"同一场景反复体验"的学习过程。
        """
        stats = {
            'initial_error': None,
            'final_error': None,
            'error_history': [],
            'total_steps': 0,
            'stage_reached': self.learner.stage,
        }

        target_idx = STAGE_ORDER.index(self.config.target_stage)

        for stage_idx in range(target_idx + 1):
            stage_name = STAGE_ORDER[stage_idx]
            max_steps = self.config.max_steps_per_stage

            for step in range(max_steps):
                obs = self.learner.perceive(raw_input)
                action = self.learner.choose_action(obs)
                error = self.learner.learn_from_experience(obs, action, actual)
                self.learner.remember(obs, action, actual, reward=1.0, error=error)

                stats['error_history'].append(error)
                self._total_steps += 1

                if stats['initial_error'] is None:
                    stats['initial_error'] = error

                # 定期评估
                if step > 0 and step % self.config.evaluation_interval == 0:
                    evaluation = self._evaluate()
                    self.learner.try_advance(evaluation)

        stats['final_error'] = stats['error_history'][-1] if stats['error_history'] else 0.0
        stats['total_steps'] = self._total_steps
        stats['stage_reached'] = self.learner.stage

        return stats

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _generate_scene(self, stage: str) -> Dict[str, torch.Tensor]:
        """根据发展阶段生成场景复杂度

        sensorimotor: 简单静态场景
        single_word: 加入物体颜色/形状变化
        two_word: 多物体场景
        complex: 动态场景 + 交互
        literacy: 复杂抽象场景
        """
        complexity = {
            'sensorimotor': 1.0,
            'single_word': 1.5,
            'two_word': 2.0,
            'complex': 3.0,
            'literacy': 4.0,
        }.get(stage, 1.0)

        return {
            'visual': torch.randn(4, 8, 8) * complexity,
            'auditory': torch.randn(13) * complexity,
            'position': torch.randn(2),
        }

    def _evaluate(self) -> Dict[str, float]:
        """评估学习者当前能力"""
        engine_stats = self.learner.get_stats()
        progress = engine_stats['learning_progress']
        avg_error = engine_stats['avg_error']

        return {
            'prediction_accuracy': max(0.0, 1.0 - avg_error),
            'learning_progress': progress,
            'vocabulary_size': 0,  # 语言系统就绪后更新
            'composition_rate': 0.0,
        }

    def _save_checkpoint(self) -> None:
        """保存训练检查点"""
        path = os.path.join(self.results_dir, f'checkpoint_{self._total_steps}.pt')
        self.learner.save(path)

    def _save_results(self, stats: Dict) -> None:
        """保存训练结果"""
        # 将 tensor 转为 float 以便 JSON 序列化
        clean_stats = {}
        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                clean_stats[k] = v.item()
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], float):
                clean_stats[k] = v
            elif isinstance(v, (int, float, str)):
                clean_stats[k] = v
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                clean_stats[k] = v

        path = os.path.join(self.results_dir, 'training_results.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(clean_stats, f, indent=2, ensure_ascii=False)


class DevelopmentalTrainer:
    """发展型训练器 — 真实环境 + 语言游戏 + 阶段自动推进

    训练流程：
      1. 创建 World 环境，按阶段配置复杂度
      2. 好奇心驱动探索，积累感知-运动经验
      3. 感知聚类 → 概念接地 → 符号涌现
      4. 参照游戏驱动语言发展
      5. 能力评估 → 阶段自动晋升
      6. 记忆巩固（模拟睡眠）
    """

    # 简化阶段映射：5 阶段 → 8 Piaget 阶段
    STAGE_TO_PIAGET = {
        'sensorimotor': 'sensorimotor',
        'single_word': 'early_preoperational',
        'two_word': 'late_preoperational',
        'complex': 'early_concrete',
        'literacy': 'late_concrete',
    }

    def __init__(self, learner_config: LearnerConfig,
                 trainer_config: TrainerConfig,
                 results_dir: str = None):
        self.learner_config = learner_config
        self.config = trainer_config
        self.results_dir = results_dir or trainer_config.results_dir

        self.learner = Learner(learner_config)
        self._total_steps = 0

        os.makedirs(self.results_dir, exist_ok=True)

    def train(self) -> Dict:
        """运行完整发展训练"""
        from src.environment.world import World
        from src.curriculum.evaluator import CapabilityEvaluator
        from src.language.communication import generate_scene

        world = World(self.learner_config)
        evaluator = CapabilityEvaluator()

        stats = {
            'initial_error': None,
            'final_error': None,
            'error_history': [],
            'total_steps': 0,
            'stage_reached': self.learner.stage,
            'stages_completed': 0,
            'lang_games': 0,
            'lang_successes': 0,
            'grounded_concepts': 0,
        }

        target_idx = STAGE_ORDER.index(self.config.target_stage)

        for stage_idx in range(target_idx + 1):
            stage_name = STAGE_ORDER[stage_idx]
            piaget_stage = self.STAGE_TO_PIAGET.get(stage_name, 'sensorimotor')

            # 配置环境
            world.configure_for_stage(piaget_stage)
            max_steps = self.config.max_steps_per_stage

            stage_errors = []
            for step in range(max_steps):
                raw_input = world.observe()
                obs = self.learner.perceive(raw_input)

                # 感知概念接地
                self.learner.ground_concept(obs)

                # 行动
                action = self.learner.choose_action(obs)
                action_tensor = torch.zeros(self.learner_config.action_dim)
                action_tensor[action] = 1.0
                next_raw, reward, done = world.step(action_tensor)
                next_obs = self.learner.perceive(next_raw)

                # 学习
                error = self.learner.learn_from_experience(obs, action, next_obs)
                self.learner.remember(obs, action, next_obs, reward, error)

                stage_errors.append(error)
                self._total_steps += 1

                if stats['initial_error'] is None:
                    stats['initial_error'] = error

                # 定期语言游戏（single_word 以后）
                if stage_idx >= 1 and step % 5 == 0:
                    scene = generate_scene(num_objects=3, complexity='simple')
                    if scene:
                        target = 0
                        success = self.learner.play_reference_game(scene, target)
                        stats['lang_games'] += 1
                        if success:
                            stats['lang_successes'] += 1

                # 环境重置
                if done:
                    world.configure_for_stage(piaget_stage)

                # 检查点
                if self._total_steps % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()

                # 词汇剪枝（防止长训练内存膨胀）
                if step > 0 and step % 500 == 0:
                    lang = getattr(self.learner, 'communication', None)
                    lang = getattr(lang, 'language', None) if lang else None
                    if lang is not None:
                        pruned = lang.prune_all()
                        if any(v > 0 for v in pruned.values()):
                            print(f"  [Step {step}] 词汇剪枝: {pruned}")

            # 阶段结束：巩固 + 评估
            self.learner.consolidate()

            evaluation = evaluator.evaluate(self.learner)
            evaluation['vocabulary_size'] = float(len(self.learner.get_vocabulary()))

            self.learner.try_advance(evaluation)
            stats['stages_completed'] += 1

        # 最终统计
        learner_stats = self.learner.get_stats()
        stats['final_error'] = stage_errors[-1] if stage_errors else 0.0
        stats['error_history'] = stage_errors
        stats['total_steps'] = self._total_steps
        stats['stage_reached'] = self.learner.stage
        stats['grounded_concepts'] = learner_stats['perceptual_clusters']
        stats['vocabulary_size'] = learner_stats['vocabulary_size']

        self._save_checkpoint()
        self._save_results(stats)

        return stats

    def _save_checkpoint(self) -> None:
        path = os.path.join(self.results_dir, f'checkpoint_{self._total_steps}.pt')
        self.learner.save(path)

    def _save_results(self, stats: Dict) -> None:
        clean = {}
        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                clean[k] = v.item()
            elif isinstance(v, (int, float, str, list)):
                clean[k] = v
        path = os.path.join(self.results_dir, 'training_results.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
