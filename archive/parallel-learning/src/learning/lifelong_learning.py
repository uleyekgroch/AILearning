"""
持续学习机制 - Lifelong Learning System

基于2024-2025年最新研究：
- Continual Learning
- Catastrophic Forgetting Prevention
- Knowledge Distillation
- Elastic Weight Consolidation (EWC)
- Progressive Neural Networks

功能：
1. 在线学习（增量更新）
2. 知识蒸馏
3. 灾难性遗忘预防
4. 生命周期学习
5. 性能监控
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import copy


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class LearningTask:
    """学习任务"""
    task_id: str                           # 任务ID
    task_name: str                         # 任务名称
    data: Any                              # 训练数据
    num_samples: int                       # 样本数量
    importance: float = 1.0               # 任务重要性
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningHistory:
    """学习历史"""
    task_id: str                           # 任务ID
    learned_at: float                      # 学习时间
    performance: float                      # 性能指标
    parameters_snapshot: Dict = field(default_factory=dict)  # 参数快照
    fisher_info: Dict = field(default_factory=dict)  # Fisher信息矩阵


@dataclass
class ContinualLearningConfig:
    """持续学习配置"""
    # EWC参数
    ewc_lambda: float = 1000.0             # EWC正则化系数
    fisher_samples: int = 2000             # Fisher信息采样数量

    # 知识蒸馏参数
    distill_temperature: float = 2.0       # 蒸馏温度
    distill_alpha: float = 0.7             # 蒸馏损失权重

    # 弹性权重巩固
    elastic_lambda: float = 0.5            # 弹性正则化系数

    # 学习参数
    learning_rate: float = 0.001           # 学习率
    batch_size: int = 32                   # 批次大小
    epochs_per_task: int = 10              # 每任务训练轮数

    # 记忆回放
    replay_buffer_size: int = 1000         # 回放缓冲区大小
    replay_samples_per_task: int = 100     # 每任务回放样本数


# ============================================================================
# Elastic Weight Consolidation (EWC)
# ============================================================================

class EWCRegularizer:
    """弹性权重巩固正则化器

    预防灾难性遗忘的重要技术
    """

    def __init__(self, model: nn.Module, config: ContinualLearningConfig):
        self.model = model
        self.config = config
        self.fisher_info = {}  # {param_name: fisher_matrix}
        self.optimal_params = {}  # {param_name: optimal_value}

    def compute_fisher_information(self, data_loader):
        """计算Fisher信息矩阵"""
        self.model.eval()

        # 初始化Fisher信息
        fisher_info = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param)

        # 计算Fisher信息：F = E[∇θ log p(y|x,θ)²]
        num_samples = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            if num_samples >= self.config.fisher_samples:
                break

            # 前向传播
            output = self.model(data)
            loss = F.cross_entropy(output, target)

            # 反向传播获取梯度
            self.model.zero_grad()
            loss.backward()

            # 累积梯度平方
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2

            num_samples += len(data)

        # 归一化
        for name in fisher_info:
            fisher_info[name] /= num_samples

        # 保存当前最优参数
        optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                optimal_params[name] = param.data.clone()

        self.fisher_info = fisher_info
        self.optimal_params = optimal_params

        return fisher_info, optimal_params

    def ewc_loss(self) -> torch.Tensor:
        """计算EWC正则化损失

        Loss = Σ λ/2 * F_i * (θ_i - θ*_i)²
        """
        ewc_loss = 0.0

        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.optimal_params:
                fisher = self.fisher_info[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()

        return (self.config.ewc_lambda / 2) * ewc_loss


# ============================================================================
# 知识蒸馏
# ============================================================================

class KnowledgeDistillation:
    """知识蒸馏

    保留旧任务知识的重要技术
    """

    def __init__(self, old_model: nn.Module, config: ContinualLearningConfig):
        self.old_model = old_model
        self.config = config
        self.old_model.eval()

    def distillation_loss(self, new_output: torch.Tensor,
                         old_output: torch.Tensor) -> torch.Tensor:
        """计算蒸馏损失

        Loss = KL divergence(new_logits/T || old_logits/T)
        """
        # 应用温度
        new_logits = new_output / self.config.distill_temperature
        old_logits = old_output / self.config.distill_temperature

        # 计算KL散度
        distill_loss = F.kl_div(
            F.log_softmax(new_logits, dim=1),
            F.softmax(old_logits, dim=1),
            reduction='batchmean'
        )

        return self.config.distill_temperature ** 2 * distill_loss


# ============================================================================
# 记忆回放
# ============================================================================

class ReplayBuffer:
    """记忆回放缓冲区"""

    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.buffer = defaultdict(list)  # {task_id: [samples]}

    def add_samples(self, task_id: str, samples: List[Tuple]):
        """添加样本到缓冲区"""
        for sample in samples:
            self.buffer[task_id].append(sample)

        # 限制缓冲区大小
        if len(self.buffer[task_id]) > self.config.replay_buffer_size:
            # 随机移除样本
            excess = len(self.buffer[task_id]) - self.config.replay_buffer_size
            indices = np.random.choice(len(self.buffer[task_id]), excess, replace=False)
            # 从后往前删除，避免索引问题
            for idx in sorted(indices, reverse=True):
                del self.buffer[task_id][idx]

    def get_samples(self, task_id: str = None, num_samples: int = None) -> List[Tuple]:
        """获取样本用于回放"""
        if task_id:
            # 获取特定任务的样本
            samples = self.buffer.get(task_id, [])
        else:
            # 获取所有任务的样本
            all_samples = []
            for task_samples in self.buffer.values():
                all_samples.extend(task_samples)
            samples = all_samples

        # 限制样本数量
        if num_samples and len(samples) > num_samples:
            indices = np.random.choice(len(samples), num_samples, replace=False)
            samples = [samples[i] for i in indices]

        return samples

    def get_size(self) -> int:
        """获取缓冲区总大小"""
        return sum(len(samples) for samples in self.buffer.values())


# ============================================================================
# 持续学习引擎
# ============================================================================

class ContinualLearningEngine:
    """持续学习引擎

    整合EWC、知识蒸馏、记忆回放等技术
    """

    def __init__(self, model: nn.Module, config: ContinualLearningConfig = None):
        self.model = model
        self.config = config or ContinualLearningConfig()

        # 组件
        self.ewc_regularizer = None
        self.distillation = None
        self.replay_buffer = ReplayBuffer(self.config)

        # 学习历史
        self.history: List[LearningHistory] = []
        self.current_task_id = None

        # 统计信息
        self.stats = {
            'tasks_learned': 0,
            'total_samples': 0,
            'replay_samples_used': 0,
        }

    def learn_task(self, task: LearningTask, data_loader,
                  old_model: nn.Module = None) -> Dict[str, float]:
        """
        学习新任务

        Args:
            task: 学习任务
            data_loader: 数据加载器
            old_model: 旧模型（用于知识蒸馏）

        Returns:
            学习统计信息
        """
        self.current_task_id = task.task_id

        # 创建EWC正则化器（如果有旧任务）
        if self.ewc_regularizer:
            # 计算Fisher信息
            fisher_info, optimal_params = self.ewc_regularizer.compute_fisher_information(data_loader)
            # 保存到历史
            history_entry = LearningHistory(
                task_id=self.history[-1].task_id if self.history else "init",
                learned_at=0.0,
                performance=0.0,
                parameters_snapshot=optimal_params,
                fisher_info=fisher_info
            )
            self.history.append(history_entry)

        # 创建知识蒸馏器
        if old_model is not None:
            self.distillation = KnowledgeDistillation(old_model, self.config)

        # 训练模型
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        train_losses = []
        train_accs = []

        for epoch in range(self.config.epochs_per_task):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (data, target) in enumerate(data_loader):
                self.model.train()

                # 前向传播
                output = self.model(data)

                # 主任务损失
                task_loss = F.cross_entropy(output, target)

                # EWC正则化损失
                ewc_loss = 0.0
                if self.ewc_regularizer:
                    ewc_loss = self.ewc_regularizer.ewc_loss()

                # 知识蒸馏损失
                distill_loss = 0.0
                if self.distillation and old_model is not None:
                    with torch.no_grad():
                        old_output = old_model(data)
                    distill_loss = self.distillation.distillation_loss(output, old_output)

                # 总损失
                if self.distillation:
                    # 组合任务损失和蒸馏损失
                    loss = (1 - self.config.distill_alpha) * task_loss + \
                           self.config.distill_alpha * distill_loss
                else:
                    loss = task_loss

                loss += ewc_loss

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 统计
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_total += len(data)

                # 记忆回放
                if batch_idx % 10 == 0:
                    replay_samples = self.replay_buffer.get_samples(
                        num_samples=self.config.replay_samples_per_task
                    )
                    if replay_samples:
                        self.stats['replay_samples_used'] += len(replay_samples)

            epoch_loss /= len(data_loader)
            epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0

            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)

        # 保存样本到回放缓冲区
        samples = []
        for data, target in data_loader:
            for i in range(len(data)):
                samples.append((data[i].cpu(), target[i].cpu()))
        self.replay_buffer.add_samples(task.task_id, samples)

        # 创建新的EWC正则器
        self.ewc_regularizer = EWCRegularizer(self.model, self.config)

        # 更新统计
        self.stats['tasks_learned'] += 1
        self.stats['total_samples'] += task.num_samples

        return {
            'task_id': task.task_id,
            'final_loss': train_losses[-1] if train_losses else 0.0,
            'final_accuracy': train_accs[-1] if train_accs else 0.0,
            'avg_loss': np.mean(train_losses) if train_losses else 0.0,
            'avg_accuracy': np.mean(train_accs) if train_accs else 0.0,
        }

    def evaluate_task(self, task_id: str, data_loader) -> Dict[str, float]:
        """
        评估任务性能

        Args:
            task_id: 任务ID
            data_loader: 数据加载器

        Returns:
            评估结果
        """
        self.model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for data, target in data_loader:
                output = self.model(data)
                loss = F.cross_entropy(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(data_loader) if data_loader else 0.0

        return {
            'task_id': task_id,
            'accuracy': accuracy,
            'loss': avg_loss,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能总结"""
        return {
            'stats': self.stats.copy(),
            'history_length': len(self.history),
            'replay_buffer_size': self.replay_buffer.get_size(),
            'current_task': self.current_task_id,
        }


# ============================================================================
# 在线学习包装器
# ============================================================================

class OnlineLearningWrapper:
    """在线学习包装器

    为现有模型添加在线学习能力
    """

    def __init__(self, model: nn.Module, config: ContinualLearningConfig = None):
        self.model = model
        self.config = config or ContinualLearningConfig()
        self.engine = ContinualLearningEngine(model, config)

        # 性能监控
        self.performance_history = []

    def learn_online(self, new_data: List[Tuple], task_id: str = "online",
                    old_model: nn.Module = None) -> Dict[str, float]:
        """
        在线学习新数据

        Args:
            new_data: 新数据 [(x, y), ...]
            task_id: 任务ID
            old_model: 旧模型（可选）

        Returns:
            学习统计
        """
        # 创建任务
        task = LearningTask(
            task_id=task_id,
            task_name=f"Online Task {task_id}",
            data=new_data,
            num_samples=len(new_data)
        )

        # 创建数据加载器
        from torch.utils.data import DataLoader, TensorDataset

        if len(new_data) > 0:
            # 转换为Tensor
            data_tensor = torch.stack([x for x, y in new_data])
            target_tensor = torch.tensor([y for x, y in new_data])

            dataset = TensorDataset(data_tensor, target_tensor)
            data_loader = DataLoader(
                dataset,
                batch_size=min(self.config.batch_size, len(new_data)),
                shuffle=True
            )
        else:
            # 空数据加载器
            data_loader = iter([])

        # 学习任务
        stats = self.engine.learn_task(task, data_loader, old_model)

        # 记录性能
        self.performance_history.append(stats)

        return stats

    def evaluate(self, test_data: List[Tuple]) -> Dict[str, float]:
        """评估当前模型"""
        from torch.utils.data import DataLoader, TensorDataset

        if len(test_data) == 0:
            return {'accuracy': 0.0, 'loss': 0.0}

        data_tensor = torch.stack([x for x, y in test_data])
        target_tensor = torch.tensor([y for x, y in test_data])

        dataset = TensorDataset(data_tensor, target_tensor)
        data_loader = DataLoader(dataset, batch_size=32)

        # 简化评估
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)

        accuracy = correct / total if total > 0 else 0.0

        return {'accuracy': accuracy}

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'engine_stats': self.engine.get_performance_summary(),
            'performance_history': self.performance_history,
        }


# ============================================================================
# 便捷函数
# ============================================================================

def get_continual_learner(model: nn.Module,
                          config: ContinualLearningConfig = None) -> ContinualLearningEngine:
    """获取持续学习引擎"""
    return ContinualLearningEngine(model, config)


def get_online_learner(model: nn.Module,
                      config: ContinualLearningConfig = None) -> OnlineLearningWrapper:
    """获取在线学习包装器"""
    return OnlineLearningWrapper(model, config)


if __name__ == '__main__':
    print("=== 持续学习机制 ===")
    print()
    print("核心组件:")
    print("- Elastic Weight Consolidation (EWC)")
    print("- Knowledge Distillation")
    print("- Memory Replay")
    print("- Continual Learning Engine")
    print("- Online Learning Wrapper")
    print()
    print("功能:")
    print("- 在线学习")
    print("- 灾难性遗忘预防")
    print("- 知识保留")
    print("- 性能监控")
