"""自修改架构

从错误中学习，自我改进。

核心能力：
1. 错误检测 — 检测预测错误
2. 原因分析 — 分析错误原因
3. 策略调整 — 调整学习策略
4. 结构修改 — 修改网络结构

运行方式：
    python training/layers/self_modification.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ErrorRecord:
    """错误记录"""
    input_data: torch.Tensor
    expected: torch.Tensor
    predicted: torch.Tensor
    error_type: str
    timestamp: float
    correction_applied: bool = False


class ErrorDetector(nn.Module):
    """错误检测器

    检测预测错误和知识冲突。
    """

    def __init__(self, input_dim: int = 128, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

        # 错误分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 错误类型预测
        self.type_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # 5种错误类型
        )

    def forward(self, predicted: torch.Tensor, expected: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """检测错误"""
        combined = torch.cat([predicted, expected], dim=1)

        # 错误概率
        error_prob = self.classifier(combined)

        # 错误类型
        error_type = self.type_predictor(combined)

        return error_prob, error_type

    def is_error(self, predicted: torch.Tensor, expected: torch.Tensor) -> bool:
        """判断是否有错误"""
        error_prob, _ = self.forward(predicted, expected)
        return error_prob.item() > self.threshold


class StrategyAdapter(nn.Module):
    """策略适配器

    根据错误调整学习策略。
    """

    def __init__(self, strategy_dim: int = 32, num_strategies: int = 10):
        super().__init__()

        # 策略编码
        self.strategy_encoder = nn.Embedding(num_strategies, strategy_dim)

        # 策略选择
        self.strategy_selector = nn.Sequential(
            nn.Linear(strategy_dim + 128, 64),  # strategy + error_features
            nn.ReLU(),
            nn.Linear(64, num_strategies),
            nn.Softmax(dim=1),
        )

        # 策略效果预测
        self.effect_predictor = nn.Sequential(
            nn.Linear(strategy_dim + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, error_features: torch.Tensor,
                current_strategy: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """选择策略"""
        # 编码当前策略
        strategy_vec = self.strategy_encoder(
            torch.tensor([current_strategy], device=error_features.device)
        )

        # 拼接
        combined = torch.cat([strategy_vec, error_features], dim=1)

        # 选择新策略
        strategy_probs = self.strategy_selector(combined)

        # 预测效果
        effect = self.effect_predictor(combined)

        return strategy_probs, effect


class StructureModifier(nn.Module):
    """结构修改器

    根据学习进度修改网络结构。
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        # 添加神经元
        self.add_neuron = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 删除神经元
        self.remove_neuron = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 修改权重
        self.modify_weight = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """决定结构修改"""
        add_prob = self.add_neuron(features)
        remove_prob = self.remove_neuron(features)
        weight_mod = self.modify_weight(features)

        return add_prob, remove_prob, weight_mod


class SelfModificationSystem:
    """自修改系统"""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 模型
        self.error_detector = ErrorDetector().to(self.device)
        self.strategy_adapter = StrategyAdapter().to(self.device)
        self.structure_modifier = StructureModifier().to(self.device)

        # 错误历史
        self.error_history: List[ErrorRecord] = []

        # 当前策略
        self.current_strategy = 0

        # 策略效果
        self.strategy_effects: Dict[int, List[float]] = {}

        # 统计
        self.stats = {
            'errors_detected': 0,
            'corrections_applied': 0,
            'strategy_changes': 0,
            'structure_modifications': 0,
        }

    def detect_error(self, predicted: torch.Tensor, expected: torch.Tensor,
                    error_type: str = "unknown") -> bool:
        """检测错误"""
        is_error = self.error_detector.is_error(predicted, expected)

        if is_error:
            record = ErrorRecord(
                input_data=predicted.clone(),
                expected=expected.clone(),
                predicted=predicted.clone(),
                error_type=error_type,
                timestamp=0.0,  # 简化
            )
            self.error_history.append(record)
            self.stats['errors_detected'] += 1

        return is_error

    def adapt_strategy(self, error_features: torch.Tensor) -> int:
        """适配策略"""
        strategy_probs, effect = self.strategy_adapter(
            error_features, self.current_strategy
        )

        # 选择新策略
        new_strategy = strategy_probs.argmax(dim=1).item()

        # 记录效果
        if self.current_strategy not in self.strategy_effects:
            self.strategy_effects[self.current_strategy] = []
        self.strategy_effects[self.current_strategy].append(effect.item())

        # 更新策略
        if new_strategy != self.current_strategy:
            self.current_strategy = new_strategy
            self.stats['strategy_changes'] += 1

        return new_strategy

    def modify_structure(self, features: torch.Tensor) -> Dict:
        """修改结构"""
        add_prob, remove_prob, weight_mod = self.structure_modifier(features)

        modifications = {
            'add_neuron': add_prob.item() > 0.5,
            'remove_neuron': remove_prob.item() > 0.5,
            'weight_modification': weight_mod,
        }

        if modifications['add_neuron'] or modifications['remove_neuron']:
            self.stats['structure_modifications'] += 1

        return modifications

    def learn_from_error(self, predicted: torch.Tensor, expected: torch.Tensor):
        """从错误中学习"""
        # 检测错误
        if not self.detect_error(predicted, expected):
            return

        # 计算误差
        error = expected - predicted

        # 适配策略
        self.adapt_strategy(error)

        # 修改结构
        self.modify_structure(error)

        self.stats['corrections_applied'] += 1

    def get_error_rate(self) -> float:
        """获取错误率"""
        if not self.error_history:
            return 0.0
        return len(self.error_history) / max(1, self.stats['corrections_applied'])

    def get_best_strategy(self) -> int:
        """获取最佳策略"""
        if not self.strategy_effects:
            return 0

        best_strategy = 0
        best_effect = 0.0

        for strategy, effects in self.strategy_effects.items():
            avg_effect = np.mean(effects) if effects else 0.0
            if avg_effect > best_effect:
                best_effect = avg_effect
                best_strategy = strategy

        return best_strategy

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'device': str(self.device),
            'error_history_size': len(self.error_history),
            'current_strategy': self.current_strategy,
            'best_strategy': self.get_best_strategy(),
        }


def test_self_modification():
    """测试自修改系统"""
    print("=" * 70)
    print("自修改系统测试")
    print("=" * 70)

    system = SelfModificationSystem()

    # 测试错误检测
    print("\n1. 错误检测测试:")
    test_cases = [
        (torch.randn(1, 128), torch.randn(1, 128), "随机错误"),
        (torch.ones(1, 128), torch.ones(1, 128), "正确"),
        (torch.randn(1, 128), torch.randn(1, 128), "预测错误"),
    ]

    for pred, expected, desc in test_cases:
        is_error = system.detect_error(pred.to(system.device), expected.to(system.device), desc)
        print(f"  {desc}: 错误={is_error}")

    # 测试策略适配
    print("\n2. 策略适配测试:")
    error_features = torch.randn(1, 128).to(system.device)

    for i in range(3):
        new_strategy = system.adapt_strategy(error_features)
        print(f"  轮次 {i+1}: 策略={new_strategy}")

    # 测试结构修改
    print("\n3. 结构修改测试:")
    features = torch.randn(1, 128).to(system.device)
    modifications = system.modify_structure(features)

    print(f"  添加神经元: {modifications['add_neuron']}")
    print(f"  删除神经元: {modifications['remove_neuron']}")
    print(f"  权重修改: {modifications['weight_modification'].shape}")

    # 测试从错误学习
    print("\n4. 从错误学习测试:")
    for i in range(5):
        predicted = torch.randn(1, 128).to(system.device)
        expected = torch.randn(1, 128).to(system.device)
        system.learn_from_error(predicted, expected)

    # 统计
    print("\n统计:")
    stats = system.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    test_self_modification()
