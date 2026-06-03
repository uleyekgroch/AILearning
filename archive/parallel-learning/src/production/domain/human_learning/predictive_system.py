"""
预测学习系统 - 从第一性原理出发

人类学习的核心是感知-预测循环：
1. 感知输入
2. 预测下一刻
3. 计算预测误差
4. 根据误差更新预测模型

这是学习的根本动力
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """预测结果"""
    predicted_value: np.ndarray
    confidence: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionError:
    """预测误差"""
    error_magnitude: float
    error_direction: np.ndarray
    prediction: Prediction
    actual: np.ndarray
    learning_signal: float


class PredictiveModel:
    """预测模型

    基于人类大脑的预测机制：
    - 不断预测下一刻会发生什么
    - 预测误差驱动学习
    - 这是学习的根本动力
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        初始化预测模型

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 预测权重（简化为线性模型）
        # 输入 -> 隐藏 -> 输出（输出维度 = 输入维度）
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.001
        self.weights_hidden_output = np.random.randn(hidden_dim, input_dim) * 0.001
        self.bias_hidden = np.zeros(hidden_dim)
        self.bias_output = np.zeros(input_dim)

        # 预测历史
        self.predictions: List[Prediction] = []
        self.errors: List[PredictionError] = []

        # 学习率
        self.learning_rate = 0.0001

        # 统计
        self.stats = {
            'total_predictions': 0,
            'total_errors': 0,
            'avg_error': 0.0,
            'learning_rate': self.learning_rate,
        }

    def predict(self, input_data: np.ndarray, context: Dict[str, Any] = None) -> Prediction:
        """
        预测下一刻

        Args:
            input_data: 输入数据
            context: 上下文信息

        Returns:
            预测结果
        """
        # 前向传播：输入 -> 隐藏 -> 输出
        hidden = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        hidden = np.maximum(0, hidden)  # ReLU激活
        predicted_value = np.dot(hidden, self.weights_hidden_output) + self.bias_output

        # 计算置信度（基于历史误差）
        confidence = self._compute_confidence()

        # 创建预测结果
        prediction = Prediction(
            predicted_value=predicted_value,
            confidence=confidence,
            timestamp=datetime.now(),
            context=context or {}
        )

        # 保存预测
        self.predictions.append(prediction)
        self.stats['total_predictions'] += 1

        return prediction

    def compute_error(self, prediction: Prediction, actual: np.ndarray) -> PredictionError:
        """
        计算预测误差

        Args:
            prediction: 预测结果
            actual: 实际值

        Returns:
            预测误差
        """
        # 计算误差
        error = actual - prediction.predicted_value
        error_magnitude = np.linalg.norm(error)
        error_direction = error / (error_magnitude + 1e-8)

        # 计算学习信号
        learning_signal = error_magnitude * prediction.confidence

        # 创建误差对象
        prediction_error = PredictionError(
            error_magnitude=error_magnitude,
            error_direction=error_direction,
            prediction=prediction,
            actual=actual,
            learning_signal=learning_signal
        )

        # 保存误差
        self.errors.append(prediction_error)
        self.stats['total_errors'] += 1

        # 更新平均误差
        self.stats['avg_error'] = (
            self.stats['avg_error'] * (self.stats['total_errors'] - 1) + error_magnitude
        ) / self.stats['total_errors']

        return prediction_error

    def learn_from_error(self, error: PredictionError, input_data: np.ndarray) -> None:
        """
        从误差中学习

        Args:
            error: 预测误差
            input_data: 输入数据
        """
        # 前向传播（保存中间结果）
        hidden = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        hidden = np.maximum(0, hidden)

        # 反向传播
        # 输出层梯度
        output_gradient = error.error_direction

        # 梯度裁剪
        max_grad_norm = 1.0
        output_gradient_norm = np.linalg.norm(output_gradient)
        if output_gradient_norm > max_grad_norm:
            output_gradient = output_gradient * (max_grad_norm / output_gradient_norm)

        self.weights_hidden_output -= self.learning_rate * np.outer(hidden, output_gradient)
        self.bias_output -= self.learning_rate * output_gradient

        # 隐藏层梯度
        hidden_gradient = np.dot(output_gradient, self.weights_hidden_output.T)
        hidden_gradient[hidden <= 0] = 0  # ReLU导数

        # 梯度裁剪
        hidden_gradient_norm = np.linalg.norm(hidden_gradient)
        if hidden_gradient_norm > max_grad_norm:
            hidden_gradient = hidden_gradient * (max_grad_norm / hidden_gradient_norm)

        self.weights_input_hidden -= self.learning_rate * np.outer(input_data, hidden_gradient)
        self.bias_hidden -= self.learning_rate * hidden_gradient

        # 更新学习率（基于误差大小）
        self._update_learning_rate(error.error_magnitude)

        logger.debug(f"Learned from error: magnitude={error.error_magnitude:.4f}")

    def _compute_confidence(self) -> float:
        """
        计算置信度

        Returns:
            置信度 [0, 1]
        """
        if len(self.errors) == 0:
            return 0.5

        # 基于最近误差计算置信度
        recent_errors = [e.error_magnitude for e in self.errors[-10:]]
        avg_error = np.mean(recent_errors)

        # 转换为置信度（误差越小，置信度越高）
        confidence = 1.0 / (1.0 + avg_error)

        return confidence

    def _update_learning_rate(self, error_magnitude: float) -> None:
        """
        更新学习率

        Args:
            error_magnitude: 误差大小
        """
        # 自适应学习率
        if error_magnitude > 1.0:
            self.learning_rate = min(0.1, self.learning_rate * 1.1)
        else:
            self.learning_rate = max(0.001, self.learning_rate * 0.99)

        self.stats['learning_rate'] = self.learning_rate

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


class PredictiveLearningSystem:
    """预测学习系统

    实现人类学习的核心机制：
    1. 感知输入
    2. 预测下一刻
    3. 计算预测误差
    4. 根据误差更新预测模型
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        初始化预测学习系统

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 预测模型
        self.model = PredictiveModel(input_dim, hidden_dim)

        # 学习历史
        self.learning_history: List[Dict[str, Any]] = []

        # 统计
        self.stats = {
            'total_learning_cycles': 0,
            'avg_prediction_error': 0.0,
            'learning_progress': 0.0,
        }

    def perceive_and_learn(self, input_data: np.ndarray,
                          actual_next: np.ndarray,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        感知并学习

        Args:
            input_data: 当前输入
            actual_next: 实际下一刻
            context: 上下文信息

        Returns:
            学习结果
        """
        # 1. 预测下一刻
        prediction = self.model.predict(input_data, context)

        # 2. 计算预测误差
        error = self.model.compute_error(prediction, actual_next)

        # 3. 从误差中学习
        self.model.learn_from_error(error, input_data)

        # 4. 记录学习历史
        learning_record = {
            'timestamp': datetime.now().isoformat(),
            'prediction_error': error.error_magnitude,
            'confidence': prediction.confidence,
            'learning_signal': error.learning_signal,
        }
        self.learning_history.append(learning_record)

        # 5. 更新统计
        self.stats['total_learning_cycles'] += 1
        self.stats['avg_prediction_error'] = self.model.stats['avg_error']
        self.stats['learning_progress'] = self._compute_learning_progress()

        return {
            'prediction': prediction.predicted_value,
            'error': error.error_magnitude,
            'confidence': prediction.confidence,
            'learning_signal': error.learning_signal,
        }

    def predict(self, input_data: np.ndarray,
               context: Dict[str, Any] = None) -> np.ndarray:
        """
        预测下一刻

        Args:
            input_data: 输入数据
            context: 上下文信息

        Returns:
            预测值
        """
        prediction = self.model.predict(input_data, context)
        return prediction.predicted_value

    def _compute_learning_progress(self) -> float:
        """
        计算学习进度

        Returns:
            学习进度 [0, 1]
        """
        if len(self.learning_history) < 10:
            return 0.0

        # 比较最近误差和早期误差
        recent_errors = [r['prediction_error'] for r in self.learning_history[-10:]]
        early_errors = [r['prediction_error'] for r in self.learning_history[:10]]

        avg_recent = np.mean(recent_errors)
        avg_early = np.mean(early_errors)

        # 计算改进比例
        if avg_early > 0:
            progress = 1.0 - (avg_recent / avg_early)
            return max(0.0, min(1.0, progress))

        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'model_stats': self.model.get_stats(),
        }
