#!/usr/bin/env python3
"""时序预测学习 (Temporal Sequence Learning)

基于论文:
- Neuron 2024 (Chen, Zhang et al.): "Predictive Sequence Learning in the
  Hippocampal Formation"
  CA3生成对未来输入的预测，CA1计算时序预测误差。
  三突触回路(EC→DG→CA3→CA1)实现序列预测。
- NeurIPS 2023: "Sequential Memory with Temporal Predictive Coding"
  时序预测编码用于序列记忆。
- bioRxiv 2024: "Sequential predictive learning is a unifying theory for
  hippocampal function"

核心思想:
  海马体不仅存储记忆，还不断预测"下一步会发生什么"：
  1. 输入序列: A → B → C → D → ...
  2. CA3预测: 给定A，预测B应该是什么
  3. CA1比较: 实际B vs 预测B → 预测误差
  4. 误差驱动学习: 预测错误越大，学习信号越强

  这解释了为什么：
  - 意外事件（预测失败）记忆深刻
  - 可预测的事件容易被忽略（预测编码Light的底层机制）
  - 时序关系（"先下雨，后地湿"）是因果推理的基础

  对学习系统的意义:
  - 从文本中提取时序关系（A在B之前/之后）
  - 预测驱动的好奇心：预测失败 → 学习动机
  - 为因果推理提供时序基础（原因必须先于结果）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class SequenceStep:
    """序列中的一个步骤"""
    item: str                    # 内容标识
    embedding: torch.Tensor      # 嵌入
    predicted_next: Optional[torch.Tensor] = None  # 预测的下一步
    prediction_error: float = 0.0  # 预测误差
    timestamp: int = 0           # 时间戳


class PredictiveSequenceModel(nn.Module):
    """预测序列模型 — 模拟CA3-CA1回路

    CA3 (预测器): 给定当前状态，预测下一步
    CA1 (比较器): 比较预测与实际，计算误差
    DG (模式分离): 处理新的输入，正交化表征
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # CA3: 预测下一个状态
        self.ca3_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # DG: 模式分离（新输入正交化）
        self.dg_separator = nn.Linear(d_model, d_model, bias=False)
        nn.init.orthogonal_(self.dg_separator.weight)

        # CA1: 整合预测和实际（计算残差）
        self.ca1_integrator = nn.Linear(d_model * 2, d_model)

    def predict_next(self, current: torch.Tensor) -> torch.Tensor:
        """CA3预测下一步"""
        if current.dim() == 1:
            current = current.unsqueeze(0)
        return self.ca3_predictor(current).squeeze(0)

    def separate(self, embedding: torch.Tensor) -> torch.Tensor:
        """DG模式分离"""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        return self.dg_separator(embedding).squeeze(0)

    def compute_error(self, predicted: torch.Tensor, actual: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """CA1计算预测误差"""
        if predicted.dim() == 1:
            predicted = predicted.unsqueeze(0)
        if actual.dim() == 1:
            actual = actual.unsqueeze(0)

        # 残差 = 实际 - 预测
        residual = actual - predicted
        error_magnitude = F.mse_loss(predicted, actual).item()

        # CA1整合预测和实际
        integrated = self.ca1_integrator(torch.cat([predicted.squeeze(0), actual.squeeze(0)]))

        return integrated, error_magnitude


class TemporalSequenceSystem:
    """时序预测学习系统

    使用方法:
    1. observe(): 观察序列中的一个事件
    2. 系统自动预测下一步
    3. 预测失败驱动学习
    4. 提取时序关系（A在B之前/之后）
    """

    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.d_model = d_model
        self.device = torch.device(device)

        # 预测模型
        self.model = PredictiveSequenceModel(d_model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # 序列历史
        self.sequence: List[SequenceStep] = []
        self.max_sequence = 1000

        # 时序关系存储
        self.temporal_relations: Dict[str, List[str]] = {}  # item → [items that followed it]

        # 预测误差历史
        self.error_history: List[float] = []

        # 统计
        self.stats = {
            'observations': 0,
            'predictions_made': 0,
            'prediction_failures': 0,
            'temporal_relations_found': 0,
            'avg_prediction_error': 0.0,
        }

        self._time = 0

    def observe(self, item: str, embedding: torch.Tensor) -> Dict:
        """观察序列中的一个事件

        1. 对比预测与实际（如果之前有预测）
        2. 预测下一步
        3. 记录时序关系
        4. 训练模型

        Args:
            item: 事件标识（实体名/文本片段）
            embedding: 事件的嵌入向量

        Returns:
            观察结果（预测误差、是否意外等）
        """
        self._time += 1
        self.stats['observations'] += 1

        emb = embedding.to(self.device)
        step = SequenceStep(
            item=item,
            embedding=emb.detach(),
            timestamp=self._time,
        )

        result = {
            'item': item,
            'prediction_error': 0.0,
            'surprising': False,
            'predicted_correctly': True,
        }

        # 如果有序列历史，检查上一个预测
        if self.sequence:
            last_step = self.sequence[-1]
            if last_step.predicted_next is not None:
                # 计算预测误差
                with torch.no_grad():
                    integrated, error = self.model.compute_error(
                        last_step.predicted_next, emb
                    )

                step.prediction_error = error
                self.error_history.append(error)

                # 判断是否意外（误差>阈值）
                error_threshold = self._adaptive_threshold()
                result['prediction_error'] = error
                result['surprising'] = error > error_threshold
                result['predicted_correctly'] = error < error_threshold

                if result['surprising']:
                    self.stats['prediction_failures'] += 1

                # 记录时序关系：last_item → item
                self._record_temporal(last_step.item, item)

        # 预测下一步
        with torch.no_grad():
            predicted_next = self.model.predict_next(emb)
        step.predicted_next = predicted_next
        self.stats['predictions_made'] += 1

        # 训练模型（如果有上一步的真实结果）
        if self.sequence:
            self._train_step(self.sequence[-1].embedding, emb)

        # 存储到序列
        self.sequence.append(step)
        if len(self.sequence) > self.max_sequence:
            self.sequence = self.sequence[-self.max_sequence:]

        # 更新统计
        if self.error_history:
            recent = self.error_history[-100:]
            self.stats['avg_prediction_error'] = sum(recent) / len(recent)

        return result

    def _train_step(self, current_emb: torch.Tensor, next_emb: torch.Tensor):
        """训练预测模型

        给定current_emb，模型应该预测next_emb。
        """
        self.model.train()

        current = current_emb.detach().to(self.device)
        target = next_emb.detach().to(self.device)

        if current.dim() == 1:
            current = current.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        predicted = self.model.ca3_predictor(current)
        loss = F.mse_loss(predicted, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _record_temporal(self, before_item: str, after_item: str):
        """记录时序关系"""
        if before_item not in self.temporal_relations:
            self.temporal_relations[before_item] = []
        if after_item not in self.temporal_relations[before_item]:
            self.temporal_relations[before_item].append(after_item)
            self.stats['temporal_relations_found'] += 1

    def _adaptive_threshold(self) -> float:
        """自适应误差阈值

        基于历史误差的中位数确定什么是"意外"。
        """
        if len(self.error_history) < 5:
            return 0.5  # 默认阈值
        recent = self.error_history[-50:]
        sorted_errors = sorted(recent)
        median = sorted_errors[len(sorted_errors) // 2]
        return median * 1.5  # 超过中位数1.5倍算意外

    def get_temporal_successors(self, item: str) -> List[str]:
        """获取时序后继（在item之后经常出现的事件）"""
        return self.temporal_relations.get(item, [])

    def get_temporal_predecessors(self, item: str) -> List[str]:
        """获取时序前驱（在item之前经常出现的事件）"""
        predecessors = []
        for pred, successors in self.temporal_relations.items():
            if item in successors:
                predecessors.append(pred)
        return predecessors

    def predict_next_item(self, current_embedding: torch.Tensor) -> torch.Tensor:
        """预测下一个事件的嵌入"""
        with torch.no_grad():
            return self.model.predict_next(current_embedding.to(self.device))

    def get_surprise_score(self) -> float:
        """获取最近的意外程度（用于驱动好奇心）"""
        if not self.error_history:
            return 0.0
        recent = self.error_history[-10:]
        return sum(recent) / len(recent)

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'sequence_length': len(self.sequence),
            'temporal_items_tracked': len(self.temporal_relations),
        }
