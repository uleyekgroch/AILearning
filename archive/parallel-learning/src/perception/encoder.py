"""
多模态编码器 — 纯 torch.nn.Module 实现

架构：
  视觉分支：Conv2d(4, 16, 3) → ReLU → flatten → Linear(16*6*6, 16)
  听觉分支：Linear(7, 16) → ReLU
  位置分支：Linear(2, 8)
  融合：concat → 40d

移植自 mvl/encoder_sensory.py（263 行），核心改造：
  - numpy 权重 → torch.nn.Conv2d / Linear（自动求导）
  - 手写反向传播 → torch autograd
  - IPerception 接口实现
"""

import torch
import torch.nn as nn
from typing import Dict

from src.core.interfaces import IPerception
from src.core.config import LearnerConfig
from src.core.device import get_device


class MultiModalEncoder(nn.Module, IPerception):
    """
    多模态编码器：视觉 + 听觉 + 位置 → 40d 融合向量

    参数量：约 6200
    - 视觉 Conv2d: 4*16*3*3 + 16 = 592
    - 视觉 Linear: 576*16 + 16 = 9232
    - 听觉 Linear: 7*16 + 16 = 128
    - 位置 Linear: 2*8 + 8 = 24
    """

    def __init__(self, config: LearnerConfig):
        nn.Module.__init__(self)

        self.config = config
        self.device = get_device(config.device)

        # 维度常量
        self.visual_out_dim = 16
        self.auditory_out_dim = 16
        self.position_out_dim = 8
        self.output_dim = self.visual_out_dim + self.auditory_out_dim + self.position_out_dim  # 40

        # 视觉分支：Conv2d(4, 16, kernel_size=3) → ReLU → flatten → Linear(16*6*6, 16)
        # 输入 (4, 8, 8) → Conv → (16, 6, 6) → flatten (576) → Linear → (16)
        conv_out_h = config.visual_size[0] - 3 + 1  # 6
        conv_out_w = config.visual_size[1] - 3 + 1  # 6
        self.conv_flat_dim = 16 * conv_out_h * conv_out_w  # 576

        self.visual_conv = nn.Conv2d(
            in_channels=config.visual_channels,
            out_channels=16,
            kernel_size=3,
        )
        self.visual_fc = nn.Linear(self.conv_flat_dim, self.visual_out_dim)

        # 听觉分支：Linear(7, 16) → ReLU
        self.auditory_linear = nn.Linear(config.audio_dim, self.auditory_out_dim)

        # 位置分支：Linear(2, 8)
        self.position_linear = nn.Linear(config.position_dim, self.position_out_dim)

        # 投影层：40d → obs_dim，让感知输出与文本编码器维度对齐
        # 这是桥接感知路径(40d)和文本路径(128d)的关键
        raw_dim = self.visual_out_dim + self.auditory_out_dim + self.position_out_dim  # 40
        self.projection = nn.Sequential(
            nn.Linear(raw_dim, config.obs_dim),
            nn.ReLU(),
        )
        self.output_dim = config.obs_dim  # 投影后维度

        # 学习率（backward 方法用）
        self.lr = config.learning_rate

        # 缓存最近一次输入（供 backward 用）
        self._cached_raw_input = None

        self.to(self.device)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        """自动将张量移到编码器所在设备"""
        return t.to(self.device)

    def _encode_visual(self, visual: torch.Tensor) -> torch.Tensor:
        """视觉分支编码

        Args:
            visual: (4, 8, 8) 视觉特征图
        Returns:
            (16,) 视觉特征向量
        """
        x = visual.unsqueeze(0)  # (1, 4, 8, 8)
        x = self.visual_conv(x)  # (1, 16, 6, 6)
        x = torch.relu(x)
        x = x.flatten(1)  # (1, 576)
        x = self.visual_fc(x)  # (1, 16)
        x = torch.relu(x)
        return x.squeeze(0)  # (16,)

    def _encode_auditory(self, auditory: torch.Tensor) -> torch.Tensor:
        """听觉分支编码

        Args:
            auditory: (7,) 音频事件向量
        Returns:
            (16,) 听觉特征向量
        """
        x = auditory.unsqueeze(0)  # (1, 7)
        x = self.auditory_linear(x)  # (1, 16)
        x = torch.relu(x)
        return x.squeeze(0)  # (16,)

    def _encode_position(self, position: torch.Tensor) -> torch.Tensor:
        """位置分支编码（无激活函数）

        Args:
            position: (2,) 位置坐标
        Returns:
            (8,) 位置特征向量
        """
        x = position.unsqueeze(0)  # (1, 2)
        x = self.position_linear(x)  # (1, 8)
        return x.squeeze(0)  # (8,)

    # ------------------------------------------------------------------
    # IPerception 接口实现
    # ------------------------------------------------------------------

    def encode(self, raw_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将原始输入编码为 obs_dim 融合向量

        Args:
            raw_input: {'visual': (4,8,8), 'auditory': (7,), 'position': (2,)}

        Returns:
            (obs_dim,) 编码向量 — 通过投影层与文本编码器维度对齐
        """
        visual = self._to_device(raw_input['visual'])
        auditory = self._to_device(raw_input['auditory'])
        position = self._to_device(raw_input['position'])

        # 缓存输入供 backward 使用
        self._cached_raw_input = raw_input

        vis_feat = self._encode_visual(visual)      # (16,)
        aud_feat = self._encode_auditory(auditory)   # (16,)
        pos_feat = self._encode_position(position)    # (8,)

        raw_vec = torch.cat([vis_feat, aud_feat, pos_feat], dim=0)  # (40,)
        return self.projection(raw_vec)  # (obs_dim=128,)

    def forward(self, raw_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """nn.Module.forward — 委托给 encode"""
        return self.encode(raw_input)

    def get_modality_weights(self) -> Dict[str, float]:
        """返回各分支权重的 L2 范数归一化（注意力权重）

        用各分支参数的 L2 范数作为模态重要性度量。
        """
        vis_norm = self.visual_conv.weight.data.norm(2).item() + \
                   self.visual_fc.weight.data.norm(2).item()
        aud_norm = self.auditory_linear.weight.data.norm(2).item()
        pos_norm = self.position_linear.weight.data.norm(2).item()

        total = vis_norm + aud_norm + pos_norm
        if total < 1e-12:
            return {'visual': 1/3, 'auditory': 1/3, 'position': 1/3}

        return {
            'visual': vis_norm / total,
            'auditory': aud_norm / total,
            'position': pos_norm / total,
        }

    def backward(self, grad_output: torch.Tensor) -> None:
        """从学习引擎传回编码器梯度，执行一步参数更新

        利用缓存的原始输入重做前向（带梯度），再用 autograd 反向传播。
        供 PredictiveCodingEngine.learn_with_input_gradient 使用。

        Args:
            grad_output: (40,) 损失对编码输出的梯度
        """
        grad_output = self._to_device(grad_output)

        # 使用缓存的原始输入重做前向传播（带梯度），然后 autograd 反向传播
        if self._cached_raw_input is not None:
            # 清除之前的梯度
            self.zero_grad()

            # 带梯度的前向传播
            raw = self._cached_raw_input
            visual = self._to_device(raw['visual']).detach().requires_grad_(False)
            auditory = self._to_device(raw['auditory']).detach().requires_grad_(False)
            position = self._to_device(raw['position']).detach().requires_grad_(False)

            # 前向（参数有梯度）
            vis_feat = self._encode_visual(visual)
            aud_feat = self._encode_auditory(auditory)
            pos_feat = self._encode_position(position)
            output = torch.cat([vis_feat, aud_feat, pos_feat], dim=0)

            # autograd 反向传播
            output.backward(grad_output)

            # 用梯度做一步 SGD
            with torch.no_grad():
                for param in self.parameters():
                    if param.grad is not None:
                        param.add_(self.lr * param.grad)
