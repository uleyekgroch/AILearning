"""
可学习感官编码器：从原始像素/声音到内部表征

核心思想：
- 视觉分支：卷积层提取空间特征
- 音频分支：MLP 提取声音特征
- 位置分支：线性层编码坐标
- 三路融合为统一的 40 维表征

这是从"手工特征"到"学习感知"的关键一步。
"""

import numpy as np
from typing import Dict, Tuple


class SensoryEncoder:
    """
    多模态感官编码器

    架构：
    - 视觉：8x8x4 → Conv2D(3x3,stride=2,8ch) → ReLU → Flatten → Linear(72→16)
    - 音频：7维 → Linear(7→16) → ReLU
    - 位置：2维 → Linear(2→8)
    - 融合：Concat = 40维

    参数量：~1600
    """

    def __init__(self, visual_shape=(8, 8, 4), audio_dim=7, pos_dim=2,
                 visual_out=16, audio_out=16, pos_out=8):
        self.visual_shape = visual_shape
        self.audio_dim = audio_dim
        self.pos_dim = pos_dim
        self.visual_out = visual_out
        self.audio_out = audio_out
        self.pos_out = pos_out

        self.output_dim = visual_out + audio_out + pos_out  # 40

        # Conv2D: 3x3 kernel, stride=2, 8 output channels
        # Input: (8,8,4) → Output: (3,3,8)
        self.conv_kernel_size = 3
        self.conv_stride = 2
        self.conv_in_channels = visual_shape[2]  # 4
        self.conv_out_channels = 8
        self.conv_out_h = (visual_shape[0] - self.conv_kernel_size) // self.conv_stride + 1  # 3
        self.conv_out_w = (visual_shape[1] - self.conv_kernel_size) // self.conv_stride + 1  # 3
        self.conv_flat_dim = self.conv_out_h * self.conv_out_w * self.conv_out_channels  # 72

        # Conv weights: (kernel_h, kernel_w, in_ch, out_ch)
        fan_in = self.conv_kernel_size * self.conv_kernel_size * self.conv_in_channels
        self.conv_W = np.random.randn(
            self.conv_kernel_size, self.conv_kernel_size,
            self.conv_in_channels, self.conv_out_channels
        ) * np.sqrt(2.0 / fan_in)
        self.conv_b = np.zeros(self.conv_out_channels)

        # Visual linear: 72 → 16
        self.vis_W = np.random.randn(self.conv_flat_dim, visual_out) * np.sqrt(2.0 / self.conv_flat_dim)
        self.vis_b = np.zeros(visual_out)

        # Audio linear: 7 → 16
        self.aud_W = np.random.randn(audio_dim, audio_out) * np.sqrt(2.0 / audio_dim)
        self.aud_b = np.zeros(audio_out)

        # Position linear: 2 → 8
        self.pos_W = np.random.randn(pos_dim, pos_out) * np.sqrt(2.0 / pos_dim)
        self.pos_b = np.zeros(pos_out)

        # Cache for backward
        self._cache = {}

        self.lr = 0.001

    def get_param_count(self) -> int:
        return (self.conv_W.size + self.conv_b.size +
                self.vis_W.size + self.vis_b.size +
                self.aud_W.size + self.aud_b.size +
                self.pos_W.size + self.pos_b.size)

    def _conv2d_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Conv2D forward pass.

        x: (H, W, C_in)
        returns: (out_h, out_w, C_out)
        """
        H, W, C_in = x.shape
        kh, kw = self.conv_kernel_size, self.conv_kernel_size
        oh, ow = self.conv_out_h, self.conv_out_w
        C_out = self.conv_out_channels

        output = np.zeros((oh, ow, C_out))
        for i in range(oh):
            for j in range(ow):
                patch = x[i*self.conv_stride:i*self.conv_stride+kh,
                          j*self.conv_stride:j*self.conv_stride+kw, :]
                for f in range(C_out):
                    output[i, j, f] = np.sum(patch * self.conv_W[:, :, :, f]) + self.conv_b[f]
        return output

    def _conv2d_backward(self, x: np.ndarray, d_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Conv2D backward pass.

        x: (H, W, C_in) - input
        d_out: (out_h, out_w, C_out) - gradient of loss w.r.t. output
        returns: (d_x, d_W, d_b)
        """
        H, W, C_in = x.shape
        kh, kw = self.conv_kernel_size, self.conv_kernel_size
        oh, ow = self.conv_out_h, self.conv_out_w
        C_out = self.conv_out_channels

        d_W = np.zeros_like(self.conv_W)
        d_b = np.zeros_like(self.conv_b)
        d_x = np.zeros_like(x)

        for i in range(oh):
            for j in range(ow):
                patch = x[i*self.conv_stride:i*self.conv_stride+kh,
                          j*self.conv_stride:j*self.conv_stride+kw, :]
                for f in range(C_out):
                    d_W[:, :, :, f] += patch * d_out[i, j, f]
                    d_b[f] += d_out[i, j, f]
                    d_x[i*self.conv_stride:i*self.conv_stride+kh,
                        j*self.conv_stride:j*self.conv_stride+kw, :] += \
                        self.conv_W[:, :, :, f] * d_out[i, j, f]

        return d_x, d_W, d_b

    def forward(self, visual: np.ndarray, audio: np.ndarray,
                position: np.ndarray) -> np.ndarray:
        """
        前向传播：原始感官输入 → 40维编码

        Args:
            visual: (8, 8, 4) 视觉输入
            audio: (7,) 音频事件向量
            position: (2,) 位置坐标

        Returns:
            (40,) 编码向量
        """
        # Visual branch: conv → relu → flatten → linear
        conv_out = self._conv2d_forward(visual)  # (3, 3, 8)
        conv_relu = np.maximum(0, conv_out)  # ReLU
        conv_flat = conv_relu.flatten()  # (72,)
        vis_feat = conv_flat @ self.vis_W + self.vis_b  # (16,)
        vis_feat = np.maximum(0, vis_feat)  # ReLU

        # Audio branch: linear → relu
        aud_feat = audio @ self.aud_W + self.aud_b  # (16,)
        aud_feat = np.maximum(0, aud_feat)  # ReLU

        # Position branch: linear (no activation)
        pos_feat = position @ self.pos_W + self.pos_b  # (8,)

        # Fusion
        encoded = np.concatenate([vis_feat, aud_feat, pos_feat])  # (40,)

        # Cache for backward
        self._cache = {
            'visual': visual,
            'audio': audio,
            'position': position,
            'conv_out': conv_out,
            'conv_relu': conv_relu,
            'conv_flat': conv_flat,
            'vis_feat_preactive': conv_flat @ self.vis_W + self.vis_b,
            'vis_feat': vis_feat,
            'aud_feat_preactive': audio @ self.aud_W + self.aud_b,
            'aud_feat': aud_feat,
            'pos_feat': pos_feat,
        }

        return encoded

    def backward(self, d_output: np.ndarray, learning_rate: float = None):
        """
        反向传播：更新编码器权重

        Args:
            d_output: (40,) 损失对编码输出的梯度
            learning_rate: 学习率（默认使用 self.lr）
        """
        if learning_rate is None:
            learning_rate = self.lr

        # Split gradient for each branch
        d_vis = d_output[:self.visual_out]
        d_aud = d_output[self.visual_out:self.visual_out + self.audio_out]
        d_pos = d_output[self.visual_out + self.audio_out:]

        # Position branch backward (linear, no activation)
        d_pos_W = np.outer(self._cache['position'], d_pos)
        d_pos_b = d_pos
        self.pos_W -= learning_rate * d_pos_W
        self.pos_b -= learning_rate * d_pos_b

        # Audio branch backward (linear + relu)
        d_aud_preactive = d_aud * (self._cache['aud_feat_preactive'] > 0).astype(float)
        d_aud_W = np.outer(self._cache['audio'], d_aud_preactive)
        d_aud_b = d_aud_preactive
        self.aud_W -= learning_rate * d_aud_W
        self.aud_b -= learning_rate * d_aud_b

        # Visual branch backward (linear + relu + flatten + conv_relu + conv)
        # Step 1: Linear backward
        d_vis_preactive = d_vis * (self._cache['vis_feat_preactive'] > 0).astype(float)
        d_vis_W = np.outer(self._cache['conv_flat'], d_vis_preactive)
        d_vis_b = d_vis_preactive

        # Step 2: Unflatten and ReLU backward (compute BEFORE weight update)
        d_conv_flat = d_vis_preactive @ self.vis_W.T  # (72,)
        d_conv_relu = d_conv_flat.reshape(self.conv_out_h, self.conv_out_w, self.conv_out_channels)
        d_conv_out = d_conv_relu * (self._cache['conv_out'] > 0).astype(float)

        # Step 3: Conv2D backward (compute BEFORE weight update)
        d_x, d_conv_W, d_conv_b = self._conv2d_backward(
            self._cache['visual'], d_conv_out
        )

        # Step 4: Update all weights (after gradients are computed)
        self.vis_W -= learning_rate * d_vis_W
        self.vis_b -= learning_rate * d_vis_b
        self.conv_W -= learning_rate * d_conv_W
        self.conv_b -= learning_rate * d_conv_b
