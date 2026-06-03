"""
感知系统测试 — 多模态编码器 (TDD 先行)

6 个测试用例覆盖：
1. 视觉输入编码形状
2. 听觉输入编码形状
3. 位置输入编码形状
4. 三路融合输出 (40,)
5. 反向传播梯度可达 Conv2d 权重
6. auto 设备选择正确
"""

import sys
sys.path.insert(0, '.')

import torch
import pytest
from src.core.config import LearnerConfig
from src.perception.encoder import MultiModalEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return LearnerConfig()


@pytest.fixture
def encoder(config):
    return MultiModalEncoder(config)


def _make_raw_input(encoder, visual=None, auditory=None, position=None):
    """构造 raw_input dict，缺失的模态填零"""
    cfg = encoder.config
    if visual is None:
        visual = torch.randn(
            cfg.visual_channels,
            cfg.visual_size[0],
            cfg.visual_size[1],
        )
    if auditory is None:
        auditory = torch.randn(cfg.audio_dim)
    if position is None:
        position = torch.randn(cfg.position_dim)
    return {
        'visual': visual,
        'auditory': auditory,
        'position': position,
    }


# ---------------------------------------------------------------------------
# Test 1: 视觉输入编码形状
# ---------------------------------------------------------------------------

class TestVisualEncoding:
    """视觉分支：Conv2d(4, 16, 3) -> flatten -> Linear -> 16d"""

    def test_encode_visual_input_shape(self, encoder):
        """视觉输入 (4, 8, 8) 经编码后 visual 分支输出 16 维"""
        raw = _make_raw_input(encoder)
        output = encoder.encode(raw)
        assert output.shape == (40,), f"期望 (40,)，实际 {output.shape}"

        # 仅视觉输入时也能正确编码
        raw_vis_only = {
            'visual': torch.randn(4, 8, 8),
            'auditory': torch.zeros(encoder.config.audio_dim),
            'position': torch.zeros(encoder.config.position_dim),
        }
        output_vis = encoder.encode(raw_vis_only)
        assert output_vis.shape == (40,)

    def test_visual_branch_internal_shape(self, encoder):
        """验证视觉分支内部 Conv2d 输出维度正确"""
        x = torch.randn(1, 4, 8, 8, device=encoder.device)
        conv_out = encoder.visual_conv(x)
        # Conv2d(4, 16, 3) with no padding: (1, 16, 6, 6)
        assert conv_out.shape == (1, 16, 6, 6), f"Conv 输出应为 (1,16,6,6)，实际 {conv_out.shape}"


# ---------------------------------------------------------------------------
# Test 2: 听觉输入编码形状
# ---------------------------------------------------------------------------

class TestAuditoryEncoding:
    """听觉分支：Linear(audio_dim, 16)"""

    def test_encode_auditory_input_shape(self, encoder):
        """听觉输入编码后 auditory 分支输出 16 维"""
        raw = _make_raw_input(encoder)
        output = encoder.encode(raw)
        assert output.shape == (40,)

    def test_auditory_branch_internal_shape(self, encoder):
        """验证听觉分支内部维度"""
        audio_dim = encoder.config.audio_dim
        x = torch.randn(1, audio_dim, device=encoder.device)
        aud_out = encoder.auditory_linear(x)
        assert aud_out.shape == (1, 16), f"听觉分支输出应为 (1,16)，实际 {aud_out.shape}"


# ---------------------------------------------------------------------------
# Test 3: 位置输入编码形状
# ---------------------------------------------------------------------------

class TestPositionEncoding:
    """位置分支：Linear(2, 8)"""

    def test_encode_position_input_shape(self, encoder):
        """位置输入 (2,) 编码后 position 分支输出 8 维"""
        raw = _make_raw_input(encoder, position=torch.randn(2))
        output = encoder.encode(raw)
        assert output.shape == (40,)

    def test_position_branch_internal_shape(self, encoder):
        """验证位置分支内部维度"""
        x = torch.randn(1, 2, device=encoder.device)
        pos_out = encoder.position_linear(x)
        assert pos_out.shape == (1, 8), f"位置分支输出应为 (1,8)，实际 {pos_out.shape}"


# ---------------------------------------------------------------------------
# Test 4: 三路融合 → 40d
# ---------------------------------------------------------------------------

class TestMultimodalFusion:
    """三路输入拼接为 40 维"""

    def test_encode_multimodal_fusion(self, encoder):
        """三路完整输入 → 拼接输出 (40,)"""
        raw = _make_raw_input(encoder)
        output = encoder.encode(raw)
        assert output.shape == (40,), f"融合输出应为 (40,)，实际 {output.shape}"

        # 验证不是全零
        assert output.abs().sum() > 0, "融合输出不应全为零"

    def test_fusion_dimensions_add_up(self, encoder):
        """16 (visual) + 16 (auditory) + 8 (position) = 40"""
        assert encoder.visual_out_dim == 16
        assert encoder.auditory_out_dim == 16
        assert encoder.position_out_dim == 8
        assert encoder.output_dim == 40

    def test_forward_matches_encode(self, encoder):
        """forward() 和 encode() 应返回相同结果"""
        raw = _make_raw_input(encoder)
        out_encode = encoder.encode(raw)
        out_forward = encoder.forward(raw)
        assert torch.allclose(out_encode, out_forward), \
            "forward() 和 encode() 输出应一致"


# ---------------------------------------------------------------------------
# Test 5: 梯度反向传播
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """反向传播梯度可达 Conv2d 权重"""

    def test_gradient_flows_through_encoder(self, encoder):
        """端到端：loss → grad → Conv2d 权重有梯度"""
        raw = _make_raw_input(encoder)
        output = encoder.encode(raw)
        # 模拟一个简单的 loss
        target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # 检查 Conv2d 权重梯度非 None 且非全零
        conv_grad = encoder.visual_conv.weight.grad
        assert conv_grad is not None, "Conv2d 权重梯度不应为 None"
        assert conv_grad.abs().sum() > 0, "Conv2d 权重梯度不应全为零"

        # 检查其他分支也有梯度
        assert encoder.auditory_linear.weight.grad is not None
        assert encoder.position_linear.weight.grad is not None

    def test_backward_method(self, encoder):
        """backward(grad) 方法正确传入梯度并更新权重"""
        raw = _make_raw_input(encoder)
        output = encoder.encode(raw)

        # 记录更新前权重
        old_conv_weight = encoder.visual_conv.weight.data.clone()

        # 构造梯度并调用 backward
        grad = torch.randn_like(output)
        encoder.backward(grad)

        # 权重应该已变化
        new_conv_weight = encoder.visual_conv.weight.data
        assert not torch.allclose(old_conv_weight, new_conv_weight), \
            "backward 后 Conv2d 权重应已更新"


# ---------------------------------------------------------------------------
# Test 6: 设备放置
# ---------------------------------------------------------------------------

class TestDevicePlacement:
    """auto 设备选择"""

    def test_device_placement_auto(self, config):
        """config.device='auto' 时自动选择可用设备"""
        enc = MultiModalEncoder(config)
        expected = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert enc.device == expected, \
            f"设备应为 {expected}，实际 {enc.device}"

    def test_device_placement_cpu(self):
        """config.device='cpu' 时强制 CPU"""
        cfg = LearnerConfig(device='cpu')
        enc = MultiModalEncoder(cfg)
        assert enc.device == torch.device('cpu')

    def test_output_on_correct_device(self, encoder):
        """编码输出应在正确设备上"""
        raw = _make_raw_input(encoder)
        output = encoder.encode(raw)
        assert output.device.type == encoder.device.type, \
            f"输出设备类型应为 {encoder.device.type}，实际 {output.device.type}"


# ---------------------------------------------------------------------------
# Test: get_modality_weights
# ---------------------------------------------------------------------------

class TestModalityWeights:
    """模态权重：L2 范数归一化"""

    def test_modality_weights_keys(self, encoder):
        """权重 dict 包含三个模态键"""
        weights = encoder.get_modality_weights()
        assert set(weights.keys()) == {'visual', 'auditory', 'position'}

    def test_modality_weights_normalized(self, encoder):
        """权重归一化后总和为 1"""
        weights = encoder.get_modality_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-5, f"权重总和应为 1.0，实际 {total}"

    def test_modality_weights_positive(self, encoder):
        """权重均为正数"""
        weights = encoder.get_modality_weights()
        for name, w in weights.items():
            assert w > 0, f"{name} 权重应为正数，实际 {w}"
