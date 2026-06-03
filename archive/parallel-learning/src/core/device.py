"""
CUDA 设备管理

自动检测 GPU，提供统一的设备管理。
零 numpy 依赖 — 所有转换通过 torch 原生操作。
"""

import torch

# 全局设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_device(device_str: str = 'auto') -> torch.device:
    """解析设备字符串为 torch.device"""
    if device_str == 'auto':
        return DEVICE
    return torch.device(device_str)


def to_device(tensor: torch.Tensor, device: torch.device = None) -> torch.Tensor:
    """将张量移到指定设备"""
    if device is None:
        device = DEVICE
    return tensor.to(device)
