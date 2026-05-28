"""
CUDA 工具层：设备管理、NumPy↔PyTorch 转换、批量操作封装

使用方式：
    from cuda_utils import DEVICE, to_tensor, to_numpy
"""

import numpy as np
import torch
import torch.nn as nn

# 设备管理
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_tensor(x: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """NumPy 数组转 PyTorch 张量，并移到指定设备"""
    if device is None:
        device = DEVICE
    return torch.from_numpy(x).float().to(device)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """PyTorch 张量转 NumPy 数组"""
    return x.detach().cpu().numpy()


def batch_predict(predictor, obs_batch: np.ndarray,
                  actions: np.ndarray) -> tuple:
    """
    批量预测：对多个 (obs, action) 对一次性计算

    Args:
        predictor: ProbabilisticPredictor 实例
        obs_batch: (batch, obs_dim) 观测批次
        actions: (batch,) 动作批次

    Returns:
        mean_batch: (batch, obs_dim) 均值预测
        logvar_batch: (batch, obs_dim) 对数方差预测
    """
    obs_t = to_tensor(obs_batch)
    action_batch = torch.zeros(len(actions), predictor.action_dim, device=DEVICE)
    action_batch[torch.arange(len(actions)), actions] = 1.0

    hidden = torch.tanh(obs_t @ to_tensor(predictor.W_obs) +
                        action_batch @ to_tensor(predictor.W_action))
    mean = hidden @ to_tensor(predictor.W_mean)
    log_var = hidden @ to_tensor(predictor.W_logvar) + to_tensor(predictor.b_logvar)

    return to_numpy(mean), to_numpy(log_var)


def batch_gef(obs_batch: np.ndarray, actions: np.ndarray,
              predictor, beliefs: np.ndarray,
              info_gain_weight: float = 1.0,
              pragmatic_weight: float = 1.0) -> np.ndarray:
    """
    批量计算期望自由能

    Args:
        obs_batch: (batch, obs_dim) 观测批次
        actions: (batch,) 动作批次
        predictor: ProbabilisticPredictor 实例
        beliefs: (batch, obs_dim) 信念批次（均值）
        info_gain_weight: 信息增益权重
        pragmatic_weight: 工具价值权重

    Returns:
        gef_batch: (batch,) 期望自由能批次
    """
    mean_batch, logvar_batch = batch_predict(predictor, obs_batch, actions)
    var_batch = np.exp(logvar_batch)

    # 信息增益：方差越大，信息增益越高
    info_gain = -0.5 * np.sum(np.log(var_batch + 1e-8), axis=1)

    # 工具价值：预测均值与信念的差异
    pragmatic_value = -0.5 * np.sum(
        (mean_batch - beliefs) ** 2 / (var_batch + 1e-8), axis=1
    )

    # 期望自由能
    gef = -info_gain_weight * info_gain - pragmatic_weight * pragmatic_value

    return gef


def vectorized_similarity(vocab_a: dict, vocab_b: dict) -> float:
    """
    向量化计算两个词汇表的相似度

    Args:
        vocab_a: Agent A 的词汇表
        vocab_b: Agent B 的词汇表

    Returns:
        similarity: 0-1 之间的相似度
    """
    if not vocab_a or not vocab_b:
        return 0.0

    keys_a = set(vocab_a.keys())
    keys_b = set(vocab_b.keys())

    intersection = keys_a & keys_b
    union = keys_a | keys_b

    if not union:
        return 0.0

    return len(intersection) / len(union)
