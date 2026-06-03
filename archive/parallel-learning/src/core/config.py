"""
统一学习系统 — 配置

所有可调参数集中定义，避免散落在代码中。
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class LearnerConfig:
    """学习体配置"""
    # 维度
    obs_dim: int = 128
    action_dim: int = 8
    hidden_dims: Tuple[int, ...] = (256, 128)

    # 设备
    device: str = 'auto'  # 'auto' | 'cuda' | 'cpu'

    # 学习参数
    learning_rate: float = 0.001
    inference_lr: float = 0.05
    max_inference_steps: int = 50
    convergence_threshold: float = 1e-4

    # 好奇心
    curiosity_alpha: float = 0.5  # 预测误差权重
    curiosity_beta: float = 0.5   # 可学习性权重
    curiosity_decay: float = 0.99

    # 发展
    initial_stage: str = 'sensorimotor'
    initial_language_stage: str = 'holophrase'  # 文本学习从单词期开始（跳过前语言期）

    # 记忆
    working_memory_capacity: int = 7
    episodic_memory_capacity: int = 1000
    forgetting_rate: float = 0.95
    consolidation_interval: int = 100

    # 感知
    visual_channels: int = 4
    visual_size: Tuple[int, int] = (8, 8)
    audio_dim: int = 13
    position_dim: int = 2

    # 语言
    vocabulary_threshold: float = 0.3  # 词汇巩固阈值
    composition_threshold: int = 3     # 组合性触发阈值

    # 3D 环境
    bounds_3d: Tuple[float, ...] = (10.0, 10.0, 5.0)

    # 主动推理 (FEP)
    use_active_inference: bool = False
    fep_risk_penalty: float = 0.0

    # 关键期可塑性
    plasticity_schedule: str = 'none'  # exponential|sigmoid|linear|step|none
    plasticity_floor: float = 0.1

    # 睡眠巩固
    consolidation_strategy: str = 'random'  # random|success|recent|surprising

    # 内在动机
    motivation_epsilon: float = 0.1

    # 统计学习（Phase 1 核心转变）
    statistical_learning_enabled: bool = True
    statistical_min_freq: int = 3        # 概念涌现最低频率
    statistical_min_pmi: float = 1.0     # 关系涌现最低PMI
    statistical_max_ngram: int = 4       # 最大n-gram长度
    statistical_max_concepts: int = 5000 # 最大概念候选数
    statistical_use_as_primary: bool = True  # True=统计学习为主，正则为辅

    # 可学习编码器（Phase 6C — 容量可配置）
    encoder_n_heads: int = 4        # 多头注意力头数（需整除 obs_dim）
    encoder_n_layers: int = 3       # Transformer 层数（2→3，提升抽象能力）
    encoder_max_len: int = 128      # 最大序列长度

    # 对比学习（Phase 4 — 解决编码器"万能相似"）
    contrastive_enabled: bool = True
    contrastive_temperature: float = 0.2     # InfoNCE 温度参数（0.07太严格，0.2更适合小数据）
    contrastive_memory_bank_size: int = 512  # 内存银行容量（存储近期概念嵌入）
    contrastive_negatives_per_pos: int = 7   # 每个正样本对应的负样本数
    contrastive_update_freq: int = 1         # 每N条文本触发一次对比学习
    contrastive_warmup_texts: int = 5        # 至少N条文本后才开始对比学习
    contrastive_learning_rate: float = 5e-4  # 对比学习专用学习率


@dataclass
class TrainerConfig:
    """训练编排器配置"""
    # 训练阶段
    target_stage: str = 'literacy'
    max_steps_per_stage: int = 10000

    # 阶段对应交互数（模拟每月交互次数）
    interactions_per_month: dict = field(default_factory=lambda: {
        'sensorimotor': 50,
        'single_word': 100,
        'two_word': 150,
        'complex': 200,
        'literacy': 300,
    })

    # 评估
    evaluation_interval: int = 200
    advancement_threshold: float = 0.7

    # 输出
    results_dir: str = 'results/'
    checkpoint_interval: int = 500

    # 日志
    log_interval: int = 100
    verbose: bool = True


@dataclass
class StageDefinition:
    """发展阶段定义"""
    name: str
    age_range_months: Tuple[int, int]
    description: str
    promotion_criteria: dict
    abilities: list
    limitations: list
    scene_complexity: dict = field(default_factory=dict)
