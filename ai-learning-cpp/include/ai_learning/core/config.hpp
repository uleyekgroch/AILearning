/**
 * @file config.hpp
 * @brief 学习体配置 — 所有可调参数集中定义
 *
 * DDD 值对象：不可变配置，构造后只读。
 * 对应 Python 版 LearnerConfig / TrainerConfig / StageDefinition。
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ai_learning::core {

/// 设备类型
enum class Device { kCpu, kCuda, kAuto };

/// 可塑性调度策略
enum class PlasticitySchedule {
    kExponential,
    kSigmoid,
    kLinear,
    kStep,
    kNone
};

/// 巩固策略
enum class ConsolidationStrategy {
    kRandom,
    kSuccess,
    kRecent,
    kSurprising
};

/// 学习体配置（值对象）
struct LearnerConfig {
    // ── 维度 ──
    int obs_dim                  = 128;
    int action_dim               = 8;
    std::vector<int> hidden_dims = {64, 32};

    // ── 设备 ──
    Device device = Device::kAuto;

    // ── 学习参数 ──
    double learning_rate        = 0.001;
    double inference_lr         = 0.05;
    int    max_inference_steps  = 50;
    double convergence_threshold = 1e-4;

    // ── 好奇心 ──
    double curiosity_alpha = 0.5;
    double curiosity_beta  = 0.5;
    double curiosity_decay = 0.99;

    // ── 发展阶段 ──
    std::string initial_stage         = "sensorimotor";
    std::string initial_language_stage = "holophrase";

    // ── 记忆 ──
    int    working_memory_capacity   = 7;
    int    episodic_memory_capacity  = 1000;
    double forgetting_rate           = 0.95;
    int    consolidation_interval    = 100;

    // ── 感知 ──
    int visual_channels = 4;
    int visual_width    = 8;
    int visual_height   = 8;
    int audio_dim       = 13;
    int position_dim    = 2;

    // ── 语言 ──
    double vocabulary_threshold  = 0.3;
    int    composition_threshold = 3;

    // ── 3D 环境 ──
    double bounds_x = 10.0;
    double bounds_y = 10.0;
    double bounds_z = 5.0;

    // ── 主动推理 ──
    bool   use_active_inference = false;
    double fep_risk_penalty     = 0.0;

    // ── 可塑性 ──
    PlasticitySchedule plasticity_schedule = PlasticitySchedule::kNone;
    double plasticity_floor                = 0.1;

    // ── 巩固 ──
    ConsolidationStrategy consolidation_strategy = ConsolidationStrategy::kRandom;

    // ── 内在动机 ──
    double motivation_epsilon = 0.1;

    // ── 统计学习 ──
    bool   statistical_learning_enabled = true;
    int    statistical_min_freq         = 3;
    double statistical_min_pmi          = 1.0;
    int    statistical_max_ngram        = 4;
    int    statistical_max_concepts     = 5000;
    bool   statistical_use_as_primary   = true;

    // ── 嵌入学习 ──
    bool   embedding_learning_enabled = false;   // 总开关，默认关闭（兼容性）
    int    embedding_dim              = 128;     // 必须与 obs_dim 一致
    int    embedding_train_interval   = 5;       // 每 N 次 consolidate 触发重训练
    int    embedding_min_count        = 2;       // 词汇频率阈值
    int    embedding_window_size      = 5;       // 上下文窗口
    int    embedding_epochs           = 3;       // SGNS 训练轮次
    int    embedding_neg_samples      = 5;       // 负采样数
    double embedding_learning_rate    = 0.025;   // SGNS 初始学习率
    int    ds_min_freq                = 3;       // DS 最低概念频率
};

/// 训练编排器配置（值对象）
struct TrainerConfig {
    std::string target_stage     = "literacy";
    int         max_steps_per_stage = 10000;

    int    evaluation_interval    = 200;
    double advancement_threshold  = 0.7;

    std::string results_dir       = "results/";
    int         checkpoint_interval = 500;

    int  log_interval = 100;
    bool verbose      = true;
};

/// 发展阶段定义（值对象）
struct StageDefinition {
    std::string name;
    int         age_min_months = 0;
    int         age_max_months = 0;
    std::string description;
    double      promotion_threshold = 0.7;
};

}  // namespace ai_learning::core
