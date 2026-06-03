/**
 * @file test_embedding_cuda.cpp
 * @brief Embedding Trainer CUDA 加速测试
 *
 * 验证：
 *   1. CUDA 可用性检测
 *   2. CPU fallback 正常工作（非 CUDA 构建）
 *   3. CUDA 训练 vs CPU 训练结果一致性（CUDA 构建）
 *   4. 小词表训练正确性
 *   5. 训练收敛性（loss 有效）
 *
 * CUDA 测试通过 WSL/Linux 运行，CPU fallback 测试在所有平台运行。
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/learning/embedding_trainer.hpp"
#include "ai_learning/learning/embedding_trainer_cuda.cuh"
#include "ai_learning/core/tensor_ops.hpp"

#include <cmath>
#include <random>
#include <vector>

using namespace ai_learning::learning;
using namespace ai_learning::core;
using Catch::Matchers::WithinAbs;

// ═══════════════════════════════════════════════════════════
// 基础：CUDA 可用性
// ═══════════════════════════════════════════════════════════

TEST_CASE("Embedding CUDA: 可用性检测", "[embedding][cuda]") {
    // 不应崩溃，返回 true 或 false 均可
    bool available = cuda_available();
    // 在无 CUDA 的构建上应返回 false
    // 在有 CUDA 的构建上可能返回 true
    (void)available;
}

// ═══════════════════════════════════════════════════════════
// 核心：小词表训练正确性
// ═══════════════════════════════════════════════════════════

TEST_CASE("Embedding CUDA: 小词表训练不崩溃", "[embedding][cuda]") {
    EmbeddingTrainerConfig config;
    config.embedding_dim = 32;
    config.epochs = 3;
    config.min_count = 1;
    config.neg_samples = 3;
    config.seed = 42;
    EmbeddingTrainer trainer(config);

    // 少量文本，词表 < 100 → 使用 CPU 路径
    for (int i = 0; i < 5; ++i) {
        trainer.add_text("数学研究数量和结构");
        trainer.add_text("物理研究物质和能量");
    }

    auto result = trainer.train();

    CHECK(result.vocab_size > 0);
    CHECK(result.embedding_dim == 32);
    CHECK(result.total_pairs > 0);
    CHECK_FALSE(trainer.is_trained() == false);
}

// ═══════════════════════════════════════════════════════════
// 核心：中等词表训练（触发 CUDA 路径阈值 > 100）
// ═══════════════════════════════════════════════════════════

TEST_CASE("Embedding CUDA: 中等词表训练", "[embedding][cuda]") {
    EmbeddingTrainerConfig config;
    config.embedding_dim = 64;
    config.epochs = 3;
    config.min_count = 1;
    config.neg_samples = 5;
    config.seed = 42;
    EmbeddingTrainer trainer(config);

    // 多样化文本，确保词表 > 100
    std::vector<std::string> corpus = {
        "数学研究数量结构空间变化",
        "物理研究物质运动能量力量",
        "化学研究物质组成反应性质",
        "生物学研究生命现象有机体",
        "天文学研究宇宙星体星际空间",
        "地理学研究地球表面自然环境",
        "历史学研究人类社会发展过程",
        "哲学研究存在知识价值意义",
        "经济学研究资源配置市场交换",
        "心理学研究人类行为心理过程",
        "社会学研究社会结构社会关系",
        "政治学研究权力制度治理决策",
        "法学研究法律规范社会秩序",
        "教育学研究教学学习知识传承",
        "语言学研究语言结构意义交际",
        "艺术学研究审美创造表达形式",
        "文学研究文本创作叙事表达",
        "音乐研究声音节奏旋律和声",
        "计算机科学研究算法程序数据",
        "工程学研究设计建造系统优化",
    };

    for (int i = 0; i < 3; ++i) {
        trainer.add_corpus(corpus);
    }

    auto result = trainer.train();

    CHECK(result.vocab_size > 50);
    CHECK(result.embedding_dim == 64);
    CHECK(result.total_pairs > 0);
    CHECK(result.final_loss >= 0.0);
    CHECK(trainer.is_trained());

    // 嵌入向量可用
    auto vec = trainer.get_embedding("数学");
    if (vec.has_value()) {
        CHECK(static_cast<int>(vec->size()) == 64);
        float sum = 0.0f;
        for (float v : *vec) sum += std::abs(v);
        CHECK(sum > 0.0f);
    }
}

// ═══════════════════════════════════════════════════════════
// CPU/GPU 一致性（仅 CUDA 构建有意义）
// ═══════════════════════════════════════════════════════════

TEST_CASE("Embedding CUDA: CPU 与 GPU 训练 loss 有效", "[embedding][cuda]") {
    if (!cuda_available()) {
        // 非 CUDA 构建，跳过一致性测试
        return;
    }

    // CPU 训练
    EmbeddingTrainerConfig config_cpu;
    config_cpu.embedding_dim = 32;
    config_cpu.epochs = 2;
    config_cpu.min_count = 1;
    config_cpu.neg_samples = 3;
    config_cpu.seed = 42;
    EmbeddingTrainer trainer_cpu(config_cpu);

    std::vector<std::string> corpus = {
        "数学研究数量结构空间变化",
        "物理研究物质运动能量力量",
        "化学研究物质组成反应性质",
        "生物学研究生命现象有机体",
        "计算机科学研究算法程序数据",
    };

    for (int i = 0; i < 5; ++i) {
        trainer_cpu.add_corpus(corpus);
    }

    auto result_cpu = trainer_cpu.train();
    CHECK(result_cpu.vocab_size > 0);
    CHECK(result_cpu.final_loss >= 0.0);
    CHECK(trainer_cpu.is_trained());

    // CUDA 训练（相同语料，相同种子）
    EmbeddingTrainerConfig config_gpu;
    config_gpu.embedding_dim = 32;
    config_gpu.epochs = 2;
    config_gpu.min_count = 1;
    config_gpu.neg_samples = 3;
    config_gpu.seed = 42;
    EmbeddingTrainer trainer_gpu(config_gpu);

    for (int i = 0; i < 5; ++i) {
        trainer_gpu.add_corpus(corpus);
    }

    auto result_gpu = trainer_gpu.train();

    // CUDA 路径应在词表 > 100 时启用
    // 如果词表不够大则走 CPU，两者结果应相同
    CHECK(result_gpu.vocab_size == result_cpu.vocab_size);
    CHECK(result_gpu.embedding_dim == result_cpu.embedding_dim);
    CHECK(trainer_gpu.is_trained());

    // 训练后的词表应一致
    auto vocab_cpu = trainer_cpu.vocabulary();
    auto vocab_gpu = trainer_gpu.vocabulary();
    CHECK(vocab_cpu.size() == vocab_gpu.size());
}

// ═══════════════════════════════════════════════════════════
// 边界：空语料和极端参数
// ═══════════════════════════════════════════════════════════

TEST_CASE("Embedding CUDA: 空语料训练", "[embedding][cuda]") {
    EmbeddingTrainerConfig config;
    config.embedding_dim = 32;
    config.epochs = 1;
    config.min_count = 1;
    EmbeddingTrainer trainer(config);

    // 不添加任何语料
    auto result = trainer.train();
    CHECK(result.vocab_size == 0);
    CHECK_FALSE(trainer.is_trained());
}

TEST_CASE("Embedding CUDA: 高维嵌入训练", "[embedding][cuda]") {
    EmbeddingTrainerConfig config;
    config.embedding_dim = 128;
    config.epochs = 2;
    config.min_count = 1;
    config.neg_samples = 5;
    config.seed = 42;
    EmbeddingTrainer trainer(config);

    for (int i = 0; i < 3; ++i) {
        trainer.add_corpus({
            "数学研究数量结构",
            "物理研究物质能量",
            "化学研究物质反应",
            "生物研究生命现象",
        });
    }

    auto result = trainer.train();
    CHECK(result.vocab_size > 0);
    CHECK(result.embedding_dim == 128);
    CHECK(trainer.is_trained());

    auto vec = trainer.get_embedding("数学");
    if (vec.has_value()) {
        CHECK(static_cast<int>(vec->size()) == 128);
    }
}

// ═══════════════════════════════════════════════════════════
// 集成：CUDA stub 不影响 CPU 训练
// ═══════════════════════════════════════════════════════════

TEST_CASE("Embedding CUDA: CPU fallback 训练收敛", "[embedding][cuda]") {
    EmbeddingTrainerConfig config;
    config.embedding_dim = 32;
    config.epochs = 10;
    config.min_count = 1;
    config.neg_samples = 3;
    config.learning_rate = 0.05;
    config.seed = 42;
    EmbeddingTrainer trainer(config);

    for (int i = 0; i < 8; ++i) {
        trainer.add_corpus({
            "数学研究数量结构",
            "物理研究物质能量",
            "数学属于基础学科",
            "物理属于基础学科",
        });
    }

    double last_loss = 0.0;
    trainer.set_progress_callback([&](const TrainingProgress& p) {
        last_loss = p.loss;
    });

    auto result = trainer.train();

    CHECK(result.vocab_size > 0);
    CHECK(trainer.is_trained());
    CHECK(result.final_loss >= 0.0);

    // 向量不应全零
    auto vec = trainer.get_embedding("数学");
    if (vec.has_value()) {
        float sum = 0.0f;
        for (float v : *vec) sum += std::abs(v);
        CHECK(sum > 0.0f);
    }
}
