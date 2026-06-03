/**
 * @file benchmark.cpp
 * @brief 性能基准测试 — 对比各模块操作耗时
 *
 * 测试模块：
 * 1. 文本学习（实体/关系提取 + 验证）
 * 2. 经验学习（预测编码 + STDP）
 * 3. 感知编码（多模态融合）
 * 4. 记忆系统（存储 + 检索 + 巩固）
 * 5. 推理引擎（统一推理）
 * 6. 自主学习循环（完整闭环）
 * 7. 社会交互（教学 + 学习）
 * 8. 持久化（保存 + 加载）
 */

#include "ai_learning/core/learner.hpp"
#include "ai_learning/domain/environment/simple_environment.hpp"
#include "ai_learning/domain/social/social_agent.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using bench_clock_t = std::chrono::high_resolution_clock;
using ms_t = std::chrono::duration<double, std::milli>;

/// 基准测试结果
struct BenchResult {
    std::string name;
    double total_ms;
    int iterations;
    double avg_ms() const { return total_ms / iterations; }
    double ops_per_sec() const { return 1000.0 * iterations / total_ms; }
};

/// 运行基准并打印结果
auto run_bench(const std::string& name, int iterations,
               const std::function<void()>& fn) -> BenchResult {
    // 预热
    fn();

    auto t0 = bench_clock_t::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    auto t1 = bench_clock_t::now();

    BenchResult r;
    r.name = name;
    r.total_ms = ms_t(t1 - t0).count();
    r.iterations = iterations;

    std::cout << std::left << std::setw(40) << r.name
              << " | " << std::setw(8) << std::fixed << std::setprecision(2) << r.total_ms << " ms"
              << " | avg " << std::setw(8) << r.avg_ms() << " ms"
              << " | " << std::setw(10) << std::fixed << std::setprecision(0) << r.ops_per_sec() << " ops/s"
              << "\n";

    return r;
}

auto main() -> int {
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           AI Learning System — Performance Benchmark                ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";

    // 小网络（CPU）
    ai_learning::core::LearnerConfig config_small;
    config_small.obs_dim = 16;
    config_small.action_dim = 4;
    // hidden_dims 默认 {64, 32}，矩阵远低于 4096 阈值，走 CPU

    // 大网络（触发 GPU 加速）
    ai_learning::core::LearnerConfig config_large;
    config_large.obs_dim = 256;
    config_large.action_dim = 16;
    config_large.hidden_dims = {512, 256};
    config_large.max_inference_steps = 10;  // 减少推理步数，避免超时
    // 矩阵: 512×272=139264, 256×512=131072 → 全部超过 4096 阈值，走 CUDA

    std::vector<BenchResult> results;

    // ────────────────────────────────────────────────────────────────
    // 0. 经验学习：小网络 vs 大网络（CPU vs GPU）
    // ────────────────────────────────────────────────────────────────
    std::cout << "── 经验学习：小网络 vs 大网络 ────────────────────────────\n";
    {
        ai_learning::core::Learner learner_s(config_small);
        auto s1s = std::vector<float>(16, 0.5f);
        auto s2s = std::vector<float>(16, 1.0f);
        results.push_back(run_bench("learn_exp [CPU] small 16→64→32", 1000, [&]() {
            learner_s.learn_from_experience(s1s, 0, s2s, 1.0f);
        }));
    }
    {
        ai_learning::core::Learner learner_l(config_large);
        auto s1l = std::vector<float>(256, 0.5f);
        auto s2l = std::vector<float>(256, 1.0f);
        results.push_back(run_bench("learn_exp [GPU] large 256→512→256", 100, [&]() {
            learner_l.learn_from_experience(s1l, 0, s2l, 1.0f);
        }));
    }

    // ────────────────────────────────────────────────────────────────
    // 1. 文本学习
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n── 文本学习 ──────────────────────────────────────────────\n";
    {
        ai_learning::core::Learner learner(config_small);
        results.push_back(run_bench("learn_from_text (单条)", 100, [&]() {
            learner.learn_from_text("人工智能是计算机科学的一个分支");
        }));
    }
    {
        ai_learning::core::Learner learner(config_small);
        results.push_back(run_bench("think (回答问题)", 50, [&]() {
            learner.learn_from_text("数学是研究数量的学科");
            learner.think("什么是数学");
        }));
    }

    // ────────────────────────────────────────────────────────────────
    // 2. 经验学习（小网络，用于对比）
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n── 经验学习 ──────────────────────────────────────────────\n";
    {
        ai_learning::core::Learner learner(config_small);
        auto s1 = std::vector<float>(16, 0.5f);
        auto s2 = std::vector<float>(16, 1.0f);
        results.push_back(run_bench("learn_from_experience", 1000, [&]() {
            learner.learn_from_experience(s1, 0, s2, 1.0f);
        }));
    }

    // ────────────────────────────────────────────────────────────────
    // 3. 感知编码
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n── 感知编码 ──────────────────────────────────────────────\n";
    {
        ai_learning::core::Learner learner(config_small);
        std::map<std::string, std::vector<float>> input = {
            {"visual", std::vector<float>(16, 0.5f)},
            {"auditory", {0.3f, 0.4f, 0.5f, 0.6f}},
            {"position", {0.5f, 0.5f}},
        };
        results.push_back(run_bench("perceive (多模态)", 10000, [&]() {
            learner.perceive(input);
        }));
    }

    // ────────────────────────────────────────────────────────────────
    // 4. 记忆系统
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n── 记忆系统 ──────────────────────────────────────────────\n";
    {
        ai_learning::core::Learner learner(config_small);
        auto s1 = std::vector<float>(16, 0.5f);
        auto s2 = std::vector<float>(16, 1.0f);
        // 先填充记忆
        for (int i = 0; i < 100; ++i) {
            learner.remember(s1, i % 4, s2, 0.5f, 0.1f);
        }
        results.push_back(run_bench("recall (检索)", 1000, [&]() {
            learner.recall(s1, 5);
        }));
    }
    {
        ai_learning::core::Learner learner(config_small);
        auto s1 = std::vector<float>(16, 0.5f);
        auto s2 = std::vector<float>(16, 1.0f);
        for (int i = 0; i < 200; ++i) {
            learner.remember(s1, i % 4, s2, 0.5f, 0.1f);
        }
        results.push_back(run_bench("consolidate (巩固)", 100, [&]() {
            learner.consolidate();
        }));
    }

    // ────────────────────────────────────────────────────────────────
    // 5. 推理引擎
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n── 推理引擎 ──────────────────────────────────────────────\n";
    {
        ai_learning::core::Learner learner(config_small);
        learner.learn_from_text("数学是研究数量的学科");
        learner.learn_from_text("物理是研究自然规律的科学");
        results.push_back(run_bench("reason (统一推理)", 100, [&]() {
            learner.reason("数学");
        }));
    }

    // ────────────────────────────────────────────────────────────────
    // 6. 自主学习循环
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n── 自主学习循环 ──────────────────────────────────────────\n";
    {
        ai_learning::core::Learner learner(config_small);
        ai_learning::domain::SimpleEnvironment env(4, 4, 42);
        results.push_back(run_bench("autonomous_learn (100步)", 10, [&]() {
            ai_learning::domain::SimpleEnvironment e(4, 4, 42);
            learner.autonomous_learn(e, 100, 0, 100);
        }));
    }

    // ────────────────────────────────────────────────────────────────
    // 7. 社会交互
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n── 社会交互 ──────────────────────────────────────────────\n";
    {
        ai_learning::core::Learner l1(config_small);
        ai_learning::core::Learner l2(config_small);
        ai_learning::social::SocialAgent a1(l1);
        ai_learning::social::SocialAgent a2(l2);
        results.push_back(run_bench("social interact", 100, [&]() {
            a1.interact(a2);
        }));
    }

    // ────────────────────────────────────────────────────────────────
    // 8. 持久化
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n── 持久化 ────────────────────────────────────────────────\n";
    {
        ai_learning::core::Learner learner(config_small);
        learner.learn_from_text("测试持久化性能");
        auto s1 = std::vector<float>(16, 0.5f);
        auto s2 = std::vector<float>(16, 1.0f);
        for (int i = 0; i < 50; ++i) {
            learner.learn_from_experience(s1, 0, s2, 1.0f);
        }
        results.push_back(run_bench("save (保存)", 100, [&]() {
            learner.save("bench_temp.txt");
        }));
        results.push_back(run_bench("load (加载)", 100, [&]() {
            learner.load("bench_temp.txt");
        }));
        std::remove("bench_temp.txt");
    }

    // ────────────────────────────────────────────────────────────────
    // 汇总
    // ────────────────────────────────────────────────────────────────
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "                            汇总报告\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << std::left
              << std::setw(42) << "测试项"
              << " | " << std::setw(10) << "总耗时(ms)"
              << " | " << std::setw(10) << "平均(ms)"
              << " | " << std::setw(12) << "吞吐量(ops/s)"
              << "\n";
    std::cout << std::string(85, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(42) << r.name
                  << " | " << std::setw(10) << std::fixed << std::setprecision(2) << r.total_ms
                  << " | " << std::setw(10) << r.avg_ms()
                  << " | " << std::setw(12) << std::fixed << std::setprecision(0) << r.ops_per_sec()
                  << "\n";
    }

    std::cout << "\n✅ Benchmark 完成！共 " << results.size() << " 项测试。\n";

    return 0;
}
