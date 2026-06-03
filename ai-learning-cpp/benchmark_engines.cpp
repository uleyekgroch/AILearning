/**
 * @file benchmark_engines.cpp
 * @brief 预测引擎 head-to-head 性能基准
 *
 * 对比三种引擎：PC (Predictive Coding) / MLP / Light
 *
 * 测试维度：
 * - predict() 单次推理延迟
 * - learn() 单次学习延迟
 * - 持续吞吐量 (ops/sec)
 * - 状态序列化大小
 *
 * 测试配置：
 * - Small:  obs=16,  action=4
 * - Medium: obs=128, action=8
 *
 * 输出格式：Markdown 表格（便于复制到文档）
 */

#include "ai_learning/core/config.hpp"
#include "ai_learning/core/learner_factory.hpp"
#include "ai_learning/learning/ipredictive_engine.hpp"
#include "ai_learning/learning/light_predictive_engine.hpp"
#include "ai_learning/learning/mlp_forward_engine.hpp"
#include "ai_learning/learning/predictive_coding_engine.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using bench_clock_t = std::chrono::high_resolution_clock;
using ns_t = std::chrono::duration<double, std::nano>;
using ms_t = std::chrono::duration<double, std::milli>;

// ── 结果结构 ─────────────────────────────────────────────────────

struct EngineBenchResult {
    std::string engine_name;
    int obs_dim = 0;
    int action_dim = 0;

    double predict_avg_ns = 0.0;
    double learn_avg_ns = 0.0;
    double throughput_ops_per_sec = 0.0;
    size_t state_size_bytes = 0;
};

// ── 计时器 ────────────────────────────────────────────────────────

static auto now() -> bench_clock_t::time_point { return bench_clock_t::now(); }
static auto elapsed_ns(bench_clock_t::time_point t0) -> double {
    return ns_t(now() - t0).count();
}

// ── 热身 ────────────────────────────────────────────────────────

static void warm_up() {
    volatile double x = 0.0;
    for (int i = 0; i < 1'000'000; ++i) {
        x = x + std::sin(i * 0.001);
    }
    (void)x;
}

// ── 生成测试数据 ─────────────────────────────────────────────────

static auto make_obs(int dim, float value) -> std::vector<float> {
    return std::vector<float>(dim, value);
}

static auto make_action(int dim, float value) -> std::vector<float> {
    return std::vector<float>(dim, value);
}

// ── 单引擎基准 ───────────────────────────────────────────────────

static auto benchmark_engine(
    const std::string& name,
    std::unique_ptr<ai_learning::learning::IPredictiveEngine> engine,
    int obs_dim, int action_dim)
    -> EngineBenchResult
{
    EngineBenchResult r;
    r.engine_name = name;
    r.obs_dim = obs_dim;
    r.action_dim = action_dim;

    auto state = make_obs(obs_dim, 0.5f);
    auto action = make_action(action_dim, 0.1f);
    auto actual = make_obs(obs_dim, 1.0f);

    // ── 预测基准 ──────────────────────────────────────────────
    {
        int predict_iters = 1000;
        // 预热
        for (int i = 0; i < 10; ++i) engine->predict(state, action);

        auto t0 = now();
        for (int i = 0; i < predict_iters; ++i) {
            engine->predict(state, action);
        }
        r.predict_avg_ns = elapsed_ns(t0) / predict_iters;
    }

    // ── 学习基准 ──────────────────────────────────────────────
    {
        int learn_iters = 500;
        // 预热
        for (int i = 0; i < 5; ++i) {
            engine->learn(state, action, actual);
        }

        auto t0 = now();
        for (int i = 0; i < learn_iters; ++i) {
            engine->learn(state, action, actual);
        }
        r.learn_avg_ns = elapsed_ns(t0) / learn_iters;
    }

    // ── 吞吐量基准（持续 1 秒或至少 1000 次）───────────────────
    {
        int throughput_iters = 0;
        auto t0 = now();
        while (elapsed_ns(t0) < 1'000'000'000.0) {  // 1 秒
            engine->learn(state, action, actual);
            ++throughput_iters;
            if (throughput_iters < 1000) continue;  // 至少跑 1000 次
        }
        double total_s = elapsed_ns(t0) * 1e-9;
        r.throughput_ops_per_sec = throughput_iters / total_s;
    }

    // ── 状态大小
    {
        auto state_data = engine->save_state();
        r.state_size_bytes = state_data.weights.size() * sizeof(float)
                           + state_data.shape.size() * sizeof(int);
    }

    return r;
}

// ── 创建引擎 ─────────────────────────────────────────────────────

static auto make_pc_engine(const ai_learning::core::LearnerConfig& cfg)
    -> std::unique_ptr<ai_learning::learning::IPredictiveEngine> {
    return ai_learning::core::LearnerFactory::make_engine("pc", cfg);
}

static auto make_mlp_engine(const ai_learning::core::LearnerConfig& cfg)
    -> std::unique_ptr<ai_learning::learning::IPredictiveEngine> {
    return ai_learning::core::LearnerFactory::make_engine("mlp", cfg);
}

static auto make_light_engine(const ai_learning::core::LearnerConfig& cfg)
    -> std::unique_ptr<ai_learning::learning::IPredictiveEngine> {
    return ai_learning::core::LearnerFactory::make_engine("light", cfg);
}

// ── 格式化输出 ───────────────────────────────────────────────────

static void print_header(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(80, '=') << "\n";
}

static void print_table(const std::vector<EngineBenchResult>& results) {
    std::cout << "\n"
              << std::left << std::setw(12) << "Engine"
              << " | " << std::setw(8) << "obs"
              << " | " << std::setw(8) << "action"
              << " | " << std::setw(12) << "predict(ns)"
              << " | " << std::setw(12) << "learn(ns)"
              << " | " << std::setw(14) << "ops/sec"
              << " | " << std::setw(10) << "state(B)"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(12) << r.engine_name
                  << " | " << std::setw(8) << r.obs_dim
                  << " | " << std::setw(8) << r.action_dim
                  << " | " << std::setw(11) << std::fixed << std::setprecision(1) << r.predict_avg_ns
                  << " | " << std::setw(11) << std::fixed << std::setprecision(1) << r.learn_avg_ns
                  << " | " << std::setw(13) << std::fixed << std::setprecision(0) << r.throughput_ops_per_sec
                  << " | " << std::setw(9) << r.state_size_bytes
                  << "\n";
    }
}

static void print_markdown(const std::vector<EngineBenchResult>& results) {
    std::cout << "\n### Markdown 格式\n\n";
    std::cout << "| Engine | obs | action | predict(ns) | learn(ns) | ops/sec | state(B) |\n";
    std::cout << "|--------|-----|--------|-------------|-----------|---------|----------|\n";
    for (const auto& r : results) {
        std::cout << "| " << r.engine_name
                  << " | " << r.obs_dim
                  << " | " << r.action_dim
                  << " | " << std::fixed << std::setprecision(1) << r.predict_avg_ns
                  << " | " << std::fixed << std::setprecision(1) << r.learn_avg_ns
                  << " | " << std::fixed << std::setprecision(0) << r.throughput_ops_per_sec
                  << " | " << r.state_size_bytes
                  << " |\n";
    }
}

// ── 主函数 ────────────────────────────────────────────────────────

auto main() -> int {
    using namespace ai_learning;

    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        Predictive Engine Head-to-Head Benchmark                     ║\n";
    std::cout << "║        PC vs MLP vs Light                                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";

    warm_up();

    std::vector<EngineBenchResult> all_results;

    // ── Small Network ──────────────────────────────────────────────
    print_header("Small Network: obs=16, action=4");
    {
        core::LearnerConfig cfg;
        cfg.obs_dim = 16;
        cfg.action_dim = 4;

        std::cout << "  Testing PC..." << std::flush;
        all_results.push_back(benchmark_engine("PC", make_pc_engine(cfg), 16, 4));
        std::cout << " done\n";

        std::cout << "  Testing MLP..." << std::flush;
        all_results.push_back(benchmark_engine("MLP", make_mlp_engine(cfg), 16, 4));
        std::cout << " done\n";

        std::cout << "  Testing Light..." << std::flush;
        all_results.push_back(benchmark_engine("Light", make_light_engine(cfg), 16, 4));
        std::cout << " done\n";
    }

    // ── Medium Network ─────────────────────────────────────────────
    print_header("Medium Network: obs=128, action=8");
    {
        core::LearnerConfig cfg;
        cfg.obs_dim = 128;
        cfg.action_dim = 8;

        std::cout << "  Testing PC..." << std::flush;
        all_results.push_back(benchmark_engine("PC", make_pc_engine(cfg), 128, 8));
        std::cout << " done\n";

        std::cout << "  Testing MLP..." << std::flush;
        all_results.push_back(benchmark_engine("MLP", make_mlp_engine(cfg), 128, 8));
        std::cout << " done\n";

        std::cout << "  Testing Light..." << std::flush;
        all_results.push_back(benchmark_engine("Light", make_light_engine(cfg), 128, 8));
        std::cout << " done\n";
    }

    // ── Summary ────────────────────────────────────────────────────
    print_header("Summary Table");
    print_table(all_results);
    print_markdown(all_results);

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Benchmark complete.\n";
    return 0;
}
