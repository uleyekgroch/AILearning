/**
 * @file benchmark_tensor_ops.cpp
 * @brief 手写 tensor_ops vs Eigen 细粒度性能基准
 *
 * 测试目标：量化当前手写矩阵/向量运算与 Eigen 的差距。
 *
 * 测试维度：
 * - mat_vec_bias: 矩阵×向量+偏置（预测编码核心）
 * - tensor_relu: 逐元素 ReLU
 * - tensor_sub: 逐元素减法
 * - mat_add_outer: 外积权重更新（Hebbian 核心）
 * - 端到端 PredictiveCodingEngine::learn()
 *
 * 输出格式：JSON（便于脚本解析和趋势追踪）
 */

#include "ai_learning/core/tensor_ops.hpp"
#include "ai_learning/learning/predictive_coding_engine.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using bench_clock_t = std::chrono::high_resolution_clock;
using ns_t = std::chrono::duration<double, std::nano>;

// ── 结果结构 ─────────────────────────────────────────────────────

struct BenchEntry {
    std::string name;
    int dim_rows = 0;    // 矩阵行数 / 向量长度
    int dim_cols = 0;    // 矩阵列数（矩阵运算时）
    int iterations = 0;
    double total_ns = 0.0;
    double avg_ns() const { return total_ns / iterations; }
    double ops() const {
        if (dim_cols > 0) return static_cast<double>(dim_rows * dim_cols);
        return static_cast<double>(dim_rows);
    }
    double gflops() const {
        return (ops() * iterations) / (total_ns * 1e-9) / 1e9;
    }
};

// ── 计时器 ────────────────────────────────────────────────────────

class Timer {
    bench_clock_t::time_point t0_;
public:
    void start() { t0_ = bench_clock_t::now(); }
    double elapsed_ns() const {
        return ns_t(bench_clock_t::now() - t0_).count();
    }
};

// ── 热身 ────────────────────────────────────────────────────────

static void warm_up() {
    volatile double x = 0.0;
    for (int i = 0; i < 1'000'000; ++i) x += std::sin(i * 0.001);
    (void)x;
}

// ── 报告输出 ─────────────────────────────────────────────────────

static void print_header() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     手写 tensor_ops vs Eigen 细粒度性能基准                              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n格式: [名称] | 维度 | 迭代次数 | 总耗时(ms) | 平均(ns) | 估算 GFLOPS\n";
    std::cout << std::string(90, '-') << "\n";
}

static void print_entry(const BenchEntry& e) {
    std::cout << std::left << std::setw(28) << e.name
              << " | " << std::setw(12)
              << (e.dim_cols > 0
                  ? std::to_string(e.dim_rows) + "x" + std::to_string(e.dim_cols)
                  : std::to_string(e.dim_rows))
              << " | " << std::setw(8) << e.iterations
              << " | " << std::setw(10) << std::fixed << std::setprecision(2)
              << (e.total_ns / 1e6)
              << " | " << std::setw(10) << std::fixed << std::setprecision(1)
              << e.avg_ns()
              << " | " << std::setw(8) << std::fixed << std::setprecision(3)
              << e.gflops()
              << "\n";
}

static void print_json(const std::vector<BenchEntry>& entries) {
    std::cout << "\n{\"benchmark\":\"tensor_ops\",\"entries\":[\n";
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        std::cout << "  {"
                  << "\"name\":\"" << e.name << "\","
                  << "\"rows\":" << e.dim_rows << ","
                  << "\"cols\":" << e.dim_cols << ","
                  << "\"iterations\":" << e.iterations << ","
                  << "\"total_ms\":" << std::fixed << std::setprecision(2)
                  << (e.total_ns / 1e6) << ","
                  << "\"avg_ns\":" << std::fixed << std::setprecision(1)
                  << e.avg_ns() << ","
                  << "\"gflops\":" << std::fixed << std::setprecision(3)
                  << e.gflops() << "}";
        if (i + 1 < entries.size()) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "]}\n";
}

// ── 手写 mat_vec_bias ────────────────────────────────────────────

static auto bench_hand_mat_vec_bias(int rows, int cols, int iters)
    -> BenchEntry {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> mat(rows * cols);
    std::vector<float> vec(cols);
    std::vector<float> bias(rows);
    for (auto& v : mat) v = dist(rng);
    for (auto& v : vec) v = dist(rng);
    for (auto& v : bias) v = dist(rng);

    Timer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        volatile auto result = ai_learning::core::mat_vec_bias(
            mat, rows, cols, vec, bias);
        (void)result;
    }

    BenchEntry e;
    e.name = "hand_mat_vec_bias";
    e.dim_rows = rows;
    e.dim_cols = cols;
    e.iterations = iters;
    e.total_ns = timer.elapsed_ns();
    return e;
}

// ── Eigen mat_vec_bias ────────────────────────────────────────────

static auto bench_eigen_mat_vec_bias(int rows, int cols, int iters)
    -> BenchEntry {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Eigen::MatrixXf M(rows, cols);
    Eigen::VectorXf v(cols);
    Eigen::VectorXf b(rows);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) M(r, c) = dist(rng);
        b(r) = dist(rng);
    }
    for (int c = 0; c < cols; ++c) v(c) = dist(rng);

    Timer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        volatile auto result = (M * v + b).eval();
        (void)result;
    }

    BenchEntry e;
    e.name = "eigen_mat_vec_bias";
    e.dim_rows = rows;
    e.dim_cols = cols;
    e.iterations = iters;
    e.total_ns = timer.elapsed_ns();
    return e;
}

// ── 手写 tensor_relu ────────────────────────────────────────────

static auto bench_hand_relu(int n, int iters) -> BenchEntry {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    ai_learning::core::Tensor a(n);
    for (auto& v : a) v = dist(rng);

    Timer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        volatile auto r = ai_learning::core::tensor_relu(a);
        (void)r;
    }

    BenchEntry e;
    e.name = "hand_relu";
    e.dim_rows = n;
    e.iterations = iters;
    e.total_ns = timer.elapsed_ns();
    return e;
}

// ── Eigen tensor_relu ───────────────────────────────────────────

static auto bench_eigen_relu(int n, int iters) -> BenchEntry {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Eigen::VectorXf v(n);
    for (int i = 0; i < n; ++i) v(i) = dist(rng);

    Timer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        volatile auto r = v.cwiseMax(0.0f).eval();
        (void)r;
    }

    BenchEntry e;
    e.name = "eigen_relu";
    e.dim_rows = n;
    e.iterations = iters;
    e.total_ns = timer.elapsed_ns();
    return e;
}

// ── 手写 tensor_sub ─────────────────────────────────────────────

static auto bench_hand_sub(int n, int iters) -> BenchEntry {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    ai_learning::core::Tensor a(n), b(n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    Timer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        volatile auto r = ai_learning::core::tensor_sub(a, b);
        (void)r;
    }

    BenchEntry e;
    e.name = "hand_sub";
    e.dim_rows = n;
    e.iterations = iters;
    e.total_ns = timer.elapsed_ns();
    return e;
}

// ── Eigen tensor_sub ────────────────────────────────────────────

static auto bench_eigen_sub(int n, int iters) -> BenchEntry {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Eigen::VectorXf a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a(i) = dist(rng);
        b(i) = dist(rng);
    }

    Timer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        volatile auto r = (a - b).eval();
        (void)r;
    }

    BenchEntry e;
    e.name = "eigen_sub";
    e.dim_rows = n;
    e.iterations = iters;
    e.total_ns = timer.elapsed_ns();
    return e;
}

// ── 手写 mat_add_outer ───────────────────────────────────────────

static auto bench_hand_outer(int a_size, int b_size, int iters)
    -> BenchEntry {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> mat(a_size * b_size);
    ai_learning::core::Tensor a(a_size), b(b_size);
    for (auto& v : mat) v = dist(rng);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    Timer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        ai_learning::core::mat_add_outer(mat, 0.01f, a, b);
    }

    BenchEntry e;
    e.name = "hand_mat_add_outer";
    e.dim_rows = a_size;
    e.dim_cols = b_size;
    e.iterations = iters;
    e.total_ns = timer.elapsed_ns();
    return e;
}

// ── Eigen outer ─────────────────────────────────────────────────

static auto bench_eigen_outer(int a_size, int b_size, int iters)
    -> BenchEntry {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Eigen::MatrixXf M(a_size, b_size);
    Eigen::VectorXf a(a_size), b_vec(b_size);
    for (int i = 0; i < a_size; ++i) {
        a(i) = dist(rng);
        for (int j = 0; j < b_size; ++j) {
            M(i, j) = dist(rng);
            if (i == 0) b_vec(j) = dist(rng);
        }
    }

    Timer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        volatile auto r = (M + 0.01f * a * b_vec.transpose()).eval();
        (void)r;
    }

    BenchEntry e;
    e.name = "eigen_outer";
    e.dim_rows = a_size;
    e.dim_cols = b_size;
    e.iterations = iters;
    e.total_ns = timer.elapsed_ns();
    return e;
}

// ── 端到端 PredictiveCodingEngine::learn() ─────────────────────

static auto bench_hand_pc_learn(int obs_dim, int hidden1, int hidden2,
                                 int iters) -> BenchEntry {
    ai_learning::learning::PredictiveCodingConfig cfg;
    cfg.obs_dim = obs_dim;
    cfg.action_dim = 8;
    cfg.hidden1_dim = hidden1;
    cfg.hidden2_dim = hidden2;
    cfg.max_inference_steps = 10;

    ai_learning::learning::PredictiveCodingEngine engine(cfg);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> obs(obs_dim);
    std::vector<float> action(8, 0.0f);
    action[0] = 1.0f;
    std::vector<float> actual(obs_dim);
    for (auto& v : obs) v = dist(rng);
    for (auto& v : actual) v = dist(rng);

    Timer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        engine.learn(obs, action, actual);
    }

    BenchEntry e;
    e.name = "hand_pc_learn";
    e.dim_rows = obs_dim;
    e.dim_cols = hidden1;
    e.iterations = iters;
    e.total_ns = timer.elapsed_ns();
    return e;
}

// ── 主函数 ──────────────────────────────────────────────────────

auto main() -> int {
    warm_up();
    print_header();

    std::vector<BenchEntry> results;

    // ═══════════════════════════════════════════════════════════════
    // 1. mat_vec_bias 对比
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── mat_vec_bias (矩阵×向量+偏置) ────────────────────\n";
    const int mat_sizes[][2] = {
        {16, 8}, {64, 32}, {128, 64}, {256, 128}, {512, 256}
    };
    for (const auto& sz : mat_sizes) {
        int rows = sz[0], cols = sz[1];
        int iters = (rows <= 64) ? 100'000 : (rows <= 128) ? 50'000 : 10'000;
        auto h = bench_hand_mat_vec_bias(rows, cols, iters);
        auto e = bench_eigen_mat_vec_bias(rows, cols, iters);
        print_entry(h);
        print_entry(e);
        results.push_back(h);
        results.push_back(e);
    }

    // ═══════════════════════════════════════════════════════════════
    // 2. tensor_relu 对比
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── tensor_relu (逐元素 ReLU) ─────────────────────────\n";
    const int vec_sizes[] = {64, 256, 512, 1024, 4096};
    for (int n : vec_sizes) {
        int iters = (n <= 256) ? 100'000 : (n <= 1024) ? 50'000 : 20'000;
        auto h = bench_hand_relu(n, iters);
        auto e = bench_eigen_relu(n, iters);
        print_entry(h);
        print_entry(e);
        results.push_back(h);
        results.push_back(e);
    }

    // ═══════════════════════════════════════════════════════════════
    // 3. tensor_sub 对比
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── tensor_sub (逐元素减法) ────────────────────────────\n";
    for (int n : vec_sizes) {
        int iters = (n <= 256) ? 100'000 : (n <= 1024) ? 50'000 : 20'000;
        auto h = bench_hand_sub(n, iters);
        auto e = bench_eigen_sub(n, iters);
        print_entry(h);
        print_entry(e);
        results.push_back(h);
        results.push_back(e);
    }

    // ═══════════════════════════════════════════════════════════════
    // 4. mat_add_outer 对比
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── mat_add_outer (外积权重更新) ───────────────────────\n";
    const int outer_sizes[][2] = {
        {64, 128}, {128, 256}, {256, 512}
    };
    for (const auto& sz : outer_sizes) {
        int a = sz[0], b = sz[1];
        int iters = (a <= 64) ? 50'000 : 10'000;
        auto h = bench_hand_outer(a, b, iters);
        auto e = bench_eigen_outer(a, b, iters);
        print_entry(h);
        print_entry(e);
        results.push_back(h);
        results.push_back(e);
    }

    // ═══════════════════════════════════════════════════════════════
    // 5. 端到端 PredictiveCodingEngine::learn()
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n── 端到端 PredictiveCodingEngine::learn() ────────────\n";
    const int pc_configs[][3] = {
        {16, 64, 32},
        {128, 256, 128},
        {256, 512, 256},
    };
    for (const auto& cfg : pc_configs) {
        int obs = cfg[0], h1 = cfg[1], h2 = cfg[2];
        int iters = (obs <= 16) ? 10'000 : 1'000;
        auto h = bench_hand_pc_learn(obs, h1, h2, iters);
        print_entry(h);
        results.push_back(h);
    }

    // ═══════════════════════════════════════════════════════════════
    // 汇总与加速比
    // ═══════════════════════════════════════════════════════════════
    std::cout << "\n══════════════════════════════════════════════════════════════════\n";
    std::cout << "                      加速比汇总\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << std::left
              << std::setw(28) << "测试项"
              << " | " << std::setw(12) << "维度"
              << " | " << std::setw(10) << "手写(ns)"
              << " | " << std::setw(10) << "Eigen(ns)"
              << " | " << std::setw(10) << "加速比"
              << "\n";
    std::cout << std::string(85, '-') << "\n";

    for (size_t i = 0; i + 1 < results.size(); i += 2) {
        const auto& h = results[i];
        if (h.name.substr(0, 4) != "hand") continue;
        const auto& e = results[i + 1];
        if (e.name.substr(0, 5) != "eigen") continue;

        double speedup = h.avg_ns() / e.avg_ns();
        std::cout << std::left << std::setw(28) << h.name
                  << " | " << std::setw(12)
                  << (h.dim_cols > 0
                      ? std::to_string(h.dim_rows) + "x" + std::to_string(h.dim_cols)
                      : std::to_string(h.dim_rows))
                  << " | " << std::setw(10) << std::fixed << std::setprecision(1)
                  << h.avg_ns()
                  << " | " << std::setw(10) << std::fixed << std::setprecision(1)
                  << e.avg_ns()
                  << " | " << std::setw(10) << std::fixed << std::setprecision(2)
                  << speedup
                  << "\n";
    }

    // JSON 输出（脚本解析用）
    print_json(results);

    std::cout << "\n✅ Tensor Ops Benchmark 完成！共 " << results.size() << " 项测试。\n";
    return 0;
}
