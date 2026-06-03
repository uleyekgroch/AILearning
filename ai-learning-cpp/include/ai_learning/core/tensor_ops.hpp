/**
 * @file tensor_ops.hpp
 * @brief 轻量张量运算 — CPU + CUDA 双路径
 *
 * 提供预测编码引擎所需的基本向量/矩阵运算。
 * 编译时自动检测 CUDA：有则启用 GPU 加速，无则纯 CPU。
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#ifdef AI_LEARNING_USE_EIGEN
#include <Eigen/Dense>
#endif

namespace ai_learning::core {

using Tensor = std::vector<float>;

// ── CUDA 可用性检测 ─────────────────────────────────────────────
// （由 tensor_ops_cuda.cu 或 stub 提供）

/// 查询 CUDA 是否可用（懒初始化，首次调用时检测）
auto cuda_available() -> bool;

/// 查询 FP16 Tensor Core 是否可用（sm_70+, 懒初始化）
/// 注: 在无 CUDA 环境下始终返回 false（由 stub 提供）
auto cuda_fp16_available() -> bool;

// CUDA 加速版本的声明（由 tensor_ops_cuda.cu 提供）
auto cuda_mat_vec(const std::vector<float>& mat, int rows, int cols,
                  const Tensor& vec) -> Tensor;
auto cuda_vec_mat(const Tensor& vec, const std::vector<float>& mat,
                  int cols, int rows) -> Tensor;
void cuda_mat_add_outer(std::vector<float>& mat, float lr,
                         const Tensor& a, const Tensor& b);
auto cuda_mat_vec_bias(const std::vector<float>& mat, int rows, int cols,
                        const Tensor& vec, const Tensor& bias) -> Tensor;

// FP16 Tensor Core 加速版本（compute-heavy 场景使用）
auto cuda_mat_vec_fp16(const std::vector<float>& mat, int rows, int cols,
                        const Tensor& vec) -> Tensor;

auto cuda_tensor_relu(const Tensor& a) -> Tensor;
auto cuda_tensor_relu_deriv(const Tensor& a) -> Tensor;
auto cuda_tensor_clamp(const Tensor& a, float lo, float hi) -> Tensor;
auto cuda_tensor_sub(const Tensor& a, const Tensor& b) -> Tensor;
auto cuda_tensor_mul(const Tensor& a, const Tensor& b) -> Tensor;
auto cuda_tensor_scale(const Tensor& a, float s) -> Tensor;

// ── 向量运算 (CPU 实现) ──────────────────────────────────────────

/// 向量加法
inline auto tensor_add(const Tensor& a, const Tensor& b) -> Tensor {
    auto result = Tensor(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

/// 向量减法 — GPU 分发
inline auto tensor_sub(const Tensor& a, const Tensor& b) -> Tensor {
    // 小向量走 CPU（避免 PCIe 开销）
    if (a.size() < 1024 || !cuda_available()) {
        auto result = Tensor(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] - b[i];
        }
        return result;
    }
    return cuda_tensor_sub(a, b);
}

/// 标量乘法 — GPU 分发
inline auto tensor_scale(const Tensor& a, float s) -> Tensor {
    if (a.size() < 1024 || !cuda_available()) {
        auto result = Tensor(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] * s;
        }
        return result;
    }
    return cuda_tensor_scale(a, s);
}

/// 逐元素乘法 — GPU 分发
inline auto tensor_mul(const Tensor& a, const Tensor& b) -> Tensor {
    if (a.size() < 1024 || !cuda_available()) {
        auto result = Tensor(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] * b[i];
        }
        return result;
    }
    return cuda_tensor_mul(a, b);
}

/// 逐元素 clamp — GPU 分发
inline auto tensor_clamp(const Tensor& a, float lo, float hi) -> Tensor {
    if (a.size() < 1024 || !cuda_available()) {
        auto result = Tensor(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = std::clamp(a[i], lo, hi);
        }
        return result;
    }
    return cuda_tensor_clamp(a, lo, hi);
}

/// ReLU — GPU 分发
inline auto tensor_relu(const Tensor& a) -> Tensor {
    if (a.size() < 1024 || !cuda_available()) {
        auto result = Tensor(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = std::max(0.0f, a[i]);
        }
        return result;
    }
    return cuda_tensor_relu(a);
}

/// ReLU 导数 — GPU 分发
inline auto tensor_relu_deriv(const Tensor& a) -> Tensor {
    if (a.size() < 1024 || !cuda_available()) {
        auto result = Tensor(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] > 0.0f ? 1.0f : 0.0f;
        }
        return result;
    }
    return cuda_tensor_relu_deriv(a);
}

/// 均方误差 (MSE) — 纯 CPU（reduce 操作，GPU 优势不大）
inline auto tensor_mse(const Tensor& a, const Tensor& b) -> float {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum / static_cast<float>(a.size());
}

/// L2 范数
inline auto tensor_norm(const Tensor& a) -> float {
    float sum = 0.0f;
    for (auto v : a) sum += v * v;
    return std::sqrt(sum);
}

/// 归一化
inline auto tensor_normalize(const Tensor& a) -> Tensor {
    auto n = tensor_norm(a);
    if (n < 1e-8f) return Tensor(a.size(), 0.0f);
    return tensor_scale(a, 1.0f / n);
}

/// 余弦相似度
inline auto cosine_similarity(const Tensor& a, const Tensor& b) -> float {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    auto denom = std::sqrt(na) * std::sqrt(nb);
    return denom < 1e-8f ? 0.0f : dot / denom;
}

/// 全零向量
inline auto tensor_zeros(size_t n) -> Tensor {
    return Tensor(n, 0.0f);
}

/// 随机向量 (uniform [-std, std])
inline auto tensor_randn(size_t n, float std = 0.1f) -> Tensor {
    auto result = Tensor(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * std;
    }
    return result;
}

// ── 矩阵运算 (flat row-major) — GPU 分发 ────────────────────────

/// 矩阵 × 向量: mat(rows×cols) × vec(cols) → result(rows)
inline auto mat_vec(const std::vector<float>& mat,
                    int rows, int cols,
                    const Tensor& vec) -> Tensor {
    // 小矩阵走 CPU（避免 GPU PCIe 开销）
    // 阈值：矩阵元素 < 4096（约 64×64）时 CPU 更快
    if (rows * cols < 4096 || !cuda_available()) {
#ifdef AI_LEARNING_USE_EIGEN
        if (rows * cols >= 2048) {
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                            Eigen::RowMajor>>
                M(mat.data(), rows, cols);
            Eigen::Map<const Eigen::VectorXf> v(vec.data(), cols);
            Eigen::VectorXf res = M * v;
            return Tensor(res.data(), res.data() + res.size());
        }
#endif
        auto result = Tensor(rows, 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                result[r] += mat[r * cols + c] * vec[c];
            }
        }
        return result;
    }
    return cuda_mat_vec(mat, rows, cols, vec);
}

/// 向量 × 矩阵: vec(cols) × mat(cols×rows) → result(rows)
inline auto vec_mat(const Tensor& vec,
                    const std::vector<float>& mat,
                    int cols, int rows) -> Tensor {
    if (cols * rows < 4096 || !cuda_available()) {
        auto result = Tensor(rows, 0.0f);
        for (int c = 0; c < cols; ++c) {
            for (int r = 0; r < rows; ++r) {
                result[r] += vec[c] * mat[c * rows + r];
            }
        }
        return result;
    }
    return cuda_vec_mat(vec, mat, cols, rows);
}

/// 外积: outer(a, b) → matrix(a.size() × b.size())
inline auto outer(const Tensor& a, const Tensor& b) -> std::vector<float> {
    auto result = std::vector<float>(a.size() * b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i * b.size() + j] = a[i] * b[j];
        }
    }
    return result;
}

/// 矩阵 + 外积更新: mat += lr * outer(a, b)
inline void mat_add_outer(std::vector<float>& mat,
                           float lr,
                           const Tensor& a,
                           const Tensor& b) {
    if (a.size() * b.size() < 4096 || !cuda_available()) {
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                mat[i * b.size() + j] += lr * a[i] * b[j];
            }
        }
        return;
    }
    cuda_mat_add_outer(mat, lr, a, b);
}

/// 矩阵 × 向量 + 偏置
inline auto mat_vec_bias(const std::vector<float>& mat,
                          int rows, int cols,
                          const Tensor& vec,
                          const Tensor& bias) -> Tensor {
    if (rows * cols < 4096 || !cuda_available()) {
#ifdef AI_LEARNING_USE_EIGEN
        if (rows * cols >= 2048) {
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                            Eigen::RowMajor>>
                M(mat.data(), rows, cols);
            Eigen::Map<const Eigen::VectorXf> v(vec.data(), cols);
            Eigen::Map<const Eigen::VectorXf> b(bias.data(), rows);
            Eigen::VectorXf res = M * v + b;
            return Tensor(res.data(), res.data() + res.size());
        }
#endif
        auto result = mat_vec(mat, rows, cols, vec);
        for (int i = 0; i < rows; ++i) {
            result[i] += bias[i];
        }
        return result;
    }
    return cuda_mat_vec_bias(mat, rows, cols, vec, bias);
}

}  // namespace ai_learning::core
