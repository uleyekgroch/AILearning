/**
 * @file tensor_ops_cuda_stub.cpp
 * @brief CUDA 函数的 CPU stub — 无 CUDA 环境时链接此文件
 *
 * 所有函数返回 false / 不执行，确保纯 CPU 构建正常工作。
 */

#include <vector>

namespace ai_learning::core {

auto cuda_available() -> bool { return false; }
auto cuda_fp16_available() -> bool { return false; }

auto cuda_mat_vec(const std::vector<float>&, int, int, const std::vector<float>&)
    -> std::vector<float> { return {}; }

auto cuda_vec_mat(const std::vector<float>&, const std::vector<float>&, int, int)
    -> std::vector<float> { return {}; }

void cuda_mat_add_outer(std::vector<float>&, float, const std::vector<float>&, const std::vector<float>&) {}

auto cuda_mat_vec_bias(const std::vector<float>&, int, int, const std::vector<float>&, const std::vector<float>&)
    -> std::vector<float> { return {}; }

auto cuda_mat_vec_fp16(const std::vector<float>&, int, int, const std::vector<float>&)
    -> std::vector<float> { return {}; }

auto cuda_tensor_relu(const std::vector<float>&) -> std::vector<float> { return {}; }
auto cuda_tensor_relu_deriv(const std::vector<float>&) -> std::vector<float> { return {}; }
auto cuda_tensor_clamp(const std::vector<float>&, float, float) -> std::vector<float> { return {}; }
auto cuda_tensor_sub(const std::vector<float>&, const std::vector<float>&) -> std::vector<float> { return {}; }
auto cuda_tensor_mul(const std::vector<float>&, const std::vector<float>&) -> std::vector<float> { return {}; }
auto cuda_tensor_scale(const std::vector<float>&, float) -> std::vector<float> { return {}; }

}  // namespace ai_learning::core
