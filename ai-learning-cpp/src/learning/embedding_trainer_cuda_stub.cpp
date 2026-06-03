/**
 * @file embedding_trainer_cuda_stub.cpp
 * @brief Embedding Trainer CUDA 函数的 CPU stub — 无 CUDA 环境时链接此文件
 *
 * 所有函数不执行操作，确保纯 CPU 构建正常工作。
 * CUDA 函数声明在 embedding_trainer_cuda.cuh 中，分发逻辑通过
 * cuda_available() == false 自动回退到 CPU 路径。
 */

#include "ai_learning/learning/embedding_trainer_cuda.cuh"

#include <random>
#include <vector>

namespace ai_learning::learning {

auto train_epoch_cuda_dispatch(
    std::vector<float>&,
    std::vector<float>&,
    const std::vector<int>&,
    const std::vector<int>&,
    int,
    int,
    int,
    int,
    double,
    double&,
    std::mt19937&) -> long long
{
    // stub: no-op, dispatch falls back to CPU path via cuda_available() check
    return 0;
}

}  // namespace ai_learning::learning
