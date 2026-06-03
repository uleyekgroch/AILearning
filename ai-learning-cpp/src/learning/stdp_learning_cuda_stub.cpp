/**
 * @file stdp_learning_cuda_stub.cpp
 * @brief STDP CUDA 函数的 CPU stub — 无 CUDA 环境时链接此文件
 *
 * 所有函数不执行操作，确保纯 CPU 构建正常工作。
 * CUDA 函数声明在 stdp_learning.hpp 中，分发逻辑通过
 * cuda_available() == false 自动回退到 CPU 路径。
 */

#include <vector>

namespace ai_learning::learning {

void cuda_stdp_batch_update(
    float*,
    const float*,
    const float*,
    int,
    int,
    float,
    float,
    float,
    float,
    float,
    float)
{
    // stub: no-op, dispatch falls back to CPU path via cuda_available() check
}

void cuda_stdp_batch_update_default(
    float*,
    const float*,
    const float*,
    int,
    int)
{
    // stub: no-op
}

}  // namespace ai_learning::learning
