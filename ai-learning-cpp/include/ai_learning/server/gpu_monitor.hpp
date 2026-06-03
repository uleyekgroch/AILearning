/**
 * @file gpu_monitor.hpp
 * @brief GPU 显存监控 — 可选，仅在 CUDA 编译时启用
 *
 * 用法:
 *   auto mem = GpuMonitor::memory_info();
 *   std::cout << "GPU used: " << mem.used_mb << " MB\n";
 */

#pragma once

#include <string>

#ifdef AI_LEARNING_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace ai_learning::server {

struct GpuMemoryInfo {
    bool available = false;
    size_t total_mb = 0;
    size_t free_mb = 0;
    size_t used_mb = 0;
    int device_id = -1;
    std::string device_name;
};

class GpuMonitor {
public:
    /// 检测 CUDA 是否可用
    [[nodiscard]] static auto cuda_available() -> bool {
#ifdef AI_LEARNING_WITH_CUDA
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
        return false;
#endif
    }

    /// 获取当前 GPU 显存信息
    [[nodiscard]] static auto memory_info() -> GpuMemoryInfo {
        GpuMemoryInfo info;
#ifdef AI_LEARNING_WITH_CUDA
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            return info;
        }
        info.available = true;
        info.device_id = 0;
        cudaSetDevice(0);

        size_t free = 0, total = 0;
        if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
            info.total_mb = total / (1024 * 1024);
            info.free_mb  = free / (1024 * 1024);
            info.used_mb  = info.total_mb - info.free_mb;
        }

        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            info.device_name = prop.name;
        }
#else
        (void)0;  // 无 CUDA 时静默返回 unavailable
#endif
        return info;
    }
};

}  // namespace ai_learning::server
