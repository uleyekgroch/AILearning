/**
 * @file activation_spread_cuda.cu
 * @brief Activation Spreading CUDA 加速实现
 *
 * CSR 格式知识图谱上的激活扩散（SpMV）:
 * - kernel_activation_spread: 单轮扩散，每个 thread 处理一个节点
 * - kernel_activation_spread_batch: 批量扩散（多查询并行）
 * - cuda_activation_spread_multi_hop: 多轮迭代
 *
 * 复用 tensor_ops_cuda.cu 的 CudaContext / cuda_available() 基础设施。
 * 本模块使用独立的轻量 GpuBuffer，避免跨模块耦合。
 *
 * 性能策略：
 * - CSR 格式: 内存连续访问，无原子操作（单查询）
 * - Batch 模式: 每个查询独立激活向量，线程块分配查询×节点
 * - FP16 可选: 节点数 > 4096 时自动启用 FP16 激活值减少显存带宽
 * - 阈值: 节点数 > 256 时启用 CUDA
 */

#include "ai_learning/reasoning/activation_spread_cuda.cuh"
#include "ai_learning/core/cuda_fp16_utils.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <unordered_map>

// ── 复用 tensor_ops 的 CUDA 基础设施 ─────────────────────────────────
// CudaContext, cuda_available() 定义在 tensor_ops_cuda.cu
// 这里 extern 声明，不重复初始化 GPU

namespace ai_learning::core {
extern auto cuda_available() -> bool;
}  // namespace ai_learning::core

namespace ai_learning::reasoning {

// ── 本地 GPU 辅助 ──────────────────────────────────────────────────────

class SpreadGpuBuffer {
public:
    explicit SpreadGpuBuffer(size_t bytes) : size_(bytes) {
        if (cudaMalloc(&ptr_, bytes) != cudaSuccess) {
            ptr_ = nullptr;
        }
    }
    ~SpreadGpuBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    SpreadGpuBuffer(const SpreadGpuBuffer&) = delete;
    SpreadGpuBuffer& operator=(const SpreadGpuBuffer&) = delete;

    void upload(const void* host, size_t bytes, cudaStream_t stream) {
        cudaMemcpyAsync(ptr_, host, bytes, cudaMemcpyHostToDevice, stream);
    }

    void download(void* host, size_t bytes, cudaStream_t stream) const {
        cudaMemcpyAsync(host, ptr_, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    template <typename T>
    auto as() -> T* { return static_cast<T*>(ptr_); }

    template <typename T>
    auto as() const -> const T* { return static_cast<const T*>(ptr_); }

    auto valid() const -> bool { return ptr_ != nullptr; }

private:
    void*  ptr_{nullptr};
    size_t size_;
};

// ── CUDA Stream 管理 ───────────────────────────────────────────────────

static cudaStream_t get_spread_stream() {
    static cudaStream_t s_stream = nullptr;
    static bool s_initialized = false;
    if (!s_initialized) {
        cudaStreamCreate(&s_stream);
        s_initialized = true;
    }
    return s_stream;
}

// ── CUDA Kernel 配置 ───────────────────────────────────────────────────

static constexpr int SPREAD_BLOCK = 256;

static int spread_grid(int n) {
    return (n + SPREAD_BLOCK - 1) / SPREAD_BLOCK;
}

// ── CUDA Kernels ───────────────────────────────────────────────────────

/**
 * 单轮激活扩散 kernel — CSR SpMV
 *
 * 对每个节点 i:
 *   new_act[i] = sum( act[j] * w[j->i] ) for all j with edge j->i
 *   new_act[i] *= decay_rate
 *   if new_act[i] < threshold: new_act[i] = 0
 *
 * CSR 格式按出边组织: row_ptr[i] 给出节点 i 的出边范围。
 * 扩散是反向收集: 节点 i 的激活贡献到其邻居。
 * 但为了高效的 CSR SpMV，我们按入边组织。
 *
 * 假设 CSR 的 row_ptr/col_indices 表示入边邻接：
 *   节点 i 的入边邻居 = col_indices[row_ptr[i] .. row_ptr[i+1])
 *   对应权重 = edge_weights[row_ptr[i] .. row_ptr[i+1])
 *
 * 计算: new_act[i] = decay * sum( act[neighbor] * weight )
 *        if new_act[i] < threshold: new_act[i] = 0
 */
__global__ void kernel_activation_spread(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ act_in,
    float* __restrict__ act_out,
    int num_nodes,
    float decay_rate,
    float threshold)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int start = row_ptr[node];
    int end   = row_ptr[node + 1];

    float sum = 0.0f;
    for (int e = start; e < end; ++e) {
        int neighbor = col_indices[e];
        float weight = edge_weights[e];
        sum += act_in[neighbor] * weight;
    }

    float new_act = sum * decay_rate;
    if (new_act < threshold) {
        new_act = 0.0f;
    }

    // 保留种子节点自身的激活值（不覆盖已有激活）
    float prev = act_in[node];
    act_out[node] = (prev > new_act) ? prev : new_act;
}

/**
 * 多轮激活扩散 kernel — 原地 ping-pong
 *
 * 启动 N 次 kernel_activation_spread，交替使用 act_in/act_out 缓冲。
 * 每轮都应用 decay（衰减因子按轮数累积: decay^hop）。
 *
 * 注：多轮通过 host 循环调用 kernel_activation_spread 实现，
 * 此 kernel 仅用于单轮。
 */

/**
 * 批量激活扩散 kernel — 多查询并行
 *
 * 每个 thread 处理 (query, node) 对。
 * 使用独立的激活向量数组: act[query * num_nodes + node]
 *
 * 对于 batch SpMV:
 *   act_out[q * N + i] = decay * sum( act_in[q * N + j] * w[e] )
 *   for edges e from j to i in CSR
 */
__global__ void kernel_activation_spread_batch(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ act_in,
    float* __restrict__ act_out,
    int num_nodes,
    int num_queries,
    float decay_rate,
    float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_queries * num_nodes;
    if (idx >= total) return;

    int q    = idx / num_nodes;
    int node = idx % num_nodes;

    int start = row_ptr[node];
    int end   = row_ptr[node + 1];

    float sum = 0.0f;
    for (int e = start; e < end; ++e) {
        int neighbor = col_indices[e];
        float weight = edge_weights[e];
        sum += act_in[q * num_nodes + neighbor] * weight;
    }

    float new_act = sum * decay_rate;
    if (new_act < threshold) {
        new_act = 0.0f;
    }

    float prev = act_in[q * num_nodes + node];
    act_out[q * num_nodes + node] = (prev > new_act) ? prev : new_act;
}

/**
 * FP16 单轮激活扩散 kernel
 *
 * 与 kernel_activation_spread 相同逻辑，但激活值使用 FP16。
 * CSR 结构（row_ptr, col_indices, edge_weights）保持 FP32 以保精度。
 */
__global__ void kernel_activation_spread_fp16(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const float* __restrict__ edge_weights,
    const half* __restrict__ act_in,
    half* __restrict__ act_out,
    int num_nodes,
    float decay_rate,
    float threshold)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int start = row_ptr[node];
    int end   = row_ptr[node + 1];

    float sum = 0.0f;
    for (int e = start; e < end; ++e) {
        int neighbor = col_indices[e];
        float weight = edge_weights[e];
        float neighbor_act = __half2float(act_in[neighbor]);
        sum += neighbor_act * weight;
    }

    float new_act = sum * decay_rate;
    if (new_act < threshold) {
        new_act = 0.0f;
    }

    float prev = __half2float(act_in[node]);
    float result = (prev > new_act) ? prev : new_act;
    act_out[node] = __float2half(result);
}

// ── CSR 构建 ──────────────────────────────────────────────────────────

auto CsrGraph::build(
    int num_nodes,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_indices,
    const std::vector<float>& edge_weights,
    std::vector<std::string> node_id_map) -> CsrGraph
{
    CsrGraph g;
    g.num_nodes = num_nodes;
    g.num_edges = static_cast<int>(col_indices.size());
    g.row_ptr = row_ptr;
    g.col_indices = col_indices;
    g.edge_weights = edge_weights;
    g.node_id_map = std::move(node_id_map);
    return g;
}

// ── GPU 多轮扩散 ──────────────────────────────────────────────────────

auto cuda_activation_spread_multi_hop(
    const CsrGraph& graph,
    const std::vector<float>& seed_activations,
    float decay_rate,
    float threshold,
    int num_hops) -> SpreadResult
{
    assert(graph.num_nodes > 0);
    assert(static_cast<int>(seed_activations.size()) == graph.num_nodes);
    assert(num_hops > 0);
    assert(decay_rate >= 0.0f && decay_rate <= 1.0f);
    assert(threshold >= 0.0f);

    SpreadResult result;
    result.hops_completed = 0;
    result.activations.resize(graph.num_nodes, 0.0f);

    if (graph.num_edges == 0) {
        // 无边图：直接返回种子激活
        result.activations = seed_activations;
        return result;
    }

    cudaStream_t stream = get_spread_stream();

    int N = graph.num_nodes;
    int E = graph.num_edges;

    size_t row_ptr_bytes     = static_cast<size_t>(N + 1) * sizeof(int);
    size_t col_idx_bytes     = static_cast<size_t>(E) * sizeof(int);
    size_t edge_w_bytes      = static_cast<size_t>(E) * sizeof(float);
    size_t act_bytes         = static_cast<size_t>(N) * sizeof(float);

    // GPU 缓冲分配
    SpreadGpuBuffer d_row_ptr(row_ptr_bytes);
    SpreadGpuBuffer d_col_indices(col_idx_bytes);
    SpreadGpuBuffer d_edge_weights(edge_w_bytes);
    SpreadGpuBuffer d_act_a(act_bytes);
    SpreadGpuBuffer d_act_b(act_bytes);

    if (!d_row_ptr.valid() || !d_col_indices.valid() ||
        !d_edge_weights.valid() || !d_act_a.valid() || !d_act_b.valid()) {
        std::cerr << "[ActivationSpread CUDA] GPU buffer alloc failed, returning seeds\n";
        result.activations = seed_activations;
        return result;
    }

    // 上传 CSR 结构（不变，只上传一次）
    d_row_ptr.upload(graph.row_ptr.data(), row_ptr_bytes, stream);
    d_col_indices.upload(graph.col_indices.data(), col_idx_bytes, stream);
    d_edge_weights.upload(graph.edge_weights.data(), edge_w_bytes, stream);

    // 上传初始激活值
    d_act_a.upload(seed_activations.data(), act_bytes, stream);

    // 判断是否使用 FP16
    bool use_fp16 = (N > 4096) && ai_learning::core::cuda_fp16_available();

    int grid = spread_grid(N);

    if (use_fp16) {
        // FP16 路径：减少显存带宽
        size_t act_fp16_bytes = static_cast<size_t>(N) * sizeof(half);
        SpreadGpuBuffer d_act_a16(act_fp16_bytes);
        SpreadGpuBuffer d_act_b16(act_fp16_bytes);

        if (!d_act_a16.valid() || !d_act_b16.valid()) {
            use_fp16 = false;  // fallback to FP32 below
        } else {
            // FP32 → FP16 初始上传
            std::vector<half> h_act_fp16(N);
            for (int i = 0; i < N; ++i) {
                h_act_fp16[i] = __float2half(seed_activations[i]);
            }
            d_act_a16.upload(h_act_fp16.data(), act_fp16_bytes, stream);

            float decay_per_hop = decay_rate;

            for (int hop = 0; hop < num_hops; ++hop) {
                kernel_activation_spread_fp16<<<grid, SPREAD_BLOCK, 0, stream>>>(
                    d_row_ptr.as<int>(),
                    d_col_indices.as<int>(),
                    d_edge_weights.as<float>(),
                    (hop % 2 == 0) ? d_act_a16.as<half>() : d_act_b16.as<half>(),
                    (hop % 2 == 0) ? d_act_b16.as<half>() : d_act_a16.as<half>(),
                    N,
                    decay_per_hop,
                    threshold);

                result.hops_completed = hop + 1;
            }

            // 下载最终结果: FP16 → FP32
            const half* final_fp16 = (num_hops % 2 == 1)
                ? d_act_b16.as<half>()   // odd hops: result in B
                : d_act_a16.as<half>();  // even hops: result in A
            std::vector<half> h_result(N);
            d_act_a16.download(h_result.data(), act_fp16_bytes, stream);
            // Re-download from correct buffer
            cudaMemcpyAsync(h_result.data(),
                           (num_hops % 2 == 1) ? d_act_b16.as<half>() : d_act_a16.as<half>(),
                           act_fp16_bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            for (int i = 0; i < N; ++i) {
                result.activations[i] = __half2float(h_result[i]);
            }
        }
    }

    if (!use_fp16) {
        // FP32 路径
        for (int hop = 0; hop < num_hops; ++hop) {
            kernel_activation_spread<<<grid, SPREAD_BLOCK, 0, stream>>>(
                d_row_ptr.as<int>(),
                d_col_indices.as<int>(),
                d_edge_weights.as<float>(),
                (hop % 2 == 0) ? d_act_a.as<float>() : d_act_b.as<float>(),
                (hop % 2 == 0) ? d_act_b.as<float>() : d_act_a.as<float>(),
                N,
                decay_rate,
                threshold);

            result.hops_completed = hop + 1;
        }

        // 下载最终结果
        float* final_buf = (num_hops % 2 == 1)
            ? d_act_b.as<float>()   // odd hops: result in B
            : d_act_a.as<float>();  // even hops: result in A
        cudaMemcpyAsync(result.activations.data(), final_buf,
                        act_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    return result;
}

// ── GPU 批量扩散 ──────────────────────────────────────────────────────

auto cuda_activation_spread_batch(
    const CsrGraph& graph,
    const std::vector<std::vector<float>>& batch_seeds,
    float decay_rate,
    float threshold,
    int num_hops) -> BatchSpreadResult
{
    int num_queries = static_cast<int>(batch_seeds.size());
    assert(num_queries > 0);

    BatchSpreadResult result;
    result.hops_completed = 0;
    result.activations.resize(num_queries);

    if (graph.num_edges == 0) {
        for (int q = 0; q < num_queries; ++q) {
            result.activations[q] = batch_seeds[q];
        }
        return result;
    }

    int N = graph.num_nodes;
    int E = graph.num_edges;

    cudaStream_t stream = get_spread_stream();

    size_t row_ptr_bytes = static_cast<size_t>(N + 1) * sizeof(int);
    size_t col_idx_bytes = static_cast<size_t>(E) * sizeof(int);
    size_t edge_w_bytes  = static_cast<size_t>(E) * sizeof(float);
    size_t batch_bytes   = static_cast<size_t>(num_queries * N) * sizeof(float);

    SpreadGpuBuffer d_row_ptr(row_ptr_bytes);
    SpreadGpuBuffer d_col_indices(col_idx_bytes);
    SpreadGpuBuffer d_edge_weights(edge_w_bytes);
    SpreadGpuBuffer d_act_a(batch_bytes);
    SpreadGpuBuffer d_act_b(batch_bytes);

    if (!d_row_ptr.valid() || !d_col_indices.valid() ||
        !d_edge_weights.valid() || !d_act_a.valid() || !d_act_b.valid()) {
        std::cerr << "[ActivationSpread CUDA Batch] GPU buffer alloc failed\n";
        for (int q = 0; q < num_queries; ++q) {
            result.activations[q] = batch_seeds[q];
        }
        return result;
    }

    // 上传 CSR
    d_row_ptr.upload(graph.row_ptr.data(), row_ptr_bytes, stream);
    d_col_indices.upload(graph.col_indices.data(), col_idx_bytes, stream);
    d_edge_weights.upload(graph.edge_weights.data(), edge_w_bytes, stream);

    // 展平并上传 batch 种子
    std::vector<float> flat_seeds(num_queries * N, 0.0f);
    for (int q = 0; q < num_queries; ++q) {
        assert(static_cast<int>(batch_seeds[q].size()) == N);
        std::copy(batch_seeds[q].begin(), batch_seeds[q].end(),
                  flat_seeds.begin() + q * N);
    }
    d_act_a.upload(flat_seeds.data(), batch_bytes, stream);

    int total_threads = num_queries * N;
    int grid = spread_grid(total_threads);

    for (int hop = 0; hop < num_hops; ++hop) {
        kernel_activation_spread_batch<<<grid, SPREAD_BLOCK, 0, stream>>>(
            d_row_ptr.as<int>(),
            d_col_indices.as<int>(),
            d_edge_weights.as<float>(),
            (hop % 2 == 0) ? d_act_a.as<float>() : d_act_b.as<float>(),
            (hop % 2 == 0) ? d_act_b.as<float>() : d_act_a.as<float>(),
            N,
            num_queries,
            decay_rate,
            threshold);

        result.hops_completed = hop + 1;
    }

    // 下载并拆分结果
    std::vector<float> flat_result(num_queries * N);
    float* final_buf = (num_hops % 2 == 1)
        ? d_act_b.as<float>()
        : d_act_a.as<float>();
    cudaMemcpyAsync(flat_result.data(), final_buf,
                    batch_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int q = 0; q < num_queries; ++q) {
        result.activations[q].assign(
            flat_result.begin() + q * N,
            flat_result.begin() + (q + 1) * N);
    }

    return result;
}

// ── CPU 多轮扩散 ──────────────────────────────────────────────────────

static auto cpu_activation_spread_multi_hop(
    const CsrGraph& graph,
    const std::vector<float>& seed_activations,
    float decay_rate,
    float threshold,
    int num_hops) -> SpreadResult
{
    int N = graph.num_nodes;

    SpreadResult result;
    result.hops_completed = 0;
    result.activations = seed_activations;

    if (graph.num_edges == 0) return result;

    std::vector<float> act_next(N, 0.0f);

    for (int hop = 0; hop < num_hops; ++hop) {
        std::fill(act_next.begin(), act_next.end(), 0.0f);

        // CSR SpMV: 对每个节点，从入边邻居收集激活
        for (int i = 0; i < N; ++i) {
            int start = graph.row_ptr[i];
            int end   = graph.row_ptr[i + 1];

            float sum = 0.0f;
            for (int e = start; e < end; ++e) {
                int neighbor = graph.col_indices[e];
                float weight = graph.edge_weights[e];
                sum += result.activations[neighbor] * weight;
            }

            float new_act = sum * decay_rate;
            if (new_act < threshold) {
                new_act = 0.0f;
            }
            // 保留已有激活值
            act_next[i] = (result.activations[i] > new_act)
                ? result.activations[i] : new_act;
        }

        result.activations = act_next;
        result.hops_completed = hop + 1;
    }

    return result;
}

// ── CPU/GPU 自动分发 ────────────────────────────────────────────────────

auto activation_spread_dispatch(
    const CsrGraph& graph,
    const std::vector<float>& seed_activations,
    float decay_rate,
    float threshold,
    int num_hops) -> SpreadResult
{
    if (graph.num_nodes >= CUDA_SPREAD_THRESHOLD &&
        ai_learning::core::cuda_available()) {
        return cuda_activation_spread_multi_hop(
            graph, seed_activations, decay_rate, threshold, num_hops);
    }
    return cpu_activation_spread_multi_hop(
        graph, seed_activations, decay_rate, threshold, num_hops);
}

// ── 便捷函数：从邻接表构建 CSR 并扩散 ──────────────────────────────────

auto activation_spread_from_adjacency(
    const std::map<std::string, std::vector<std::pair<std::string, float>>>& adjacency,
    const std::vector<std::string>& node_ids,
    const std::vector<std::string>& seed_entity_ids,
    const std::vector<float>& seed_values,
    float decay_rate,
    float threshold,
    int num_hops) -> SpreadResult
{
    int N = static_cast<int>(node_ids.size());
    assert(static_cast<int>(seed_entity_ids.size()) == static_cast<int>(seed_values.size()));

    // 构建 entity_id → node_index 映射
    std::unordered_map<std::string, int> id_to_idx;
    for (int i = 0; i < N; ++i) {
        id_to_idx[node_ids[i]] = i;
    }

    // 构建入边 CSR（激活扩散是收集入边贡献）
    std::vector<int> row_ptr(N + 1, 0);
    std::vector<int> col_indices;
    std::vector<float> edge_weights;

    // 先统计每个节点的入边数
    std::vector<int> in_degree(N, 0);
    for (const auto& [src_id, neighbors] : adjacency) {
        auto src_it = id_to_idx.find(src_id);
        if (src_it == id_to_idx.end()) continue;
        int src_idx = src_it->second;
        for (const auto& [dst_id, weight] : neighbors) {
            auto dst_it = id_to_idx.find(dst_id);
            if (dst_it == id_to_idx.end()) continue;
            in_degree[dst_it->second]++;
        }
    }

    // 累积偏移
    row_ptr[0] = 0;
    for (int i = 0; i < N; ++i) {
        row_ptr[i + 1] = row_ptr[i] + in_degree[i];
    }

    // 填充边数据
    col_indices.resize(row_ptr[N]);
    edge_weights.resize(row_ptr[N]);
    std::vector<int> current_pos(N, 0);

    for (const auto& [src_id, neighbors] : adjacency) {
        auto src_it = id_to_idx.find(src_id);
        if (src_it == id_to_idx.end()) continue;
        int src_idx = src_it->second;
        for (const auto& [dst_id, weight] : neighbors) {
            auto dst_it = id_to_idx.find(dst_id);
            if (dst_it == id_to_idx.end()) continue;
            int dst_idx = dst_it->second;
            int pos = row_ptr[dst_idx] + current_pos[dst_idx];
            col_indices[pos] = src_idx;
            edge_weights[pos] = weight;
            current_pos[dst_idx]++;
        }
    }

    auto graph = CsrGraph::build(
        N, row_ptr, col_indices, edge_weights,
        std::vector<std::string>(node_ids.begin(), node_ids.end()));

    // 构建种子激活向量
    std::vector<float> seed_activations(N, 0.0f);
    for (size_t i = 0; i < seed_entity_ids.size(); ++i) {
        auto it = id_to_idx.find(seed_entity_ids[i]);
        if (it != id_to_idx.end()) {
            seed_activations[it->second] = seed_values[i];
        }
    }

    return activation_spread_dispatch(
        graph, seed_activations, decay_rate, threshold, num_hops);
}

}  // namespace ai_learning::reasoning
