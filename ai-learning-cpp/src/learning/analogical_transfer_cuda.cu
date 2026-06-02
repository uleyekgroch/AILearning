/**
 * @file analogical_transfer_cuda.cu
 * @brief Analogical Transfer Alignment CUDA 加速实现
 *
 * 实现 N^2 概念对齐的 GPU 加速：
 * - kernel_jaccard_batch: 每个 CUDA thread 计算一对 (source, target) 的 Jaccard
 * - kernel_greedy_alignment: 每个 block 处理一个 source 的贪心匹配
 * - cuda_analogical_align: host 端完整流程
 *
 * 性能策略：
 * - Jaccard 计算：N_source * N_target 个 thread 完全并行
 * - FP16 相似度矩阵：节省 N^2 内存带宽
 * - 贪心匹配：atomicCAS 保证 target 分配互斥
 * - 阈值：concept pairs > CUDA_ALIGNMENT_THRESHOLD 时启用 CUDA
 *
 * 复用模式参考 stdp_learning_cuda.cu:
 * - 本地 StdpGpuBuffer → 本地 AlignGpuBuffer
 * - extern cuda_available() 避免 GPU 重复初始化
 */

#include "ai_learning/learning/analogical_transfer_cuda.cuh"
#include "ai_learning/learning/analogical_transfer.hpp"
#include "ai_learning/core/cuda_fp16_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

// ── 复用 tensor_ops 的 CUDA 基础设施 ─────────────────────────────
namespace ai_learning::core {
extern auto cuda_available() -> bool;
}  // namespace ai_learning::core

namespace ai_learning::learning {

// ── GPU Buffer Helper ──────────────────────────────────────────────

/// 轻量 GPU 缓冲（仅用于 analogical transfer，避免依赖 tensor_ops 内部类）
class AlignGpuBuffer {
public:
    explicit AlignGpuBuffer(size_t bytes) : size_(bytes) {
        if (cudaMalloc(&ptr_, bytes) != cudaSuccess) {
            ptr_ = nullptr;
            std::cerr << "[Analogical CUDA] cudaMalloc failed for "
                      << bytes << " bytes\n";
        }
    }
    ~AlignGpuBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    AlignGpuBuffer(const AlignGpuBuffer&) = delete;
    AlignGpuBuffer& operator=(const AlignGpuBuffer&) = delete;

    void upload(const void* host, size_t bytes, cudaStream_t stream) {
        cudaMemcpyAsync(ptr_, host, bytes, cudaMemcpyHostToDevice, stream);
    }

    void download(void* host, size_t bytes, cudaStream_t stream) const {
        cudaMemcpyAsync(host, ptr_, bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    auto ptr() -> void* { return ptr_; }
    auto ptr() const -> const void* { return ptr_; }
    auto as_u32() -> uint32_t* { return static_cast<uint32_t*>(ptr_); }
    auto as_u32() const -> const uint32_t* { return static_cast<const uint32_t*>(ptr_); }
    auto as_half() -> half* { return static_cast<half*>(ptr_); }
    auto as_half() const -> const half* { return static_cast<const half*>(ptr_); }
    auto as_float() -> float* { return static_cast<float*>(ptr_); }
    auto as_float() const -> const float* { return static_cast<const float*>(ptr_); }
    auto as_int() -> int* { return static_cast<int*>(ptr_); }
    auto as_bytes() -> void* { return ptr_; }
    auto valid() const -> bool { return ptr_ != nullptr; }

private:
    void*  ptr_{nullptr};
    size_t size_;
};

// ── CUDA Constants ─────────────────────────────────────────────────

static constexpr int ALIGN_BLOCK = 256;

static int align_grid(int n) {
    return (n + ALIGN_BLOCK - 1) / ALIGN_BLOCK;
}

// ── Device-side GpuConceptSet (POD, GPU-friendly) ──────────────────

struct DeviceConceptSet {
    int      hash_offset;
    int      hash_count;
    uint32_t domain_tag;
};

// ── CUDA Stream Management ─────────────────────────────────────────

static cudaStream_t get_align_stream() {
    static cudaStream_t s_stream = nullptr;
    static bool s_initialized = false;
    if (!s_initialized) {
        cudaStreamCreate(&s_stream);
        s_initialized = true;
    }
    return s_stream;
}

// ── Kernel 1: Batch Jaccard Similarity ─────────────────────────────

/**
 * kernel_jaccard_batch — 计算 N_source x N_target Jaccard 相似度矩阵。
 *
 * 每个 CUDA thread 处理一对 (source_idx, target_idx)：
 *   1. 读取 source 的属性哈希集合 A 和 target 的属性哈希集合 B
 *   2. 计算 |A ∩ B| / |A ∪ B|
 *   3. 写入 FP16 相似度矩阵 sim_matrix[s * n_target + t]
 *
 * 属性哈希已排序，用双指针法求交集：
 *   i = j = 0
 *   while i < |A| && j < |B|:
 *       if A[i] == B[j]: intersection++; i++; j++
 *       elif A[i] < B[j]: i++
 *       else: j++
 *
 * @param d_concepts    概念集数组 [n_source + n_target]
 * @param d_hashes      属性哈希大数组
 * @param d_sim_matrix  输出 FP16 相似度矩阵 [n_source * n_target]
 * @param n_source      source 概念数量
 * @param n_target      target 概念数量
 * @param total_pairs   n_source * n_target（用于 bounds check）
 */
__global__ void kernel_jaccard_batch(
    const DeviceConceptSet* __restrict__ d_concepts,
    const uint32_t*        __restrict__ d_hashes,
    half*                  __restrict__ d_sim_matrix,
    int n_source,
    int n_target,
    int total_pairs)
{
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) return;

    // 从线性索引恢复 (source, target) 二维坐标
    int s = pair_idx / n_target;
    int t = pair_idx % n_target;

    // source 概念在前 n_source 个位置
    const DeviceConceptSet& src = d_concepts[s];
    // target 概念在后 n_target 个位置
    const DeviceConceptSet& tgt = d_concepts[n_source + t];

    // 交集计数：双指针法（属性哈希已排序）
    int intersection = 0;
    int i = 0;
    int j = 0;

    while (i < src.hash_count && j < tgt.hash_count) {
        uint32_t a = d_hashes[src.hash_offset + i];
        uint32_t b = d_hashes[tgt.hash_offset + j];
        if (a == b) {
            intersection++;
            i++;
            j++;
        } else if (a < b) {
            i++;
        } else {
            j++;
        }
    }

    int union_size = src.hash_count + tgt.hash_count - intersection;
    float jaccard = (union_size > 0)
        ? __uint2float_rn(intersection) / __uint2float_rn(union_size)
        : 0.0f;

    // FP16 存储
    d_sim_matrix[pair_idx] = __float2half(jaccard);
}

// ── Kernel 2: FP16 → FP32 Conversion ──────────────────────────────

/**
 * kernel_half_to_float_batch — 将 FP16 相似度矩阵转为 FP32。
 * 后续贪心对齐和 host 下载均使用 FP32。
 */
__global__ void kernel_half_to_float_batch(
    const half* __restrict__ d_half,
    float*     __restrict__ d_float,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_float[idx] = __half2float(d_half[idx]);
}

// ── Kernel 3: Greedy Alignment ─────────────────────────────────────

/**
 * kernel_greedy_alignment — 基于相似度矩阵的贪心匹配。
 *
 * 每个 CUDA block 处理一个 source 概念：
 *   1. block 内协作找该 source 的最佳未分配 target
 *   2. 用 atomicCAS 尝试占锁，失败则退而求其次
 *
 * @param d_sim_matrix  FP32 相似度矩阵 [n_source * n_target]
 * @param d_target_lock 原子锁数组 [n_target]，0=空闲，1=已分配
 * @param d_alignment   输出对齐结果 [n_source * 2]：(target_idx, score) 对
 * @param n_source      source 概念数量
 * @param n_target      target 概念数量
 * @param min_score     最低对齐分数阈值
 */
__global__ void kernel_greedy_alignment(
    const float* __restrict__ d_sim_matrix,
    int*         __restrict__ d_target_lock,
    float*       __restrict__ d_alignment,
    int n_source,
    int n_target,
    float min_score)
{
    // 一个 block 处理一个 source
    int s = blockIdx.x;
    if (s >= n_source) return;

    // block 内协作找最佳 target：归约求 max
    // 使用 shared memory 加速
    __shared__ int   s_best_idx;
    __shared__ float s_best_score;

    // 多轮贪心：每轮找当前最佳未分配 target
    for (int attempt = 0; attempt < n_target; ++attempt) {
        // 初始化为无效
        if (threadIdx.x == 0) {
            s_best_idx   = -1;
            s_best_score = min_score;  // 只考虑超过阈值的
        }
        __syncthreads();

        // 每个 thread 检查一部分 target
        for (int t = threadIdx.x; t < n_target; t += blockDim.x) {
            float score = d_sim_matrix[s * n_target + t];
            if (score > s_best_score && d_target_lock[t] == 0) {
                // 原子更新全局最佳（简化：直接写入 shared，最后一个 wins）
                // 使用简单的 shared memory 归约
                if (score > s_best_score) {
                    s_best_score = score;
                    s_best_idx   = t;
                }
            }
        }
        __syncthreads();

        // 没有找到超过阈值的 target，结束
        if (s_best_idx < 0) break;

        // thread 0 尝试用 atomicCAS 占锁
        if (threadIdx.x == 0) {
            int expected = 0;
            int got = atomicCAS(&d_target_lock[s_best_idx], 0, 1);
            if (got == 0) {
                // 成功占用：记录对齐结果
                // d_alignment[s * 2 + 0] = target_idx (as float bits)
                // d_alignment[s * 2 + 1] = score
                d_alignment[s * 2 + 0] = __int2float_rn(s_best_idx);
                d_alignment[s * 2 + 1] = s_best_score;
                break;  // 每个 source 只匹配一个 target
            }
            // else: 锁被其他 block 占了，下一轮尝试次优
        }
        __syncthreads();

        // 检查是否已经成功匹配（由 thread 0 设置）
        // 如果 d_alignment[s*2] 已被设置（> -0.5f），则退出
        if (d_alignment[s * 2] >= 0.0f) break;
    }
}

// ── Public: CUDA Analogical Align ──────────────────────────────────

auto cuda_analogical_align(const AlignmentInput& input) -> AlignmentResult {
    assert(!input.source_concepts.empty());
    assert(!input.target_concepts.empty());

    int n_source = static_cast<int>(input.source_concepts.size());
    int n_target = static_cast<int>(input.target_concepts.size());
    int total_pairs = n_source * n_target;

    cudaStream_t stream = get_align_stream();

    // ── 准备 device-side concept 数据 ─────────────────────────────
    // target 哈希在拼接数组中的偏移 = source_hashes.size()
    int tgt_hash_offset = static_cast<int>(input.source_hashes.size());

    std::vector<DeviceConceptSet> d_concepts_host;
    d_concepts_host.reserve(n_source + n_target);
    for (const auto& c : input.source_concepts) {
        d_concepts_host.push_back({c.hash_offset, c.hash_count, c.domain_tag});
    }
    for (const auto& c : input.target_concepts) {
        // target 哈希偏移需要加上 source 哈希数组的大小
        d_concepts_host.push_back({c.hash_offset + tgt_hash_offset, c.hash_count, c.domain_tag});
    }

    // ── 分配 GPU 缓冲 ─────────────────────────────────────────────
    size_t concepts_bytes = (n_source + n_target) * sizeof(DeviceConceptSet);
    size_t src_hashes_bytes = input.source_hashes.size() * sizeof(uint32_t);
    size_t tgt_hashes_bytes = input.target_hashes.size() * sizeof(uint32_t);
    size_t all_hashes_bytes = src_hashes_bytes + tgt_hashes_bytes;
    size_t sim_half_bytes   = total_pairs * sizeof(half);
    size_t sim_float_bytes  = total_pairs * sizeof(float);
    size_t lock_bytes       = n_target * sizeof(int);
    size_t alignment_bytes  = n_source * 2 * sizeof(float);

    AlignGpuBuffer d_concepts(concepts_bytes);
    AlignGpuBuffer d_hashes(all_hashes_bytes);
    AlignGpuBuffer d_sim_half(sim_half_bytes);
    AlignGpuBuffer d_sim_float(sim_float_bytes);
    AlignGpuBuffer d_target_lock(lock_bytes);
    AlignGpuBuffer d_alignment(alignment_bytes);

    if (!d_concepts.valid() || !d_hashes.valid() ||
        !d_sim_half.valid() || !d_sim_float.valid() ||
        !d_target_lock.valid() || !d_alignment.valid()) {
        std::cerr << "[Analogical CUDA] GPU buffer allocation failed\n";
        return cpu_analogical_align(input);
    }

    // ── 上传数据 ──────────────────────────────────────────────────
    d_concepts.upload(d_concepts_host.data(), concepts_bytes, stream);

    // 拼接 source + target 哈希到统一数组
    // target hashes 的偏移 = source_hashes.size()
    std::vector<uint32_t> all_hashes = input.source_hashes;
    all_hashes.insert(all_hashes.end(),
                      input.target_hashes.begin(), input.target_hashes.end());
    d_hashes.upload(all_hashes.data(), all_hashes_bytes, stream);

    // 初始化对齐结果为 -1 (未匹配)
    std::vector<float> alignment_init(n_source * 2, -1.0f);
    d_alignment.upload(alignment_init.data(), alignment_bytes, stream);

    // 初始化锁为 0 (空闲)
    std::vector<int> lock_init(n_target, 0);
    d_target_lock.upload(lock_init.data(), lock_bytes, stream);

    // ── Kernel 1: Jaccard 相似度 (FP16) ───────────────────────────
    int grid_jaccard = align_grid(total_pairs);
    auto* d_concepts_ptr = static_cast<DeviceConceptSet*>(d_concepts.as_bytes());
    kernel_jaccard_batch<<<grid_jaccard, ALIGN_BLOCK, 0, stream>>>(
        d_concepts_ptr,
        d_hashes.as_u32(),
        d_sim_half.as_half(),
        n_source,
        n_target,
        total_pairs);

    // ── Kernel 1b: FP16 → FP32 转换 ──────────────────────────────
    int grid_convert = align_grid(total_pairs);
    kernel_half_to_float_batch<<<grid_convert, ALIGN_BLOCK, 0, stream>>>(
        d_sim_half.as_half(),
        d_sim_float.as_float(),
        total_pairs);

    // ── Kernel 2: Greedy Alignment ────────────────────────────────
    // 每个 block 处理一个 source
    kernel_greedy_alignment<<<n_source, ALIGN_BLOCK, 0, stream>>>(
        d_sim_float.as_float(),
        d_target_lock.as_int(),
        d_alignment.as_float(),
        n_source,
        n_target,
        input.min_alignment_score);

    // ── 下载结果 ──────────────────────────────────────────────────
    std::vector<float> sim_matrix(total_pairs);
    d_sim_float.download(sim_matrix.data(), sim_float_bytes, stream);

    std::vector<float> alignment_host(n_source * 2);
    d_alignment.download(alignment_host.data(), alignment_bytes, stream);

    // ── 构建 AlignmentResult ──────────────────────────────────────
    AlignmentResult result;
    result.n_source = n_source;
    result.n_target = n_target;
    result.similarity_matrix = std::move(sim_matrix);

    for (int s = 0; s < n_source; ++s) {
        float t_idx  = alignment_host[s * 2 + 0];
        float score  = alignment_host[s * 2 + 1];
        if (t_idx >= 0.0f && score >= input.min_alignment_score) {
            result.pairs.push_back({
                s,
                static_cast<int>(t_idx),
                score
            });
        }
    }

    // 按分数降序排列
    std::sort(result.pairs.begin(), result.pairs.end(),
              [](const AlignmentPair& a, const AlignmentPair& b) {
                  return a.jaccard_score > b.jaccard_score;
              });

    return result;
}

// ── CPU Fallback ───────────────────────────────────────────────────

namespace {

/// FNV-1a 32-bit hash（与 GPU 端一致）
auto fnv1a_hash(const std::string& s) -> uint32_t {
    uint32_t h = 2166136261u;
    for (char c : s) {
        h ^= static_cast<uint32_t>(static_cast<unsigned char>(c));
        h *= 16777619u;
    }
    return h;
}

/// CPU Jaccard 计算
auto cpu_jaccard(const std::vector<uint32_t>& a,
                 const std::vector<uint32_t>& b) -> float {
    if (a.empty() && b.empty()) return 0.0f;
    int intersection = 0;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) { intersection++; i++; j++; }
        else if (a[i] < b[j]) { i++; }
        else { j++; }
    }
    int union_size = static_cast<int>(a.size() + b.size()) - intersection;
    return union_size > 0 ? static_cast<float>(intersection) / union_size : 0.0f;
}

}  // anonymous namespace

auto cpu_analogical_align(const AlignmentInput& input) -> AlignmentResult {
    int n_source = static_cast<int>(input.source_concepts.size());
    int n_target = static_cast<int>(input.target_concepts.size());

    AlignmentResult result;
    result.n_source = n_source;
    result.n_target = n_target;
    result.similarity_matrix.resize(n_source * n_target, 0.0f);

    // 为每个概念提取排序后的哈希子数组
    auto get_hashes = [&](const GpuConceptSet& c, const std::vector<uint32_t>& pool)
        -> std::vector<uint32_t> {
        std::vector<uint32_t> h(
            pool.begin() + c.hash_offset,
            pool.begin() + c.hash_offset + c.hash_count);
        std::sort(h.begin(), h.end());
        return h;
    };

    // N^2 Jaccard
    for (int s = 0; s < n_source; ++s) {
        auto src_hashes = get_hashes(input.source_concepts[s], input.source_hashes);
        for (int t = 0; t < n_target; ++t) {
            auto tgt_hashes = get_hashes(input.target_concepts[t], input.target_hashes);
            float sim = cpu_jaccard(src_hashes, tgt_hashes);
            result.similarity_matrix[s * n_target + t] = sim;
        }
    }

    // 贪心匹配：每个 source 选最佳未分配 target
    std::vector<bool> target_used(n_target, false);

    // 先按分数排序所有 pair
    struct ScoredPair { int s, t; float score; };
    std::vector<ScoredPair> all_pairs;
    all_pairs.reserve(n_source * n_target);
    for (int s = 0; s < n_source; ++s) {
        for (int t = 0; t < n_target; ++t) {
            all_pairs.push_back({s, t, result.similarity_matrix[s * n_target + t]});
        }
    }
    std::sort(all_pairs.begin(), all_pairs.end(),
              [](const ScoredPair& a, const ScoredPair& b) {
                  return a.score > b.score;
              });

    std::vector<bool> source_matched(n_source, false);
    for (const auto& p : all_pairs) {
        if (source_matched[p.s] || target_used[p.t]) continue;
        if (p.score < input.min_alignment_score) break;
        source_matched[p.s] = true;
        target_used[p.t] = true;
        result.pairs.push_back({p.s, p.t, p.score});
    }

    return result;
}

// ── CPU/GPU Auto-Dispatch ──────────────────────────────────────────

auto dispatch_analogical_align(const AlignmentInput& input) -> AlignmentResult {
    int n_source = static_cast<int>(input.source_concepts.size());
    int n_target = static_cast<int>(input.target_concepts.size());
    int total_pairs = n_source * n_target;

    if (total_pairs >= CUDA_ALIGNMENT_THRESHOLD && ai_learning::core::cuda_available()) {
        return cuda_analogical_align(input);
    }
    return cpu_analogical_align(input);
}

// ── Build AlignmentInput from ConceptDescriptors ───────────────────

auto build_alignment_input(
    const std::vector<ConceptDescriptor>& source,
    const std::vector<ConceptDescriptor>& target,
    float min_score) -> AlignmentInput
{
    AlignmentInput input;
    input.min_alignment_score = min_score;

    // Helper: hash sorted unique attributes of a concept
    auto hash_concept = [](const ConceptDescriptor& cd,
                          std::vector<GpuConceptSet>& concepts,
                          std::vector<uint32_t>& hashes,
                          std::vector<std::string>& ids) {
        int offset = static_cast<int>(hashes.size());

        // 去重排序后哈希
        std::set<std::string> unique_attrs(cd.attributes.begin(), cd.attributes.end());
        std::vector<uint32_t> attr_hashes;
        attr_hashes.reserve(unique_attrs.size());
        for (const auto& attr : unique_attrs) {
            attr_hashes.push_back(fnv1a_hash(attr));
        }
        std::sort(attr_hashes.begin(), attr_hashes.end());

        // 去重哈希（可能碰撞但可接受）
        attr_hashes.erase(
            std::unique(attr_hashes.begin(), attr_hashes.end()),
            attr_hashes.end());

        int count = static_cast<int>(attr_hashes.size());

        concepts.push_back({
            static_cast<int>(concepts.size()),
            offset,
            count,
            fnv1a_hash(cd.domain)
        });

        hashes.insert(hashes.end(), attr_hashes.begin(), attr_hashes.end());
        ids.push_back(cd.id);
    };

    for (const auto& cd : source) {
        hash_concept(cd, input.source_concepts, input.source_hashes, input.source_ids);
    }
    for (const auto& cd : target) {
        hash_concept(cd, input.target_concepts, input.target_hashes, input.target_ids);
    }

    return input;
}

}  // namespace ai_learning::learning
