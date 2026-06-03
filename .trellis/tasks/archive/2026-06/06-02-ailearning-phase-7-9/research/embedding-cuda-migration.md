# Research: Embedding Training CUDA Migration

- **Query**: Understand existing CUDA module patterns to implement the 6th CUDA algorithm: Embedding Training CUDA migration
- **Scope**: Internal (codebase analysis)
- **Date**: 2026-06-03

## Findings

### Files Found

| File Path | Description |
|---|---|
| `ai-learning-cpp/src/learning/embedding_trainer_cuda.cu` | Existing CUDA implementation (375 lines) -- fully written, not a stub |
| `ai-learning-cpp/src/learning/embedding_trainer.cpp` | CPU implementation + main train() loop (729 lines) |
| `ai-learning-cpp/include/ai_learning/learning/embedding_trainer.hpp` | Header with EmbeddingTrainer class, config structs (206 lines) |
| `ai-learning-cpp/src/learning/stdp_learning_cuda.cu` | Reference CUDA module -- STDP pattern (263 lines) |
| `ai-learning-cpp/src/learning/active_experimenter_cuda.cu` | Reference CUDA module -- Bayesian pattern (574 lines) |
| `ai-learning-cpp/include/ai_learning/learning/stdp_learning_cuda.cuh` | STDP CUDA header (function declarations) |
| `ai-learning-cpp/include/ai_learning/learning/active_experimenter_cuda.cuh` | Bayesian CUDA header (types + function declarations) |
| `ai-learning-cpp/src/learning/stdp_learning_cuda_stub.cpp` | STDP CPU stub (no-op when CUDA unavailable) |
| `ai-learning-cpp/src/learning/active_experimenter_cuda_stub.cpp` | Bayesian CPU stub (includes full CPU fallback) |
| `ai-learning-cpp/src/core/tensor_ops_cuda.cu` | Core CUDA infrastructure (CudaContext, cuBLAS, cuda_available()) |
| `ai-learning-cpp/tests/learning/test_embedding_trainer.cpp` | Existing tests for embedding trainer (344 lines) |
| `ai-learning-cpp/CMakeLists.txt` | Build configuration -- CUDA sources auto-collected via GLOB_RECURSE |

### Key Observation: Missing Pieces

1. **No `embedding_trainer_cuda_stub.cpp` exists** -- all other 5 CUDA modules have stub files. This must be created.
2. **No `embedding_trainer_cuda.cuh` header exists** -- the .cu file declares `train_epoch_cuda_dispatch` as extern directly.
3. **The .cu file IS NOT wired into the training loop** -- `embedding_trainer.cpp` line 201-202 computes `use_cuda` but `(void)use_cuda;` suppresses it, and `train()` always calls `train_epoch_cpu_()`.
4. **CMakeLists.txt does NOT exclude an embedding_trainer_cuda_stub** -- since the stub does not exist, there's no FILTER line for it (unlike the other 5 modules).

### Existing CUDA Code Analysis (embedding_trainer_cuda.cu)

The .cu file at `src/learning/embedding_trainer_cuda.cu` is a **complete, working CUDA implementation** with 375 lines. It contains:

#### Kernels (lines 54-136)

**kernel_sgns_train** (lines 54-124):
- Each CUDA thread processes one (center, context) training pair
- Computes dot product between center vector and all targets (positive + negatives)
- Applies sigmoid and computes SGD gradient
- Uses `atomicAdd` for W_out updates (conflicting targets across threads) and W_in_grad accumulation
- Uses shared memory for per-thread gradient buffer: `extern __shared__ float s_grad[]`
- Outputs per-pair loss

**kernel_apply_grad** (lines 127-136):
- Simple elementwise add of gradient to W_in matrix

#### GPU Memory Management (lines 140-190)

**EmbeddingGpuState** struct (lines 140-151):
- Holds all GPU pointers (d_W_in, d_W_out, d_W_in_grad, d_centers, d_contexts, d_neg_ids, d_loss)
- Tracks vocab_size and dim for reallocation logic

**gpu_alloc/gpu_cleanup** (lines 155-190):
- Global singleton `g_emb_gpu` -- allocates persistent W_in, W_out, W_in_grad matrices
- Uses `goto fail` pattern for error cleanup
- Reuses allocations when vocab_size/dim haven't changed

#### Host Dispatch (lines 196-373)

**train_epoch_cuda_dispatch** (lines 196-373):
- Signature matches the CPU `train_epoch_cpu_()` parameters closely
- Takes `std::vector<float>& W_in, W_out` (host references) + corpus + neg_table + params
- Batch size hardcoded to 512
- Loop: generate training pairs on CPU -> upload batch -> launch kernel -> download loss
- **Critical issue**: Temp buffers (d_centers, d_contexts, d_neg_ids, d_loss) are cudaMalloc'd and cudaFree'd **every batch** (lines 282-327) -- massive allocation churn
- Downloads final W_in/W_out from GPU at end (lines 368-369)
- Returns 0 on GPU alloc failure (no fallback signal)

### Pattern Comparison: Established CUDA Module Architecture

#### Pattern A: STDP (simple, direct kernel call)

Structure:
1. **Header** (`stdp_learning_cuda.cuh`): function declarations only
2. **CUDA impl** (`stdp_learning_cuda.cu`):
   - Local `StdpGpuBuffer` RAII class (cudaMalloc in ctor, cudaFree in dtor)
   - `get_stdp_stream()` -- module-local CUDA stream (static)
   - Kernel: one thread per synapse
   - Public function: `cuda_stdp_batch_update()` -- alloc, upload, kernel, download
3. **Stub** (`stdp_learning_cuda_stub.cpp`): no-op functions (dispatch in caller checks cuda_available())
4. **Dispatch**: Done in the calling code (stdp_learning.cpp) via `if (cuda_available())`

#### Pattern B: Bayesian (struct-based input/output, CPU fallback in stub)

Structure:
1. **Header** (`active_experimenter_cuda.cuh`): GPU data types (GpuHypothesis, BayesianUpdateInput, BayesianUpdateResult) + function declarations + dispatch function + threshold constant
2. **CUDA impl** (`active_experimenter_cuda.cu`):
   - Local `BayesianGpuBuffer` RAII class
   - `get_bayesian_stream()` -- module-local stream
   - Multiple kernels: kernel_bayesian_update, kernel_bayesian_normalize, kernel_information_gain, kernel_experiment_gain
   - Two-pass normalization (kernel + CPU reduction + kernel)
   - Public functions: `cuda_bayesian_batch_update()`, `cpu_bayesian_batch_update()`, `dispatch_bayesian_batch_update()`, `build_bayesian_input()`
3. **Stub** (`active_experimenter_cuda_stub.cpp`): includes full `cpu_bayesian_batch_update()` + `dispatch_bayesian_batch_update()` (always CPU) + `build_bayesian_input()`
4. **Dispatch**: Via `dispatch_bayesian_batch_update()` which checks threshold + cuda_available()

### EmbeddingTrainer Class Analysis

#### SGNS Training Algorithm (train_epoch_cpu_, lines 254-341)

Core loop per corpus position:
1. Skip document separators (corpus_[pos] < 0)
2. Random window size: `win_dist(rng_)` gives [1, window_size]
3. For each context word in window:
   - Reset gradient accumulator (dim-sized)
   - For positive + N negative samples:
     - Dot product: center . target (D multiply-adds)
     - Sigmoid activation
     - Gradient: `lr * (label - sigmoid)`
     - Accumulate center gradient: `grad_center += grad * v_target`
     - Update target vector: `v_target += grad * v_center` (immediate)
   - Apply accumulated gradient to center vector

#### Data Storage

- `W_in_`: vector<float> of size vocab_size * embedding_dim (row-major, center vectors)
- `W_out_`: vector<float> of size vocab_size * embedding_dim (row-major, context vectors)
- `corpus_`: vector<int> of token indices, -1 as doc separator
- `neg_table_`: vector<int> of size NEG_TABLE_SIZE (100000), unigram^0.75 distribution
- Default dim = 128, window = 5, neg_samples = 5, epochs = 5

#### train() Method (lines 175-252)

Current flow:
1. build_vocabulary_() -> prepare_corpus_() -> build_neg_table_() -> init_embeddings_()
2. `bool use_cuda = core::cuda_available() && vocab_size() > 100;` (line 201)
3. **Always calls train_epoch_cpu_()** -- CUDA dispatch is not wired (line 218)
4. Tracks loss, progress callback

### What Needs to Be Done (Gap Analysis)

#### 1. Create `embedding_trainer_cuda_stub.cpp`
Must provide:
- `train_epoch_cuda_dispatch()` that returns 0 (indicating "not handled, use CPU")

#### 2. Create `embedding_trainer_cuda.cuh` header
Should declare:
- `train_epoch_cuda_dispatch()` function signature
- Any GPU-specific types if needed

#### 3. Wire dispatch into `embedding_trainer.cpp` train() method
Change lines 212-218 from:
```cpp
long long pairs = train_epoch_cpu_(epoch, lr, epoch_loss);
```
To a CPU/GPU dispatch pattern like:
```cpp
long long pairs;
if (use_cuda) {
    pairs = train_epoch_cuda_dispatch(W_in_, W_out_, corpus_, neg_table_,
        config_.embedding_dim, config_.neg_samples, config_.window_size,
        epoch, lr, epoch_loss, rng_);
    if (pairs == 0) pairs = train_epoch_cpu_(epoch, lr, epoch_loss);
} else {
    pairs = train_epoch_cpu_(epoch, lr, epoch_loss);
}
```

#### 4. Fix existing .cu file issues
- **Memory churn**: d_centers, d_contexts, d_neg_ids, d_loss are malloc'd/freed every batch -- should use persistent buffers or RAII
- **Global singleton**: `g_emb_gpu` is a raw global -- inconsistent with RAII pattern used in other modules
- **No CUDA stream**: Uses default stream (0) -- other modules use dedicated streams
- **No error checking after kernel launches**: Missing `cudaGetLastError()` / `cudaDeviceSynchronize()`
- **Shared memory sizing**: `CUDA_BLOCK * D * sizeof(float)` -- for D=128, that's 128KB which exceeds typical 48KB shared memory limit. This kernel will fail for embedding_dim >= 192.

#### 5. Update CMakeLists.txt
- Add stub exclusion filter: `list(FILTER AI_LEARNING_SOURCES EXCLUDE REGEX "embedding_trainer_cuda_stub\\.cpp$")`
- Add stub source in else() block: `list(APPEND AI_LEARNING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/learning/embedding_trainer_cuda_stub.cpp")`

#### 6. Add CUDA consistency test
- Compare CPU and CUDA training output for identical input (precision threshold 1e-3 as per PRD)

### PRD Requirements (Phase 8, Algorithm #1)

From prd.md lines 176-178:
- **Algorithm**: Embedding vector training
- **CUDA Strategy**: FP16 GEMM + SGD kernel
- **Flash Attention**: Yes (attention-weighted)
- **Batch Processing**: Batch sample gradients

The existing .cu file uses FP32 only. The PRD calls for FP16 GEMM optimization using cuBLAS GemmEx. The existing kernel uses custom dot-product loops rather than cuBLAS GEMM.

### Related Specs

- `.trellis/tasks/06-02-ailearning-phase-7-9/prd.md` -- Phase 8 roadmap, Algorithm #1 specification

## Caveats / Not Found

- **embedding_trainer_cuda_stub.cpp does not exist** -- must be created from scratch
- **embedding_trainer_cuda.cuh does not exist** -- must be created from scratch
- **Shared memory overflow risk**: The existing kernel uses `extern __shared__ float s_grad[]` sized to `CUDA_BLOCK * D * sizeof(float)`. For the default D=128, this is 256*128*4 = 128KB, well above the 48KB shared memory limit on most GPUs. This kernel WILL fail at runtime for default config.
- **No existing integration test** for CUDA path in embedding trainer
- **The existing .cu includes `<cublas_v2.h>` but never uses cuBLAS** -- the header is imported but all math is done in custom kernels
