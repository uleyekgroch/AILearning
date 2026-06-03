export const meta = {
  name: 'phase-8-cuda',
  description: 'Phase 8: CUDA performance optimization - FP16 infrastructure, 5 algorithm migrations, Flash Attention',
  phases: [
    { title: 'FP16 Infra', detail: 'sm_89 upgrade + FP16 Tensor Core infrastructure' },
    { title: 'CUDA Kernels', detail: '5 algorithm kernels in parallel (STDP, activation, graph, analogy, bayesian)' },
    { title: 'Flash Attention', detail: 'Flash Attention integration for embedding + activation' },
    { title: 'Wire Up', detail: 'Wire CUDA kernels into existing C++ code + update CMakeLists.txt' },
  ],
}

// Phase 1: FP16 infrastructure + sm_89 upgrade
phase('FP16 Infra')

const fp16 = await agent(
  'Active task: .trellis/tasks/06-02-ailearning-phase-7-9\n\n## Phase 8.1: FP16 Mixed Precision Infrastructure\n\nUpgrade CUDA infrastructure for RTX 4060 Ada (sm_89) with FP16 Tensor Core support.\n\nTasks:\n1. Update ai-learning-cpp/CMakeLists.txt: change CUDA_ARCHITECTURES from "86" to "89"\n2. Create ai-learning-cpp/include/ai_learning/core/cuda_fp16_utils.cuh with:\n   - FP16 wrapper functions (half2 operations)\n   - Tensor Core GEMM helper using cublasGemmEx with CUDA_R_16F\n   - FP16<->FP32 conversion utilities\n   - Batch memory pool (reduce cudaMalloc overhead)\n3. Update tensor_ops_cuda.cu to add FP16 mat_vec variants:\n   - cuda_mat_vec_fp16(): uses cublasGemmEx with half precision\n   - Precision control: knowledge queries use FP32, compute-heavy use FP16\n\nKey files to READ first:\n- ai-learning-cpp/CMakeLists.txt (current CUDA config)\n- ai-learning-cpp/src/core/tensor_ops_cuda.cu (current CUDA implementation pattern)\n- ai-learning-cpp/include/ai_learning/core/tensor_ops.hpp (dispatch interface)\n\nDesign:\n- Use cublasGemmEx with CUDA_R_16F for Tensor Core acceleration\n- Keep FP32 accumulation for accuracy (cublasComputeType::CUBLAS_COMPUTE_32F)\n- Add cuda_fp16_available() check (sm_70+)\n- GpuBufferPool class for memory reuse\n\nDo NOT modify any .cpp files (only .cu and .cuh). Maintain backward compatibility.',
  { label: 'fp16-infra', phase: 'FP16 Infra' }
)

log('FP16 infra: ' + (fp16 ? 'done' : 'failed'))

// Phase 2: 5 CUDA kernels in parallel
phase('CUDA Kernels')

const stdpKernel = await agent(
  'Active task: .trellis/tasks/06-02-ailearning-phase-7-9\n\n## Phase 8.2a: STDP Weight Update CUDA Kernel\n\nCreate CUDA kernel for STDP (Spike-Timing-Dependent Plasticity) weight updates.\n\nFile: ai-learning-cpp/src/learning/stdp_learning_cuda.cu\n\nREAD first: ai-learning-cpp/include/ai_learning/learning/stdp_learning.hpp to understand the STDP data structures and update rules.\n\nImplement:\n1. kernel_stdpc_update: parallel STDP weight update for all synapses\n   - Each CUDA thread handles one synapse\n   - Weight change: dw = A_plus * exp(-delta_t / tau_plus) if pre before post, else -A_minus * exp(delta_t / tau_minus)\n   - Batch mode: process N pre-post spike pairs per synapse\n2. Batch interface: cuda_stpdc_batch_update() that accepts arrays of weights, pre_times, post_times\n3. CPU fallback dispatch in stdp_learning.hpp (add inline GPU dispatch like tensor_ops.hpp pattern)\n\nAlso create header: ai-learning-cpp/include/ai_learning/learning/stdp_learning_cuda.cuh\n\nUse the existing GpuBuffer and CudaContext from tensor_ops_cuda.cu (declare as extern).\nFP16 not needed here - STDP uses scalar timing values.\nFocus on batch processing: maximize parallel synapse updates per kernel launch.',
  { label: 'cuda-stdp', phase: 'CUDA Kernels' }
)

const activationKernel = await agent(
  'Active task: .trellis/tasks/06-02-ailearning-phase-7-9\n\n## Phase 8.2b: Activation Spread CUDA Kernel\n\nCreate CUDA kernel for activation spreading in knowledge graph (message passing).\n\nFile: ai-learning-cpp/src/reasoning/activation_spread_cuda.cu\nHeader: ai-learning-cpp/include/ai_learning/reasoning/activation_spread_cuda.cuh\n\nREAD first:\n- ai-learning-cpp/include/ai_learning/reasoning/activation_spread.hpp - understand ActivationSpread class, spread algorithm\n- ai-learning-cpp/include/ai_learning/domain/knowledge/knowledge_graph.hpp - graph structure\n\nImplement:\n1. kernel_activation_spread: one iteration of activation spreading\n   - Each CUDA thread handles one node\n   - Gather activations from neighbors via edges\n   - Apply decay factor and threshold\n   - AtomicAdd for concurrent edge contributions\n2. Multi-hop spread: launch kernel N times for N hops\n3. Batch query: spread activation from multiple source nodes simultaneously\n\nKey optimization: Use CSR (Compressed Sparse Row) format for graph edges on GPU.\nThis is essentially a sparse matrix-vector multiply (SpMV).\n\nUse FP16 for activation values (via cuda_fp16_utils.cuh if available, else FP32).\nProvide cpu_gpu dispatch wrapper.',
  { label: 'cuda-activation', phase: 'CUDA Kernels' }
)

const graphKernel = await agent(
  'Active task: .trellis/tasks/06-02-ailearning-phase-7-9\n\n## Phase 8.2c: Knowledge Graph Query CUDA Kernel\n\nCreate CUDA kernel for accelerated knowledge graph queries.\n\nFile: ai-learning-cpp/src/domain/knowledge/knowledge_graph_cuda.cu\nHeader: ai-learning-cpp/include/ai_learning/domain/knowledge/knowledge_graph_cuda.cuh\n\nREAD first:\n- ai-learning-cpp/include/ai_learning/domain/knowledge/knowledge_graph.hpp - KnowledgeGraph class API\n- ai-learning-cpp/include/ai_learning/domain/knowledge/entity.hpp - Entity structure\n- ai-learning-cpp/include/ai_learning/domain/knowledge/relation.hpp - Relation structure\n\nImplement:\n1. GPU graph representation:\n   - CSR adjacency list (nodes + edges on GPU)\n   - Entity properties as GPU arrays\n   - Relation types as integer arrays\n2. CUDA kernels:\n   - cuda_graph_find_neighbors: parallel neighbor lookup for batch queries\n   - cuda_graph_shortest_path: BFS-based shortest path on GPU\n   - cuda_graph_entity_search: parallel entity property matching\n3. Graph upload: cuda_graph_upload() copies knowledge graph to GPU once, reuse for queries\n4. L2 cache persistence hint for frequently accessed graph data (cudaFuncSetAttribute)\n\nFocus on batch query throughput - process multiple queries in parallel.\nUse CSR format for memory-efficient graph storage.\nProvide cpu_gpu dispatch.',
  { label: 'cuda-graph', phase: 'CUDA Kernels' }
)

const analogyKernel = await agent(
  'Active task: .trellis/tasks/06-02-ailearning-phase-7-9\n\n## Phase 8.2d: Analogical Transfer Alignment CUDA Kernel\n\nCreate CUDA kernel for Jaccard similarity + N2 concept alignment.\n\nFile: ai-learning-cpp/src/learning/analogical_transfer_cuda.cu\nHeader: ai-learning-cpp/include/ai_learning/learning/analogical_transfer_cuda.cuh\n\nREAD first:\n- ai-learning-cpp/include/ai_learning/learning/analogical_transfer.hpp - AnalogicalTransferEngine class, alignment algorithm\n\nImplement:\n1. kernel_jaccard_batch: compute Jaccard similarity for all source-target concept pairs\n   - Each CUDA thread handles one (source, target) pair\n   - Compute |intersection| / |union| of attribute sets\n   - Batch: N_source x N_target pairs in parallel\n2. kernel_greedy_alignment: greedy matching based on similarity scores\n   - One CUDA thread block handles one source concept\n   - Atomic operations for target assignment\n3. cuda_analogical_align() wrapper: upload concepts, run kernels, download alignment\n\nUse FP16 for similarity scores (via cuda_fp16_utils.cuh if available).\nOptimize for N2 parallelism - the main bottleneck.\nProvide cpu_gpu dispatch.',
  { label: 'cuda-analogy', phase: 'CUDA Kernels' }
)

const bayesianKernel = await agent(
  'Active task: .trellis/tasks/06-02-ailearning-phase-7-9\n\n## Phase 8.2e: Bayesian Hypothesis Update CUDA Kernel\n\nCreate CUDA kernel for parallel Bayesian posterior computation.\n\nFile: ai-learning-cpp/src/learning/active_experimenter_cuda.cu\nHeader: ai-learning-cpp/include/ai_learning/learning/active_experimenter_cuda.cuh\n\nREAD first:\n- ai-learning-cpp/include/ai_learning/learning/active_experimenter.hpp - ActiveExperimenter class, hypothesis structure, bayesian_update method\n\nImplement:\n1. kernel_bayesian_update: parallel posterior update for all hypotheses\n   - Each CUDA thread handles one hypothesis\n   - Compute: posterior = prior * likelihood / evidence\n   - Support binary and continuous evidence\n   - Renormalize across all hypotheses (two-pass: compute sum, then divide)\n2. kernel_information_gain: compute expected information gain for experiment design\n   - H(prior) - E[H(posterior)] for each candidate experiment\n   - Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)\n3. cuda_bayesian_batch_update() wrapper: upload hypotheses, run kernels, download results\n\nUse FP32 for probability values (precision important for Bayesian).\nBatch processing: update 100+ hypotheses simultaneously.\nProvide cpu_gpu dispatch.',
  { label: 'cuda-bayesian', phase: 'CUDA Kernels' }
)

log('STDP: ' + (stdpKernel ? 'done' : 'failed'))
log('Activation: ' + (activationKernel ? 'done' : 'failed'))
log('Graph: ' + (graphKernel ? 'done' : 'failed'))
log('Analogy: ' + (analogyKernel ? 'done' : 'failed'))
log('Bayesian: ' + (bayesianKernel ? 'done' : 'failed'))

// Phase 3: Flash Attention
phase('Flash Attention')

const flashAttn = await agent(
  'Active task: .trellis/tasks/06-02-ailearning-phase-7-9\n\n## Phase 8.3: Flash Attention Integration\n\nCreate Flash Attention kernel for embedding training and activation spread.\n\nFile: ai-learning-cpp/src/core/flash_attention_cuda.cu\nHeader: ai-learning-cpp/include/ai_learning/core/flash_attention_cuda.cuh\n\nDesign based on Flash Attention 2 algorithm (Dao, 2023), adapted for C++/CUDA:\n\n1. kernel_flash_attention: tiled attention computation\n   - Input: Q (query), K (key), V (value) matrices [seq_len x dim]\n   - Output: O (output) [seq_len x dim]\n   - Tiled computation: load Q_block and K_block into shared memory, compute attention scores\n   - Online softmax: maintain running max and sum for numerical stability\n   - Output accumulation in registers, write once at end\n\n2. Key parameters:\n   - Block size: 64 or 128 (tune for RTX 4060 8GB)\n   - Support seq_len up to 4096, dim up to 256\n   - Memory: O(N*d) instead of O(N^2) standard attention\n\n3. cuda_flash_attention() wrapper:\n   - Allocate GPU memory for Q, K, V, O\n   - Launch kernel with appropriate grid/block\n   - Synchronize and download result\n\n4. Integration points:\n   - Use in embedding_trainer for attention-weighted context scoring\n   - Use in activation_spread for attention-based message passing\n\nConstraints:\n   - RTX 4060 has 8GB VRAM - keep tile size small\n   - Use FP16 for Q*K^T computation, FP32 for softmax/accumulation\n   - Must handle seq_len not divisible by block_size\n\nReference the existing tensor_ops_cuda.cu for CudaContext and GpuBuffer patterns.',
  { label: 'flash-attention', phase: 'Flash Attention' }
)

log('Flash Attention: ' + (flashAttn ? 'done' : 'failed'))

// Phase 4: Wire everything together
phase('Wire Up')

const wireUp = await agent(
  'Active task: .trellis/tasks/06-02-ailearning-phase-7-9\n\n## Phase 8 Final: Wire CUDA kernels into existing code + update CMakeLists.txt\n\nTasks:\n\n1. Update ai-learning-cpp/CMakeLists.txt:\n   - Add all new .cu files to CUDA sources:\n     - src/core/flash_attention_cuda.cu\n     - src/learning/stdp_learning_cuda.cu\n     - src/reasoning/activation_spread_cuda.cu\n     - src/domain/knowledge/knowledge_graph_cuda.cu\n     - src/learning/analogical_transfer_cuda.cu\n     - src/learning/active_experimenter_cuda.cu\n   - Also add new .cuh includes path\n   - Verify CUDA_ARCHITECTURES is "89"\n\n2. For each of the 5 algorithms, ensure the CPU code has a GPU dispatch path:\n   - stdp_learning.hpp/cpp: add cuda_stpdc_batch_update() call path\n   - activation_spread.hpp/cpp: add cuda_activation_spread() call path\n   - knowledge_graph.hpp/cpp: add cuda_graph_*() call paths\n   - analogical_transfer.hpp/cpp: add cuda_analogical_align() call path\n   - active_experimenter.hpp/cpp: add cuda_bayesian_batch_update() call path\n   \n   Pattern: if (cuda_available() && data_size > threshold) use CUDA, else CPU\n\n3. Check that all .cuh files have proper include guards and forward declarations\n4. Ensure no circular dependencies between headers\n\nREAD all modified files before editing to understand current state.\nDo NOT break existing CPU-only builds (all CUDA calls must be guarded by cuda_available()).\nDo NOT modify test files.',
  { label: 'wire-up', phase: 'Wire Up' }
)

log('Wire up: ' + (wireUp ? 'done' : 'failed'))

return {
  fp16: fp16 ? 'completed' : 'failed',
  stdp: stdpKernel ? 'completed' : 'failed',
  activation: activationKernel ? 'completed' : 'failed',
  graph: graphKernel ? 'completed' : 'failed',
  analogy: analogyKernel ? 'completed' : 'failed',
  bayesian: bayesianKernel ? 'completed' : 'failed',
  flashAttention: flashAttn ? 'completed' : 'failed',
  wireUp: wireUp ? 'completed' : 'failed',
}
