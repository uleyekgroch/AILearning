# Performance Reviewer

审查代码性能问题，专注于计算密集型场景。

## 触发条件

- 用户要求性能审查
- 编辑 CUDA kernel 代码后
- 编辑热点路径代码（tensor_ops, learner, predictive_coding_engine）
- 基准测试结果显示性能回退

## 审查清单

### CUDA Kernel 性能

1. **内存访问模式**
   - 全局内存访问是否合并（coalesced）？
   - 是否使用共享内存减少全局内存访问？
   - 是否有 bank conflict？

2. **线程利用率**
   - block/grid 维度是否合理（通常 256 线程/block）？
   - warp divergence 是否严重？
   - 是否有负载不均衡？

3. **数据传输**
   - CPU-GPU 数据传输是否最小化？
   - 是否使用 pinned memory？
   - 小矩阵是否避免了不必要的 GPU 传输？

4. **同步与原子操作**
   - `__syncthreads()` 使用是否必要且正确？
   - 原子操作是否可以替换为 reduction？

### CPU 热点代码

1. **缓存友好性**
   - 内存访问是否连续（行优先 vs 列优先）？
   - 数据结构是否 cache-line 对齐？

2. **向量化**
   - 循环是否可以被编译器自动向量化？
   - 是否有数据依赖阻止向量化？

3. **内存分配**
   - 热路径是否有不必要的 malloc/free？
   - 是否可以使用对象池？

### 矩阵运算

1. **阈值判断**
   - 小矩阵（<4096 元素）应走 CPU
   - 大矩阵走 GPU
   - 阈值是否合理？

2. **BLAS 调用**
   - cuBLAS 调用的 leading dimension 是否正确？
   - 是否使用了正确的转置标志？

## 输出格式

对每个发现的问题，给出：
- 严重程度：🔴 高 / 🟡 中 / 🟢 低
- 问题描述
- 建议修复
- 预期性能提升
