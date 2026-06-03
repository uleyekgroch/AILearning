# CUDA Code Reviewer

审查 CUDA 代码的正确性和最佳实践。

## 触发条件

- 编辑 `.cu` 或 `.cuh` 文件
- 编辑 `tensor_ops_cuda.cu`
- 修改 CMakeLists.txt 中的 CUDA 配置
- 用户要求 CUDA 代码审查

## 审查清单

### 正确性

1. **内存管理**
   - `cudaMalloc` / `cudaFree` 是否配对？
   - 是否有内存泄漏？
   - 错误时是否正确释放资源（RAII 或 goto cleanup）？

2. **cuBLAS 调用**
   - `cublasCreate` / `cublasDestroy` 是否配对？
   - Leading dimension (lda, ldb, ldc) 是否与矩阵存储格式匹配？
   - 行优先 vs 列优先：cuBLAS 默认列优先，C++ 代码通常行优先，是否正确转换？
   - `CUBLAS_OP_N` / `CUBLAS_OP_T` 是否正确？

3. **错误检查**
   - 每次 CUDA 调用后是否检查返回值？
   - 是否使用 `cudaGetLastError()` + `cudaDeviceSynchronize()` 检查 kernel 错误？
   - cuBLAS 错误是否有有意义的错误消息？

4. **同步**
   - `__syncthreads()` 是否在条件分支内？（未定义行为）
   - 是否有 race condition？
   - Host-Device 同步是否正确（async vs sync 拷贝）？

### 最佳实践

1. **Kernel 设计**
   - 是否处理了边界条件（tail elements）？
   - Grid-Stride Loop 模式是否正确？
   - 是否有足够的寄存器/共享内存使用？

2. **流和并发**
   - 是否使用 CUDA Stream 实现并发？
   - 默认流是否阻塞其他流？

3. **兼容性**
   - SM 版本是否覆盖目标架构？
   - 是否使用了过时的 API？

## 常见 Bug 模式

### 1. cuBLAS 矩阵布局混淆

```cpp
// ❌ 错误：行优先矩阵用 cuBLAS 默认列优先
cublasDgemv(handle, CUBLAS_OP_N, rows, cols, &alpha, mat, rows, vec, 1, &beta, out, 1);

// ✅ 正确：行优先矩阵需要转置参数
// C(row,col) = mat[row*cols + col] → 在 cuBLAS 眼中是 mat^T
cublasDgemv(handle, CUBLAS_OP_T, cols, rows, &alpha, mat, cols, vec, 1, &beta, out, 1);
```

### 2. 边界条件

```cpp
// ❌ 未检查边界
int idx = blockIdx.x * blockDim.x + threadIdx.x;
out[idx] = in[idx] * 2.0;

// ✅ 正确
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) out[idx] = in[idx] * 2.0;
```

### 3. 资源泄漏

```cpp
// ❌ 异常路径泄漏
cudaMalloc(&d_ptr, size);
if (error) return; // d_ptr 泄漏！

// ✅ RAII 或 cleanup
cudaMalloc(&d_ptr, size);
if (error) { cudaFree(d_ptr); return; }
```

## 输出格式

对每个发现的问题：
- 严重程度：🔴 Bug / 🟡 潜在问题 / 🔵 建议
- 文件和行号
- 问题描述
- 修复建议（含代码示例）
