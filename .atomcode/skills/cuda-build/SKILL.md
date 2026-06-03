# CUDA Build & Test

一键构建、测试和基准对比 CUDA/C++ 项目。

## 用法

```
/cuda-build [build|test|bench|all]
```

## 参数

- `build` — 仅编译（默认）
- `test` — 编译 + 运行测试
- `bench` — 编译 + 运行基准测试
- `all` — 编译 + 测试 + 基准（默认）

## 指令

根据参数执行以下步骤：

### 构建步骤

1. 进入 WSL 构建目录：
   ```bash
   cd /mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda
   ```
2. 如果目录不存在，创建并配置：
   ```bash
   mkdir -p build-cuda && cd build-cuda
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```
3. 编译：
   ```bash
   cmake --build . --parallel
   ```

### 测试步骤（test 或 all）

```bash
./ai_learning_tests
```

### 基准步骤（bench 或 all）

```bash
./ai_learning_benchmark
```

### WSL 命令模板

所有命令通过 Windows PowerShell 调用 WSL：
```
powershell -NoProfile -Command "wsl -d Ubuntu-22.04 -- bash -c 'cd /mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda && <命令> 2>&1'"
```

### 环境变量

WSL 中需要设置 CUDA 路径：
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 错误处理

- 如果 CMake 找不到 CUDA，检查 `/usr/local/cuda/bin/nvcc` 是否存在
- 如果链接错误 cuBLAS，确保安装了 `cuda-libraries-12-6` 和 `libcublas-dev-12-6`
- 编译超时：增加 `--parallel` 数量或分步编译
