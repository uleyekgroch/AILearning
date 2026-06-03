# Benchmark Compare

对比 CPU vs GPU 基准性能，生成对比报告。

## 用法

```
/benchmark-compare [quick|full]
```

## 参数

- `quick` — 只跑核心热点（经验学习、自主学习）
- `full` — 跑完整基准（默认）

## 指令

### 步骤 1：运行 GPU 基准

```bash
powershell -NoProfile -Command "wsl -d Ubuntu-22.04 -- bash -c 'export PATH=/usr/local/cuda/bin:$PATH && cd /mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda && ./ai_learning_benchmark 2>&1'"
```

### 步骤 2：运行 CPU 基准

```bash
powershell -NoProfile -Command "wsl -d Ubuntu-22.04 -- bash -c 'cd /mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-wsl && ./ai_learning_benchmark 2>&1'"
```

### 步骤 3：生成对比表格

解析两个基准输出，生成 Markdown 对比表格：

| 测试项 | CPU (ms) | GPU (ms) | 加速比 |
|--------|----------|----------|--------|
| ... | ... | ... | GPU/CPU |

### 步骤 4：分析瓶颈

- 加速比 < 2x 的项目：可能是矩阵太小未触发 GPU 路径
- 加速比 > 10x 的项目：GPU 加速显著
- 无变化的项目：纯 CPU 逻辑（文本处理等）

### 注意事项

- CPU 基准在 `build-wsl/` 目录（纯 CPU 构建）
- GPU 基准在 `build-cuda/` 目录（CUDA 构建）
- 如果 build-wsl 不存在，先用 `cmake -B build-wsl -DCMAKE_BUILD_TYPE=Release && cmake --build build-wsl` 构建
- GPU 首次运行有 CUDA 初始化开销（~100ms），连续运行更准确
