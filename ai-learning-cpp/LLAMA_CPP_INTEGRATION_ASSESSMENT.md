/**
 * @file LLAMA_CPP_INTEGRATION_ASSESSMENT.md
 * @brief llama.cpp 集成评估报告
 *
 * 评估目标：是否应将 llama.cpp 集成到 ai-learning-cpp 项目中，
 * 以提升语义理解层（embedding、对话、文本推理）的能力。
 */

# llama.cpp 集成评估报告

## 一、执行摘要

**建议：渐进式集成（Phase 1: Embedding → Phase 2: 对话 → Phase 3: 推理辅助）**

llama.cpp 的集成能为项目带来显著的语义理解能力提升，但需要以"可选依赖"方式引入，避免破坏现有轻量级部署场景。核心收益在 embedding 质量（从规则/统计 → 预训练语义）和对话能力（从 stub → 本地 LLM）。

---

## 二、项目现状分析

### 2.1 当前语义理解模块

| 模块 | 当前实现 | 能力边界 |
|------|----------|----------|
| `EmbeddingTrainer` | 从零训练 Skip-gram SGNS | 需要大量语料，冷启动差，语义质量有限 |
| `DistributionalSemantics` | 统计共现矩阵 | 无法捕捉深层语义，只有表层关联 |
| `TextLearner` | 规则模式匹配实体/关系提取 | 中文支持有限，无法处理歧义、隐喻 |
| `DialogManager` | 依赖 `ILLMProvider` 接口 | 当前无真实后端，仅返回 stub |
| `UnifiedReasoningEngine` | 知识图谱符号推理 | 无神经语义推理，无法理解自然语言问题 |
| `Tokenizer` | 简单字符/词分割 | 无子词切分，OOV 严重 |

### 2.2 已预留的扩展点

```cpp
// LearnerConfig 中已预留
bool embedding_learning_enabled = false;  // 总开关
int  embedding_dim = 128;               // 与 obs_dim 对齐

// DialogManager 已抽象
class ILLMProvider {
    virtual auto generate(const std::string& prompt) -> std::string = 0;
};
```

**评估**：项目架构已为 LLM 集成预留了清晰的插入点，集成侵入性可控。

---

## 三、llama.cpp 概述

### 3.1 基本信息

| 属性 | 详情 |
|------|------|
| **许可证** | MIT（代码）+ 模型文件各自许可证 |
| **语言** | C/C++（C API + C++ wrapper） |
| **构建** | CMake，支持 FetchContent 嵌入 |
| **后端** | CPU（AVX/AVX2/NEON）、CUDA、Metal、Vulkan |
| **量化** | Q4_0 / Q4_K_M / Q5_K_M / Q8_0 / FP16 / FP32 |
| **内存** | 7B Q4 ≈ 4GB RAM，3B Q4 ≈ 2GB |

### 3.2 与项目匹配的 API

```cpp
// 1. 文本嵌入（替换 EmbeddingTrainer）
llama_get_embeddings(ctx);           // mean/max pool
llama_get_embeddings_seq(ctx, seq); // 序列级嵌入

// 2. 文本生成（实现 ILLMProvider）
llama_sampler_sample(smpl, ctx, -1); // 采样下一个 token
llama_decode(ctx, batch);            // 前向解码

// 3. 模型加载
llama_model_load_from_file(path, mparams);
llama_init_from_model(model, cparams);
```

---

## 四、集成价值分析

### 4.1 能解决的问题

| 问题 | 当前状态 | llama.cpp 解决后 |
|------|----------|------------------|
| **Embedding 冷启动** | 从零训练，1000+ 文档才有意义 | 加载预训练模型，立即可用 |
| **语义相似度** | 仅词频共现，"king - man + woman ≈ queen" 不可能 | 向量运算直接支持 |
| **实体提取质量** | 规则匹配，"苹果"歧义（水果/公司）无法区分 | 上下文感知提取 |
| **对话能力** | stub 返回固定文本 | 真实多轮对话 |
| **推理辅助** | 纯符号推理，无法理解自然语言问题 | 神经+符号混合推理 |
| **OOV 处理** | 简单字符分割，新词无法表示 | BPE/SentencePiece 子词切分 |

### 4.2 无法解决的问题

| 问题 | 原因 |
|------|------|
| **预测编码引擎性能** | llama.cpp 是 LLM 推理，与 PC/MLP 引擎是不同的抽象层次 |
| **知识图谱自动构建** | 需要额外的信息抽取管道（llama.cpp 只提供文本生成） |
| **在线学习** | llama.cpp 当前不支持 LoRA/QLoRA 在线微调 |

---

## 五、集成方案设计

### 5.1 架构原则

1. **可选依赖**：`AI_LEARNING_WITH_LLAMA_CPP` CMake 选项，默认 OFF
2. **接口隔离**：新增 `IEmbeddingProvider` / `ILLMProvider` 实现
3. **最小侵入**：不修改现有 Learner、PredictiveCodingEngine 核心逻辑
4. **资源懒加载**：模型文件在首次调用时加载，允许运行时失败回退

### 5.2 模块映射

```
ai-learning-cpp
├── include/ai_learning/language/
│   ├── llm_provider.hpp              ← 已有 ILLMProvider
│   ├── llama_cpp_provider.hpp        ← 新增：llama.cpp 实现
│   └── embedding_provider.hpp        ← 新增：嵌入抽象接口
│
├── src/language/
│   ├── llama_cpp_provider.cpp        ← 新增：llama.cpp 封装
│   └── embedding_provider.cpp        ← 新增：统一嵌入层
│
└── CMakeLists.txt
    └── option(AI_LEARNING_WITH_LLAMA_CPP "Enable llama.cpp integration" OFF)
```

### 5.3 三阶段集成路线图

#### Phase 1: Embedding 替换（Week 1-2）

```cpp
// 新增：LlamaCppEmbeddingProvider
class LlamaCppEmbeddingProvider : public IEmbeddingProvider {
public:
    explicit LlamaCppEmbeddingProvider(const std::string& model_path);
    auto embed(const std::string& text) -> std::vector<float> override;
    auto embed_batch(const std::vector<std::string>& texts)
        -> std::vector<std::vector<float>> override;
};
```

**改动点：**
- `DistributionalSemantics`：可选使用 LlamaCppEmbeddingProvider 替代 SGNS
- `TextLearner`：实体相似度计算从共现 → 余弦相似度
- `LearnerConfig`：`embedding_backend = "llama_cpp" | "sgns" | "none"`

#### Phase 2: 对话后端（Week 3-4）

```cpp
// 新增：LlamaCppLLMProvider
class LlamaCppLLMProvider : public ILLMProvider {
public:
    auto generate(const std::string& prompt) -> std::string override;
    auto chat(const std::vector<DialogTurn>& history,
              const std::string& user_msg) -> std::string override;
};
```

**改动点：**
- `DialogManager`：已有 `ILLMProvider&`，直接注入新实现
- `server_main.cpp`：`--llm-model path/to/model.gguf` CLI 参数

#### Phase 3: 推理增强（Month 2）

```cpp
// UnifiedReasoningEngine 增强
auto UnifiedReasoningEngine::neural_reason(
    const std::string& question) const -> ReasoningResult {
    // 当符号推理置信度 < 0.5 时，调用 LLM 进行语义推理
    // 将 LLM 输出解析为结构化的 ReasoningResult
}
```

---

## 六、风险评估

### 6.1 技术风险

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| **构建时间剧增** | 中 | FetchContent 缓存 + 预编译二进制分发 |
| **模型文件过大** | 高 | 默认关闭；推荐 Q4_K_M 量化（7B≈4GB）；支持 3B/1B 小模型 |
| **运行时内存** | 高 | 可选 CPU-only 模式；模型卸载到磁盘（mmap）；提示词缓存 |
| **WSL 兼容性** | 低 | llama.cpp 原生支持 Linux/WSL，无额外风险 |
| **API 稳定性** | 中 | 封装隔离层，llama.cpp 版本升级只影响 provider 实现 |

### 6.2 模型选择建议

| 用途 | 推荐模型 | 大小 | 许可证 |
|------|----------|------|--------|
| **Embedding** | bge-small-zh-v1.5 (Q4) | ~300MB | MIT |
| **Embedding (英文)** | gte-base (Q4) | ~500MB | MIT |
| **对话 (中文)** | Qwen2.5-3B-Instruct (Q4) | ~2GB | Apache 2.0 |
| **对话 (通用)** | Llama-3.2-3B-Instruct (Q4) | ~2GB | Llama 3.2 License |
| **轻量对话** | Phi-3-mini (Q4) | ~2GB | MIT |

### 6.3 许可证风险

- llama.cpp 代码：MIT（安全）
- 模型文件：需根据具体模型选择（Llama 系列需接受 Meta 许可；Qwen/Phi 系列 Apache 2.0/MIT 更宽松）
- **建议**：文档中明确标注模型许可证要求，提供脚本自动下载并展示许可证文本

---

## 七、工作量估算

| 阶段 | 内容 | 估算人日 |
|------|------|----------|
| **Phase 1** | Embedding 接口 + LlamaCppEmbeddingProvider | 3-5 天 |
| **Phase 2** | ILLMProvider 实现 + 对话集成 | 3-5 天 |
| **Phase 3** | 推理增强 + 性能调优 | 5-7 天 |
| **基础设施** | CMake FetchContent、错误回退、文档 | 2-3 天 |
| **测试** | 单元测试 + 集成测试 + 模型兼容性测试 | 3-5 天 |
| **总计** | — | **16-25 天** |

---

## 八、决策建议

### 8.1 立即启动 Phase 1 的理由

1. **架构已就绪**：`ILLMProvider` 接口、`embedding_learning_enabled` 开关都已存在
2. **零破坏承诺**：可选依赖，不开启时现有代码完全不受影响
3. **收益确定**：Embedding 质量从"能用"到"好用"是确定性提升
4. **llama.cpp 成熟度**：已广泛应用于生产（Ollama、koboldcpp 等），C API 稳定

### 8.2 暂缓集成的理由

1. **部署复杂度**：需要额外下载模型文件，不能仅靠 `git clone && cmake && make`
2. **资源门槛**：最小模型也需要 ~2GB 内存，嵌入式/边缘场景不适用
3. **当前不是瓶颈**：系统当前的核心挑战是预测引擎性能和知识表示，而非语义理解

### 8.3 推荐决策

**启动 Phase 1（Embedding 层），暂缓 Phase 2-3。**

理由：
- Embedding 是现有系统的明确短板（从零训练 SGNS 在小型部署中不现实）
- Embedding 层是"只读"使用 llama.cpp（加载模型 → 编码文本），不涉及对话生成的复杂状态管理
- 即使不集成对话后端，高质量的文本嵌入也能立即提升 `DistributionalSemantics` 和 `TextLearner` 的效果

---

## 九、下一步行动

1. **本周**：创建 `feature/llama-embedding` 分支，实现 `LlamaCppEmbeddingProvider` stub
2. **下周**：下载 bge-small-zh-v1.5 Q4 模型，验证嵌入质量和内存占用
3. **月底**：完成 Phase 1 集成，PR 审查，文档更新
4. **Q3**：根据 Phase 1 反馈，决定是否继续 Phase 2（对话）

---

*评估日期：2025-06-03*
*评估人：Devin AI*
*版本：v1.0*
