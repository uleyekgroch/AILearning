# 研究发现 — 通用学习AI前沿

## 核心发现

### 1. LLM的"假智能"本质
- LLM通过暴力缩放实现了元学习目标，但方式是"暴力记忆"而非"真正理解"
- Nature HBM 2025: LLM在感觉运动域与人类表示显著偏离
- 统计关联 ≠ 因果理解

### 2. 当前系统的真实差距
- **编码层**：字符级 bag-of-characters，丢失词序、上下文、语义
- **提取层**：硬编码正则，不可学习，中文专用
- **推理层**：图遍历，无逻辑/归纳/反事实推理
- **记忆层**：静态存储，无真正的巩固和遗忘
- **世界模型**：线性预测器，无对象分解，无想象规划
- **模块集成**：40+注册模块，大多未被调用

### 3. 最值得深入的方向
- **Friston主动推理**：统一感知-行动-学习的数学框架
- **对象中心世界模型**（AXIOM）：从向量到对象的认知跃迁
- **多时间尺度学习**（Nested Learning）：解决灾难性遗忘
- **Empowerment驱动探索**：比prediction error更有效的探索策略

## 论文来源

- LeCun (2022) "A Path Towards Autonomous Machine Intelligence"
- Hafner et al. (2025) DreamerV3 (Nature)
- Heins et al. (2025) AXIOM (arXiv:2505.24784)
- Behrouz et al. (2025) Nested Learning (NeurIPS 2025)
- Scholkopf et al. (2021) "Toward Causal Representation Learning"
- Mantiuk et al. (2025) "From Curiosity to Competence"
- Xu et al. (2025) Symbol Grounding (Nature HBM)
- ORBIT (2026) Cross-Episode Meta-RL
