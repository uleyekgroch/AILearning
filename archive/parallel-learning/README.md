# parallel-learning — 历史实验代码（已废弃）

> **状态：已归档，不再维护**
>
> 本目录包含的历史 Python 代码仅用于记录早期研究过程，不代表当前项目状态。
> 所有生产开发已迁移至 `ai-learning-cpp/`（C++20 实现）。

## 归档时间

2026-06-03

## 说明

- `src/core/learner.py`（5140 行）等 Python 代码已被 C++ 版 `ai-learning-cpp/include/ai_learning/core/learner.hpp`（529 行瘦编排器）完全替代
- 数据资产 `data/wiki_zh/` 已移至项目根目录，不受本归档影响
- 如需查看实验结果或验证历史假设，可参考本目录中的代码
- 请勿基于本目录代码继续开发

## 替代方案

| Python 原模块 | C++ 替代 |
|-------------|---------|
| `src/core/learner.py` | `ai_learning::core::Learner` |
| `src/perception/encoder.py` | `ai_learning::perception::MultiModalEncoder` |
| `src/knowledge/graph.py` | `ai_learning::domain::knowledge::KnowledgeGraph` |
| `src/reasoning/engine.py` | `ai_learning::reasoning::UnifiedReasoningEngine` |
| `server.py` | `ai_learning_server` (Crow REST API) |
