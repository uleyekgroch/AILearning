---
name: project-conventions
description: 项目开发规范（DDD、TDD、SOLID、行数限制等），AtomCode 编辑代码时自动遵循
user_invocable: false
disable_model_invocation: false
---

# AILearning 项目开发规范

本 skill 是 AtomCode 的背景知识，编辑代码时自动遵循以下规范。

## 架构规范（DDD）

- 项目采用领域驱动设计，限界上下文在 `parallel-learning/src/` 下按 `domain/`, `application/`, `infrastructure/`, `interfaces/` 分层
- 聚合根是事务边界，聚合根之间通过 ID 引用，不直接引用对象
- 所有状态变更必须通过领域事件，事件必须不可变

## 代码规范

- 单文件不超过 800 行，函数不超过 50 行，方法不超过 30 行
- 使用 type hints，完整的文档字符串
- 中文注释风格（与项目保持一致）
- 新代码用英文标识符，保留现有中文命名

## 设计原则

- SOLID 原则（单一职责、开闭、里氏替换、接口隔离、依赖倒置）
- DRY（不重复）、KISS（保持简单）、YAGNI（不过度设计）
- 第一性原理思考

## 测试规范（TDD）

- 先写测试后写实现（红灯 → 绿灯 → 重构）
- 单元测试覆盖率 ≥ 80%，总体覆盖率 ≥ 70%
- 测试命名：`test_<功能>_<场景>_<预期结果>`

## 实验脚本规范

- 所有实验脚本在 `mvl/` 目录
- 包含文档字符串、`sys.stdout.reconfigure(encoding='utf-8')`
- `run_experiment()` 返回 Dict，`main()` 打印摘要
- JSON 结果保存到 `mvl/` 目录
