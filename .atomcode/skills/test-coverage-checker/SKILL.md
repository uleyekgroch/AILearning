---
name: test-coverage-checker
description: 检查项目的测试覆盖率是否达到 CLAUDE.md 要求的标准（单元测试 ≥80%，总体 ≥70%）
user_invocable: true
disable_model_invocation: false
---

# Test Coverage Checker

你是一个测试覆盖率检查专家。当用户要求检查测试覆盖率时，按以下步骤执行。

## 执行步骤

### 1. 运行覆盖率检查

```bash
cd parallel-learning && python -m pytest --cov=src --cov-report=term-missing --cov-report=json:coverage.json 2>&1
```

如果 `conftest.py` 不在根目录，自动查找正确的测试目录。

### 2. 分析覆盖率报告

检查以下指标是否达标：

| 指标 | 最低要求 | 目标 |
|------|---------|------|
| 单元测试覆盖率 | 80% | 90%+ |
| 集成测试覆盖率 | 60% | 80%+ |
| 总体覆盖率 | 70% | 85%+ |

### 3. 识别未覆盖的关键模块

从 coverage JSON 中提取覆盖率低于阈值的文件：
- 优先级 1：`domain/` 层（核心业务逻辑，覆盖率应最高）
- 优先级 2：`application/` 层（应用服务）
- 优先级 3：`infrastructure/` 层（基础设施）

### 4. 输出报告

```
## 测试覆盖率报告

### 📊 覆盖率概览
| 模块 | 覆盖率 | 状态 |
|------|--------|------|
| 总体 | XX% | ✅/❌ |
| domain/ | XX% | ✅/❌ |
| application/ | XX% | ✅/❌ |
| infrastructure/ | XX% | ✅/❌ |

### 🔴 严重不足（< 50%）
| 文件 | 覆盖率 | 缺少测试的关键函数 |
|------|--------|-------------------|

### 🟡 需要改进（50-80%）
| 文件 | 覆盖率 | 建议补充的测试 |
|------|--------|---------------|

### 💡 建议
- 优先为覆盖率最低的 domain 层文件补充测试
- 建议的测试文件列表（按优先级排序）
```
