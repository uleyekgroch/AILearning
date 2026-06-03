---
name: run-experiment
description: 运行或创建 mvl 目录下的实验脚本
user_invocable: true
disable_model_invocation: false
arguments:
  - name: experiment
    description: 实验脚本名称（不含 experiment_ 前缀和 .py 后缀），如 language、physics、causal_reasoning。留空则列出所有可用实验。
    required: false
  - name: action
    description: run（运行实验）或 create（创建新实验模板）。默认 run。
    required: false
---

# Run Experiment Skill

## 用户意图
运行、列出或创建 `mvl/` 目录下的实验脚本。

## 执行步骤

### 如果 action=run（默认）

1. 如果未提供 `experiment` 参数：
   ```bash
   dir /b mvl\experiment_*.py
   ```
   列出所有可用实验，提示用户选择。

2. 如果提供了 `experiment`：
   - 构建脚本路径：`mvl/experiment_{experiment}.py`
   - 检查文件是否存在
   - 运行：
   ```bash
   cd mvl && python experiment_{experiment}.py
   ```

### 如果 action=create

1. 读取一个现有实验脚本（如 `experiment_language.py`）了解模板模式
2. 创建新文件 `mvl/experiment_{experiment}.py`，包含：
   - 文档字符串（描述实验目的和核心问题）
   - `sys.stdout.reconfigure(encoding='utf-8')`
   - `run_experiment()` 函数返回 Dict 结果
   - `main()` 函数打印结果摘要
   - JSON 结果保存到 `mvl/` 目录
   - `if __name__ == '__main__': main()` 入口
3. 遵循项目现有的中文注释风格和代码结构

## 注意事项
- 所有实验在 `mvl/` 子目录下
- 结果 JSON 文件也保存在 `mvl/` 下
- 保持与现有实验脚本一致的代码风格（中文注释、类型标注）
