---
name: run
description: 快速运行 mvl 目录下的实验脚本，自动处理编码和路径
---

运行 mvl 目录下的实验脚本。

## 用法

用户提供实验名称（不含 `experiment_` 前缀和 `.py` 后缀）。

## 步骤

1. 如果用户未指定实验名称，列出所有可用实验：
   ```bash
   dir /b mvl\experiment_*.py
   ```

2. 构建命令并运行：
   ```bash
   cd D:\mayAi\AILearning_v0527\mvl && python experiment_{name}.py
   ```

3. 如果运行出错，读取错误信息并帮助诊断。

## 注意事项
- 所有实验脚本都在 `mvl/` 子目录
- 脚本内部已处理 `sys.stdout.reconfigure(encoding='utf-8')`
- 结果 JSON 保存到 `mvl/` 目录
