运行 ruff 和 mypy 对项目进行代码检查。

## 用法

用户提供可选的路径参数。留空则检查整个项目。

## 步骤

1. 运行 ruff 检查：
   ```bash
   cd D:\mayAi\AILearning_v0527\parallel-learning && python -m ruff check src [路径] 2>&1
   ```

2. 运行 mypy 类型检查：
   ```bash
   cd D:\mayAi\AILearning_v0527\parallel-learning && python -m mypy src [路径] --ignore-missing-imports 2>&1
   ```

3. 汇总报告：
   - ruff 错误数量和类别
   - mypy 错误数量和类别
   - 建议修复方案

## 注意事项
- ruff 用于代码风格和常见错误检查
- mypy 用于类型检查（项目要求使用 type hints）
- 两个工具并行运行，汇总结果
