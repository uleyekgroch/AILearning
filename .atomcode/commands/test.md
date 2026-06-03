运行项目的 pytest 测试套件，带覆盖率报告。

## 用法

用户提供可选的路径参数（测试目录或具体测试文件）。留空则运行全部测试。

## 步骤

1. 如果用户未指定路径，运行全部测试：
   ```bash
   cd D:\mayAi\AILearning_v0527\parallel-learning && python -m pytest --cov=src --cov-report=term-missing -v 2>&1
   ```

2. 如果用户指定了路径（如 `tests/test_learner.py`）：
   ```bash
   cd D:\mayAi\AILearning_v0527\parallel-learning && python -m pytest [路径] --cov=src --cov-report=term-missing -v 2>&1
   ```

3. 如果测试失败，读取失败信息并帮助诊断。

## 注意事项
- 使用 `-v` 显示详细输出
- 使用 `--cov=src` 生成覆盖率报告
- 使用 `--cov-report=term-missing` 显示未覆盖的行号
- 如果覆盖率低于 70%，提醒用户注意
