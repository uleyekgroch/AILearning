# AI-Learning-CPP 开发经验教训

## Lesson 1: Windows/WSL 构建环境差异
**问题**: UCRT64 编译的 exe 在 MINGW64 shell 下无法运行 (exit code 127)
**原因**: 不同 MSYS2 子系统的 DLL 路径不互通
**解决**: 统一用 WSL 运行 Linux 构建，或在 Windows 上确保 shell 与编译器子系统一致
**规则**: 构建和运行必须在同一环境

## Lesson 2: CMake 隐式依赖
**问题**: 增量构建掩盖了缺失的库依赖 (nlohmann_json, asio)
**原因**: Windows 增量构建复用缓存，WSL 干净构建暴露了问题
**解决**: 每个 target 显式声明所有依赖 (`target_link_libraries`)
**规则**: 每次改 CMake 后在干净环境验证

## Lesson 3: 内存泄漏隐蔽性
**问题**: EmbeddingTrainer::train() 不释放 corpus_ 和 neg_table_ (百万级 int)
**原因**: 训练后数据不再使用但未清理
**解决**: `clear()` + `shrink_to_fit()` 释放内存
**规则**: batch 操作后检查临时资源释放

## Lesson 4: 单文件膨胀预警
**问题**: server.cpp 膨胀到 1,687 行才被发现超标
**原因**: 路由端点逐个添加，每次只增加 20-30 行
**解决**: 定期审计文件行数，超过 600 行时主动拆分
**规则**: 每次提交前检查 `wc -l`，600 行为预警线

## Lesson 5: 分词策略的语言绑定
**问题**: 硬编码中文 n-gram 规则，换英文完全失效
**原因**: DS/ET 内部 segment_() 只处理中文
**解决**: 提取为独立 Tokenizer 模块，Language 枚举控制策略
**规则**: NLP 相关逻辑必须有语言抽象层

## Lesson 6: C++ 分区实现
**发现**: 同一类的成员函数可以分布在多个 .cpp 文件中
**适用场景**: 文件行数超标但不想引入新类
**限制**: 所有 .cpp 必须看到完整的类定义（#include 同一头文件）
**收益**: 零 API 变化，零调用方影响
