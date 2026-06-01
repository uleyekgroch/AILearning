/**
 * @file execution_sandbox.hpp
 * @brief 执行沙箱 — 让学习体通过"动手做"获得真实反馈
 *
 * 参考：Darwin Gödel Machine (Sakana AI, 2025)
 * 核心闭环：写代码 → 编译/执行 → 解析输出/错误 → 从反馈中学习
 *
 * 能力 1：动手做（Learning by Doing）
 * 人类学习核心：写代码 → 跑 → 看报错 → 改 → 再跑。
 * 本模块为学习体提供安全的代码执行环境。
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::learning {

/// 执行结果
struct ExecutionResult {
    bool success = false;              ///< 是否成功（退出码 0）
    int exit_code = -1;                ///< 进程退出码
    std::string stdout_output;         ///< 标准输出
    std::string stderr_output;         ///< 标准错误
    double elapsed_ms = 0.0;           ///< 执行耗时（毫秒）
    std::string error_type;            ///< 错误类型：compile_error / runtime_error / timeout / none
};

/// 编译错误详情（结构化解析）
struct CompileError {
    std::string file;                  ///< 文件名
    int line = 0;                      ///< 行号
    std::string severity;              ///< error / warning
    std::string message;               ///< 错误消息
    std::string code_snippet;          ///< 出错代码片段
};

/// 执行反馈 — 从执行结果中提取的学习信号
struct ExecutionFeedback {
    bool compiled = false;             ///< 是否编译通过
    bool ran = false;                  ///< 是否运行成功
    std::vector<CompileError> errors;  ///< 编译错误列表
    std::string runtime_error;         ///< 运行时错误
    std::string output_summary;        ///< 输出摘要（前 500 字符）
    double execution_time_ms = 0.0;    ///< 执行时间
    std::map<std::string, double> metrics;  ///< 额外指标（如通过的测试数）
};

/// 执行沙箱配置
struct SandboxConfig {
    int timeout_ms = 5000;             ///< 执行超时（毫秒）
    std::string work_dir;              ///< 工作目录
    std::string compiler = "g++";      ///< 编译器路径
    int max_output_bytes = 10000;      ///< 最大输出字节数
};

/// 执行沙箱接口 — 策略模式，可替换不同实现
class IExecutionSandbox {
public:
    virtual ~IExecutionSandbox() = default;

    /// 编译并执行代码
    virtual auto execute(const std::string& code,
                         const std::string& language = "cpp")
        -> ExecutionResult = 0;

    /// 只编译不执行
    virtual auto compile(const std::string& code)
        -> ExecutionResult = 0;

    /// 运行测试（执行测试代码并解析结果）
    virtual auto run_tests(const std::string& test_code)
        -> ExecutionFeedback = 0;

    /// 获取沙箱名称
    [[nodiscard]] virtual auto name() const -> std::string = 0;
};

/// 本地进程沙箱 — 通过子进程执行代码
class LocalProcessSandbox : public IExecutionSandbox {
public:
    explicit LocalProcessSandbox(const SandboxConfig& config);

    auto execute(const std::string& code,
                 const std::string& language = "cpp")
        -> ExecutionResult override;

    auto compile(const std::string& code)
        -> ExecutionResult override;

    auto run_tests(const std::string& test_code)
        -> ExecutionFeedback override;

    [[nodiscard]] auto name() const -> std::string override {
        return "local_process";
    }

private:
    SandboxConfig config_;

    /// 写代码到临时文件
    auto write_temp_file_(const std::string& code,
                          const std::string& suffix) const
        -> std::string;

    /// 执行子进程并捕获输出（带超时）
    auto run_process_(const std::string& command,
                      int timeout_ms = 10000) const
        -> ExecutionResult;

    /// 解析编译器错误输出为结构化错误列表
    static auto parse_compile_errors_(const std::string& stderr_output)
        -> std::vector<CompileError>;

    /// 从执行结果提取学习反馈
    static auto extract_feedback_(const ExecutionResult& result)
        -> ExecutionFeedback;
};

/// 反馈解析器 — 从编译/运行错误中提取可学习的模式
class FeedbackParser {
public:
    /// 解析编译错误为因果假设
    /// 返回：{错误类型, 建议修复, 相关代码模式}
    static auto analyze_compile_error(const CompileError& error)
        -> std::map<std::string, std::string>;

    /// 解析运行时错误
    static auto analyze_runtime_error(const std::string& error_msg)
        -> std::map<std::string, std::string>;

    /// 从成功执行中提取性能指标
    static auto extract_performance_metrics(const std::string& output)
        -> std::map<std::string, double>;
};

}  // namespace ai_learning::learning
