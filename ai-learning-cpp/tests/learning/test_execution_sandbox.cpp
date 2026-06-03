/**
 * @file test_execution_sandbox.cpp
 * @brief 执行沙箱单元测试
 *
 * 验证：
 *   1. 编译合法 C++ 代码 → 成功
 *   2. 编译非法 C++ 代码 → compile_error 含错误详情
 *   3. 执行简单程序 → 正确 stdout
 *   4. 超时处理
 *   5. 输出截断
 *   6. FeedbackParser 错误分析
 *   7. 性能指标提取
 */

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include "ai_learning/learning/execution_sandbox.hpp"

using namespace ai_learning::learning;

// ═══════════════════════════════════════════════════════════
// 辅助：获取 g++ 完整路径（子进程可能找不到 PATH 中的 g++）
// ═══════════════════════════════════════════════════════════

static std::string find_gpp() {
    // 优先使用 MSYS2 UCRT64 路径
    for (const char* candidate : {
        "D:/msys64/ucrt64/bin/g++.exe",
        "C:/msys64/ucrt64/bin/g++.exe",
        "/d/msys64/ucrt64/bin/g++.exe",
        "/c/msys64/ucrt64/bin/g++.exe",
    }) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    // 回退到 PATH 中的 g++
    return "g++";
}

// ═══════════════════════════════════════════════════════════
// 编译测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("沙箱: 编译合法代码", "[sandbox]") {
    SandboxConfig config;
    config.compiler = find_gpp();
    LocalProcessSandbox sandbox(config);

    std::string valid_code = R"(
#include <iostream>
int main() {
    std::cout << "hello";
    return 0;
}
)";

    auto result = sandbox.compile(valid_code);

    CHECK(result.success);
    CHECK(result.error_type == "none");
    CHECK(result.exit_code == 0);
}

TEST_CASE("沙箱: 编译非法代码", "[sandbox]") {
    SandboxConfig config;
    config.compiler = find_gpp();
    LocalProcessSandbox sandbox(config);

    std::string invalid_code = R"(
int main() {
    undeclared_variable x;
    return 0;
}
)";

    auto result = sandbox.compile(invalid_code);

    CHECK_FALSE(result.success);
    CHECK(result.error_type == "compile_error");
    CHECK_FALSE(result.stderr_output.empty());
}

// ═══════════════════════════════════════════════════════════
// 执行测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("沙箱: 执行简单程序", "[sandbox]") {
    SandboxConfig config;
    config.compiler = find_gpp();
    config.timeout_ms = 5000;
    LocalProcessSandbox sandbox(config);

    std::string code = R"(
#include <iostream>
int main() {
    std::cout << "42";
    return 0;
}
)";

    auto result = sandbox.execute(code, "cpp");

    CHECK(result.success);
    CHECK(result.exit_code == 0);
    CHECK(result.error_type == "none");
    // stdout 应包含 "42"
    CHECK_FALSE(result.stdout_output.find("42") == std::string::npos);
}

TEST_CASE("沙箱: 执行运行时错误程序", "[sandbox]") {
    SandboxConfig config;
    config.compiler = find_gpp();
    config.timeout_ms = 5000;
    LocalProcessSandbox sandbox(config);

    std::string code = R"(
#include <vector>
int main() {
    std::vector<int> v;
    return v.at(999);  // 抛出 out_of_range
}
)";

    auto result = sandbox.execute(code, "cpp");

    CHECK_FALSE(result.success);
    // 退出码非零
    CHECK(result.exit_code != 0);
}

// ═══════════════════════════════════════════════════════════
// 超时测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("沙箱: 超时处理", "[sandbox]") {
    SandboxConfig config;
    config.compiler = find_gpp();
    config.timeout_ms = 500;  // 500ms 超时
    LocalProcessSandbox sandbox(config);

    std::string infinite_code = R"(
int main() {
    while (true) {}
    return 0;
}
)";

    auto result = sandbox.execute(infinite_code, "cpp");

    CHECK_FALSE(result.success);
    CHECK(result.error_type == "timeout");
}

// ═══════════════════════════════════════════════════════════
// 输出截断测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("沙箱: 输出截断", "[sandbox]") {
    SandboxConfig config;
    config.compiler = find_gpp();
    config.timeout_ms = 5000;
    config.max_output_bytes = 50;  // 截断阈值
    LocalProcessSandbox sandbox(config);

    // 输出 200 字节（< 管道缓冲区 64KB，不会死锁），但 > max_output_bytes
    std::string code = R"(
#include <cstdio>
int main() {
    for (int i = 0; i < 10; ++i)
        printf("ABCDEFGHIJABCDEFGHIJ");
    return 0;
}
)";

    auto result = sandbox.execute(code, "cpp");

    CHECK(result.success);
    // 输出应被截断到 max_output_bytes
    CHECK(static_cast<int>(result.stdout_output.size()) <= config.max_output_bytes);
}

// ═══════════════════════════════════════════════════════════
// run_tests 测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("沙箱: run_tests 成功", "[sandbox]") {
    SandboxConfig config;
    config.compiler = find_gpp();
    config.timeout_ms = 5000;
    LocalProcessSandbox sandbox(config);

    std::string test_code = R"(
#include <iostream>
int main() {
    std::cout << "Tests passed: 3/3" << std::endl;
    std::cout << "Time: 100ms" << std::endl;
    return 0;
}
)";

    auto feedback = sandbox.run_tests(test_code);

    CHECK(feedback.compiled);
    CHECK(feedback.ran);
    CHECK_FALSE(feedback.output_summary.empty());
}

TEST_CASE("沙箱: run_tests 编译失败", "[sandbox]") {
    SandboxConfig config;
    config.compiler = find_gpp();
    LocalProcessSandbox sandbox(config);

    std::string bad_code = R"(
int main() {
    bad_syntax_here
}
)";

    auto feedback = sandbox.run_tests(bad_code);

    CHECK_FALSE(feedback.compiled);
    CHECK_FALSE(feedback.ran);
}

// ═══════════════════════════════════════════════════════════
// FeedbackParser 测试
// ═══════════════════════════════════════════════════════════

TEST_CASE("FeedbackParser: 编译错误分析 — undeclared", "[sandbox][feedback]") {
    CompileError err;
    err.file = "test.cpp";
    err.line = 5;
    err.severity = "error";
    err.message = "'x' was not declared in this scope";

    auto analysis = FeedbackParser::analyze_compile_error(err);

    CHECK(analysis.at("category") == "undeclared_identifier");
    CHECK(analysis.at("file") == "test.cpp");
    CHECK_FALSE(analysis.at("suggestion").empty());
}

TEST_CASE("FeedbackParser: 编译错误分析 — syntax", "[sandbox][feedback]") {
    CompileError err;
    err.file = "main.cpp";
    err.line = 3;
    err.severity = "error";
    err.message = "expected ';' before '}' token";

    auto analysis = FeedbackParser::analyze_compile_error(err);

    CHECK(analysis.at("category") == "syntax_error");
}

TEST_CASE("FeedbackParser: 运行时错误分析", "[sandbox][feedback]") {
    auto analysis = FeedbackParser::analyze_runtime_error(
        "terminate called after throwing an instance of 'std::out_of_range'");

    CHECK(analysis.at("category") == "memory_access");
    CHECK_FALSE(analysis.at("suggestion").empty());
}

TEST_CASE("FeedbackParser: 性能指标提取", "[sandbox][feedback]") {
    std::string output = "Tests passed: 8/10\nTime: 250ms\nScore: 0.85\n";

    auto metrics = FeedbackParser::extract_performance_metrics(output);

    CHECK(metrics.count("tests_passed") > 0);
    CHECK(metrics.count("tests_total") > 0);
    CHECK(metrics.count("time_ms") > 0);
    CHECK(metrics.count("score") > 0);
    CHECK(metrics.at("tests_passed") == 8.0);
    CHECK(metrics.at("tests_total") == 10.0);
}

TEST_CASE("FeedbackParser: 空输出返回空指标", "[sandbox][feedback]") {
    auto metrics = FeedbackParser::extract_performance_metrics("");
    CHECK(metrics.empty());
}

// ═══════════════════════════════════════════════════════════
// 沙箱名称
// ═══════════════════════════════════════════════════════════

TEST_CASE("沙箱: name 返回正确名称", "[sandbox]") {
    SandboxConfig config;
    LocalProcessSandbox sandbox(config);
    CHECK(sandbox.name() == "local_process");
}
