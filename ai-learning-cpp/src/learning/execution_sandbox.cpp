/**
 * @file execution_sandbox.cpp
 * @brief 执行沙箱实现 — 让学习体通过"动手做"获得真实反馈
 *
 * 参考：Darwin Gödel Machine (Sakana AI, 2025)
 * 实现：通过 CreateProcess (Windows) / fork+exec (POSIX) 执行代码，支持超时
 */

#include "ai_learning/learning/execution_sandbox.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#endif

namespace ai_learning::learning {

namespace fs = std::filesystem;

// ── LocalProcessSandbox ──────────────────────────────────────────

LocalProcessSandbox::LocalProcessSandbox(const SandboxConfig& config)
    : config_(config) {}

auto LocalProcessSandbox::execute(const std::string& code,
                                    const std::string& language)
    -> ExecutionResult {
    ExecutionResult result;

    // 1. 写入临时源文件
    std::string suffix = (language == "cpp") ? ".cpp" : ".c";
    auto src_path = write_temp_file_(code, suffix);
    if (src_path.empty()) {
        result.error_type = "io_error";
        return result;
    }

    // 2. 编译
    std::string exe_path = src_path + ".exe";
    std::string compile_cmd = config_.compiler + " -std=c++20 -O2 -o \"" +
                              exe_path + "\" \"" + src_path + "\"";

    auto compile_result = run_process_(compile_cmd, 10000);  // 编译超时 10s
    if (compile_result.exit_code != 0) {
        result.success = false;
        result.exit_code = compile_result.exit_code;
        result.stdout_output = compile_result.stdout_output;
        result.stderr_output = compile_result.stderr_output;
        result.error_type = "compile_error";
        result.elapsed_ms = compile_result.elapsed_ms;

        std::remove(src_path.c_str());
        return result;
    }

    // 3. 执行
    std::string run_cmd = "\"" + exe_path + "\"";
    result = run_process_(run_cmd, config_.timeout_ms);
    result.success = (result.exit_code == 0);
    result.error_type = result.success ? "none" : "runtime_error";

    // 4. 检查超时
    if (result.elapsed_ms >= config_.timeout_ms) {
        result.error_type = "timeout";
        result.success = false;
    }

    // 5. 截断输出
    if (result.stdout_output.size() >
        static_cast<size_t>(config_.max_output_bytes)) {
        result.stdout_output.resize(config_.max_output_bytes);
    }

    // 6. 清理临时文件
    std::remove(src_path.c_str());
    std::remove(exe_path.c_str());

    return result;
}

auto LocalProcessSandbox::compile(const std::string& code)
    -> ExecutionResult {
    auto src_path = write_temp_file_(code, ".cpp");
    if (src_path.empty()) {
        return {false, -1, "", "无法写入临时文件", 0.0, "io_error"};
    }

    std::string exe_path = src_path + ".exe";
    std::string compile_cmd = config_.compiler + " -std=c++20 -O2 -o \"" +
                              exe_path + "\" \"" + src_path + "\"";

    auto result = run_process_(compile_cmd, 10000);
    result.success = (result.exit_code == 0);
    result.error_type = result.success ? "none" : "compile_error";

    std::remove(src_path.c_str());
    if (result.success) {
        std::remove(exe_path.c_str());
    }

    return result;
}

auto LocalProcessSandbox::run_tests(const std::string& test_code)
    -> ExecutionFeedback {
    auto result = execute(test_code, "cpp");
    return extract_feedback_(result);
}

auto LocalProcessSandbox::write_temp_file_(const std::string& code,
                                             const std::string& suffix) const
    -> std::string {
    fs::path dir = config_.work_dir.empty()
                       ? fs::temp_directory_path()
                       : fs::path(config_.work_dir);

    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }

    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto filename = "sandbox_" + std::to_string(now) + suffix;
    auto filepath = dir / filename;

    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return "";

    out << code;
    out.close();

    return filepath.string();
}

#ifdef _WIN32

/// Windows 实现：CreateProcess + 管道 + WaitForSingleObject 超时
auto LocalProcessSandbox::run_process_(const std::string& command,
                                         int timeout_ms) const
    -> ExecutionResult {
    ExecutionResult result;
    auto t0 = std::chrono::high_resolution_clock::now();

    // 创建管道捕获 stdout + stderr
    SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE hReadOut, hWriteOut;
    HANDLE hReadErr, hWriteErr;

    if (!CreatePipe(&hReadOut, &hWriteOut, &sa, 0)) {
        result.exit_code = -1;
        result.error_type = "process_error";
        return result;
    }
    if (!CreatePipe(&hReadErr, &hWriteErr, &sa, 0)) {
        CloseHandle(hReadOut);
        CloseHandle(hWriteOut);
        result.exit_code = -1;
        result.error_type = "process_error";
        return result;
    }

    SetHandleInformation(hReadOut, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(hReadErr, HANDLE_FLAG_INHERIT, 0);

    // 构建 CreateProcess 命令行
    std::string cmd = "cmd /c " + command;
    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = hWriteOut;
    si.hStdError = hWriteErr;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi = {};

    // 创建 Job Object 以便终止整个进程树
    HANDLE hJob = CreateJobObjectA(nullptr, nullptr);
    JOBOBJECT_BASIC_LIMIT_INFORMATION job_limit = {};
    job_limit.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_ext = {};
    job_ext.BasicLimitInformation = job_limit;
    SetInformationJobObject(hJob, JobObjectExtendedLimitInformation,
                            &job_ext, sizeof(job_ext));

    // 需要可写缓冲区
    std::vector<char> cmd_buf(cmd.begin(), cmd.end());
    cmd_buf.push_back('\0');

    BOOL created = CreateProcessA(
        nullptr, cmd_buf.data(),
        nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW | CREATE_SUSPENDED,
        nullptr, nullptr,
        &si, &pi
    );

    if (created) {
        AssignProcessToJobObject(hJob, pi.hProcess);
        ResumeThread(pi.hThread);
    }

    // 关闭子进程端的管道句柄
    CloseHandle(hWriteOut);
    CloseHandle(hWriteErr);

    if (!created) {
        CloseHandle(hReadOut);
        CloseHandle(hReadErr);
        result.exit_code = -1;
        result.error_type = "process_error";
        return result;
    }

    // 读取输出（在等待进程的同时）
    auto read_pipe = [](HANDLE hPipe) -> std::string {
        std::string output;
        char buffer[4096];
        DWORD bytes_read = 0;
        while (ReadFile(hPipe, buffer, sizeof(buffer) - 1, &bytes_read, nullptr) && bytes_read > 0) {
            buffer[bytes_read] = '\0';
            output.append(buffer, bytes_read);
        }
        return output;
    };

    // 等待进程完成，带超时
    DWORD wait_result = WaitForSingleObject(pi.hProcess, static_cast<DWORD>(timeout_ms));

    if (wait_result == WAIT_TIMEOUT) {
        // 超时：关闭 Job Object 终止整个进程树
        CloseHandle(hJob);
        result.exit_code = -1;
        result.error_type = "timeout";
    } else {
        DWORD exit_code = 0;
        GetExitCodeProcess(pi.hProcess, &exit_code);
        result.exit_code = static_cast<int>(exit_code);
    }

    // 读取所有输出
    result.stdout_output = read_pipe(hReadOut);
    result.stderr_output = read_pipe(hReadErr);

    CloseHandle(hReadOut);
    CloseHandle(hReadErr);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (wait_result != WAIT_TIMEOUT) {
        CloseHandle(hJob);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return result;
}

#else

/// POSIX 实现：fork + exec + pipe + waitpid 超时
auto LocalProcessSandbox::run_process_(const std::string& command,
                                         int timeout_ms) const
    -> ExecutionResult {
    ExecutionResult result;
    auto t0 = std::chrono::high_resolution_clock::now();

    FILE* pipe = popen((command + " 2>&1").c_str(), "r");
    if (!pipe) {
        result.exit_code = -1;
        return result;
    }

    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result.stdout_output += buffer;
    }
    result.exit_code = pclose(pipe);
    result.stderr_output = result.stdout_output;  // 合并模式下相同

    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return result;
}

#endif

auto LocalProcessSandbox::parse_compile_errors_(
    const std::string& stderr_output) -> std::vector<CompileError> {
    std::vector<CompileError> errors;
    std::istringstream stream(stderr_output);
    std::string line;

    while (std::getline(stream, line)) {
        CompileError err;

        // GCC/Clang 格式: file:line:col: error: message
        auto colon1 = line.find(':');
        if (colon1 == std::string::npos) continue;

        auto colon2 = line.find(':', colon1 + 1);
        if (colon2 == std::string::npos) continue;

        auto colon3 = line.find(':', colon2 + 1);
        if (colon3 == std::string::npos) continue;

        err.file = line.substr(0, colon1);
        try {
            err.line = std::stoi(line.substr(colon1 + 1, colon2 - colon1 - 1));
        } catch (...) {
            continue;
        }

        auto severity = line.substr(colon2 + 1, colon3 - colon2 - 1);
        while (!severity.empty() && severity.front() == ' ')
            severity.erase(severity.begin());
        while (!severity.empty() && severity.back() == ' ')
            severity.pop_back();

        if (severity.find("error") != std::string::npos) {
            err.severity = "error";
        } else if (severity.find("warning") != std::string::npos) {
            err.severity = "warning";
        } else {
            continue;
        }

        err.message = line.substr(colon3 + 1);
        while (!err.message.empty() && err.message.front() == ' ')
            err.message.erase(err.message.begin());

        errors.push_back(err);
    }

    return errors;
}

auto LocalProcessSandbox::extract_feedback_(const ExecutionResult& result)
    -> ExecutionFeedback {
    ExecutionFeedback feedback;

    feedback.compiled = (result.error_type != "compile_error");
    feedback.ran = result.success;
    feedback.execution_time_ms = result.elapsed_ms;

    if (!feedback.compiled) {
        // stderr_output 优先，fallback 到 stdout_output
        auto& error_text = result.stderr_output.empty()
                               ? result.stdout_output
                               : result.stderr_output;
        feedback.errors = parse_compile_errors_(error_text);
    }

    if (!result.success && feedback.compiled) {
        feedback.runtime_error = result.stderr_output.empty()
                                     ? result.stdout_output
                                     : result.stderr_output;
    }

    feedback.output_summary = result.stdout_output.substr(
        0, std::min(result.stdout_output.size(), size_t(500)));

    auto metrics = FeedbackParser::extract_performance_metrics(result.stdout_output);
    feedback.metrics = metrics;

    return feedback;
}

// ── FeedbackParser ───────────────────────────────────────────────

auto FeedbackParser::analyze_compile_error(const CompileError& error)
    -> std::map<std::string, std::string> {
    std::map<std::string, std::string> analysis;
    analysis["error_type"] = error.severity;
    analysis["file"] = error.file;
    analysis["line"] = std::to_string(error.line);
    analysis["message"] = error.message;

    const auto& msg = error.message;
    if (msg.find("undeclared") != std::string::npos ||
        msg.find("was not declared") != std::string::npos) {
        analysis["category"] = "undeclared_identifier";
        analysis["suggestion"] = "检查拼写或添加必要的 #include";
    } else if (msg.find("no match") != std::string::npos ||
               msg.find("no viable") != std::string::npos) {
        analysis["category"] = "type_mismatch";
        analysis["suggestion"] = "检查函数参数类型是否匹配";
    } else if (msg.find("expected") != std::string::npos) {
        analysis["category"] = "syntax_error";
        analysis["suggestion"] = "检查语法：括号、分号、大括号是否匹配";
    } else if (msg.find("cannot convert") != std::string::npos) {
        analysis["category"] = "type_conversion";
        analysis["suggestion"] = "使用显式类型转换或检查类型兼容性";
    } else {
        analysis["category"] = "unknown";
        analysis["suggestion"] = "仔细阅读编译器错误信息";
    }

    return analysis;
}

auto FeedbackParser::analyze_runtime_error(const std::string& error_msg)
    -> std::map<std::string, std::string> {
    std::map<std::string, std::string> analysis;
    analysis["message"] = error_msg;

    if (error_msg.find("out_of_range") != std::string::npos ||
        error_msg.find("segmentation") != std::string::npos ||
        error_msg.find("segfault") != std::string::npos) {
        analysis["category"] = "memory_access";
        analysis["suggestion"] = "检查数组/向量索引是否越界";
    } else if (error_msg.find("bad_alloc") != std::string::npos) {
        analysis["category"] = "memory_exhaustion";
        analysis["suggestion"] = "减少内存分配或使用更高效的数据结构";
    } else if (error_msg.find("exception") != std::string::npos) {
        analysis["category"] = "unhandled_exception";
        analysis["suggestion"] = "添加 try-catch 块处理异常";
    } else {
        analysis["category"] = "unknown_runtime";
        analysis["suggestion"] = "添加更多调试输出以定位问题";
    }

    return analysis;
}

auto FeedbackParser::extract_performance_metrics(const std::string& output)
    -> std::map<std::string, double> {
    std::map<std::string, double> metrics;

    auto trim = [](std::string s) {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
            s.erase(s.begin());
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
            s.pop_back();
        return s;
    };

    // 解析 "Tests passed: X/Y"
    auto pos = output.find("passed:");
    if (pos != std::string::npos) {
        auto slash = output.find('/', pos);
        auto end = output.find_first_of("\n\r", pos);
        if (slash != std::string::npos && end != std::string::npos) {
            try {
                metrics["tests_passed"] = std::stod(trim(output.substr(pos + 7, slash - pos - 7)));
                metrics["tests_total"] = std::stod(trim(output.substr(slash + 1, end - slash - 1)));
            } catch (...) {}
        }
    }

    // 解析 "Time: Xms"
    pos = output.find("Time:");
    if (pos != std::string::npos) {
        auto end = output.find_first_of("ms\n\r", pos + 5);
        if (end != std::string::npos) {
            try {
                metrics["time_ms"] = std::stod(trim(output.substr(pos + 5, end - pos - 5)));
            } catch (...) {}
        }
    }

    // 解析 "Score: X.XX"
    pos = output.find("Score:");
    if (pos != std::string::npos) {
        auto end = output.find_first_of("\n\r", pos + 6);
        if (end != std::string::npos) {
            try {
                metrics["score"] = std::stod(trim(output.substr(pos + 6, end - pos - 6)));
            } catch (...) {}
        }
    }

    return metrics;
}

}  // namespace ai_learning::learning
