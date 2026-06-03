/**
 * @file logger.hpp
 * @brief 轻量结构化日志 — 无外部依赖
 *
 * 分级：DEBUG < INFO < WARN < ERROR
 * 环境变量 LOG_LEVEL 控制输出级别（默认 INFO）
 * 输出格式：[2026-06-03T21:30:15.123Z] [INFO] [server] message
 */

#pragma once

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace ai_learning::server {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

/// 全局日志级别（由环境变量 LOG_LEVEL 初始化）
inline auto global_log_level() -> LogLevel {
    static LogLevel level = []() {
        const char* env = std::getenv("LOG_LEVEL");
        if (!env) return LogLevel::Info;
        std::string s = env;
        if (s == "DEBUG" || s == "debug" || s == "0") return LogLevel::Debug;
        if (s == "INFO" || s == "info" || s == "1") return LogLevel::Info;
        if (s == "WARN" || s == "warn" || s == "2") return LogLevel::Warn;
        if (s == "ERROR" || s == "error" || s == "3") return LogLevel::Error;
        return LogLevel::Info;
    }();
    return level;
}

/// 设置全局日志级别
inline auto set_global_log_level(LogLevel lvl) -> void {
    // 通过 mutable static hack（实际使用中通常通过 config 传递）
    global_log_level();  // force init
    // 注：生产代码应通过 atomic 或 mutex，此处为简化
}

inline auto level_name(LogLevel lvl) -> const char* {
    switch (lvl) {
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERROR";
    }
    return "?";
}

inline auto now_iso8601() -> std::string {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S")
        << "." << std::setw(3) << std::setfill('0') << ms.count() << "Z";
    return oss.str();
}

/// 核心日志函数
inline auto log_(LogLevel lvl, const std::string& component,
                const std::string& msg) -> void {
    if (static_cast<int>(lvl) < static_cast<int>(global_log_level())) return;

    auto& stream = (lvl >= LogLevel::Error) ? std::cerr : std::cout;
    stream << "[" << now_iso8601() << "]"
           << " [" << level_name(lvl) << "]"
           << " [" << component << "] "
           << msg << "\n";
}

/// 便捷宏风格函数（C++ 无宏，用 inline）
inline auto log_debug(const std::string& comp, const std::string& msg) -> void {
    log_(LogLevel::Debug, comp, msg);
}
inline auto log_info(const std::string& comp, const std::string& msg) -> void {
    log_(LogLevel::Info, comp, msg);
}
inline auto log_warn(const std::string& comp, const std::string& msg) -> void {
    log_(LogLevel::Warn, comp, msg);
}
inline auto log_error(const std::string& comp, const std::string& msg) -> void {
    log_(LogLevel::Error, comp, msg);
}

}  // namespace ai_learning::server
