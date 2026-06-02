/**
 * @file server_config.hpp
 * @brief REST 服务配置 — 服务启动参数
 *
 * 值对象：不可变配置，构造后只读。
 * 与 LearnerConfig 同风格。
 */
#pragma once

#include <string>

namespace ai_learning::server {

/// REST 服务配置（值对象）
struct ServerConfig {
    /// 监听端口
    int port = 8080;

    /// 工作线程数
    int threads = 4;

    /// 是否启用 CORS
    bool cors_enabled = true;

    /// 静态文件目录（空=不提供静态文件）
    std::string static_dir;

    /// 服务版本号
    std::string version = "0.2.0";
};

}  // namespace ai_learning::server
