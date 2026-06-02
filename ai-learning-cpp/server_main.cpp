/**
 * @file server_main.cpp
 * @brief AILearning REST 服务入口
 *
 * 创建 Learner 实例并启动 HTTP 服务。
 * 用法：ai_learning_server [--port PORT] [--threads N]
 */

#include "ai_learning/core/learner.hpp"
#include "src/server/server.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

/// 解析命令行整数参数
static auto parse_int_arg(int argc, char* argv[],
                          const std::string& flag, int default_val) -> int {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) {
            try {
                return std::stoi(argv[i + 1]);
            } catch (...) {
                std::cerr << "[Warning] Invalid value for " << flag
                          << ", using default: " << default_val << "\n";
                return default_val;
            }
        }
    }
    return default_val;
}

auto main(int argc, char* argv[]) -> int {
    using namespace ai_learning;

    // 解析配置
    server::ServerConfig server_config;
    server_config.port = parse_int_arg(argc, argv, "--port", 8080);
    server_config.threads = parse_int_arg(argc, argv, "--threads", 4);

    std::cout << "=== AILearning REST Server ===\n";
    std::cout << "  Port: " << server_config.port << "\n";
    std::cout << "  Threads: " << server_config.threads << "\n\n";

    // 创建学习体
    core::LearnerConfig learner_config;
    learner_config.obs_dim = 128;
    learner_config.action_dim = 8;

    core::Learner learner(learner_config);

    // 创建并启动服务
    server::LearningServer server(learner, server_config);
    server.run();

    return 0;
}
