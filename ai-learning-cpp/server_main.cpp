/**
 * @file server_main.cpp
 * @brief AILearning REST 服务入口
 *
 * 创建 Learner 实例并启动 HTTP 服务。
 * 用法：ai_learning_server [--port PORT] [--threads N] [--engine pc|mlp|light]
 */

#include "ai_learning/core/learner.hpp"
#include "ai_learning/core/learner_factory.hpp"
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

/// 解析命令行字符串参数
static auto parse_str_arg(int argc, char* argv[],
                          const std::string& flag,
                          const std::string& default_val) -> std::string {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) {
            return argv[i + 1];
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
    auto engine_name = parse_str_arg(argc, argv, "--engine", "pc");

    std::cout << "=== AILearning REST Server ===\n";
    std::cout << "  Port: " << server_config.port << "\n";
    std::cout << "  Threads: " << server_config.threads << "\n";
    std::cout << "  Engine: " << engine_name << "\n\n";

    // 创建学习体（支持引擎选择）
    core::LearnerConfig learner_config;
    learner_config.obs_dim = 128;
    learner_config.action_dim = 8;

    auto learner = core::LearnerFactory::create_with_engine(
        learner_config,
        core::LearnerFactory::make_engine(engine_name, learner_config));

    // 创建并启动服务
    server::LearningServer server(*learner, server_config);
    server.run();

    return 0;
}
