/**
 * @file route_groups.hpp
 * @brief 路由组注册函数声明 — 将 server.cpp 拆分为多个路由组文件
 *
 * 每个路由组文件实现一个 register_*_routes 自由函数，
 * 由 server.cpp 编排器统一调用。
 *
 * 设计原则：
 * - SharedState 结构体持有所有路由共享的可变状态
 * - 每个函数接收 crow::SimpleApp、Learner 引用、配置和共享状态
 * - 纯结构性拆分，无行为变更
 */
#pragma once

#include "server_config.hpp"

#include "ai_learning/core/learner.hpp"
#include "ai_learning/society/society.hpp"
#include "ai_learning/language/dialog_manager.hpp"
#include "ai_learning/language/llm_provider.hpp"
#include "ai_learning/learning/continuous_loop.hpp"

#include <crow.h>

#include <nlohmann/json.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace ai_learning::server {

/// 异步任务状态
struct AsyncTask {
    std::string task_id;
    std::string status;           // pending / running / completed / failed
    nlohmann::json result;        // 完成后的结果
    std::string error;            // 失败原因
};

/// 路由间共享的可变状态
struct SharedState {
    /// 异步任务管理
    std::mutex tasks_mutex;
    std::map<std::string, AsyncTask> tasks;

    /// Phase 9: 懒初始化组件
    std::unique_ptr<society::Society> society;
    std::unique_ptr<language::DialogManager> dialog;
    std::unique_ptr<learning::ContinuousLearningLoop> continuous_loop;
    std::unique_ptr<language::ILLMProvider> llm_provider;
};

/// 注册系统 + 核心路由（健康检查、统计、阶段、任务、学习、推理、记忆、持久化）
void register_core_routes(
    crow::SimpleApp& app,
    core::Learner& learner,
    const ServerConfig& config,
    SharedState& state);

/// 注册高级认知路由（Phase 3-6：类比、持续学习、抽象、社会观察、情感、顿悟、元认知、实验、整合）
void register_advanced_routes(
    crow::SimpleApp& app,
    core::Learner& learner,
    SharedState& state);

/// 注册社会路由（Phase 9：创建/删除/列出 Agent、观察、广播、指标）
void register_society_routes(
    crow::SimpleApp& app,
    core::Learner& learner,
    SharedState& state);

/// 注册对话路由（Phase 9：聊天、历史、会话管理）
void register_chat_routes(
    crow::SimpleApp& app,
    core::Learner& learner,
    SharedState& state);

/// 注册运行时路由（Phase 9：状态、启动/停止循环、检查点、喂数据、恢复）
void register_runtime_routes(
    crow::SimpleApp& app,
    core::Learner& learner,
    SharedState& state);

/// 注册目标系统路由（Goals：创建、分解、进度、查询）
void register_goals_routes(
    crow::SimpleApp& app,
    core::Learner& learner,
    SharedState& state);

/// 注册 OpenAPI 文档路由（/api/openapi.json, /api/docs）
void register_openapi_routes(
    crow::SimpleApp& app,
    const ServerConfig& config);

/// 生成唯一 task_id
auto generate_task_id() -> std::string;

}  // namespace ai_learning::server
