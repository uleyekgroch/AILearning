/**
 * @file society.hpp
 * @brief 多智能体社会 — 管理多个独立 Learner 进程的协作学习
 *
 * 架构：
 *   Society (port 8090)
 *     管理多个 AgentHandle 实例
 *     每个 AgentHandle 通过 REST 连接到一个 Learner 服务
 *     Society 可以生成新进程或连接已有进程
 *
 * 设计原则：
 * - 每个 Agent 是独立进程（子进程）
 * - 通信仅通过 REST HTTP（无共享内存）
 * - 跨平台：Windows 用 CreateProcess，Linux 用 fork+exec
 * - Society 本身也可以暴露为 REST 服务
 */
#pragma once

#include "agent_handle.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ai_learning::society {

/// Agent 信息（查询用）
struct AgentInfo {
    std::string agent_id;
    int port;
    int pid = 0;            // 进程 ID（0 = 未由 Society 生成）
    AgentStatus status = AgentStatus::Stopped;
    int knowledge_count = 0;
};

/// Society 配置
struct SocietyConfig {
    int base_port = 8080;          ///< 第一个 Agent 的端口
    int max_agents = 10;           ///< 最大 Agent 数量
    int spawn_timeout_ms = 5000;   ///< 进程启动超时（毫秒）
    std::string host = "127.0.0.1";
    std::string server_binary;     ///< ai_learning_server 可执行文件路径
};

/// Society 指标
struct SocietyMetrics {
    int total_agents = 0;
    int active_agents = 0;
    double avg_knowledge_per_agent = 0.0;
    double knowledge_diversity = 0.0;        ///< 0~1，知识多样性
    int cultural_transmission_count = 0;      ///< 知识传播次数
    double collective_learning_speed = 0.0;   ///< 集体学习速度
};

/// Society — 多智能体社会管理器
///
/// 管理多个 Learner 进程的生命周期，
/// 并提供社会性学习能力（观察、知识传播等）。
class Society {
public:
    explicit Society(SocietyConfig config = SocietyConfig{});

    ~Society();

    // 禁止拷贝
    Society(const Society&) = delete;
    auto operator=(const Society&) -> Society& = delete;

    // 允许移动
    Society(Society&&) noexcept = default;
    auto operator=(Society&&) noexcept -> Society& = default;

    // ── Agent 管理 ─────────────────────────────────────────────────

    /// 创建新 Agent：启动 Learner 子进程
    /// @param port 指定端口（0 = 自动分配）
    /// @return agent_id
    auto create_agent(int port = 0) -> std::string;

    /// 移除 Agent：终止子进程
    auto remove_agent(const std::string& agent_id) -> bool;

    /// 列出所有 Agent 信息
    auto list_agents() const -> std::vector<AgentInfo>;

    /// 获取指定 Agent 的句柄
    auto get_agent(const std::string& agent_id) -> AgentHandle*;

    // ── 社会性学习 ─────────────────────────────────────────────────

    /// 触发观察：让 observer 观察 model 的行为
    auto trigger_observation(const std::string& observer_id,
                             const std::string& model_id,
                             const std::string& domain)
        -> nlohmann::json;

    /// 广播知识：将一个 Agent 的知识分享给所有其他 Agent
    auto broadcast_knowledge(const std::string& from_id,
                             const std::string& domain)
        -> nlohmann::json;

    /// 获取社会指标
    auto social_metrics() -> SocietyMetrics;

    // ── 生命周期 ───────────────────────────────────────────────────

    /// 关闭所有 Agent 进程
    void shutdown_all();

private:
    /// 生成唯一 agent_id
    auto generate_agent_id_() -> std::string;

    /// 启动子进程（平台相关）
    auto spawn_process_(int port) -> int;

    /// 终止子进程（平台相关）
    void terminate_process_(int pid);

    /// 等待 Agent 健康检查通过
    auto wait_for_health_(AgentHandle& agent) -> bool;

    SocietyConfig config_;

    /// 下一个可分配的端口号
    int next_port_;

    /// Agent 注册表：id → (handle, info)
    struct AgentEntry {
        std::unique_ptr<AgentHandle> handle;
        AgentInfo info;
    };
    std::map<std::string, AgentEntry> agents_;

    /// 统计
    int cultural_transmission_count_ = 0;
    int next_agent_counter_ = 0;
};

}  // namespace ai_learning::society
