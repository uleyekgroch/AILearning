/**
 * @file society.cpp
 * @brief 多智能体社会实现 — 进程管理与协作学习
 *
 * 平台支持：
 * - Windows: CreateProcess 创建子进程
 * - Linux/macOS: fork + exec
 */
#include "ai_learning/society/society.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace ai_learning::society {

// ── 构造/析构 ──────────────────────────────────────────────────────

Society::Society(SocietyConfig config)
    : config_(std::move(config))
    , next_port_(config_.base_port) {}

Society::~Society() {
    shutdown_all();
}

// ── Agent 管理 ──────────────────────────────────────────────────────

auto Society::create_agent(int port) -> std::string {
    if (static_cast<int>(agents_.size()) >= config_.max_agents) {
        throw std::runtime_error("max agents reached: " + std::to_string(config_.max_agents));
    }

    auto agent_id = generate_agent_id_();
    int use_port = (port > 0) ? port : next_port_++;

    // 启动子进程
    int pid = spawn_process_(use_port);

    // 创建 Agent 句柄
    auto handle = std::make_unique<AgentHandle>(agent_id, config_.host, use_port);

    // 等待健康检查
    AgentInfo info;
    info.agent_id = agent_id;
    info.port = use_port;
    info.pid = pid;

    if (wait_for_health_(*handle)) {
        info.status = AgentStatus::Active;
    } else {
        info.status = AgentStatus::Stopped;
        std::cerr << "[Society] Agent " << agent_id
                  << " health check failed on port " << use_port << "\n";
    }

    AgentEntry entry{std::move(handle), info};
    agents_.emplace(agent_id, std::move(entry));

    return agent_id;
}

auto Society::remove_agent(const std::string& agent_id) -> bool {
    auto it = agents_.find(agent_id);
    if (it == agents_.end()) {
        return false;
    }

    // 终止子进程
    if (it->second.info.pid > 0) {
        terminate_process_(it->second.info.pid);
    }

    agents_.erase(it);
    return true;
}

auto Society::list_agents() const -> std::vector<AgentInfo> {
    std::vector<AgentInfo> result;
    result.reserve(agents_.size());
    for (const auto& [id, entry] : agents_) {
        result.push_back(entry.info);
    }
    return result;
}

auto Society::get_agent(const std::string& agent_id) -> AgentHandle* {
    auto it = agents_.find(agent_id);
    if (it == agents_.end()) {
        return nullptr;
    }
    return it->second.handle.get();
}

// ── 社会性学习 ──────────────────────────────────────────────────────

auto Society::trigger_observation(const std::string& observer_id,
                                   const std::string& model_id,
                                   const std::string& domain)
    -> nlohmann::json {
    auto* observer = get_agent(observer_id);
    auto* model = get_agent(model_id);

    if (!observer || !model) {
        return {{"error", "agent not found"}};
    }

    // 获取被观察者的统计信息作为行为数据
    auto model_stats = model->get_stats();

    // 构造观察数据
    nlohmann::json observation;
    observation["agent_id"] = model_id;
    observation["action"] = "learning_in_" + domain;
    observation["context"] = "social_observation";
    observation["domain"] = domain;
    observation["outcome_quality"] = 0.8;

    // 让观察者调用 observe-behavior
    auto result = observer->observe_behavior(observation);

    cultural_transmission_count_++;

    return {
        {"observer", observer_id},
        {"model", model_id},
        {"domain", domain},
        {"observation_result", result}
    };
}

auto Society::broadcast_knowledge(const std::string& from_id,
                                   const std::string& domain)
    -> nlohmann::json {
    auto* source = get_agent(from_id);
    if (!source) {
        return {{"error", "source agent not found"}};
    }

    // 获取源 Agent 的统计
    auto source_stats = source->get_stats();

    nlohmann::json results = nlohmann::json::array();
    int success_count = 0;

    for (auto& [id, entry] : agents_) {
        if (id == from_id) {
            continue;
        }

        // 让每个 Agent 观察源 Agent 的学习行为
        nlohmann::json observation;
        observation["agent_id"] = from_id;
        observation["action"] = "knowledge_sharing_in_" + domain;
        observation["context"] = "broadcast";
        observation["domain"] = domain;
        observation["outcome_quality"] = 1.0;

        try {
            auto result = entry.handle->observe_behavior(observation);
            results.push_back({{"agent_id", id}, {"success", true}});
            success_count++;
            cultural_transmission_count_++;
        } catch (const std::exception& e) {
            results.push_back({{"agent_id", id}, {"success", false}, {"error", e.what()}});
        }
    }

    return {
        {"source", from_id},
        {"domain", domain},
        {"recipients", static_cast<int>(agents_.size() - 1)},
        {"successful", success_count},
        {"details", results}
    };
}

auto Society::social_metrics() -> SocietyMetrics {
    SocietyMetrics m;
    m.total_agents = static_cast<int>(agents_.size());
    m.cultural_transmission_count = cultural_transmission_count_;

    double total_knowledge = 0.0;
    std::map<std::string, int> domain_counts;
    int active = 0;

    for (auto& [id, entry] : agents_) {
        if (entry.info.status == AgentStatus::Active) {
            active++;
        }

        try {
            auto stats = entry.handle->get_stats();
            double kc = stats.value("knowledge_count", 0.0);
            entry.info.knowledge_count = static_cast<int>(kc);
            total_knowledge += kc;
        } catch (...) {
            // Agent 不可达
            entry.info.status = AgentStatus::Stopped;
        }
    }

    m.active_agents = active;

    // 平均知识量
    if (m.total_agents > 0) {
        m.avg_knowledge_per_agent = total_knowledge / m.total_agents;
    }

    // 知识多样性：基于各 Agent 知识量标准差（归一化到 0~1）
    if (m.total_agents > 1 && total_knowledge > 0) {
        double mean = total_knowledge / m.total_agents;
        double variance = 0.0;
        for (const auto& [id, entry] : agents_) {
            double diff = entry.info.knowledge_count - mean;
            variance += diff * diff;
        }
        variance /= m.total_agents;
        double stddev = std::sqrt(variance);
        // 归一化：stddev / mean 作为多样性度量，clamp 到 [0, 1]
        m.knowledge_diversity = std::min(1.0, stddev / (mean + 1.0));
    }

    // 集体学习速度：总知识量 / Agent 数量（简化度量）
    if (m.active_agents > 0) {
        m.collective_learning_speed = total_knowledge / m.active_agents;
    }

    return m;
}

// ── 生命周期 ────────────────────────────────────────────────────────

void Society::shutdown_all() {
    for (auto& [id, entry] : agents_) {
        if (entry.info.pid > 0) {
            terminate_process_(entry.info.pid);
        }
    }
    agents_.clear();
}

// ── 内部方法 ────────────────────────────────────────────────────────

auto Society::generate_agent_id_() -> std::string {
    return "agent_" + std::to_string(next_agent_counter_++);
}

auto Society::spawn_process_(int port) -> int {
    std::string binary = config_.server_binary;
    if (binary.empty()) {
        // 尝试自动查找可执行文件
#ifdef _WIN32
        binary = "ai_learning_server.exe";
#else
        binary = "./ai_learning_server";
#endif
    }

#ifdef _WIN32
    // Windows: CreateProcess
    std::string cmd = binary + " --port " + std::to_string(port) + " --threads 2";

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // 创建可修改的命令行字符串
    std::vector<char> cmd_buf(cmd.begin(), cmd.end());
    cmd_buf.push_back('\0');

    BOOL ok = CreateProcessA(
        nullptr,                    // 应用名
        cmd_buf.data(),             // 命令行
        nullptr,                    // 进程安全属性
        nullptr,                    // 线程安全属性
        FALSE,                      // 不继承句柄
        CREATE_NEW_CONSOLE,         // 新控制台窗口
        nullptr,                    // 环境
        nullptr,                    // 当前目录
        &si,
        &pi
    );

    if (!ok) {
        throw std::runtime_error(
            "CreateProcess failed for port " + std::to_string(port) +
            ": error " + std::to_string(GetLastError()));
    }

    int pid = static_cast<int>(pi.dwProcessId);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return pid;
#else
    // Linux/macOS: fork + exec
    pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("fork failed");
    }

    if (pid == 0) {
        // 子进程
        std::string port_str = std::to_string(port);
        execl(binary.c_str(), binary.c_str(),
              "--port", port_str.c_str(),
              "--threads", "2",
              static_cast<char*>(nullptr));
        // exec 失败
        _exit(1);
    }

    return static_cast<int>(pid);
#endif
}

void Society::terminate_process_(int pid) {
#ifdef _WIN32
    HANDLE hProcess = OpenProcess(PROCESS_TERMINATE, FALSE, static_cast<DWORD>(pid));
    if (hProcess) {
        TerminateProcess(hProcess, 0);
        CloseHandle(hProcess);
    }
#else
    kill(static_cast<pid_t>(pid), SIGTERM);
    // 等待一小段时间，如果还没退出则 SIGKILL
    int status = 0;
    pid_t result = waitpid(static_cast<pid_t>(pid), &status, WNOHANG);
    if (result == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        kill(static_cast<pid_t>(pid), SIGKILL);
        waitpid(static_cast<pid_t>(pid), &status, 0);
    }
#endif
}

auto Society::wait_for_health_(AgentHandle& agent) -> bool {
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(config_.spawn_timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        if (agent.health()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

}  // namespace ai_learning::society
