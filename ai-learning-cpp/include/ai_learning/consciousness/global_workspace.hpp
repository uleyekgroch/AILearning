/**
 * @file global_workspace.hpp
 * @brief 全局工作空间 — 模拟意识的广播瓶颈 (Global Workspace Theory)
 *
 * 理论基础：
 *   - Baars (1988) Global Workspace Theory (GWT)
 *   - Dehaene et al. (2021) Deep Learning and the Global Workspace Theory
 *
 * 核心机制：
 *   大脑有许多并行、无意识的专用模块（视觉、语言、记忆），但意识是一个狭窄的瓶颈。
 *   只有当某个模块产生高价值或高惊讶（Surprise）信号时，信息才会进入 Workspace，
 *   然后向全局广播。这协调了所有模块解决同一个复杂问题。
 */
#pragma once

#include <string>
#include <vector>
#include <queue>
#include <map>
#include <mutex>
#include <memory>
#include <optional>

namespace ai_learning::consciousness {

/// 进入意识空间的想法或表征
struct WorkspaceThought {
    std::string source_module;       ///< 提交此想法的模块 (e.g., "perception", "memory")
    std::vector<float> latent_state; ///< 想法的神经表征 (潜空间向量)
    std::string symbolic_content;    ///< 符号化内容 (可选，如果已经符号接地)
    double surprise_value = 0.0;     ///< 预测误差 / 惊讶度 / 显著性
    double emotion_arousal = 0.0;    ///< 情绪唤醒度
};

/// 接收意识广播的模块接口
class IWorkspaceObserver {
public:
    virtual ~IWorkspaceObserver() = default;
    virtual void on_broadcast(const WorkspaceThought& thought) = 0;
};

/// 全局工作空间 (意识瓶颈)
class GlobalWorkspace {
public:
    explicit GlobalWorkspace(int capacity = 1);

    /// 注册监听模块
    void register_observer(IWorkspaceObserver* observer);

    /// 无意识模块竞争进入工作空间
    void submit_thought(const WorkspaceThought& thought);

    /// 执行意识融合与广播 (每一个认知 Tick 调用一次)
    auto process_and_broadcast() -> std::optional<WorkspaceThought>;

    /// 获取当前维持在意识中的焦点想法
    [[nodiscard]] auto current_focus() const -> std::optional<WorkspaceThought> {
        return current_focus_;
    }

private:
    int capacity_;
    std::optional<WorkspaceThought> current_focus_;
    std::vector<WorkspaceThought> pending_thoughts_;
    std::vector<IWorkspaceObserver*> observers_;
};

} // namespace ai_learning::consciousness
