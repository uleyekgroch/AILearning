#pragma once
/**
 * @file dialog_manager.hpp
 * @brief 多轮对话管理器 — 连接用户、LLM 和学习系统
 *
 * 职责：
 * 1. 管理多会话对话历史
 * 2. 检测用户意图并路由到对应学习动作
 * 3. 构建包含系统状态的上下文提示词
 *
 * 设计约束：
 * - 单会话最多 20 轮对话
 * - 最多 10 个并发会话
 * - LRU 淘汰超限会话
 */

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace ai_learning::core {
class Learner;
}  // namespace ai_learning::core

namespace ai_learning::language {

class ILLMProvider;

/// 单轮对话记录
struct DialogTurn {
    std::string role;      ///< "user" 或 "assistant"
    std::string content;   ///< 对话内容
    std::string timestamp; ///< ISO 格式时间戳
};

/// 对话响应
struct DialogResponse {
    std::string assistant_message;      ///< 助手回复文本
    std::vector<std::string> knowledge_learned;  ///< 本次学习到的知识点
    std::string learner_action;         ///< 学习动作：learn/reason/explore/reflect
    std::string intent;                 ///< 检测到的意图
};

/// 多轮对话管理器
class DialogManager {
public:
    /// 构造对话管理器
    /// @param llm LLM 提供者引用
    /// @param learner 学习体引用
    explicit DialogManager(ILLMProvider& llm, core::Learner& learner);

    /// 处理用户消息并返回响应
    /// @param user_message 用户输入
    /// @param session_id 会话标识
    auto chat(const std::string& user_message,
              const std::string& session_id = "default")
        -> DialogResponse;

    /// 获取会话历史
    /// @param session_id 会话标识
    /// @param last_n 返回最近 N 轮（默认 20）
    [[nodiscard]] auto get_history(const std::string& session_id,
                                   int last_n = 20) const
        -> std::vector<DialogTurn>;

    /// 清除指定会话
    void clear_session(const std::string& session_id);

    /// 列出所有活跃会话
    [[nodiscard]] auto list_sessions() const -> std::vector<std::string>;

private:
    /// 构建包含学习体状态的系统提示词
    [[nodiscard]] auto build_system_prompt_() const -> std::string;

    /// 基于关键词的意图检测
    [[nodiscard]] static auto detect_intent_(const std::string& user_message)
        -> std::string;

    /// 将最近对话历史格式化为消息列表（供 LLM 使用）
    [[nodiscard]] auto format_history_(const std::string& session_id) const
        -> std::string;

    /// 生成当前时间的 ISO 格式字符串
    [[nodiscard]] static auto now_timestamp_() -> std::string;

    /// 获取或创建会话（带 LRU 淘汰）
    auto get_or_create_session_(const std::string& session_id)
        -> std::vector<DialogTurn>&;

    ILLMProvider& llm_;
    core::Learner& learner_;

    /// 会话存储：session_id → 对话历史
    std::map<std::string, std::vector<DialogTurn>> sessions_;

    /// 会话访问顺序（用于 LRU 淘汰）
    std::vector<std::string> session_order_;

    static constexpr int kMaxTurnsPerSession = 20;
    static constexpr int kMaxSessions = 10;
};

}  // namespace ai_learning::language
