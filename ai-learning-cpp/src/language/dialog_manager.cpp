/**
 * @file dialog_manager.cpp
 * @brief 多轮对话管理器实现
 */

#include "ai_learning/language/dialog_manager.hpp"

#include "ai_learning/core/learner.hpp"
#include "ai_learning/language/llm_provider.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <sstream>
#include <string>

namespace ai_learning::language {

// ═══════════════════════════════════════════════════════════════════
// 构造
// ═══════════════════════════════════════════════════════════════════

DialogManager::DialogManager(ILLMProvider& llm, core::Learner& learner)
    : llm_(llm), learner_(learner) {}

// ═══════════════════════════════════════════════════════════════════
// 公开接口
// ═══════════════════════════════════════════════════════════════════

auto DialogManager::chat(const std::string& user_message,
                         const std::string& session_id) -> DialogResponse {
    // 1. 检测意图
    std::string intent = detect_intent_(user_message);

    // 2. 记录用户消息
    auto& history = get_or_create_session_(session_id);
    history.push_back({"user", user_message, now_timestamp_()});

    // 3. 执行对应的学习动作
    std::vector<std::string> learned;
    std::string action;

    if (intent == "teach") {
        action = "learn";
        auto result = learner_.learn_from_text(user_message);
        if (!result.entities.empty()) {
            learned = result.entities;
        }
    } else if (intent == "ask") {
        action = "reason";
        // 使用统一推理
        learner_.reason(user_message);
    } else if (intent == "explore") {
        action = "explore";
        learner_.design_experiment();
    } else if (intent == "reflect") {
        action = "reflect";
        learner_.meta_reflect();
    } else {
        // general — 事实性内容则学习，否则推理
        action = "learn";
        auto result = learner_.learn_from_text(user_message);
        if (!result.entities.empty()) {
            learned = result.entities;
        }
    }

    // 4. 构建系统提示词
    std::string sys_prompt = build_system_prompt_();

    // 5. 附加历史上下文
    std::string context = format_history_(session_id);
    std::string full_prompt = context.empty()
                                  ? user_message
                                  : context + "\n用户：" + user_message;

    // 6. 调用 LLM
    std::string reply = llm_.complete(full_prompt, sys_prompt);

    // 7. 记录助手响应（截断超长会话）
    history.push_back({"assistant", reply, now_timestamp_()});
    if (static_cast<int>(history.size()) > kMaxTurnsPerSession) {
        int excess = static_cast<int>(history.size()) - kMaxTurnsPerSession;
        history.erase(history.begin(), history.begin() + excess);
    }

    return {reply, learned, action, intent};
}

auto DialogManager::get_history(const std::string& session_id,
                                int last_n) const
    -> std::vector<DialogTurn> {
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return {};
    }
    const auto& turns = it->second;
    int start = std::max(0, static_cast<int>(turns.size()) - last_n);
    return {turns.begin() + start, turns.end()};
}

void DialogManager::clear_session(const std::string& session_id) {
    sessions_.erase(session_id);
    session_order_.erase(
        std::remove(session_order_.begin(), session_order_.end(), session_id),
        session_order_.end());
}

auto DialogManager::list_sessions() const -> std::vector<std::string> {
    std::vector<std::string> result;
    result.reserve(sessions_.size());
    for (const auto& [id, _] : sessions_) {
        result.push_back(id);
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════
// 私有方法
// ═══════════════════════════════════════════════════════════════════

auto DialogManager::build_system_prompt_() const -> std::string {
    auto stats = learner_.get_stats();
    int node_count = 0;
    if (stats.count("knowledge_nodes") > 0) {
        node_count = static_cast<int>(stats.at("knowledge_nodes"));
    }

    // 获取情感状态
    std::string emotion = "平静";
    try {
        auto params = learner_.emotion_modulated_params();
        emotion = params.emotion_label;
    } catch (...) {
        // 情感模块可能未初始化
    }

    std::ostringstream ss;
    ss << "你是一个AI学习系统助手。"
       << "当前发展阶段：" << learner_.stage() << "。"
       << "已掌握" << node_count << "个知识节点。"
       << "情感状态：" << emotion << "。"
       << "你可以帮助用户教系统新知识，或回答关于系统状态的问题。";
    return ss.str();
}

auto DialogManager::detect_intent_(const std::string& msg) -> std::string {
    // 教学意图
    if (msg.find("教") != std::string::npos ||
        msg.find("学习") != std::string::npos ||
        msg.find("是") != std::string::npos ||
        msg.find("关于") != std::string::npos) {
        return "teach";
    }

    // 提问意图
    if (msg.find("为什么") != std::string::npos ||
        msg.find("怎么") != std::string::npos ||
        msg.find("如何") != std::string::npos) {
        return "ask";
    }

    // 探索意图
    if (msg.find("试试") != std::string::npos ||
        msg.find("实验") != std::string::npos) {
        return "explore";
    }

    // 反思意图
    if (msg.find("思考") != std::string::npos ||
        msg.find("反思") != std::string::npos) {
        return "reflect";
    }

    return "general";
}

auto DialogManager::format_history_(const std::string& session_id) const
    -> std::string {
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return "";
    }

    const auto& turns = it->second;
    // 只取最近几轮作为上下文（避免 prompt 过长）
    int start = std::max(0, static_cast<int>(turns.size()) - 6);

    std::ostringstream ss;
    for (int i = start; i < static_cast<int>(turns.size()); ++i) {
        const auto& t = turns[i];
        if (t.role == "user") {
            ss << "用户：" << t.content << "\n";
        } else {
            ss << "助手：" << t.content << "\n";
        }
    }
    return ss.str();
}

auto DialogManager::now_timestamp_() -> std::string {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
    return buf;
}

auto DialogManager::get_or_create_session_(const std::string& session_id)
    -> std::vector<DialogTurn>& {
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        // 更新 LRU 顺序
        session_order_.erase(
            std::remove(session_order_.begin(), session_order_.end(),
                        session_id),
            session_order_.end());
        session_order_.push_back(session_id);
        return it->second;
    }

    // 新会话：如果超限，淘汰最久未访问的
    if (static_cast<int>(sessions_.size()) >= kMaxSessions) {
        if (!session_order_.empty()) {
            std::string evict = session_order_.front();
            sessions_.erase(evict);
            session_order_.erase(session_order_.begin());
        }
    }

    session_order_.push_back(session_id);
    return sessions_[session_id];
}

}  // namespace ai_learning::language
