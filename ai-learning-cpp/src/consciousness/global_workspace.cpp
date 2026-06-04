/**
 * @file global_workspace.cpp
 * @brief 全局工作空间实现
 */

#include "ai_learning/consciousness/global_workspace.hpp"
#include <algorithm>

namespace ai_learning::consciousness {

GlobalWorkspace::GlobalWorkspace(int capacity)
    : capacity_(capacity) {}

void GlobalWorkspace::register_observer(IWorkspaceObserver* observer) {
    if (observer) {
        observers_.push_back(observer);
    }
}

void GlobalWorkspace::submit_thought(const WorkspaceThought& thought) {
    // 门控机制：如果不够惊讶，且情绪不强烈，无法进入意识队列
    if (thought.surprise_value < 0.1 && thought.emotion_arousal < 0.2) {
        return; // 无意识处理过滤掉
    }
    pending_thoughts_.push_back(thought);
}

auto GlobalWorkspace::process_and_broadcast() -> std::optional<WorkspaceThought> {
    if (pending_thoughts_.empty()) {
        // 意识逐渐消退（或维持前一个念头但显著性降低）
        if (current_focus_) {
            current_focus_->surprise_value *= 0.8;
            current_focus_->emotion_arousal *= 0.9;
            if (current_focus_->surprise_value < 0.05) {
                current_focus_ = std::nullopt;
            }
        }
        return current_focus_;
    }

    // 竞争机制：惊讶度或情绪最强烈的念头夺取意识焦点 (Winner-takes-all)
    auto winner_it = std::max_element(
        pending_thoughts_.begin(), pending_thoughts_.end(),
        [](const WorkspaceThought& a, const WorkspaceThought& b) {
            return (a.surprise_value + a.emotion_arousal) < 
                   (b.surprise_value + b.emotion_arousal);
        });

    current_focus_ = *winner_it;
    pending_thoughts_.clear(); // 其他念头被抑制 (掩蔽效应)

    // 全局广播 (Global Broadcast)
    for (auto* obs : observers_) {
        obs->on_broadcast(*current_focus_);
    }

    return current_focus_;
}

} // namespace ai_learning::consciousness
