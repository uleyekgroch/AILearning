#pragma once
#include "ai_learning/consciousness/global_workspace.hpp"
#include <iostream>

namespace ai_learning::consciousness {

// 日志监听器，用于把 AI 冒出的念头打印到终端
class ConsoleWorkspaceObserver : public IWorkspaceObserver {
public:
    void on_broadcast(const WorkspaceThought& thought) override {
        std::cout << "\n[🧠 GLOBAL WORKSPACE BROADCAST] (" << thought.source_module << ")\n"
                  << "  内容: " << thought.symbolic_content << "\n"
                  << "  惊讶度: " << thought.surprise_value << "\n";
    }
};

}