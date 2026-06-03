/**
 * @file perf_profile.cpp
 * @brief 快速性能剖析 — 找出 learn_from_text() 瓶颈
 */
#include "ai_learning/core/learner.hpp"
#include "ai_learning/learning/tokenizer.hpp"
#include <chrono>
#include <iostream>

using namespace ai_learning::core;
using namespace std::chrono;

int main() {
    // 模拟一篇 wiki 文章的典型长度
    std::string short_text = "人工智能是计算机科学的一个分支。机器学习是人工智能的核心技术。";
    std::string medium_text = std::string(500, '数') + "学是研究数量结构的学科";  // ~1.5KB
    std::string long_text;
    for (int i = 0; i < 100; ++i) {
        long_text += "人工智能是计算机科学的一个重要分支，机器学习是人工智能的核心技术，"
                     "深度学习是机器学习的一个子领域，自然语言处理是人工智能的应用方向。";
    }
    // long_text ~ 7.5KB, ~2500 CJK chars

    auto bench = [](const std::string& label, const std::string& text) {
        LearnerConfig config;
        config.embedding_learning_enabled = true;
        config.embedding_predictive_learning = true;
        config.pc_max_steps_per_text = 20;
        Learner learner(config);

        auto t0 = steady_clock::now();
        learner.learn_from_text(text, "test");
        auto t1 = steady_clock::now();
        double ms = duration<double, std::milli>(t1 - t0).count();

        auto tokens = ai_learning::learning::tokenize(text, config.language);
        std::cout << label << ": " << ms << " ms, "
                  << text.size() << " bytes, "
                  << tokens.size() << " tokens\n";
    };

    // 逐步增大文本，看非线性增长
    bench("short  (~60B)", short_text);

    // 中等：只开统计学习（关掉 embedding 和 PC）
    {
        LearnerConfig config;
        config.embedding_learning_enabled = false;
        config.embedding_predictive_learning = false;
        Learner learner(config);

        auto t0 = steady_clock::now();
        learner.learn_from_text(long_text, "test");
        auto t1 = steady_clock::now();
        double ms = duration<double, std::milli>(t1 - t0).count();
        std::cout << "long (no embed): " << ms << " ms\n";
    }

    // 全开
    bench("long   (~7KB)", long_text);

    // 超长文本（模拟 18KB wiki 文章）
    std::string huge_text;
    for (int i = 0; i < 250; ++i) {
        huge_text += "人工智能是计算机科学的一个重要分支，机器学习是人工智能的核心技术，"
                     "深度学习是机器学习的一个子领域，自然语言处理是人工智能的应用方向。";
    }

    // 只开 PC 不开 DS
    {
        LearnerConfig config;
        config.embedding_learning_enabled = true;
        config.embedding_predictive_learning = true;
        config.pc_max_steps_per_text = 20;
        Learner learner(config);

        auto t0 = steady_clock::now();
        learner.learn_from_text(huge_text, "test");
        auto t1 = steady_clock::now();
        double ms = duration<double, std::milli>(t1 - t0).count();

        auto tokens = ai_learning::learning::tokenize(huge_text, config.language);
        std::cout << "huge   (~18KB): " << ms << " ms, "
                  << huge_text.size() << " bytes, "
                  << tokens.size() << " tokens\n";
    }

    return 0;
}
