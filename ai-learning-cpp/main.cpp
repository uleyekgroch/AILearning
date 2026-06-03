/**
 * @file main.cpp
 * @brief AI Learning System — C++20 重构版
 *
 * 演示统一学习体的核心循环：
 *   perceive → predict → choose_action → learn → remember → think
 */

#include "ai_learning/core/learner.hpp"

#include <iostream>
#include <map>
#include <vector>

auto main() -> int {
    using namespace ai_learning::core;

    std::cout << "=== AI Learning System (C++20) ===\n\n";

    // 配置学习体
    LearnerConfig config;
    config.obs_dim    = 16;
    config.action_dim = 4;

    Learner learner(config);

    // 1. 从文本学习
    std::cout << "--- 文本学习 ---\n";
    auto r1 = learner.learn_from_text("人工智能是计算机科学的一个分支");
    std::cout << "  提取实体数: " << r1.entities.size() << "\n";
    std::cout << "  提取三元组数: " << r1.triples.size() << "\n";
    std::cout << "  验证通过: " << (r1.verification_passed ? "是" : "否") << "\n";

    auto r2 = learner.learn_from_text("Python是一种编程语言");
    std::cout << "  学习 Python, 实体数: " << r2.entities.size() << "\n";

    auto r3 = learner.learn_from_text("因为下雨所以地面湿了");
    std::cout << "  因果关系数: " << r3.causal_links.size() << "\n";

    // 2. 思考/回答
    std::cout << "\n--- 思考 ---\n";
    auto answer = learner.think("什么是人工智能");
    std::cout << "  Q: 什么是人工智能?\n";
    std::cout << "  A: " << answer << "\n";

    // 3. 感知循环
    std::cout << "\n--- 感知循环 ---\n";
    auto obs = learner.perceive({
        {"visual", std::vector<float>(16, 0.5f)},
        {"auditory", std::vector<float>(4, 0.3f)},
    });
    auto action = learner.choose_action(obs);
    std::cout << "  选择动作: " << action << "\n";

    // 4. 经验学习
    std::cout << "\n--- 经验学习 ---\n";
    auto s  = std::vector<float>(16, 0.5f);
    auto s2 = std::vector<float>(16, 1.0f);
    auto err1 = learner.learn_from_experience(s, 0, s2, 1.0f);
    for (int i = 0; i < 50; ++i) {
        learner.learn_from_experience(s, 0, s2, 1.0f);
    }
    auto err2 = learner.learn_from_experience(s, 0, s2, 1.0f);
    std::cout << "  初始误差: " << err1 << "\n";
    std::cout << "  最终误差: " << err2 << "\n";
    std::cout << "  误差下降: " << (err1 > err2 ? "是" : "否") << "\n";

    // 5. 记忆系统
    std::cout << "\n--- 记忆系统 ---\n";
    learner.remember(s, 0, s2, 1.0f, static_cast<float>(err2));
    learner.remember(s2, 1, s, 0.5f, 0.1f);

    auto recalled = learner.recall(s, 3);
    std::cout << "  检索到记忆: " << recalled.size() << " 条\n";

    auto report = learner.consolidate();
    std::cout << "  巩固结果: ";
    for (const auto& [k, v] : report) {
        std::cout << k << "=" << v << " ";
    }
    std::cout << "\n";

    // 6. 统计概念涌现
    std::cout << "\n--- 统计概念涌现 ---\n";
    for (int i = 0; i < 10; ++i) {
        learner.learn_from_text("数学是研究数量和结构的学科");
        learner.learn_from_text("数学和物理是基础学科");
        learner.learn_from_text("物理学研究自然规律");
    }
    auto sl_stats = learner.observe_text("数学和物理相辅相成");
    std::cout << "  涌现概念数: " << sl_stats["new_concepts"].size() << "\n";

    // 7. 统一推理
    std::cout << "\n--- 统一推理 ---\n";
    auto results = learner.reason("数学");
    std::cout << "  推理结果数: " << results.size() << "\n";
    for (const auto& r : results) {
        std::cout << "  [" << r.method << "] " << r.content
                  << " (conf=" << r.confidence << ")\n";
    }

    // 8. 符号接地与语言发展
    std::cout << "\n--- 符号接地与语言发展 ---\n";
    auto& grounding = learner.knowledge_graph();  // 访问子系统
    (void)grounding;
    // 通过发展阶段评估
    std::map<std::string, double> eval;
    eval["prediction_accuracy"] = 0.8;
    eval["vocabulary_size"] = 10;
    bool advanced = learner.try_advance(eval);
    std::cout << "  发展阶段: " << learner.stage() << "\n";
    std::cout << "  晋升成功: " << (advanced ? "是" : "否") << "\n";

    // 9. 自主进化
    std::cout << "\n--- 自主进化 ---\n";
    auto evolve_result = learner.evolve(1);
    std::cout << "  进化前分数: " << evolve_result["score_before"] << "\n";
    std::cout << "  进化后分数: " << evolve_result["score_after"] << "\n";

    // 10. 海马记忆与睡眠巩固
    std::cout << "\n--- 海马记忆与睡眠巩固 ---\n";
    auto stats_before = learner.get_stats();
    std::cout << "  海马情景数: " << stats_before["hippocampal_episodes"] << "\n";
    std::cout << "  皮层事实数: " << stats_before["cortical_facts"] << "\n";

    // 学习新知识并存入海马
    learner.learn_from_text("量子计算利用量子力学原理进行计算");
    learner.learn_from_text("量子比特是量子计算的基本单位");
    learner.learn_from_text("量子纠缠是量子力学的重要现象");

    auto stats_mid = learner.get_stats();
    std::cout << "  学习后海马情景: " << stats_mid["hippocampal_episodes"] << "\n";

    // 巩固：海马→皮层转移
    auto con_report = learner.consolidate();
    std::cout << "  巩固转移: " << con_report["consolidated"] << "\n";
    std::cout << "  遗忘数: " << con_report["forgotten"] << "\n";

    auto stats_after = learner.get_stats();
    std::cout << "  巩固后皮层事实: " << stats_after["cortical_facts"] << "\n";
    std::cout << "  巩固后海马情景: " << stats_after["hippocampal_episodes"] << "\n";

    std::cout << "\n=== 完成 ===\n";
    return 0;
}
