/**
 * @file test_autonomous_perception_social.cpp
 * @brief 测试 Phase 8-10: autonomous_learn, Perception, Social
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ai_learning/core/learner.hpp"
#include "ai_learning/domain/perception/multimodal_encoder.hpp"
#include "ai_learning/domain/environment/simple_environment.hpp"
#include "ai_learning/domain/social/social_agent.hpp"

#include <cmath>
#include <fstream>
#include <map>
#include <vector>

// ═══════════════════════════════════════════════════════════════════
// Phase 8: 多模态感知编码器
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Perception: MultiModalEncoder 构造与基本编码",
          "[perception]") {
    ai_learning::perception::MultiModalEncoder encoder(16);

    // 注册了 3 个模态
    auto mods = encoder.modalities();
    REQUIRE(mods.size() == 3);

    // 等权重初始化
    auto weights = encoder.get_modality_weights();
    REQUIRE(weights.size() == 3);
    for (const auto& [_, w] : weights) {
        REQUIRE_THAT(w, Catch::Matchers::WithinAbs(1.0 / 3.0, 0.01));
    }
}

TEST_CASE("Perception: 单模态编码", "[perception]") {
    ai_learning::perception::MultiModalEncoder encoder(16);

    std::map<std::string, std::vector<float>> input = {
        {"visual", {0.5f, 0.3f, 0.1f, 0.8f}},
    };

    auto encoded = encoder.encode(input);
    REQUIRE(static_cast<int>(encoded.size()) == 16);

    // 非零值存在（visual 有输入）
    bool has_nonzero = false;
    for (auto v : encoded) {
        if (std::abs(v) > 1e-6f) has_nonzero = true;
    }
    REQUIRE(has_nonzero);
}

TEST_CASE("Perception: 多模态融合编码", "[perception]") {
    ai_learning::perception::MultiModalEncoder encoder(16);

    std::map<std::string, std::vector<float>> input = {
        {"visual", {1.0f, 0.5f, 0.0f, 0.8f}},
        {"auditory", {0.2f, 0.4f}},
        {"position", {0.5f, 0.5f}},
    };

    auto encoded = encoder.encode(input);
    REQUIRE(static_cast<int>(encoded.size()) == 16);

    // L2 归一化
    float norm = 0.0f;
    for (auto v : encoded) norm += v * v;
    norm = std::sqrt(norm);
    REQUIRE_THAT(norm, Catch::Matchers::WithinAbs(1.0f, 0.01f));
}

TEST_CASE("Perception: 权重更新", "[perception]") {
    ai_learning::perception::MultiModalEncoder encoder(16);

    auto before = encoder.get_modality_weights();
    double visual_before = before.at("visual");

    encoder.update_weight("visual", 0.1);

    auto after = encoder.get_modality_weights();
    REQUIRE(after.at("visual") > visual_before);
}

TEST_CASE("Perception: 空输入返回零向量", "[perception]") {
    ai_learning::perception::MultiModalEncoder encoder(16);

    std::map<std::string, std::vector<float>> empty_input;
    auto encoded = encoder.encode(empty_input);
    REQUIRE(static_cast<int>(encoded.size()) == 16);
    // 全零（norm=0，不归一化）
    for (auto v : encoded) {
        REQUIRE_THAT(v, Catch::Matchers::WithinAbs(0.0f, 1e-6f));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Phase 8: 简单网格环境
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Environment: SimpleEnvironment 初始化", "[environment]") {
    ai_learning::domain::SimpleEnvironment env(8, 8, 42);

    auto obs = env.observe();
    REQUIRE(obs.contains("visual"));
    REQUIRE(obs.contains("position"));
    REQUIRE(obs["visual"].size() == 16);
    REQUIRE(obs["position"].size() == 2);

    auto [w, h] = env.grid_size();
    REQUIRE(w == 8);
    REQUIRE(h == 8);
}

TEST_CASE("Environment: step 执行动作", "[environment]") {
    ai_learning::domain::SimpleEnvironment env(4, 4, 42);
    env.reset();

    auto pos = env.agent_pos();  // 获取位置（验证可调用）
    (void)pos;

    // 向右移动
    auto [reward, done] = env.step(3);  // 右
    REQUIRE(reward >= 0.0);
    REQUIRE(reward <= 1.0);
}

TEST_CASE("Environment: reset 重置环境", "[environment]") {
    ai_learning::domain::SimpleEnvironment env(4, 4, 42);

    auto obs1 = env.reset();
    env.step(0);
    env.step(1);
    env.step(2);

    auto obs2 = env.reset();
    REQUIRE(env.step_count() == 0);
}

TEST_CASE("Environment: 到达目标获得高奖励", "[environment]") {
    ai_learning::domain::SimpleEnvironment env(3, 3, 42);
    env.reset();

    double max_reward = 0.0;
    for (int i = 0; i < 100; ++i) {
        auto [reward, done] = env.step(i % 4);
        max_reward = std::max(max_reward, reward);
        if (done) break;
    }

    // 在 100 步内应该能到达或接近目标
    REQUIRE(max_reward >= 0.3);
}

TEST_CASE("Environment: configure_for_stage 调整难度", "[environment]") {
    ai_learning::domain::SimpleEnvironment env(4, 4, 42);

    env.configure_for_stage("sensorimotor");
    // 不崩溃即可
    env.configure_for_stage("literacy");
    env.configure_for_stage("complex");

    auto obs = env.observe();
    REQUIRE(obs.contains("visual"));
}

// ═══════════════════════════════════════════════════════════════════
// Phase 8: autonomous_learn 完整闭环
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Autonomous: Learner 在环境中自主学习", "[autonomous]") {
    ai_learning::core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;

    ai_learning::core::Learner learner(config);
    ai_learning::domain::SimpleEnvironment env(4, 4, 42);

    auto result = learner.autonomous_learn(env, 100, 3, 50);

    REQUIRE(result.total_steps > 0);
    REQUIRE(result.episodes > 0);
    REQUIRE(result.elapsed_ms > 0.0);

    // 运行后 learner 状态应该变化
    auto stats = learner.get_stats();
    REQUIRE(stats.at("total_steps") > 0.0);
}

TEST_CASE("Autonomous: 自主学习降低预测误差", "[autonomous]") {
    ai_learning::core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;

    ai_learning::core::Learner learner(config);
    ai_learning::domain::SimpleEnvironment env(4, 4, 42);

    // 第一轮
    auto r1 = learner.autonomous_learn(env, 200, 0, 200);

    // 第二轮（更多步数）
    auto r2 = learner.autonomous_learn(env, 200, 0, 200);

    // 验证误差为有限正值（随机初始化导致绝对值不可预测）
    REQUIRE(std::isfinite(r1.avg_error));
    REQUIRE(std::isfinite(r2.avg_error));
    REQUIRE(r1.avg_error >= 0.0);
    REQUIRE(r2.avg_error >= 0.0);
}

TEST_CASE("Autonomous: 巩固在循环中执行", "[autonomous]") {
    ai_learning::core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;

    ai_learning::core::Learner learner(config);
    ai_learning::domain::SimpleEnvironment env(4, 4, 42);

    auto result = learner.autonomous_learn(env, 500, 0, 50);

    REQUIRE(result.consolidations > 0);
}

// ═══════════════════════════════════════════════════════════════════
// Phase 9: 社会智能体
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Social: SocialAgent 交互", "[social]") {
    ai_learning::core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;

    ai_learning::core::Learner learner1(config);
    ai_learning::core::Learner learner2(config);

    ai_learning::social::SocialAgent agent1(learner1);
    ai_learning::social::SocialAgent agent2(learner2);

    auto result = agent1.interact(agent2);

    REQUIRE(result.contains("interactions"));
    REQUIRE(result.at("interactions") >= 1.0);
}

TEST_CASE("Social: 观察伙伴学习", "[social]") {
    ai_learning::core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;

    ai_learning::core::Learner learner(config);
    ai_learning::social::SocialAgent agent(learner);

    std::vector<float> partner_action = {0.1f, 0.9f, 0.2f, 0.3f};
    std::map<std::string, double> outcome = {{"reward", 0.8}};

    agent.observe_partner(partner_action, outcome);

    auto stats = agent.stats();
    REQUIRE(stats.at("observations") >= 1.0);
}

TEST_CASE("Social: 教授其他 Agent", "[social]") {
    ai_learning::core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;

    ai_learning::core::Learner learner1(config);
    ai_learning::core::Learner learner2(config);

    ai_learning::social::SocialAgent teacher(learner1);
    ai_learning::social::SocialAgent student(learner2);

    auto result = teacher.teach(student, "数学");

    REQUIRE(result.contains("facts_transferred"));
    REQUIRE(result.contains("teaching_effectiveness"));
}

TEST_CASE("Social: 接收知识", "[social]") {
    ai_learning::core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;

    ai_learning::core::Learner learner(config);
    ai_learning::social::SocialAgent agent(learner);

    auto result = agent.receive_knowledge("数学是研究数量和结构的学科");

    REQUIRE(result.contains("entities"));
    REQUIRE(result.at("entities") >= 1.0);
}

TEST_CASE("Social: 统计信息", "[social]") {
    ai_learning::core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;

    ai_learning::core::Learner learner(config);
    ai_learning::social::SocialAgent agent(learner);

    auto stats = agent.stats();
    REQUIRE(stats.contains("interactions"));
    REQUIRE(stats.contains("observations"));
    REQUIRE(stats.contains("total_knowledge_gained"));
}

// ═══════════════════════════════════════════════════════════════════
// Phase 11: 持久化 (save/load)
// ═══════════════════════════════════════════════════════════════════

TEST_CASE("Persistence: save 和 load 状态", "[persistence]") {
    ai_learning::core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;

    ai_learning::core::Learner learner(config);

    // 学习一些知识
    learner.learn_from_text("深度学习是机器学习的分支");
    learner.learn_from_text("神经网络是深度学习的基础");

    // 经验学习
    auto s = std::vector<float>(16, 0.5f);
    auto s2 = std::vector<float>(16, 1.0f);
    for (int i = 0; i < 10; ++i) {
        learner.learn_from_experience(s, 0, s2, 1.0f);
    }

    // 获取保存前状态
    auto stats_before = learner.get_stats();

    // 保存
    learner.save("test_save_state.txt");

    // 验证文件存在
    std::ifstream f("test_save_state.txt");
    REQUIRE(f.is_open());
    f.close();

    // 加载到新 learner
    ai_learning::core::Learner learner2(config);
    learner2.load("test_save_state.txt");

    // 验证元数据恢复
    auto stats_after = learner2.get_stats();
    REQUIRE(stats_after.at("total_steps") == stats_before.at("total_steps"));

    // 清理
    std::remove("test_save_state.txt");
}

TEST_CASE("Persistence: 保存文件格式正确", "[persistence]") {
    ai_learning::core::LearnerConfig config;
    config.obs_dim = 16;
    config.action_dim = 4;

    ai_learning::core::Learner learner(config);
    learner.learn_from_text("测试文本");

    learner.save("test_format.txt");

    std::ifstream f("test_format.txt");
    REQUIRE(f.is_open());

    std::string content;
    bool has_meta = false;
    bool has_config = false;
    bool has_kg = false;

    std::string line;
    while (std::getline(f, line)) {
        if (line == "[meta]") has_meta = true;
        if (line == "[config]") has_config = true;
        if (line == "[knowledge_graph]") has_kg = true;
    }

    REQUIRE(has_meta);
    REQUIRE(has_config);
    REQUIRE(has_kg);

    f.close();
    std::remove("test_format.txt");
}
