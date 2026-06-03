/**
 * @file test_phase9_dialog.cpp
 * @brief Phase 9 LLM Dialog verification tests
 *
 * Tests the DialogManager, StubLLMProvider, and OpenAICompatibleProvider:
 *   1. StubLLMProvider returns deterministic responses
 *   2. Multi-turn dialog with context preservation
 *   3. Intent detection (Chinese keywords)
 *   4. Session management (create, list, clear, LRU eviction)
 *   5. Knowledge learning through dialog
 *   6. DialogManager + Learner integration
 *   7. OpenAICompatibleProvider construction (no real API calls)
 *
 * All tests are deterministic, offline, single-process.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "ai_learning/core/learner.hpp"
#include "ai_learning/language/dialog_manager.hpp"
#include "ai_learning/language/llm_provider.hpp"

using namespace ai_learning;
using namespace ai_learning::language;
using namespace ai_learning::core;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::ContainsSubstring;

// ═══════════════════════════════════════════════════════════
// StubLLMProvider
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Dialog: StubLLMProvider name", "[phase9][dialog]") {
    StubLLMProvider stub;
    CHECK(stub.name() == "stub");
}

TEST_CASE("Phase 9 Dialog: StubLLMProvider explain intent", "[phase9][dialog]") {
    StubLLMProvider stub;
    auto reply = stub.complete("请解释什么是光合作用");
    CHECK_FALSE(reply.empty());
    CHECK_THAT(reply, ContainsSubstring("解释"));
}

TEST_CASE("Phase 9 Dialog: StubLLMProvider why intent", "[phase9][dialog]") {
    StubLLMProvider stub;
    auto reply = stub.complete("为什么会下雨");
    CHECK_FALSE(reply.empty());
}

TEST_CASE("Phase 9 Dialog: StubLLMProvider how intent", "[phase9][dialog]") {
    StubLLMProvider stub;
    auto reply = stub.complete("怎么学习物理");
    CHECK_FALSE(reply.empty());
}

TEST_CASE("Phase 9 Dialog: StubLLMProvider teach intent", "[phase9][dialog]") {
    StubLLMProvider stub;
    auto reply = stub.complete("教系统数学知识");
    CHECK_FALSE(reply.empty());
    CHECK_THAT(reply, ContainsSubstring("理解"));
}

TEST_CASE("Phase 9 Dialog: StubLLMProvider default fallback",
          "[phase9][dialog]") {
    StubLLMProvider stub;
    auto reply = stub.complete("这是一段普通文本");
    CHECK_FALSE(reply.empty());
}

TEST_CASE("Phase 9 Dialog: StubLLMProvider with system prompt",
          "[phase9][dialog]") {
    StubLLMProvider stub;
    auto reply = stub.complete("学习新知识", "你是一个AI助手");
    CHECK_FALSE(reply.empty());
}

// ═══════════════════════════════════════════════════════════
// DialogManager — Multi-turn Conversation
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Dialog: Single turn conversation", "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    auto response = mgr.chat("物理学是研究自然规律的科学");

    CHECK_FALSE(response.assistant_message.empty());
    CHECK_FALSE(response.intent.empty());
    CHECK_FALSE(response.learner_action.empty());
}

TEST_CASE("Phase 9 Dialog: Multi-turn context preservation",
          "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    // Turn 1
    mgr.chat("数学是研究数量和结构的学科", "session1");
    // Turn 2
    mgr.chat("化学是研究物质变化的学科", "session1");
    // Turn 3
    auto r3 = mgr.chat("物理是研究力和运动的学科", "session1");

    CHECK_FALSE(r3.assistant_message.empty());

    // History should have 6 entries (3 user + 3 assistant)
    auto history = mgr.get_history("session1");
    CHECK(history.size() == 6);

    // Check history roles alternate
    CHECK(history[0].role == "user");
    CHECK(history[1].role == "assistant");
    CHECK(history[2].role == "user");
    CHECK(history[3].role == "assistant");
    CHECK(history[4].role == "user");
    CHECK(history[5].role == "assistant");
}

TEST_CASE("Phase 9 Dialog: History last_n parameter", "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    mgr.chat("第一轮", "test");
    mgr.chat("第二轮", "test");
    mgr.chat("第三轮", "test");

    auto last2 = mgr.get_history("test", 2);
    CHECK(last2.size() == 2);
    CHECK(last2[0].role == "user");
    CHECK(last2[0].content == "第三轮");
}

// ═══════════════════════════════════════════════════════════
// Intent Detection
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Dialog: Intent teach detected", "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    auto r = mgr.chat("物理学是研究力和运动的科学");
    CHECK(r.intent == "teach");
    CHECK(r.learner_action == "learn");
}

TEST_CASE("Phase 9 Dialog: Intent ask detected", "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    // "为什么" triggers "ask" intent (before "是" triggers "teach")
    auto r = mgr.chat("为什么会下雨");
    CHECK(r.intent == "ask");
    CHECK(r.learner_action == "reason");
}

TEST_CASE("Phase 9 Dialog: Intent explore detected", "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    auto r = mgr.chat("试试做实验验证假设");
    CHECK(r.intent == "explore");
}

TEST_CASE("Phase 9 Dialog: Intent reflect detected", "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    auto r = mgr.chat("思考一下学过的知识");
    CHECK(r.intent == "reflect");
}

// ═══════════════════════════════════════════════════════════
// Session Management
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Dialog: Multiple sessions", "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    mgr.chat("会话A第一条", "sessionA");
    mgr.chat("会话B第一条", "sessionB");

    auto sessions = mgr.list_sessions();
    CHECK(sessions.size() == 2);

    auto hist_a = mgr.get_history("sessionA");
    auto hist_b = mgr.get_history("sessionB");
    CHECK(hist_a.size() == 2);  // user + assistant
    CHECK(hist_b.size() == 2);
}

TEST_CASE("Phase 9 Dialog: Clear session", "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    mgr.chat("有内容的消息", "temp_session");
    CHECK(mgr.list_sessions().size() == 1);

    mgr.clear_session("temp_session");
    CHECK(mgr.list_sessions().empty());

    auto hist = mgr.get_history("temp_session");
    CHECK(hist.empty());
}

TEST_CASE("Phase 9 Dialog: LRU eviction when exceeding max sessions",
          "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    // Create 10 sessions (max)
    for (int i = 0; i < 10; ++i) {
        mgr.chat("消息" + std::to_string(i), "session_" + std::to_string(i));
    }
    CHECK(mgr.list_sessions().size() == 10);

    // 11th session should evict the oldest (session_0)
    mgr.chat("第11个会话", "session_10");

    auto sessions = mgr.list_sessions();
    CHECK(sessions.size() == 10);

    // session_0 should be evicted
    auto hist = mgr.get_history("session_0");
    CHECK(hist.empty());

    // session_10 should exist
    auto hist10 = mgr.get_history("session_10");
    CHECK_FALSE(hist10.empty());
}

TEST_CASE("Phase 9 Dialog: Nonexistent session returns empty history",
          "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    auto hist = mgr.get_history("does_not_exist");
    CHECK(hist.empty());
}

// ═══════════════════════════════════════════════════════════
// Knowledge Learning via Dialog
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Dialog: Teaching dialog adds knowledge",
          "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    auto stats_before = learner.get_stats();

    auto r = mgr.chat("物理学是研究力、能量和运动的自然科学");
    CHECK(r.intent == "teach");
    CHECK(r.learner_action == "learn");

    // Knowledge should have been learned
    CHECK_FALSE(r.knowledge_learned.empty());
}

TEST_CASE("Phase 9 Dialog: Ask dialog triggers reasoning",
          "[phase9][dialog]") {
    LearnerConfig config;
    Learner learner(config);
    StubLLMProvider stub;
    DialogManager mgr(stub, learner);

    // First teach some knowledge
    mgr.chat("重力是地球对物体的吸引力");

    // Then ask about it
    auto r = mgr.chat("为什么物体会掉到地上");
    CHECK(r.intent == "ask");
    CHECK(r.learner_action == "reason");
}

// ═══════════════════════════════════════════════════════════
// OpenAICompatibleProvider Construction
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Dialog: OpenAICompatibleProvider construction",
          "[phase9][dialog]") {
    // Construct with explicit parameters (no real API calls)
    OpenAICompatibleProvider provider(
        "api.example.com",
        "test-key-123",
        "test-model");

    CHECK(provider.name() == "openai_compatible");
}

TEST_CASE("Phase 9 Dialog: OpenAICompatibleProvider default key from env",
          "[phase9][dialog]") {
    // No env var set in test — should fall back to default
    OpenAICompatibleProvider provider(
        "dashscope.aliyuncs.com",
        "",       // empty → use env var → fallback
        "qwen-plus-latest");

    CHECK(provider.name() == "openai_compatible");
}
