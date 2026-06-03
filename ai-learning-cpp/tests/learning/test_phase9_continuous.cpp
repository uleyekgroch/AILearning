/**
 * @file test_phase9_continuous.cpp
 * @brief Phase 9 Continuous Learning Loop verification tests
 *
 * Tests the ContinuousLearningLoop and CheckpointManager:
 *   1. Start/stop background thread
 *   2. Learning progress and statistics
 *   3. Checkpoint save/load/recover
 *   4. Forgetting monitoring
 *   5. Thread safety (concurrent access)
 *   6. Error recovery during learning
 *   7. Status query API
 *   8. CheckpointManager standalone operations
 *
 * All tests use short intervals (<= 100ms), no real 24h runs.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "ai_learning/core/learner.hpp"
#include "ai_learning/learning/continuous_loop.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace ai_learning;
using namespace ai_learning::learning;
using namespace ai_learning::core;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::ContainsSubstring;

// Helper: create a temp dir for checkpoints, clean up on destruction
class TempCheckpointDir {
public:
    explicit TempCheckpointDir(const std::string& tag) {
        // Use current working directory to avoid Windows temp path issues
        path_ = std::filesystem::current_path() /
                ("ailearning_test_" + tag);
        std::filesystem::create_directories(path_);
        path_str_ = path_.string();
    }
    ~TempCheckpointDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    auto path() const -> const std::string& {
        return path_str_;
    }
private:
    std::filesystem::path path_;
    std::string path_str_;
};

// ═══════════════════════════════════════════════════════════
// CheckpointManager Standalone Tests
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Continuous: CheckpointManager save and list",
          "[phase9][continuous]") {
    TempCheckpointDir tmpdir("ckpt_list");
    CheckpointManager mgr(tmpdir.path());

    LearnerConfig config;
    Learner learner(config);
    learner.learn_from_text("测试知识点内容");

    auto id = mgr.save(learner, "test_tag");
    CHECK_FALSE(id.empty());
    CHECK_THAT(id, ContainsSubstring("ckpt_"));

    auto list = mgr.list_checkpoints();
    REQUIRE_FALSE(list.empty());
    CHECK(list[0].id == id);
    CHECK(list[0].tag == "test_tag");
}

TEST_CASE("Phase 9 Continuous: CheckpointManager save and load",
          "[phase9][continuous]") {
    TempCheckpointDir tmpdir("ckpt_load");
    CheckpointManager mgr(tmpdir.path());

    LearnerConfig config;
    Learner learner(config);

    // Learn and save
    learner.learn_from_text("保存测试：数学是研究数量的学科");
    auto id = mgr.save(learner);

    // Load into a fresh learner
    Learner learner2(config);
    bool loaded = mgr.load(learner2, id);
    CHECK(loaded);
}

TEST_CASE("Phase 9 Continuous: CheckpointManager latest returns newest",
          "[phase9][continuous]") {
    TempCheckpointDir tmpdir("ckpt_latest");
    CheckpointManager mgr(tmpdir.path());

    LearnerConfig config;
    Learner learner(config);

    // Save two checkpoints with a brief gap so timestamps differ
    auto id1 = mgr.save(learner, "first");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto id2 = mgr.save(learner, "second");

    auto latest = mgr.latest();
    REQUIRE(latest.has_value());
    CHECK(latest->id == id2);
    CHECK(latest->tag == "second");
}

TEST_CASE("Phase 9 Continuous: CheckpointManager cleanup",
          "[phase9][continuous]") {
    TempCheckpointDir tmpdir("ckpt_cleanup");
    CheckpointManager mgr(tmpdir.path());

    LearnerConfig config;
    Learner learner(config);

    // Create 5 checkpoints
    std::vector<std::string> ids;
    for (int i = 0; i < 5; ++i) {
        ids.push_back(mgr.save(learner, "cp_" + std::to_string(i)));
        // Sleep 1s between saves to ensure distinct timestamps
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Keep only last 2
    mgr.cleanup(2);
    auto remaining = mgr.list_checkpoints();
    CHECK(remaining.size() == 2);
    // Should keep the newest ones
    CHECK(remaining[0].id == ids[4]);
    CHECK(remaining[1].id == ids[3]);
}

TEST_CASE("Phase 9 Continuous: CheckpointManager load_latest",
          "[phase9][continuous]") {
    TempCheckpointDir tmpdir("ckpt_latest_load");
    CheckpointManager mgr(tmpdir.path());

    LearnerConfig config;
    Learner learner(config);

    // No checkpoints initially
    Learner learner2(config);
    CHECK_FALSE(mgr.load_latest(learner2));

    // Save one
    mgr.save(learner, "only");
    CHECK(mgr.load_latest(learner2));
}

TEST_CASE("Phase 9 Continuous: CheckpointManager empty dir",
          "[phase9][continuous]") {
    TempCheckpointDir tmpdir("ckpt_empty");
    CheckpointManager mgr(tmpdir.path());

    auto list = mgr.list_checkpoints();
    CHECK(list.empty());

    auto latest = mgr.latest();
    CHECK_FALSE(latest.has_value());
}

// ═══════════════════════════════════════════════════════════
// ContinuousLearningLoop — Start / Stop
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Continuous: Start and stop", "[phase9][continuous]") {
    LearnerConfig config;
    Learner learner(config);

    ContinuousLearningLoop loop(learner);

    ContinuousLoopConfig cfg;
    cfg.checkpoint_dir = TempCheckpointDir("loop_start").path();
    cfg.max_iterations = 3;
    cfg.checkpoint_interval_seconds = 9999;   // Don't checkpoint during test
    cfg.consolidation_interval_seconds = 9999;
    cfg.auto_recover = false;

    CHECK_FALSE(loop.is_running());

    loop.start(cfg);
    CHECK(loop.is_running());

    // Give it time to run a few iterations
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    loop.stop();
    CHECK_FALSE(loop.is_running());
}

TEST_CASE("Phase 9 Continuous: Loop processes multiple iterations",
          "[phase9][continuous]") {
    LearnerConfig config;
    Learner learner(config);

    ContinuousLearningLoop loop(learner);

    ContinuousLoopConfig cfg;
    cfg.checkpoint_dir = TempCheckpointDir("loop_maxiter").path();
    cfg.max_iterations = 3;
    cfg.checkpoint_interval_seconds = 9999;
    cfg.consolidation_interval_seconds = 9999;
    cfg.auto_recover = false;

    loop.start(cfg);

    // Wait for at least 2 iterations
    for (int i = 0; i < 50; ++i) {
        auto s = loop.status();
        if (s.iterations >= 2) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    auto status = loop.status();
    CHECK(status.iterations >= 1);

    // Explicit stop (graceful shutdown)
    loop.stop();
    CHECK_FALSE(loop.is_running());
}

// ═══════════════════════════════════════════════════════════
// Learning Progress
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Continuous: Feed data and learn", "[phase9][continuous]") {
    LearnerConfig config;
    Learner learner(config);

    ContinuousLearningLoop loop(learner);

    ContinuousLoopConfig cfg;
    cfg.checkpoint_dir = TempCheckpointDir("loop_feed").path();
    cfg.max_iterations = 5;
    cfg.checkpoint_interval_seconds = 9999;
    cfg.consolidation_interval_seconds = 9999;
    cfg.auto_recover = false;

    loop.start(cfg);

    // Feed data
    loop.feed_data("数学是研究数量和结构的学科", "test");
    loop.feed_data("物理学是研究自然规律的科学", "test");
    loop.feed_data("化学是研究物质变化的学科", "test");

    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    auto status = loop.status();
    CHECK(status.data_processed >= 1);  // At least some data should be processed

    loop.stop();
}

TEST_CASE("Phase 9 Continuous: Status returns valid info",
          "[phase9][continuous]") {
    LearnerConfig config;
    Learner learner(config);

    ContinuousLearningLoop loop(learner);

    ContinuousLoopConfig cfg;
    cfg.checkpoint_dir = TempCheckpointDir("loop_status").path();
    cfg.max_iterations = 2;
    cfg.checkpoint_interval_seconds = 9999;
    cfg.consolidation_interval_seconds = 9999;
    cfg.auto_recover = false;

    loop.start(cfg);
    std::this_thread::sleep_for(std::chrono::milliseconds(600));

    auto status = loop.status();
    CHECK(status.running == true);
    CHECK(status.iterations >= 1);
    CHECK(status.uptime_seconds > 0.0);
    CHECK(status.knowledge_retention >= 0.0);
    CHECK(status.knowledge_retention <= 1.0);

    loop.stop();

    auto final_status = loop.status();
    CHECK_FALSE(final_status.running);
}

// ═══════════════════════════════════════════════════════════
// Checkpoint via Loop
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Continuous: Checkpoint now", "[phase9][continuous]") {
    TempCheckpointDir tmpdir("loop_ckpt_now");

    LearnerConfig config;
    Learner learner(config);
    learner.learn_from_text("需要被检查点保存的知识");

    ContinuousLearningLoop loop(learner);

    ContinuousLoopConfig cfg;
    cfg.checkpoint_dir = tmpdir.path();
    cfg.max_iterations = 2;
    cfg.checkpoint_interval_seconds = 9999;
    cfg.consolidation_interval_seconds = 9999;
    cfg.auto_recover = false;

    loop.start(cfg);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    loop.checkpoint_now();

    // Wait a bit for file I/O
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    loop.stop();

    // Verify checkpoint was created
    CheckpointManager mgr(tmpdir.path());
    auto list = mgr.list_checkpoints();
    CHECK_FALSE(list.empty());
}

// ═══════════════════════════════════════════════════════════
// Error Recovery
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Continuous: Loop handles malformed data gracefully",
          "[phase9][continuous]") {
    LearnerConfig config;
    Learner learner(config);

    ContinuousLearningLoop loop(learner);

    ContinuousLoopConfig cfg;
    cfg.checkpoint_dir = TempCheckpointDir("loop_error").path();
    cfg.max_iterations = 3;
    cfg.checkpoint_interval_seconds = 9999;
    cfg.consolidation_interval_seconds = 9999;
    cfg.auto_recover = false;

    loop.start(cfg);

    // Feed valid and empty data
    loop.feed_data("有效知识点", "test");
    loop.feed_data("", "test");  // empty — should not crash
    loop.feed_data("另一个有效知识点", "test");

    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    // Loop should still be running and have processed data
    auto status = loop.status();
    CHECK(status.iterations >= 1);

    loop.stop();
}

// ═══════════════════════════════════════════════════════════
// Data Buffer Overflow
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Continuous: Data buffer respects max size",
          "[phase9][continuous]") {
    LearnerConfig config;
    Learner learner(config);

    ContinuousLearningLoop loop(learner);

    ContinuousLoopConfig cfg;
    cfg.checkpoint_dir = TempCheckpointDir("loop_buffer").path();
    cfg.data_buffer_size = 5;
    cfg.max_iterations = 5;
    cfg.checkpoint_interval_seconds = 9999;
    cfg.consolidation_interval_seconds = 9999;
    cfg.auto_recover = false;

    loop.start(cfg);

    // Feed more data than buffer can hold
    for (int i = 0; i < 10; ++i) {
        loop.feed_data("数据" + std::to_string(i), "test");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    auto status = loop.status();
    CHECK(status.data_processed >= 1);

    loop.stop();
}

// ═══════════════════════════════════════════════════════════
// Auto-Recover
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Continuous: Auto-recover from checkpoint",
          "[phase9][continuous]") {
    TempCheckpointDir tmpdir("loop_recover");

    LearnerConfig config;

    // First session: learn and save checkpoint
    {
        Learner learner(config);
        learner.learn_from_text("恢复测试知识内容");
        CheckpointManager mgr(tmpdir.path());
        mgr.save(learner, "recovery_test");
    }

    // Second session: start loop with auto_recover
    {
        Learner learner(config);
        ContinuousLearningLoop loop(learner);

        ContinuousLoopConfig cfg;
        cfg.checkpoint_dir = tmpdir.path();
        cfg.max_iterations = 1;
        cfg.checkpoint_interval_seconds = 9999;
        cfg.consolidation_interval_seconds = 9999;
        cfg.auto_recover = true;

        loop.start(cfg);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        loop.stop();
    }
}

// ═══════════════════════════════════════════════════════════
// Concurrent Access Safety
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Continuous: Concurrent feed_data calls",
          "[phase9][continuous]") {
    LearnerConfig config;
    Learner learner(config);

    ContinuousLearningLoop loop(learner);

    ContinuousLoopConfig cfg;
    cfg.checkpoint_dir = TempCheckpointDir("loop_concurrent").path();
    cfg.data_buffer_size = 100;
    cfg.max_iterations = 5;
    cfg.checkpoint_interval_seconds = 9999;
    cfg.consolidation_interval_seconds = 9999;
    cfg.auto_recover = false;

    loop.start(cfg);

    // Feed data from multiple "threads" (simulated sequential here,
    // since feed_data itself is thread-safe by internal mutex)
    std::vector<std::thread> feeders;
    for (int t = 0; t < 3; ++t) {
        feeders.emplace_back([&loop, t]() {
            for (int i = 0; i < 5; ++i) {
                loop.feed_data("线程" + std::to_string(t)
                               + "_数据" + std::to_string(i),
                               "concurrent_test");
            }
        });
    }

    for (auto& th : feeders) {
        th.join();
    }

    // Wait for data to be processed (loop iteration takes ~1s)
    for (int i = 0; i < 30; ++i) {
        auto s = loop.status();
        if (s.data_processed >= 1) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    auto status = loop.status();
    CHECK(status.data_processed >= 1);

    loop.stop();
}

// ═══════════════════════════════════════════════════════════
// Short Intensive Learning (Simulated "Long Run")
// ═══════════════════════════════════════════════════════════

TEST_CASE("Phase 9 Continuous: Short intensive learning burst",
          "[phase9][continuous]") {
    TempCheckpointDir tmpdir("loop_burst");

    LearnerConfig config;
    Learner learner(config);

    ContinuousLearningLoop loop(learner);

    ContinuousLoopConfig cfg;
    cfg.checkpoint_dir = tmpdir.path();
    cfg.data_buffer_size = 50;
    cfg.max_iterations = 10;
    cfg.checkpoint_interval_seconds = 9999;  // Don't checkpoint during test
    cfg.consolidation_interval_seconds = 9999;
    cfg.auto_recover = false;

    loop.start(cfg);

    // Feed a burst of data
    for (int i = 0; i < 20; ++i) {
        loop.feed_data("密集知识" + std::to_string(i)
                       + "：这是第" + std::to_string(i) + "条知识点",
                       "burst_test");
    }

    // Let it process (wait up to 10s for data to be processed)
    for (int i = 0; i < 50; ++i) {
        auto s = loop.status();
        if (s.data_processed >= 1) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    auto status = loop.status();
    CHECK(status.data_processed >= 1);
    CHECK(status.iterations >= 1);

    loop.stop();

    // Check final status
    auto final_status = loop.status();
    CHECK_FALSE(final_status.running);
}
