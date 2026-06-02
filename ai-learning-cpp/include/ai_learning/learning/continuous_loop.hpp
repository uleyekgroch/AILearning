/**
 * @file continuous_loop.hpp
 * @brief Continuous Online Learning — Loop Engine + Checkpoint Manager
 *
 * ContinuousLearningLoop: background thread that ingests data, learns,
 * consolidates, replays, and checkpoints on a timer-driven cycle.
 *
 * CheckpointManager: save/load learner state with metadata JSON,
 * keep last N checkpoints, recover latest on startup.
 *
 * Design constraints:
 *   - std::filesystem for file I/O
 *   - std::thread + std::atomic for background execution
 *   - Graceful shutdown: stop() waits for current iteration
 *   - Total under 500 lines
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// Forward-declare to avoid heavy include; implementation file includes learner.hpp
namespace ai_learning::core { class Learner; }

namespace ai_learning::learning {

// ── Configuration ────────────────────────────────────────────────

/// Continuous loop configuration
struct ContinuousLoopConfig {
    int checkpoint_interval_seconds  = 300;   ///< Save checkpoint every N s
    int consolidation_interval_seconds = 60;  ///< Consolidate every N s
    int max_iterations               = 0;     ///< 0 = unlimited
    int data_buffer_size             = 100;   ///< Max pending items
    std::string checkpoint_dir       = "./checkpoints";
    bool auto_recover                = true;  ///< Recover latest on start()
};

// ── Status & result types ────────────────────────────────────────

/// Snapshot of the loop's current state (thread-safe readout)
struct LoopStatus {
    bool   running                = false;
    int    iterations             = 0;
    int    data_processed         = 0;
    std::string last_checkpoint_time;
    std::string last_consolidation_time;
    double knowledge_retention    = 1.0;
    double uptime_seconds         = 0.0;
    int    pending_data_count     = 0;
};

/// Result of a single loop iteration
struct LoopStepResult {
    int    learned      = 0;
    bool   consolidated = false;
    bool   checkpointed = false;
    double retention    = 1.0;
    std::vector<std::string> errors;
};

/// Metadata for a single checkpoint
struct CheckpointInfo {
    std::string id;            ///< Unique checkpoint id (timestamp-based)
    std::string tag;           ///< Optional user tag
    std::string timestamp;     ///< ISO-like timestamp string
    int    knowledge_count     = 0;
    int64_t file_size_bytes    = 0;
};

// ── CheckpointManager ────────────────────────────────────────────

/// Manages checkpoint persistence: save, load, list, cleanup.
class CheckpointManager {
public:
    explicit CheckpointManager(const std::string& checkpoint_dir);

    /// Save learner state; returns the checkpoint id.
    auto save(const core::Learner& learner,
              const std::string& tag = "") -> std::string;

    /// Load the most recent checkpoint into learner.
    auto load_latest(core::Learner& learner) -> bool;

    /// Load a specific checkpoint by id.
    auto load(core::Learner& learner,
              const std::string& checkpoint_id) -> bool;

    /// List all checkpoints (newest first).
    auto list_checkpoints() const -> std::vector<CheckpointInfo>;

    /// Delete old checkpoints, keeping only the last N.
    void cleanup(int keep_last = 5);

    /// Get info about the latest checkpoint.
    auto latest() const -> std::optional<CheckpointInfo>;

private:
    std::filesystem::path dir_;

    /// Build a checkpoint id from the current time.
    static auto make_id_() -> std::string;

    /// Write metadata JSON file alongside the learner save file.
    void write_meta_(const std::filesystem::path& meta_path,
                     const std::string& id,
                     const std::string& tag,
                     int knowledge_count,
                     const std::filesystem::path& save_path) const;

    /// Parse a metadata JSON file into CheckpointInfo.
    auto parse_meta_(const std::filesystem::path& meta_path) const
        -> std::optional<CheckpointInfo>;

    /// Ensure the checkpoint directory exists.
    void ensure_dir_();
};

// ── ContinuousLearningLoop ───────────────────────────────────────

/// Background loop: ingest data → learn → consolidate → replay → checkpoint.
class ContinuousLearningLoop {
public:
    explicit ContinuousLearningLoop(core::Learner& learner);

    /// Start the background loop.  If auto_recover is set, loads latest
    /// checkpoint before the first iteration.
    void start(const ContinuousLoopConfig& config = {});

    /// Signal the loop to stop and wait for the thread to finish.
    void stop();

    /// Whether the background thread is currently running.
    [[nodiscard]] bool is_running() const;

    /// Thread-safe snapshot of the loop's status.
    [[nodiscard]] LoopStatus status() const;

    /// Enqueue data for the loop to learn from.
    void feed_data(const std::string& data, const std::string& source);

    /// Force an immediate checkpoint.
    void checkpoint_now();

    /// Attempt to recover from the latest checkpoint in the given dir.
    bool recover(const std::string& checkpoint_dir);

private:
    core::Learner& learner_;
    CheckpointManager ckpt_mgr_;

    // Configuration (immutable after start)
    ContinuousLoopConfig config_;

    // Thread control
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    // Status (guarded by status_mutex_)
    mutable std::mutex status_mutex_;
    LoopStatus status_;

    // Pending data buffer (guarded by data_mutex_)
    mutable std::mutex data_mutex_;
    struct PendingItem { std::string data; std::string source; };
    std::vector<PendingItem> pending_data_;

    // Timing (last operation timestamps)
    std::chrono::steady_clock::time_point last_checkpoint_;
    std::chrono::steady_clock::time_point last_consolidation_;
    std::chrono::steady_clock::time_point start_time_;

    // ── Internal helpers ───────────────────────────────────────

    /// Execute one iteration of the loop.
    auto loop_iteration_() -> LoopStepResult;

    /// Whether enough time has elapsed for consolidation.
    [[nodiscard]] bool should_consolidate_() const;

    /// Whether enough time has elapsed for a checkpoint.
    [[nodiscard]] bool should_checkpoint_() const;

    /// Drain and learn from all pending data; returns count learned.
    auto ingest_pending_data_() -> int;

    /// The main loop body (runs on background thread).
    void run_loop_(ContinuousLoopConfig config);

    /// Update status_ under lock.
    void update_status_(const LoopStepResult& result);

    /// Write a formatted timestamp.
    static auto now_string_() -> std::string;
};

}  // namespace ai_learning::learning
