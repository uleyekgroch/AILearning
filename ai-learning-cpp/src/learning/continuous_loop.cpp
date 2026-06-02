/**
 * @file continuous_loop.cpp
 * @brief Continuous Online Learning — Loop Engine + Checkpoint Manager
 */

#include "ai_learning/learning/continuous_loop.hpp"
#include "ai_learning/core/learner.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace ai_learning::learning {

// =====================================================================
//  CheckpointManager
// =====================================================================

CheckpointManager::CheckpointManager(const std::string& checkpoint_dir)
    : dir_(checkpoint_dir)
{
    ensure_dir_();
}

void CheckpointManager::ensure_dir_()
{
    if (!std::filesystem::exists(dir_)) {
        std::filesystem::create_directories(dir_);
    }
}

auto CheckpointManager::make_id_() -> std::string
{
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    // Windows: use localtime_s, POSIX: localtime_r — use std::strftime
    std::tm buf{};
#ifdef _WIN32
    localtime_s(&buf, &t);
#else
    localtime_r(&t, &buf);
#endif
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &buf);
    return "ckpt_" + std::string(ts);
}

void CheckpointManager::write_meta_(
    const std::filesystem::path& meta_path,
    const std::string& id,
    const std::string& tag,
    int knowledge_count,
    const std::filesystem::path& save_path) const
{
    std::ofstream out(meta_path);
    if (!out.is_open()) return;

    // Minimal JSON — no external library needed
    out << "{\n";
    out << "  \"id\": \"" << id << "\",\n";

    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm buf{};
#ifdef _WIN32
    localtime_s(&buf, &t);
#else
    localtime_r(&t, &buf);
#endif
    char ts[64];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", &buf);
    out << "  \"timestamp\": \"" << ts << "\",\n";
    out << "  \"tag\": \"" << tag << "\",\n";
    out << "  \"knowledge_count\": " << knowledge_count << ",\n";

    auto fsize = std::filesystem::exists(save_path)
                     ? std::filesystem::file_size(save_path)
                     : 0;
    out << "  \"file_size_bytes\": " << fsize << ",\n";
    out << "  \"save_file\": \"" << save_path.filename().string() << "\"\n";
    out << "}\n";
}

auto CheckpointManager::parse_meta_(
    const std::filesystem::path& meta_path) const
    -> std::optional<CheckpointInfo>
{
    std::ifstream in(meta_path);
    if (!in.is_open()) return std::nullopt;

    CheckpointInfo info;
    std::string line;
    while (std::getline(in, line)) {
        // Simple key-value extraction from pseudo-JSON
        auto extract = [&](const std::string& key) -> std::string {
            auto pos = line.find(key);
            if (pos == std::string::npos) return {};
            auto colon = line.find(':', pos + key.size());
            if (colon == std::string::npos) return {};
            auto val = line.substr(colon + 1);
            // Trim whitespace and quotes
            auto b = val.find_first_not_of(" \"");
            auto e = val.find_last_not_of(" \"\r\n,");
            if (b == std::string::npos) return {};
            return val.substr(b, e == std::string::npos ? std::string::npos : e - b + 1);
        };

        if (line.find("\"id\"") != std::string::npos)
            info.id = extract("\"id\"");
        else if (line.find("\"timestamp\"") != std::string::npos)
            info.timestamp = extract("\"timestamp\"");
        else if (line.find("\"tag\"") != std::string::npos)
            info.tag = extract("\"tag\"");
        else if (line.find("\"knowledge_count\"") != std::string::npos) {
            auto v = extract("\"knowledge_count\"");
            info.knowledge_count = std::stoi(v);
        }
        else if (line.find("\"file_size_bytes\"") != std::string::npos) {
            auto v = extract("\"file_size_bytes\"");
            info.file_size_bytes = std::stoll(v);
        }
    }
    if (info.id.empty()) return std::nullopt;
    return info;
}

auto CheckpointManager::save(const core::Learner& learner,
                             const std::string& tag) -> std::string
{
    ensure_dir_();
    auto id = make_id_();

    auto save_path = dir_ / (id + ".dat");
    learner.save(save_path.string());

    auto meta_path = dir_ / (id + ".meta.json");

    // Knowledge count = entity_count + relation_count
    auto stats = learner.get_stats();
    int kcount = static_cast<int>(stats.count("entity_count")
                                      ? stats.at("entity_count") : 0.0)
               + static_cast<int>(stats.count("relation_count")
                                      ? stats.at("relation_count") : 0.0);

    write_meta_(meta_path, id, tag, kcount, save_path);
    return id;
}

auto CheckpointManager::load_latest(core::Learner& learner) -> bool
{
    auto latest_cp = latest();
    if (!latest_cp) return false;
    return load(learner, latest_cp->id);
}

auto CheckpointManager::load(core::Learner& learner,
                             const std::string& checkpoint_id) -> bool
{
    auto save_path = dir_ / (checkpoint_id + ".dat");
    if (!std::filesystem::exists(save_path)) return false;
    learner.load(save_path.string());
    return true;
}

auto CheckpointManager::list_checkpoints() const
    -> std::vector<CheckpointInfo>
{
    std::vector<CheckpointInfo> result;
    if (!std::filesystem::exists(dir_)) return result;

    for (const auto& entry : std::filesystem::directory_iterator(dir_)) {
        if (entry.path().extension() != ".json") continue;
        auto info = parse_meta_(entry.path());
        if (info) result.push_back(*info);
    }
    // Newest first (id contains timestamp, lexicographic = chronological)
    std::sort(result.begin(), result.end(),
              [](const CheckpointInfo& a, const CheckpointInfo& b) {
                  return a.id > b.id;
              });
    return result;
}

void CheckpointManager::cleanup(int keep_last)
{
    auto all = list_checkpoints();
    if (static_cast<int>(all.size()) <= keep_last) return;

    for (int i = keep_last; i < static_cast<int>(all.size()); ++i) {
        auto dat  = dir_ / (all[i].id + ".dat");
        auto meta = dir_ / (all[i].id + ".meta.json");
        std::filesystem::remove(dat);
        std::filesystem::remove(meta);
    }
}

auto CheckpointManager::latest() const -> std::optional<CheckpointInfo>
{
    auto all = list_checkpoints();
    if (all.empty()) return std::nullopt;
    return all.front();
}

// =====================================================================
//  ContinuousLearningLoop
// =====================================================================

ContinuousLearningLoop::ContinuousLearningLoop(core::Learner& learner)
    : learner_(learner)
    , ckpt_mgr_("./checkpoints")
{}

void ContinuousLearningLoop::start(const ContinuousLoopConfig& config)
{
    if (running_.load()) return;

    config_    = config;
    ckpt_mgr_  = CheckpointManager(config.checkpoint_dir);
    running_.store(true);
    stop_requested_.store(false);
    start_time_ = std::chrono::steady_clock::now();
    last_checkpoint_     = start_time_;
    last_consolidation_  = start_time_;

    {
        std::lock_guard<std::mutex> lk(status_mutex_);
        status_ = LoopStatus{};
        status_.running = true;
    }

    // Auto-recover from latest checkpoint
    if (config.auto_recover) {
        recover(config.checkpoint_dir);
    }

    thread_ = std::thread(&ContinuousLearningLoop::run_loop_, this, config);
}

void ContinuousLearningLoop::stop()
{
    stop_requested_.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }
    running_.store(false);

    std::lock_guard<std::mutex> lk(status_mutex_);
    status_.running = false;
}

bool ContinuousLearningLoop::is_running() const
{
    return running_.load();
}

LoopStatus ContinuousLearningLoop::status() const
{
    std::lock_guard<std::mutex> lk(status_mutex_);
    auto s = status_;
    s.pending_data_count = [&] {
        std::lock_guard<std::mutex> dl(data_mutex_);
        return static_cast<int>(pending_data_.size());
    }();
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    s.uptime_seconds = std::chrono::duration<double>(elapsed).count();
    return s;
}

void ContinuousLearningLoop::feed_data(const std::string& data,
                                       const std::string& source)
{
    std::lock_guard<std::mutex> lk(data_mutex_);
    if (static_cast<int>(pending_data_.size()) >= config_.data_buffer_size) {
        // Drop oldest to make room
        pending_data_.erase(pending_data_.begin());
    }
    pending_data_.push_back({data, source});
}

void ContinuousLearningLoop::checkpoint_now()
{
    try {
        auto id = ckpt_mgr_.save(learner_, "manual");
        ckpt_mgr_.cleanup(5);
        std::lock_guard<std::mutex> lk(status_mutex_);
        status_.last_checkpoint_time = now_string_();
    } catch (...) {
        // Best-effort checkpoint — don't crash the loop
    }
}

bool ContinuousLearningLoop::recover(const std::string& checkpoint_dir)
{
    CheckpointManager mgr(checkpoint_dir);
    return mgr.load_latest(learner_);
}

// ── Background thread entry ──────────────────────────────────────

void ContinuousLearningLoop::run_loop_(ContinuousLoopConfig config)
{
    int iterations = 0;
    while (!stop_requested_.load()) {
        auto result = loop_iteration_();
        update_status_(result);

        ++iterations;
        if (config.max_iterations > 0 && iterations >= config.max_iterations) {
            break;
        }

        // Sleep 1s in small increments for responsive shutdown
        for (int i = 0; i < 10 && !stop_requested_.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // Final checkpoint on shutdown
    try {
        ckpt_mgr_.save(learner_, "shutdown");
        ckpt_mgr_.cleanup(5);
    } catch (...) {}
}

// ── Single iteration ─────────────────────────────────────────────

auto ContinuousLearningLoop::loop_iteration_() -> LoopStepResult
{
    LoopStepResult result;

    // 1. Ingest pending data
    result.learned = ingest_pending_data_();

    // 2. Consolidation timer
    if (should_consolidate_()) {
        try {
            learner_.consolidate();
            last_consolidation_ = std::chrono::steady_clock::now();
            result.consolidated = true;
        } catch (const std::exception& e) {
            result.errors.emplace_back(e.what());
        }
    }

    // 3. Check forgetting — if retention low, replay
    try {
        double retention = learner_.continual_learner().retention_rate();
        result.retention = retention;
        if (retention < 0.8) {
            learner_.continual_learner().replay(5);
        }
    } catch (const std::exception& e) {
        result.errors.emplace_back(e.what());
    }

    // 4. Checkpoint timer
    if (should_checkpoint_()) {
        try {
            ckpt_mgr_.save(learner_);
            ckpt_mgr_.cleanup(5);
            last_checkpoint_ = std::chrono::steady_clock::now();
            result.checkpointed = true;
        } catch (const std::exception& e) {
            result.errors.emplace_back(e.what());
        }
    }

    return result;
}

// ── Helpers ──────────────────────────────────────────────────────

bool ContinuousLearningLoop::should_consolidate_() const
{
    auto elapsed = std::chrono::steady_clock::now() - last_consolidation_;
    return std::chrono::duration<double>(elapsed).count()
           >= config_.consolidation_interval_seconds;
}

bool ContinuousLearningLoop::should_checkpoint_() const
{
    auto elapsed = std::chrono::steady_clock::now() - last_checkpoint_;
    return std::chrono::duration<double>(elapsed).count()
           >= config_.checkpoint_interval_seconds;
}

auto ContinuousLearningLoop::ingest_pending_data_() -> int
{
    std::vector<PendingItem> batch;
    {
        std::lock_guard<std::mutex> lk(data_mutex_);
        batch.swap(pending_data_);
    }

    int learned = 0;
    for (const auto& item : batch) {
        try {
            learner_.learn_from_text(item.data, item.source);
            ++learned;
        } catch (const std::exception&) {
            // Skip malformed data, continue
        }
    }
    return learned;
}

void ContinuousLearningLoop::update_status_(const LoopStepResult& result)
{
    std::lock_guard<std::mutex> lk(status_mutex_);
    ++status_.iterations;
    status_.data_processed += result.learned;
    status_.knowledge_retention = result.retention;
    if (result.consolidated)
        status_.last_consolidation_time = now_string_();
    if (result.checkpointed)
        status_.last_checkpoint_time = now_string_();
}

auto ContinuousLearningLoop::now_string_() -> std::string
{
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm buf{};
#ifdef _WIN32
    localtime_s(&buf, &t);
#else
    localtime_r(&t, &buf);
#endif
    char ts[64];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &buf);
    return ts;
}

}  // namespace ai_learning::learning
