/**
 * @file corpus_main.cpp
 * @brief wiki_zh 语料学习 — 用 AILearning 系统学习中文维基百科
 *
 * 流程：
 *   1. 扫描 wiki_zh/ 目录下所有文件
 *   2. 解析 JSON 文章（{id, url, title, text}）
 *   3. 逐篇调用 learner.learn_from_text()
 *   4. 每 100 篇巩固记忆 + 打印统计
 *   5. 输出学习报告 JSON
 */

#include "ai_learning/core/learner.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace ai_learning::core;

// ── wiki_zh JSON 解析器 ──────────────────────────────────────────

struct WikiArticle {
    std::string id;
    std::string title;
    std::string text;
};

/// 从单个 wiki_zh 文件中解析所有文章
/// 格式：每行一个 JSON 对象 {id, url, title, text}
auto parse_wiki_file(const std::string& filepath)
    -> std::vector<WikiArticle>
{
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) return {};

    std::vector<WikiArticle> articles;
    std::string line;

    while (std::getline(ifs, line)) {
        // 跳过空行
        if (line.empty() || line == "\r") continue;
        // 跳过非 JSON 行
        if (line[0] != '{') continue;

        try {
            auto obj = json::parse(line);
            WikiArticle art;
            art.id    = obj.value("id", "");
            art.title = obj.value("title", "");
            art.text  = obj.value("text", "");
            if (!art.title.empty() && !art.text.empty()) {
                articles.push_back(std::move(art));
            }
        } catch (...) {
            // 跳过解析失败的行
        }
    }
    ifs.close();
    return articles;
}

/// 扫描目录下所有 wiki 文件（支持一层和两层目录结构）
auto scan_wiki_files(const std::string& base_dir)
    -> std::vector<std::string>
{
    std::vector<std::string> files;
    if (!fs::exists(base_dir)) return files;

    // 辅助：检查文件名是否为 wiki_* 格式
    auto is_wiki_file = [](const std::string& name) -> bool {
        return name.size() > 5 && name.substr(0, 5) == "wiki_";
    };

    for (const auto& entry : fs::directory_iterator(base_dir)) {
        if (entry.is_regular_file()) {
            // 直接在 base_dir 下: base_dir/wiki_00
            if (is_wiki_file(entry.path().filename().string())) {
                files.push_back(entry.path().string());
            }
        } else if (entry.is_directory()) {
            // 一层子目录: base_dir/AA/wiki_00
            for (const auto& file : fs::directory_iterator(entry.path())) {
                if (file.is_regular_file() &&
                    is_wiki_file(file.path().filename().string())) {
                    files.push_back(file.path().string());
                }
            }
        }
    }
    // 按路径排序确保可重现
    std::sort(files.begin(), files.end());
    return files;
}

// ── 学习报告 ────────────────────────────────────────────────────

struct LearningStats {
    int total_articles = 0;
    int total_entities = 0;
    int total_triples = 0;
    int total_causal = 0;
    int total_numerical = 0;
    int verified = 0;
    int failed = 0;
    int files_processed = 0;
    double elapsed_seconds = 0.0;
    std::map<std::string, int> entity_frequency;   // 实体出现频率
    std::map<std::string, int> relation_types;     // 关系类型分布
};

void save_report(const LearningStats& stats,
                 const Learner& learner,
                 const std::string& output_path)
{
    json report;

    // 基本统计
    report["stats"]["total_articles"]    = stats.total_articles;
    report["stats"]["total_entities"]    = stats.total_entities;
    report["stats"]["total_triples"]     = stats.total_triples;
    report["stats"]["total_causal"]      = stats.total_causal;
    report["stats"]["total_numerical"]   = stats.total_numerical;
    report["stats"]["verified"]          = stats.verified;
    report["stats"]["failed"]            = stats.failed;
    report["stats"]["verification_rate"] = stats.total_articles > 0
        ? static_cast<double>(stats.verified) / stats.total_articles
        : 0.0;
    report["stats"]["files_processed"]   = stats.files_processed;
    report["stats"]["elapsed_seconds"]   = stats.elapsed_seconds;

    // 知识图谱统计
    auto& kg = learner.knowledge_graph();
    report["knowledge_graph"]["entity_count"]  = kg.entity_count();

    // Top-30 高频实体
    std::vector<std::pair<std::string, int>> freq_vec(
        stats.entity_frequency.begin(), stats.entity_frequency.end());
    std::sort(freq_vec.begin(), freq_vec.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    json top_entities = json::array();
    for (int i = 0; i < std::min(30, (int)freq_vec.size()); ++i) {
        top_entities.push_back({{"entity", freq_vec[i].first},
                                 {"count", freq_vec[i].second}});
    }
    report["top_entities"] = top_entities;

    // 关系类型分布
    std::vector<std::pair<std::string, int>> rel_vec(
        stats.relation_types.begin(), stats.relation_types.end());
    std::sort(rel_vec.begin(), rel_vec.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    json relation_dist = json::array();
    for (int i = 0; i < std::min(20, (int)rel_vec.size()); ++i) {
        relation_dist.push_back({{"relation", rel_vec[i].first},
                                  {"count", rel_vec[i].second}});
    }
    report["relation_distribution"] = relation_dist;

    // 推理测试
    json reasoning_tests = json::array();
    for (const auto& query : {"数学", "物理", "中国", "历史", "计算机",
                               "人工智能", "生物", "化学", "地球", "语言"}) {
        auto answer = learner.think(query);
        reasoning_tests.push_back({{"question", query}, {"answer", answer}});
    }
    report["reasoning_tests"] = reasoning_tests;

    // 系统统计
    auto sys_stats = learner.get_stats();
    report["system_stats"] = sys_stats;

    // 嵌入学习统计
    if (learner.config().embedding_learning_enabled) {
        auto& ds = learner.distributional_semantics();
        auto& et = learner.embedding_trainer();
        auto ds_stats = ds.stats();
        report["distributional_semantics"]["concepts"]      = ds_stats.cpts_represented;
        report["distributional_semantics"]["dimensions"]     = ds_stats.total_dimensions;
        report["distributional_semantics"]["texts_processed"]= ds_stats.texts_processed;
        report["distributional_semantics"]["avg_density"]    = ds_stats.avg_vector_density;
        report["distributional_semantics"]["clusters"]       = ds_stats.clusters_found;
        report["embedding"]["vocab_size"]  = et.vocab_size();
        report["embedding"]["is_trained"]  = et.is_trained();

        // DS 相似度样例：查看语义空间质量
        auto ds_cpts = ds.all_cpts();
        json ds_sim = json::array();
        for (int i = 0; i < std::min(10, (int)ds_cpts.size()); i += 2) {
            auto& pa = ds_cpts[i];
            if (pa.size() < 6) continue;  // 跳过单字
            auto top = ds.most_similar(pa, 5);
            if (top.empty()) continue;
            json entry;
            entry["word"] = pa;
            json sim_list = json::array();
            for (auto& s : top) {
                sim_list.push_back({{s.cpt_b, std::round(s.similarity * 1000) / 1000.0}});
            }
            entry["similar"] = sim_list;
            ds_sim.push_back(entry);
        }
        report["distributional_semantics"]["similarity_examples"] = ds_sim;

        // Embedding 相似度样例（如果已训练）
        if (et.is_trained()) {
            json emb_sim = json::array();
            for (const auto& word : {"数学", "物理", "化学", "中国", "历史",
                                     "计算机", "人工智能", "生物", "语言", "文化"}) {
                auto top = et.most_similar(word, 5);
                if (top.empty()) continue;
                json entry;
                entry["word"] = word;
                json sim_list = json::array();
                for (auto& s : top) {
                    sim_list.push_back({{s.word, std::round(s.similarity * 1000) / 1000.0}});
                }
                entry["similar"] = sim_list;
                emb_sim.push_back(entry);
            }
            report["embedding"]["similarity_examples"] = emb_sim;
        }
    }

    std::ofstream ofs(output_path);
    ofs << report.dump(2, ' ', true) << std::endl;
}

// ── main ────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::cout << "=== AILearning 语料学习系统 ===\n\n";

    // 语料目录（可命令行覆盖）
    std::string corpus_dir = "../parallel-learning/data/wiki_zh";
    int max_files = 0;  // 0 = 全部
    int max_articles_per_file = 0;  // 0 = 全部
    std::string output_path = "wiki_learning_report.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--corpus" && i + 1 < argc) {
            corpus_dir = argv[++i];
        } else if (arg == "--max-files" && i + 1 < argc) {
            max_files = std::stoi(argv[++i]);
        } else if (arg == "--max-articles" && i + 1 < argc) {
            max_articles_per_file = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    // 扫描语料文件
    std::cout << "扫描语料目录: " << corpus_dir << "\n";
    auto files = scan_wiki_files(corpus_dir);
    if (max_files > 0 && (int)files.size() > max_files) {
        files.resize(max_files);
    }
    std::cout << "找到 " << files.size() << " 个语料文件\n\n";

    if (files.empty()) {
        std::cerr << "错误: 未找到语料文件！请检查路径。\n";
        return 1;
    }

    // 初始化学习器
    LearnerConfig config;
    config.obs_dim = 128;
    config.embedding_learning_enabled = true;
    config.embedding_train_interval = 5; // 调大以提升性能
    config.embedding_epochs = 2;         // 调低以提升性能
    config.embedding_predictive_learning = true;
    config.pc_max_steps_per_text = 20;   // 限制推演步数提升速度
    Learner learner(config);

    // ★ 加载脑快照（如果存在的话，这使得能够断点续传）
    std::string checkpoint_path = "brain_checkpoint.json";
    learner.load(checkpoint_path);

    LearningStats stats;
    auto t_start = std::chrono::steady_clock::now();

    // ── 主学习循环 ──────────────────────────────────────────────
    int articles_since_consolidate = 0;
    const int CONSOLIDATE_INTERVAL = 200; // 调大以提升性能

    for (size_t fi = 0; fi < files.size(); ++fi) {
        auto articles = parse_wiki_file(files[fi]);
        if (max_articles_per_file > 0 && (int)articles.size() > max_articles_per_file) {
            articles.resize(max_articles_per_file);
        }

        for (const auto& art : articles) {
            // 组合标题和正文
            std::string input = art.title + "：" + art.text;

            // 使用主动阅读机制（The Conscious Autotelic Loop Phase 3）
            auto result = learner.active_read(input, art.title);

            // 累计统计
            stats.total_articles++;
            stats.total_entities += (int)result.entities.size();
            stats.total_triples  += (int)result.triples.size();
            stats.total_causal   += (int)result.causal_links.size();
            stats.total_numerical += (int)result.numerical_facts.size();

            if (result.verification_passed) {
                stats.verified++;
            } else {
                stats.failed++;
            }

            // 实体频率统计
            for (const auto& e : result.entities) {
                stats.entity_frequency[e]++;
            }

            // 关系类型统计
            for (const auto& t : result.triples) {
                stats.relation_types[t.relation]++;
            }

            articles_since_consolidate++;

            // 定期巩固 + 打印进度
            if (articles_since_consolidate >= CONSOLIDATE_INTERVAL) {
                learner.consolidate();
                articles_since_consolidate = 0;

                auto t_now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t_start).count();

                std::cout << "  [进度] 文件 " << (fi + 1) << "/" << files.size()
                          << " | 文章 " << stats.total_articles
                          << " | 实体 " << stats.total_entities
                          << " | 三元组 " << stats.total_triples
                          << " | 验证率 "
                          << std::fixed << std::setprecision(1)
                          << (stats.total_articles > 0
                              ? 100.0 * stats.verified / stats.total_articles : 0.0)
                          << "% | KG节点 " << learner.knowledge_graph().entity_count()
                          << " | " << (int)elapsed << "s\n";
            }
        }

        stats.files_processed++;
    }

    // 最终巩固
    std::cout << "\n最终巩固记忆...\n";
    learner.consolidate();
    
    // ★ 退出前保存整个大脑状态
    learner.save(checkpoint_path);

    auto t_end = std::chrono::steady_clock::now();
    stats.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();

    // ── 输出报告 ────────────────────────────────────────────────
    save_report(stats, learner, output_path);

    std::cout << "\n=== 学习完成 ===\n";
    std::cout << "  总文章: " << stats.total_articles << "\n";
    std::cout << "  总实体: " << stats.total_entities << "\n";
    std::cout << "  总三元组: " << stats.total_triples << "\n";
    std::cout << "  因果关系: " << stats.total_causal << "\n";
    std::cout << "  数值事实: " << stats.total_numerical << "\n";
    std::cout << "  验证通过: " << stats.verified
              << " / " << stats.total_articles
              << " (" << std::fixed << std::setprecision(1)
              << (stats.total_articles > 0
                  ? 100.0 * stats.verified / stats.total_articles : 0.0)
              << "%)\n";
    std::cout << "  KG 节点: " << learner.knowledge_graph().entity_count() << "\n";
    std::cout << "  唯一实体: " << stats.entity_frequency.size() << "\n";
    std::cout << "  关系类型: " << stats.relation_types.size() << "\n";
    std::cout << "  耗时: " << (int)stats.elapsed_seconds << " 秒\n";
    std::cout << "  报告已保存: " << output_path << "\n";

    return 0;
}
