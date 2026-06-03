/**
 * @file test_tokenizer.cpp
 * @brief 语无关分词器单元测试
 *
 * 验证：
 *   1. 中文分词：CJK char + bigram + trigram
 *   2. 英文分词：lowercase + stopword 过滤
 *   3. 自动语言检测
 *   4. 空字符串、纯标点等边界情况
 *   5. 停用词过滤正确性
 */

#include <catch2/catch_test_macros.hpp>

#include "ai_learning/learning/tokenizer.hpp"

using namespace ai_learning::learning;

// ═══════════════════════════════════════════════════════════
// 基础：空输入与边界
// ═══════════════════════════════════════════════════════════

TEST_CASE("分词器: 空字符串返回空向量", "[tokenizer]") {
    auto tokens = tokenize("", Language::kChinese);
    CHECK(tokens.empty());

    tokens = tokenize("", Language::kEnglish);
    CHECK(tokens.empty());

    tokens = tokenize("", Language::kAuto);
    CHECK(tokens.empty());
}

TEST_CASE("分词器: 纯空格返回空向量", "[tokenizer]") {
    auto tokens = tokenize("   ", Language::kEnglish);
    CHECK(tokens.empty());
}

TEST_CASE("分词器: 纯标点返回空向量", "[tokenizer]") {
    auto tokens = tokenize(".,;:!?", Language::kEnglish);
    CHECK(tokens.empty());
}

// ═══════════════════════════════════════════════════════════
// 英文分词
// ═══════════════════════════════════════════════════════════

TEST_CASE("分词器: 英文简单句子", "[tokenizer]") {
    auto tokens = tokenize("Artificial intelligence is a branch of computer science",
                           Language::kEnglish);

    // 停用词 "is", "a", "of" 应被过滤
    CHECK_FALSE(tokens.empty());

    // 应包含非停用词（小写）
    bool has_artificial = false, has_intelligence = false;
    bool has_branch = false, has_computer = false, has_science = false;
    for (const auto& t : tokens) {
        if (t == "artificial") has_artificial = true;
        if (t == "intelligence") has_intelligence = true;
        if (t == "branch") has_branch = true;
        if (t == "computer") has_computer = true;
        if (t == "science") has_science = true;
    }
    CHECK(has_artificial);
    CHECK(has_intelligence);
    CHECK(has_branch);
    CHECK(has_computer);
    CHECK(has_science);
}

TEST_CASE("分词器: 英文停用词过滤", "[tokenizer]") {
    auto tokens = tokenize("this is the a an", Language::kEnglish);
    // 全部是停用词或单字符词，应返回空
    CHECK(tokens.empty());
}

TEST_CASE("分词器: 英文大小写统一", "[tokenizer]") {
    auto tokens1 = tokenize("Hello World", Language::kEnglish);
    auto tokens2 = tokenize("hello world", Language::kEnglish);

    REQUIRE(tokens1.size() == tokens2.size());
    for (size_t i = 0; i < tokens1.size(); ++i) {
        CHECK(tokens1[i] == tokens2[i]);
    }
}

TEST_CASE("分词器: 英文单字符词被过滤", "[tokenizer]") {
    auto tokens = tokenize("I a", Language::kEnglish);
    // "I" 和 "a" 都是单字符，应被过滤
    CHECK(tokens.empty());
}

// ═══════════════════════════════════════════════════════════
// 中文分词
// ═══════════════════════════════════════════════════════════

TEST_CASE("分词器: 中文基本分词", "[tokenizer]") {
    // "学习数学" = 学习(非停用) + 数学(非停用)
    auto tokens = tokenize("学习数学", Language::kChinese);
    CHECK_FALSE(tokens.empty());

    // 应包含 unigram
    bool has_learn = false, has_math = false;
    for (const auto& t : tokens) {
        if (t == "学") has_learn = true;
        if (t == "数") has_math = true;
    }
    CHECK(has_learn);
    CHECK(has_math);
}

TEST_CASE("分词器: 中文 bigram 和 trigram", "[tokenizer]") {
    // 4个中文字符，应产生 unigram + bigram + trigram
    auto tokens = tokenize("人工智能技术", Language::kChinese);
    CHECK_FALSE(tokens.empty());

    // 检查是否包含 bigram（2个CJK字符 = 6字节）
    bool has_bigram = false;
    for (const auto& t : tokens) {
        if (t.size() == 6) has_bigram = true;  // 2个UTF-8中文字符 = 6字节
    }
    CHECK(has_bigram);

    // 检查是否包含 trigram（3个CJK字符 = 9字节）
    bool has_trigram = false;
    for (const auto& t : tokens) {
        if (t.size() == 9) has_trigram = true;  // 3个UTF-8中文字符 = 9字节
    }
    CHECK(has_trigram);
}

TEST_CASE("分词器: 中文停用词过滤", "[tokenizer]") {
    // "的", "了" 是中文停用词
    auto tokens = tokenize("我的了", Language::kChinese);
    // "的", "了" 被过滤，只有 "我"（但"我"也是停用词）
    // 实际上 "我", "的", "了" 都是停用词
    bool has_de = false, has_le = false;
    for (const auto& t : tokens) {
        if (t == "的") has_de = true;
        if (t == "了") has_le = true;
    }
    CHECK_FALSE(has_de);
    CHECK_FALSE(has_le);
}

// ═══════════════════════════════════════════════════════════
// 自动检测
// ═══════════════════════════════════════════════════════════

TEST_CASE("分词器: 自动检测中文", "[tokenizer]") {
    // 高比例 CJK 字节 → 应按中文模式分词
    auto tokens = tokenize("人工智能是计算机科学的一个分支", Language::kAuto);
    CHECK_FALSE(tokens.empty());

    // 中文模式应该产出 bigram
    bool has_bigram = false;
    for (const auto& t : tokens) {
        if (t.size() == 6) has_bigram = true;
    }
    CHECK(has_bigram);
}

TEST_CASE("分词器: 自动检测英文", "[tokenizer]") {
    // 纯 ASCII → 英文模式
    auto tokens = tokenize("Hello world example", Language::kAuto);
    CHECK_FALSE(tokens.empty());

    // 英文模式不应产出 bigram/trigram（全是小写单词）
    bool all_ascii_words = true;
    for (const auto& t : tokens) {
        for (char c : t) {
            if (static_cast<unsigned char>(c) >= 0x80) {
                all_ascii_words = false;
            }
        }
    }
    CHECK(all_ascii_words);
}

// ═══════════════════════════════════════════════════════════
// 特殊情况
// ═══════════════════════════════════════════════════════════

TEST_CASE("分词器: 中英混合文本", "[tokenizer][chinese]") {
    // 包含中英文混合 — 中文模式下 ASCII 部分也处理
    auto tokens = tokenize("Python编程语言", Language::kChinese);
    CHECK_FALSE(tokens.empty());
}

TEST_CASE("分词器: 数字处理", "[tokenizer]") {
    // 英文模式下，纯数字被当作单词处理
    auto tokens = tokenize("test123 data456", Language::kEnglish);
    CHECK_FALSE(tokens.empty());
}

TEST_CASE("分词器: 重复分词一致性", "[tokenizer]") {
    auto tokens1 = tokenize("Hello World Test", Language::kEnglish);
    auto tokens2 = tokenize("Hello World Test", Language::kEnglish);

    REQUIRE(tokens1.size() == tokens2.size());
    for (size_t i = 0; i < tokens1.size(); ++i) {
        CHECK(tokens1[i] == tokens2[i]);
    }
}

TEST_CASE("分词器: 长文本分词", "[tokenizer]") {
    std::string long_text;
    for (int i = 0; i < 100; ++i) {
        long_text += "learning system ";
    }

    auto tokens = tokenize(long_text, Language::kEnglish);
    CHECK_FALSE(tokens.empty());
    // "learning" 和 "system" 应各出现 100 次
    int learning_count = 0, system_count = 0;
    for (const auto& t : tokens) {
        if (t == "learning") ++learning_count;
        if (t == "system") ++system_count;
    }
    CHECK(learning_count == 100);
    CHECK(system_count == 100);
}
