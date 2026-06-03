/**
 * @file tokenizer.hpp
 * @brief 语无关分词器 — 支持中文和英文文本的统一分词
 *
 * 中文分词：UTF-8 char + bigram + trigram + 停用词过滤
 * 英文分词：whitespace/标点分割 + 小写 + 停用词过滤
 * 自动检测：根据 CJK 字节比例选择分词策略
 */
#pragma once

#include <string>
#include <vector>

namespace ai_learning::learning {

/// 语言模式
enum class Language {
    kAuto,     // 自动检测（首字节 CJK → Chinese，否则 English）
    kChinese,  // 中文：char + bigram + trigram
    kEnglish,  // 英文：word split + lowercase
};

/// 语无关分词
auto tokenize(const std::string& text, Language lang) -> std::vector<std::string>;

}  // namespace ai_learning::learning
