/**
 * @file utf8.hpp
 * @brief UTF-8 字符遍历工具 — 消除跨模块重复代码
 *
 * 使用场景：
 * - 统计学习器分词
 * - 推理引擎关键词提取
 * - 目标管理器描述分割
 * - tokenizer 字符级处理
 */
#pragma once

#include <string>
#include <vector>

namespace ai_learning::utils {

/// 获取单个 UTF-8 字符的字节长度
inline auto utf8_char_len(unsigned char c) -> size_t {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;  // 无效 UTF-8，按单字节处理
}

/// 从 pos 位置提取下一个 UTF-8 字符（返回字符和前进字节数）
inline auto utf8_next_char(const std::string& text, size_t pos)
    -> std::pair<std::string, size_t> {
    if (pos >= text.size()) return {"", 0};

    auto len = utf8_char_len(static_cast<unsigned char>(text[pos]));
    if (pos + len > text.size()) len = text.size() - pos;

    return {text.substr(pos, len), len};
}

/// 遍历字符串中的每个 UTF-8 字符，调用回调
/// 回调签名: void(const std::string& char_str, size_t byte_pos)
template<typename Fn>
void utf8_foreach(const std::string& text, Fn&& callback) {
    for (size_t i = 0; i < text.size(); ) {
        auto [ch, len] = utf8_next_char(text, i);
        if (len == 0) break;
        callback(ch, i);
        i += len;
    }
}

/// 检查一个 UTF-8 标点是否为分隔符
/// pos 必须是字符的起始位置
inline auto is_utf8_delimiter(const std::string& text, size_t pos) -> bool {
    if (pos >= text.size()) return false;

    unsigned char c0 = static_cast<unsigned char>(text[pos]);

    // ASCII 分隔符
    if (c0 == ',' || c0 == '!' || c0 == '?' || c0 == ';'
        || c0 == '\n' || c0 == '\r' || c0 == '\t' || c0 == ' ') {
        return true;
    }

    // 多字节 UTF-8 标点
    size_t len = utf8_char_len(c0);
    if (pos + len > text.size()) return false;

    if (len == 3) {
        // U+3002 = 。 (E3 80 82)
        // U+3001 = 、 (E3 80 81)
        // U+FF0C = ， (EF BC 8C)
        // U+FF1B = ； (EF BC 9B)
        unsigned char c1 = static_cast<unsigned char>(text[pos + 1]);
        unsigned char c2 = static_cast<unsigned char>(text[pos + 2]);
        if (c0 == 0xE3 && c1 == 0x80 && (c2 == 0x82 || c2 == 0x81)) return true;
        if (c0 == 0xEF && c1 == 0xBC && (c2 == 0x8C || c2 == 0x9B)) return true;
    }
    return false;
}

/// 提取所有 UTF-8 字符（用于需要逐字符操作的场景）
inline auto utf8_chars(const std::string& text) -> std::vector<std::string> {
    std::vector<std::string> result;
    utf8_foreach(text, [&](const std::string& ch, size_t) {
        result.push_back(ch);
    });
    return result;
}

}  // namespace ai_learning::utils
