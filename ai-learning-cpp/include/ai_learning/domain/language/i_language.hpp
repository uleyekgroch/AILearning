/**
 * @file i_language.hpp
 * @brief 语言领域接口 — 语言产出与理解
 *
 * DDD 限界上下文：语言
 */
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ai_learning::domain {

class ILanguage {
public:
    virtual ~ILanguage() = default;

    /// 从内部意图产生话语（符号序列）
    virtual auto produce(const std::map<std::string, std::string>& intention) const
        -> std::vector<std::string> = 0;

    /// 理解话语，返回解析后的意图/指称
    virtual auto comprehend(const std::vector<std::string>& utterance,
                            const std::map<std::string, std::string>& context) const
        -> std::map<std::string, std::string> = 0;

    /// 获取当前词汇表大小
    virtual auto get_vocabulary_size() const -> int = 0;
};

}  // namespace ai_learning::domain
