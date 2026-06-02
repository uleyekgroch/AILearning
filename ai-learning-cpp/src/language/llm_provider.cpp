/**
 * @file llm_provider.cpp
 * @brief LLM 提供者实现
 */

#include "ai_learning/language/llm_provider.hpp"

#include <cstdlib>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

namespace ai_learning::language {

// ═══════════════════════════════════════════════════════════════════
// StubLLMProvider
// ═══════════════════════════════════════════════════════════════════

auto StubLLMProvider::complete(const std::string& prompt,
                               const std::string& /*system_prompt*/)
    -> std::string {
    // 基于关键词的简单模式匹配
    if (prompt.find("解释") != std::string::npos ||
        prompt.find("explain") != std::string::npos) {
        return "这是一个很好的问题。让我从基本概念开始解释："
               "每个知识点都可以分解为更小的组成部分，"
               "通过理解这些基础部分之间的关系，"
               "我们可以建立对整体概念的深入理解。";
    }

    if (prompt.find("为什么") != std::string::npos ||
        prompt.find("why") != std::string::npos) {
        return "这是一个关于因果关系的问题。"
               "根据已有的知识，我们可以从多个角度分析原因："
               "首先是直接原因，其次是潜在的系统性因素。"
               "建议通过实验来验证这些假设。";
    }

    if (prompt.find("怎么") != std::string::npos ||
        prompt.find("如何") != std::string::npos ||
        prompt.find("how") != std::string::npos) {
        return "关于这个问题，建议采用以下步骤："
               "首先明确目标和约束条件，"
               "然后分解为可执行的小任务，"
               "逐步验证每个步骤的正确性。";
    }

    if (prompt.find("学习") != std::string::npos ||
        prompt.find("教") != std::string::npos ||
        prompt.find("teach") != std::string::npos ||
        prompt.find("learn") != std::string::npos) {
        return "我已理解您传授的知识。"
               "让我将这些信息整合到我的知识体系中，"
               "建立相关的概念关联，并在需要时进行验证。";
    }

    // 默认响应
    return "我理解了。让我处理这些信息。";
}

// ═══════════════════════════════════════════════════════════════════
// OpenAICompatibleProvider
// ═══════════════════════════════════════════════════════════════════

OpenAICompatibleProvider::OpenAICompatibleProvider(
    const std::string& base_url,
    const std::string& api_key,
    const std::string& model)
    : base_url_(base_url), model_(model) {
    // API Key 优先级：构造参数 > 环境变量 > 硬编码默认值
    if (!api_key.empty()) {
        api_key_ = api_key;
    } else {
        const char* env_key = std::getenv("DASHSCOPE_API_KEY");
        api_key_ = (env_key != nullptr) ? env_key
                    : "sk-b68b1187aba542c3b2fec09cdc02634c";
    }
}

auto OpenAICompatibleProvider::complete(const std::string& prompt,
                                        const std::string& system_prompt)
    -> std::string {
    // 构建 OpenAI 兼容请求体
    nlohmann::json messages = nlohmann::json::array();

    if (!system_prompt.empty()) {
        messages.push_back({{"role", "system"}, {"content", system_prompt}});
    }
    messages.push_back({{"role", "user"}, {"content", prompt}});

    nlohmann::json body = {
        {"model", model_},
        {"messages", messages}
    };

    std::string url = "https://" + base_url_ +
                      "/compatible-mode/v1/chat/completions";
    std::string raw = exec_curl_(url, body.dump());

    if (raw.empty()) {
        return "[LLM 请求失败：无响应]";
    }

    try {
        auto resp = nlohmann::json::parse(raw);
        if (resp.contains("choices") && !resp["choices"].empty()) {
            return resp["choices"][0]["message"]["content"].get<std::string>();
        }
        if (resp.contains("error")) {
            return "[LLM 错误：" +
                   resp["error"]["message"].get<std::string>() + "]";
        }
        return "[LLM 响应格式异常]";
    } catch (const nlohmann::json::exception&) {
        return "[LLM 响应解析失败]";
    }
}

auto OpenAICompatibleProvider::exec_curl_(const std::string& url,
                                          const std::string& body_json) const
    -> std::string {
    // 通过 curl 子进程执行 HTTPS POST，避免 OpenSSL 依赖
    // 将 JSON body 写入临时文件以避免 shell 转义问题
    std::string tmp_path = std::tmpnam(nullptr);

    // 写入请求体到临时文件
    std::FILE* tmp = std::fopen(tmp_path.c_str(), "w");
    if (!tmp) {
        return "";
    }
    std::fputs(body_json.c_str(), tmp);
    std::fclose(tmp);

    // 构造 curl 命令
    // -s 静默模式  -X POST  -d @file 从文件读取 body
    std::ostringstream cmd;
    cmd << "curl -s -X POST \"" << url << "\""
        << " -H \"Authorization: Bearer " << api_key_ << "\""
        << " -H \"Content-Type: application/json\""
        << " -d @" << tmp_path
        << " --max-time 30";

    // 执行并捕获输出
    std::string output;
    constexpr int kBufSize = 4096;
    char buffer[kBufSize];

#ifdef _WIN32
    auto pipe = _popen(cmd.str().c_str(), "r");
#else
    auto pipe = popen(cmd.str().c_str(), "r");
#endif

    if (pipe) {
        while (std::fgets(buffer, kBufSize, pipe) != nullptr) {
            output += buffer;
        }
#ifdef _WIN32
        _pclose(pipe);
#else
        pclose(pipe);
#endif
    }

    // 清理临时文件
    std::remove(tmp_path.c_str());

    return output;
}

}  // namespace ai_learning::language
