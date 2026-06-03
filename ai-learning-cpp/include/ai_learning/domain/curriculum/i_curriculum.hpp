/**
 * @file i_curriculum.hpp
 * @brief 课程领域接口 — 发展阶段调度
 *
 * DDD 限界上下文：课程
 */
#pragma once

#include <map>
#include <string>

namespace ai_learning::domain {

class ICurriculum {
public:
    virtual ~ICurriculum() = default;

    /// 获取当前发展阶段名称
    virtual auto get_current_stage() const -> std::string = 0;

    /// 评估学习者各项能力指标
    virtual auto evaluate()
        -> std::map<std::string, double> = 0;

    /// 判断是否满足晋升下一阶段的条件
    virtual auto should_advance(
        const std::map<std::string, double>& evaluation) const -> bool = 0;
};

}  // namespace ai_learning::domain
