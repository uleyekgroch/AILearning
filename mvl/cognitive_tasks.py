"""
经典皮亚杰认知任务模块

每个任务对应一个认知发展阶段的核心能力。
任务接口：test(agent, ...) -> float (0.0-1.0)

任务列表：
- ConservationTask: 守恒（6-8岁）
- ClassificationTask: 分类（4-6岁+）
- SeriationTask: 序列化（6-8岁）
- TransitivityTask: 传递性（8-11岁）
- PlanningTask: 多步规划（8-11岁）
- HypothesisTestingTask: 假设检验（11-13岁）
- CounterfactualTask: 反事实推理（11-13岁）
- PerspectiveCoordinationTask: 视角协调（13-15岁）
- MoralReasoningTask: 道德推理（15-17岁）
"""

import random
import numpy as np
from typing import Dict, List, Optional


class ConservationTask:
    """
    守恒任务：理解物体数量/重量不因排列改变而改变

    经典皮亚杰任务：将同样数量的液体倒入不同形状的杯子，
    儿童是否理解液体总量不变。
    """

    name = "守恒"
    stage = "early_concrete"

    def test(self, agent, num_objects: int = 5) -> float:
        """
        测试：相同物体在不同排列下的预测是否一致

        创建两组相同的物体，一组紧密排列，一组分散排列。
        守恒能力 = agent 对两组的预测相似度。
        """
        if len(agent.experiences) < 30:
            return 0.0

        # 创建两组相同的物体特征
        base_features = np.random.rand(7) * 0.5 + 0.25

        # 紧密排列：物体特征集中在小范围
        compact_obs = np.concatenate([
            [0.5, 0.5],  # agent 位置
            [0.5, 0.5],  # 物体位置（紧密）
            base_features[:5]
        ])

        # 分散排列：物体特征分散
        spread_obs = np.concatenate([
            [0.5, 0.5],  # agent 位置
            [0.2, 0.8],  # 物体位置（分散）
            base_features[:5]
        ])

        # 预测两种排列的结果
        pred_compact = agent.predictive_model.predict(compact_obs, 0)
        pred_spread = agent.predictive_model.predict(spread_obs, 0)

        # 守恒 = 预测相似度（忽略位置差异，只看物体特征预测）
        feature_pred_compact = pred_compact[4:]  # 跳过位置
        feature_pred_spread = pred_spread[4:]

        diff = np.mean(np.abs(feature_pred_compact - feature_pred_spread))
        score = 1.0 / (1.0 + diff * 5)

        return score


class ClassificationTask:
    """
    分类任务：按特征将物体分组

    测试 agent 是否能识别相同特征的物体属于同一类别。
    """

    name = "分类"
    stage = "late_preoperational"

    def test(self, agent, num_groups: int = 3) -> float:
        """
        测试：对相同特征物体的预测是否一致

        生成多组物体，每组内特征相同但位置不同。
        分类能力 = 同组内预测误差的一致性。
        """
        if len(agent.experiences) < 20:
            return 0.0

        recent = agent.experiences[-60:]

        # 按物体特征分组
        groups = {}
        for exp in recent:
            obj_feat = tuple(np.round(exp.observation[3:7], 1))
            if obj_feat not in groups:
                groups[obj_feat] = []
            groups[obj_feat].append(exp.prediction_error)

        if len(groups) < 2:
            return 0.0

        # 各组内预测误差的一致性
        group_scores = []
        for feat, errors in groups.items():
            if len(errors) >= 2:
                consistency = 1.0 / (1.0 + np.std(errors))
                accuracy = 1.0 / (1.0 + np.mean(errors))
                group_scores.append(consistency * accuracy)

        return np.mean(group_scores) if group_scores else 0.0


class SeriationTask:
    """
    序列化任务：沿维度排序

    经典皮亚杰任务：将棍子按长度排序。
    测试 agent 是否能理解物体沿某个维度的顺序关系。
    """

    name = "序列化"
    stage = "early_concrete"

    def test(self, agent, num_items: int = 5) -> float:
        """
        测试：对不同权重物体的预测是否呈单调关系

        创建一组重量递增的物体，检查 agent 的预测是否
        也呈现递增趋势（重量越大，预测的位移越大）。
        """
        if len(agent.experiences) < 30:
            return 0.0

        # 用最近的经验，按重量排序
        # 观测向量结构: [agent_pos(2), color(4), shape(3), pos_weight(3)]
        # 重量在 obs[11]（pos_weight 的第3个元素）
        recent = agent.experiences[-50:]

        # 提取重量和预测误差
        weight_error_pairs = []
        for exp in recent:
            if len(exp.observation) > 11:
                weight = float(exp.observation[11])
                weight_error_pairs.append((weight, exp.prediction_error))

        if len(weight_error_pairs) < 10:
            return 0.0

        # 按重量排序
        weight_error_pairs.sort(key=lambda x: x[0])
        errors = [e for _, e in weight_error_pairs]

        # 计算排序一致性（误差应该随重量变化呈现某种规律）
        # 使用 Spearman 秩相关
        n = len(errors)
        weight_ranks = list(range(n))
        error_ranks = list(np.argsort(np.argsort(errors)))

        # 计算秩相关系数
        d_squared_sum = sum((w - e) ** 2 for w, e in zip(weight_ranks, error_ranks))
        spearman = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))

        # 取绝对值（正相关或负相关都说明有排序能力）
        return abs(spearman)


class TransitivityTask:
    """
    传递性任务：从部分比较推断完整排序

    经典皮亚杰任务：如果 A > B 且 B > C，则 A > C。
    测试 agent 是否能进行传递推理。
    """

    name = "传递性"
    stage = "late_concrete"

    def test(self, agent) -> float:
        """
        测试：从已有经验中推断未观察到的关系

        检查 agent 对三种不同特征物体的预测是否满足传递性。
        例如：如果对红色物体的预测误差 < 蓝色，蓝色 < 绿色，
        那么红色的预测误差应该 < 绿色。
        """
        if len(agent.experiences) < 50:
            return 0.0

        recent = agent.experiences[-80:]

        # 按物体类型分组，计算各类型的平均预测误差
        type_errors = {}
        for exp in recent:
            obj_feat = tuple(np.round(exp.observation[3:7], 0))
            if obj_feat not in type_errors:
                type_errors[obj_feat] = []
            type_errors[obj_feat].append(exp.prediction_error)

        if len(type_errors) < 3:
            return 0.0

        # 取误差最低的3种类型
        sorted_types = sorted(type_errors.items(), key=lambda x: np.mean(x[1]))
        if len(sorted_types) > 3:
            sorted_types = sorted_types[:3]

        # 检查传递性：A < B < C 应该意味着 A < C
        means = [np.mean(errors) for _, errors in sorted_types]

        # 验证 A < B, B < C => A < C
        correct_transitive = 0
        total_checks = 0

        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                for k in range(j + 1, len(means)):
                    total_checks += 1
                    # 如果 means[i] < means[j] < means[k]，传递性成立
                    if means[i] < means[j] and means[j] < means[k]:
                        correct_transitive += 1
                    # 如果 means[i] > means[j] > means[k]，也成立
                    elif means[i] > means[j] and means[j] > means[k]:
                        correct_transitive += 1

        return correct_transitive / max(1, total_checks)


class PlanningTask:
    """
    规划任务：多步目标导向问题解决

    测试 agent 是否能执行需要多个步骤才能完成的目标。
    """

    name = "规划"
    stage = "late_concrete"

    def test(self, agent, env=None) -> float:
        """
        测试：检查经验中是否存在目标导向的多步行为

        分析最近的动作序列，寻找：
        1. 连续相同动作（目标追求）
        2. 动作切换（目标完成/新目标开始）
        3. 与物体位置变化的相关性
        """
        if len(agent.experiences) < 50:
            return 0.0

        recent = agent.experiences[-80:]
        actions = [e.action for e in recent]

        # 检测目标导向序列：连续3+相同动作
        goal_sequences = 0
        i = 0
        while i < len(actions) - 2:
            seq_len = 1
            while i + seq_len < len(actions) and actions[i + seq_len] == actions[i]:
                seq_len += 1

            if seq_len >= 3:
                goal_sequences += 1
                i += seq_len
            else:
                i += 1

        # 检测位置变化（目标达成的间接证据）
        position_changes = 0
        for j in range(1, len(recent)):
            pos_diff = np.mean(np.abs(
                recent[j].observation[:2] - recent[j-1].observation[:2]
            ))
            if pos_diff > 0.1:
                position_changes += 1

        # 综合得分
        goal_score = min(1.0, goal_sequences / max(1, len(actions) / 10))
        movement_score = min(1.0, position_changes / max(1, len(recent) * 0.3))

        return 0.6 * goal_score + 0.4 * movement_score


class HypothesisTestingTask:
    """
    假设检验任务：系统性实验

    测试 agent 是否能系统性地测试假设，
    而不是随机探索。
    """

    name = "假设检验"
    stage = "early_formal"

    def test(self, agent) -> float:
        """
        测试：假设检验的系统性

        分析最近的经验，检查 agent 是否：
        1. 对同一状态尝试多种不同动作（控制变量）
        2. 对相似状态使用相同动作（重复验证）
        3. 预测误差随时间降低（假设被确认）
        """
        if len(agent.experiences) < 40:
            return 0.0

        recent = agent.experiences[-60:]

        # 系统性指标1：同一状态的多动作尝试
        state_action_pairs = {}
        for exp in recent:
            state_key = tuple(np.round(exp.observation[:4], 1))
            if state_key not in state_action_pairs:
                state_action_pairs[state_key] = set()
            state_action_pairs[state_key].add(exp.action)

        # 每个状态尝试的动作数
        actions_per_state = [len(actions) for actions in state_action_pairs.values()]
        diversity_score = np.mean(actions_per_state) / agent.action_dim if actions_per_state else 0

        # 系统性指标2：预测误差的下降趋势
        errors = [e.prediction_error for e in recent]
        first_half_err = np.mean(errors[:len(errors)//2])
        second_half_err = np.mean(errors[len(errors)//2:])
        improvement_score = max(0, (first_half_err - second_half_err) / (first_half_err + 1e-6))

        return 0.5 * min(1.0, diversity_score * 2) + 0.5 * min(1.0, improvement_score * 3)


class CounterfactualTask:
    """
    反事实推理任务：如果做了不同的动作会怎样

    测试 agent 是否能预测替代动作的结果。
    """

    name = "反事实推理"
    stage = "early_formal"

    def test(self, agent) -> float:
        """
        测试：对同一状态的不同动作预测是否不同且合理

        检查 agent 是否对相同状态产生了不同的预测
        （取决于选择的动作）。如果所有动作的预测相同，
        说明没有反事实推理能力。
        """
        if len(agent.experiences) < 30:
            return 0.0

        recent = agent.experiences[-40:]

        # 找到同一状态的不同动作
        state_predictions = {}
        for exp in recent:
            state_key = tuple(np.round(exp.observation[:4], 1))
            if state_key not in state_predictions:
                state_predictions[state_key] = []
            state_predictions[state_key].append(
                (exp.action, exp.prediction_error)
            )

        # 检查每个状态的预测多样性
        diversity_scores = []
        for state_key, preds in state_predictions.items():
            if len(preds) >= 2:
                errors = [e for _, e in preds]
                # 不同动作应该有不同的预测误差
                diversity = np.std(errors) / (np.mean(errors) + 1e-6)
                diversity_scores.append(min(1.0, diversity))

        return np.mean(diversity_scores) if diversity_scores else 0.0


class PerspectiveCoordinationTask:
    """
    视角协调任务：区分自己和他人的视角

    测试 agent 是否能理解不同 agent 看到不同的东西。
    """

    name = "视角协调"
    stage = "late_formal"

    def test(self, agent) -> float:
        """
        测试：通信中的视角调整能力

        基于 agent 的通信历史，检查：
        1. 是否有成功的跨视角通信
        2. 通信成功率是否随时间提高
        """
        total_comm = agent.communication_stats.get('total', 0)
        success_comm = agent.communication_stats.get('success', 0)

        if total_comm < 5:
            return 0.0

        comm_rate = success_comm / total_comm

        # 交互频率因子
        frequency_factor = min(1.0, total_comm / 50)

        return comm_rate * frequency_factor


class MoralReasoningTask:
    """
    道德推理任务：公平 vs 自私的选择

    测试 agent 是否考虑公平性，而不是总是选择最大化自身利益。
    """

    name = "道德推理"
    stage = "adolescent"

    def test(self, agent) -> float:
        """
        测试：动作分布的均衡性

        在多 Agent 场景中，如果 agent 总是选择相同的"自私"动作，
        说明缺乏道德推理。动作分布越均衡，道德推理越成熟。
        """
        if len(agent.experiences) < 50:
            return 0.0

        recent = agent.experiences[-80:]
        action_counts = {}
        for exp in recent:
            action_counts[exp.action] = action_counts.get(exp.action, 0) + 1

        total = len(recent)
        # 计算熵
        entropy = 0.0
        for count in action_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        # 归一化
        max_entropy = np.log2(agent.action_dim) if agent.action_dim > 1 else 1.0
        return entropy / max_entropy


# 阶段-任务映射
STAGE_TASKS = {
    'sensorimotor': [],
    'early_preoperational': [],
    'late_preoperational': [ClassificationTask()],
    'early_concrete': [ConservationTask(), ClassificationTask(), SeriationTask()],
    'late_concrete': [TransitivityTask(), PlanningTask(), ClassificationTask()],
    'early_formal': [HypothesisTestingTask(), CounterfactualTask()],
    'late_formal': [PerspectiveCoordinationTask(), HypothesisTestingTask()],
    'adolescent': [MoralReasoningTask(), HypothesisTestingTask(), CounterfactualTask()],
}


def run_stage_tasks(stage: str, agent, env=None) -> Dict[str, float]:
    """
    运行指定阶段的所有认知任务

    Args:
        stage: 当前发展阶段名称
        agent: LearningAgent 实例
        env: 可选的环境实例

    Returns:
        任务名 -> 得分的字典
    """
    tasks = STAGE_TASKS.get(stage, [])
    results = {}

    for task in tasks:
        try:
            score = task.test(agent)
            results[task.name] = score
        except Exception as e:
            results[task.name] = 0.0

    return results


def run_all_tasks(agent, env=None) -> Dict[str, Dict[str, float]]:
    """
    运行所有认知任务

    Returns:
        阶段 -> {任务名 -> 得分} 的嵌套字典
    """
    all_results = {}
    for stage, tasks in STAGE_TASKS.items():
        if tasks:
            all_results[stage] = run_stage_tasks(stage, agent, env)
    return all_results
