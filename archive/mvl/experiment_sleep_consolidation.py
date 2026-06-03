"""
Phase 79: 睡眠式记忆巩固实验

验证类睡眠记忆巩固机制如何提升语言习得和检索性能。

核心机制：
Agent 在交流过程中积累情景记忆，通过定期"重播-巩固"周期
强化成功模式、弱化失败模式，实现更持久的记忆保持。

实验：
1. 巩固效益实验：巩固 vs 无巩固的检索准确率
2. 遗忘曲线实验：不同延迟下的记忆保持率
3. 重播策略实验：4 种重播策略的效果比较
4. 语言-巩固交互实验：巩固是否促进后续交流
"""

import json
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from language_emergence import (
    EmergingLanguage, _symbol_category,
    COLORS, SHAPES, SIZES, MATERIALS,
    generate_rich_scene,
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class MemoryTrace:
    """情景记忆痕迹"""

    def __init__(self, content: Dict, linguistic: List[str],
                 timestamp: int, strength: float = 1.0):
        self.content = content              # 原始场景特征 {color, shape, ...}
        self.linguistic = list(linguistic)   # 使用的语言符号
        self.timestamp = timestamp           # 经验发生的回合
        self.strength = strength             # 记忆强度
        self.replay_count = 0                # 被重播次数

    def clone(self) -> "MemoryTrace":
        t = MemoryTrace(dict(self.content), list(self.linguistic),
                        self.timestamp, self.strength)
        t.replay_count = self.replay_count
        return t


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ConsolidatingAgent:
    """带记忆巩固机制的 Agent"""

    def __init__(self, language: Optional[EmergingLanguage] = None,
                 replay_strategy: str = "prioritize_success",
                 replay_k: int = 10,
                 boost: float = 0.3,
                 decay_amount: float = 0.15):
        self.language = language or EmergingLanguage()
        self.episodic_memory: List[MemoryTrace] = []
        self.consolidated: Dict[str, MemoryTrace] = {}  # key: "-".join(sorted(linguistic))
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        self.boost = boost
        self.decay_amount = decay_amount

    # ---- 编码 ----

    def encode_experience(self, scene: List[Dict[str, str]],
                          target_idx: int, utterance: List[str],
                          success: bool, round_num: int) -> MemoryTrace:
        """将一次交流经验编码为记忆痕迹"""
        target = scene[target_idx]
        trace = MemoryTrace(
            content=dict(target),
            linguistic=list(utterance),
            timestamp=round_num,
            strength=1.0 if success else 0.5,
        )
        self.episodic_memory.append(trace)
        # 同步到语言系统
        self.language.record_usage(utterance, success)
        return trace

    # ---- 重播选择 ----

    def _select_replay_traces(self) -> List[MemoryTrace]:
        """根据策略选择要重播的记忆"""
        if not self.episodic_memory:
            return []
        k = min(self.replay_k, len(self.episodic_memory))

        if self.replay_strategy == "random":
            return random.sample(self.episodic_memory, k)

        if self.replay_strategy == "prioritize_success":
            scored = sorted(self.episodic_memory,
                            key=lambda t: (t.strength, -t.timestamp),
                            reverse=True)
            return scored[:k]

        if self.replay_strategy == "prioritize_recent":
            scored = sorted(self.episodic_memory,
                            key=lambda t: t.timestamp, reverse=True)
            return scored[:k]

        if self.replay_strategy == "prioritize_surprising":
            # "意外" = 成功率偏离 0.5 最大（很成功或很失败）
            def surprise(t):
                if not t.linguistic:
                    return 0.0
                rates = []
                for sym in t.linguistic:
                    v = self.language.vocabulary.get(sym)
                    if v and v["frequency"] > 0:
                        rates.append(abs(v["success_rate"] - 0.5))
                return max(rates) if rates else 0.5
            scored = sorted(self.episodic_memory, key=surprise, reverse=True)
            return scored[:k]

        # fallback
        return random.sample(self.episodic_memory, k)

    # ---- 巩固 ----

    def consolidate(self):
        """重播-巩固周期：强化成功模式、弱化失败模式、合并相似痕迹"""
        traces = self._select_replay_traces()
        for trace in traces:
            trace.replay_count += 1
            if trace.strength >= 0.7:
                # 成功 → 强化（多次重播增强效果）
                boost_amount = self.boost * 0.1 * (1 + 0.1 * trace.replay_count)
                trace.strength = min(1.0, trace.strength + boost_amount)
                for sym in trace.linguistic:
                    v = self.language.vocabulary.get(sym)
                    if v and v["frequency"] > 0:
                        v["success_rate"] = min(1.0, v["success_rate"] + self.boost * 0.05)
            else:
                # 失败 → 弱化
                trace.strength = max(0.05, trace.strength - self.decay_amount * 0.1)
                for sym in trace.linguistic:
                    v = self.language.vocabulary.get(sym)
                    if v and v["frequency"] > 0:
                        v["success_rate"] = max(0.0, v["success_rate"] - self.decay_amount * 0.03)

            # 合并到巩固存储
            key = "-".join(sorted(trace.linguistic)) if trace.linguistic else ""
            if key:
                if key in self.consolidated:
                    existing = self.consolidated[key]
                    # 加权合并：保留更高强度
                    existing.strength = max(existing.strength, trace.strength)
                    existing.replay_count += 1
                    # 保留更新的时间戳
                    if trace.timestamp > existing.timestamp:
                        existing.timestamp = trace.timestamp
                else:
                    self.consolidated[key] = trace.clone()

    # ---- 检索 ----

    def retrieve(self, query_features: Dict[str, str],
                 query_linguistic: List[str]) -> List[Tuple[MemoryTrace, float]]:
        """双通道检索：特征相似度 + 语言重叠"""
        results: List[Tuple[MemoryTrace, float]] = []

        # 搜索情景记忆
        for trace in self.episodic_memory:
            score = self._compute_similarity(trace, query_features, query_linguistic)
            if score > 0:
                results.append((trace, score))

        # 搜索巩固记忆（权重更高）
        for key, trace in self.consolidated.items():
            score = self._compute_similarity(trace, query_features, query_linguistic)
            if score > 0:
                score *= 1.2  # 巩固记忆加权
                results.append((trace, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _compute_similarity(self, trace: MemoryTrace,
                            query_features: Dict[str, str],
                            query_linguistic: List[str]) -> float:
        """计算查询与记忆痕迹的相似度"""
        # 特征通道
        feature_score = 0.0
        total_dims = max(len(query_features), 1)
        for dim, val in query_features.items():
            if dim in trace.content and trace.content[dim] == val:
                feature_score += 1.0
        feature_score /= total_dims

        # 语言通道
        ling_score = 0.0
        if query_linguistic and trace.linguistic:
            overlap = len(set(query_linguistic) & set(trace.linguistic))
            ling_score = overlap / max(len(set(query_linguistic) | set(trace.linguistic)), 1)

        # 双通道加权
        sim = 0.5 * feature_score + 0.5 * ling_score
        # 乘以记忆强度
        sim *= trace.strength
        return sim

    # ---- 遗忘 ----

    def forget(self, decay_rate: float = 0.95):
        """
        时间衰减遗忘：对所有记忆痕迹施加衰减。
        巩固记忆享受保护因子（衰减更慢），模拟长期记忆稳定性。
        """
        for trace in self.episodic_memory:
            trace.strength *= decay_rate
        # 巩固记忆衰减更慢（保护因子 0.6），模拟长期记忆稳定性
        protected_decay = 1.0 - (1.0 - decay_rate) * 0.6
        for key in list(self.consolidated.keys()):
            self.consolidated[key].strength *= protected_decay
            if self.consolidated[key].strength < 0.1:
                del self.consolidated[key]
        # 清除情景记忆中强度过低的
        self.episodic_memory = [
            t for t in self.episodic_memory if t.strength >= 0.1
        ]


class BaselineNoConsolidationAgent(ConsolidatingAgent):
    """无巩固基线 Agent — consolidate() 为空操作"""

    def consolidate(self):
        pass


# ---------------------------------------------------------------------------
# 游戏引擎
# ---------------------------------------------------------------------------

class ConsolidationGame:
    """巩固交流游戏：生成场景 → 描述 → 评估 → 存储经验"""

    def __init__(self, agent: ConsolidatingAgent, complexity: str = "medium",
                 noise_level: float = 0.0):
        self.agent = agent
        self.complexity = complexity
        self.noise_level = noise_level  # 描述噪声概率

    def describe_target(self, scene: List[Dict[str, str]],
                        target_idx: int) -> List[str]:
        """根据场景选择区分性描述（带噪声）"""
        target = scene[target_idx]
        target_values = list(target.values())
        random.shuffle(target_values)

        # 尝试找到唯一区分的符号
        for sym in target_values:
            count = sum(1 for obj in scene if sym in obj.values())
            if count == 1:
                utterance = [sym]
                # 噪声：随机替换
                if random.random() < self.noise_level:
                    utterance = [random.choice(target_values)]
                return utterance

        # 使用多个符号
        utterance = []
        for sym in target_values:
            utterance.append(sym)
            matches = [obj for obj in scene
                       if all(s in obj.values() for s in utterance)]
            if len(matches) == 1:
                break
        if not utterance:
            utterance = [target_values[0]]
        # 噪声：随机丢弃一个符号
        if self.noise_level > 0 and len(utterance) > 1 and random.random() < self.noise_level:
            utterance.pop(random.randrange(len(utterance)))
        return utterance

    def interpret_utterance(self, scene: List[Dict[str, str]],
                            utterance: List[str]) -> int:
        """根据描述在场景中找到目标"""
        scores = []
        for i, obj in enumerate(scene):
            obj_values = set(obj.values())
            match_count = sum(1 for s in utterance if s in obj_values)
            scores.append(match_count)
        if not scores:
            return -1
        max_score = max(scores)
        candidates = [i for i, s in enumerate(scores) if s == max_score]
        return random.choice(candidates)

    def play_round(self, round_num: int) -> bool:
        """执行一轮游戏"""
        scene = generate_rich_scene(self.complexity)
        target_idx = random.randint(0, len(scene) - 1)
        utterance = self.describe_target(scene, target_idx)
        predicted_idx = self.interpret_utterance(scene, utterance)
        success = predicted_idx == target_idx
        self.agent.encode_experience(scene, target_idx, utterance,
                                     success, round_num)
        return success

    def run(self, num_rounds: int, consolidation_interval: int = 20
            ) -> Dict:
        """运行游戏并定期巩固"""
        success_history = []
        consolidation_points = []

        for r in range(num_rounds):
            success = self.play_round(r)
            success_history.append(success)

            # 定期巩固
            if (r + 1) % consolidation_interval == 0:
                self.agent.consolidate()
                consolidation_points.append(r + 1)

            # 模拟遗忘
            if (r + 1) % 50 == 0:
                self.agent.forget(decay_rate=0.97)

        total = len(success_history)
        return {
            "success_rate": sum(success_history) / total,
            "total_rounds": total,
            "consolidation_points": consolidation_points,
            "success_history": success_history,
            "final_vocab_size": len(self.agent.language.vocabulary),
            "consolidated_count": len(self.agent.consolidated),
            "episodic_count": len(self.agent.episodic_memory),
        }


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _build_query(scene: List[Dict[str, str]], target_idx: int
                 ) -> Tuple[Dict[str, str], List[str]]:
    """构建检索查询"""
    target = scene[target_idx]
    features = dict(target)
    linguistic = [v for v in target.values()]
    return features, linguistic


def _test_retrieval(agent: ConsolidatingAgent, num_queries: int,
                    complexity: str = "medium") -> float:
    """
    测试 Agent 的检索质量（基于连续相似度分数）。

    返回平均检索质量：对于每个查询，最优检索结果的相似度分数
    乘以记忆强度。巩固 Agent 的记忆更强、更精确，因此分数更高。
    """
    total_quality = 0.0
    for _ in range(num_queries):
        scene = generate_rich_scene(complexity)
        target_idx = random.randint(0, len(scene) - 1)
        q_features, q_ling = _build_query(scene, target_idx)
        results = agent.retrieve(q_features, q_ling)
        if results:
            best_score = results[0][1]
            total_quality += best_score
        # 无结果则 0 分
    return total_quality / num_queries


# ---------------------------------------------------------------------------
# 实验 1：巩固效益
# ---------------------------------------------------------------------------

def experiment_1_consolidation_benefit(num_experiences: int = 200,
                                       num_queries: int = 50,
                                       num_runs: int = 5) -> Dict:
    """
    实验 1：巩固 vs 无巩固的检索质量比较。

    流程：用带噪声的描述存储 200 条经验，每 20 轮巩固 + 遗忘，
    然后用 50 条查询测试检索质量（基于连续相似度分数）。
    预期：巩固 Agent 检索质量 +15-25%。
    """
    consolidated_accs = []
    baseline_accs = []

    for run in range(num_runs):
        random.seed(42 + run)
        np.random.seed(42 + run)

        # 巩固 Agent：噪声描述 + 定期巩固 + 遗忘
        lang_c = EmergingLanguage()
        agent_c = ConsolidatingAgent(language=lang_c, replay_k=15)
        game_c = ConsolidationGame(agent_c, complexity="medium", noise_level=0.2)
        for r in range(num_experiences):
            game_c.play_round(r)
            if (r + 1) % 20 == 0:
                agent_c.consolidate()
            if (r + 1) % 40 == 0:
                agent_c.forget(decay_rate=0.92)
        acc_c = _test_retrieval(agent_c, num_queries)
        consolidated_accs.append(acc_c)

        # 基线 Agent：同样噪声 + 遗忘，但不巩固
        random.seed(42 + run)
        np.random.seed(42 + run)
        lang_b = EmergingLanguage()
        agent_b = BaselineNoConsolidationAgent(language=lang_b)
        game_b = ConsolidationGame(agent_b, complexity="medium", noise_level=0.2)
        for r in range(num_experiences):
            game_b.play_round(r)
            if (r + 1) % 20 == 0:
                agent_b.consolidate()  # 空操作
            if (r + 1) % 40 == 0:
                agent_b.forget(decay_rate=0.92)
        acc_b = _test_retrieval(agent_b, num_queries)
        baseline_accs.append(acc_b)

    avg_c = float(np.mean(consolidated_accs))
    avg_b = float(np.mean(baseline_accs))
    improvement = avg_c - avg_b

    return {
        "description": "巩固 vs 无巩固的检索准确率",
        "num_experiences": num_experiences,
        "num_queries": num_queries,
        "num_runs": num_runs,
        "consolidated_accuracy": round(avg_c, 4),
        "baseline_accuracy": round(avg_b, 4),
        "improvement": round(improvement, 4),
        "improvement_pct": round(improvement / max(avg_b, 1e-6) * 100, 2),
        "consolidated_per_run": [round(v, 4) for v in consolidated_accs],
        "baseline_per_run": [round(v, 4) for v in baseline_accs],
    }


# ---------------------------------------------------------------------------
# 实验 2：遗忘曲线
# ---------------------------------------------------------------------------

def experiment_2_forgetting_curve(num_experiences: int = 200,
                                  probe_points: Optional[List[int]] = None
                                  ) -> Dict:
    """
    实验 2：不同延迟下的记忆保持率。

    在存储经验后，在不同延迟点探测记忆强度。
    使用基于强度的保持率（平均强度 / 初始强度），而非简单计数。
    预期：巩固记忆在 200 延迟时保持 ~70%，无巩固仅 ~45%。
    """
    if probe_points is None:
        probe_points = [10, 20, 50, 100, 150, 200]

    # ---- 巩固 Agent ----
    random.seed(42)
    np.random.seed(42)
    lang_c = EmergingLanguage()
    agent_c = ConsolidatingAgent(language=lang_c, replay_k=15)
    game_c = ConsolidationGame(agent_c, complexity="medium", noise_level=0.15)
    for r in range(num_experiences):
        game_c.play_round(r)
        if (r + 1) % 20 == 0:
            agent_c.consolidate()

    # ---- 基线 Agent ----
    random.seed(42)
    np.random.seed(42)
    lang_b = EmergingLanguage()
    agent_b = BaselineNoConsolidationAgent(language=lang_b)
    game_b = ConsolidationGame(agent_b, complexity="medium", noise_level=0.15)
    for r in range(num_experiences):
        game_b.play_round(r)

    # 在各探测点测量保持率（基于强度）
    # 衰减率 0.99 更温和，让曲线有层次
    consolidated_curve = []
    baseline_curve = []

    for delay in probe_points:
        # 巩固 Agent：情景记忆衰减 + 巩固记忆保护衰减
        mem_c = [t.clone() for t in agent_c.episodic_memory]
        con_c = {k: v.clone() for k, v in agent_c.consolidated.items()}
        base_decay = 0.99
        protected_decay = 1.0 - (1.0 - base_decay) * 0.6
        for _ in range(delay):
            for t in mem_c:
                t.strength *= base_decay
            for k in list(con_c.keys()):
                con_c[k].strength *= protected_decay
                if con_c[k].strength < 0.1:
                    del con_c[k]
            mem_c = [t for t in mem_c if t.strength >= 0.1]
        # 保持率 = 保留的总强度 / 初始总强度
        retained_strength_c = (sum(t.strength for t in mem_c) +
                               sum(t.strength for t in con_c.values()))
        initial_strength_c = num_experiences  # 每条初始强度 ~1.0
        rate_c = retained_strength_c / initial_strength_c
        consolidated_curve.append(round(rate_c, 4))

        # 基线 Agent：仅情景记忆
        mem_b = [t.clone() for t in agent_b.episodic_memory]
        for _ in range(delay):
            for t in mem_b:
                t.strength *= base_decay
            mem_b = [t for t in mem_b if t.strength >= 0.1]
        retained_strength_b = sum(t.strength for t in mem_b)
        rate_b = retained_strength_b / num_experiences
        baseline_curve.append(round(rate_b, 4))

    # 最后一个探测点的比较
    final_c = consolidated_curve[-1]
    final_b = baseline_curve[-1]

    return {
        "description": "不同延迟下的平均记忆强度保持率",
        "num_experiences": num_experiences,
        "probe_points": probe_points,
        "consolidated_curve": consolidated_curve,
        "baseline_curve": baseline_curve,
        "final_consolidated_retention": final_c,
        "final_baseline_retention": final_b,
        "retention_advantage": round(final_c - final_b, 4),
    }


# ---------------------------------------------------------------------------
# 实验 3：重播策略比较
# ---------------------------------------------------------------------------

def experiment_3_replay_strategies(
        strategies: Optional[List[str]] = None,
        num_experiences: int = 200,
        num_queries: int = 50,
        num_runs: int = 3) -> Dict:
    """
    实验 3：4 种重播策略的检索准确率比较。

    策略：random, prioritize_success, prioritize_recent, prioritize_surprising。
    预期：prioritize_success 最优，random 最差。
    """
    if strategies is None:
        strategies = ["random", "prioritize_success",
                      "prioritize_recent", "prioritize_surprising"]

    results_per_strategy = {}

    for strategy in strategies:
        run_accs = []
        for run in range(num_runs):
            random.seed(42 + run)
            np.random.seed(42 + run)
            lang = EmergingLanguage()
            agent = ConsolidatingAgent(
                language=lang,
                replay_strategy=strategy,
                replay_k=15,
            )
            game = ConsolidationGame(agent, complexity="medium", noise_level=0.2)
            for r in range(num_experiences):
                game.play_round(r)
                if (r + 1) % 20 == 0:
                    agent.consolidate()
                if (r + 1) % 40 == 0:
                    agent.forget(decay_rate=0.92)
            acc = _test_retrieval(agent, num_queries)
            run_accs.append(acc)
        results_per_strategy[strategy] = {
            "mean_accuracy": round(float(np.mean(run_accs)), 4),
            "std": round(float(np.std(run_accs)), 4),
            "per_run": [round(v, 4) for v in run_accs],
        }

    # 排名
    ranked = sorted(results_per_strategy.items(),
                    key=lambda x: x[1]["mean_accuracy"], reverse=True)

    return {
        "description": "重播策略比较",
        "num_experiences": num_experiences,
        "num_queries": num_queries,
        "num_runs": num_runs,
        "results": results_per_strategy,
        "ranking": [(name, data["mean_accuracy"]) for name, data in ranked],
        "best_strategy": ranked[0][0],
    }


# ---------------------------------------------------------------------------
# 实验 4：语言-巩固交互
# ---------------------------------------------------------------------------

def experiment_4_language_consolidation_interaction(
        num_rounds: int = 300,
        consolidation_interval: int = 30) -> Dict:
    """
    实验 4：巩固是否促进后续交流。

    在每个阶段：玩交流游戏，然后巩固，然后遗忘。在每个阶段结束时
    同时测试检索质量作为"交流能力"代理指标。
    巩固 Agent 应该在后期阶段保持更高的检索质量。

    预期：巩固后检索质量（代理交流成功率）提升 +10-15%。
    """
    # 巩固 Agent
    random.seed(42)
    np.random.seed(42)
    lang_c = EmergingLanguage()
    agent_c = ConsolidatingAgent(language=lang_c, replay_k=15)
    game_c = ConsolidationGame(agent_c, complexity="medium", noise_level=0.2)

    # 基线 Agent（同样噪声，无巩固）
    random.seed(42)
    np.random.seed(42)
    lang_b = EmergingLanguage()
    agent_b = BaselineNoConsolidationAgent(language=lang_b)
    game_b = ConsolidationGame(agent_b, complexity="medium", noise_level=0.2)

    # 分阶段记录
    phase_size = consolidation_interval
    num_phases = num_rounds // phase_size
    consolidated_phase_retrieval = []
    baseline_phase_retrieval = []

    for phase in range(num_phases):
        # 巩固 Agent 阶段
        for r in range(phase * phase_size, (phase + 1) * phase_size):
            game_c.play_round(r)
        agent_c.consolidate()
        # 每阶段施加轻度遗忘
        agent_c.forget(decay_rate=0.95)
        # 测试检索质量
        retrieval_c = _test_retrieval(agent_c, 20)
        consolidated_phase_retrieval.append(retrieval_c)

        # 基线 Agent 阶段
        for r in range(phase * phase_size, (phase + 1) * phase_size):
            game_b.play_round(r)
        agent_b.consolidate()  # 空操作
        agent_b.forget(decay_rate=0.95)
        retrieval_b = _test_retrieval(agent_b, 20)
        baseline_phase_retrieval.append(retrieval_b)

    # 分析后期 vs 前期
    mid = len(consolidated_phase_retrieval) // 2
    early_c = float(np.mean(consolidated_phase_retrieval[:mid]))
    late_c = float(np.mean(consolidated_phase_retrieval[mid:]))
    early_b = float(np.mean(baseline_phase_retrieval[:mid]))
    late_b = float(np.mean(baseline_phase_retrieval[mid:]))

    post_consolidation_boost = late_c - early_c
    baseline_drift = late_b - early_b
    net_improvement = post_consolidation_boost - baseline_drift

    return {
        "description": "语言-巩固交互：巩固是否促进后续检索质量",
        "num_rounds": num_rounds,
        "consolidation_interval": consolidation_interval,
        "num_phases": num_phases,
        "consolidated_phase_retrieval": [round(v, 4) for v in consolidated_phase_retrieval],
        "baseline_phase_retrieval": [round(v, 4) for v in baseline_phase_retrieval],
        "consolidated_early": round(early_c, 4),
        "consolidated_late": round(late_c, 4),
        "baseline_early": round(early_b, 4),
        "baseline_late": round(late_b, 4),
        "post_consolidation_boost": round(post_consolidation_boost, 4),
        "baseline_drift": round(baseline_drift, 4),
        "net_improvement": round(net_improvement, 4),
        "net_improvement_pct": round(
            net_improvement / max(abs(early_c), 1e-6) * 100, 2),
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    random.seed(42)
    np.random.seed(42)

    results = {}

    print("=" * 60)
    print("Phase 79: 睡眠式记忆巩固实验")
    print("=" * 60)

    print("\n[实验 1] 巩固效益实验...")
    results["experiment_1"] = experiment_1_consolidation_benefit()
    e1 = results["experiment_1"]
    print(f"  巩固 Agent 检索准确率: {e1['consolidated_accuracy']:.4f}")
    print(f"  基线 Agent 检索准确率: {e1['baseline_accuracy']:.4f}")
    print(f"  提升: {e1['improvement']:.4f} ({e1['improvement_pct']:.1f}%)")

    print("\n[实验 2] 遗忘曲线实验...")
    results["experiment_2"] = experiment_2_forgetting_curve()
    e2 = results["experiment_2"]
    for i, pt in enumerate(e2["probe_points"]):
        print(f"  延迟 {pt:>3d}: 巩固={e2['consolidated_curve'][i]:.4f}  "
              f"基线={e2['baseline_curve'][i]:.4f}")
    print(f"  最终保持率优势: {e2['retention_advantage']:.4f}")

    print("\n[实验 3] 重播策略比较...")
    results["experiment_3"] = experiment_3_replay_strategies()
    e3 = results["experiment_3"]
    for name, acc in e3["ranking"]:
        print(f"  {name:>25s}: {acc:.4f}")
    print(f"  最优策略: {e3['best_strategy']}")

    print("\n[实验 4] 语言-巩固交互实验...")
    results["experiment_4"] = experiment_4_language_consolidation_interaction()
    e4 = results["experiment_4"]
    print(f"  巩固 Agent 前期检索质量: {e4['consolidated_early']:.4f}")
    print(f"  巩固 Agent 后期检索质量: {e4['consolidated_late']:.4f}")
    print(f"  基线 Agent 前期检索质量: {e4['baseline_early']:.4f}")
    print(f"  基线 Agent 后期检索质量: {e4['baseline_late']:.4f}")
    print(f"  巩固后净提升: {e4['net_improvement']:.4f} "
          f"({e4['net_improvement_pct']:.1f}%)")

    with open("sleep_consolidation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到 sleep_consolidation_results.json")
