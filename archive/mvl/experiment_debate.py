"""
Phase 75: 多 Agent 辩论与说服 —— 论证标记涌现

核心思想：
论证标记（"because"/"but"/"so"/"wrong"）从辩论压力中涌现。
当 Agent 用证据支持论点时，比单纯重复主张更有说服力。

本阶段测试：
- 论证标记是否从辩论交互中自发涌现
- 结构化论证（claim + evidence）是否优于重复主张
- 论证复杂度是否随交互增长
- 多 Agent 辩论是否达成共识

涌现条件：
1. 场景有多种合理解读，需要论证才能消歧
2. 提供证据的 Agent 比重复主张的 Agent 更有说服力
3. 论证标记提供结构化优势

理论依据：
-论证语用学（Argumentative Pragmatics）
- 说服的知识模型（Persuasion Knowledge Model）
- 信念修正（Belief Revision）
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from language_emergence import EmergingLanguage, _symbol_category, COLORS, SHAPES


# ============================================================
# 辩论话题
# ============================================================

class DebateTopic:
    """
    可辩论的场景解读

    场景有歧义：多个合理解读，但只有一个是正确的。
    不同 Agent 看到不同的证据片段。
    """

    def __init__(self):
        self.scene_description: Dict = {}
        self.possible_interpretations: List[Dict] = []
        self.correct_interpretation: int = 0

    @classmethod
    def generate(cls) -> 'DebateTopic':
        """生成一个有歧义的场景话题"""
        topic = cls()

        colors = list(COLORS)
        shapes = list(SHAPES)

        # 生成核心物体（有歧义的）
        obj_color = random.choice(colors)
        obj_shape = random.choice(shapes)

        # 场景描述：包含模糊证据
        topic.scene_description = {
            'primary_color': obj_color,
            'primary_shape': obj_shape,
            'secondary_color': random.choice(colors),
            'secondary_shape': random.choice(shapes),
            'context': random.choice(['indoor', 'outdoor', 'underwater', 'space']),
            'lighting': random.choice(['bright', 'dim', 'shadowed']),
        }

        # 生成 3 种合理解读
        n_interp = 3
        topic.possible_interpretations = []
        for i in range(n_interp):
            interp = {
                'id': f'interp_{i}',
                'label': f'解读{i+1}',
                'description': f'{obj_color} {obj_shape} 是 {random.choice(["工具", "装饰", "武器", "容器", "信号"])}',
                'evidence_favor': random.uniform(0.2, 0.8),
            }
            topic.possible_interpretations.append(interp)

        # 随机选定正确解读
        topic.correct_interpretation = random.randint(0, n_interp - 1)

        # 为正确解读分配更多证据权重
        topic.possible_interpretations[topic.correct_interpretation]['evidence_favor'] = \
            min(0.95, topic.possible_interpretations[topic.correct_interpretation]['evidence_favor'] + 0.2)

        return topic

    def get_evidence_for(self, interp_id: str) -> Dict[str, float]:
        """
        为特定解读生成证据片段

        不同 Agent 看到的证据可能不同（信息不对称）
        """
        evidence = {}
        scene = self.scene_description

        # 基础特征证据
        features = ['primary_color', 'primary_shape', 'context', 'lighting']
        for feat in features:
            val = scene.get(feat, '')
            # 每个特征对每种解读有不同的支持度
            support = random.uniform(0.1, 0.9)
            evidence[f'{feat}={val}'] = round(support, 2)

        return evidence

    def check_correct(self, interp_id: str) -> bool:
        """检查解读是否正确"""
        idx = int(interp_id.split('_')[1])
        return idx == self.correct_interpretation


# ============================================================
# 论证结构
# ============================================================

class Argument:
    """
    结构化论证

    claim = 主张
    support = 支持证据列表
    markers_used = 使用的论证标记
    quality = 论证质量分数
    """

    # 论证标记池
    MARKER_BECAUSE = 'because'
    MARKER_BUT = 'but'
    MARKER_SO = 'so'
    MARKER_WRONG = 'wrong'
    ALL_MARKERS = {MARKER_BECAUSE, MARKER_BUT, MARKER_SO, MARKER_WRONG}

    def __init__(self, claim: str, support: Optional[List[str]] = None,
                 markers_used: Optional[List[str]] = None):
        self.claim = claim
        self.support = support or []
        self.markers_used = markers_used or []
        self.quality: float = 0.0

    def compute_quality(self) -> float:
        """
        计算论证质量

        基于：
        - 证据数量
        - 标记多样性
        - 论证结构完整性
        """
        score = 0.0

        # 基础分：有主张
        score += 0.2

        # 每条证据加分
        score += min(len(self.support) * 0.2, 0.4)

        # 标记多样性加分
        unique_markers = set(self.markers_used)
        if self.MARKER_BECAUSE in unique_markers:
            score += 0.15  # 因果连接
        if self.MARKER_BUT in unique_markers:
            score += 0.1   # 反驳能力
        if self.MARKER_SO in unique_markers:
            score += 0.1   # 推理能力
        if self.MARKER_WRONG in unique_markers:
            score += 0.05  # 否定能力

        self.quality = min(score, 1.0)
        return self.quality

    def to_utterance(self) -> List[str]:
        """转换为符号序列"""
        tokens = [self.claim]

        if self.support:
            for i, supp in enumerate(self.support):
                # 在第一条证据前插入 "because"
                if i == 0 and self.MARKER_BECAUSE in self.markers_used:
                    tokens.append(self.MARKER_BECAUSE)
                # 在第二条证据前插入 "but"（让步反驳）
                if i == 1 and self.MARKER_BUT in self.markers_used:
                    tokens.append(self.MARKER_BUT)
                tokens.append(supp)

            # 在证据末尾插入 "so"（推理结论）
            if self.MARKER_SO in self.markers_used:
                tokens.append(self.MARKER_SO)

        # "wrong" 可以独立于证据出现（否定其他解读）
        if self.MARKER_WRONG in self.markers_used:
            tokens.append(self.MARKER_WRONG)

        return tokens


# ============================================================
# 信念 Agent
# ============================================================

class BeliefAgent:
    """
    带信念的辩论 Agent

    维护对各种解读的置信度，通过论证更新信念。
    """

    def __init__(self, agent_id: int, language: EmergingLanguage):
        self.agent_id = agent_id
        self.language = language
        self.beliefs: Dict[str, float] = {}
        self.argument_markers: Dict[str, Dict] = {
            'because': {'used': 0, 'success': 0},
            'but':     {'used': 0, 'success': 0},
            'so':      {'used': 0, 'success': 0},
            'wrong':   {'used': 0, 'success': 0},
        }
        self.argument_history: List[Argument] = []
        self.persuasion_successes: int = 0
        self.total_arguments: int = 0
        # 经验驱动的标记采纳概率（初期低，随成功使用增长）
        self.marker_adoption = {
            'because': 0.3,   # 基础因果标记，起步较快
            'but':     0.15,  # 反驳标记
            'so':      0.15,  # 推理标记
            'wrong':   0.15,  # 否定标记
        }

    def initialize_beliefs(self, interpretations: List[Dict],
                           evidence_bias: Optional[Dict[str, float]] = None):
        """初始化信念（带可能的偏置）"""
        self.beliefs = {}
        n = len(interpretations)
        for interp in interpretations:
            iid = interp['id']
            if evidence_bias and iid in evidence_bias:
                self.beliefs[iid] = evidence_bias[iid]
            else:
                self.beliefs[iid] = 1.0 / n

        # 归一化
        self._normalize_beliefs()

    def _normalize_beliefs(self):
        """归一化信念"""
        total = sum(self.beliefs.values())
        if total > 0:
            for k in self.beliefs:
                self.beliefs[k] /= total

    def construct_argument(self, interpretation: str,
                           evidence: Dict[str, float]) -> List[str]:
        """
        为特定解读构建论证

        标记使用概率由 marker_adoption 控制（经验驱动），
        每次成功使用后概率上升，模拟真实语言习得的渐进过程。
        """
        # 选择最支持的证据
        sorted_evidence = sorted(evidence.items(), key=lambda x: x[1], reverse=True)
        # 经验越多，使用的证据越多
        max_evidence = 1 + min(self.total_arguments // 20, 2)
        top_evidence = sorted_evidence[:min(max_evidence, len(sorted_evidence))]

        # 基于采纳概率决定标记使用
        markers = []

        # "because": 连接证据，随经验增长
        if top_evidence and random.random() < self.marker_adoption['because']:
            markers.append(Argument.MARKER_BECAUSE)

        # "so": 推理总结，需要多条证据
        if len(top_evidence) >= 2 and random.random() < self.marker_adoption['so']:
            markers.append(Argument.MARKER_SO)

        # "but": 让步反驳，需要置信度不够高
        confidence = self.beliefs.get(interpretation, 0.5)
        if confidence < 0.6 and random.random() < self.marker_adoption['but']:
            markers.append(Argument.MARKER_BUT)

        # "wrong": 否定其他解读，需要其他解读有竞争力
        other_max = max((v for k, v in self.beliefs.items() if k != interpretation),
                        default=0.0)
        if other_max >= 0.25 and random.random() < self.marker_adoption['wrong']:
            markers.append(Argument.MARKER_WRONG)

        # 构建论证
        support_items = [f'{k}:{v:.1f}' for k, v in top_evidence]

        arg = Argument(
            claim=interpretation,
            support=support_items,
            markers_used=markers,
        )
        arg.compute_quality()
        self.argument_history.append(arg)
        self.total_arguments += 1

        # 更新标记使用统计
        for m in markers:
            self.argument_markers[m]['used'] += 1
            # 成功使用增加未来采纳概率（学习效应）
            self.marker_adoption[m] = min(self.marker_adoption[m] + 0.008, 0.9)

        # 记录到语言系统
        utterance = arg.to_utterance()
        self.language.record_usage(utterance, True)

        return utterance

    def evaluate_argument(self, argument_utterance: List[str]) -> float:
        """
        评估收到的论证的说服力分数

        基于：
        - 是否包含 "because"（证据连接）
        - 是否包含 "so"（推理结论）
        - 论证长度（更多 = 更详细）
        - 是否有 "wrong"（直接反驳）
        """
        score = 0.3  # 基础分

        if Argument.MARKER_BECAUSE in argument_utterance:
            score += 0.25  # 证据支持
        if Argument.MARKER_SO in argument_utterance:
            score += 0.2   # 推理结论
        if Argument.MARKER_BUT in argument_utterance:
            score += 0.1   # 让步反驳
        if Argument.MARKER_WRONG in argument_utterance:
            score += 0.05  # 直接否定

        # 长度加分（更多证据 = 更好）
        score += min(len(argument_utterance) * 0.05, 0.2)

        # 小随机扰动
        score += random.uniform(-0.05, 0.05)

        return min(max(score, 0.0), 1.0)

    def update_belief(self, argument_score: float, interpretation: str):
        """
        Bayesian 更新信念

        高说服力的论证提高对应解读的置信度
        """
        if interpretation not in self.beliefs:
            return

        prior = self.beliefs[interpretation]

        # 似然：论证分数越高，该解读为真的概率越大
        likelihood = 0.5 + argument_score * 0.4

        # Bayesian 更新
        posterior = (prior * likelihood) / (
            prior * likelihood +
            (1 - prior) * (1 - likelihood + 0.01)
        )

        self.beliefs[interpretation] = np.clip(posterior, 0.01, 0.99)
        self._normalize_beliefs()

        # 更新标记成功统计
        if argument_score > 0.5:
            self.persuasion_successes += 1

    def get_persuasion_power(self) -> float:
        """
        说服力：基于论证质量历史

        更多成功论证 = 更高说服力
        """
        if self.total_arguments == 0:
            return 0.3
        return 0.3 + 0.7 * (self.persuasion_successes / self.total_arguments)

    def get_top_belief(self) -> Tuple[str, float]:
        """返回最高置信度及其解读"""
        if not self.beliefs:
            return ('', 0.0)
        best = max(self.beliefs.items(), key=lambda x: x[1])
        return best

    def get_marker_summary(self) -> Dict[str, float]:
        """获取标记使用率"""
        total = max(sum(m['used'] for m in self.argument_markers.values()), 1)
        return {
            m: self.argument_markers[m]['used'] / total
            for m in self.argument_markers
        }


# ============================================================
# 强力 Agent（对照组）
# ============================================================

class ForcefulAgent:
    """
    对照组：重复主张而不提供证据

    没有论证结构，只是增加重复次数。
    测试假设：论证结构 > 单纯重复
    """

    def __init__(self, agent_id: int, language: EmergingLanguage):
        self.agent_id = agent_id
        self.language = language
        self.beliefs: Dict[str, float] = {}
        self.repetition_count: int = 0

    def initialize_beliefs(self, interpretations: List[Dict],
                           evidence_bias: Optional[Dict[str, float]] = None):
        """初始化信念"""
        self.beliefs = {}
        n = len(interpretations)
        for interp in interpretations:
            iid = interp['id']
            if evidence_bias and iid in evidence_bias:
                self.beliefs[iid] = evidence_bias[iid]
            else:
                self.beliefs[iid] = 1.0 / n
        self._normalize_beliefs()

    def _normalize_beliefs(self):
        total = sum(self.beliefs.values())
        if total > 0:
            for k in self.beliefs:
                self.beliefs[k] /= total

    def forceful_claim(self, interpretation: str) -> List[str]:
        """
        强力主张：重复同一声明多次

        没有证据、没有标记，只有重复
        """
        # 重复 2-4 次
        reps = random.randint(2, 4)
        self.repetition_count += reps

        utterance = [interpretation] * reps

        self.language.record_usage(utterance, True)
        return utterance

    def evaluate_argument(self, argument_utterance: List[str]) -> float:
        """评估论证（靠重复量评分，而非结构）"""
        # 对强力 Agent 来说，重复 = 力量
        score = 0.2 + min(len(argument_utterance) * 0.1, 0.3)
        score += random.uniform(-0.05, 0.05)
        return min(max(score, 0.0), 1.0)

    def update_belief(self, argument_score: float, interpretation: str):
        """更新信念（更容易被重复说服，而非证据）"""
        if interpretation not in self.beliefs:
            return

        prior = self.beliefs[interpretation]
        # 弱更新：对论证结构不敏感
        likelihood = 0.5 + argument_score * 0.2

        posterior = (prior * likelihood) / (
            prior * likelihood +
            (1 - prior) * (1 - likelihood + 0.01)
        )

        self.beliefs[interpretation] = np.clip(posterior, 0.01, 0.99)
        self._normalize_beliefs()

    def get_top_belief(self) -> Tuple[str, float]:
        if not self.beliefs:
            return ('', 0.0)
        return max(self.beliefs.items(), key=lambda x: x[1])

    def get_persuasion_power(self) -> float:
        return 0.3  # 固定低说服力


# ============================================================
# 辩论游戏
# ============================================================

class DebateGame:
    """
    双 Agent 辩论游戏

    3 轮辩论：
    Round 1: Agent A 提出主张
    Round 2: Agent B 反驳或支持
    Round 3: Agent A 回应

    追踪信念收敛和论证标记使用
    """

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.argument_history: List[Dict] = []
        self.marker_usage: Dict[str, int] = defaultdict(int)
        self.total_debates: int = 0
        self.convergence_history: List[float] = []
        self.accuracy_history: List[float] = []

    def play_debate(self, agent_a: BeliefAgent, agent_b: BeliefAgent,
                    topic: DebateTopic) -> Dict:
        """
        进行一轮完整辩论

        Returns:
            辩论结果字典
        """
        self.total_debates += 1
        round_args = []

        # 获取两个 Agent 的初始信念
        belief_a_initial = dict(agent_a.beliefs)
        belief_b_initial = dict(agent_b.beliefs)

        # --- Round 1: Agent A 提出主张 ---
        top_a, conf_a = agent_a.get_top_belief()
        evidence_a = topic.get_evidence_for(top_a)
        utterance_a1 = agent_a.construct_argument(top_a, evidence_a)

        # 记录标记使用
        for sym in utterance_a1:
            if sym in Argument.ALL_MARKERS:
                self.marker_usage[sym] += 1

        round_args.append({
            'round': 1,
            'agent': agent_a.agent_id,
            'interpretation': top_a,
            'utterance': utterance_a1,
        })

        # --- Round 2: Agent B 评估并回应 ---
        score_a1 = agent_b.evaluate_argument(utterance_a1)
        agent_b.update_belief(score_a1, top_a)

        top_b, conf_b = agent_b.get_top_belief()
        evidence_b = topic.get_evidence_for(top_b)
        utterance_b = agent_b.construct_argument(top_b, evidence_b)

        for sym in utterance_b:
            if sym in Argument.ALL_MARKERS:
                self.marker_usage[sym] += 1

        round_args.append({
            'round': 2,
            'agent': agent_b.agent_id,
            'interpretation': top_b,
            'utterance': utterance_b,
        })

        # --- Round 3: Agent A 回应 ---
        score_b = agent_a.evaluate_argument(utterance_b)
        agent_a.update_belief(score_b, top_b)

        top_a2, conf_a2 = agent_a.get_top_belief()
        evidence_a2 = topic.get_evidence_for(top_a2)
        utterance_a2 = agent_a.construct_argument(top_a2, evidence_a2)

        for sym in utterance_a2:
            if sym in Argument.ALL_MARKERS:
                self.marker_usage[sym] += 1

        round_args.append({
            'round': 3,
            'agent': agent_a.agent_id,
            'interpretation': top_a2,
            'utterance': utterance_a2,
        })

        # Agent B 最终更新
        score_a2 = agent_b.evaluate_argument(utterance_a2)
        agent_b.update_belief(score_a2, top_a2)

        # 标记成功
        for agent in [agent_a, agent_b]:
            for m in agent.argument_markers:
                if agent.argument_markers[m]['used'] > 0:
                    agent.argument_markers[m]['success'] += int(
                        (agent_a.get_persuasion_power() + agent_b.get_persuasion_power()) / 2
                    )

        # 记录历史
        self.argument_history.extend(round_args)

        # 计算信念收敛度
        convergence = self._compute_convergence(agent_a, agent_b)
        self.convergence_history.append(convergence)

        # 计算准确率
        correct = topic.correct_interpretation
        correct_id = f'interp_{correct}'
        top_a_final, _ = agent_a.get_top_belief()
        top_b_final, _ = agent_b.get_top_belief()
        accuracy = (1.0 if top_a_final == correct_id else 0.0) + \
                   (1.0 if top_b_final == correct_id else 0.0)
        accuracy /= 2.0
        self.accuracy_history.append(accuracy)

        return {
            'debate_id': self.total_debates,
            'rounds': round_args,
            'convergence': convergence,
            'accuracy': accuracy,
            'belief_a_final': dict(agent_a.beliefs),
            'belief_b_final': dict(agent_b.beliefs),
            'correct_interp': correct_id,
        }

    def play_debate_forceful(self, belief_agent: BeliefAgent,
                             forceful_agent: ForcefulAgent,
                             topic: DebateTopic) -> Dict:
        """
        信念 Agent vs 强力 Agent 的辩论
        """
        self.total_debates += 1

        # Round 1: 信念 Agent 论证
        top_ba, _ = belief_agent.get_top_belief()
        evidence = topic.get_evidence_for(top_ba)
        utt_ba = belief_agent.construct_argument(top_ba, evidence)

        for sym in utt_ba:
            if sym in Argument.ALL_MARKERS:
                self.marker_usage[sym] += 1

        # Round 2: 强力 Agent 回应
        score_ba = forceful_agent.evaluate_argument(utt_ba)
        forceful_agent.update_belief(score_ba, top_ba)

        top_fa, _ = forceful_agent.get_top_belief()
        utt_fa = forceful_agent.forceful_claim(top_fa)

        # Round 3: 信念 Agent 再论证
        score_fa = belief_agent.evaluate_argument(utt_fa)
        belief_agent.update_belief(score_fa, top_fa)

        top_ba2, _ = belief_agent.get_top_belief()
        evidence2 = topic.get_evidence_for(top_ba2)
        utt_ba2 = belief_agent.construct_argument(top_ba2, evidence2)

        for sym in utt_ba2:
            if sym in Argument.ALL_MARKERS:
                self.marker_usage[sym] += 1

        # 强力 Agent 最终更新
        score_ba2 = forceful_agent.evaluate_argument(utt_ba2)
        forceful_agent.update_belief(score_ba2, top_ba2)

        # 收敛和准确率
        convergence = self._compute_convergence_mixed(belief_agent, forceful_agent)
        self.convergence_history.append(convergence)

        correct = topic.correct_interpretation
        correct_id = f'interp_{correct}'
        top_ba_f, _ = belief_agent.get_top_belief()
        top_fa_f, _ = forceful_agent.get_top_belief()
        accuracy = (1.0 if top_ba_f == correct_id else 0.0) + \
                   (1.0 if top_fa_f == correct_id else 0.0)
        accuracy /= 2.0
        self.accuracy_history.append(accuracy)

        return {
            'convergence': convergence,
            'accuracy': accuracy,
        }

    def _compute_convergence(self, agent_a: BeliefAgent,
                             agent_b: BeliefAgent) -> float:
        """计算两个 Agent 的信念收敛度"""
        all_keys = set(agent_a.beliefs.keys()) | set(agent_b.beliefs.keys())
        if not all_keys:
            return 0.0
        diff = sum(abs(agent_a.beliefs.get(k, 0) - agent_b.beliefs.get(k, 0))
                   for k in all_keys)
        return 1.0 - diff / 2.0  # 归一化到 [0, 1]

    def _compute_convergence_mixed(self, agent_a: BeliefAgent,
                                   agent_b: ForcefulAgent) -> float:
        """计算 BeliefAgent 和 ForcefulAgent 的收敛度"""
        all_keys = set(agent_a.beliefs.keys()) | set(agent_b.beliefs.keys())
        if not all_keys:
            return 0.0
        diff = sum(abs(agent_a.beliefs.get(k, 0) - agent_b.beliefs.get(k, 0))
                   for k in all_keys)
        return 1.0 - diff / 2.0

    def get_marker_stats(self) -> Dict[str, float]:
        """获取标记使用频率"""
        total = sum(self.marker_usage.values())
        if total == 0:
            return {m: 0.0 for m in Argument.ALL_MARKERS}
        return {m: self.marker_usage.get(m, 0) / total for m in Argument.ALL_MARKERS}


# ============================================================
# 实验 1：论证标记涌现
# ============================================================

def experiment_1_argument_markers(num_rounds: int = 300):
    """
    实验 1：追踪 "because"/"but"/"so" 标记涌现

    两个 BeliefAgent 进行多轮辩论，观察论证标记是否自发涌现。
    """
    print("=" * 60)
    print("实验 1: 论证标记涌现")
    print("=" * 60)

    language = EmergingLanguage()
    game = DebateGame(language)

    # 创建两个 Agent
    agent_a = BeliefAgent(0, EmergingLanguage())
    agent_b = BeliefAgent(1, EmergingLanguage())

    snapshots = []
    marker_emergence = {m: -1 for m in Argument.ALL_MARKERS}

    for r in range(num_rounds):
        # 生成话题
        topic = DebateTopic.generate()

        # 初始化信念（带信息不对称）
        bias_a = {f'interp_{i}': random.uniform(0.1, 0.7)
                  for i in range(len(topic.possible_interpretations))}
        bias_b = {f'interp_{i}': random.uniform(0.1, 0.7)
                  for i in range(len(topic.possible_interpretations))}

        agent_a.initialize_beliefs(topic.possible_interpretations, bias_a)
        agent_b.initialize_beliefs(topic.possible_interpretations, bias_b)

        # 进行辩论
        result = game.play_debate(agent_a, agent_b, topic)

        # 检查标记涌现
        for m in Argument.ALL_MARKERS:
            if marker_emergence[m] == -1 and game.marker_usage.get(m, 0) > 0:
                marker_emergence[m] = r

        # 每 50 轮打印快照
        if (r + 1) % 50 == 0:
            conv = np.mean(game.convergence_history[-50:]) if game.convergence_history else 0
            acc = np.mean(game.accuracy_history[-50:]) if game.accuracy_history else 0
            marker_stats = game.get_marker_stats()

            snapshot = {
                'round': r + 1,
                'convergence': round(conv, 3),
                'accuracy': round(acc, 3),
                'markers': {m: round(v, 3) for m, v in marker_stats.items()},
            }
            snapshots.append(snapshot)

            print(f"  Round {r+1}: "
                  f"收敛={conv:.3f}, "
                  f"准确率={acc:.3f}, "
                  f"标记={dict(marker_stats)}")

    # 最终统计
    final_convergence = np.mean(game.convergence_history[-50:])
    final_accuracy = np.mean(game.accuracy_history[-50:])
    final_markers = game.get_marker_stats()

    print(f"\n最终结果:")
    print(f"  平均收敛度: {final_convergence:.3f}")
    print(f"  平均准确率: {final_accuracy:.3f}")
    print(f"  标记涌现轮次: {marker_emergence}")
    print(f"  最终标记分布: {final_markers}")

    return {
        'final_convergence': round(final_convergence, 3),
        'final_accuracy': round(final_accuracy, 3),
        'marker_emergence_rounds': marker_emergence,
        'final_marker_distribution': {m: round(v, 3) for m, v in final_markers.items()},
        'snapshots': snapshots,
    }


# ============================================================
# 实验 2：论证 vs 重复
# ============================================================

def experiment_2_persuasion_comparison(num_rounds: int = 200, num_runs: int = 5):
    """
    实验 2：BeliefAgent（论证）vs ForcefulAgent（重复）

    比较两种说服策略的信念收敛率和准确率。
    """
    print(f"\n{'=' * 60}")
    print("实验 2: 论证 vs 重复 —— 说服效果对比")
    print(f"{'=' * 60}")

    structured_results = []
    forceful_results = []

    for run in range(num_runs):
        # --- 结构化论证组 ---
        lang_struct = EmergingLanguage()
        game_struct = DebateGame(lang_struct)

        agent_struct_a = BeliefAgent(0, EmergingLanguage())
        agent_struct_b = BeliefAgent(1, EmergingLanguage())

        for r in range(num_rounds):
            topic = DebateTopic.generate()
            bias_a = {f'interp_{i}': random.uniform(0.1, 0.7)
                      for i in range(len(topic.possible_interpretations))}
            bias_b = {f'interp_{i}': random.uniform(0.1, 0.7)
                      for i in range(len(topic.possible_interpretations))}
            agent_struct_a.initialize_beliefs(topic.possible_interpretations, bias_a)
            agent_struct_b.initialize_beliefs(topic.possible_interpretations, bias_b)
            game_struct.play_debate(agent_struct_a, agent_struct_b, topic)

        struct_conv = np.mean(game_struct.convergence_history)
        struct_acc = np.mean(game_struct.accuracy_history)
        structured_results.append({
            'convergence': struct_conv,
            'accuracy': struct_acc,
        })

        # --- 强力重复组 ---
        lang_force = EmergingLanguage()
        game_force = DebateGame(lang_force)

        agent_belief = BeliefAgent(0, EmergingLanguage())
        agent_force = ForcefulAgent(1, EmergingLanguage())

        for r in range(num_rounds):
            topic = DebateTopic.generate()
            bias_b = {f'interp_{i}': random.uniform(0.1, 0.7)
                      for i in range(len(topic.possible_interpretations))}
            bias_f = {f'interp_{i}': random.uniform(0.1, 0.7)
                      for i in range(len(topic.possible_interpretations))}
            agent_belief.initialize_beliefs(topic.possible_interpretations, bias_b)
            agent_force.initialize_beliefs(topic.possible_interpretations, bias_f)
            game_force.play_debate_forceful(agent_belief, agent_force, topic)

        force_conv = np.mean(game_force.convergence_history)
        force_acc = np.mean(game_force.accuracy_history)
        forceful_results.append({
            'convergence': force_conv,
            'accuracy': force_acc,
        })

        print(f"  Run {run+1}/{num_runs}: "
              f"论证组 收敛={struct_conv:.3f}/准确={struct_acc:.3f}, "
              f"重复组 收敛={force_conv:.3f}/准确={force_acc:.3f}")

    # 汇总
    avg_struct_conv = np.mean([r['convergence'] for r in structured_results])
    avg_struct_acc = np.mean([r['accuracy'] for r in structured_results])
    avg_force_conv = np.mean([r['convergence'] for r in forceful_results])
    avg_force_acc = np.mean([r['accuracy'] for r in forceful_results])

    print(f"\n最终对比:")
    print(f"  论证组: 平均收敛={avg_struct_conv:.3f}, 平均准确={avg_struct_acc:.3f}")
    print(f"  重复组: 平均收敛={avg_force_conv:.3f}, 平均准确={avg_force_acc:.3f}")
    print(f"  论证优势: 收敛+{avg_struct_conv - avg_force_conv:.3f}, "
          f"准确+{avg_struct_acc - avg_force_acc:.3f}")

    return {
        'structured_avg_convergence': round(avg_struct_conv, 3),
        'structured_avg_accuracy': round(avg_struct_acc, 3),
        'forceful_avg_convergence': round(avg_force_conv, 3),
        'forceful_avg_accuracy': round(avg_force_acc, 3),
        'convergence_advantage': round(avg_struct_conv - avg_force_conv, 3),
        'accuracy_advantage': round(avg_struct_acc - avg_force_acc, 3),
        'per_run': {
            'structured': structured_results,
            'forceful': forceful_results,
        },
    }


# ============================================================
# 实验 3：论证复杂度演化
# ============================================================

def experiment_3_argument_complexity(num_rounds: int = 500):
    """
    实验 3：追踪论证结构演化

    从简单主张 → claim+because → claim+because+but+so
    测量论证复杂度的增长轨迹。
    """
    print(f"\n{'=' * 60}")
    print("实验 3: 论证复杂度演化")
    print(f"{'=' * 60}")

    language = EmergingLanguage()
    game = DebateGame(language)

    agent_a = BeliefAgent(0, EmergingLanguage())
    agent_b = BeliefAgent(1, EmergingLanguage())

    complexity_history = []  # 每轮的平均论证复杂度
    snapshots = []

    # 复杂度等级定义
    def measure_complexity(utterance: List[str]) -> int:
        """测量论证复杂度 (0-4)"""
        level = 0
        if len(utterance) > 1:
            level = 1  # 有主张 + 至少一个元素
        if Argument.MARKER_BECAUSE in utterance:
            level = max(level, 2)  # 有证据连接
        if Argument.MARKER_BUT in utterance or Argument.MARKER_SO in utterance:
            level = max(level, 3)  # 有反驳或推理
        if Argument.MARKER_WRONG in utterance:
            level = max(level, 4)  # 有直接否定
        return level

    for r in range(num_rounds):
        topic = DebateTopic.generate()

        bias_a = {f'interp_{i}': random.uniform(0.1, 0.7)
                  for i in range(len(topic.possible_interpretations))}
        bias_b = {f'interp_{i}': random.uniform(0.1, 0.7)
                  for i in range(len(topic.possible_interpretations))}

        agent_a.initialize_beliefs(topic.possible_interpretations, bias_a)
        agent_b.initialize_beliefs(topic.possible_interpretations, bias_b)

        result = game.play_debate(agent_a, agent_b, topic)

        # 测量本轮复杂度
        round_complexities = []
        for rd in result['rounds']:
            c = measure_complexity(rd['utterance'])
            round_complexities.append(c)

        avg_complexity = np.mean(round_complexities) if round_complexities else 0
        complexity_history.append(avg_complexity)

        # 每 100 轮打印
        if (r + 1) % 100 == 0:
            window = complexity_history[-100:]
            avg = np.mean(window)
            max_c = max(window) if window else 0

            # 统计各复杂度等级的比例
            levels = defaultdict(int)
            for c in window:
                levels[int(c)] += 1
            total_w = len(window)
            level_dist = {f'level_{k}': round(v / total_w, 3)
                          for k, v in sorted(levels.items())}

            marker_stats = game.get_marker_stats()

            snapshot = {
                'round': r + 1,
                'avg_complexity': round(avg, 2),
                'max_complexity': max_c,
                'level_distribution': level_dist,
                'marker_stats': {m: round(v, 3) for m, v in marker_stats.items()},
            }
            snapshots.append(snapshot)

            print(f"  Round {r+1}: "
                  f"平均复杂度={avg:.2f}, "
                  f"最大={max_c}, "
                  f"等级分布={level_dist}")

    # 最终复杂度统计
    final_window = complexity_history[-100:]
    final_avg = np.mean(final_window)
    final_levels = defaultdict(int)
    for c in final_window:
        final_levels[int(c)] += 1
    final_dist = {f'level_{k}': round(v / len(final_window), 3)
                  for k, v in sorted(final_levels.items())}

    # 复杂度增长趋势
    early = np.mean(complexity_history[:50])
    late = np.mean(complexity_history[-50:])
    growth = late - early

    print(f"\n最终结果:")
    print(f"  早期复杂度: {early:.2f}")
    print(f"  晚期复杂度: {late:.2f}")
    print(f"  复杂度增长: {growth:+.2f}")
    print(f"  最终等级分布: {final_dist}")

    return {
        'early_complexity': round(early, 2),
        'late_complexity': round(late, 2),
        'complexity_growth': round(growth, 2),
        'final_level_distribution': final_dist,
        'snapshots': snapshots,
    }


# ============================================================
# 实验 4：多 Agent 共识
# ============================================================

def experiment_4_group_consensus(num_agents: int = 5, num_rounds: int = 300):
    """
    实验 4：多 Agent 辩论共识

    N 个 Agent 从随机信念出发，通过辩论达成共识。
    追踪达成共识的时间和最终准确率。
    """
    print(f"\n{'=' * 60}")
    print(f"实验 4: {num_agents} Agent 群体共识")
    print(f"{'=' * 60}")

    language = EmergingLanguage()

    # 创建 Agent
    agents = [BeliefAgent(i, EmergingLanguage()) for i in range(num_agents)]

    consensus_history = []
    accuracy_history = []
    snapshots = []
    consensus_round = -1

    def compute_group_consensus(agent_list):
        """计算群体共识度：所有 Agent 信念的 pairwise 相似度平均"""
        if len(agent_list) < 2:
            return 1.0

        all_keys = set()
        for a in agent_list:
            all_keys.update(a.beliefs.keys())

        if not all_keys:
            return 0.0

        # 计算所有 Agent 信念的均值向量
        mean_belief = {k: np.mean([a.beliefs.get(k, 0) for a in agent_list])
                       for k in all_keys}

        # 每个 Agent 与均值的平均偏差
        deviations = []
        for a in agent_list:
            dev = sum(abs(a.beliefs.get(k, 0) - mean_belief[k]) for k in all_keys)
            deviations.append(dev / len(all_keys))

        avg_deviation = np.mean(deviations)
        return 1.0 - min(avg_deviation * 2, 1.0)

    def compute_group_accuracy(agent_list, correct_id):
        """计算群体准确率"""
        correct_count = sum(
            1 for a in agent_list
            if a.get_top_belief()[0] == correct_id
        )
        return correct_count / len(agent_list)

    for r in range(num_rounds):
        topic = DebateTopic.generate()

        # 每轮随机初始化信念（带偏置）
        for agent in agents:
            bias = {f'interp_{i}': random.uniform(0.05, 0.6)
                    for i in range(len(topic.possible_interpretations))}
            agent.initialize_beliefs(topic.possible_interpretations, bias)

        # 随机配对辩论
        indices = list(range(num_agents))
        random.shuffle(indices)
        for i in range(0, len(indices) - 1, 2):
            a_idx = indices[i]
            b_idx = indices[i + 1]
            game = DebateGame(language)
            game.play_debate(agents[a_idx], agents[b_idx], topic)

        # 测量共识和准确率
        consensus = compute_group_consensus(agents)
        correct_id = f'interp_{topic.correct_interpretation}'
        accuracy = compute_group_accuracy(agents, correct_id)

        consensus_history.append(consensus)
        accuracy_history.append(accuracy)

        # 检查是否达成共识
        if consensus_round == -1 and consensus > 0.8:
            consensus_round = r

        # 每 50 轮打印
        if (r + 1) % 50 == 0:
            avg_consensus = np.mean(consensus_history[-50:])
            avg_accuracy = np.mean(accuracy_history[-50:])

            snapshot = {
                'round': r + 1,
                'consensus': round(avg_consensus, 3),
                'accuracy': round(avg_accuracy, 3),
            }
            snapshots.append(snapshot)

            print(f"  Round {r+1}: "
                  f"共识度={avg_consensus:.3f}, "
                  f"准确率={avg_accuracy:.3f}")

    # 最终统计
    final_consensus = np.mean(consensus_history[-50:])
    final_accuracy = np.mean(accuracy_history[-50:])

    # 标记汇总
    all_marker_stats = defaultdict(float)
    for agent in agents:
        ms = agent.get_marker_summary()
        for m, v in ms.items():
            all_marker_stats[m] += v
    for m in all_marker_stats:
        all_marker_stats[m] /= num_agents

    print(f"\n最终结果:")
    print(f"  最终共识度: {final_consensus:.3f}")
    print(f"  最终准确率: {final_accuracy:.3f}")
    print(f"  达成共识轮次: {consensus_round}")
    print(f"  Agent 标记使用: {dict(all_marker_stats)}")

    return {
        'num_agents': num_agents,
        'final_consensus': round(final_consensus, 3),
        'final_accuracy': round(final_accuracy, 3),
        'consensus_round': consensus_round,
        'agent_marker_summary': {m: round(v, 3) for m, v in all_marker_stats.items()},
        'snapshots': snapshots,
    }


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    random.seed(42)
    np.random.seed(42)

    results = {}
    results['experiment_1'] = experiment_1_argument_markers()
    results['experiment_2'] = experiment_2_persuasion_comparison()
    results['experiment_3'] = experiment_3_argument_complexity()
    results['experiment_4'] = experiment_4_group_consensus()

    with open('debate_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 debate_results.json")
