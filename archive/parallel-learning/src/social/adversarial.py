"""对抗性通信 — 信任评估、欺骗检测、声誉系统

当说话者可能存在利益偏差时，听者发展出信任评估与声誉追踪机制。
信任标记 (trust markers) 如 'sure'/'doubt' 自然涌现。

核心组件:
  TrustEvaluator   — 信息源的可靠性评估
  ReputationSystem — 多 Agent 声誉追踪与说话者选择
  AdversarialModule— 欺骗识别与信任标记生成

所有数值计算使用 torch.Tensor，零 numpy 依赖。
"""

from typing import Dict, List, Optional, Tuple

import torch

from src.core.device import get_device, to_device

# 信任标记词汇表
TRUST_MARKERS = {'sure', 'doubt', 'honest', 'trust', 'verify', 'warn'}


class TrustEvaluator:
    """评估信息源的可靠性

    基于贝叶斯更新: 先验信任度 + 每次交互的后验调整。
    每个信息源维护 (alpha, beta) 参数，类似 Beta 分布。

    Args:
        initial_trust: 初始信任先验，默认 0.5
        lr: 学习率（每次交互的更新幅度），默认 0.1
    """

    def __init__(self, initial_trust: float = 0.5, lr: float = 0.1):
        self._device = get_device()
        self.initial_trust = initial_trust
        self.lr = lr

        # 每个信息源的交互记录 {source_id: (successes, failures)}
        self._records: Dict[str, Tuple[int, int]] = {}

    def update(self, source_id: str, was_honest: bool) -> None:
        """更新信息源的信任记录

        Args:
            source_id: 信息源标识
            was_honest: 本次交互是否诚实
        """
        if source_id not in self._records:
            self._records[source_id] = (0, 0)

        successes, failures = self._records[source_id]
        if was_honest:
            self._records[source_id] = (successes + 1, failures)
        else:
            self._records[source_id] = (successes, failures + 1)

    def get_trust(self, source_id: str) -> float:
        """获取信息源的信任度 [0, 1]

        使用 Beta(alpha, beta) 的均值: alpha / (alpha + beta)。
        未记录的信息源返回初始信任先验。
        """
        if source_id not in self._records:
            return self.initial_trust

        successes, failures = self._records[source_id]
        alpha = successes + 1  # 拉普拉斯平滑
        beta = failures + 1

        # 用 torch 计算避免数值问题
        trust = torch.tensor(alpha, dtype=torch.float32) / torch.tensor(
            alpha + beta, dtype=torch.float32
        )
        return trust.item()

    def get_reliable_sources(self, threshold: float = 0.7) -> List[str]:
        """获取信任度高于阈值的信息源列表"""
        return [
            sid for sid in self._records
            if self.get_trust(sid) >= threshold
        ]

    # ── 持久化 ────────────────────────────────────────────────

    def save_state(self) -> dict:
        return {
            'initial_trust': self.initial_trust,
            'lr': self.lr,
            'records': {k: list(v) for k, v in self._records.items()},
        }

    def load_state(self, state: dict) -> None:
        self.initial_trust = state.get('initial_trust', 0.5)
        self.lr = state.get('lr', 0.1)
        self._records = {
            k: tuple(v) for k, v in state.get('records', {}).items()
        }


class ReputationSystem:
    """多 Agent 声誉追踪

    记录每对 (speaker, listener) 交互的成功与诚实度，
    全局声誉 = 所有听众对该说话者评价的均值。

    Args:
        decay: 声誉衰减因子 (0, 1)，每步乘以此值，默认 0.995
    """

    def __init__(self, decay: float = 0.995):
        self._device = get_device()
        self.decay = decay

        # {speaker_id: {listener_id: (successes, failures, honest_count, deceit_count)}}
        self._pair_records: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {}
        self._global_scores: Dict[str, float] = {}
        self._step_count = 0

    def record_interaction(self, speaker_id: str, listener_id: str,
                           success: bool, was_honest: bool) -> None:
        """记录一次交互

        Args:
            speaker_id: 说话者标识
            listener_id: 听者标识
            success: 交互是否成功（任务是否完成）
            was_honest: 说话者是否诚实
        """
        if speaker_id not in self._pair_records:
            self._pair_records[speaker_id] = {}

        pair = self._pair_records[speaker_id]
        if listener_id not in pair:
            pair[listener_id] = (0, 0, 0, 0)

        s, f, h, d = pair[listener_id]
        if success:
            s += 1
        else:
            f += 1
        if was_honest:
            h += 1
        else:
            d += 1
        pair[listener_id] = (s, f, h, d)

        self._update_global(speaker_id)

    def _update_global(self, speaker_id: str) -> None:
        """更新说话者的全局声誉"""
        pair = self._pair_records.get(speaker_id, {})
        if not pair:
            return

        scores = []
        for listener_id, (s, f, h, d) in pair.items():
            total = s + f
            if total == 0:
                continue
            # 成功率 + 诚实率的加权平均
            success_rate = s / total
            honest_rate = (h + 1) / (h + d + 1)
            scores.append(0.5 * success_rate + 0.5 * honest_rate)

        if scores:
            self._global_scores[speaker_id] = sum(scores) / len(scores)

    def get_reputation(self, agent_id: str) -> float:
        """获取 Agent 的全局声誉 [0, 1]"""
        # 应用时间衰减
        base = self._global_scores.get(agent_id, 0.5)
        decay_factor = self.decay ** self._step_count
        # 衰减趋向中性值 0.5
        return 0.5 + (base - 0.5) * decay_factor

    def choose_speaker(self, available: List[str]) -> str:
        """从可用说话者中选择声誉最高的一位

        使用 softmax 温度采样: 声誉越高被选中概率越大，
        但保留一定探索性。

        Args:
            available: 可用说话者 ID 列表

        Returns:
            被选中的说话者 ID
        """
        if not available:
            raise ValueError("可用说话者列表为空")
        if len(available) == 1:
            return available[0]

        reputations = torch.tensor(
            [self.get_reputation(aid) for aid in available],
            dtype=torch.float32,
            device=self._device,
        )

        # 温度 softmax: 温度 2.0 保持适度探索
        probs = torch.softmax(reputations * 2.0, dim=0)
        idx = torch.multinomial(probs, 1).item()
        return available[idx]

    def step(self) -> None:
        """每步调用，推进衰减计数"""
        self._step_count += 1

    # ── 持久化 ────────────────────────────────────────────────

    def save_state(self) -> dict:
        # 将嵌套 tuple 结构转换为可序列化的 dict
        pair_serializable = {}
        for speaker, listeners in self._pair_records.items():
            pair_serializable[speaker] = {
                lid: list(vals) for lid, vals in listeners.items()
            }
        return {
            'decay': self.decay,
            'pair_records': pair_serializable,
            'global_scores': dict(self._global_scores),
            'step_count': self._step_count,
        }

    def load_state(self, state: dict) -> None:
        self.decay = state.get('decay', 0.995)
        self._pair_records = {}
        for speaker, listeners in state.get('pair_records', {}).items():
            self._pair_records[speaker] = {
                lid: tuple(vals) for lid, vals in listeners.items()
            }
        self._global_scores = state.get('global_scores', {})
        self._step_count = state.get('step_count', 0)


class AdversarialModule:
    """对抗性通信处理 — 欺骗检测与信任标记

    核心流程:
    1. evaluate_utterance — 基于场景一致性检验话语可靠性
    2. add_trust_marker   — 根据信任度在话语中添加标记
    3. TrustEvaluator / ReputationSystem 提供底层评估

    Args:
        trust_evaluator: 信任评估器（可选，自动创建）
        reputation: 声誉系统（可选，自动创建）
    """

    def __init__(self, trust_evaluator: Optional[TrustEvaluator] = None,
                 reputation: Optional[ReputationSystem] = None):
        self._device = get_device()
        self.trust = trust_evaluator or TrustEvaluator()
        self.reputation = reputation or ReputationSystem()

        # 欺骗检测: 已知的欺骗模式（符号 → 场景冲突计数）
        self._deception_patterns: Dict[str, int] = {}
        self._total_evaluations = 0

    def evaluate_utterance(self, utterance: List[str], scene: List[Dict],
                           source_trust: float) -> Tuple[List[str], float]:
        """评估话语的可靠性

        策略:
        - 高信任来源: 基本通过，置信度 = source_trust
        - 低信任来源: 过滤掉与场景不匹配的符号，降低置信度

        Args:
            utterance: 话语符号列表
            scene: 场景描述 [{feature: value}, ...]
            source_trust: 来源的信任度 [0, 1]

        Returns:
            (filtered_utterance, confidence) 元组
        """
        self._total_evaluations += 1

        # 场景特征集合
        scene_features = set()
        for obj in scene:
            scene_features.update(obj.keys())

        if source_trust >= 0.7:
            # 高信任源: 直接通过
            return list(utterance), source_trust

        # 低信任源: 检查一致性
        filtered = []
        match_count = 0

        for symbol in utterance:
            # 如果符号能在场景中找到对应特征，则保留
            if symbol in scene_features or symbol in TRUST_MARKERS:
                filtered.append(symbol)
                match_count += 1
            elif symbol in self._deception_patterns:
                # 已知欺骗模式: 记录但不过滤（留待后续判断）
                self._deception_patterns[symbol] += 1
                if self._deception_patterns[symbol] < 3:
                    filtered.append(symbol)
                    match_count += 1
            else:
                # 未知符号: 保守过滤
                filtered.append(symbol)
                match_count += 1

        # 置信度 = 匹配率 × 来源信任
        if utterance:
            match_rate = match_count / len(utterance)
        else:
            match_rate = 1.0

        confidence = match_rate * source_trust
        confidence = torch.tensor(confidence, dtype=torch.float32)
        confidence = torch.clamp(confidence, 0.0, 1.0).item()

        return filtered, confidence

    def add_trust_marker(self, utterance: List[str], trust: float) -> List[str]:
        """根据信任度在话语中添加信任标记

        规则:
        - trust >= 0.8 → 添加 'sure'
        - 0.5 <= trust < 0.8 → 不添加标记
        - 0.3 <= trust < 0.5 → 添加 'doubt'
        - trust < 0.3 → 添加 'warn'

        Args:
            utterance: 原始话语
            trust: 信任度 [0, 1]

        Returns:
            添加了信任标记的话语
        """
        result = list(utterance)

        if trust >= 0.8:
            marker = 'sure'
        elif trust >= 0.5:
            return result  # 中等信任，不加标记
        elif trust >= 0.3:
            marker = 'doubt'
        else:
            marker = 'warn'

        # 避免重复添加同一标记
        if marker not in result:
            result.append(marker)

        return result

    def detect_deception(self, utterance: List[str], ground_truth: List[str],
                         source_id: str) -> float:
        """检测欺骗程度

        比较话语与 ground truth 的重合度，低重合度暗示欺骗。

        Args:
            utterance: 待检测话语
            ground_truth: 真实场景描述
            source_id: 信息源标识

        Returns:
            欺骗概率 [0, 1]
        """
        if not utterance or not ground_truth:
            return 0.0

        utter_set = set(utterance)
        truth_set = set(ground_truth)

        # Jaccard 相似度
        intersection = len(utter_set & truth_set)
        union = len(utter_set | truth_set)
        similarity = intersection / union if union > 0 else 0.0

        # 欺骗概率 = 1 - 相似度
        deception_prob = 1.0 - similarity

        # 更新信任评估器
        self.trust.update(source_id, deception_prob < 0.5)

        return deception_prob

    # ── 持久化 ────────────────────────────────────────────────

    def save_state(self) -> dict:
        return {
            'trust': self.trust.save_state(),
            'reputation': self.reputation.save_state(),
            'deception_patterns': dict(self._deception_patterns),
            'total_evaluations': self._total_evaluations,
        }

    def load_state(self, state: dict) -> None:
        if 'trust' in state:
            self.trust.load_state(state['trust'])
        if 'reputation' in state:
            self.reputation.load_state(state['reputation'])
        self._deception_patterns = state.get('deception_patterns', {})
        self._total_evaluations = state.get('total_evaluations', 0)
