"""预测编码Light — 抑制可预测信号，只传输新信息

基于 Nature Communications 2025 论文：
"Predictive Coding Light" — 提出了一种新的脉冲神经网络模型，
不传递预测误差到高层，而是抑制最可预测的脉冲，
仅传输输入的压缩表征。

核心思想：
- 传统预测编码：传递预测误差（prediction error）
- 预测编码Light：抑制可预测信号，只传输新信息
- 优势：大幅降低通信带宽和能耗

对当前系统的应用：
- 学习时：只学习新信息（不可预测的部分）
- 推理时：只关注新信息（抑制已知模式）
- 通信时：只传输新信息（减少冗余）
"""

import torch
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PredictionState:
    """预测状态"""
    predicted: torch.Tensor
    actual: torch.Tensor
    novel_part: torch.Tensor  # 不可预测的部分
    predictable_part: torch.Tensor  # 可预测的部分


class PredictiveCodingLight:
    """预测编码Light系统

    抑制可预测信号，只传输新信息。
    """

    def __init__(self, d_model: int = 128, suppression_threshold: float = 0.5, device: str = 'cpu'):
        self.d_model = d_model
        self.suppression_threshold = suppression_threshold
        self.device = torch.device(device)

        # 预测模型（简单线性预测器）
        self.prediction_weights = torch.randn(d_model, d_model, device=self.device) * 0.01
        self.prediction_bias = torch.zeros(d_model, device=self.device)

        # 历史
        self.prediction_history: List[PredictionState] = []
        self.novelty_history: deque = deque(maxlen=100)

        # 学习率
        self.lr = 0.01

    def predict(self, context: torch.Tensor) -> torch.Tensor:
        """基于上下文预测下一个状态"""
        # 确保在同一设备上
        context = context.to(self.device)
        # 简单线性预测
        predicted = torch.matmul(context, self.prediction_weights) + self.prediction_bias
        return predicted

    def suppress_predictable(self, input_tensor: torch.Tensor,
                           context: torch.Tensor) -> PredictionState:
        """抑制可预测部分，保留新信息

        Args:
            input_tensor: 输入信号
            context: 上下文（用于预测）

        Returns:
            PredictionState: 包含新信息和可预测部分
        """
        # 预测
        predicted = self.predict(context)

        # 计算新信息（不可预测的部分）
        novel_part = input_tensor - predicted

        # 计算可预测部分
        predictable_part = predicted

        # 计算新颖性分数
        novelty_score = torch.norm(novel_part).item()

        # 记录
        state = PredictionState(
            predicted=predicted,
            actual=input_tensor,
            novel_part=novel_part,
            predictable_part=predictable_part,
        )
        self.prediction_history.append(state)
        self.novelty_history.append(novelty_score)

        return state

    def learn_from_prediction(self, state: PredictionState):
        """从预测结果中学习（更新预测模型）"""
        # 使用梯度下降更新预测权重
        error = state.actual - state.predicted

        # 更新权重
        self.prediction_weights += self.lr * torch.outer(
            error, state.actual
        )
        self.prediction_bias += self.lr * error

    def get_novelty_score(self) -> float:
        """获取当前新颖性分数"""
        if not self.novelty_history:
            return 1.0
        return sum(self.novelty_history) / len(self.novelty_history)

    def is_novel(self, input_tensor: torch.Tensor, context: torch.Tensor) -> bool:
        """判断输入是否新颖（不可预测）"""
        state = self.suppress_predictable(input_tensor, context)
        novelty = torch.norm(state.novel_part).item()
        return novelty > self.suppression_threshold

    def get_novel_part(self, input_tensor: torch.Tensor,
                      context: torch.Tensor) -> torch.Tensor:
        """获取输入的新颖部分（抑制可预测部分）"""
        state = self.suppress_predictable(input_tensor, context)
        return state.novel_part

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'prediction_history': len(self.prediction_history),
            'avg_novelty': self.get_novelty_score(),
            'suppression_threshold': self.suppression_threshold,
        }


class RewardRepresentationShift:
    """奖励表征后移 — 信用分配的时间迁移

    基于 Nature 2026 哈佛大学研究：
    "Predictive coding of reward in the hippocampus"

    核心发现：
    - 随着学习经验积累，神经元活动从编码奖励本身
      逐渐转移为编码先行任务特征
    - 即经验驱动神经活动向后偏移以预测奖励

    对当前系统的应用：
    - 学习时：将价值信号从结果逐步迁移到先行线索
    - 推理时：通过先行线索预测价值，无需等待结果
    """

    def __init__(self, d_model: int = 128, shift_rate: float = 0.1):
        self.d_model = d_model
        self.shift_rate = shift_rate

        # 奖励表征：从结果到先行线索的映射
        self.reward_cues: Dict[str, float] = {}  # 线索 -> 预期奖励

        # 经验计数
        self.experience_count = 0

        # 历史
        self.shift_history: List[Dict] = []

    def update_reward_representation(self, cue: str, actual_reward: float):
        """更新奖励表征（将奖励信号迁移到先行线索）

        Args:
            cue: 先行线索（如"下雨"）
            actual_reward: 实际奖励（如"地面湿了"的价值）
        """
        if cue in self.reward_cues:
            # 渐进更新：旧值 + 学习率 * (新值 - 旧值)
            old_value = self.reward_cues[cue]
            self.reward_cues[cue] = old_value + self.shift_rate * (actual_reward - old_value)
        else:
            # 新线索：直接赋值
            self.reward_cues[cue] = actual_reward * self.shift_rate

        self.experience_count += 1

        # 记录
        self.shift_history.append({
            'cue': cue,
            'actual_reward': actual_reward,
            'predicted_reward': self.reward_cues[cue],
        })

    def predict_reward(self, cue: str) -> float:
        """通过先行线索预测奖励"""
        return self.reward_cues.get(cue, 0.0)

    def get_reward_cues(self) -> Dict[str, float]:
        """获取所有奖励线索"""
        return self.reward_cues.copy()

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'experience_count': self.experience_count,
            'reward_cues': len(self.reward_cues),
            'shift_history': len(self.shift_history),
        }


class CompositionalGeneralization:
    """组合泛化 — 分离what和how + 共享子空间

    基于 Nature 2025 和 eLife 2025 研究：
    - 人类大脑通过共享神经子空间实现组合泛化
    - 已学习的计算组件可以在新任务中被重新组合
    - "what"（内容）和"how"（计算）的分离是关键

    对当前系统的应用：
    - 学习时：分离内容和计算方式
    - 推理时：将已学到的计算方式应用到新内容
    """

    def __init__(self):
        # "what"空间：内容表示
        self.what_space: Dict[str, torch.Tensor] = {}

        # "how"空间：计算方式
        self.how_space: Dict[str, torch.Tensor] = {}

        # 组合历史
        self.combinations: List[Dict] = []

    def learn_what(self, concept: str, representation: torch.Tensor):
        """学习内容表示（what）"""
        self.what_space[concept] = representation.detach().clone()

    def learn_how(self, operation: str, transformation: torch.Tensor):
        """学习计算方式（how）"""
        self.how_space[operation] = transformation.detach().clone()

    def compose(self, concept: str, operation: str) -> Optional[torch.Tensor]:
        """组合内容和计算方式

        Args:
            concept: 内容概念
            operation: 计算方式

        Returns:
            组合后的表示
        """
        if concept not in self.what_space:
            return None
        if operation not in self.how_space:
            return None

        what = self.what_space[concept]
        how = self.how_space[operation]

        # 组合：将计算方式应用到内容
        result = what + how  # 简化：加法组合

        self.combinations.append({
            'concept': concept,
            'operation': operation,
            'result_norm': torch.norm(result).item(),
        })

        return result

    def decompose(self, representation: torch.Tensor) -> Dict[str, str]:
        """分解表示为内容和计算方式

        Args:
            representation: 要分解的表示

        Returns:
            {'what': 最匹配的概念, 'how': 最匹配的计算方式}
        """
        best_what = None
        best_how = None
        best_what_sim = -1.0
        best_how_sim = -1.0

        # 找最匹配的内容
        for concept, what_repr in self.what_space.items():
            sim = torch.cosine_similarity(
                representation.unsqueeze(0), what_repr.unsqueeze(0)
            ).item()
            if sim > best_what_sim:
                best_what_sim = sim
                best_what = concept

        # 找最匹配的计算方式
        for operation, how_repr in self.how_space.items():
            sim = torch.cosine_similarity(
                representation.unsqueeze(0), how_repr.unsqueeze(0)
            ).item()
            if sim > best_how_sim:
                best_how_sim = sim
                best_how = operation

        return {
            'what': best_what,
            'how': best_how,
            'what_similarity': best_what_sim,
            'how_similarity': best_how_sim,
        }

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'what_concepts': len(self.what_space),
            'how_operations': len(self.how_space),
            'combinations': len(self.combinations),
        }


class SocialContingencyLearning:
    """社会偶联学习 — 即时反馈循环

    基于 Royal Society 2026 研究：
    "Pathways from social contingency to infant language learning"

    核心发现：
    - 社会偶联性（对婴儿行为的及时回应）促进学习
    - 婴儿不是被动接收信息，而是通过互动反馈循环主动建构语言

    对当前系统的应用：
    - 学习时：对每次学习尝试给予即时反馈
    - 推理时：根据反馈调整推理策略
    """

    def __init__(self):
        # 反馈历史
        self.feedback_history: List[Dict] = []

        # 反馈强度
        self.feedback_strength: Dict[str, float] = {}

        # 学习速率调整
        self.learning_rate_adjustments: Dict[str, float] = {}

    def give_feedback(self, action: str, outcome: float, context: str = ''):
        """给予即时反馈

        Args:
            action: 行动（如学习的文本）
            outcome: 结果（如验证分数）
            context: 上下文
        """
        feedback = {
            'action': action,
            'outcome': outcome,
            'context': context,
            'timestamp': len(self.feedback_history),
        }
        self.feedback_history.append(feedback)

        # 更新反馈强度
        if action in self.feedback_strength:
            old = self.feedback_strength[action]
            self.feedback_strength[action] = old + 0.1 * (outcome - old)
        else:
            self.feedback_strength[action] = outcome

        # 根据反馈调整学习速率
        if outcome > 0.7:
            # 正反馈：降低学习速率（已掌握）
            self.learning_rate_adjustments[action] = 0.5
        elif outcome < 0.3:
            # 负反馈：提高学习速率（需要更多学习）
            self.learning_rate_adjustments[action] = 2.0
        else:
            # 中等反馈：保持学习速率
            self.learning_rate_adjustments[action] = 1.0

    def get_learning_rate_adjustment(self, action: str) -> float:
        """获取学习速率调整"""
        return self.learning_rate_adjustments.get(action, 1.0)

    def get_feedback_strength(self, action: str) -> float:
        """获取反馈强度"""
        return self.feedback_strength.get(action, 0.5)

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'total_feedback': len(self.feedback_history),
            'feedback_actions': len(self.feedback_strength),
            'avg_outcome': sum(f['outcome'] for f in self.feedback_history) / max(len(self.feedback_history), 1),
        }


class SymbolGrounding:
    """符号接地 — 交互式环境因果学习

    基于 arXiv 2026 USC 研究：
    "Intelligence Requires Grounding But Not Embodiment"

    核心发现：
    - 智能需要接地但不需要具身
    - 接地是符号获得外部一致意义的机制
    - 可以在数字环境中通过工具使用、代码执行等方式实现接地

    对当前系统的应用：
    - 学习时：通过交互建立符号与环境的因果关系
    - 推理时：使用接地的符号进行因果推理
    """

    def __init__(self):
        # 符号到环境的映射
        self.symbol_to_env: Dict[str, Dict] = {}

        # 环境到符号的映射
        self.env_to_symbol: Dict[str, str] = {}

        # 接地强度
        self.grounding_strength: Dict[str, float] = {}

    def ground_symbol(self, symbol: str, env_state: Dict, interaction_result: float):
        """将符号接地到环境状态

        Args:
            symbol: 符号（如"下雨"）
            env_state: 环境状态（如{'moisture': 0.8, 'cloud': 0.9}）
            interaction_result: 交互结果（如因果关系强度）
        """
        # 记录符号到环境的映射
        self.symbol_to_env[symbol] = env_state.copy()

        # 记录环境到符号的映射
        env_key = str(sorted(env_state.items()))
        self.env_to_symbol[env_key] = symbol

        # 更新接地强度
        if symbol in self.grounding_strength:
            old = self.grounding_strength[symbol]
            self.grounding_strength[symbol] = old + 0.1 * (interaction_result - old)
        else:
            self.grounding_strength[symbol] = interaction_result

    def get_grounded_meaning(self, symbol: str) -> Optional[Dict]:
        """获取符号的接地含义"""
        return self.symbol_to_env.get(symbol)

    def get_symbol_for_env(self, env_state: Dict) -> Optional[str]:
        """根据环境状态获取符号"""
        env_key = str(sorted(env_state.items()))
        return self.env_to_symbol.get(env_key)

    def is_grounded(self, symbol: str, threshold: float = 0.3) -> bool:
        """判断符号是否已接地"""
        return self.grounding_strength.get(symbol, 0.0) > threshold

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'grounded_symbols': len(self.symbol_to_env),
            'env_mappings': len(self.env_to_symbol),
            'avg_strength': sum(self.grounding_strength.values()) / max(len(self.grounding_strength), 1),
        }
