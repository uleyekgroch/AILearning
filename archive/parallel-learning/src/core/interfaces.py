"""
统一学习系统 — 核心抽象接口

DDD 限界上下文的"公约数"。所有领域通过这些接口解耦。
所有输入/输出严格使用 torch.Tensor，零 numpy 暴露。

7 个接口对应 7 个领域：
  IPerception    — 感知领域
  IMemory        — 记忆领域
  ILanguage      — 语言领域
  ILearningEngine — 学习引擎
  IEnvironment   — 环境领域
  ICurriculum    — 课程领域
  ISocialAgent   — 社会领域
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, List, Optional, Any, Tuple


class IPerception(ABC):
    """感知接口：将原始多模态输入编码为内部表示"""

    @abstractmethod
    def encode(self, raw_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将原始输入（视觉/听觉/位置）编码为统一内部表示向量"""

    @abstractmethod
    def get_modality_weights(self) -> Dict[str, float]:
        """获取各模态的注意力权重（语言引导注意力调制用）"""


class IMemory(ABC):
    """记忆接口：三层记忆系统的统一协议"""

    @abstractmethod
    def store(self, representation: torch.Tensor, metadata: Dict) -> None:
        """存储经验（表示向量 + 元数据）"""

    @abstractmethod
    def retrieve(self, cue: torch.Tensor, k: int = 5) -> List[Dict]:
        """按线索检索最相似的 k 条经验"""

    @abstractmethod
    def consolidate(self) -> Dict:
        """巩固记忆（模拟睡眠），返回巩固报告"""


class ILanguage(ABC):
    """语言接口：语言产出与理解"""

    @abstractmethod
    def produce(self, intention: Dict) -> List[str]:
        """从内部意图产生话语（符号序列）"""

    @abstractmethod
    def comprehend(self, utterance: List[str], context: Dict) -> Dict:
        """理解话语，返回解析后的意图/指称"""

    @abstractmethod
    def get_vocabulary(self) -> Dict[str, Any]:
        """获取当前词汇表状态"""


class ILearningEngine(ABC):
    """学习引擎接口：预测编码核心"""

    @abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """预测下一个状态"""

    @abstractmethod
    def learn(self, predicted: torch.Tensor, actual: torch.Tensor) -> float:
        """从预测误差中学习，返回预测误差（MSE）"""

    @abstractmethod
    def get_curiosity(self, state: torch.Tensor) -> float:
        """计算好奇心值（内在奖励 = 预测误差 × 可学习性）"""


class IEnvironment(ABC):
    """环境接口：统一的环境交互协议"""

    @abstractmethod
    def observe(self) -> Dict[str, torch.Tensor]:
        """获取当前环境观测"""

    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool]:
        """执行动作，返回 (观测, 奖励, 是否结束)"""

    @abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境"""

    def configure_for_stage(self, stage: str) -> None:
        """按发展阶段配置环境（可选覆盖）"""


class ICurriculum(ABC):
    """课程接口：发展阶段调度"""

    @abstractmethod
    def get_current_stage(self) -> str:
        """获取当前发展阶段名称"""

    @abstractmethod
    def evaluate(self, learner: Any) -> Dict[str, float]:
        """评估学习者各项能力指标"""

    @abstractmethod
    def should_advance(self, evaluation: Dict[str, float]) -> bool:
        """判断是否满足晋升下一阶段的条件"""


class ISocialAgent(ABC):
    """社会 Agent 接口：多 Agent 交互协议"""

    @abstractmethod
    def interact(self, partner: 'ISocialAgent', env: IEnvironment) -> Dict:
        """与另一个 Agent 在环境中交互"""

    @abstractmethod
    def observe_partner(self, partner_action: torch.Tensor,
                        partner_outcome: Dict) -> None:
        """观察伙伴的行为和结果（社会学习）"""

    @abstractmethod
    def teach(self, learner: 'ISocialAgent', topic: str) -> Dict:
        """向另一个 Agent 教授特定主题"""
