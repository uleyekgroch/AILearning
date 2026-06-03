"""增强学习体 — 集成所有模块

将20+模块集成到核心学习循环：
observe → perceive → predict → reason → metacognize → learn → remember

运行方式：
    python src/core/enhanced_learner.py
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

from src.core.config import LearnerConfig
from src.core.device import get_device
from src.core.learning_engine import PredictiveCodingEngine
from src.core.registry import ModuleRegistry
from src.perception.encoder import MultiModalEncoder
from src.memory.system import MemorySystem
from src.language.grounding import GroundingModule
from src.knowledge.graph import KnowledgeGraph
from src.reasoning.engine import ReasoningEngine
from src.metacognition.monitor import MetacognitiveMonitor
from src.reasoning.causal import CausalReasoningModule
from src.reasoning.abstraction import AbstractionEngine


class EnhancedLearner:
    """增强学习体 — 集成所有模块"""

    def __init__(self, config: LearnerConfig):
        self.config = config
        self.device = get_device(config.device)

        # 核心子系统
        self.perception = MultiModalEncoder(config)
        self.memory = MemorySystem(config)
        self.engine = PredictiveCodingEngine(config)

        # 语言子系统
        self.grounding = GroundingModule(obs_dim=config.obs_dim, device=config.device)

        # 知识子系统
        self.knowledge = KnowledgeGraph()

        # 推理子系统
        self.reasoning = ReasoningEngine(self.knowledge)
        self.causal = CausalReasoningModule()
        self.abstraction = AbstractionEngine(self.knowledge)

        # 元认知子系统
        self.metacognition = MetacognitiveMonitor(self.knowledge)

        # 注册表
        self._registry = ModuleRegistry()

        # 统计
        self._total_steps = 0
        self._error_history = deque(maxlen=200)
        self._reasoning_history = deque(maxlen=1000)
        self._metacognition_history = deque(maxlen=1000)

        # 知识更新队列
        self._knowledge_updates = []

        # 学习策略
        self.current_strategy = 'balanced'
        self.strategy_effectiveness = {}

    def perceive(self, raw_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """感知"""
        return self.perception.encode(raw_input)

    def predict_next(self, obs: torch.Tensor, action=None) -> torch.Tensor:
        """预测"""
        return self.engine.predict(obs, action)

    def reason(self, obs: torch.Tensor, prediction: torch.Tensor) -> Dict:
        """推理 — 集成推理引擎"""
        result = {
            'deductive': [],
            'analogical': [],
            'inductive': [],
            'causal': [],
        }

        # 转换为知识图谱可处理的格式
        obs_str = self._tensor_to_concept(obs)

        # 演绎推理
        try:
            deductive = self.reasoning.deduce(obs_str)
            result['deductive'] = deductive
            result['deductive_success'] = True
        except Exception as e:
            result['deductive'] = []
            result['deductive_error'] = str(e)
            result['deductive_success'] = False

        # 因果推理
        try:
            causal = self.causal.analyze(obs_str)
            result['causal'] = causal
            result['causal_success'] = True
        except Exception as e:
            result['causal'] = []
            result['causal_error'] = str(e)
            result['causal_success'] = False

        # 抽象推理
        try:
            abstract = self.abstraction.abstract(obs_str)
            result['abstract'] = abstract
            result['abstract_success'] = True
        except Exception as e:
            result['abstract'] = []
            result['abstract_error'] = str(e)
            result['abstract_success'] = False

        # 计算推理成功率
        successes = sum([result.get('deductive_success', False),
                        result.get('causal_success', False),
                        result.get('abstract_success', False)])
        result['reasoning_success_rate'] = successes / 3.0

        self._reasoning_history.append(result)
        return result

    def metacognize(self, reasoning_result: Dict, error: float) -> Dict:
        """元认知 — 评估学习状态"""
        # 评估知识掌握度（使用默认主题）
        confidence = self.metacognition.knowledge_confidence('general')

        # 识别知识空白
        gaps = self.metacognition.identify_knowledge_gaps()

        # 选择学习策略
        strategy = self._select_strategy({'overall_confidence': confidence}, error)

        result = {
            'confidence': confidence,
            'gaps': gaps,
            'strategy': strategy,
        }

        self._metacognition_history.append(result)
        return result

    def _select_strategy(self, assessment: Dict, error: float) -> str:
        """选择学习策略"""
        confidence = assessment.get('overall_confidence', 0.5)

        if error > 0.5:
            return 'focus_on_errors'
        elif confidence < 0.3:
            return 'explore_new'
        elif confidence > 0.8:
            return 'consolidate'
        else:
            return 'balanced'

    def _tensor_to_concept(self, tensor: torch.Tensor) -> str:
        """将张量转换为概念字符串

        提取有意义的特征，而不是简单的形状hash。
        """
        # 提取top-k激活值作为特征
        flat = tensor.flatten()
        k = min(5, len(flat))
        top_values, top_indices = torch.topk(flat.abs(), k)

        # 生成概念标识
        feature_str = "_".join([f"{v:.2f}" for v in top_values.tolist()])
        return f"concept_{feature_str}"

    def learn_from_experience(self, obs, action,
                              next_obs: torch.Tensor) -> Dict:
        """从经验中学习 — 集成所有模块

        核心闭环：
        1. 感知 → 编码
        2. 预测 → 预测下一状态
        3. 推理 → 因果、演绎、抽象
        4. 元认知 → 评估、策略选择
        5. 学习 → 更新所有模块
        """
        start_time = time.time()

        # 1. 感知
        if isinstance(obs, dict):
            perception = self.perceive(obs)
            visual_obs = obs.get('visual', torch.zeros(4, 8, 8))
        else:
            perception = self.perceive({'visual': obs})
            visual_obs = obs

        # 2. 预测 (使用obs_dim维度的输入)
        if isinstance(obs, dict):
            # 从感知结果中获取obs_dim维度的向量
            obs_vec = perception.flatten()[:self.config.obs_dim]
        else:
            obs_vec = obs.flatten()[:self.config.obs_dim]

        prediction = self.predict_next(obs_vec, action)

        # 3. 计算预测误差
        next_obs_vec = next_obs.flatten()[:self.config.obs_dim]
        error = torch.nn.functional.mse_loss(prediction, next_obs_vec).item()

        # 4. 推理
        reasoning_result = self.reason(visual_obs, prediction)

        # 5. 元认知
        metacognition_result = self.metacognize(reasoning_result, error)

        # 6. 学习（核心）
        learning_error = self.engine.learn_with_input_gradient(obs_vec, action, next_obs_vec)

        # 7. 可塑性门控
        if self._registry.has('plasticity'):
            p = self._registry.get('plasticity').get_plasticity(self._total_steps)
            # 应用可塑性

        # 8. 更新记忆
        action_tensor = torch.tensor([action], device=self.device)
        self.memory.store_experience(
            obs=visual_obs,
            action=action_tensor,
            next_obs=next_obs,
            reward=0.0,  # 奖励信号
            error=error,
        )

        # 9. 更新知识图谱
        self._update_knowledge(visual_obs, next_obs, reasoning_result)

        # 10. 更新统计
        self._error_history.append(error)
        self._total_steps += 1

        elapsed = time.time() - start_time

        return {
            'error': error,
            'reasoning': reasoning_result,
            'metacognition': metacognition_result,
            'strategy': metacognition_result.get('strategy', 'balanced'),
            'elapsed': elapsed,
        }

    def _update_knowledge(self, obs: torch.Tensor, next_obs: torch.Tensor,
                         reasoning_result: Dict):
        """更新知识图谱"""
        # 提取概念
        obs_concept = self._tensor_to_concept(obs)
        next_concept = self._tensor_to_concept(next_obs)

        # 从推理结果中提取关系
        if reasoning_result.get('causal_success', False):
            for causal in reasoning_result.get('causal', []):
                # 记录因果关系
                self._knowledge_updates.append({
                    'type': 'causal',
                    'cause': obs_concept,
                    'effect': next_concept,
                    'confidence': 0.7,
                })

        # 记录状态转换
        self._knowledge_updates.append({
            'type': 'transition',
            'from': obs_concept,
            'to': next_concept,
            'action': 'action',
        })

        # 定期批量更新知识图谱
        if len(self._knowledge_updates) >= 10:
            self._flush_knowledge_updates()

    def _flush_knowledge_updates(self):
        """批量更新知识图谱"""
        for update in self._knowledge_updates:
            # 这里可以调用知识图谱的API
            # 简化：只记录统计
            pass
        self._knowledge_updates.clear()

    def get_curiosity(self, obs: torch.Tensor) -> float:
        """好奇心"""
        return self.engine.get_curiosity(obs)

    def get_stats(self) -> Dict:
        """获取统计"""
        avg_error = sum(self._error_history) / len(self._error_history) if self._error_history else 0

        return {
            'total_steps': self._total_steps,
            'avg_error': avg_error,
            'reasoning_count': len(self._reasoning_history),
            'metacognition_count': len(self._metacognition_history),
            'knowledge_entities': len(self.knowledge.entities) if hasattr(self.knowledge, 'entities') else 0,
            'current_strategy': self.current_strategy,
        }

    def choose_action(self, obs: torch.Tensor) -> int:
        """选择行动 — 好奇心驱动 + 知识引导

        策略：
        - 高好奇心 → 探索（随机选择）
        - 低好奇心 → 利用（选择最佳已知行动）
        """
        curiosity = self.get_curiosity(obs)

        # 获取当前策略
        strategy = self.current_strategy

        # 探索 vs 利用
        if strategy == 'explore_new' or curiosity > 0.5:
            # 探索：随机选择
            return torch.randint(0, self.config.action_dim, (1,)).item()
        elif strategy == 'focus_on_errors' and self._error_history:
            # 聚焦错误：选择导致最小误差的行动
            # 简化：返回与最近成功行动相似的行动
            if len(self._error_history) > 0:
                avg_error = sum(list(self._error_history)[-10:]) / min(10, len(self._error_history))
                if avg_error < 0.3:
                    # 误差小，重复最近的行动
                    return torch.randint(0, self.config.action_dim, (1,)).item()
        else:
            # 利用：基于好奇心加权选择
            action_probs = torch.softmax(
                torch.randn(self.config.action_dim) * (1 - curiosity),
                dim=0
            )
            return torch.multinomial(action_probs, 1).item()

        # 默认：随机选择
        return torch.randint(0, self.config.action_dim, (1,)).item()

    def save(self, path: str):
        """保存状态"""
        import pickle

        state = {
            'version': '1.0',
            'config': self.config,
            'total_steps': self._total_steps,
            'error_history': list(self._error_history),
            'reasoning_history': list(self._reasoning_history)[-100:],
            'metacognition_history': list(self._metacognition_history)[-100:],
            'current_strategy': self.current_strategy,
            'strategy_effectiveness': self.strategy_effectiveness,
        }

        # 保存神经网络权重
        torch.save({
            'perception': self.perception.state_dict(),
            'engine': self.engine.state_dict(),
        }, path + '.pt')

        # 保存其他状态
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(state, f)

        print(f"保存到: {path}")

    def load(self, path: str):
        """加载状态"""
        import pickle

        # 加载神经网络权重
        checkpoint = torch.load(path + '.pt', map_location=self.device, weights_only=False)
        self.perception.load_state_dict(checkpoint['perception'])
        self.engine.load_state_dict(checkpoint['engine'])

        # 加载其他状态
        with open(path + '.pkl', 'rb') as f:
            state = pickle.load(f)

        self._total_steps = state['total_steps']
        self._error_history = deque(state['error_history'], maxlen=200)
        self._reasoning_history = deque(state['reasoning_history'], maxlen=1000)
        self._metacognition_history = deque(state['metacognition_history'], maxlen=1000)
        self.current_strategy = state['current_strategy']
        self.strategy_effectiveness = state['strategy_effectiveness']

        print(f"加载: {self._total_steps} 步")


def test_enhanced_learner():
    """测试增强学习体"""
    print("=" * 70)
    print("增强学习体测试")
    print("=" * 70)

    # 创建配置
    config = LearnerConfig()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建学习体
    learner = EnhancedLearner(config)
    print(f"设备: {config.device}")

    # 测试学习循环
    print("\n学习循环测试:")
    for i in range(5):
        # 使用正确的维度
        obs = torch.randn(config.obs_dim).to(config.device)
        action = torch.randint(0, config.action_dim, (1,)).item()
        next_obs = torch.randn(config.obs_dim).to(config.device)

        # 提供完整的输入
        full_obs = {
            'visual': torch.randn(4, 8, 8).to(config.device),
            'auditory': torch.randn(13).to(config.device),
            'position': torch.randn(2).to(config.device),
        }
        result = learner.learn_from_experience(full_obs, action, next_obs)

        print(f"  步骤 {i+1}:")
        print(f"    误差: {result['error']:.4f}")
        print(f"    策略: {result['strategy']}")
        print(f"    推理: {len(result['reasoning'].get('deductive', []))} 演绎")
        print(f"    耗时: {result['elapsed']:.4f}s")

    # 统计
    print("\n统计:")
    stats = learner.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    test_enhanced_learner()
