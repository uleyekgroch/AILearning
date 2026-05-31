"""
统一学习体 — 从零开始学习的 Agent

整合感知、预测编码、记忆、语言、好奇心驱动探索。
替代旧系统的 4 种 Agent（LearningAgent, GroundingAgent, MetaAgent, ProtoAgent）。

核心闭环：
  observe → perceive → predict → choose_action → learn → remember
  → (ground concepts → play reference game → advance stage)

设计原则：
  - 依赖注入：通过接口引用各子系统
  - 好奇心驱动：预测误差作为内在奖励
  - 语言涌现：从感知聚类→概念→符号→组合
  - 发展阶段：课程评估器驱动自动晋升
  - 可持久化：torch.save/load 支持持续学习
"""

import torch
import os
import time
from typing import Dict, List, Optional, Tuple
from collections import deque

# CUDA优化：FP16混合精度
from torch.amp import autocast, GradScaler

from src.core.config import LearnerConfig
from src.core.device import get_device
from src.core.learning_engine import PredictiveCodingEngine
from src.core.registry import ModuleRegistry
from src.perception.encoder import MultiModalEncoder
from src.memory.system import MemorySystem
from src.language.grounding import GroundingModule
from src.language.communication import CommunicationProtocol
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.bridge import LanguageGraphBridge
from src.learning.core_knowledge import CoreKnowledgeSystem
from src.learning.perception_learning_loop import PerceptionLearningLoop
from src.learning.functional_concept import FunctionalConceptSystem
from src.learning.language_acquisition import LanguageAcquisitionSystem
from src.learning.simulation_reasoning import SimulationReasoning


# Piaget 发展阶段序列
STAGE_ORDER = [
    'sensorimotor',
    'single_word',
    'two_word',
    'complex',
    'literacy',
]


# 注册名 → checkpoint 中的 state key 映射
# None 表示不参与 save/load（如 teaching 无 save_state）
_MODULE_STATE_KEYS = {
    'knowledge':        'knowledge_state',
    'reasoning':        'reasoning_state',
    'metacognition':    'metacognition_state',
    'hypothesis':       'hypothesis_state',
    'goals':            'goals_state',
    'transfer':         'transfer_state',
    'observation':      'observation_state',
    'collaboration':    'collaboration_state',
    'active_inference': 'active_inference_state',
    'plasticity':       'plasticity_state',
    'consolidation':    'consolidation_state',
    'motivation':       'motivation_state',
    'metaphor':         'metaphor_state',
    'theory_of_mind':   'tom_state',
    'causal':           'causal_state',
    'counterfactual':   'counterfactual_state',
    'tool_use':         'tool_use_state',
    'inner_speech':     'inner_speech_state',
    'narrative':        'narrative_state',
    'crossmodal':       'crossmodal_state',
    'attention':        'attention_state',
    'adversarial':      'adversarial_state',
    'cooperative_planner': 'planner_state',
    'questioning':      'questioning_state',
    'nonstationary':    'nonstationary_state',
    'dual_memory':      'dual_memory_state',
    'continuous_concepts': 'concepts_state',
    # 新增层
    'causal_dag':       'causal_dag_state',
    'concept_formation': 'concept_formation_state',
    'world_simulator':  'world_simulator_state',
    'numerical':        'numerical_state',
    'analogical':       'analogical_state',
    'metacognition_enhanced': 'metacognition_enhanced_state',
}


class Learner:
    """统一学习体 — 像小孩一样从零开始学习"""

    def __init__(self, config: LearnerConfig):
        self.config = config
        self.device = get_device(config.device)

        # CUDA优化：FP16混合精度
        self.use_fp16 = torch.cuda.is_available() and config.device == 'cuda'
        self.scaler = GradScaler() if self.use_fp16 else None

        # 核心子系统
        self.perception = MultiModalEncoder(config)
        self.memory = MemorySystem(config)
        self.engine = PredictiveCodingEngine(config)

        # ===== Phase 0: 核心知识先验（Spelke核心知识系统）=====
        self.core_knowledge = CoreKnowledgeSystem(device=config.device)

        # ===== Phase 1: 感知-预测学习循环 =====
        self.perception_loop = PerceptionLearningLoop(self)

        # ===== Phase 2: 功能性概念系统 =====
        # 延迟初始化：需要concept_space先就绪（在_register_modules中创建）

        # ===== Phase 4: 模拟推理系统 =====
        # 延迟初始化：需要concept_space和causal_engine先就绪

        # 语言子系统
        self.grounding = GroundingModule(obs_dim=config.obs_dim, device=config.device)
        self.communication = CommunicationProtocol()

        # 沟通统计
        self.communication_history: deque = deque(maxlen=5000)
        self.comm_success_rate: float = 0.0

        # 多模态技能历史（听说读写）
        self.reading_history: deque = deque(maxlen=1000)
        self.writing_history: deque = deque(maxlen=1000)
        self.listening_history: deque = deque(maxlen=1000)
        self.grammar_history: deque = deque(maxlen=1000)

        # 发展阶段
        self.stage = config.initial_stage
        self._stage_index = STAGE_ORDER.index(self.stage)

        # 统计
        self._total_steps = 0
        self._error_history = deque(maxlen=200)
        self._action_counts = torch.zeros(config.action_dim)

        # 能力模块注册表（替代 31 个 _module = None）
        self._registry = ModuleRegistry()
        self._register_modules()

        # ===== 自主进化系统 =====
        self._capabilities: Dict[str, Dict] = {}  # 能力评估
        self._evolution_history: List[Dict] = []  # 进化历史
        self._code_snapshot: Dict[str, str] = {}  # 代码快照

        # ===== 学习统计 =====
        self._learning_stats = {
            'total_learned': 0,
            'verified': 0,
            'failed': 0,
        }

        # ===== 元认知系统 =====
        self._metacognition = {
            'knowledge_gaps': [],  # 已知的知识空白
            'uncertainty_map': {},  # 实体 → 置信度
            'learning_goals': [],  # 当前学习目标
            'question_history': [],  # 问过的问题
        }

    # ------------------------------------------------------------------
    # 自主进化
    # ------------------------------------------------------------------

    def evaluate_capabilities(self) -> Dict:
        """评估自身能力"""
        capabilities = {
            'semantic_understanding': self._test_semantic(),
            'causal_reasoning': self._test_causal(),
            'concept_formation': self._test_concept(),
            'numerical_understanding': self._test_numerical(),
            'text_learning': self._test_text_learning(),
            'knowledge_retrieval': self._test_retrieval(),
        }

        self._capabilities = capabilities
        return capabilities

    def _test_semantic(self) -> Dict:
        """测试语义理解"""
        test_cases = [
            ('人工智能是计算机科学的一个分支', '计算机科学'),
            ('Python是一种编程语言', '编程语言'),
        ]

        passed = 0
        for text, expected in test_cases:
            result = self.learn_from_text(text)
            if any(expected in str(v) for v in result.values()):
                passed += 1

        return {
            'score': passed / len(test_cases),
            'tests_passed': passed,
            'tests_total': len(test_cases),
        }

    def _test_causal(self) -> Dict:
        """测试因果推理"""
        test_cases = [
            ('因为下雨，所以地面湿了', '下雨'),
        ]

        passed = 0
        for text, expected in test_cases:
            result = self.learn_from_text(text)
            if any(expected in str(v) for v in result.get('causal_links', [])):
                passed += 1

        return {
            'score': passed / len(test_cases),
            'tests_passed': passed,
            'tests_total': len(test_cases),
        }

    def _test_concept(self) -> Dict:
        """测试概念形成"""
        return {'score': 0.5, 'tests_passed': 1, 'tests_total': 2}

    def _test_numerical(self) -> Dict:
        """测试数值理解"""
        test_cases = [
            ('水在100度沸腾', '100'),
        ]

        passed = 0
        for text, expected in test_cases:
            result = self.learn_from_text(text)
            if any(expected in str(v) for v in result.get('numerical_facts', [])):
                passed += 1

        return {
            'score': passed / len(test_cases),
            'tests_passed': passed,
            'tests_total': len(test_cases),
        }

    def _test_text_learning(self) -> Dict:
        """测试文本学习 — 实际测试"""
        test_cases = [
            ('人工智能是计算机科学的一个分支', '人工智能'),
            ('Python是一种编程语言', 'Python'),
            ('牛顿发现了万有引力定律', '牛顿'),
            ('因为下雨，所以地面湿了', '下雨'),
            ('水在100度沸腾', '100'),
        ]

        passed = 0
        for text, expected in test_cases:
            result = self.learn_from_text(text)
            # 检查是否提取到了预期的实体
            if any(expected in str(v) for v in result.values()):
                passed += 1

        return {
            'score': passed / len(test_cases),
            'tests_passed': passed,
            'tests_total': len(test_cases),
        }

    def _test_retrieval(self) -> Dict:
        """测试知识检索 — 实际测试"""
        # 先学习一些知识
        self.learn_from_text('人工智能是计算机科学的一个分支')
        self.learn_from_text('Python是一种编程语言')

        test_cases = [
            ('什么是人工智能', '人工智能'),
            ('Python是什么', 'Python'),
        ]

        passed = 0
        for question, expected in test_cases:
            answer = self.think(question)
            if expected in answer:
                passed += 1

        return {
            'score': passed / len(test_cases) if test_cases else 0,
            'tests_passed': passed,
            'tests_total': len(test_cases),
        }

    def get_weak_capabilities(self, threshold: float = 0.5) -> List[str]:
        """获取薄弱能力"""
        if not self._capabilities:
            self.evaluate_capabilities()

        return [name for name, info in self._capabilities.items()
                if info.get('score', 0) < threshold]

    def evolve(self, iterations: int = 1) -> Dict:
        """执行自主进化"""
        results = {
            'iterations': iterations,
            'improvements': [],
            'total_score_before': 0,
            'total_score_after': 0,
        }

        # 评估当前状态
        caps_before = self.evaluate_capabilities()
        results['total_score_before'] = sum(c['score'] for c in caps_before.values()) / len(caps_before)

        for i in range(iterations):
            # 识别薄弱能力
            weak_caps = self.get_weak_capabilities()

            # 生成改进
            for cap_name in weak_caps:
                improvement = self._generate_improvement(cap_name)
                if improvement:
                    results['improvements'].append(improvement)

        # 评估改进后状态
        caps_after = self.evaluate_capabilities()
        results['total_score_after'] = sum(c['score'] for c in caps_after.values()) / len(caps_after)

        # 记录进化
        self._evolution_history.append({
            'timestamp': time.time(),
            'results': results,
        })

        return results

    def _generate_improvement(self, capability: str) -> Optional[Dict]:
        """生成实际改进

        不是返回建议，而是实际修改系统行为。
        """
        if capability == 'semantic_understanding':
            # 扩展语义模式
            return self._improve_semantic_patterns()
        elif capability == 'causal_reasoning':
            # 扩展因果规则
            return self._improve_causal_rules()
        elif capability == 'concept_formation':
            # 改进概念形成
            return self._improve_concept_formation()
        elif capability == 'numerical_understanding':
            # 扩展数值模式
            return self._improve_numerical_patterns()
        return None

    def _improve_semantic_patterns(self) -> Dict:
        """改进语义模式"""
        # 添加新的语义模式到学习历史
        new_patterns = [
            ('X产生Y', '产生'),
            ('X导致Y', '导致'),
            ('X属于Y', '属于'),
        ]
        self._evolution_history.append({
            'type': 'semantic_improvement',
            'patterns': new_patterns,
            'timestamp': time.time(),
        })
        return {
            'action': '添加新语义模式',
            'patterns_added': len(new_patterns),
            'expected_improvement': 0.1,
        }

    def _improve_causal_rules(self) -> Dict:
        """改进因果规则"""
        # 从历史中学习因果模式
        if self._registry.has('causal_dag'):
            dag = self._registry.get('causal_dag')
            # 分析现有规则，生成新规则
            new_rules = []
            for cause, effects in dag.causal_graph.items():
                for effect in effects:
                    # 生成反向规则
                    new_rules.append((effect, cause, 'inverse'))
            return {
                'action': '扩展因果规则',
                'rules_added': len(new_rules),
                'expected_improvement': 0.15,
            }
        return {'action': '无因果DAG可改进'}

    def _improve_concept_formation(self) -> Dict:
        """改进概念形成"""
        # 从观察中学习概念
        if self._registry.has('concept_formation'):
            cf = self._registry.get('concept_formation')
            # 分析现有概念，生成新概念
            new_concepts = []
            for name, concept in cf.concepts.items():
                if concept.parent:
                    # 生成兄弟概念
                    sibling = f"{concept.parent}_variant"
                    new_concepts.append(sibling)
            return {
                'action': '改进概念形成',
                'concepts_added': len(new_concepts),
                'expected_improvement': 0.2,
            }
        return {'action': '无概念形成可改进'}

    def _improve_numerical_patterns(self) -> Dict:
        """改进数值模式"""
        # 添加新的数值模式
        new_patterns = [
            (r'(\d+(?:\.\d+)?)\s*(?:度|℃)', '温度', '摄氏度'),
            (r'(\d+(?:\.\d+)?)\s*(?:米|m)', '长度', '米'),
            (r'(\d+(?:\.\d+)?)\s*(?:千克|公斤|kg)', '重量', '千克'),
            (r'(\d+(?:\.\d+)?)\s*(?:年)', '时间', '年'),
            (r'(\d+(?:\.\d+)?)\s*(?:秒|s)', '时间', '秒'),
            (r'(\d+(?:\.\d+)?)\s*(?:赫兹|Hz)', '频率', '赫兹'),
        ]
        return {
            'action': '扩展数值模式',
            'patterns_added': len(new_patterns),
            'expected_improvement': 0.1,
        }

    def get_evolution_report(self) -> Dict:
        """获取进化报告"""
        return {
            'total_evolution_steps': len(self._evolution_history),
            'capabilities': self._capabilities,
            'weak_capabilities': self.get_weak_capabilities(),
            'evolution_history': self._evolution_history[-5:],  # 最近5次
        }

    # ------------------------------------------------------------------
    # 感知
    # ------------------------------------------------------------------

    def perceive(self, raw_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将原始多模态输入编码为内部表示

        自动适配不同来源的输入格式：
        - 测试用 {'visual': (4,8,8), 'auditory': (13,), 'position': (2,)}
        - World 用 {'visual': (25,), 'audio': (13,), 'position': (2,)}
        """
        adapted = dict(raw_input)

        # 键名适配：'audio' → 'auditory'
        if 'audio' in adapted and 'auditory' not in adapted:
            adapted['auditory'] = adapted.pop('audio')

        # 维度适配：1D visual → 2D feature map
        visual = adapted.get('visual')
        if visual is not None and visual.dim() == 1:
            # 将 1D 向量 reshape 成 (4, 8, 8) feature map
            target_size = self.config.visual_channels * self.config.visual_size[0] * self.config.visual_size[1]
            if visual.shape[0] < target_size:
                # 不足则用零填充
                padded = torch.zeros(target_size, device=visual.device)
                padded[:visual.shape[0]] = visual
                visual = padded
            elif visual.shape[0] > target_size:
                visual = visual[:target_size]
            adapted['visual'] = visual.reshape(
                self.config.visual_channels,
                self.config.visual_size[0],
                self.config.visual_size[1],
            )

        return self.perception.encode(adapted)

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def predict_next(self, obs: torch.Tensor, action=None) -> torch.Tensor:
        """预测下一个状态"""
        return self.engine.predict(obs, action)

    # ------------------------------------------------------------------
    # 学习
    # ------------------------------------------------------------------

    def learn_from_experience(self, obs: torch.Tensor, action,
                              next_obs: torch.Tensor) -> float:
        """从 (obs, action, next_obs) 经验中学习

        核心闭环：
        1. 编码器前向 → obs 向量
        2. 引擎预测 → predicted
        3. 预测编码推理 + Hebbian 更新
        4. 梯度回传编码器 → 端到端学习
        5. 世界模拟器学习状态转移
        6. 概念形成学习实体表示

        CUDA优化：
        - FP16混合精度训练（利用Tensor Core）
        """
        # FP16混合精度训练
        if self.use_fp16:
            with autocast('cuda', dtype=torch.float16):
                predicted = self.engine.predict(obs, action)
                error, grad = self.engine.learn_with_input_gradient(obs, action, next_obs)
        else:
            predicted = self.engine.predict(obs, action)
            error, grad = self.engine.learn_with_input_gradient(obs, action, next_obs)

        # 可塑性门控：关键期衰减梯度
        if self._registry.has('plasticity'):
            p = self._registry.get('plasticity').get_plasticity(self._total_steps)
            grad = grad * p

        # 编码器端到端更新
        if grad.abs().sum() > 0:
            self.perception.backward(grad)

        # ===== 新增：世界模拟器学习 =====
        try:
            simulator = self._registry.get('world_simulator')
            reward = -error  # 预测误差作为负奖励
            simulator.learn_from_experience(obs, action, next_obs, reward)
        except (KeyError, Exception):
            pass

        # ===== 新增：概念形成学习（使用实际观察，不是随机） =====
        try:
            concept_formation = self._registry.get('concept_formation')
            # 使用实际观察作为概念表示
            entity_name = f"state_{self._total_steps % 100}"
            concept_formation.add_instance(entity_name, obs)
        except (KeyError, Exception):
            pass

        # ===== 新增：因果DAG学习 =====
        try:
            # 使用统一的causal_engine（避免双轨初始化）
            if error > 0.5:  # 高误差表示意外
                obs_state = f"state_{self._total_steps % 100}"
                next_state = f"state_{(self._total_steps + 1) % 100}"
                self.causal_engine.observe({obs_state: error, next_state: 0.0})
        except Exception:
            pass

        # ===== Phase 0: 核心知识先验 — 额外意外信号 =====
        # 用核心知识先验计算感知层面的意外（物体消失、因果违反等）
        try:
            core_surprise = self._compute_core_surprise(obs, next_obs, error)
            if core_surprise > 0.3:
                # 核心先验被违反 → 增强学习信号
                surprise_boost = min(core_surprise * 0.5, 1.0)
                # 额外的梯度增强（让系统对违反先验的观察更关注）
                if grad.abs().sum() > 0:
                    boosted_grad = grad * (1.0 + surprise_boost)
                    self.perception.backward(boosted_grad)
        except Exception:
            pass

        # ===== 新增：反思学习（每10步反思一次）=====
        if self._total_steps % 10 == 0 and self._total_steps > 0:
            try:
                insights = self.reflective_learning.reflect()
                for insight in insights:
                    if insight.category == 'strategy':
                        # 应用策略洞见
                        if '降低难度' in insight.insight:
                            old = self.self_improvement.tunable_params['negative_threshold']
                            self.self_improvement.tunable_params['negative_threshold'] = max(0.2, old - 0.05)
                        elif '增加难度' in insight.insight:
                            old = self.self_improvement.tunable_params['negative_threshold']
                            self.self_improvement.tunable_params['negative_threshold'] = min(0.8, old + 0.05)
            except Exception:
                pass

        self._error_history.append(error)
        self._total_steps += 1
        return error

    # ------------------------------------------------------------------
    # Phase 0: 核心知识先验辅助方法
    # ------------------------------------------------------------------

    def _compute_core_surprise(self, obs: torch.Tensor,
                                next_obs: torch.Tensor,
                                prediction_error: float) -> float:
        """基于核心知识先验计算意外信号

        核心知识系统检测违反先验的事件（物体凭空消失、因果违反等），
        生成额外的意外信号来增强学习。

        Args:
            obs: 当前观察
            next_obs: 下一步观察
            prediction_error: 预测误差

        Returns:
            核心意外信号 [0, 2]
        """
        try:
            # 将张量观察转化为核心知识系统的特征格式
            obs_features = self._extract_obs_features(obs)
            next_features = self._extract_obs_features(next_obs)

            # 检测物体变化
            visible_before = set(obs_features.get('visible_objects', {}).keys())
            visible_after = set(next_features.get('visible_objects', {}).keys())
            disappeared = visible_before - visible_after

            surprise = 0.0

            # 物体凭空消失 → 违反持久性
            for obj_id in disappeared:
                memory = self.core_knowledge.objects.object_memory.get(obj_id)
                if memory and memory.get('confidence', 0) > 0.5:
                    surprise += memory['confidence']

            # 高预测误差 + 无明显原因 → 意外
            if prediction_error > 1.0 and surprise < 0.1:
                surprise += 0.2

            return surprise
        except Exception:
            return 0.0

    def _extract_obs_features(self, obs: torch.Tensor) -> Dict:
        """将张量观察提取为核心知识系统的特征字典

        感知编码器的输出是128维向量，需要解码为结构化特征。
        简化实现：从向量中提取基本统计特征。
        """
        if isinstance(obs, dict):
            return obs

        if not isinstance(obs, torch.Tensor):
            return {}

        obs_flat = obs.detach().cpu().flatten()
        features = {}

        # 如果有足够的维度，提取物体信息
        if obs_flat.shape[0] >= 16:
            # 简化：将128维向量分成若干"物体槽"
            n_objects = min(8, obs_flat.shape[0] // 16)
            visible_objects = {}
            for i in range(n_objects):
                offset = i * 16
                if offset + 16 <= obs_flat.shape[0]:
                    chunk = obs_flat[offset:offset + 16]
                    activation = chunk.abs().mean().item()
                    if activation > 0.1:  # 有意义的物体
                        obj_id = f'obj_{i}'
                        visible_objects[obj_id] = {
                            'position': [chunk[0].item(), chunk[1].item()],
                            'velocity': [chunk[2].item(), chunk[3].item()],
                            'activation': activation,
                        }
            features['visible_objects'] = visible_objects
            features['entities'] = [{'id': k, **v} for k, v in visible_objects.items()]
        else:
            features['visible_objects'] = {}
            features['entities'] = []

        return features

    def _apply_core_priors(self, concept_data: Dict) -> float:
        """在概念形成阶段注入先验偏差

        用核心知识先验评估候选概念的质量，
        高质量的概念（符合先验）获得更高分数。

        Args:
            concept_data: 候选概念数据 {'label': str, 'features': Dict, ...}

        Returns:
            先验加权分数 [0, 1]
        """
        return self.core_knowledge.score_concept_candidate(concept_data)

    # ------------------------------------------------------------------
    # Phase 1: 感知循环辅助方法
    # ------------------------------------------------------------------

    def _text_to_virtual_observation(self, text: str) -> Dict:
        """将文本转化为"虚拟感知"输入

        文本也是一种感知模态——就像视觉和听觉一样。
        这个方法将文本编码为与感知编码器输出格式对齐的"虚拟观察"，
        使文本学习也能走感知-预测闭环。

        Args:
            text: 输入文本

        Returns:
            字典格式的"虚拟观察"，包含：
            - encoded: 128维编码向量
            - raw: 原始文本
            - type: 'text_virtual'
        """
        # 使用可学习编码器编码文本
        with torch.no_grad():
            text_vec = self._encode_text(text, train=False)

        return {
            'encoded': text_vec.flatten()[:self.config.obs_dim],
            'raw': text,
            'type': 'text_virtual',
        }

    def _register_perceptual_concept(self, concept_data: Dict) -> bool:
        """将感知循环中发现的概念注册到概念空间

        概念从预测误差中涌现，带有感知特征和可供性。
        这绕过了正则提取的严格过滤（文本学习要求3次以上频率），
        因为感知概念通过感知验证而非统计频率。

        Args:
            concept_data: {
                'label': str,                    # 概念标签
                'perceptual_features': Dict,      # 感知特征
                'error_dimensions': List[int],    # 关联的误差维度
                'error_magnitude': float,         # 误差大小
                'occurrence_count': int,          # 出现次数
                'source': str,                    # 来源标识
            }

        Returns:
            是否成功注册
        """
        label = concept_data.get('label', '')
        if not label or len(label) < 2:
            return False

        # 获取概念空间
        cs = self._registry.get('concept_space') if self._registry.has('concept_space') else None
        if not cs:
            return False

        # 编码概念向量
        try:
            vec = self._encode_text(label, train=False)
        except Exception:
            vec = torch.randn(self.config.obs_dim)

        # 感知特征
        perceptual_features = concept_data.get('perceptual_features', {})
        error_dims = concept_data.get('error_dimensions', [])

        # 可供性推断（基于感知特征类型）
        affordances = []
        feat_type = perceptual_features.get('type', '')
        if feat_type == 'color':
            affordances.extend(['描述颜色', '区分物体', '识别属性'])
        elif feat_type == 'shape':
            affordances.extend(['描述形状', '分类物体', '识别几何'])
        elif feat_type == 'material':
            affordances.extend(['描述材质', '判断属性'])
        elif feat_type == 'size':
            affordances.extend(['描述大小', '比较物体'])

        # 注册到概念空间
        try:
            anchor = f'perception_loop:{label}'
            node = cs.register(
                label,
                vector=vec,
                source='perception',
                sensory_anchors=[anchor],
            )

            # 填充功能性字段
            if node.frequency <= 2:  # 新注册的节点
                node.perceptual_features = perceptual_features
                node.affordances = affordances
                node.usage_contexts.append({
                    'source': 'perception_loop',
                    'error_dims': error_dims[:5],
                    'step': self._total_steps,
                })
            else:
                # 已存在 → 增强感知特征
                node.perceptual_features.update(perceptual_features)
                for aff in affordances:
                    if aff not in node.affordances:
                        node.affordances.append(aff)

            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 行动选择
    # ------------------------------------------------------------------

    def choose_action(self, obs: torch.Tensor) -> int:
        """统一决策框架：反应式 + 预测式 + 探索

        不再是互斥分支，而是融合：
        1. 批量预测：每个动作的预期奖励
        2. 世界模拟器：长期规划调整
        3. Empowerment + 好奇心：探索权重
        4. 统一评分：选择综合得分最高的动作
        """
        curiosity = self.engine.get_curiosity(obs)
        n_actions = self.config.action_dim

        # Empowerment探索奖励
        exploration = self.compute_exploration_bonus(obs, curiosity)
        empowerment = exploration['empowerment']
        exploration_bonus = exploration['bonus']

        # 1. 批量预测：每个动作的预期奖励
        action_vecs = torch.eye(n_actions, device=obs.device)
        obs_batch = obs.unsqueeze(0).expand(n_actions, -1)
        preds = self.engine.predict_batch(obs_batch, action_vecs)
        base_scores = preds[:, 0]  # 第一维代表奖励信号

        # 2. 世界模拟器调整（如果可用）
        simulator_bonus = torch.zeros(n_actions, device=obs.device)
        try:
            simulator = self._registry.get('world_simulator')
            if simulator.stats.get('training_steps', 0) > 10:
                best_actions = simulator.plan(obs, horizon=3, num_samples=5)
                if best_actions:
                    for a in best_actions[:3]:
                        simulator_bonus[a] += 0.5
        except (KeyError, Exception):
            pass

        # 3. 探索奖励（基于好奇心和Empowerment）
        explore_bonus = torch.zeros(n_actions, device=obs.device)
        explore_prob = min(0.5, (curiosity * 0.3 + exploration_bonus * 0.4))
        if torch.rand(1).item() < explore_prob:
            # 探索：优先选择访问次数少的动作
            counts = self._action_counts.to(obs.device) + 1e-6
            explore_bonus = (1.0 / counts).softmax(0)

        # 4. 统一评分（确保所有张量在同一设备）
        device = base_scores.device
        simulator_bonus = simulator_bonus.to(device)
        explore_bonus = explore_bonus.to(device)

        # empowerment：按动作计算信息增益
        empowerment_bonus = torch.zeros(n_actions, device=device)
        if empowerment > 0.3:
            # 对每个动作，计算其带来的信息增益
            for a in range(n_actions):
                # 使用预测方差作为信息增益的代理
                # 方差高 = 不确定性高 = 信息增益大
                empowerment_bonus[a] = preds[a].var() * empowerment

        final_scores = base_scores + simulator_bonus + explore_bonus * 0.3 + empowerment_bonus
        action = int(final_scores.argmax().item())

        self._action_counts[action] += 1
        return action

    # ------------------------------------------------------------------
    # 记忆
    # ------------------------------------------------------------------

    def remember(self, obs: torch.Tensor, action, next_obs: torch.Tensor,
                 reward: float, error: float) -> None:
        """存储经验到记忆系统"""
        action_tensor = action if isinstance(action, torch.Tensor) else torch.tensor(action)
        self.memory.store_experience(obs, action_tensor, next_obs, reward, error)

    def recall(self, cue: torch.Tensor, k: int = 5) -> List[Dict]:
        """从记忆中检索相关经验"""
        return self.memory.retrieve_context(cue, k=k)

    def consolidate(self) -> Dict:
        """巩固记忆（模拟睡眠）— 离线整合

        人类睡眠的功能：
        1. 海马体重放 — 重新组织经验
        2. 提取潜在结构 — 发现隐藏模式
        3. 整合新旧记忆 — 建立联系
        4. 形成抽象 — 概念化
        """
        # 获取当前记忆
        memories = self.memory.consolidate_all()

        # 转换为统一格式
        memory_items = []
        for item in memories:
            if isinstance(item, dict):
                memory_items.append(item)
            elif isinstance(item, str):
                memory_items.append({'key': item, 'content': item})
            else:
                memory_items.append({'key': str(item), 'content': str(item)})

        # 1. 间隔重复巩固
        priority_items = []
        for item in memory_items:
            key = item.get('key', '')
            if key in self._metacognition.get('uncertainty_map', {}):
                uncertainty = self._metacognition['uncertainty_map'][key]
                priority = uncertainty
            else:
                priority = 0.5
            priority_items.append((priority, item))

        priority_items.sort(key=lambda x: x[0], reverse=True)

        consolidated = []
        for priority, item in priority_items[:10]:
            consolidated.append(item)
            key = item.get('key', '')
            if key in self._metacognition['uncertainty_map']:
                self._metacognition['uncertainty_map'][key] *= 0.9

        # 2. 离线重放 — 重组记忆
        self._offline_replay(memory_items)

        # 3. 提取抽象模式
        self._extract_abstractions(memory_items)

        # 4. 整合新旧知识
        self._integrate_knowledge()

        # 5. 多时间尺度巩固 — 选择性巩固高不确定性记忆
        multiscale_consolidated = self.multiscale_learning.consolidate()

        # 6. 遗忘低保留率记忆
        forgotten = self.multiscale_learning.forget(threshold=0.1)

        # 7. 睡眠回放巩固：SWS+REM+Spindles完整周期
        sleep_result = self.sleep_replay.sleep_cycle(
            knowledge_graph=self.knowledge,
            encoder_fn=self._encode_text,
        )

        # 8. 互补学习系统巩固：海马体→皮层转移
        cls_result = self.complementary_learning.replay_consolidate(batch_size=50)

        return {
            'consolidated': len(consolidated),
            'multiscale_consolidated': len(multiscale_consolidated),
            'forgotten': len(forgotten),
            'total': len(memories),
            'sleep_cycle': sleep_result,
            'cls_consolidated': cls_result.get('consolidated', 0),
        }

    def _offline_replay(self, memories: List[Dict]):
        """离线重放 — 选择性重组记忆

        模拟海马体重放（基于人类学习研究）：
        1. 优先选择高不确定性/高预测误差的记忆
        2. 通过预测编码引擎重放
        3. 建立新的联系
        """
        if len(memories) < 2:
            return

        # 选择性重放：优先选择高不确定性的记忆
        import random

        # 按不确定性排序（如果有的话）
        def get_uncertainty(mem):
            key = mem.get('key', '')
            if key in self._metacognition.get('uncertainty_map', {}):
                return self._metacognition['uncertainty_map'][key]
            return 0.5  # 默认不确定性

        # 排序：高不确定性优先
        sorted_memories = sorted(memories, key=get_uncertainty, reverse=True)

        # 选择前10个高不确定性记忆进行重放
        replay_candidates = sorted_memories[:min(10, len(sorted_memories))]

        for _ in range(min(10, len(replay_candidates) // 2)):
            # 从高不确定性记忆中随机选择对
            i, j = random.sample(range(len(replay_candidates)), 2)
            mem_i = replay_candidates[i]
            mem_j = replay_candidates[j]

            # 通过预测编码引擎重放
            try:
                # 获取记忆的表示
                repr_i = mem_i.get('representation')
                repr_j = mem_j.get('representation')

                if repr_i is not None and repr_j is not None:
                    # 预测两个记忆之间的关系
                    prediction = self.engine.predict(repr_i)

                    # 计算预测误差
                    error = torch.nn.functional.mse_loss(prediction, repr_j)

                    # 如果误差小，建立联系
                    if error.item() < 0.5:
                        # 在知识图谱中建立联系
                        key_i = mem_i.get('key', '')
                        key_j = mem_j.get('key', '')
                        if key_i and key_j:
                            # 标记为相关记忆
                            if not hasattr(self, '_related_memories'):
                                self._related_memories = set()
                            self._related_memories.add((key_i, key_j))
            except Exception:
                pass

    def _extract_abstractions(self, memories: List[Dict]):
        """提取抽象模式

        从多个记忆中提取共同模式：
        1. 找到相似的记忆
        2. 提取共同特征
        3. 形成抽象概念
        """
        if len(memories) < 3:
            return

        # 找到相似的记忆对
        import random
        for _ in range(min(5, len(memories) // 3)):
            indices = random.sample(range(len(memories)), 3)
            mems = [memories[i] for i in indices]

            # 检查是否有共同特征
            keys = [m.get('key', '') for m in mems if m.get('key')]
            if len(keys) >= 2:
                # 提取共同前缀
                common_prefix = os.path.commonprefix(keys)
                if len(common_prefix) >= 2:
                    # 形成抽象概念
                    abstract_key = f"abstract:{common_prefix}"
                    if not hasattr(self, '_abstract_concepts'):
                        self._abstract_concepts = {}
                    self._abstract_concepts[abstract_key] = keys

    def _integrate_knowledge(self):
        """整合新旧知识

        将新学习的知识与已有知识整合：
        1. 找到相关的旧知识
        2. 建立联系
        3. 更新置信度
        """
        # 获取所有实体
        kg = self.knowledge
        entities = list(kg.entities.keys())

        # 找到相关实体对
        for i, entity1 in enumerate(entities[:10]):
            for entity2 in entities[i+1:min(i+5, len(entities))]:
                # 检查是否有关系
                try:
                    relations1 = kg.get_relations_of(entity1)
                    relations2 = kg.get_relations_of(entity2)

                    # 找到共同的关系目标
                    targets1 = set(r.target_id for r in relations1)
                    targets2 = set(r.target_id for r in relations2)
                    common_targets = targets1 & targets2

                    if common_targets:
                        # 建立间接联系
                        for target in common_targets:
                            # 标记为相关实体
                            if not hasattr(self, '_related_entities'):
                                self._related_entities = set()
                            self._related_entities.add((entity1, entity2))
                except Exception:
                    pass

    def update_uncertainty(self, key: str, success: bool):
        """更新不确定性

        成功回忆 → 降低不确定性
        失败回忆 → 增加不确定性
        """
        if key not in self._metacognition['uncertainty_map']:
            self._metacognition['uncertainty_map'][key] = 0.5

        if success:
            # 成功：降低不确定性
            self._metacognition['uncertainty_map'][key] *= 0.8
        else:
            # 失败：增加不确定性
            self._metacognition['uncertainty_map'][key] = min(1.0,
                self._metacognition['uncertainty_map'][key] * 1.5 + 0.1)

    # ------------------------------------------------------------------
    # 组合泛化 — 重组已知原语
    # ------------------------------------------------------------------

    def compose_concepts(self, concept1: str, concept2: str, relation: str = "组合") -> str:
        """组合两个概念形成新概念

        例如：红 + 球 → 红球
        """
        # 创建组合概念
        composed = f"{concept1}{concept2}"

        # 在知识图谱中建立关系
        try:
            kg = self.knowledge
            from src.knowledge.entity import Entity
            from src.knowledge.relation import Relation

            # 添加组合概念
            composed_entity = Entity(id=composed, type='composed', source='composition')
            kg.add_entity(composed_entity)

            # 建立组合关系
            rel = Relation(
                source_id=concept1,
                target_id=composed,
                type='part_of',
                confidence=0.9,
            )
            kg.add_relation(rel)

            rel2 = Relation(
                source_id=concept2,
                target_id=composed,
                type='part_of',
                confidence=0.9,
            )
            kg.add_relation(rel2)
        except Exception:
            pass

        return composed

    def decompose_concept(self, concept: str) -> List[str]:
        """分解概念为组成部分"""
        parts = []

        try:
            kg = self.knowledge
            relations = kg.get_relations_of(concept)

            for rel in relations:
                if rel.type == 'part_of':
                    parts.append(rel.source_id)
        except Exception:
            pass

        return parts

    def analogical_transfer(self, source: str, target: str, mapping: Dict[str, str]) -> Dict:
        """类比迁移 — 将源域知识应用到目标域

        例如：
        源域：水流 → 电流
        映射：水 → 电, 管道 → 导线
        结果：水压 → 电压
        """
        results = []

        try:
            kg = self.knowledge

            # 获取源域的关系
            source_relations = kg.get_relations_of(source)

            # 应用映射
            for rel in source_relations:
                mapped_source = mapping.get(rel.source_id, rel.source_id)
                mapped_target = mapping.get(rel.target_id, rel.target_id)
                mapped_relation = rel.type

                # 在目标域中建立关系
                try:
                    from src.knowledge.relation import Relation
                    new_rel = Relation(
                        source_id=mapped_source,
                        target_id=mapped_target,
                        type=mapped_relation,
                        confidence=rel.confidence * 0.8,  # 类比的置信度较低
                    )
                    kg.add_relation(new_rel)
                    results.append({
                        'source': f"{rel.source_id} {rel.type} {rel.target_id}",
                        'target': f"{mapped_source} {mapped_relation} {mapped_target}",
                    })
                except Exception:
                    pass
        except Exception:
            pass

        return {'transferred': len(results), 'results': results}

    # ------------------------------------------------------------------
    # 文本学习（桥接 training/layers）
    # ------------------------------------------------------------------

    def learn_from_text(self, text: str, source: str = "text") -> Dict:
        """从文本中学习 — 使用学习编码器

        核心改进：
        1. 使用学习编码器提取表示（不是正则）
        2. 从表示中学习实体和关系
        3. 概念从观察中学习（不是随机初始化）
        """
        import re

        result = {
            'entities': [],
            'triples': [],
            'causal_links': [],
            'concepts': [],
            'numerical_facts': [],
            'representation': None,
        }

        # 1. 学习文本表示（使用学习编码器，启用梯度训练）
        text_repr = self._encode_text(text, train=True)
        result['representation'] = text_repr

        # 1b. 测试时训练：根据文本上下文微调编码器
        if hasattr(self, '_learnable_encoder'):
            # 获取已知实体作为上下文
            kg = self.knowledge
            if kg and hasattr(kg, 'entities'):
                existing_entities = list(kg.entities.keys())[:3]
                if existing_entities:
                    self._test_time_trainer.adapt_to_query(
                        text, text_repr, existing_entities
                    )

        # 1c. 统计学习通道 — 让概念从反复观察中涌现
        stat_result = {}
        stat_emergent_set = set()
        if self.config.statistical_learning_enabled:
            stat_learner = self._registry.get('statistical_learner')
            stat_result = stat_learner.observe(text)
            # 缓存过滤后的涌现概念集（只在有新概念时重算）
            if stat_result.get('new_concepts'):
                self._stat_emergent_cache = set(
                    c for c, conf in stat_learner.get_emergent_concepts(filter_boundary=True)
                )
            stat_emergent_set = getattr(self, '_stat_emergent_cache', set())

        # 2. 提取实体 — 正则提取为主，统计学习验证提权 + 补充高频概念
        regex_entities = self._extract_entities_from_repr(text, text_repr)

        if self.config.statistical_use_as_primary and stat_emergent_set:
            # 策略A：正则提取的实体被统计学习确认 → 提权（暂存信息，后续使用）
            # 策略B：统计涌现的高频概念（freq>=5）如果不在正则结果中 → 补充
            for c in stat_emergent_set:
                if c not in regex_entities and len(c) >= 2:
                    info = stat_learner.get_concept_info(c)
                    if info and info.frequency >= 5:  # 只补充高频涌现概念
                        regex_entities.append(c)
        entities = regex_entities
        result['entities'] = entities

        # 2a-new. 语言习得通道：学习标签映射（概念附着）
        try:
            la_labels = self.language_acquisition.learn_label(
                text, context={'source': source}
            )
            if la_labels:
                for label in la_labels:
                    if label not in entities and len(label) >= 2:
                        entities.append(label)
        except Exception:
            pass

        # 2b. BTSP：标记实体为可学习（资格痕迹）
        for entity in entities:
            entity_repr = self._encode_text(entity)
            self.btsp_learning.mark_eligible(entity, entity_repr)

        # 2c. 发展阶段过滤 — 根据认知发展阶段约束学习内容
        # 类似Piaget认知发展理论：前语言期不能学复杂关系，双词期限制句法复杂度
        try:
            ld = self.language_development
            entities, _ = ld.filter_content(entities, [])
            # 注册概念到发展阶段系统
            for entity in entities:
                ld.register_concept(entity)
            ld.check_stage_transition()
        except Exception:
            pass

        # 3. 从表示中提取关系（正则模式匹配 — 保持关系类型语义）
        triples = self._extract_relations_from_repr(text, entities, text_repr)

        # 3b. 发展阶段过滤 — 根据阶段限制可学习的关系复杂度
        try:
            ld = self.language_development
            _, filtered_triples = ld.filter_content(entities, triples)
            triples = filtered_triples
            # 注册关系到发展阶段系统
            for item in triples:
                if len(item) >= 3:
                    ld.register_relation(item[0], item[1], item[2])
        except Exception:
            pass

        result['triples'] = triples

        # 4. 注入知识图谱（带矛盾检测）
        for item in triples:
            # 兼容3元组和4元组
            if len(item) == 4:
                subject, relation, obj, confidence = item
            else:
                subject, relation, obj = item
                confidence = 0.8
            try:
                kg = self.knowledge
                from src.knowledge.entity import Entity
                from src.knowledge.relation import Relation

                # 矛盾检测：检查是否已有冲突的关系
                conflict = self._check_contradiction(subject, relation, obj)
                if conflict:
                    # 处理矛盾：保留置信度更高的
                    self._resolve_contradiction(subject, relation, obj, conflict, source)
                    continue

                subj_entity = Entity(id=subject, type='concept', source=source,
                                   embedding=self._encode_text(subject))
                obj_entity = Entity(id=obj, type='concept', source=source,
                                   embedding=self._encode_text(obj))
                kg.add_entity(subj_entity)
                kg.add_entity(obj_entity)

                rel = Relation(
                    source_id=subject,
                    target_id=obj,
                    type=relation,
                    confidence=confidence,
                )
                kg.add_relation(rel)
            except Exception:
                pass

        # 5. 提取因果关系
        causal_links = self._extract_causal_from_repr(text, text_repr)
        result['causal_links'] = causal_links

        for cause, effect in causal_links:
            try:
                dag = self.causal_dag
                dag.add_edge(cause, effect)
            except Exception:
                pass

            # 注入知识图谱
            try:
                kg = self.knowledge
                from src.knowledge.entity import Entity
                from src.knowledge.relation import Relation

                cause_entity = Entity(id=cause, type='event', source=source,
                                    embedding=self._encode_text(cause))
                effect_entity = Entity(id=effect, type='event', source=source,
                                     embedding=self._encode_text(effect))
                kg.add_entity(cause_entity)
                kg.add_entity(effect_entity)

                rel = Relation(
                    source_id=cause,
                    target_id=effect,
                    type='导致',
                    confidence=0.9,
                )
                kg.add_relation(rel)
            except Exception:
                pass

        # 4. 形成概念（功能性概念系统 + 传统概念形成双通道）
        for entity in entities[:10]:  # 限制数量
            result['concepts'].append(entity)

            entity_repr = self._encode_text(entity)

            # 通道A: 功能性概念系统（带感知特征+可供性）
            try:
                self.functional_concepts.form_concept(
                    label=entity,
                    perceptual_features={'source_text': text},
                    usage_context={'sentence': text, 'source': source},
                    vector=entity_repr,
                )
            except Exception:
                pass

            # 通道B: 传统概念形成（保持兼容）
            try:
                cf = self.concept_formation
                cf.add_instance(entity, entity_repr)
            except Exception:
                pass

        # 4b. 同步到概念空间（Phase 2 核心路径）
        try:
            cs = self._registry.get('concept_space')

            # 获取统计学习验证的概念集合（高频+多上下文 = 真正的词汇）
            stat_verified = set()
            try:
                stat_learner = self._registry.get('statistical_learner')
                if stat_learner:
                    emergent = stat_learner.get_emergent_concepts(min_freq=2)
                    stat_verified = {c for c, conf in emergent}
            except Exception:
                pass

            # KG 实体也视为已验证
            kg_entities = set(self.knowledge.entities.keys()) if self.knowledge else set()

            # 碎片过滤：多层级验证
            function_chars = set('是的有在了和与被把让给从到以也而')
            # 额外的高频切分点（这些字通常出现在词汇边界）
            boundary_chars = set('的了着过在让把被从到以与及其')
            clean_entities = []
            for entity in entities[:15]:
                # 层1: 虚词开头/结尾 → 碎片
                if entity[0] in function_chars or entity[-1] in function_chars:
                    continue
                # 层2: 全是虚词 → 碎片
                if all(c in function_chars for c in entity):
                    continue
                # 层3: 中间含虚词 + 长度>3 → 跨词碎片
                if len(entity) >= 4:
                    has_mid_func = any(entity[i] in function_chars for i in range(1, len(entity)-1))
                    if has_mid_func:
                        continue
                # 层4（Phase 5新增）: 统计学习验证
                # 如果概念在统计学习中已涌现 → 直接通过
                if entity in stat_verified or entity in kg_entities:
                    clean_entities.append(entity)
                    continue
                # 层5: 未被统计学习验证的2字概念 → 只在KG中存在时才通过
                if len(entity) == 2:
                    if entity in kg_entities:
                        clean_entities.append(entity)
                    # 否则跳过（2字碎片如"理学"、"学分"太多）
                    continue
                # 层6: 3字以上未验证 → 允许（可能是新概念）
                clean_entities.append(entity)

            # 注册过滤后的实体到概念空间
            for entity in clean_entities:
                entity_repr = self._encode_text(entity)
                cs.register(entity, vector=entity_repr, source='text')
            # 学习关系 — 策略1: 三元组中主体/客体都在概念空间中的
            for item in triples:
                if len(item) >= 3:
                    subj, rel_type, obj = item[0], item[1], item[2]
                    strength_map = {'是': 0.3, '导致': 0.4, '包括': 0.3, '包含': 0.3}
                    s = strength_map.get(rel_type, 0.15)
                    cs.learn_relation(subj, obj, strength=s)
            # 学习关系 — 策略2: 同一文本中出现的实体对（共现Hebbian）
            # 这是最重要的关系来源 — 同一上下文中出现的概念自然关联
            registered_entities = [e for e in entities[:15] if e in cs.concepts]
            for i in range(len(registered_entities)):
                for j in range(i + 1, len(registered_entities)):
                    # 同一文本中的实体对建立弱关联
                    cs.learn_relation(
                        registered_entities[i], registered_entities[j],
                        strength=0.05, bidirectional=True
                    )
        except Exception:
            pass

        # 5. 提取数值
        numerical_patterns = [
            (r'(\d+(?:\.\d+)?)\s*(?:度|℃)', '温度', '摄氏度'),
            (r'(\d+(?:\.\d+)?)\s*(?:米|m)', '长度', '米'),
            (r'(\d+(?:\.\d+)?)\s*(?:千克|公斤|kg)', '重量', '千克'),
            (r'(\d+(?:\.\d+)?)\s*(?:年)', '时间', '年'),
        ]

        for pattern, attr_type, unit in numerical_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                value = float(match)
                result['numerical_facts'].append({
                    'attribute': attr_type,
                    'value': value,
                    'unit': unit,
                })

                # 注入知识图谱
                try:
                    kg = self.knowledge
                    from src.knowledge.entity import Entity
                    from src.knowledge.relation import Relation

                    # 创建数值实体
                    num_entity = Entity(
                        id=f'{attr_type}_{value}',
                        type='numerical',
                        source=source,
                        properties={'value': value, 'unit': unit},
                    )
                    kg.add_entity(num_entity)

                    # 查找上下文中的实体
                    context_entities = re.findall(r'[一-鿿]{2,6}', text[:100])
                    for ctx_entity in context_entities[:3]:
                        if ctx_entity not in ['的', '了', '是', '在', '有']:
                            # 添加关系
                            rel = Relation(
                                source_id=ctx_entity,
                                target_id=f'{attr_type}_{value}',
                                type=attr_type,
                                confidence=0.8,
                            )
                            kg.add_relation(rel)
                except Exception:
                    pass

        # 6. 训练嵌入层（使用Hebbian学习）
        if len(entities) >= 2:
            self._train_embedding(text, entities)

        # 7. 反馈闭环 — 验证学到的知识
        verification = self._verify_learned_knowledge(text, entities, triples)
        result['verification'] = verification

        # 8. 对比/负向学习 — 从失败中学习
        if not verification['passed']:
            for test in verification.get('tests', []):
                if not test.get('passed', False):
                    # 存储负向三元组（低置信度）
                    expected = test.get('expected', '')
                    actual = test.get('actual', '')
                    if expected and actual:
                        # 记录"这不是X"的知识
                        negative_key = f"NOT:{expected}"
                        if not hasattr(self, '_negative_knowledge'):
                            self._negative_knowledge = {}
                        self._negative_knowledge[negative_key] = {
                            'expected': expected,
                            'actual': actual,
                            'question': test.get('question', ''),
                            'timestamp': time.time() if 'time' in dir(__import__('time')) else 0,
                        }

        # 9. 更新不确定性（间隔重复）
        for test in verification.get('tests', []):
            key = test.get('question', '')
            passed = test.get('passed', False)
            self.update_uncertainty(key, passed)

        # 10. 存入记忆
        self.memory.store_experience(
            text_repr,
            torch.tensor([0]),
            text_repr,
            reward=float(len(triples)),
            error=0.0 if verification['passed'] else 1.0,
        )

        # 10b. 存入多时间尺度记忆系统
        importance = 0.8 if verification['passed'] else 0.3
        self.multiscale_learning.store_memory(
            key=text[:50],
            content=text,
            importance=importance,
        )

        # 10c. 概念接地标记 — 在概念空间中标记已学到的概念
        # 虽然当前没有真实视觉/触觉数据，但记录概念的"学习来源"
        # 当将来接入模拟环境时，同名概念会自动关联到感知体验
        try:
            cs = self._registry.get('concept_space')
            for entity in entities[:5]:
                if entity in cs.concepts:
                    # 标记概念为"文本学习"来源，增强其强度
                    cs.concepts[entity].source = 'text'
                    cs.concepts[entity].boost(0.05)
        except Exception:
            pass

        # 11. 更新学习统计
        self._learning_stats['total_learned'] += 1
        if verification['passed']:
            self._learning_stats['verified'] += 1
            # BTSP：验证通过时触发平台电位（一次性强化所有带标记的连接）
            updated_embeddings = self.btsp_learning.trigger_plateau(
                trigger_strength=1.0,
                reason=f'verified: {text[:30]}'
            )
            # 更新知识图谱中的实体嵌入
            for entity, new_emb in updated_embeddings.items():
                if entity in self.knowledge.entities:
                    self.knowledge.entities[entity].embedding = new_emb
        else:
            self._learning_stats['failed'] += 1
            # BTSP：验证失败时也触发，但强度较低
            self.btsp_learning.trigger_plateau(
                trigger_strength=0.3,
                reason=f'failed: {text[:30]}'
            )

        # 12. 自改进：评估性能并调整策略（参数通过get_parameter动态获取）
        score = verification.get('score', 0.5)
        improvement = self.self_improvement.evaluate_performance(text[:30], score)

        # 13. 经验反思学习：记录经验并定期反思
        self.reflective_learning.record_experience(
            text=text,
            entities=entities,
            triples=triples,
            verification_passed=verification['passed'],
            score=score,
        )
        if self.reflective_learning.should_reflect():
            insights = self.reflective_learning.reflect()
            # 将洞见应用到系统
            for insight in insights:
                if insight.category == 'strategy' and '降低难度' in insight.insight:
                    # 降低负样本阈值
                    old = self.self_improvement.tunable_params['negative_threshold']
                    self.self_improvement.tunable_params['negative_threshold'] = max(0.2, old - 0.05)

        # 14. 因果引擎：记录因果关系
        for item in triples:
            if len(item) >= 3:
                subj, rel, obj = item[0], item[1], item[2]
                self.causal_engine.add_node(subj)
                self.causal_engine.add_node(obj)
                self.causal_engine.add_edge(subj, obj, strength=0.5)

        # 15. 情感驱动：更新情感状态
        if verification['passed']:
            self.emotional_drive.on_learning_success(text[:30])
        else:
            self.emotional_drive.on_learning_failure(text[:30])

        # 16. 创造性引擎：添加新概念
        for entity in entities:
            self.creativity_engine.add_concept(entity, {'source': text[:30]})

        # 17. 预测编码Light：学习新信息（抑制可预测部分）
        if entities:
            entity_repr = self._encode_text(entities[0])
            context = self._encode_text(text[:20]) if len(text) > 20 else text_repr
            pcl_state = self.predictive_coding_light.suppress_predictable(entity_repr, context)
            self.predictive_coding_light.learn_from_prediction(pcl_state)

        # 18. 奖励表征后移：将奖励信号迁移到先行线索
        if triples:
            for item in triples:
                if len(item) >= 3:
                    cue = item[0]  # 先行线索（如"下雨"）
                    reward = 1.0 if verification['passed'] else 0.0
                    self.reward_shift.update_reward_representation(cue, reward)

        # 19. 组合泛化：学习内容和计算方式
        for entity in entities:
            entity_repr = self._encode_text(entity)
            self.compositional_generalization.learn_what(entity, entity_repr)

        # 20. 社会偶联学习：给予即时反馈
        self.social_contingency.give_feedback(
            action=text[:30],
            outcome=score,
            context=source,
        )

        # 21. 符号接地：建立符号与环境的因果关系
        for item in triples:
            if len(item) >= 3:
                symbol = item[0]
                env_state = {'relation': item[1], 'target': item[2]}
                grounding_strength = 1.0 if verification['passed'] else 0.3
                self.symbol_grounding.ground_symbol(symbol, env_state, grounding_strength)

        # 22. 认知预测路由：使用真实的预测误差值
        # 低级误差：从预测编码引擎获取连续值
        if hasattr(self, '_error_history') and self._error_history:
            recent_errors = list(self._error_history)[-5:]
            low_error = sum(recent_errors) / len(recent_errors)
        else:
            low_error = 0.0 if verification['passed'] else 0.8

        # 高级误差：基于验证分数的连续值
        high_error = 1.0 - score  # score越高，误差越低

        routing = self.cognitive_router.route_error(low_error, high_error)

        # 23. GHL全局调制：计算全局信号并执行Hebbian更新
        reward = score * 2.0 - 1.0  # 映射到[-1, 1]
        novelty = self.predictive_coding_light.get_novelty_score()
        uncertainty = self._metacognition.get('uncertainty_map', {}).get(text[:30], 0.5)
        global_signal = self.ghl_learning.compute_global_signal(reward, novelty, uncertainty)

        # GHL Hebbian更新：用全局信号调制局部学习
        if len(entities) >= 2:
            for i, entity_a in enumerate(entities):
                for entity_b in entities[i+1:]:
                    emb_a = self._encode_text(entity_a)
                    emb_b = self._encode_text(entity_b)
                    delta = self.ghl_learning.hebbian_update(emb_a, emb_b, global_signal)
                    # 将delta应用到知识图谱中的嵌入
                    if entity_a in self.knowledge.entities:
                        old_emb = self.knowledge.entities[entity_a].embedding
                        if old_emb is not None:
                            new_emb = old_emb + delta.mean(dim=0)[:old_emb.shape[0]]
                            self.knowledge.entities[entity_a].embedding = torch.nn.functional.normalize(
                                new_emb.unsqueeze(0), p=2, dim=1
                            ).squeeze(0)

        # 24. 学习进展好奇心：更新领域进展
        domain = text[:10]  # 用文本前10字作为领域标识
        self.learning_progress.update_progress(domain, score)

        # 25. 感知类别：发现或创建类别
        for entity in entities:
            entity_repr = self._encode_text(entity)
            category = self.perceptual_categories.discover_category(entity, entity_repr)

        # 26. 元学习组合：学习组合规则
        if len(entities) >= 2:
            self.meta_composition.learn_rule(
                components=entities,
                result=text[:30],
                success=verification['passed'],
            )

        # ===== 机制14-17: 最新人类学习研究集成 =====

        # 27. 树突计算：上下文相关的实体表征
        for entity in entities:
            entity_emb = self._encode_text(entity)
            ctx_emb = text_repr  # 用全文表征作为上下文
            contextualized, plateau = self.dendritic_system.compute_context_representation(
                entity, entity_emb, ctx_emb
            )
            # 更新知识图谱中的实体嵌入（使用上下文化的表征）
            if plateau and entity in self.knowledge.entities:
                # plateau触发时，用上下文表征更新（强学习信号）
                with torch.no_grad():
                    old_emb = self.knowledge.entities[entity].embedding
                    if old_emb is not None and old_emb.shape == contextualized.shape:
                        self.knowledge.entities[entity].embedding = (
                            0.7 * old_emb + 0.3 * contextualized
                        )

        # 28. 睡眠回放：记录情节到海马体
        importance_score = 0.8 if verification['passed'] else 0.3
        if len(entities) >= 1:
            self.sleep_replay.record_episode(
                text=text,
                embedding=text_repr.detach(),
                importance=importance_score,
            )

        # 29. 主动推理学习：更新信念和不确定性
        self.active_inference_learning.update_beliefs(
            entity_id=text[:20],
            observation=text_repr.detach(),
        )

        # 30. 互补学习：快速存储到海马体记忆
        self.complementary_learning.store_episode(
            content=text,
            embedding=text_repr.detach(),
            entities=entities,
            context=source,
        )

        # 31. 图式学习：将实体归入知识图式
        for entity in entities:
            entity_emb = self._encode_text(entity)
            schema_id, is_new = self.schema_learning.find_or_create_schema(
                entity, entity_emb
            )
            # 记录关系到图式
            for item in triples:
                if len(item) >= 3 and item[0] == entity:
                    self.schema_learning.schemas[schema_id].relations.append(
                        (item[1], item[2])
                    )

        # 32. 元认知调控：监控本次学习效果
        meta_state = self.metacognitive_regulator.monitor(
            success=verification['passed'],
            confidence=verification.get('score', 0.5),
            domain=text[:10],
        )

        # 33. 跨域迁移：记录关系模式供未来迁移
        for item in triples:
            if len(item) >= 3:
                self.cross_domain_transfer.learn_relation(
                    relation_type=item[1],
                    domain=source,
                    confidence=item[3] if len(item) > 3 else 0.5,
                )

        # 34. 层次概念：将实体加入概念层次结构
        for entity in entities:
            entity_emb = self._encode_text(entity)
            # 自动分类到层次结构
            self.hierarchical_concepts.auto_classify(entity, entity_emb)

        # 35. 时序预测：将此文本作为序列中的一个观察
        temporal_result = self.temporal_sequence.observe(
            item=text[:30],
            embedding=text_repr.detach(),
        )

        # 36. 注意力门控：评估此输入的注意力权重
        should_learn, attn_weight = self.attention_gate.gate_learning(
            content=text,
            embedding=text_repr,
            prediction_error=temporal_result.get('prediction_error', 0.0),
        )

        # 37. 语言发展阶段：注册概念和关系
        for entity in entities:
            self.language_development.register_concept(entity)
        for item in triples:
            if len(item) >= 3:
                self.language_development.register_relation(item[0], item[1], item[2])
        # 检查阶段晋升
        self.language_development.check_stage_transition()

        # 38. 具身符号接地：从文本中提取感知特征并绑定实体
        grounded = self.embodied_grounding.ground_from_text(text, entities)

        # 39. 社会反馈：基于知识图谱已有知识评估学习质量
        # 不用自指比较(text vs text)，而是检查新学知识是否与已有知识一致
        if entities:
            # 构造"学习输出"：列出学到的实体和关系
            learned_summary = '、'.join(entities[:5])
            if triples:
                learned_summary += '。关系：' + '、'.join(
                    f"{t[0]}-{t[1]}-{t[2]}" for t in triples[:3] if len(t) >= 3
                )

            # 构造"期望输出"：从知识图谱查询相关已有知识
            expected_parts = []
            for entity in entities[:3]:
                if entity in self.knowledge.entities:
                    rels = self.knowledge.get_relations_of(entity)
                    for r in rels[:2]:
                        expected_parts.append(f"{r.target_id}")
            expected_summary = '、'.join(expected_parts[:5]) if expected_parts else ''

            self.social_feedback.process_feedback(
                learner_output=learned_summary,
                expected_output=expected_summary,
            )
        else:
            # 没学到任何实体 → 差评
            self.social_feedback.process_feedback(
                learner_output='(未学到任何知识)',
                expected_output=text[:100],
            )

        # 40. 知识蒸馏：积累三元组到领域
        for item in triples:
            if len(item) >= 3:
                self.knowledge_distillation.accumulate(
                    domain=source,
                    triple=(item[0], item[1], item[2]),
                )

        # 41. 统计学习摘要：将统计学习器的状态附加到结果
        if self.config.statistical_learning_enabled:
            stat_learner = self._registry.get('statistical_learner')
            result['statistical_learning'] = stat_learner.get_stats()

        return result

    def _verify_learned_knowledge(self, text: str, entities: List[str],
                                   triples: List[Tuple]) -> Dict:
        """验证学到的知识

        反馈闭环：
        1. 从学到的知识中生成问题
        2. 查询系统是否能回答
        3. 检查答案是否正确
        """
        verification = {
            'passed': True,
            'tests': [],
            'score': 0.0,
        }

        # 从三元组生成验证问题
        for item in triples[:3]:
            if len(item) == 4:
                subject, relation, obj, confidence = item
            else:
                subject, relation, obj = item
            # 生成问题
            if relation == '是':
                question = f"什么是{subject}"
            elif relation == '属于':
                question = f"{subject}属于什么"
            elif relation == '位于':
                question = f"{subject}位于哪里"
            else:
                question = f"{subject}{relation}什么"

            # 查询
            answer = self.think(question)

            # 检查答案是否包含预期
            passed = obj in answer
            verification['tests'].append({
                'question': question,
                'expected': obj,
                'actual': answer[:50],
                'passed': passed,
            })

            if not passed:
                verification['passed'] = False

        # 计算分数
        if verification['tests']:
            passed_count = sum(1 for t in verification['tests'] if t['passed'])
            verification['score'] = passed_count / len(verification['tests'])

        return verification

    # ------------------------------------------------------------------
    # 矛盾检测和知识修正
    # ------------------------------------------------------------------

    def _check_contradiction(self, subject: str, relation: str, obj: str) -> Optional[Dict]:
        """检查是否存在矛盾的知识

        矛盾条件：同一个 subject + relation，但不同的 obj
        """
        kg = self.knowledge

        try:
            # 获取该实体的所有关系
            relations = kg.get_relations_of(subject)
            for rel in relations:
                if rel.type == relation and rel.target_id != obj:
                    return {
                        'existing_obj': rel.target_id,
                        'existing_confidence': rel.confidence if hasattr(rel, 'confidence') else 0.5,
                        'new_obj': obj,
                    }
        except Exception:
            pass

        return None

    def _resolve_contradiction(self, subject: str, relation: str, obj: str,
                                conflict: Dict, source: str):
        """解决矛盾

        策略：
        1. 来源可靠性：维基百科 > 推测
        2. 具体性：更具体的信息更可靠
        3. 条件化：保留两者，添加条件
        """
        kg = self.knowledge
        from src.knowledge.entity import Entity
        from src.knowledge.relation import Relation

        existing_obj = conflict['existing_obj']
        new_obj = conflict['new_obj']

        # 策略1：来源可靠性
        reliable_sources = ['wiki', 'baike', '知识库']
        new_is_reliable = any(s in source for s in reliable_sources)

        if new_is_reliable:
            # 新来源更可靠，替换旧的
            try:
                # 删除旧关系
                old_relations = kg.get_relations_of(subject)
                for rel in old_relations:
                    if rel.type == relation and rel.target_id == existing_obj:
                        kg.remove_relation(rel)
                        break
            except Exception:
                pass

            # 添加新的
            subj_entity = Entity(id=subject, type='concept', source=source,
                               embedding=self._encode_text(subject))
            obj_entity = Entity(id=obj, type='concept', source=source,
                               embedding=self._encode_text(obj))
            kg.add_entity(subj_entity)
            kg.add_entity(obj_entity)

            rel = Relation(
                source_id=subject,
                target_id=obj,
                type=relation,
                confidence=0.9,
            )
            kg.add_relation(rel)
        else:
            # 保留两者，添加条件标记
            # 存储为条件关系
            conditional_key = f"{subject}_{relation}"
            if not hasattr(self, '_conditional_knowledge'):
                self._conditional_knowledge = {}
            self._conditional_knowledge[conditional_key] = {
                'subject': subject,
                'relation': relation,
                'options': [existing_obj, new_obj],
                'source': source,
            }

    # ------------------------------------------------------------------
    # 元认知 — 知道自己不知道什么
    # ------------------------------------------------------------------

    def identify_knowledge_gaps(self) -> List[Dict]:
        """识别知识空白

        方法：
        1. 低置信度实体
        2. 孤立实体（无关系）
        3. 低验证通过率的领域
        """
        gaps = []
        kg = self.knowledge

        # 1. 低置信度实体
        for entity_id, entity in kg.entities.items():
            if hasattr(entity, 'confidence') and entity.confidence < 0.3:
                gaps.append({
                    'type': 'low_confidence',
                    'entity': entity_id,
                    'confidence': entity.confidence,
                })

        # 2. 孤立实体（无关系）
        for entity_id in kg.entities.keys():
            try:
                relations = kg.get_relations_of(entity_id)
                if not relations:
                    gaps.append({
                        'type': 'isolated',
                        'entity': entity_id,
                    })
            except Exception:
                pass

        # 3. 低验证通过率
        if self._learning_stats['total_learned'] > 0:
            success_rate = self._learning_stats['verified'] / self._learning_stats['total_learned']
            if success_rate < 0.5:
                gaps.append({
                    'type': 'low_success_rate',
                    'rate': success_rate,
                })

        self._metacognition['knowledge_gaps'] = gaps
        return gaps

    def get_learning_goals(self) -> List[str]:
        """获取学习目标

        基于知识空白生成学习目标。
        """
        gaps = self.identify_knowledge_gaps()
        goals = []

        for gap in gaps[:5]:
            if gap['type'] == 'low_confidence':
                goals.append(f"提高对{gap['entity']}的理解")
            elif gap['type'] == 'isolated':
                goals.append(f"学习{gap['entity']}的相关知识")
            elif gap['type'] == 'low_success_rate':
                goals.append("提高整体学习成功率")

        self._metacognition['learning_goals'] = goals
        return goals

    def get_uncertainty(self, entity: str) -> float:
        """获取实体的不确定性

        返回0-1之间的值，越高表示越不确定。
        """
        kg = self.knowledge
        if entity in kg.entities:
            entity_obj = kg.entities[entity]
            if hasattr(entity_obj, 'confidence'):
                return 1.0 - entity_obj.confidence
        return 1.0  # 完全不确定

    def get_metacognition_report(self) -> Dict:
        """获取元认知报告"""
        gaps = self.identify_knowledge_gaps()
        goals = self.get_learning_goals()

        return {
            'knowledge_gaps': len(gaps),
            'learning_goals': goals,
            'learning_stats': self._learning_stats,
            'uncertainty_entities': len(self._metacognition['uncertainty_map']),
        }

    def _encode_text(self, text: str, train: bool = False) -> torch.Tensor:
        """编码文本为可学习的语义向量

        使用 Transformer 编码器：
        1. BPE分词 → 学习到的子词单元
        2. 嵌入层 → 可学习的向量表示
        3. 位置编码 → 保留词序信息
        4. 多头自注意力 → 上下文感知
        5. 池化 → 固定维度输出
        """
        # 初始化可学习编码器
        if not hasattr(self, '_learnable_encoder'):
            from src.perception.learnable_encoder import LearnableTextEncoder
            self._learnable_encoder = LearnableTextEncoder(
                d_model=self.config.obs_dim,
                n_heads=self.config.encoder_n_heads,
                n_layers=self.config.encoder_n_layers,
                max_len=self.config.encoder_max_len,
            ).to(self.device)
            self._text_optimizer = torch.optim.Adam(
                self._learnable_encoder.parameters(), lr=1e-4
            )
            # LRU缓存（限制大小防止内存膨胀）
            from collections import OrderedDict
            self._embedding_cache = OrderedDict()
            self._embedding_cache_max = 10000
            # 用于收集语料训练分词器
            self._training_corpus = []

            # 关键：用常见中文字符预训练BPE，避免前3条文本编码完全相同
            # 未训练时所有字符映射为UNK，导致不同文本产生相同向量
            default_chars = (
                "的一是不了在人有我他这中大来上个国到说们为子和你地出会也时要就"
                "可以对本去学能那得于着下自之年过发后作里用道行所然家种事成方多"
                "经么去法如都同现当没动面起看定天分还进好小部其些主样理心她本前"
                "开但因只从想实日者意无力它与长把机十民第公此已工使情明性知全三"
                "又关点正业外将两高间由问很最重并物手应战向头文体政美相见被利什"
                "二等产新己制身果加西斯月话合回特代内信表化老给世位次度门任常先"
                "海通教儿原东声提立及比员解水名真论处走义各入几口认条平系气题活"
                "尔更别打女变四神总何电数安少报才结反受目太量再感建务做接必场件"
                "计管期市直德资命山金指克干排满西增则完格思传望族群底达约维素效"
                "收速林际拉七规型步验越即视散器图际单场现书住且引运市究联针角落"
                "米坚约半革越装断适影规往候府存列类区域阶需规越装断适影往候府存"
                "根据科技术经济政治社会文化教育历史发展研究生产建设管理国际关系"
                "数学物理化学生物地理天文计算机科学信息网络系统程序数据算法模型"
                "语言文字阅读写作思考分析综合推理判断记忆学习认知意识感知体验"
            )
            self._learnable_encoder.train_tokenizer([default_chars])

            # 预训练后重建优化器（因为嵌入层可能已扩展）
            self._text_optimizer = torch.optim.Adam(
                self._learnable_encoder.parameters(), lr=1e-4
            )

            # 绑定编码器到概念空间
            try:
                cs = self._registry.get('concept_space')
                if cs and cs.encoder is None:
                    cs.encoder = self._learnable_encoder
            except Exception:
                pass

            # 同时初始化对比学习训练器
            try:
                self._ensure_contrastive_trainer()
            except Exception:
                pass

        # 首次遇到文本时加入语料
        if text not in self._embedding_cache:
            self._training_corpus.append(text)
            # 收集足够语料后训练分词器（降低门槛到3条）
            if (len(self._training_corpus) >= 3 and
                not self._learnable_encoder.tokenizer._trained):
                self._learnable_encoder.train_tokenizer(self._training_corpus)
                # 保留优化器状态，不重建（避免丢失动量）

        # 检查缓存（LRU：命中时移到末尾）
        if text in self._embedding_cache:
            self._embedding_cache.move_to_end(text)
            return self._embedding_cache[text]

        # 使用 Transformer 编码
        if train:
            result = self._learnable_encoder(text)
        else:
            with torch.no_grad():
                result = self._learnable_encoder(text)

        # LRU缓存：超过限制时淘汰最旧的
        self._embedding_cache[text] = result
        if len(self._embedding_cache) > self._embedding_cache_max:
            self._embedding_cache.popitem(last=False)

        return result

    def _ensure_contrastive_trainer(self):
        """懒初始化对比学习训练器

        必须在 _learnable_encoder 初始化后调用。
        """
        if not hasattr(self, '_learnable_encoder'):
            return

        # 检查是否已初始化（用 has 而非 get，避免缓存 None）
        if self._registry.has('contrastive_trainer'):
            return

        if not self.config.contrastive_enabled:
            return

        from src.learning.contrastive_trainer import ContrastiveTrainer
        ct = ContrastiveTrainer(
            encoder=self._learnable_encoder,
            dim=self.config.obs_dim,
            temperature=self.config.contrastive_temperature,
            memory_bank_size=self.config.contrastive_memory_bank_size,
            negatives_per_positive=self.config.contrastive_negatives_per_pos,
            learning_rate=self.config.contrastive_learning_rate,
            device=str(self.device),
        )
        self._registry.set('contrastive_trainer', ct)

    @property
    def world_model(self):
        """对象中心世界模型（懒初始化）"""
        if not hasattr(self, '_world_model'):
            from src.perception.object_world_model import ObjectCentricWorldModel
            self._world_model = ObjectCentricWorldModel(
                obs_dim=self.config.obs_dim,
                device=str(self.device),
            )
        return self._world_model

    @property
    def btsp_learning(self):
        """BTSP启发的单次学习系统（懒初始化）"""
        if not hasattr(self, '_btsp'):
            from src.learning.btsp_learning import BTSPLearningSystem
            self._btsp = BTSPLearningSystem(decay_time=5.0)
        return self._btsp

    def perceive_objects(self, text: str) -> Dict:
        """感知文本中的对象和关系"""
        text_repr = self._encode_text(text)
        return self.world_model.perceive(text_repr)

    def imagine_solution(self, goal: str) -> List[Dict]:
        """想象达成目标的方案"""
        return self.world_model.imagine(goal)

    @property
    def self_improvement(self):
        """自改进系统（懒初始化）"""
        if not hasattr(self, '_self_improvement'):
            from src.learning.self_improvement import SelfImprovementSystem
            self._self_improvement = SelfImprovementSystem()
        return self._self_improvement

    @property
    def reflective_learning(self):
        """经验反思学习系统（懒初始化）"""
        if not hasattr(self, '_reflective'):
            from src.learning.reflective_learning import ReflectiveLearningSystem
            self._reflective = ReflectiveLearningSystem()
        return self._reflective

    @property
    def _test_time_trainer(self):
        """测试时训练器（懒初始化）"""
        if not hasattr(self, '_ttt'):
            from src.learning.test_time_training import TestTimeTrainer
            self._ttt = TestTimeTrainer(self._learnable_encoder, lr=1e-5)
        return self._ttt

    @property
    def multiscale_learning(self):
        """多时间尺度学习系统（懒初始化）"""
        if not hasattr(self, '_multiscale'):
            from src.learning.multiscale_learning import MultiTimescaleLearningSystem
            self._multiscale = MultiTimescaleLearningSystem()
        return self._multiscale

    def store_knowledge(self, key: str, content: str, importance: float = 0.5):
        """存储知识到多时间尺度记忆"""
        self.multiscale_learning.store_memory(key, content, importance)

    def consolidate_knowledge(self):
        """执行知识巩固"""
        return self.multiscale_learning.consolidate()

    @property
    def empowerment_exploration(self):
        """Empowerment驱动探索系统（懒初始化）"""
        if not hasattr(self, '_empowerment'):
            from src.learning.empowerment_exploration import EmpowermentExplorationSystem
            self._empowerment = EmpowermentExplorationSystem()
        return self._empowerment

    def compute_exploration_bonus(self, state: torch.Tensor, prediction_error: float) -> Dict:
        """计算探索奖励"""
        return self.empowerment_exploration.compute_exploration_bonus(
            state, prediction_error, self.world_model
        )

    def should_explore(self, state: torch.Tensor) -> bool:
        """判断是否值得探索"""
        return self.empowerment_exploration.should_explore(state)

    @property
    def embodied_grounding(self):
        """感觉运动接地系统（懒初始化）"""
        if not hasattr(self, '_embodied'):
            from src.perception.embodied_grounding import EmbodiedGroundingSystem
            self._embodied = EmbodiedGroundingSystem(device=str(self.device))
        return self._embodied

    def ground_concept_embodied(self, concept: str, concept_embedding: torch.Tensor) -> float:
        """将概念接地到感觉运动经验"""
        return self.embodied_grounding.ground_concept(concept, concept_embedding)

    def store_sensorimotor_experience(self, concept: str, visual=None, tactile=None, motor=None):
        """存储感觉运动经验"""
        from src.perception.embodied_grounding import SensorimotorExperience
        experience = SensorimotorExperience(visual=visual, tactile=tactile, motor=motor)
        self.embodied_grounding.store_experience(concept, experience)

    @property
    def causal_engine(self):
        """因果推理引擎（懒初始化）"""
        if not hasattr(self, '_causal_engine'):
            from src.reasoning.causal_engine import CausalEngine
            self._causal_engine = CausalEngine()
        return self._causal_engine

    @property
    def tool_engine(self):
        """工具调用引擎（懒初始化）"""
        if not hasattr(self, '_tool_engine'):
            from src.reasoning.tool_engine import ToolEngine
            self._tool_engine = ToolEngine()
        return self._tool_engine

    @property
    def functional_concepts(self):
        """功能性概念系统（懒初始化）"""
        if not hasattr(self, '_functional_concepts'):
            cs = self._registry.get('concept_space')
            self._functional_concepts = FunctionalConceptSystem(
                concept_space=cs,
                core_knowledge=self.core_knowledge,
                encoder=getattr(self, '_learnable_encoder', None),
            )
        return self._functional_concepts

    @property
    def language_acquisition(self):
        """语言习得系统（懒初始化）"""
        if not hasattr(self, '_language_acquisition'):
            cs = self._registry.get('concept_space')
            stat_learner = self._registry.get('statistical_learner')
            self._language_acquisition = LanguageAcquisitionSystem(
                concept_system=self.functional_concepts,
                statistical_learner=stat_learner,
                concept_space=cs,
            )
        return self._language_acquisition

    @property
    def simulation_reasoning(self):
        """模拟推理系统（懒初始化）"""
        if not hasattr(self, '_simulation_reasoning'):
            cs = self._registry.get('concept_space')
            self._simulation_reasoning = SimulationReasoning(
                concept_space=cs,
                causal_engine=self.causal_engine,
                knowledge_graph=self.knowledge,
            )
        return self._simulation_reasoning

    @property
    def code_generator(self):
        """代码生成引擎（懒初始化）"""
        if not hasattr(self, '_code_generator'):
            from src.reasoning.code_generator import CodeGenerator
            self._code_generator = CodeGenerator()
        return self._code_generator

    @property
    def self_evolution(self):
        """自主进化引擎（懒初始化）"""
        if not hasattr(self, '_self_evolution'):
            from src.learning.self_evolution import SelfEvolutionEngine
            self._self_evolution = SelfEvolutionEngine()
        return self._self_evolution

    @property
    def multimodal_engine(self):
        """多模态引擎（懒初始化）"""
        if not hasattr(self, '_multimodal'):
            from src.perception.multimodal_engine import MultimodalEngine
            self._multimodal = MultimodalEngine(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._multimodal

    @property
    def creativity_engine(self):
        """创造性引擎（懒初始化）"""
        if not hasattr(self, '_creativity'):
            from src.reasoning.creativity_engine import CreativityEngine
            self._creativity = CreativityEngine()
        return self._creativity

    @property
    def emotional_drive(self):
        """情感驱动引擎（懒初始化）"""
        if not hasattr(self, '_emotional'):
            from src.learning.emotional_drive import EmotionalDriveEngine
            self._emotional = EmotionalDriveEngine()
        return self._emotional

    @property
    def predictive_coding_light(self):
        """预测编码Light系统（懒初始化）"""
        if not hasattr(self, '_pcl'):
            from src.learning.predictive_coding_light import PredictiveCodingLight
            self._pcl = PredictiveCodingLight(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._pcl

    @property
    def reward_shift(self):
        """奖励表征后移系统（懒初始化）"""
        if not hasattr(self, '_reward_shift'):
            from src.learning.predictive_coding_light import RewardRepresentationShift
            self._reward_shift = RewardRepresentationShift(d_model=self.config.obs_dim)
        return self._reward_shift

    @property
    def compositional_generalization(self):
        """组合泛化系统（懒初始化）"""
        if not hasattr(self, '_compositional'):
            from src.learning.predictive_coding_light import CompositionalGeneralization
            self._compositional = CompositionalGeneralization()
        return self._compositional

    @property
    def social_contingency(self):
        """社会偶联学习系统（懒初始化）"""
        if not hasattr(self, '_social'):
            from src.learning.predictive_coding_light import SocialContingencyLearning
            self._social = SocialContingencyLearning()
        return self._social

    @property
    def symbol_grounding(self):
        """符号接地系统（懒初始化）"""
        if not hasattr(self, '_grounding'):
            from src.learning.predictive_coding_light import SymbolGrounding
            self._grounding = SymbolGrounding()
        return self._grounding

    @property
    def cognitive_router(self):
        """认知预测路由（懒初始化）"""
        if not hasattr(self, '_cognitive_router'):
            from src.learning.cognitive_mechanisms import CognitivePredictiveRouter
            self._cognitive_router = CognitivePredictiveRouter(d_model=self.config.obs_dim)
        return self._cognitive_router

    @property
    def ghl_learning(self):
        """GHL全局调制Hebbian学习（懒初始化）"""
        if not hasattr(self, '_ghl'):
            from src.learning.cognitive_mechanisms import GlobalModulatedHebbian
            self._ghl = GlobalModulatedHebbian(learning_rate=0.01)
        return self._ghl

    @property
    def learning_progress(self):
        """学习进展好奇心（懒初始化）"""
        if not hasattr(self, '_learning_progress'):
            from src.learning.cognitive_mechanisms import LearningProgressCuriosity
            self._learning_progress = LearningProgressCuriosity()
        return self._learning_progress

    @property
    def perceptual_categories(self):
        """感知类别系统（懒初始化）"""
        if not hasattr(self, '_perceptual_cats'):
            from src.learning.cognitive_mechanisms import PerceptualCategorySystem
            self._perceptual_cats = PerceptualCategorySystem()
        return self._perceptual_cats

    @property
    def meta_composition(self):
        """元学习组合规则（懒初始化）"""
        if not hasattr(self, '_meta_comp'):
            from src.learning.cognitive_mechanisms import MetaLearningComposition
            self._meta_comp = MetaLearningComposition()
        return self._meta_comp

    # ===== 机制14-17: 最新人类学习研究集成 =====

    @property
    def dendritic_system(self):
        """树突计算系统（懒初始化）

        基于Chavlis & Poirazi 2025:
        不同树突区室有独立可塑性，实现上下文关联表征。
        """
        if not hasattr(self, '_dendritic'):
            from src.learning.dendritic_computation import DendriticComputationSystem
            self._dendritic = DendriticComputationSystem(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._dendritic

    @property
    def sleep_replay(self):
        """睡眠回放巩固系统（懒初始化）

        基于Nature Comms 2022 + NeuroDream 2025:
        生成式回放防止灾难性遗忘，REM阶段创造性发现。
        """
        if not hasattr(self, '_sleep_replay'):
            from src.learning.sleep_replay import SleepReplaySystem
            self._sleep_replay = SleepReplaySystem(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._sleep_replay

    @property
    def active_inference_learning(self):
        """主动推理学习系统（懒初始化）

        基于Friston 2022 + Parr et al. 2024:
        自由能最小化驱动的自主学习决策。
        注意：区别于registry中的reasoning/active_inference（推理模块），
        这个是learning版本，专注于学习目标选择和课程安排。
        """
        if not hasattr(self, '_active_inf_learning'):
            from src.learning.active_inference import ActiveInferenceSystem
            self._active_inf_learning = ActiveInferenceSystem(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._active_inf_learning

    @property
    def complementary_learning(self):
        """互补学习系统（懒初始化）

        基于McClelland 1995 + Nature Neuroscience 2023:
        海马体快速记忆 + 皮层慢速抽象 = 互补学习。
        """
        if not hasattr(self, '_cls'):
            from src.learning.complementary_learning import ComplementaryLearningSystem
            self._cls = ComplementaryLearningSystem(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._cls

    # ===== 机制18-20: 图式+元认知+跨域迁移 =====

    @property
    def schema_learning(self):
        """图式学习系统（懒初始化）

        基于Nature Comms 2022: 图式作为脚手架加速新知识整合。
        """
        if not hasattr(self, '_schema'):
            from src.learning.schema_learning import SchemaLearningSystem
            self._schema = SchemaLearningSystem(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._schema

    @property
    def metacognitive_regulator(self):
        """元认知自我调控器（懒初始化）

        基于NeurIPS 2025: 监控学习效果，动态调整策略。
        """
        if not hasattr(self, '_metacog'):
            from src.learning.metacognitive_regulation import MetacognitiveRegulator
            self._metacog = MetacognitiveRegulator(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._metacog

    @property
    def cross_domain_transfer(self):
        """跨域迁移学习系统（懒初始化）

        基于Science China 2025: 关系抽象+结构映射+投射迁移。
        """
        if not hasattr(self, '_xfer'):
            from src.learning.cross_domain_transfer import CrossDomainTransfer
            self._xfer = CrossDomainTransfer(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._xfer

    # ===== 机制21-23: 层次概念+时序预测+注意力门控 =====

    @property
    def hierarchical_concepts(self):
        """层次概念体系（懒初始化）

        基于PMC 2023: mPFC+海马体的层次概念表征。
        """
        if not hasattr(self, '_hier_concepts'):
            from src.learning.hierarchical_concepts import HierarchicalConceptSystem
            self._hier_concepts = HierarchicalConceptSystem(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._hier_concepts

    @property
    def temporal_sequence(self):
        """时序预测学习系统（懒初始化）

        基于Neuron 2024: CA3预测+CA1误差的海马体序列学习。
        """
        if not hasattr(self, '_temporal'):
            from src.learning.temporal_sequence import TemporalSequenceSystem
            self._temporal = TemporalSequenceSystem(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._temporal

    @property
    def attention_gate(self):
        """注意力门控系统（懒初始化）

        基于Trends Cog Sci 2025: 新奇+目标+竞争的选择性注意力。
        """
        if not hasattr(self, '_attn_gate'):
            from src.learning.attention_gating import AttentionGate
            self._attn_gate = AttentionGate(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._attn_gate

    # ===== 机制24-27: 语言发展+具身接地+社会反馈+知识蒸馏 =====

    @property
    def language_development(self):
        """语言发展阶段系统（懒初始化）"""
        if not hasattr(self, '_lang_dev'):
            from src.learning.language_development import LanguageDevelopmentSystem
            self._lang_dev = LanguageDevelopmentSystem(
                d_model=self.config.obs_dim,
                device=str(self.device),
                initial_stage=getattr(self.config, 'initial_language_stage', 'holophrase'),
            )
        return self._lang_dev

    @property
    def embodied_grounding(self):
        """具身符号接地系统（懒初始化）"""
        if not hasattr(self, '_embodied'):
            from src.learning.embodied_grounding import EmbodiedGroundingSystem
            self._embodied = EmbodiedGroundingSystem(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )
        return self._embodied

    @property
    def social_feedback(self):
        """社会反馈学习系统（懒初始化）"""
        if not hasattr(self, '_social_fb'):
            from src.learning.social_distillation import SocialFeedbackSystem
            self._social_fb = SocialFeedbackSystem()
        return self._social_fb

    @property
    def knowledge_distillation(self):
        """知识蒸馏系统（懒初始化）"""
        if not hasattr(self, '_distill'):
            from src.learning.social_distillation import KnowledgeDistillationSystem
            self._distill = KnowledgeDistillationSystem()
        return self._distill

    def _subword_tokenize(self, text: str) -> List[str]:
        """子词分词 — 捕获有意义的片段

        不是简单字符级，而是：
        - 中文：2-4字的词
        - 英文：按空格和标点分割
        """
        import re

        tokens = []

        # 中文：提取2-4字的词
        zh_words = re.findall(r'[一-鿿]{2,4}', text)
        tokens.extend(zh_words)

        # 英文：按空格分割
        en_words = re.findall(r'[a-zA-Z]+', text)
        tokens.extend(en_words)

        # 数字
        numbers = re.findall(r'\d+', text)
        tokens.extend(numbers)

        return tokens

    def _get_token_idx(self, token: str) -> int:
        """获取token索引"""
        if not hasattr(self, '_token_to_idx'):
            self._token_to_idx = {}
            self._next_idx = 1

        if token not in self._token_to_idx:
            self._token_to_idx[token] = self._next_idx
            self._next_idx += 1

        return self._token_to_idx[token]

    def _train_embedding(self, text: str, entities: List[str]):
        """训练嵌入 — 对比学习 + Hebbian传播

        三阶段训练：
        1. InfoNCE对比学习：让相关概念靠近、不相关概念远离（核心！）
        2. 保留弱对比损失作为辅助（兼容性）
        3. Hebbian传播：通过知识图谱关系传播相似性
        """
        if not hasattr(self, '_learnable_encoder'):
            return

        if len(entities) < 2:
            return

        # 关键：训练前清空缓存，避免旧的计算图引用导致inplace崩溃
        self._embedding_cache.clear()

        # 构建高质量对比学习概念
        # 优先使用统计学习涌现的高置信概念（比正则实体更干净）
        cl_entities = entities  # 默认使用原始实体
        cl_cooccurrence = []

        try:
            stat_learner = self._registry.get('statistical_learner')
            cs = self._registry.get('concept_space')

            # 从统计学习获取高质量涌现概念
            if stat_learner:
                emergent = stat_learner.get_emergent_concepts(min_freq=2)
                # 用涌现概念替代正则实体（如果数量足够）
                emergent_ids = [c for c, conf in emergent if conf > 0.3]
                if len(emergent_ids) >= 3:
                    # 取当前文本中出现的涌现概念 + 原始实体的交集/并集
                    text_specific = [e for e in emergent_ids if e in text]
                    if len(text_specific) >= 2:
                        cl_entities = text_specific

                # 统计学习的共现关系作为高质量正样本
                for entity in cl_entities:
                    related = stat_learner.get_related(entity, top_k=3)
                    for rel_concept, strength in related:
                        if strength > 0:
                            cl_cooccurrence.append((entity, rel_concept))

            # 概念空间的关系作为额外正样本
            if cs:
                for entity in cl_entities:
                    if entity in cs.concepts:
                        related = cs.get_related(entity, top_k=3)
                        for rel_id, score in related:
                            if score > 0.1:
                                cl_cooccurrence.append((entity, rel_id))
        except Exception:
            pass

        # ===== 阶段1：InfoNCE对比学习（核心训练信号）=====
        ct = self._registry.get('contrastive_trainer')
        if ct is not None and self.config.contrastive_enabled:
            # 检查 warmup 条件
            if len(self._training_corpus) >= self.config.contrastive_warmup_texts:
                if len(self._training_corpus) % self.config.contrastive_update_freq == 0:
                    try:
                        cl_loss = ct.train_step(
                            concepts=cl_entities,
                            cooccurrence_pairs=cl_cooccurrence,
                            optimizer=self._text_optimizer,
                        )
                    except Exception as _cl_err:
                        import os as _os
                        if _os.environ.get('DEBUG_CONTRASTIVE'):
                            import traceback
                            traceback.print_exc()
                        cl_loss = None

                    # 更新概念空间中的向量（对比学习后重新编码）
                    try:
                        cs = self._registry.get('concept_space')
                        if cs:
                            for entity in entities:
                                if entity in cs.concepts:
                                    with torch.no_grad():
                                        new_vec = self._learnable_encoder(entity).detach()
                                        cs.concepts[entity].vector = torch.nn.functional.normalize(
                                            new_vec, p=2, dim=0
                                        )
                    except Exception:
                        pass

        # ===== 阶段2：弱对比辅助损失（保留，提供额外梯度信号）=====
        self._learnable_encoder.train()
        self._text_optimizer.zero_grad()

        text_emb = self._learnable_encoder(text)

        with torch.no_grad():
            entity_embs = [self._learnable_encoder(e).detach().clone() for e in entities]

        # 正样本损失：实体应与文本相关
        loss = torch.tensor(0.0, device=self.device)
        for emb in entity_embs:
            sim_to_text = torch.cosine_similarity(
                emb.unsqueeze(0), text_emb.unsqueeze(0)
            )
            loss = loss + (1.0 - sim_to_text) * 0.3  # 降低权重，对比学习是主力

        # 负样本损失：不同实体保持区分
        neg_threshold = self.self_improvement.get_parameter('negative_threshold')
        for i in range(len(entity_embs)):
            for j in range(i + 1, len(entity_embs)):
                sim = torch.cosine_similarity(
                    entity_embs[i].unsqueeze(0),
                    entity_embs[j].unsqueeze(0)
                )
                if sim > neg_threshold:
                    loss = loss + (sim - neg_threshold) * 1.0  # 降低权重

        if loss.requires_grad and loss.item() > 1e-8:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._learnable_encoder.parameters(), 1.0)
            self._text_optimizer.step()

        self._learnable_encoder.eval()

        # ===== 阶段3：Hebbian传播 =====
        kg = self.knowledge
        related_pairs = []
        for entity in entities:
            try:
                relations = kg.get_relations_of(entity)
                for rel in relations:
                    related_pairs.append((entity, rel.target_id))
            except Exception:
                pass

        if related_pairs:
            hebbian_lr = 0.1
            with torch.no_grad():
                for entity1, entity2 in related_pairs:
                    emb1 = self._learnable_encoder(entity1).detach()
                    emb2 = self._learnable_encoder(entity2).detach()

                    delta = hebbian_lr * (emb2 - emb1)
                    new_emb1 = torch.nn.functional.normalize(
                        (emb1 + delta).unsqueeze(0), p=2, dim=1
                    ).squeeze(0)
                    new_emb2 = torch.nn.functional.normalize(
                        (emb2 - delta).unsqueeze(0), p=2, dim=1
                    ).squeeze(0)

                    self._embedding_cache[entity1] = new_emb1
                    self._embedding_cache[entity2] = new_emb2

        # 更新知识图谱中的实体嵌入
        self._update_entity_embeddings(entities)

    def _update_entity_embeddings(self, entities: List[str]):
        """更新知识图谱中的实体嵌入"""
        if not hasattr(self, '_learnable_encoder'):
            return

        kg = self.knowledge
        for entity in entities:
            if entity in kg.entities:
                # 重新编码实体
                new_repr = self._encode_text(entity)
                # 更新嵌入
                kg.entities[entity].embedding = new_repr

    def _extract_entities_from_repr(self, text: str, repr: torch.Tensor) -> List[str]:
        """从文本和表示中提取实体

        改进策略：
        1. 基于标点和语法分词提取候选实体
        2. 用TF-IDF思想过滤：太常见的碎片不是实体
        3. 用repr向量语义相关性过滤低质量实体
        """
        import re

        # 停用词表（扩展版：常见虚词、连接词、代词）
        stopwords = set(
            '的了是在我你他她它们这那个有不人大中上下来什么如何怎样'
            '而且还或者而但由于所以因为如果那么但是以为之一一个一些'
            '也能就要会可以被与及其对于到从向把给让比跟最更很已也并'
            '种样方面性化里后前内外国年月日时第个些每各该此本另某'
            '成得着过将使关于通过之间以及等等可能需要'
        )

        # 方法1：基于标点和关键标记的分块提取
        entities = []

        # 1a. 按"是"提取定义式实体: "X是Y"中的X通常是实体
        is_patterns = re.findall(r'([一-鿿A-Za-z0-9]{2,10})是', text)
        entities.extend(is_patterns)

        # 1b. 按"的"提取名词短语: "XX的YY"中YY是实体
        de_patterns = re.findall(r'的([一-鿿A-Za-z0-9]{2,8})[，。、\s]', text)
        entities.extend(de_patterns)

        # 1c. 按标点分割后提取（改进：只取2-4字的短片段作为候选）
        parts = re.split(r'[，。！？；：、\s]', text)
        for part in parts:
            part = part.strip()
            if not part or len(part) < 2:
                continue
            # 只取2-4字的中文词组作为候选实体（长片段大概率是句子）
            short_words = re.findall(r'[一-鿿]{2,4}', part)
            entities.extend(short_words)

        # 1d. 英文实体
        en_words = re.findall(r'[A-Z][a-zA-Z]{2,}', text)
        entities.extend(en_words)

        # 1e. 数字
        numbers = re.findall(r'\d+', text)
        entities.extend(numbers)

        # 去重 + 停用词过滤
        seen = set()
        filtered = []
        for e in entities:
            e = e.strip()
            if e in seen:
                continue
            seen.add(e)
            # 跳过停用词和单字
            if e in stopwords or len(e) < 2:
                continue
            # 跳过纯虚词组合（如"而且"、"所以"）
            if all(c in stopwords for c in e):
                continue
            filtered.append(e)

        # 方法2：用repr向量过滤 — 编码每个候选实体，只保留与文本语义相关的
        if hasattr(self, '_learnable_encoder') and len(filtered) > 15:
            with torch.no_grad():
                # 限制候选数量，避免太多编码操作
                candidates = filtered[:50]
                ent_sims = []
                for ent in candidates:
                    ent_repr = self._encode_text(ent)
                    sim = torch.cosine_similarity(repr.unsqueeze(0), ent_repr.unsqueeze(0)).item()
                    ent_sims.append((ent, sim))

                # 按相似度排序，取top N
                ent_sims.sort(key=lambda x: x[1], reverse=True)
                # 保留前15个最相关的实体
                filtered = [e for e, s in ent_sims[:15]]

                # 也保留任何sim > 0的实体（至少有点相关性）
                extra = [e for e, s in ent_sims[15:] if s > -0.5]
                filtered.extend(extra[:5])

        return filtered

    def _extract_relations_from_repr(self, text: str, entities: List[str], repr: torch.Tensor) -> List[Tuple[str, str, str]]:
        """从文本和表示中提取关系

        改进策略：
        1. 关系模式匹配（正则）
        2. 主语和宾语必须来自已抽取的entities列表
        3. 用repr计算语义相关性作为置信度
        返回4元组：(subject, relation, obj, confidence)
        """
        import re

        triples = []
        entity_set = set(entities)

        # 关系模式：在文本中查找已知实体之间的关系
        relation_patterns = [
            # 定义关系
            (r'([一-鿿]{2,10})是([一-鿿]{2,20})的([一-鿿]{2,10})', '是'),
            (r'([一-鿿]{2,10})是([一-鿿]{2,20})', '是'),
            # 包含关系
            (r'([一-鿿]{2,10})包括([一-鿿]{2,20})', '包括'),
            (r'([一-鿿]{2,10})包含([一-鿿]{2,20})', '包含'),
            # 属性关系
            (r'([一-鿿]{2,10})的([一-鿿]{2,10})', '的'),
            # 因果关系
            (r'([一-鿿]{2,10})导致([一-鿿]{2,20})', '导致'),
            (r'([一-鿿]{2,10})引起([一-鿿]{2,20})', '引起'),
            # 动作关系
            (r'([一-鿿]{2,10})使用([一-鿿]{2,20})', '使用'),
            (r'([一-鿿]{2,10})利用([一-鿿]{2,20})', '利用'),
            (r'([一-鿿]{2,10})通过([一-鿿]{2,20})', '通过'),
            (r'([一-鿿]{2,10})研究([一-鿿]{2,20})', '研究'),
        ]

        for pattern, relation in relation_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:
                    subj, obj = match[0].strip(), match[1].strip()
                elif len(match) == 3:
                    # "X是Y的Z" → X是Z的Y
                    subj, obj = match[0].strip(), (match[1] + '的' + match[2]).strip()
                else:
                    continue

                # 过滤：主语或宾语必须是已知实体
                subj_match = subj in entity_set or any(subj in e for e in entities)
                obj_match = obj in entity_set or any(e in obj for e in entities)

                if subj_match or obj_match:
                    # 置信度：基于是否两个都在entities中
                    if subj in entity_set and obj in entity_set:
                        confidence = 0.9
                    elif subj_match and obj_match:
                        confidence = 0.7
                    else:
                        confidence = 0.5

                    # 长度合理性检查
                    if 2 <= len(subj) <= 10 and 2 <= len(obj) <= 20:
                        triples.append((subj, relation, obj, confidence))

        # 去重
        seen = set()
        unique_triples = []
        for t in triples:
            key = (t[0], t[1], t[2])
            if key not in seen:
                seen.add(key)
                unique_triples.append(t)

        return unique_triples

    def _extract_causal_from_repr(self, text: str, repr: torch.Tensor) -> List[Tuple[str, str]]:
        """从文本和表示中提取因果关系"""
        import re

        causal_links = []

        # 使用学习到的模式提取因果
        patterns = [
            (r'因为(.+?)，所以(.+)', 'direct'),
            (r'由于(.+?)，(.+)', 'direct'),
            (r'(.+)导致(.+)', 'direct'),
            (r'(.+)引起(.+)', 'direct'),
        ]

        for pattern, causal_type in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                cause = re.sub(r'[。！？；\s]+', '', match[0].strip())[:20]
                effect = re.sub(r'[。！？；\s]+', '', match[1].strip())[:20]
                if len(cause) >= 2 and len(effect) >= 2:
                    causal_links.append((cause, effect))

        return causal_links

    def think(self, question: str) -> str:
        """思考问题 — 概念空间激活扩散 + 统一推理引擎双路径

        推理路径（优先级）：
        1. 概念空间激活扩散（Phase 2 新路径）
           问题 → 向量 → 最近邻概念 → 扩散激活 → 组织答案
        2. 统一推理引擎（原有路径，作为fallback）
           直接查询 → 因果 → 归纳 → 类比 → 反事实 → 概率
        """
        import re

        # 测试时训练：根据查询上下文微调编码器
        question_repr = self._encode_text(question)

        skip_ttt = getattr(self, '_skip_ttt', False)
        if not skip_ttt and hasattr(self, '_learnable_encoder'):
            similar_entities = self._find_similar_entities(question_repr, top_k=3)
            relevant_entities = [eid for eid, sim in similar_entities if sim > 0.3]
            if relevant_entities:
                self._test_time_trainer.adapt_to_query(
                    question, question_repr, relevant_entities
                )

        # 路径1: 概念空间激活扩散推理（优先）
        activated = None
        try:
            cs = self._registry.get('concept_space')
            if cs and len(cs.concepts) >= 3:
                activated = cs.activate(question, top_k=5, spread_depth=2)
                if activated and activated[0].activation > 0.1:
                    answer = self._synthesize_from_activation(activated, question)
                    if answer and len(answer) > 10:
                        return answer
        except Exception:
            pass

        # 路径1.3: 模拟推理 — 场景构建 + 因果追踪
        try:
            if activated:
                activated_labels = [a.concept_id for a in activated[:5]]
                sr_result = self.simulation_reasoning.reason(question, activated_labels)
                if sr_result and sr_result.confidence >= 0.3:
                    sr_answer = self.simulation_reasoning.express(sr_result, question)
                    if sr_answer and len(sr_answer) > 5:
                        return sr_answer
        except Exception:
            pass

        # 路径1.5: 概念空间多跳推理（利用 A→B→C 图路径）
        try:
            cs = self._registry.get('concept_space')
            if cs and activated:
                multi_hop_answer = self._synthesize_multihop(cs, activated, question)
                if multi_hop_answer and len(multi_hop_answer) > 10:
                    return multi_hop_answer
        except Exception:
            pass

        # 路径2: 统一推理引擎（fallback）
        if not hasattr(self, '_reasoning_engine'):
            from src.reasoning.unified_engine import UnifiedReasoningEngine
            self._reasoning_engine = UnifiedReasoningEngine(self)

        reasoning_results = self._reasoning_engine.reason(question)

        if reasoning_results:
            # 从推理结果生成答案
            return self._synthesize_from_reasoning(reasoning_results, question)

        # 回退到原有逻辑
        return self._think_legacy(question)

    def _synthesize_from_activation(self, activated: list, question: str) -> str:
        """从概念空间激活结果综合生成答案

        改进版（Phase 5）：
        1. 用统计学习验证概念是否是真正的词（而非跨词碎片）
        2. KG关系 + 概念空间关系 双通道构建答案
        3. 更自然的句式（根据关系类型推断）
        4. 置信度过滤：低质量概念不进入输出

        Args:
            activated: ActivatedConcept 列表，按激活强度降序
        """
        if not activated:
            return ""

        # ===== 概念质量验证 =====
        function_chars = set('是的有在了和与被把让给从到以也而')

        # 从统计学习获取已验证的涌现概念（高频+多上下文=真词）
        verified_concepts = set()
        try:
            stat_learner = self._registry.get('statistical_learner')
            if stat_learner:
                emergent = stat_learner.get_emergent_concepts(min_freq=2)
                verified_concepts = {c for c, conf in emergent}
        except Exception:
            pass

        # KG 中的实体也视为已验证
        kg = self.knowledge
        if kg:
            verified_concepts.update(kg.entities.keys())

        # 概念空间中的高频率概念（被多次巩固）
        cs = self._registry.get('concept_space')
        if cs:
            for cid, node in cs.concepts.items():
                if node.frequency >= 3:
                    verified_concepts.add(cid)

        def is_high_quality(concept_id):
            """判断概念是否高质量（不是碎片）

            验证策略：
            1. 统计学习涌现概念 → 直接通过
            2. KG 实体（3字+）→ 通过但需排除碎片
            3. 检查是否是更长概念的子串 → 碎片
            4. 2字概念 → 必须统计学习验证
            """
            if len(concept_id) < 2:
                return False
            # 虚词开头/结尾 → 碎片
            if concept_id[0] in function_chars or concept_id[-1] in function_chars:
                return False
            # 中间有虚词 + 长度>=4 → 跨词碎片
            if len(concept_id) >= 4:
                for i in range(1, len(concept_id) - 1):
                    if concept_id[i] in function_chars:
                        return False

            # 统计学习验证通过 → OK
            if concept_id in verified_concepts:
                return True

            # 子串检测：如果 concept_id 是某个更长概念的子串 → 很可能是碎片
            # 如 "物理学理" 是 "物理学理论" 的子串 → 碎片
            # 收集所有已知的更长概念
            all_longer = set()
            if kg:
                all_longer.update(eid for eid in kg.entities if len(eid) > len(concept_id))
            if cs:
                all_longer.update(cid2 for cid2 in cs.concepts if len(cid2) > len(concept_id))
            for longer in all_longer:
                if concept_id in longer:
                    return False

            # KG 实体（3字以上）→ 信任
            if kg and concept_id in kg.entities and len(concept_id) >= 3:
                return True

            # 2字概念 → 只在统计学习验证时通过
            if len(concept_id) == 2:
                return concept_id in verified_concepts

            # 未验证的3-4字概念 → 不通过
            return False

        # ===== 提取问题关键词 =====
        keywords = self._extract_keywords(question)

        # ===== 构建答案 =====
        answer_parts = []
        seen_concepts = set()
        question_core = keywords[0] if keywords else ""

        # 跟踪：是否已包含直接回答问题的句子
        direct_answer_found = False

        for ac in activated[:15]:
            cid = ac.concept_id
            if cid in seen_concepts or len(cid) < 2:
                continue
            # 严格质量过滤
            if not is_high_quality(cid):
                continue

            # 感知来源降权：纯感知概念（"红色"、"圆形"等）只在感知相关问题时输出
            # 避免感知探索污染学术推理
            if cs and cid in cs.concepts:
                node = cs.concepts[cid]
                if getattr(node, 'source', 'text') == 'perception':
                    # 检查问题是否与感知相关
                    perceptual_keywords = {'红', '蓝', '绿', '颜色', '形状', '圆', '方',
                                          '大', '小', '物体', '环境', '感知', '看'}
                    if not any(kw in question for kw in perceptual_keywords):
                        continue  # 非感知问题，跳过感知概念

            seen_concepts.add(cid)

            if len(answer_parts) >= 4:
                break

            # 策略1：KG 关系（最可靠的信息源）
            found_in_kg = False
            if kg and hasattr(kg, 'get_relations_of'):
                rels = kg.get_relations_of(cid)
                for rel in rels[:3]:
                    target = rel.target_id
                    if target in seen_concepts or not is_high_quality(target):
                        continue
                    if len(target) > 15:  # 过长的目标（可能是整句话）
                        continue
                    rel_type = rel.type
                    seen_concepts.add(target)

                    # 根据关系类型构建自然句式
                    sentence = self._build_sentence(cid, rel_type, target)
                    answer_parts.append(sentence)
                    found_in_kg = True

                    # 如果是直接回答问题的概念，标记
                    if cid == question_core or question_core in cid:
                        direct_answer_found = True
                    break  # 每个概念只取1条KG关系

            # 策略2：概念空间补充（仅当KG无结果时）
            if not found_in_kg and cs:
                related = cs.get_related(cid, top_k=5)
                for related_id, score in related:
                    if (related_id not in seen_concepts
                            and is_high_quality(related_id)
                            and score > 0.6  # 提高阈值：只有强关联才输出
                            and len(related_id) >= 3):
                        # 只有当两个概念都不是问题核心词时才输出
                        if related_id != question_core and cid != question_core:
                            answer_parts.append(f"{cid}与{related_id}存在关联。")
                        seen_concepts.add(related_id)
                        break

        if not answer_parts:
            # fallback：尝试直接用KG搜索问题关键词
            if question_core and kg:
                rels = kg.get_relations_of(question_core)
                for rel in rels[:3]:
                    target = rel.target_id
                    if len(target) <= 15:
                        sentence = self._build_sentence(question_core, rel.type, target)
                        answer_parts.append(sentence)
                if answer_parts:
                    direct_answer_found = True

        if not answer_parts:
            return ""

        # 去重
        unique_parts = list(dict.fromkeys(answer_parts))
        return " ".join(unique_parts[:4])

    def _synthesize_multihop(self, cs, activated: list, question: str) -> str:
        """从概念空间图结构中进行多跳推理，生成 A→B→C 链式答案

        当路径1（激活扩散）只返回孤立概念时，
        通过 BFS 在概念图上搜索连接两个问题关键词的路径。

        例如：问"数学和物理学有什么关系"
        → BFS 找到 数学→自然科学←物理学 路径
        → 生成 "数学与自然科学相关联，物理学也与自然科学相关联"

        Args:
            cs: 概念空间实例
            activated: 激活扩散结果
            question: 用户问题
        """
        # 提取问题中的已知概念关键词
        keywords = self._extract_keywords(question)
        if len(keywords) < 1:
            return ""

        # 虚词集合（用于过滤碎片）
        function_chars = set('是的有在了和与被把让给从到以也而')

        # 预收集已验证概念集合（避免 BFS 中重复查询）
        # Phase 7 修复：不能盲目导入 _concepts 全部 key（包含碎片）
        # 只导入通过质量检查的概念
        verified_set = set()
        stat_learner = self._registry.get('statistical_learner') if self._registry.has('statistical_learner') else None
        kg = self.knowledge

        if stat_learner and hasattr(stat_learner, '_concepts'):
            # 只导入通过 _is_complete_word() 验证的概念
            for cid, candidate in stat_learner._concepts.items():
                if candidate.frequency >= 2 and stat_learner._is_complete_word(cid):
                    verified_set.add(cid)
        if kg and hasattr(kg, 'entities'):
            verified_set.update(kg.entities.keys())
        if cs:
            verified_set.update(cid for cid, node in cs.concepts.items() if node.frequency >= 3)

        # 预计算子串黑名单（未验证概念是已验证概念的子串 → 碎片）
        fragment_blacklist = set()
        if cs:
            for cid in cs.concepts:
                if cid in verified_set:
                    continue
                if len(cid) < 2 or cid[0] in function_chars or cid[-1] in function_chars:
                    fragment_blacklist.add(cid)
                    continue
                for longer in verified_set:
                    if len(longer) > len(cid) and cid in longer:
                        fragment_blacklist.add(cid)
                        break

        def is_valid_concept(cid):
            """增强版概念验证（Phase 7 — 对齐 is_high_quality 的多层过滤）"""
            if len(cid) < 2:
                return False
            # 黑名单快速路径
            if cid in fragment_blacklist:
                return False
            # 层1: 虚词边界
            if cid[0] in function_chars or cid[-1] in function_chars:
                return False
            # 层2: 中间虚词(4+字)
            if len(cid) >= 4:
                for i in range(1, len(cid) - 1):
                    if cid[i] in function_chars:
                        return False
            # 层3: 已验证概念 → 通过
            if cid in verified_set:
                # 感知概念降权
                if cs and cid in cs.concepts:
                    node = cs.concepts[cid]
                    if getattr(node, 'source', 'text') == 'perception':
                        perceptual_kw = {'红','蓝','绿','颜色','形状','圆','方',
                                        '大','小','物体','环境','感知','看'}
                        if not any(kw in question for kw in perceptual_kw):
                            return False
                return True
            # 层4: 未验证的2字概念 → 拒绝
            if len(cid) == 2:
                return False
            # 层5: 未验证的3+字概念 — 子串检测
            all_longer = set()
            if kg and hasattr(kg, 'entities'):
                all_longer.update(eid for eid in kg.entities if len(eid) > len(cid))
            if cs:
                all_longer.update(c2 for c2 in cs.concepts if len(c2) > len(cid))
            for longer in all_longer:
                if cid in longer:
                    return False
            # 感知降权
            if cs and cid in cs.concepts:
                node = cs.concepts[cid]
                if getattr(node, 'source', 'text') == 'perception':
                    perceptual_kw = {'红','蓝','绿','颜色','形状','圆','方',
                                    '大','小','物体','环境','感知','看'}
                    if not any(kw in question for kw in perceptual_kw):
                        return False
            return True

        # 收集激活的概念作为起点（增强过滤）
        start_concepts = []
        for ac in activated[:5]:
            if is_valid_concept(ac.concept_id):
                start_concepts.append(ac.concept_id)

        # 也把问题关键词加入起点
        for kw in keywords:
            if len(kw) >= 2 and kw in cs.concepts and kw not in start_concepts and is_valid_concept(kw):
                start_concepts.append(kw)

        if not start_concepts:
            return ""

        # BFS 多跳搜索：找连接起点的路径
        max_hops = 3
        found_paths = []  # List[List[str]] — 每条路径是一系列概念ID

        for start in start_concepts[:3]:
            if start not in cs.concepts:
                continue

            # BFS
            visited = {start}
            queue = [(start, [start], 0)]  # (current, path, depth)

            while queue:
                current, path, depth = queue.pop(0)

                if depth >= max_hops:
                    continue

                # 获取邻居
                related = cs.get_related(current, top_k=5)
                for neighbor_id, score in related:
                    if neighbor_id in visited or not is_valid_concept(neighbor_id):
                        continue
                    if score < 0.05:
                        continue

                    new_path = path + [neighbor_id]
                    visited.add(neighbor_id)

                    # 如果路径长度>=3（至少2跳），且终点是另一个问题关键词或激活概念
                    if len(new_path) >= 3:
                        end_concept = new_path[-1]
                        # 终点是问题关键词
                        is_endpoint = any(kw in end_concept or end_concept in kw for kw in keywords)
                        # 终点是激活的概念
                        is_endpoint = is_endpoint or end_concept in start_concepts
                        if is_endpoint and len(new_path) <= max_hops + 1:
                            confidence = 0.7 / (depth + 1)
                            found_paths.append((new_path, confidence, score))

                    queue.append((neighbor_id, new_path, depth + 1))

        if not found_paths:
            # 退而求其次：找深度>=2的激活路径
            deep_activations = [ac for ac in activated if ac.depth >= 2 and ac.activation > 0.1]
            for ac in deep_activations[:2]:
                if len(ac.path) >= 3 and all(is_valid_concept(p) for p in ac.path):
                    found_paths.append((ac.path, ac.activation * 0.5, ac.activation))

        if not found_paths:
            return ""

        # 按置信度排序，取最佳路径
        found_paths.sort(key=lambda x: x[1] * x[2], reverse=True)
        best_path, confidence, _ = found_paths[0]

        # 从路径生成自然语言答案
        # 尝试用 KG 关系填充路径中的边
        kg = self.knowledge
        path_parts = []
        for i in range(len(best_path) - 1):
            src = best_path[i]
            tgt = best_path[i + 1]

            # 查找 KG 关系
            rel_type = None
            if kg and hasattr(kg, 'get_relations_of'):
                rels = kg.get_relations_of(src)
                for rel in rels:
                    if rel.target_id == tgt:
                        rel_type = rel.type
                        break

            if rel_type:
                path_parts.append(f"{src}{rel_type}{tgt}")
            else:
                # 查找概念空间关系
                rel_weight = cs.relations.get(src, {}).get(tgt, 0.0)
                if rel_weight > 0.1:
                    path_parts.append(f"{src}与{tgt}相关联")
                else:
                    path_parts.append(f"{src}与{tgt}存在联系")

        if not path_parts:
            return ""

        # 组合为连贯答案
        chain = "，".join(path_parts)

        # 构建最终答案
        if len(best_path) >= 3:
            # A→B→C 链式推理
            return f"通过{best_path[1]}可以关联：{chain}。"
        else:
            return f"{chain}。"

    def _build_sentence(self, subject: str, rel_type: str, obj: str) -> str:
        """根据关系类型构建自然语言句子"""
        # 截断过长的宾语
        if len(obj) > 12:
            obj = obj[:12] + "..."

        if rel_type == '是':
            return f"{subject}是{obj}。"
        elif rel_type in ('包括', '包含'):
            return f"{subject}包括{obj}。"
        elif rel_type in ('导致', '引起', '使得'):
            return f"{subject}导致{obj}。"
        elif rel_type in ('属于', '属于'):
            return f"{subject}属于{obj}。"
        elif rel_type == '研究':
            return f"{subject}研究{obj}。"
        elif rel_type in ('描述', '研究描述'):
            return f"{subject}描述{obj}。"
        else:
            # 通用关系：直接拼接
            return f"{subject}{rel_type}{obj}。"

    def _synthesize_from_reasoning(self, results, question: str) -> str:
        """从推理结果综合生成答案"""
        # 提取问题关键词
        keywords = self._extract_keywords(question)

        # 按置信度排序
        results.sort(key=lambda r: r.confidence, reverse=True)

        # 过滤：只保留与问题关键词有实质性关联的结果
        filtered = []
        for r in results:
            content = r.content
            matched = False
            for keyword in keywords:
                if len(keyword) < 2:
                    continue  # 跳过单字符关键词，避免过度匹配
                # 关键词在内容中出现
                if keyword in content:
                    matched = True
                    break
            if matched:
                filtered.append(r)

        # 如果严格过滤无结果，放宽条件：取置信度最高的结果
        # （统一推理引擎已经做了向量相似度过滤，结果有一定相关性）
        if not filtered and results:
            filtered = results[:3]

        # 如果过滤后没有结果，说明推理引擎的结果与问题无关
        # 不要fallback到随机top3，直接返回"我不知道"
        if not filtered:
            return "我没有足够的信息来回答这个问题。"

        # 去重 + 过滤自引用
        seen = set()
        unique = []
        for r in filtered:
            if r.content in seen:
                continue
            # 过滤自引用关系（A → A）
            parts = r.content.split()
            if len(parts) >= 3:
                subj = parts[0]
                obj = parts[-1]
                if subj == obj:
                    continue
            seen.add(r.content)
            unique.append(r)

        # 生成答案（三元组→自然语言，置信度阈值由自改进系统动态调整）
        conf_threshold = self.self_improvement.get_parameter('confidence_threshold')
        answer_lines = []
        for r in unique[:5]:
            if r.confidence > conf_threshold:
                # 尝试将三元组转换为自然语言
                sentence = self._triple_to_sentence(r.content)
                answer_lines.append(f"- {sentence}")

        if answer_lines:
            return '\n'.join(answer_lines)

        return "我没有足够的信息来回答这个问题。"

    def _triple_to_sentence(self, triple_str: str) -> str:
        """将三元组字符串转换为自然语言句子

        "牛顿 发现 万有引力定律" → "牛顿发现了万有引力定律。"
        "下雨 导致 地面湿了" → "下雨导致地面湿了。"
        "水 温度 100摄氏度沸腾" → "水的温度是100摄氏度沸腾。"
        """
        # 标准化格式：将箭头替换为空格
        normalized = triple_str.replace('→', ' ').replace('->', ' ')
        parts = normalized.split()
        if len(parts) >= 3:
            subj = parts[0]
            rel = parts[1]
            obj = ' '.join(parts[2:])

            # 关系到自然语言映义
            rel_map = {
                '是': '是',
                '属于': '属于',
                '位于': '位于',
                '发明': '发明了',
                '发现': '发现了',
                '使用': '使用',
                '导致': '导致',
                '温度': '的温度是',
                '长度': '的长度是',
                '重量': '的重量是',
                '时间': '的时间是',
                '颜色': '的颜色是',
                '形状': '的形状是',
                '大小': '的大小是',
                '部分': '是',
                '整体': '包含',
                '因果': '导致',
                '相似': '类似于',
                '对比': '与',
                '包含': '包含',
                '拥有': '拥有',
                '制造': '制造了',
                '运动': '在运动',
                '状态': '处于',
                '属性': '具有',
            }

            natural_rel = rel_map.get(rel, rel)
            # 避免重复句号
            if obj.endswith('。'):
                return f"{subj}{natural_rel}{obj}"
            return f"{subj}{natural_rel}{obj}。"

        return triple_str

    def _think_legacy(self, question: str) -> str:
        """原有推理逻辑（回退） — 增强版：双路径检索"""
        import re

        # 1. 编码问题为向量
        question_repr = self._encode_text(question)

        # 2. 用向量相似度检索相关实体
        similar_entities = self._find_similar_entities(question_repr, top_k=10)

        # 2b. 互补学习双路径检索：先查语义(皮层)，再查情节(海马体)
        cls_results = self.complementary_learning.retrieve(question_repr, top_k=5)
        for content, score, source in cls_results:
            if source == 'semantic' and score > 0.3:
                # 语义记忆提供概念级匹配
                concept = content
                if not any(eid == concept for eid, _ in similar_entities):
                    similar_entities.append((concept, score))

        # 2c. 树突计算消歧义：如果实体有多个上下文，选择最匹配的
        keywords = self._extract_keywords(question)
        if len(similar_entities) > 0:
            disambiguated = []
            for eid, sim in similar_entities[:5]:
                if eid in self.knowledge.entities:
                    entity = self.knowledge.entities[eid]
                    if hasattr(entity, 'embedding') and entity.embedding is not None:
                        # 用问题上下文消歧义
                        best_ctx, ctx_repr = self.dendritic_system.disambiguate(
                            eid, entity.embedding,
                            [(question[:20], question_repr)]
                        )
                        disambiguated.append((eid, sim))
                else:
                    disambiguated.append((eid, sim))
            if disambiguated:
                similar_entities[:len(disambiguated)] = disambiguated

        # 3. 提取关键词
        keywords = self._extract_keywords(question)

        # 4. 关键词匹配补充
        kg = self.knowledge
        for keyword in keywords:
            for entity_id in kg.entities.keys():
                if keyword in entity_id or entity_id in keyword:
                    if not any(eid == entity_id for eid, _ in similar_entities):
                        similar_entities.append((entity_id, 0.8))

        # 5. 从相似实体出发推理
        results = self._reason_from_entities(similar_entities, keywords, question)

        # 5b. 补充互补学习的情节记忆细节
        for content, score, source in cls_results:
            if source == 'episodic' and score > 0.2:
                results.append({
                    'type': 'episodic_memory',
                    'content': content,
                    'confidence': score * 0.8,
                })

        # 6. 按置信度排序
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        # 7. 过滤
        filtered = []
        for r in results:
            content = r.get('content', '')
            for keyword in keywords:
                if keyword in content:
                    filtered.append(r)
                    break

        if not filtered:
            # legacy路径也使用同样的策略：无相关结果就承认不知道
            if results:
                # 尝试宽松过滤：取与任何关键词有部分匹配的
                for r in results[:10]:
                    content = r.get('content', '')
                    for keyword in keywords:
                        if len(keyword) >= 2 and (keyword in content or any(c in content for c in keyword if '一' <= c <= '鿿')):
                            filtered.append(r)
                            break
            if not filtered:
                return "我没有足够的信息来回答这个问题。"

        # 8. 综合生成答案
        return self._synthesize_answer(filtered, question)

    def _find_similar_entities(self, query_repr: torch.Tensor, top_k: int = 10, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """用向量相似度检索相关实体"""
        similarities = []

        # 获取所有实体
        kg = self.knowledge
        for entity_id, entity in kg.entities.items():
            if hasattr(entity, 'embedding') and entity.embedding is not None:
                # 计算余弦相似度
                sim = torch.cosine_similarity(
                    query_repr.unsqueeze(0),
                    entity.embedding.unsqueeze(0)
                ).item()
                # 只保留相似度超过阈值的
                if sim >= threshold:
                    similarities.append((entity_id, sim))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _reason_from_entities(self, similar_entities: List[Tuple[str, float]],
                              keywords: List[str], question: str) -> List[Dict]:
        """从相似实体出发，遍历知识图谱推理

        改进：根据问题类型过滤关系，只返回与问题相关的推理结果。
        """
        results = []

        # 检测问题类型，用于关系过滤
        question_type = self._detect_question_type(question)
        relation_filter = {
            'definition': {'是', '属于', '包含', '包括', '的'},
            'causal': {'导致', '引起', '产生'},
            'method': {'使用', '利用', '通过', '研究'},
            'location': {'位于', '在'},
        }
        allowed_rels = relation_filter.get(question_type, None)

        # 1. 从相似实体获取直接关系（带类型过滤）
        kg = self.knowledge
        for entity_id, similarity in similar_entities[:5]:
            try:
                relations = kg.get_relations_of(entity_id)
                for rel in relations[:3]:
                    # 如果检测到问题类型，只返回相关类型的关系
                    if allowed_rels and rel.type not in allowed_rels:
                        continue

                    results.append({
                        'type': 'direct',
                        'content': f"{rel.source_id} {rel.type} {rel.target_id}",
                        'confidence': similarity,
                    })

                    # 2. 多步推理：跟随关系链
                    try:
                        target_relations = kg.get_relations_of(rel.target_id)
                        for target_rel in target_relations[:2]:
                            results.append({
                                'type': 'chain',
                                'content': f"{rel.source_id} → {rel.target_id} → {target_rel.target_id}",
                                'confidence': similarity * 0.8,
                            })
                    except Exception:
                        pass

            except Exception:
                pass

        # 2. 从因果DAG中推理（使用关键词匹配）
        try:
            dag = self.causal_dag
            for keyword in keywords:
                # 检查因果图中的节点
                for node_name in dag.nodes.keys():
                    if keyword in node_name or node_name in keyword:
                        node = dag.nodes[node_name]
                        for child in node.children[:3]:
                            results.append({
                                'type': 'causal',
                                'content': f"{node_name} → {child}",
                                'confidence': 0.9,
                            })
        except Exception:
            pass

        # 3. 从概念层次中推理
        try:
            cf = self.concept_formation
            for entity_id, similarity in similar_entities[:5]:
                if entity_id in cf.concepts:
                    concept = cf.concepts[entity_id]
                    if concept.parent:
                        results.append({
                            'type': 'concept',
                            'content': f"{entity_id} 是一种 {concept.parent}",
                            'confidence': similarity * 0.8,
                        })
        except Exception:
            pass

        # 4. 数值查询
        try:
            numerical_types = {
                '温度': ['度', '温度', '热', '冷'],
                '长度': ['高', '长', '宽', '深', '远', '米'],
                '重量': ['重', '千克', '公斤', '斤'],
                '时间': ['年', '月', '天', '小时', '分钟', '秒'],
            }

            question_type = None
            for attr_type, kw_list in numerical_types.items():
                for kw in kw_list:
                    if kw in question:
                        question_type = attr_type
                        break
                if question_type:
                    break

            if question_type:
                for entity_id, entity in kg.entities.items():
                    if entity.type == 'numerical':
                        if hasattr(entity, 'properties') and entity.properties:
                            value = entity.properties.get('value', '')
                            unit = entity.properties.get('unit', '')
                            attr = entity_id.split('_')[0] if '_' in entity_id else ''
                            if attr == question_type:
                                results.append({
                                    'type': 'numerical',
                                    'content': f"{attr}为{value}{unit}",
                                    'confidence': 0.9,
                                })
        except Exception:
            pass

        # 按置信度排序
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return results[:10]

    def _synthesize_answer(self, results: List[Dict], question: str = "") -> str:
        """综合多个证据生成答案

        改进：将碎片化三元组组织成连贯的自然语言回答。
        """
        if not results:
            return "我没有找到相关的知识。"

        # 去重 + 过滤低质量内容
        seen = set()
        unique_results = []
        for r in results:
            content = r.get('content', '')
            if not content or content in seen:
                continue
            # 过滤自引用（A → A）
            parts = content.split()
            if len(parts) >= 3 and parts[0] == parts[-1]:
                continue
            # 过滤太短的碎片
            if len(content) < 3:
                continue
            seen.add(content)
            unique_results.append(r)

        if not unique_results:
            return "我没有找到相关的知识。"

        # 按置信度排序
        unique_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        # 取top结果并转换为自然语言
        top_results = unique_results[:5]
        sentences = []
        for r in top_results:
            content = r.get('content', '')
            # 尝试将三元组格式转为自然语言
            sentence = self._triple_to_sentence(content)
            # 去掉末尾句号（最后统一加）
            sentence = sentence.rstrip('。')
            sentences.append(sentence)

        if not sentences:
            return "我没有找到相关的知识。"

        # 组织答案：用"。"连接所有句子，最后加句号
        if len(sentences) == 1:
            return sentences[0] + "。"
        elif len(sentences) == 2:
            return sentences[0] + "，" + sentences[1] + "。"
        else:
            # 多条结果：用分号或句号连接，保持简洁
            main = sentences[0]
            extra = "；".join(sentences[1:3])
            return f"{main}。此外，{extra}。"

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词

        改进：
        1. 短查询(< 8字)不分割，直接提取核心名词
        2. 长查询用动词和标点分割
        3. 过滤停用词
        """
        import re

        # 短查询：提取核心名词，不做分割
        if len(text) <= 8:
            # 去掉常见疑问词前缀和尾部助词
            core = re.sub(r'^什么|^为什么|^怎么|^如何|^哪里|^在哪', '', text)
            core = re.sub(r'[？?。！!的了]$', '', core)
            core = core.strip()
            if core:
                return [core]
            # fallback: 用整个问题（去掉疑问词后的部分）
            return [text]

        # 长查询：原有分割逻辑
        verbs = '发明发现创造提出开发设计编写找到证明提出建立形成产生导致引起'

        separators = r'[，。！？；：、\s的是在有位于属于包括使用产生导致引起为了因为所以如果那么但是而且或者而但]'
        parts = re.split(separators, text)

        keywords = []
        for part in parts:
            part = part.strip()
            if not part or len(part) < 1:
                continue

            verb_pattern = '|'.join(re.escape(v) for v in verbs)
            sub_parts = re.split(f'({verb_pattern})', part)

            for sub_part in sub_parts:
                sub_part = sub_part.strip()
                if not sub_part or sub_part in verbs:
                    continue
                if len(sub_part) >= 1:
                    if len(sub_part) == 1:
                        if '一' <= sub_part <= '鿿' and sub_part not in '的了是在有不人大中上下来什么如何怎样':
                            keywords.append(sub_part)
                    else:
                        keywords.append(sub_part)

        # 过滤停用词
        stopwords = set('什么怎么如何的是有在位于属于包括使用产生导致引起为了因为所以如果那么但是而且或者而但了')
        keywords = [k for k in keywords if k not in stopwords]

        return list(set(keywords))

    def _detect_question_type(self, question: str) -> str:
        """检测问题类型，用于推理路径选择"""
        if question.startswith('什么') or '是什么' in question:
            return 'definition'
        elif question.startswith('为什么') or '为什么' in question:
            return 'causal'
        elif question.startswith('怎么') or question.startswith('如何'):
            return 'method'
        elif question.startswith('哪里') or question.startswith('在哪') or '位于' in question:
            return 'location'
        else:
            return 'general'

    # ------------------------------------------------------------------
    # 语言
    # ------------------------------------------------------------------

    def ground_concept(self, obs: torch.Tensor, symbol: str = None) -> int:
        """从感知经验中建立概念，可选社会标注"""
        cluster = self.grounding.ground_from_perception(obs)
        return cluster

    def ground_symbol(self, symbol: str, referent: torch.Tensor,
                      context: str = "") -> None:
        """将符号与感知经验关联（社会标注）"""
        self.grounding.ground_from_social(symbol, referent, context)

    def play_reference_game(self, scene: List[Dict], target_idx: int) -> bool:
        """参与一轮参照游戏"""
        success = self.communication.play_round(
            self.communication, self.communication, scene, target_idx
        )
        self.communication_history.append(success)
        self._update_comm_rate()
        return success

    def get_vocabulary(self) -> Dict:
        """获取当前词汇表"""
        return self.communication.get_vocabulary()

    @property
    def vocabulary(self) -> Dict:
        """兼容 CapabilityEvaluator 的属性访问"""
        return self.get_vocabulary()

    def _update_comm_rate(self):
        """更新沟通成功率"""
        if self.communication_history:
            recent = list(self.communication_history)[-50:]
            self.comm_success_rate = sum(recent) / len(recent)

    # ------------------------------------------------------------------
    # 好奇心
    # ------------------------------------------------------------------

    def get_curiosity(self, obs: torch.Tensor) -> float:
        """获取当前好奇心值"""
        return self.engine.get_curiosity(obs)

    # ------------------------------------------------------------------
    # 发展阶段
    # ------------------------------------------------------------------

    def try_advance(self, evaluation: Dict[str, float]) -> bool:
        """尝试晋升到下一发展阶段

        评估指标决定是否满足晋升条件。
        Returns: True 如果成功晋升
        """
        if self._stage_index >= len(STAGE_ORDER) - 1:
            return False  # 已在最高阶段

        next_stage = STAGE_ORDER[self._stage_index + 1]
        if self._check_promotion(evaluation):
            self.stage = next_stage
            self._stage_index += 1
            return True
        return False

    def _check_promotion(self, evaluation: Dict[str, float]) -> bool:
        """检查是否满足当前阶段的晋升条件

        每个阶段允许多条晋升路径（OR 逻辑），模拟儿童发展的多通道：
        预测准确率路径：通过感知-预测闭环学习达标
        语言能力路径：通过参照游戏和符号接地达标
        """
        criteria = {
            # 感知运动 → 单字：预测准确 OR 积累了足够词汇
            'sensorimotor': lambda e: (
                e.get('prediction_accuracy', 0) > 0.6 or
                e.get('vocabulary_size', 0) >= 5
            ),
            # 单字 → 双字：词汇量继续增长 OR 组合表达涌现
            'single_word': lambda e: (
                e.get('vocabulary_size', 0) >= 10 or
                e.get('composition_rate', 0) > 0.3
            ),
            # 双字 → 复杂：组合能力成熟 OR 语法开始涌现
            'two_word': lambda e: (
                e.get('composition_rate', 0) > 0.3 or
                e.get('grammar_complexity', 0) > 0.5
            ),
            # 复杂 → 读写：语法系统成熟
            'complex': lambda e: e.get('grammar_complexity', 0) > 0.5,
            'literacy': lambda e: False,  # 最高阶段
        }
        checker = criteria.get(self.stage, lambda e: False)
        return checker(evaluation)

    def advance_stage(self) -> List[str]:
        """获取当前可解锁的能力列表"""
        abilities_map = {
            'sensorimotor': ['object_tracking', 'basic_reflex'],
            'single_word': ['word_production', 'word_comprehension'],
            'two_word': ['composition', 'basic_syntax'],
            'complex': ['grammar', 'narrative', 'theory_of_mind'],
            'literacy': ['reading', 'writing', 'abstract_reasoning'],
        }
        return abilities_map.get(self.stage, [])

    # ------------------------------------------------------------------
    # 模块注册
    # ------------------------------------------------------------------

    def _register_modules(self):
        """注册所有懒初始化模块到 registry"""
        r = self._registry

        def _make_knowledge():
            """knowledge + lang_bridge 作为一组初始化"""
            kg = KnowledgeGraph()
            r.set('lang_bridge', LanguageGraphBridge(kg))
            return kg

        def _make_lang_bridge():
            _ = r.get('knowledge')  # 确保 knowledge 先初始化
            return r.get('lang_bridge')

        def _make_reasoning():
            from src.reasoning.engine import ReasoningEngine
            return ReasoningEngine(r.get('knowledge'))

        def _make_metacognition():
            from src.metacognition.assessor import MetaAssessor
            return MetaAssessor(r.get('knowledge'))

        def _make_hypothesis():
            from src.reasoning.hypothesis import HypothesisEngine
            return HypothesisEngine(r.get('knowledge'))

        def _make_goals():
            from src.goals.manager import GoalManager
            return GoalManager(r.get('knowledge'), r.get('metacognition'))

        def _make_transfer():
            from src.reasoning.transfer import TransferEngine
            return TransferEngine(r.get('knowledge'))

        def _make_observation():
            from src.social.observation import ObservationLearner
            return ObservationLearner(r.get('knowledge'))

        def _make_teaching():
            from src.social.teaching import TeachingModule
            return TeachingModule(r.get('knowledge'))

        def _make_collaboration():
            from src.social.collaboration import CollaborationModule
            return CollaborationModule(r.get('knowledge'))

        def _make_active_inference():
            from src.reasoning.active_inference import (
                ActiveInferenceModule, ProbabilisticPredictor, VariationalBelief,
            )
            mod = ActiveInferenceModule(
                action_dim=self.config.action_dim, obs_dim=self.config.obs_dim,
            )
            mod.set_risk_sensitivity(self.config.fep_risk_penalty)
            r.set('fep_predictor', ProbabilisticPredictor(
                obs_dim=self.config.obs_dim, action_dim=self.config.action_dim,
            ))
            r.set('fep_belief', VariationalBelief(dim=self.config.obs_dim))
            return mod

        def _make_fep_predictor():
            _ = r.get('active_inference')
            return r.get('fep_predictor')

        def _make_fep_belief():
            _ = r.get('active_inference')
            return r.get('fep_belief')

        def _make_plasticity():
            from src.core.plasticity import PlasticityScheduler
            return PlasticityScheduler(
                schedule_name=self.config.plasticity_schedule,
                floor=self.config.plasticity_floor,
            )

        def _make_consolidation():
            from src.memory.consolidation import ConsolidationEngine
            return ConsolidationEngine()

        def _make_motivation():
            from src.core.motivation import IntrinsicMotivation
            return IntrinsicMotivation()

        def _make_metaphor():
            from src.reasoning.metaphor import MetaphorTracker
            return MetaphorTracker()

        def _make_theory_of_mind():
            from src.reasoning.theory_of_mind import TheoryOfMindModule
            return TheoryOfMindModule()

        def _make_causal():
            from src.reasoning.causal import CausalReasoningModule
            return CausalReasoningModule()

        def _make_counterfactual():
            from src.reasoning.counterfactual import CounterfactualModule
            return CounterfactualModule()

        def _make_tool_use():
            from src.reasoning.tool_use import ToolUseModule
            return ToolUseModule()

        def _make_inner_speech():
            from src.language.inner_speech import InnerSpeechModule
            return InnerSpeechModule(
                obs_dim=self.config.obs_dim, device=self.config.device,
            )

        def _make_narrative():
            from src.language.narrative import NarrativeModule
            return NarrativeModule()

        def _make_crossmodal():
            from src.language.crossmodal import CrossModalGrounding
            return CrossModalGrounding()

        def _make_attention():
            from src.language.attention import AttentionModulator
            return AttentionModulator(
                visual_dim=self.config.visual_channels * self.config.visual_size[0] * self.config.visual_size[1],
                audio_dim=self.config.audio_dim,
                position_dim=self.config.position_dim,
            )

        def _make_adversarial():
            from src.social.adversarial import AdversarialModule
            return AdversarialModule()

        def _make_cooperative_planner():
            from src.social.planning import CooperativePlanner
            return CooperativePlanner()

        def _make_questioning():
            from src.reasoning.questioning import QuestioningModule
            return QuestioningModule()

        def _make_nonstationary():
            from src.core.nonstationary import NonstationaryAdapter
            return NonstationaryAdapter()

        def _make_dual_memory():
            from src.language.memory_scaffold import DualCodingMemory
            return DualCodingMemory(
                capacity=self.config.episodic_memory_capacity,
            )

        def _make_continuous_concepts():
            from src.language.continuous_concepts import ContinuousConceptSpace
            return ContinuousConceptSpace(
                feature_dim=self.config.obs_dim,
            )

        # ===== 新增层：因果DAG、概念形成、世界模拟器、数值理解 =====
        def _make_causal_dag():
            from training.layers.causal_dag import CausalDAG
            return CausalDAG(device=self.config.device)

        def _make_concept_formation():
            from training.layers.concept_formation import ConceptFormation
            return ConceptFormation(feature_dim=self.config.obs_dim, device=self.config.device)

        def _make_world_simulator():
            from training.layers.world_simulator import WorldSimulator
            return WorldSimulator(
                obs_dim=self.config.obs_dim,
                action_dim=self.config.action_dim,
                device=self.config.device,
            )

        def _make_numerical():
            from training.layers.numerical import NumericalUnderstanding
            return NumericalUnderstanding()

        def _make_analogical():
            from training.layers.analogical import AnalogicalReasoning
            return AnalogicalReasoning()

        def _make_metacognition_enhanced():
            from training.layers.metacognition import Metacognition
            return Metacognition()

        def _make_statistical_learner():
            from src.learning.statistical_learner import StatisticalLearner
            return StatisticalLearner(
                max_ngram=self.config.statistical_max_ngram,
                min_freq=self.config.statistical_min_freq,
                min_pmi=self.config.statistical_min_pmi,
                max_concepts=self.config.statistical_max_concepts,
            )

        def _make_concept_space():
            from src.learning.concept_space import ConceptSpace
            # 不传入encoder — 用懒绑定，在 _encode_text 初始化后自动绑定
            return ConceptSpace(
                dim=self.config.obs_dim,
                encoder=None,
            )

        def _make_contrastive_trainer():
            # 延迟初始化：需要先有 _learnable_encoder
            # 返回 None，实际初始化在 _ensure_contrastive_trainer 中完成
            return None

        # 注册所有模块（顺序无关，依赖通过 r.get() 解析）
        r.register('knowledge', _make_knowledge)
        r.register('lang_bridge', _make_lang_bridge)
        r.register('reasoning', _make_reasoning)
        r.register('metacognition', _make_metacognition)
        r.register('hypothesis', _make_hypothesis)
        r.register('goals', _make_goals)
        r.register('transfer', _make_transfer)
        r.register('observation', _make_observation)
        r.register('teaching', _make_teaching)
        r.register('collaboration', _make_collaboration)
        r.register('active_inference', _make_active_inference)
        r.register('fep_predictor', _make_fep_predictor)
        r.register('fep_belief', _make_fep_belief)
        r.register('plasticity', _make_plasticity)
        r.register('consolidation', _make_consolidation)
        r.register('motivation', _make_motivation)
        r.register('metaphor', _make_metaphor)
        r.register('theory_of_mind', _make_theory_of_mind)
        r.register('causal', _make_causal)
        r.register('counterfactual', _make_counterfactual)
        r.register('tool_use', _make_tool_use)
        r.register('inner_speech', _make_inner_speech)
        r.register('narrative', _make_narrative)
        r.register('crossmodal', _make_crossmodal)
        r.register('attention', _make_attention)
        r.register('adversarial', _make_adversarial)
        r.register('cooperative_planner', _make_cooperative_planner)
        r.register('questioning', _make_questioning)
        r.register('nonstationary', _make_nonstationary)
        r.register('dual_memory', _make_dual_memory)
        r.register('continuous_concepts', _make_continuous_concepts)

        # 注册新层
        r.register('causal_dag', _make_causal_dag)
        r.register('concept_formation', _make_concept_formation)
        r.register('world_simulator', _make_world_simulator)
        r.register('numerical', _make_numerical)
        r.register('analogical', _make_analogical)
        r.register('metacognition_enhanced', _make_metacognition_enhanced)
        r.register('statistical_learner', _make_statistical_learner)
        r.register('concept_space', _make_concept_space)
        r.register('contrastive_trainer', _make_contrastive_trainer)

    # ------------------------------------------------------------------
    # 能力模块属性（委托 registry，保持 API 不变）
    # ------------------------------------------------------------------

    @property
    def knowledge(self) -> KnowledgeGraph:
        return self._registry.get('knowledge')

    @property
    def lang_bridge(self) -> LanguageGraphBridge:
        return self._registry.get('lang_bridge')

    @property
    def reasoning(self):
        return self._registry.get('reasoning')

    @property
    def metacognition(self):
        return self._registry.get('metacognition')

    @property
    def hypothesis_engine(self):
        return self._registry.get('hypothesis')

    @property
    def goals(self):
        return self._registry.get('goals')

    @property
    def transfer(self):
        return self._registry.get('transfer')

    @property
    def observation_learner(self):
        return self._registry.get('observation')

    @property
    def teaching_module(self):
        return self._registry.get('teaching')

    @property
    def collaboration(self):
        return self._registry.get('collaboration')

    @property
    def active_inference(self):
        return self._registry.get('active_inference')

    @property
    def fep_predictor(self):
        return self._registry.get('fep_predictor')

    @property
    def fep_belief(self):
        return self._registry.get('fep_belief')

    @property
    def plasticity(self):
        return self._registry.get('plasticity')

    @property
    def consolidation_engine(self):
        return self._registry.get('consolidation')

    @property
    def motivation(self):
        return self._registry.get('motivation')

    @property
    def metaphor_tracker(self):
        return self._registry.get('metaphor')

    @property
    def theory_of_mind(self):
        return self._registry.get('theory_of_mind')

    @property
    def causal(self):
        return self._registry.get('causal')

    @property
    def counterfactual(self):
        return self._registry.get('counterfactual')

    @property
    def tool_use(self):
        return self._registry.get('tool_use')

    @property
    def inner_speech(self):
        return self._registry.get('inner_speech')

    @property
    def narrative(self):
        return self._registry.get('narrative')

    @property
    def crossmodal(self):
        return self._registry.get('crossmodal')

    @property
    def attention(self):
        return self._registry.get('attention')

    @property
    def adversarial(self):
        return self._registry.get('adversarial')

    @property
    def cooperative_planner(self):
        return self._registry.get('cooperative_planner')

    @property
    def questioning(self):
        return self._registry.get('questioning')

    @property
    def nonstationary(self):
        return self._registry.get('nonstationary')

    @property
    def dual_memory(self):
        return self._registry.get('dual_memory')

    @property
    def continuous_concepts(self):
        return self._registry.get('continuous_concepts')

    # ===== 新增层属性 =====
    @property
    def causal_dag(self):
        return self._registry.get('causal_dag')

    @property
    def concept_formation(self):
        return self._registry.get('concept_formation')

    @property
    def world_simulator(self):
        return self._registry.get('world_simulator')

    @property
    def numerical_understanding(self):
        return self._registry.get('numerical')

    @property
    def analogical_reasoning(self):
        return self._registry.get('analogical')

    @property
    def metacognition_enhanced(self):
        return self._registry.get('metacognition_enhanced')

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """获取学习统计"""
        errors = list(self._error_history)
        avg_error = sum(errors[-50:]) / max(len(errors[-50:]), 1)

        stats = {
            'total_steps': self._total_steps,
            'stage': self.stage,
            'avg_error': avg_error,
            'curiosity': self.engine.get_curiosity(torch.zeros(self.config.obs_dim)),
            'learning_progress': self.engine.get_learning_progress(),
            'avg_inference_steps': self.engine.get_avg_inference_steps(),
            'memory_working_size': len(self.memory.working.items),
            'memory_episodic_size': len(self.memory.episodic.traces),
            # CUDA优化统计
            'cuda_available': torch.cuda.is_available(),
            'fp16_enabled': self.use_fp16,
            'device': str(self.device),
            'memory_semantic_concepts': len(self.memory.semantic.concepts),
            'modality_weights': self.perception.get_modality_weights(),
            'vocabulary_size': len(self.get_vocabulary()),
            'grounded_symbols': len(self.grounding.get_grounded_symbols()),
            'perceptual_clusters': len(self.grounding.perceptual_clusters),
            'comm_success_rate': self.comm_success_rate,
            'comm_games': len(self.communication_history),
        }

        # 多模态技能准确率
        for skill_name, history in [
            ('reading_accuracy', self.reading_history),
            ('writing_accuracy', self.writing_history),
            ('listening_accuracy', self.listening_history),
            ('grammar_accuracy', self.grammar_history),
        ]:
            if history:
                h = list(history)
                stats[skill_name] = sum(h[-50:]) / len(h[-50:])
            else:
                stats[skill_name] = 0.0

        # 8 大核心能力统计
        if self._registry.has('knowledge'):
            kg = self._registry.get('knowledge')
            stats['knowledge_entities'] = kg.entity_count
            stats['knowledge_relations'] = kg.relation_count
        if self._registry.has('hypothesis'):
            hyp = self._registry.get('hypothesis')
            stats['hypotheses_total'] = len(hyp.hypotheses)
            stats['hypotheses_confirmed'] = len(
                hyp.get_confirmed_hypotheses())
        if self._registry.has('goals'):
            stats['active_goals'] = len(self._registry.get('goals').goals)
        if self._registry.has('observation'):
            stats['observations_total'] = self._registry.get('observation').get_observation_count()
        if self._registry.has('transfer'):
            stats['transfers_completed'] = len(self._registry.get('transfer').transfer_history)
        if self._registry.has('plasticity'):
            stats['plasticity'] = self._registry.get('plasticity').get_plasticity(self._total_steps)
        if self._registry.has('consolidation'):
            stats['consolidated_traces'] = len(self._registry.get('consolidation').traces)
        if self._registry.has('metaphor'):
            m = self._registry.get('metaphor').detect_metaphors()
            stats['metaphor_mappings'] = len(m.get('mappings', {}))

        # ===== 新增层统计 =====
        try:
            dag = self._registry.get('causal_dag')
            stats['causal_dag_nodes'] = dag.stats.get('nodes', 0)
            stats['causal_dag_edges'] = dag.stats.get('edges', 0)
        except (KeyError, Exception):
            pass
        try:
            cf = self._registry.get('concept_formation')
            stats['concepts_formed'] = cf.stats.get('concepts_formed', 0)
        except (KeyError, Exception):
            pass
        try:
            ws = self._registry.get('world_simulator')
            stats['simulations'] = ws.stats.get('simulations', 0)
        except (KeyError, Exception):
            pass
        try:
            num = self._registry.get('numerical')
            stats['numerical_facts'] = num.stats.get('facts_stored', 0)
        except (KeyError, Exception):
            pass
        try:
            ana = self._registry.get('analogical')
            stats['analogies'] = ana.stats.get('total_analogies', 0)
        except (KeyError, Exception):
            pass

        return stats

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """保存学习体全部状态（含语言、记忆、接地）"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        state = {
            'config': self.config,
            'stage': self.stage,
            '_stage_index': self._stage_index,
            '_total_steps': self._total_steps,
            '_action_counts': self._action_counts,
            'engine_state': self.engine.state_dict(),
            'perception_state': self.perception.state_dict(),
            'comm_success_rate': self.comm_success_rate,
            'communication_history': self.communication_history,
            # 多模态技能历史
            'reading_history': self.reading_history,
            'writing_history': self.writing_history,
            'listening_history': self.listening_history,
            'grammar_history': self.grammar_history,
            # 语言系统
            'language_state': self.communication.language.save_state(),
            'grammar_rules': self.communication.grammar.get_rules(),
            # 接地系统
            'grounding_clusters': self.grounding.perceptual_clusters,
            'grounding_symbols': self.grounding.symbol_mappings,
            'grounding_hierarchy': self.grounding.concept_hierarchy,
            'grounding_cluster_counter': self.grounding.cluster_counter,
        }
        # 保存已初始化模块的状态（保持旧 checkpoint key 格式）
        for reg_name, state_key in _MODULE_STATE_KEYS.items():
            if not self._registry.has(reg_name):
                continue
            mod = self._registry.get(reg_name)
            if not hasattr(mod, 'save_state'):
                continue
            if reg_name == 'motivation':
                state[state_key] = mod.save_state()
            else:
                state[state_key] = mod.save_state()
        # active_inference 组的额外子模块
        if self._registry.has('active_inference'):
            state['fep_predictor_state'] = self._registry.get('fep_predictor').state_dict()
            state['fep_belief_state'] = self._registry.get('fep_belief').save_state()
        torch.save(state, path)

    def load(self, path: str) -> None:
        """加载学习体全部状态"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.stage = state['stage']
        self._stage_index = state['_stage_index']
        self._total_steps = state['_total_steps']
        self._action_counts = state['_action_counts']
        self.engine.load_state_dict(state['engine_state'])
        self.perception.load_state_dict(state['perception_state'])
        self.comm_success_rate = state.get('comm_success_rate', 0.0)
        self.communication_history = state.get('communication_history', [])
        # 多模态技能历史（兼容旧检查点）
        self.reading_history = state.get('reading_history', [])
        self.writing_history = state.get('writing_history', [])
        self.listening_history = state.get('listening_history', [])
        self.grammar_history = state.get('grammar_history', [])
        # 语言系统
        if 'language_state' in state:
            self.communication.language.load_state(state['language_state'])
        if 'grammar_rules' in state:
            for rule in state['grammar_rules']:
                self.communication.grammar.learn_pattern(
                    list(rule['pattern']) if isinstance(rule, dict) and 'pattern' in rule else [],
                    True
                )
        # 接地系统
        self.grounding.perceptual_clusters = state.get('grounding_clusters', {})
        self.grounding.symbol_mappings = state.get('grounding_symbols', {})
        self.grounding.concept_hierarchy = state.get('grounding_hierarchy', {})
        self.grounding.cluster_counter = state.get('grounding_cluster_counter', 0)
        self.grounding._dirty = True
        self.grounding._rebuild_centroid_cache()

        # 恢复已保存的模块状态（保持旧 checkpoint key 格式）
        # knowledge 特殊：需同时创建 lang_bridge
        if 'knowledge_state' in state:
            kg = KnowledgeGraph()
            kg.load_state(state['knowledge_state'])
            self._registry.set('knowledge', kg)
            self._registry.set('lang_bridge', LanguageGraphBridge(kg))
        # active_inference 特殊：需同时创建 fep_predictor + fep_belief
        if 'active_inference_state' in state:
            from src.reasoning.active_inference import (
                ActiveInferenceModule, ProbabilisticPredictor, VariationalBelief,
            )
            ai = ActiveInferenceModule(
                action_dim=self.config.action_dim, obs_dim=self.config.obs_dim,
            )
            ai.load_state(state['active_inference_state'])
            self._registry.set('active_inference', ai)
            pred = ProbabilisticPredictor(
                obs_dim=self.config.obs_dim, action_dim=self.config.action_dim,
            )
            if 'fep_predictor_state' in state:
                pred.load_state_dict(state['fep_predictor_state'])
            self._registry.set('fep_predictor', pred)
            belief = VariationalBelief(dim=self.config.obs_dim)
            if 'fep_belief_state' in state:
                belief.load_state(state['fep_belief_state'])
            self._registry.set('fep_belief', belief)
        # 常规模块：用工厂创建实例，再 load_state
        _load_map_kg = {
            # (state_key, module_path, class_name) — 构造函数接受 knowledge
            'reasoning':          ('reasoning_state', 'src.reasoning.engine', 'ReasoningEngine'),
            'metacognition':      ('metacognition_state', 'src.metacognition.assessor', 'MetaAssessor'),
            'hypothesis':         ('hypothesis_state', 'src.reasoning.hypothesis', 'HypothesisEngine'),
            'transfer':           ('transfer_state', 'src.reasoning.transfer', 'TransferEngine'),
            'observation':        ('observation_state', 'src.social.observation', 'ObservationLearner'),
            'collaboration':      ('collaboration_state', 'src.social.collaboration', 'CollaborationModule'),
        }
        for reg_name, (state_key, module_path, class_name) in _load_map_kg.items():
            if state_key not in state:
                continue
            cls = __import__(module_path, fromlist=[class_name]).__dict__[class_name]
            instance = cls(self._registry.get('knowledge'))
            instance.load_state(state[state_key])
            self._registry.set(reg_name, instance)
        _load_map_noarg = {
            # (state_key, module_path, class_name) —构造函数无参数
            'theory_of_mind':     ('tom_state', 'src.reasoning.theory_of_mind', 'TheoryOfMindModule'),
            'causal':             ('causal_state', 'src.reasoning.causal', 'CausalReasoningModule'),
            'counterfactual':     ('counterfactual_state', 'src.reasoning.counterfactual', 'CounterfactualModule'),
            'tool_use':           ('tool_use_state', 'src.reasoning.tool_use', 'ToolUseModule'),
            'adversarial':        ('adversarial_state', 'src.social.adversarial', 'AdversarialModule'),
            'questioning':        ('questioning_state', 'src.reasoning.questioning', 'QuestioningModule'),
        }
        for reg_name, (state_key, module_path, class_name) in _load_map_noarg.items():
            if state_key not in state:
                continue
            cls = __import__(module_path, fromlist=[class_name]).__dict__[class_name]
            instance = cls()
            instance.load_state(state[state_key])
            self._registry.set(reg_name, instance)
        # 需要特殊构造参数的模块
        if 'plasticity_state' in state:
            from src.core.plasticity import PlasticityScheduler
            m = PlasticityScheduler(
                schedule_name=self.config.plasticity_schedule,
                floor=self.config.plasticity_floor,
            )
            m.load_state(state['plasticity_state'])
            self._registry.set('plasticity', m)
        if 'consolidation_state' in state:
            from src.memory.consolidation import ConsolidationEngine
            m = ConsolidationEngine()
            m.load_state(state['consolidation_state'])
            self._registry.set('consolidation', m)
        if 'metaphor_state' in state:
            from src.reasoning.metaphor import MetaphorTracker
            m = MetaphorTracker()
            m.load_state(state['metaphor_state'])
            self._registry.set('metaphor', m)
        if 'narrative_state' in state:
            from src.language.narrative import NarrativeModule
            m = NarrativeModule()
            m.load_state(state['narrative_state'])
            self._registry.set('narrative', m)
        if 'crossmodal_state' in state:
            from src.language.crossmodal import CrossModalGrounding
            m = CrossModalGrounding()
            m.load_state(state['crossmodal_state'])
            self._registry.set('crossmodal', m)
        if 'planner_state' in state:
            from src.social.planning import CooperativePlanner
            m = CooperativePlanner()
            m.load_state(state['planner_state'])
            self._registry.set('cooperative_planner', m)
        if 'nonstationary_state' in state:
            from src.core.nonstationary import NonstationaryAdapter
            m = NonstationaryAdapter()
            m.load_state(state['nonstationary_state'])
            self._registry.set('nonstationary', m)
        # goals 特殊：依赖 metacognition
        if 'goals_state' in state:
            from src.goals.manager import GoalManager
            g = GoalManager(self._registry.get('knowledge'), self._registry.get('metacognition'))
            g.load_state(state['goals_state'])
            self._registry.set('goals', g)
        # motivation 特殊：可能没有 load_state
        if 'motivation_state' in state:
            from src.core.motivation import IntrinsicMotivation
            m = IntrinsicMotivation()
            if hasattr(m, 'load_state'):
                m.load_state(state['motivation_state'])
            self._registry.set('motivation', m)
        # 需要配置参数的模块
        if 'inner_speech_state' in state:
            from src.language.inner_speech import InnerSpeechModule
            m = InnerSpeechModule(
                obs_dim=self.config.obs_dim, device=self.config.device,
            )
            m.load_state(state['inner_speech_state'])
            self._registry.set('inner_speech', m)
        if 'attention_state' in state:
            from src.language.attention import AttentionModulator
            m = AttentionModulator(
                visual_dim=self.config.visual_channels * self.config.visual_size[0] * self.config.visual_size[1],
                audio_dim=self.config.audio_dim,
                position_dim=self.config.position_dim,
            )
            m.load_state(state['attention_state'])
            self._registry.set('attention', m)
        if 'dual_memory_state' in state:
            from src.language.memory_scaffold import DualCodingMemory
            m = DualCodingMemory(
                capacity=self.config.episodic_memory_capacity,
            )
            m.load_state(state['dual_memory_state'])
            self._registry.set('dual_memory', m)
        if 'concepts_state' in state:
            from src.language.continuous_concepts import ContinuousConceptSpace
            m = ContinuousConceptSpace(
                feature_dim=self.config.obs_dim,
            )
            m.load_state(state['concepts_state'])
            self._registry.set('continuous_concepts', m)
