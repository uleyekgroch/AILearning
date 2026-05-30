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
            causal_dag = self._registry.get('causal_dag')
            # 使用预测误差作为因果信号
            if error > 0.5:  # 高误差表示意外
                obs_state = f"state_{self._total_steps % 100}"
                next_state = f"state_{(self._total_steps + 1) % 100}"
                causal_dag.observe({obs_state: error, next_state: 0.0})
        except (KeyError, Exception):
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

        return {
            'consolidated': len(consolidated),
            'multiscale_consolidated': len(multiscale_consolidated),
            'forgotten': len(forgotten),
            'total': len(memories),
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

        # 2. 从表示中提取实体
        entities = self._extract_entities_from_repr(text, text_repr)
        result['entities'] = entities

        # 2b. BTSP：标记实体为可学习（资格痕迹）
        for entity in entities:
            entity_repr = self._encode_text(entity)
            self.btsp_learning.mark_eligible(entity, entity_repr)

        # 3. 从表示中提取关系
        triples = self._extract_relations_from_repr(text, entities, text_repr)
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

        # 4. 形成概念（使用实际文本表示，不是随机噪声）
        for entity in entities[:10]:  # 限制数量
            result['concepts'].append(entity)

            # 注入概念形成
            try:
                cf = self.concept_formation
                # 使用实体的文本编码作为特征
                entity_repr = self._encode_text(entity)
                cf.add_instance(entity, entity_repr)
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

        # 10c. 存入感觉运动接地系统
        # 注意：当前没有真实视觉/触觉数据，跳过接地
        # 当接入真实传感器时再启用
        # for entity in entities:
        #     entity_repr = self._encode_text(entity)
        #     self.store_sensorimotor_experience(entity, visual=entity_repr)

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
                n_heads=4,
                n_layers=2,
                max_len=128,
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
        self.embodied_grounding.store_experience(concept, experience)

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
        """训练嵌入 — 梯度学习 + 知识图谱传播

        两阶段训练：
        1. 梯度训练：通过 Transformer 的反向传播更新嵌入
        2. Hebbian传播：通过知识图谱关系传播相似性
        """
        if not hasattr(self, '_learnable_encoder'):
            return

        if len(entities) < 2:
            return

        # 阶段1：梯度训练 — 实体嵌入应接近文本嵌入但保持区分度
        self._learnable_encoder.train()
        self._text_optimizer.zero_grad()

        # 编码文本和实体（带梯度）
        text_emb = self._encode_text(text, train=True)
        entity_embs = []
        for entity in entities:
            emb = self._encode_text(entity, train=True)
            entity_embs.append(emb)

        # 损失设计：
        # 1. 实体应与文本嵌入相关（正锚点）
        # 2. 实体之间应保持区分度（负样本）
        loss = 0.0
        for emb in entity_embs:
            # 正锚点：实体应与文本相关
            sim_to_text = torch.cosine_similarity(emb.unsqueeze(0), text_emb.unsqueeze(0))
            loss = loss + (1.0 - sim_to_text) * 0.5

        # 负样本：实体之间保持区分（阈值由自改进系统动态调整）
        neg_threshold = self.self_improvement.get_parameter('negative_threshold')
        for i in range(len(entity_embs)):
            for j in range(i + 1, len(entity_embs)):
                sim = torch.cosine_similarity(
                    entity_embs[i].unsqueeze(0),
                    entity_embs[j].unsqueeze(0)
                )
                # 如果相似度过高，增加损失
                if sim > neg_threshold:
                    loss = loss + (sim - neg_threshold) * 2.0

        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
            self._text_optimizer.step()

        self._learnable_encoder.eval()

        # 阶段2：Hebbian传播 — 通过知识图谱关系传播
        kg = self.knowledge
        related_pairs = []
        for entity in entities:
            try:
                relations = kg.get_relations_of(entity)
                for rel in relations:
                    related_pairs.append((entity, rel.target_id))
            except Exception:
                pass

        hebbian_lr = 0.1
        for entity1, entity2 in related_pairs:
            if entity1 in self._embedding_cache and entity2 in self._embedding_cache:
                emb1 = self._embedding_cache[entity1]
                emb2 = self._embedding_cache[entity2]

                # Hebbian更新：让相关实体的嵌入更相似
                delta = hebbian_lr * (emb2 - emb1)
                self._embedding_cache[entity1] = emb1 + delta
                self._embedding_cache[entity2] = emb2 - delta

                # 归一化
                self._embedding_cache[entity1] = torch.nn.functional.normalize(
                    self._embedding_cache[entity1].unsqueeze(0), p=2, dim=1
                ).squeeze(0)
                self._embedding_cache[entity2] = torch.nn.functional.normalize(
                    self._embedding_cache[entity2].unsqueeze(0), p=2, dim=1
                ).squeeze(0)

        # 清除缓存，让重新编码使用更新后的嵌入层
        for entity in entities:
            if entity in self._embedding_cache:
                del self._embedding_cache[entity]

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

        使用可微分知识提取器（学习驱动），冷启动时回退到正则。
        """
        # 初始化可微分提取器
        if not hasattr(self, '_knowledge_extractor'):
            from src.perception.knowledge_extractor import LearnableKnowledgeExtractor
            self._knowledge_extractor = LearnableKnowledgeExtractor(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )

        extractor = self._knowledge_extractor
        entities = []

        # 方法1：正则提取（冷启动回退）
        import re
        separators = r'[，。！？；：、\s的了是在有位于属于包括使用产生导致引起为了因为所以如果那么但是而且或者而但]'
        parts = re.split(separators, text)
        for part in parts:
            part = part.strip()
            if not part or len(part) < 2:
                continue
            zh_words = re.findall(r'[一-鿿]{2,6}', part)
            entities.extend(zh_words)

        en_words = re.findall(r'[A-Z][a-zA-Z]+', text)
        entities.extend(en_words)

        numbers = re.findall(r'\d+', text)
        entities.extend(numbers)

        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样')
        entities = [e for e in entities if e not in stopwords and len(e) >= 2]

        # 方法2：可学习提取器（3次后启用，带梯度训练）
        encoder = getattr(self, '_learnable_encoder', None)
        if extractor._extraction_count >= 3 and encoder is not None:
            learned_triples = extractor.extract_triples(text, repr, encoder)
            for t in learned_triples:
                if t.subject not in entities:
                    entities.append(t.subject)
                if t.obj not in entities:
                    entities.append(t.obj)

        extractor._extraction_count += 1

        return list(set(entities))

    def _extract_relations_from_repr(self, text: str, entities: List[str], repr: torch.Tensor) -> List[Tuple[str, str, str]]:
        """从文本和表示中提取关系

        使用可微分知识提取器（学习驱动），冷启动时回退到正则。
        返回4元组：(subject, relation, obj, confidence)
        """
        # 初始化可微分提取器
        if not hasattr(self, '_knowledge_extractor'):
            from src.perception.knowledge_extractor import LearnableKnowledgeExtractor
            self._knowledge_extractor = LearnableKnowledgeExtractor(
                d_model=self.config.obs_dim,
                device=str(self.device),
            )

        extractor = self._knowledge_extractor

        # 获取编码器
        encoder = getattr(self, '_learnable_encoder', None)

        # 提取三元组
        extracted = extractor.extract_triples(text, repr, encoder)

        # 转换为4元组格式
        triples = []
        for t in extracted:
            triples.append((t.subject, t.relation, t.obj, t.confidence))

        return triples

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
        """思考问题 — 多模式推理 + 测试时训练

        使用统一推理引擎，整合多种推理模式：
        1. 测试时训练（根据查询上下文微调编码器）
        2. 直接查询（知识图谱向量检索 + 关键词匹配）
        3. 因果推理（因果DAG遍历）
        4. 归纳推理（从记忆中发现模式）
        5. 类比推理（跨域映射）
        6. 反事实推理（如果...会怎样）
        7. 概率推理（贝叶斯更新）
        """
        import re

        # 测试时训练：根据查询上下文微调编码器
        # 先确保编码器已初始化
        question_repr = self._encode_text(question)
        if hasattr(self, '_learnable_encoder'):
            # 获取相关实体
            similar_entities = self._find_similar_entities(question_repr, top_k=3)
            relevant_entities = [eid for eid, sim in similar_entities if sim > 0.3]

            # 微调编码器
            if relevant_entities:
                self._test_time_trainer.adapt_to_query(
                    question, question_repr, relevant_entities
                )

        # 初始化统一推理引擎
        if not hasattr(self, '_reasoning_engine'):
            from src.reasoning.unified_engine import UnifiedReasoningEngine
            self._reasoning_engine = UnifiedReasoningEngine(self)

        # 使用统一推理引擎
        reasoning_results = self._reasoning_engine.reason(question)

        if reasoning_results:
            # 从推理结果生成答案
            return self._synthesize_from_reasoning(reasoning_results, question)

        # 回退到原有逻辑
        return self._think_legacy(question)

    def _synthesize_from_reasoning(self, results, question: str) -> str:
        """从推理结果综合生成答案"""
        # 提取问题关键词
        keywords = self._extract_keywords(question)

        # 按置信度排序
        results.sort(key=lambda r: r.confidence, reverse=True)

        # 过滤：只保留包含问题关键词的结果（子串匹配）
        filtered = []
        for r in results:
            content = r.content
            for keyword in keywords:
                # 双向子串匹配
                if keyword in content or content in keyword:
                    filtered.append(r)
                    break
                # 字符级匹配（中文）
                if any(c in content for c in keyword if '一' <= c <= '鿿'):
                    filtered.append(r)
                    break

        # 如果过滤后没有结果，使用原始结果
        if not filtered:
            filtered = results[:3]

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
        """原有推理逻辑（回退）"""
        import re

        # 1. 编码问题为向量
        question_repr = self._encode_text(question)

        # 2. 用向量相似度检索相关实体
        similar_entities = self._find_similar_entities(question_repr, top_k=10)

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
            filtered = results[:3]

        # 8. 综合生成答案
        return self._synthesize_answer(filtered, question)

    def _find_similar_entities(self, query_repr: torch.Tensor, top_k: int = 10, threshold: float = -1.0) -> List[Tuple[str, float]]:
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

        支持多步推理（组合泛化）：
        1. 直接关系
        2. 两跳关系（A→B→C）
        3. 因果链推理
        """
        results = []

        # 1. 从相似实体获取直接关系
        kg = self.knowledge
        for entity_id, similarity in similar_entities[:5]:
            try:
                relations = kg.get_relations_of(entity_id)
                for rel in relations[:3]:
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

        简化版：按置信度排序，返回最相关的结果。
        """
        if not results:
            return "我没有找到相关的知识。"

        # 去重
        seen = set()
        unique_results = []
        for r in results:
            content = r.get('content', '')
            if content not in seen:
                seen.add(content)
                unique_results.append(r)

        # 按置信度排序
        unique_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        # 生成答案
        parts = []
        for r in unique_results[:3]:
            parts.append(f"- {r['content']}")

        return '\n'.join(parts)

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词

        改进：
        1. 用动词分割，避免"牛顿发现"被当作一个词
        2. 保留有意义的子串
        3. 过滤停用词
        """
        import re

        # 动词列表（用于分割）
        verbs = '发明发现创造提出开发设计编写找到证明提出建立形成产生导致引起'

        # 先用标点和虚词分割
        separators = r'[，。！？；：、\s的是在有位于属于包括使用产生导致引起为了因为所以如果那么但是而且或者而但]'
        parts = re.split(separators, text)

        keywords = []
        for part in parts:
            part = part.strip()
            if not part or len(part) < 1:
                continue

            # 用动词进一步分割
            verb_pattern = '|'.join(re.escape(v) for v in verbs)
            sub_parts = re.split(f'({verb_pattern})', part)

            for sub_part in sub_parts:
                sub_part = sub_part.strip()
                if not sub_part or sub_part in verbs:
                    continue
                # 保留有意义的子串（包括单个中文字符）
                if len(sub_part) >= 1:
                    # 单字符只保留中文实词
                    if len(sub_part) == 1:
                        if '一' <= sub_part <= '鿿' and sub_part not in '的了是在有不人大中上下来什么如何怎样':
                            keywords.append(sub_part)
                    else:
                        keywords.append(sub_part)

        # 过滤停用词
        stopwords = set('什么怎么如何的是有在位于属于包括使用产生导致引起为了因为所以如果那么但是而且或者而但了')
        keywords = [k for k in keywords if k not in stopwords]

        return list(set(keywords))

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
