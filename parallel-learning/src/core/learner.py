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

        # ===== 新增：概念形成学习 =====
        try:
            concept_formation = self._registry.get('concept_formation')
            entity_name = f"entity_{self._total_steps % 100}"
            concept_formation.add_instance(entity_name, obs)
        except (KeyError, Exception):
            pass

        # ===== 新增：因果DAG学习 =====
        try:
            causal_dag = self._registry.get('causal_dag')
            obs_state = f"state_{hash(str(obs.shape)) % 1000}"
            next_state = f"state_{hash(str(next_obs.shape)) % 1000}"
            causal_dag.observe({obs_state: 1.0, next_state: 1.0})
        except (KeyError, Exception):
            pass

        self._error_history.append(error)
        self._total_steps += 1
        return error

    # ------------------------------------------------------------------
    # 行动选择
    # ------------------------------------------------------------------

    def choose_action(self, obs: torch.Tensor) -> int:
        """好奇心驱动探索 — 批量预测 + softmax"""
        curiosity = self.engine.get_curiosity(obs)
        n_actions = self.config.action_dim

        explore_prob = min(0.3, curiosity * 0.5)
        if torch.rand(1).item() < explore_prob:
            counts = self._action_counts + 1e-6
            probs = (1.0 / counts).softmax(0)
            action = torch.multinomial(probs, 1).item()
        else:
            # 批量预测：1 次矩阵运算替代 n 次前向传播
            action_vecs = torch.eye(n_actions, device=obs.device)
            obs_batch = obs.unsqueeze(0).expand(n_actions, -1)
            preds = self.engine.predict_batch(obs_batch, action_vecs)
            action = int(preds.var(dim=1).argmax().item())

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
        """巩固记忆（模拟睡眠）"""
        return self.memory.consolidate_all()

    # ------------------------------------------------------------------
    # 文本学习（桥接 training/layers）
    # ------------------------------------------------------------------

    def learn_from_text(self, text: str, source: str = "text") -> Dict:
        """从文本中学习 — 桥接 training/layers 模块

        将文本知识注入到：
        1. 知识图谱
        2. 因果DAG
        3. 概念形成
        4. 数值理解
        5. 类比推理
        """
        import re
        result = {
            'entities': [],
            'triples': [],
            'causal_links': [],
            'concepts': [],
            'numerical_facts': [],
        }

        # 1. 提取实体（简单分词）
        entities = re.findall(r'[一-鿿]{2,6}', text)
        entities = [e for e in entities if len(e) >= 2]
        result['entities'] = entities

        # 2. 提取三元组
        triple_patterns = [
            (r'(.{2,10}?)是(.{2,30})', '是'),
            (r'(.{2,10}?)属于(.{2,20})', '属于'),
            (r'(.{2,10}?)位于(.{2,20})', '位于'),
            (r'(.{2,10}?)发明了?(.{2,20})', '发明'),
            (r'(.{2,10}?)发现了?(.{2,20})', '发现'),
        ]

        for pattern, relation in triple_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                subject = match[0].strip()
                obj = match[1].strip()
                if 2 <= len(subject) <= 15 and 2 <= len(obj) <= 30:
                    result['triples'].append((subject, relation, obj))

                    # 注入知识图谱
                    try:
                        kg = self.knowledge  # 使用属性访问，会自动初始化
                        from src.knowledge.entity import Entity
                        from src.knowledge.relation import Relation

                        # 添加实体
                        subj_entity = Entity(id=subject, type='concept', source=source)
                        obj_entity = Entity(id=obj, type='concept', source=source)
                        kg.add_entity(subj_entity)
                        kg.add_entity(obj_entity)

                        # 添加关系
                        rel = Relation(
                            source_id=subject,
                            target_id=obj,
                            type=relation,
                            confidence=0.8,
                        )
                        kg.add_relation(rel)
                    except Exception as e:
                        pass

        # 3. 提取因果关系
        causal_patterns = [
            (r'因为(.+?)，所以(.+)', 'direct'),
            (r'由于(.+?)，(.+)', 'direct'),
            (r'(.+)导致(.+)', 'direct'),
            (r'(.+)引起(.+)', 'direct'),
        ]

        for pattern, causal_type in causal_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # 去除标点符号
                cause = re.sub(r'[。！？；\s]+', '', match[0].strip())[:20]
                effect = re.sub(r'[。！？；\s]+', '', match[1].strip())[:20]
                if len(cause) >= 2 and len(effect) >= 2:
                    result['causal_links'].append((cause, effect))

                    # 注入因果DAG
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

                        cause_entity = Entity(id=cause, type='event', source=source)
                        effect_entity = Entity(id=effect, type='event', source=source)
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

        # 4. 形成概念
        for entity in entities[:10]:  # 限制数量
            result['concepts'].append(entity)

            # 注入概念形成
            try:
                cf = self.concept_formation
                features = torch.randn(self.config.obs_dim).to(self.device)
                cf.add_instance(entity, features)
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

        # 6. 存入记忆
        self.memory.store_experience(
            torch.zeros(self.config.obs_dim).to(self.device),
            torch.tensor([0]),
            torch.zeros(self.config.obs_dim).to(self.device),
            reward=0.0,
            error=0.0,
        )

        return result

    def think(self, question: str) -> str:
        """思考问题 — 从知识图谱中检索答案"""
        import re

        # 提取关键词：滑动窗口匹配知识图谱中的实体
        keywords = []
        kg = self.knowledge if self._registry.has('knowledge') else None

        # 从知识图谱中获取所有实体
        known_entities = set()
        if kg:
            known_entities = set(kg.entities.keys())

        # 滑动窗口匹配
        for length in range(6, 1, -1):  # 从长到短
            for i in range(len(question) - length + 1):
                word = question[i:i+length]
                if word in known_entities:
                    keywords.append(word)

        # 如果没有匹配到，使用简单分词
        if not keywords:
            # 用标点和常见词分割
            separators = r'[，。！？；：、\s的了是在有位于属于包括使用产生导致引起为了因为所以如果那么但是而且或者而但]'
            parts = re.split(separators, question)
            for part in parts:
                part = part.strip()
                if part and len(part) >= 2:
                    keywords.append(part)

        keywords = list(set(keywords))

        if not keywords:
            return "我不太理解你的问题。"

        # 从知识图谱中搜索
        results = []
        try:
            kg = self.knowledge
            for keyword in keywords:
                try:
                    # 搜索相关实体
                    entity = kg.get_entity(keyword)
                    if entity:
                        # 获取相关关系
                        relations = kg.get_relations_of(keyword)
                        for rel in relations[:3]:
                            results.append({
                                'type': 'knowledge',
                                'content': f"{rel.source_id} {rel.type} {rel.target_id}",
                            })
                except Exception:
                    pass
        except Exception:
            pass

        # 从因果DAG中搜索
        try:
            dag = self.causal_dag
            for keyword in keywords:
                if keyword in dag.nodes:
                    node = dag.nodes[keyword]
                    for child in node.children[:3]:
                        results.append({
                            'type': 'causal',
                            'content': f"{keyword} → {child}",
                        })
        except Exception:
            pass

        # 从概念形成中搜索
        try:
            cf = self.concept_formation
            for keyword in keywords:
                if keyword in cf.concepts:
                    concept = cf.concepts[keyword]
                    if concept.parent:
                        results.append({
                            'type': 'concept',
                            'content': f"{keyword} 是一种 {concept.parent}",
                        })
        except Exception:
            pass

        # 从知识图谱中搜索数值关系
        try:
            kg = self.knowledge

            # 检测数值问题类型
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

            # 搜索数值实体
            for entity_id, entity in kg.entities.items():
                if entity.type == 'numerical':
                    # 获取数值属性
                    if hasattr(entity, 'properties') and entity.properties:
                        value = entity.properties.get('value', '')
                        unit = entity.properties.get('unit', '')
                        attr = entity_id.split('_')[0] if '_' in entity_id else ''

                        # 如果是匹配的类型，添加结果
                        if question_type and attr == question_type:
                            results.append({
                                'type': 'numerical',
                                'content': f"{attr}为{value}{unit}",
                            })
                        # 或者包含关键词
                        elif any(kw in entity_id for kw in keywords):
                            results.append({
                                'type': 'numerical',
                                'content': f"{attr}为{value}{unit}",
                            })
        except Exception:
            pass

        if not results:
            return f"我没有关于{', '.join(keywords[:3])}的知识。"

        # 生成答案
        parts = []
        seen = set()
        for r in results[:5]:
            content = r.get('content', str(r))
            if content not in seen:
                seen.add(content)
                parts.append(f"- {content}")

        return '\n'.join(parts)

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
