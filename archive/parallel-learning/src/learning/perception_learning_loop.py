"""感知-预测学习循环 — 真正的学习核心

认知科学基础：
    人类学习的本质不是"记住信息"，而是"预测世界并从错误中学习"。

    Friston的自由能原理（Active Inference, 2010）：
    - 大脑是一个预测机器，不断生成对下一个感知状态的预测
    - 预测误差 = 实际感知 - 预测感知
    - 学习 = 最小化预测误差（更新内部模型）
    - 好奇心 = 选择能最大化信息增益的行动

    与当前系统（正则提取→存储）的根本区别：
    - 正则提取：文本 → 模式匹配 → 存储（不理解任何东西）
    - 感知循环：感知 → 预测 → 误差 → 更新模型（从失败中学习）

    关键洞察：概念从预测误差中涌现。
    - 如果系统总是预测错"红色"的颜色通道 → "红色"概念涌现
    - 如果系统总是预测错"圆形"的形状特征 → "圆形"概念涌现
    - 这不是统计频率的涌现（StatisticalLearner），而是功能需求的涌现

核心设计：
    1. 感知编码：多模态感知 → 统一128维向量
    2. 预测：基于当前状态预测下一个状态
    3. 误差分析：哪些维度的误差最大 → 缺乏概念
    4. 概念检测：高误差聚类 → 新概念候选
    5. 好奇心：误差 × 可学习性 × 信息增益 → 行动选择

参考文献：
    Friston, K. (2010). The free-energy principle. Nature Reviews Neuroscience.
    Oudeyer, P.-Y. et al. (2016). Intrinsically motivated learning in real and
    virtual robots. Neurons, Behaviors, Data analysis, and Theory.
"""

import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque


@dataclass
class ConceptCandidate:
    """从预测误差中检测到的概念候选"""
    label: str                        # 候选标签
    error_dimensions: List[int]       # 高误差维度索引
    error_magnitude: float            # 误差大小
    perceptual_features: Dict         # 关联的感知特征
    occurrence_count: int = 1         # 出现次数
    last_seen_step: int = 0           # 最后出现步骤
    confirmed: bool = False           # 是否已确认为概念


class PerceptionLearningLoop:
    """感知-预测-学习闭环

    使用方式：
        loop = PerceptionLearningLoop(learner)

        # 环境探索模式
        for step in range(100):
            obs = world.observe()
            action = loop.choose_curious_action(obs)
            next_obs, reward, done = world.step(action)
            result = loop.perceive_and_learn(obs, action, next_obs)
            # result['new_concepts'] 包含从误差中涌现的新概念

        # 文本学习模式（文本作为虚拟感知）
        text_obs = learner._text_to_virtual_observation("数学是研究数量和结构的学科")
        result = loop.perceive_and_learn(text_obs)
    """

    def __init__(self, learner, concept_threshold: float = 0.6):
        """
        Args:
            learner: Learner 实例
            concept_threshold: 概念确认阈值（出现次数 × 误差大小）
        """
        self.learner = learner
        self.concept_threshold = concept_threshold

        # 预测误差历史（用于概念检测和好奇心计算）
        self.error_history: deque = deque(maxlen=200)

        # 概念候选缓冲区（从误差中检测但尚未确认）
        self.concept_candidates: Dict[str, ConceptCandidate] = {}

        # 维度级误差统计（检测哪些维度系统性高误差）
        self.dim_error_accumulator = torch.zeros(
            learner.config.obs_dim if hasattr(learner.config, 'obs_dim') else 128
        )
        self.dim_error_count = 0

        # 好奇心状态
        self.curiosity_state = {
            'current_curiosity': 1.0,
            'peak_errors': deque(maxlen=50),
            'learning_progress': 0.0,
        }

        # 统计
        self._stats = {
            'total_steps': 0,
            'concepts_detected': 0,
            'concepts_confirmed': 0,
            'avg_error': 0.0,
        }

    def perceive_and_learn(self, obs, action=None, next_obs=None) -> Dict:
        """一次完整的感知-预测-学习循环

        流程：
        1. 多模态感知编码
        2. 预测（基于当前状态 + 动作）
        3. 观察实际结果
        4. 计算预测误差
        5. 误差分析（哪些维度出错？）
        6. 学习更新（PredictiveCodingEngine）
        7. 概念检测（高误差区域 → 新概念）
        8. 好奇心更新

        Args:
            obs: 当前观察（张量或字典）
            action: 动作（可选）
            next_obs: 下一步观察（可选，如果None则自预测）

        Returns:
            {
                'error': float,             # 预测误差
                'error_structure': Dict,     # 误差结构分析
                'new_concepts': List[Dict],  # 新涌现的概念
                'curiosity': float,          # 当前好奇心
                'confirmed_concepts': int,   # 已确认概念数
            }
        """
        self._stats['total_steps'] += 1

        # 1. 多模态感知编码
        features = self._extract_features(obs)

        # 2 & 3. 预测与实际
        device = self.learner.device
        encoded = features['encoded'].to(device)

        if next_obs is not None:
            next_features = self._extract_features(next_obs)
            actual = next_features['encoded'].to(device)
            predicted = self.learner.engine.predict(encoded, action)
        else:
            # 自预测模式：预测自身的编码
            predicted = self._self_predict(encoded)
            actual = encoded

        # 4. 计算预测误差
        error_tensor = (predicted - actual).pow(2)
        total_error = error_tensor.mean().item()

        # 5. 误差结构分析
        error_structure = self._analyze_error(error_tensor, predicted, actual, features)

        # 6. 学习更新
        if next_obs is not None and action is not None:
            self.learner.engine.learn(predicted, actual, encoded, action)

        # 7. 从误差中检测概念
        new_concepts = self._detect_concepts_from_error(error_structure, features)

        # 8. 好奇心更新
        curiosity = self._update_curiosity(total_error, error_structure)

        # 更新统计
        self.error_history.append(total_error)
        self._stats['avg_error'] = (
            sum(self.error_history) / len(self.error_history)
            if self.error_history else 0.0
        )

        return {
            'error': total_error,
            'error_structure': error_structure,
            'new_concepts': new_concepts,
            'curiosity': curiosity,
            'confirmed_concepts': self._stats['concepts_confirmed'],
        }

    def choose_curious_action(self, obs, action_dim: int = 8) -> torch.Tensor:
        """好奇心驱动的行动选择

        不是随机探索，而是选择"预期学习最多"的方向。
        基于Friston的预期自由能：
        G = -info_gain - pragmatic_value + risk

        简化实现：选择预测误差最高的方向（那里有最多的东西可学）
        """
        features = self._extract_features(obs)
        encoded = features['encoded'].to(self.learner.device)

        # 采样几个候选动作，评估每个的"信息价值"
        n_candidates = 5
        best_action = None
        best_score = -float('inf')

        for _ in range(n_candidates):
            candidate = torch.randn(action_dim) * 0.5

            # 预测该动作后的状态
            predicted = self.learner.engine.predict(encoded, candidate)

            # 评估信息价值：预测不确定性高的地方信息价值高
            # 用当前维度误差分布作为不确定性估计
            if self.dim_error_count > 10:
                normalized_errors = self.dim_error_accumulator / max(self.dim_error_count, 1)
                uncertainty = torch.sigmoid(normalized_errors)
                # 预测变化越大 → 越可能学到新东西
                change_score = (predicted - encoded).abs().mean().item()
                info_score = uncertainty.mean().item() * change_score
            else:
                # 早期：随机探索
                info_score = torch.randn(1).item() * 0.5 + 0.5

            # 加入少量随机性（ε-贪心）
            score = info_score + torch.randn(1).item() * 0.1

            if score > best_score:
                best_score = score
                best_action = candidate

        return best_action if best_action is not None else torch.randn(action_dim) * 0.5

    def _extract_features(self, obs) -> Dict:
        """从观察中提取多模态特征

        支持：
        - torch.Tensor → 直接编码
        - Dict → 提取各模态后融合
        """
        if isinstance(obs, torch.Tensor):
            encoded = obs.detach().flatten()
            # 确保维度匹配
            target_dim = self.learner.config.obs_dim if hasattr(self.learner.config, 'obs_dim') else 128
            if encoded.shape[0] < target_dim:
                encoded = F.pad(encoded, (0, target_dim - encoded.shape[0]))
            elif encoded.shape[0] > target_dim:
                encoded = encoded[:target_dim]
            return {
                'encoded': encoded,
                'raw': obs,
                'type': 'tensor',
            }
        elif isinstance(obs, dict):
            # 字典格式：提取并编码各模态
            try:
                encoded = self.learner.perception.encode(obs)
                return {
                    'encoded': encoded.detach().flatten(),
                    'raw': obs,
                    'type': 'dict_multimodal',
                }
            except Exception:
                # fallback：拼接所有张量
                tensors = []
                for v in obs.values():
                    if isinstance(v, torch.Tensor):
                        tensors.append(v.flatten())
                if tensors:
                    encoded = torch.cat(tensors)
                    target_dim = self.learner.config.obs_dim
                    if encoded.shape[0] < target_dim:
                        encoded = F.pad(encoded, (0, target_dim - encoded.shape[0]))
                    elif encoded.shape[0] > target_dim:
                        encoded = encoded[:target_dim]
                    return {'encoded': encoded, 'raw': obs, 'type': 'dict_flat'}
                return {'encoded': torch.zeros(128), 'raw': obs, 'type': 'dict_empty'}
        else:
            return {'encoded': torch.zeros(128), 'raw': obs, 'type': 'unknown'}

    def _self_predict(self, encoded: torch.Tensor) -> torch.Tensor:
        """自预测：在没有动作的情况下预测当前编码

        这实现了"预测自身感知"——系统试图理解当前的感知输入。
        预测误差表示"系统不理解当前输入的哪些部分"。
        """
        # 使用引擎的无动作预测
        return self.learner.engine.predict(encoded, action=None)

    def _analyze_error(self, error_tensor: torch.Tensor,
                       predicted: torch.Tensor,
                       actual: torch.Tensor,
                       features: Dict) -> Dict:
        """分析预测误差的结构

        不只看总误差，而是分析哪些维度、哪些区域出错最多。
        这为概念检测提供基础。
        """
        # 累积维度级误差（用于长期概念检测）
        self.dim_error_accumulator += error_tensor.detach().cpu()
        self.dim_error_count += 1

        # 找到高误差维度
        error_flat = error_tensor.flatten()
        if error_flat.numel() > 0:
            threshold = error_flat.mean() + error_flat.std()
            high_error_dims = (error_flat > threshold).nonzero(as_tuple=True)[0].tolist()
        else:
            high_error_dims = []

        # 计算误差集中度（误差是否集中在少数维度）
        if error_flat.numel() > 0 and error_flat.sum() > 0:
            concentration = (error_flat.max() / error_flat.sum()).item()
        else:
            concentration = 0.0

        return {
            'total': error_tensor.mean().item(),
            'max': error_tensor.max().item(),
            'high_error_dims': high_error_dims,
            'concentration': concentration,  # 1.0=集中在一个维度，0.0=均匀分布
            'error_tensor': error_tensor.detach().cpu(),
        }

    def _detect_concepts_from_error(self, error_structure: Dict,
                                     features: Dict) -> List[Dict]:
        """从预测误差中检测新概念

        核心算法：
        1. 检查维度级误差累积 — 系统性高误差的维度表示缺乏概念
        2. 将高误差维度聚类 — 相关的高误差维度组成一个"概念候选"
        3. 关联感知特征 — 将候选与实际感知特征绑定
        4. 用核心知识先验评分 — 符合先验的候选更容易被确认
        5. 多次出现后确认 — 避免噪声导致的假概念

        Returns:
            新确认的概念列表
        """
        new_concepts = []

        if self.dim_error_count < 5:
            # 至少5步数据才开始检测
            return new_concepts

        # 1. 找到系统性高误差维度（累积误差显著高于平均）
        avg_dim_error = self.dim_error_accumulator / max(self.dim_error_count, 1)
        if avg_dim_error.max() == 0:
            return new_concepts

        # 标准化：找高于平均值2个标准差的维度
        mean_err = avg_dim_error.mean().item()
        std_err = avg_dim_error.std().item()
        if std_err < 1e-6:
            return new_concepts

        high_dims = (avg_dim_error > mean_err + 1.5 * std_err).nonzero(as_tuple=True)[0].tolist()

        if not high_dims:
            return new_concepts

        # 2. 尝试与感知特征关联
        raw = features.get('raw', {})

        # 如果raw是环境观察字典，提取场景特征
        perceptual_features = self._extract_perceptual_features(raw, high_dims)

        if not perceptual_features:
            # 无法关联感知特征 → 跳过
            return new_concepts

        # 3. 为每个感知特征创建/更新概念候选
        for label, feat_data in perceptual_features.items():
            error_mag = feat_data.get('error_contribution', 0.0)

            if label in self.concept_candidates:
                # 已有候选 → 更新
                candidate = self.concept_candidates[label]
                candidate.occurrence_count += 1
                candidate.last_seen_step = self._stats['total_steps']
                candidate.error_magnitude = max(candidate.error_magnitude, error_mag)
                candidate.perceptual_features.update(feat_data)
            else:
                # 新候选
                self.concept_candidates[label] = ConceptCandidate(
                    label=label,
                    error_dimensions=high_dims,
                    error_magnitude=error_mag,
                    perceptual_features=feat_data,
                    last_seen_step=self._stats['total_steps'],
                )

            # 4. 用核心知识先验评分
            candidate = self.concept_candidates[label]
            prior_score = self.learner.core_knowledge.score_concept_candidate({
                'label': label,
                'frequency': candidate.occurrence_count,
                'features': candidate.perceptual_features,
            })

            # 5. 确认条件：出现>=2次 AND 先验评分>阈值
            if (candidate.occurrence_count >= 2
                    and not candidate.confirmed
                    and prior_score > 0.3):

                candidate.confirmed = True
                self._stats['concepts_confirmed'] += 1

                concept_data = {
                    'label': label,
                    'perceptual_features': candidate.perceptual_features,
                    'error_dimensions': candidate.error_dimensions,
                    'error_magnitude': candidate.error_magnitude,
                    'occurrence_count': candidate.occurrence_count,
                    'prior_score': prior_score,
                    'source': 'perception_loop',
                }
                new_concepts.append(concept_data)

        self._stats['concepts_detected'] = len(self.concept_candidates)
        return new_concepts

    def _extract_perceptual_features(self, raw_obs, high_error_dims: List[int]) -> Dict:
        """从原始观察中提取可命名的感知特征

        将高误差维度映射回有意义的感知概念。
        例如：颜色通道的高误差 → "红色"、"蓝色"等
        """
        features = {}

        if isinstance(raw_obs, dict):
            # 字典格式的环境观察
            scene_features = raw_obs.get('scene_features', [])
            for f in scene_features:
                # 颜色特征
                color = f.get('color')
                if color:
                    cn_color = self._color_to_cn(color)
                    if cn_color:
                        features[cn_color] = {
                            'type': 'color',
                            'raw_value': color,
                            'error_contribution': 0.3,
                        }

                # 形状特征
                shape = f.get('shape')
                if shape:
                    cn_shape = self._shape_to_cn(shape)
                    if cn_shape:
                        features[cn_shape] = {
                            'type': 'shape',
                            'raw_value': shape,
                            'error_contribution': 0.3,
                        }

                # 材质特征
                material = f.get('material')
                if material:
                    from src.learning.perception_explorer import MATERIAL_MAP
                    cn_material = MATERIAL_MAP.get(material)
                    if cn_material:
                        features[cn_material] = {
                            'type': 'material',
                            'raw_value': material,
                            'error_contribution': 0.2,
                        }

                # 大小特征
                size = f.get('size')
                if size:
                    from src.learning.perception_explorer import SIZE_MAP
                    cn_size = SIZE_MAP.get(size)
                    if cn_size:
                        features[cn_size] = {
                            'type': 'size',
                            'raw_value': size,
                            'error_contribution': 0.2,
                        }

        # 如果有高误差维度信息，增强特征描述
        if high_error_dims and features:
            for label, feat in features.items():
                feat['related_dims'] = high_error_dims[:5]  # 记录关联维度

        return features

    def _color_to_cn(self, color: str) -> Optional[str]:
        """英文颜色名→中文"""
        color_map = {
            'red': '红色', 'blue': '蓝色', 'green': '绿色',
            'yellow': '黄色', 'orange': '橙色', 'purple': '紫色',
            'pink': '粉色', 'brown': '棕色', 'black': '黑色',
            'white': '白色', 'gray': '灰色', 'cyan': '青色',
        }
        return color_map.get(color)

    def _shape_to_cn(self, shape: str) -> Optional[str]:
        """英文形状名→中文"""
        shape_map = {
            'circle': '圆形', 'square': '方形', 'triangle': '三角形',
            'rectangle': '长方形', 'pentagon': '五边形',
            'hexagon': '六边形', 'star': '星形', 'diamond': '菱形',
        }
        return shape_map.get(shape)

    def _update_curiosity(self, error: float, error_structure: Dict) -> float:
        """基于Friston主动推理的好奇心更新

        好奇心 = 预测误差 × 可学习性 × 信息增益

        - 高误差 + 高可学习性 = 有趣！应该探索
        - 高误差 + 低可学习性 = 太难了，放弃
        - 低误差 = 无聊，需要新挑战
        """
        # 可学习性：误差在"最佳区域"（不太简单也不太难）
        optimal_error = 0.5
        learnability = math.exp(-((error - optimal_error) ** 2) / (2 * 0.3 ** 2))

        # 信息增益：高误差集中度 → 高信息增益（集中=特定概念缺失）
        info_gain = error_structure.get('concentration', 0.0)

        # 综合好奇心
        curiosity = error * learnability * (1.0 + info_gain)

        # 平滑更新
        alpha = 0.3
        self.curiosity_state['current_curiosity'] = (
            alpha * curiosity +
            (1 - alpha) * self.curiosity_state['current_curiosity']
        )

        # 学习进度（误差是否在下降）
        if len(self.error_history) >= 20:
            recent = list(self.error_history)[-10:]
            older = list(self.error_history)[-20:-10]
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older)
            if older_avg > 0:
                self.curiosity_state['learning_progress'] = (
                    (older_avg - recent_avg) / older_avg
                )
            else:
                self.curiosity_state['learning_progress'] = 0.0

        return self.curiosity_state['current_curiosity']

    def get_stats(self) -> Dict:
        """获取感知循环统计"""
        return {
            **self._stats,
            'avg_error': self._stats['avg_error'],
            'curiosity': self.curiosity_state['current_curiosity'],
            'learning_progress': self.curiosity_state['learning_progress'],
            'candidates_pending': sum(
                1 for c in self.concept_candidates.values() if not c.confirmed
            ),
        }
