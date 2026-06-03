"""认知机制 — 基于2025-2026最新人类学习研究

5个核心机制：
1. 认知预测路由 — 区分低级感觉误差和高级认知误差
2. GHL全局调制Hebbian — 神经调质信号调制局部学习
3. 学习进展好奇心 — 追踪学习速度，探索甜蜜区
4. 先类别后语言 — 感知分类先于语言涌现
5. 元学习组合规则 — 学习如何组合，而非记住什么组合

论文来源：
- "Rethinking Predictive Processing" Annual Review of Neuroscience 2026
- "Hebbian Learning with Global Direction" arXiv 2026
- "Curiosity" Oudeyer HAL-Inria 2026
- "Two-month-old babies making sense of the world" Nature Neuroscience 2026
- "Human-like systematic generalization through MLC" Nature 2023
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field


# ============================================================
# 1. 认知预测路由
# ============================================================

class CognitivePredictiveRouter:
    """认知预测路由 — 区分低级和高级预测误差

    基于 "Rethinking Predictive Processing" (Annual Review 2026)：
    - 低级感觉误差：简单残差，局部处理
    - 高级认知误差：需要前额叶式全局评估
    - 误差信号路由：不同层级的误差用不同方式处理
    """

    def __init__(self, d_model: int = 128):
        self.d_model = d_model

        # 低级误差历史（感觉层）
        self.low_level_errors: deque = deque(maxlen=100)

        # 高级误差历史（认知层）
        self.high_level_errors: deque = deque(maxlen=100)

        # 误差路由权重
        self.routing_weights = {
            'low_level': 0.3,   # 低级误差权重
            'high_level': 0.7,  # 高级误差权重（认知更重要）
        }

    def compute_low_level_error(self, predicted: torch.Tensor,
                                actual: torch.Tensor) -> float:
        """计算低级感觉预测误差（简单残差）"""
        error = torch.nn.functional.mse_loss(predicted, actual).item()
        self.low_level_errors.append(error)
        return error

    def compute_high_level_error(self, predicted_concept: torch.Tensor,
                                 actual_concept: torch.Tensor,
                                 context: torch.Tensor) -> float:
        """计算高级认知预测误差（考虑上下文）"""
        # 高级误差不仅看预测是否准确，还看是否与上下文一致
        concept_error = torch.nn.functional.mse_loss(
            predicted_concept, actual_concept
        ).item()

        # 上下文一致性
        context_similarity = torch.cosine_similarity(
            predicted_concept.unsqueeze(0), context.unsqueeze(0)
        ).item()

        # 高级误差 = 概念误差 + (1 - 上下文一致性)
        high_error = concept_error + (1.0 - context_similarity) * 0.5
        self.high_level_errors.append(high_error)
        return high_error

    def route_error(self, low_error: float, high_error: float) -> Dict:
        """路由误差信号 — 决定哪个层级需要更多学习"""
        # 计算各层级的学习需求
        low_need = low_error * self.routing_weights['low_level']
        high_need = high_error * self.routing_weights['high_level']

        # 动态调整路由权重
        if len(self.low_level_errors) > 10:
            low_trend = self._compute_trend(self.low_level_errors)
            high_trend = self._compute_trend(self.high_level_errors)

            # 误差增加的层级需要更多关注
            if low_trend > high_trend:
                self.routing_weights['low_level'] = min(0.5, self.routing_weights['low_level'] + 0.01)
                self.routing_weights['high_level'] = 1.0 - self.routing_weights['low_level']
            elif high_trend > low_trend:
                self.routing_weights['high_level'] = min(0.8, self.routing_weights['high_level'] + 0.01)
                self.routing_weights['low_level'] = 1.0 - self.routing_weights['high_level']

        return {
            'low_level_need': low_need,
            'high_level_need': high_need,
            'dominant': 'high_level' if high_need > low_need else 'low_level',
            'routing_weights': self.routing_weights.copy(),
        }

    def _compute_trend(self, errors: deque) -> float:
        """计算误差趋势"""
        if len(errors) < 5:
            return 0.0
        recent = list(errors)[-5:]
        older = list(errors)[-10:-5] if len(errors) >= 10 else list(errors)[:5]
        return sum(recent) / len(recent) - sum(older) / len(older)

    def get_stats(self) -> Dict:
        return {
            'low_level_errors': len(self.low_level_errors),
            'high_level_errors': len(self.high_level_errors),
            'routing_weights': self.routing_weights,
        }


# ============================================================
# 2. GHL全局调制Hebbian学习
# ============================================================

class GlobalModulatedHebbian:
    """全局调制Hebbian学习 — 神经调质信号

    基于 "Hebbian Learning with Global Direction" (arXiv 2026)：
    - 纯Hebbian学习缺乏全局优化方向
    - 用全局信号的符号（sign）来调制局部更新方向
    - 大脑用多巴胺、去甲肾上腺素等传递全局方向信号

    核心公式：
    Δw = η × sign(global_signal) × pre × post

    其中 sign(global_signal) 只取+1或-1，不传递精确梯度。
    """

    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

        # 全局信号历史
        self.global_signals: deque = deque(maxlen=100)

        # 神经调质水平
        self.neuromodulators = {
            'dopamine': 0.5,      # 奖励信号
            'norepinephrine': 0.5, # 警觉信号
            'acetylcholine': 0.5,  # 注意信号
            'serotonin': 0.5,      # 情绪信号
        }

    def compute_global_signal(self, reward: float, novelty: float,
                             uncertainty: float) -> float:
        """计算全局信号（类似神经调质）

        Args:
            reward: 奖励信号
            novelty: 新颖性信号
            uncertainty: 不确定性信号

        Returns:
            全局信号值
        """
        # 综合信号
        global_signal = (
            reward * 0.5 +
            novelty * 0.3 +
            uncertainty * 0.2
        )

        self.global_signals.append(global_signal)

        # 更新神经调质水平
        self.neuromodulators['dopamine'] = 0.3 * reward + 0.7 * self.neuromodulators['dopamine']
        self.neuromodulators['norepinephrine'] = 0.3 * uncertainty + 0.7 * self.neuromodulators['norepinephrine']
        self.neuromodulators['acetylcholine'] = 0.3 * novelty + 0.7 * self.neuromodulators['acetylcholine']

        return global_signal

    def hebbian_update(self, pre: torch.Tensor, post: torch.Tensor,
                      global_signal: float) -> torch.Tensor:
        """全局调制的Hebbian更新

        Δw = η × sign(global_signal) × pre × post
        """
        # 只取符号，不传递精确梯度
        direction = 1.0 if global_signal > 0 else -1.0

        # Hebbian更新
        delta = self.lr * direction * torch.outer(post, pre)

        return delta

    def get_neuromodulator_level(self, name: str) -> float:
        """获取神经调质水平"""
        return self.neuromodulators.get(name, 0.5)

    def get_stats(self) -> Dict:
        return {
            'global_signals': len(self.global_signals),
            'neuromodulators': self.neuromodulators.copy(),
        }


# ============================================================
# 3. 学习进展好奇心
# ============================================================

class LearningProgressCuriosity:
    """学习进展好奇心 — 探索甜蜜区

    基于 Oudeyer "Curiosity" (HAL-Inria 2026)：
    - 好奇心不是对新奇事物的盲目追求
    - 而是对"学习进展最大的区域"的敏感
    - 甜蜜区：不太简单（已掌握）也不太难（无法学习）

    核心机制：
    1. 追踪每个领域的学习速度
    2. 优先探索学习进展最大的领域
    3. 当学习速度下降时转向新领域
    """

    def __init__(self):
        # 每个领域的学习进展
        self.domain_progress: Dict[str, deque] = {}

        # 每个领域的学习速度
        self.learning_speed: Dict[str, float] = {}

        # 探索历史
        self.exploration_history: List[Dict] = []

    def update_progress(self, domain: str, performance: float):
        """更新某个领域的学习进展"""
        if domain not in self.domain_progress:
            self.domain_progress[domain] = deque(maxlen=50)

        self.domain_progress[domain].append(performance)

        # 计算学习速度（最近5次 vs 之前5次的差异）
        history = list(self.domain_progress[domain])
        if len(history) >= 10:
            recent = sum(history[-5:]) / 5
            older = sum(history[-10:-5]) / 5
            self.learning_speed[domain] = recent - older
        elif len(history) >= 2:
            self.learning_speed[domain] = history[-1] - history[0]

    def get_exploration_priority(self) -> Dict[str, float]:
        """获取每个领域的探索优先级

        优先级 = 学习进展（正值=正在进步，负值=需要改变策略）
        """
        priorities = {}
        for domain, speed in self.learning_speed.items():
            if speed > 0.01:
                # 正在进步，继续探索
                priorities[domain] = 0.7
            elif speed < -0.01:
                # 退步，需要改变策略
                priorities[domain] = 0.3
            else:
                # 饱和，探索新领域
                priorities[domain] = 0.9

        return priorities

    def should_explore_new(self) -> bool:
        """判断是否应该探索新领域"""
        if not self.learning_speed:
            return True

        # 如果大多数领域的学习速度都很低，应该探索新领域
        low_speed_count = sum(1 for s in self.learning_speed.values() if abs(s) < 0.01)
        return low_speed_count > len(self.learning_speed) * 0.7

    def get_sweet_spot_domain(self) -> Optional[str]:
        """找到甜蜜区领域（学习进展最大的）"""
        if not self.learning_speed:
            return None

        # 找学习速度最接近0.05的领域（不太快也不太慢）
        target = 0.05
        best_domain = None
        best_distance = float('inf')

        for domain, speed in self.learning_speed.items():
            distance = abs(speed - target)
            if distance < best_distance:
                best_distance = distance
                best_domain = domain

        return best_domain

    def get_stats(self) -> Dict:
        return {
            'domains': len(self.domain_progress),
            'learning_speeds': dict(self.learning_speed),
            'should_explore_new': self.should_explore_new(),
        }


# ============================================================
# 4. 先类别后语言
# ============================================================

class PerceptualCategorySystem:
    """感知类别系统 — 先类别后语言

    基于 "Two-month-old babies making sense of the world"
    (Nature Neuroscience 2026)：
    - 婴儿不需要语言就能进行类别识别
    - 感知分类是语言习得的前提条件
    - 关键insight：类别先于语言

    对当前系统的应用：
    - 在语言涌现之前先建立纯感知类别系统
    - 通过聚类发现自然类别
    - 语言符号从类别中涌现
    """

    def __init__(self, n_categories: int = 20):
        self.n_categories = n_categories

        # 类别原型
        self.prototypes: Dict[str, torch.Tensor] = {}

        # 类别成员
        self.members: Dict[str, List[str]] = {}

        # 类别统计
        self.category_counts: Dict[str, int] = {}

    def discover_category(self, entity: str, representation: torch.Tensor,
                         threshold: float = 0.7) -> str:
        """发现或创建类别

        Args:
            entity: 实体名称
            representation: 实体的感知表示
            threshold: 相似度阈值

        Returns:
            类别名称
        """
        # 找最相似的已有类别
        best_category = None
        best_similarity = -1.0

        for cat_name, prototype in self.prototypes.items():
            sim = torch.cosine_similarity(
                representation.unsqueeze(0), prototype.unsqueeze(0)
            ).item()
            if sim > best_similarity:
                best_similarity = sim
                best_category = cat_name

        # 如果相似度够高，归入已有类别
        if best_category and best_similarity > threshold:
            self.members[best_category].append(entity)
            self.category_counts[best_category] = self.category_counts.get(best_category, 0) + 1

            # 更新原型（移动平均）
            n = len(self.members[best_category])
            self.prototypes[best_category] = (
                self.prototypes[best_category] * (n - 1) + representation
            ) / n

            return best_category

        # 否则创建新类别
        if len(self.prototypes) < self.n_categories:
            new_cat = f"category_{len(self.prototypes)}"
            self.prototypes[new_cat] = representation.detach().clone()
            self.members[new_cat] = [entity]
            self.category_counts[new_cat] = 1
            return new_cat

        # 已满，归入最相似的
        if best_category:
            self.members[best_category].append(entity)
            self.category_counts[best_category] = self.category_counts.get(best_category, 0) + 1

        return best_category or 'unknown'

    def get_category_members(self, category: str) -> List[str]:
        """获取类别成员"""
        return self.members.get(category, [])

    def get_category_prototype(self, category: str) -> Optional[torch.Tensor]:
        """获取类别原型"""
        return self.prototypes.get(category)

    def get_stats(self) -> Dict:
        return {
            'categories': len(self.prototypes),
            'total_members': sum(len(m) for m in self.members.values()),
            'category_sizes': {k: len(v) for k, v in self.members.items()},
        }


# ============================================================
# 5. 元学习组合规则
# ============================================================

class MetaLearningComposition:
    """元学习组合规则 — 学习如何组合

    基于 "Human-like systematic generalization through MLC"
    (Nature 2023)：
    - 关键不是架构设计，而是训练过程
    - 让网络学会"如何组合"，而非"记住什么"
    - 通过元学习获得组合泛化的归纳偏置

    对当前系统的应用：
    - 记录成功的组合模式
    - 提取组合规则（而非组合实例）
    - 将规则应用到新组合
    """

    def __init__(self):
        # 组合规则库
        self.rules: Dict[str, Dict] = {}

        # 组合历史
        self.composition_history: List[Dict] = []

        # 规则统计
        self.rule_usage: Dict[str, int] = {}

    def learn_rule(self, components: List[str], result: str,
                  success: bool):
        """从组合经验中学习规则

        Args:
            components: 组合的组成部分
            result: 组合结果
            success: 是否成功
        """
        # 提取规则模式
        rule_pattern = self._extract_pattern(components)

        if rule_pattern not in self.rules:
            self.rules[rule_pattern] = {
                'pattern': rule_pattern,
                'examples': [],
                'success_count': 0,
                'total_count': 0,
            }

        self.rules[rule_pattern]['examples'].append({
            'components': components,
            'result': result,
            'success': success,
        })
        self.rules[rule_pattern]['total_count'] += 1
        if success:
            self.rules[rule_pattern]['success_count'] += 1

        self.composition_history.append({
            'components': components,
            'result': result,
            'success': success,
            'rule': rule_pattern,
        })

    def apply_rule(self, components: List[str]) -> Optional[str]:
        """应用已学到的规则进行组合

        Args:
            components: 要组合的部分

        Returns:
            预测的组合结果，如果没有匹配的规则返回None
        """
        rule_pattern = self._extract_pattern(components)

        if rule_pattern in self.rules:
            rule = self.rules[rule_pattern]
            if rule['success_count'] > 0:
                # 找最成功的示例
                best_example = max(
                    [e for e in rule['examples'] if e['success']],
                    key=lambda x: 1,  # 简化：取第一个成功的
                    default=None,
                )
                if best_example:
                    self.rule_usage[rule_pattern] = self.rule_usage.get(rule_pattern, 0) + 1
                    return best_example['result']

        return None

    def _extract_pattern(self, components: List[str]) -> str:
        """提取组合模式（抽象化）"""
        # 简化：用组件数量和类型作为模式
        n = len(components)
        types = []
        for c in components:
            if any('一' <= ch <= '鿿' for ch in c):
                types.append('zh')
            elif c[0].isupper():
                types.append('en_upper')
            else:
                types.append('en_lower')

        return f"{n}_{'_'.join(types)}"

    def get_best_rules(self, n: int = 5) -> List[Dict]:
        """获取最成功的规则"""
        sorted_rules = sorted(
            self.rules.values(),
            key=lambda r: r['success_count'] / max(r['total_count'], 1),
            reverse=True,
        )
        return sorted_rules[:n]

    def get_stats(self) -> Dict:
        return {
            'rules': len(self.rules),
            'compositions': len(self.composition_history),
            'rule_usage': dict(self.rule_usage),
        }
