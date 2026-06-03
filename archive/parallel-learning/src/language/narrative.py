"""
叙事 / 话语模块 — 多事件序列描述与连接词

核心理念：当单句无法描述完整场景时，叙事结构（事件序列 + 时序/因果连接词）
从通信压力中涌现。

纯 PyTorch 实现，零 numpy 依赖。
"""

from typing import Dict, List, Optional, Tuple

import torch

from src.core.device import get_device


# =====================================================================
# 连接词集合
# =====================================================================

NARRATIVE_MARKERS = {'then', 'because', 'so', 'but', 'and'}
TEMPORAL_MARKERS = {'then', 'before', 'after'}
CAUSAL_MARKERS = {'because', 'so', 'therefore'}


# =====================================================================
# Event — 叙事中的单个事件
# =====================================================================

class Event:
    """
    叙事中的单个事件。

    属性：
        subject: 事件主体（施事者特征字典）
        action: 动作描述
        object_features: 受事者特征（可选）
    """

    def __init__(self, subject: Dict[str, str], action: str,
                 object_features: Optional[Dict[str, str]] = None):
        self.subject = subject
        self.action = action
        self.object_features = object_features or {}

    def to_symbols(self) -> List[str]:
        """将事件转换为符号序列"""
        symbols: List[str] = []
        # 主体属性
        for val in self.subject.values():
            if val:
                symbols.append(val)
        # 动作
        if self.action:
            symbols.append(self.action)
        # 受事者属性
        for val in self.object_features.values():
            if val:
                symbols.append(val)
        return symbols

    def __repr__(self) -> str:
        subj = ' '.join(self.subject.values())
        obj = ' '.join(self.object_features.values()) if self.object_features else ''
        return f'Event({subj} {self.action} {obj})'.strip()


# =====================================================================
# Narrative — 事件序列 + 连接词
# =====================================================================

class Narrative:
    """
    叙事 = 事件序列 + 连接词。

    属性：
        events: 事件列表
        connections: 事件间的连接词列表（长度 = len(events) - 1）
    """

    def __init__(self, events: List[Event], connections: List[str]):
        self.events = events
        self.connections = connections

    def to_symbols(self) -> List[str]:
        """将叙事转换为符号序列（含连接词）"""
        if not self.events:
            return []

        symbols: List[str] = []
        for i, event in enumerate(self.events):
            symbols.extend(event.to_symbols())
            # 在事件之间插入连接词
            if i < len(self.connections):
                symbols.append(self.connections[i])
        return symbols

    @property
    def length(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        parts = []
        for i, event in enumerate(self.events):
            parts.append(str(event))
            if i < len(self.connections):
                parts.append(self.connections[i])
        return ' '.join(parts)


# =====================================================================
# NarrativeModule — 叙事 / 话语生成
# =====================================================================

class NarrativeModule:
    """
    叙事 / 话语生成模块。

    职责：
    1. 将原始事件字典列表构建为结构化叙事
    2. 选择合适的时序/因果连接词
    3. 解析符号序列恢复叙事结构
    4. 追踪连接词使用频率与涌现
    """

    def __init__(self, device: str = 'auto'):
        self.device = get_device(device)

        # 连接词使用频率
        self._connector_usage: Dict[str, int] = {m: 0 for m in NARRATIVE_MARKERS}
        # 因果连接词额外追踪
        for m in CAUSAL_MARKERS:
            self._connector_usage.setdefault(m, 0)

        # 叙事历史（最近 100 条）
        self._narrative_history: List[Narrative] = []

        # 连接词选择权重张量 — 可学习
        connector_list = sorted(self._connector_usage.keys())
        self._connector_names: List[str] = connector_list
        self._connector_weights = torch.ones(
            len(connector_list), device=self.device,
        )

        # 事件间关系类型统计
        self._relation_stats: Dict[str, int] = {
            'temporal': 0,
            'causal': 0,
            'contrast': 0,
            'additive': 0,
        }

    # ------------------------------------------------------------------
    # 核心方法：构建叙事
    # ------------------------------------------------------------------

    def build_narrative(self, events: List[Dict]) -> Narrative:
        """
        从原始事件字典列表构建叙事。

        每个 event dict 应包含：
        - 'subject': Dict[str, str] 主体特征
        - 'action': str 动作
        - 'object': Optional[Dict[str, str]] 受事者特征

        Args:
            events: 事件字典列表

        Returns:
            Narrative 对象（事件序列 + 连接词）
        """
        if not events:
            return Narrative([], [])

        # 构建 Event 对象
        event_objects: List[Event] = []
        for ev in events:
            subject = ev.get('subject', {})
            action = ev.get('action', '')
            obj_features = ev.get('object', None)
            event_objects.append(Event(subject, action, obj_features))

        # 为每对相邻事件选择连接词
        connections: List[str] = []
        for i in range(len(event_objects) - 1):
            connector = self.choose_connector(
                events[i], events[i + 1],
            )
            connections.append(connector)
            self._connector_usage[connector] = (
                self._connector_usage.get(connector, 0) + 1
            )

        narrative = Narrative(event_objects, connections)
        self._narrative_history.append(narrative)
        if len(self._narrative_history) > 100:
            self._narrative_history = self._narrative_history[-100:]

        return narrative

    def choose_connector(self, event_a: Dict, event_b: Dict) -> str:
        """
        为相邻事件选择连接词。

        启发式规则：
        - 事件有因果链 → 'because' / 'so'
        - 事件有对比关系 → 'but'
        - 事件有时序关系 → 'then'
        - 默认 → 'and'

        Args:
            event_a: 前一个事件
            event_b: 后一个事件

        Returns:
            连接词字符串
        """
        # 因果关系检测
        if self._has_causal_link(event_a, event_b):
            self._relation_stats['causal'] += 1
            # 因果关系：如果 B 是 A 的结果用 'so'，如果 A 是 B 的原因用 'because'
            if event_b.get('caused_by') == self._event_id(event_a):
                return 'because'
            return 'so'

        # 对比关系检测
        if self._has_contrast(event_a, event_b):
            self._relation_stats['contrast'] += 1
            return 'but'

        # 时序关系检测
        if self._has_temporal_order(event_a, event_b):
            self._relation_stats['temporal'] += 1
            return 'then'

        # 默认：并列
        self._relation_stats['additive'] += 1
        return 'and'

    # ------------------------------------------------------------------
    # 叙事解析
    # ------------------------------------------------------------------

    def parse_narrative(self, symbols: List[str]) -> Narrative:
        """
        将符号序列解析为叙事结构。

        以连接词为分割点，将符号序列还原为事件 + 连接词。

        Args:
            symbols: 符号序列

        Returns:
            Narrative 对象
        """
        if not symbols:
            return Narrative([], [])

        # 按连接词分割
        all_markers = NARRATIVE_MARKERS | TEMPORAL_MARKERS | CAUSAL_MARKERS
        segments: List[List[str]] = []
        connections: List[str] = []
        current: List[str] = []

        for sym in symbols:
            if sym in all_markers and current:
                # 遇到连接词，结束当前段
                connections.append(sym)
                segments.append(current)
                current = []
            else:
                current.append(sym)

        if current:
            segments.append(current)

        # 从每个段构建 Event
        events: List[Event] = []
        for seg in segments:
            if not seg:
                continue
            # 简单启发式：第一个词为主语，中间词为动作，其余为宾语特征
            if len(seg) >= 3:
                subject = {'entity': seg[0]}
                action = seg[1]
                obj_features = {'entity': seg[2]}
                # 剩余词追加到宾语特征
                for i, s in enumerate(seg[3:], start=3):
                    obj_features[f'attr_{i}'] = s
            elif len(seg) == 2:
                subject = {'entity': seg[0]}
                action = seg[1]
                obj_features = {}
            else:
                subject = {'entity': seg[0]}
                action = ''
                obj_features = {}
            events.append(Event(subject, action, obj_features))

        return Narrative(events, connections)

    # ------------------------------------------------------------------
    # 统计与涌现
    # ------------------------------------------------------------------

    def get_connector_stats(self) -> Dict[str, float]:
        """
        返回连接词使用频率。

        Returns:
            connector -> 归一化频率 (0.0 ~ 1.0)
        """
        total = max(sum(self._connector_usage.values()), 1)
        return {
            conn: count / total
            for conn, count in self._connector_usage.items()
        }

    def get_emergent_connectors(self, threshold: float = 0.1) -> List[str]:
        """
        返回已涌现的连接词（频率超过阈值）。

        Args:
            threshold: 涌现频率阈值

        Returns:
            已涌现的连接词列表
        """
        stats = self.get_connector_stats()
        return [c for c, freq in stats.items() if freq >= threshold]

    def get_relation_distribution(self) -> Dict[str, float]:
        """返回事件间关系类型的分布"""
        total = max(sum(self._relation_stats.values()), 1)
        return {k: v / total for k, v in self._relation_stats.items()}

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _has_causal_link(self, event_a: Dict, event_b: Dict) -> bool:
        """检测两个事件之间是否有因果关系"""
        # 显式因果标记
        if event_b.get('caused_by') or event_a.get('causes'):
            a_target = event_a.get('causes', '')
            b_id = self._event_id(event_b)
            if a_target and a_target == b_id:
                return True
            b_cause = event_b.get('caused_by', '')
            a_id = self._event_id(event_a)
            if b_cause and b_cause == a_id:
                return True

        # 隐式因果：动作-结果模式
        action_a = event_a.get('action', '')
        action_b = event_b.get('action', '')
        causal_pairs = {
            ('push', 'fall'), ('heat', 'melt'), ('drop', 'break'),
            ('touch', 'move'), ('pull', 'come'),
        }
        return (action_a, action_b) in causal_pairs

    def _has_contrast(self, event_a: Dict, event_b: Dict) -> bool:
        """检测两个事件之间是否有对比关系"""
        subj_a = event_a.get('subject', {})
        subj_b = event_b.get('subject', {})
        # 同一主语但不同动作 → 对比
        if subj_a == subj_b:
            return event_a.get('action') != event_b.get('action')
        # 显式对比标记
        return bool(event_a.get('contrast_with') or event_b.get('contrast_with'))

    def _has_temporal_order(self, event_a: Dict, event_b: Dict) -> bool:
        """检测两个事件之间是否有时序关系"""
        t_a = event_a.get('time')
        t_b = event_b.get('time')
        if t_a is not None and t_b is not None:
            return t_a < t_b
        # 没有时间戳时默认无时序（避免过度使用 'then'）
        return False

    def _event_id(self, event: Dict) -> str:
        """获取事件的标识符"""
        return event.get('id', '')

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        """序列化模块状态"""
        return {
            'connector_usage': dict(self._connector_usage),
            'connector_weights': self._connector_weights.detach().cpu(),
            'relation_stats': dict(self._relation_stats),
            'narrative_count': len(self._narrative_history),
        }

    def load_state(self, state: dict) -> None:
        """从字典恢复模块状态"""
        self._connector_usage = state.get('connector_usage', {
            m: 0 for m in NARRATIVE_MARKERS
        })
        if 'connector_weights' in state:
            self._connector_weights = state['connector_weights'].to(self.device)
        self._relation_stats = state.get('relation_stats', {
            'temporal': 0, 'causal': 0, 'contrast': 0, 'additive': 0,
        })
