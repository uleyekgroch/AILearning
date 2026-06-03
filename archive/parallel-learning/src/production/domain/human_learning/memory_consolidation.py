"""
记忆巩固系统 - 从第一性原理出发

人类记忆的双重系统：
1. 海马体：快速编码，容量有限
2. 皮层：长期存储，容量巨大
3. 睡眠：记忆巩固的关键
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """记忆"""
    memory_id: str
    content: Any
    strength: float  # 记忆强度 [0, 1]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationResult:
    """巩固结果"""
    memories_consolidated: int
    memories_forgotten: int
    avg_strength: float
    consolidation_time: float


class HippocampalMemory:
    """海马体记忆

    快速编码，容量有限
    特点：
    - 单次学习
    - 容量有限（~7项）
    - 快速遗忘
    """

    def __init__(self, capacity: int = 7):
        """
        初始化海马体记忆

        Args:
            capacity: 记忆容量
        """
        self.capacity = capacity
        self.memories: deque = deque(maxlen=capacity)
        self.memory_index: Dict[str, Memory] = {}

    def encode(self, content: Any, importance: float = 0.5) -> Memory:
        """
        编码记忆

        Args:
            content: 记忆内容
            importance: 重要性 [0, 1]

        Returns:
            记忆对象
        """
        # 生成记忆ID
        memory_id = f"hipp_{len(self.memories)}_{datetime.now().timestamp()}"

        # 创建记忆
        memory = Memory(
            memory_id=memory_id,
            content=content,
            strength=1.0,  # 新记忆强度最高
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            importance=importance,
        )

        # 如果容量已满，移除最旧的记忆（deque会自动移除，但需要清理索引）
        if len(self.memories) >= self.capacity:
            # 获取将被移除的元素（deque的第一个元素）
            oldest = self.memories[0]
            if oldest.memory_id in self.memory_index:
                del self.memory_index[oldest.memory_id]

        # 添加新记忆
        self.memories.append(memory)
        self.memory_index[memory_id] = memory

        logger.debug(f"Encoded memory: {memory_id}")
        return memory

    def retrieve(self, query: Any = None) -> List[Memory]:
        """
        检索记忆

        Args:
            query: 查询（可选）

        Returns:
            记忆列表
        """
        # 返回所有记忆（简化实现）
        return list(self.memories)

    def get_strongest_memories(self, n: int = 3) -> List[Memory]:
        """
        获取最强的记忆

        Args:
            n: 返回数量

        Returns:
            记忆列表
        """
        sorted_memories = sorted(
            self.memories,
            key=lambda m: m.strength * m.importance,
            reverse=True
        )
        return sorted_memories[:n]

    def forget_weak_memories(self, threshold: float = 0.3) -> List[Memory]:
        """
        遗忘弱记忆

        Args:
            threshold: 强度阈值

        Returns:
            被遗忘的记忆列表
        """
        forgotten = []
        remaining = deque()

        for memory in self.memories:
            if memory.strength < threshold:
                forgotten.append(memory)
                del self.memory_index[memory.memory_id]
            else:
                remaining.append(memory)

        self.memories = remaining
        return forgotten


class CorticalMemory:
    """皮层记忆

    长期存储，容量巨大
    特点：
    - 容量巨大
    - 存储稳定
    - 需要巩固
    """

    def __init__(self):
        """初始化皮层记忆"""
        self.memories: Dict[str, Memory] = {}
        self.consolidation_history: List[Dict[str, Any]] = []

    def store(self, memory: Memory) -> None:
        """
        存储记忆

        Args:
            memory: 记忆对象
        """
        self.memories[memory.memory_id] = memory
        logger.debug(f"Stored memory in cortex: {memory.memory_id}")

    def retrieve(self, query: Any = None, top_k: int = 10) -> List[Memory]:
        """
        检索记忆

        Args:
            query: 查询（可选）
            top_k: 返回数量

        Returns:
            记忆列表
        """
        # 按强度排序
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: m.strength,
            reverse=True
        )
        return sorted_memories[:top_k]

    def strengthen_memory(self, memory_id: str, amount: float = 0.1) -> None:
        """
        增强记忆

        Args:
            memory_id: 记忆ID
            amount: 增强量
        """
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.strength = min(1.0, memory.strength + amount)
            memory.last_accessed = datetime.now()
            memory.access_count += 1

    def decay_memories(self, decay_rate: float = 0.01) -> None:
        """
        衰减记忆

        Args:
            decay_rate: 衰减率
        """
        for memory in self.memories.values():
            # 基于时间的衰减
            time_since_access = datetime.now() - memory.last_accessed
            days = time_since_access.total_seconds() / 86400

            # 衰减公式：strength *= (1 - decay_rate)^days
            memory.strength *= (1 - decay_rate) ** days
            memory.strength = max(0.0, memory.strength)


class SleepReplay:
    """睡眠重放

    睡眠是记忆巩固的关键机制：
    - 重放白天的经历
    - 巩固重要记忆
    - 遗忘不重要记忆
    """

    def __init__(self, hippocampal: HippocampalMemory, cortical: CorticalMemory):
        """
        初始化睡眠重放

        Args:
            hippocampal: 海马体记忆
            cortical: 皮层记忆
        """
        self.hippocampal = hippocampal
        self.cortical = cortical
        self.replay_history: List[Dict[str, Any]] = []

    def replay_and_consolidate(self) -> ConsolidationResult:
        """
        重放并巩固记忆

        Returns:
            巩固结果
        """
        start_time = datetime.now()

        # 1. 获取海马体中的强记忆
        strong_memories = self.hippocampal.get_strongest_memories(n=5)

        # 2. 巩固到皮层
        consolidated_count = 0
        for memory in strong_memories:
            # 增强记忆强度
            memory.strength = min(1.0, memory.strength * 1.2)
            # 存储到皮层
            self.cortical.store(memory)
            consolidated_count += 1

        # 3. 遗忘弱记忆
        forgotten_memories = self.hippocampal.forget_weak_memories(threshold=0.3)
        forgotten_count = len(forgotten_memories)

        # 4. 衰减皮层记忆
        self.cortical.decay_memories()

        # 5. 计算平均强度
        all_memories = list(self.cortical.memories.values())
        avg_strength = np.mean([m.strength for m in all_memories]) if all_memories else 0.0

        # 计算时间
        consolidation_time = (datetime.now() - start_time).total_seconds()

        # 记录历史
        result = ConsolidationResult(
            memories_consolidated=consolidated_count,
            memories_forgotten=forgotten_count,
            avg_strength=avg_strength,
            consolidation_time=consolidation_time
        )
        self.replay_history.append({
            'timestamp': datetime.now().isoformat(),
            'consolidated': consolidated_count,
            'forgotten': forgotten_count,
            'avg_strength': avg_strength,
        })

        logger.info(f"Sleep replay: consolidated={consolidated_count}, forgotten={forgotten_count}")
        return result


class MemoryConsolidation:
    """记忆巩固系统

    实现人类记忆的双重系统：
    1. 海马体：快速编码
    2. 皮层：长期存储
    3. 睡眠：记忆巩固
    """

    def __init__(self, hippocampal_capacity: int = 7):
        """
        初始化记忆巩固系统

        Args:
            hippocampal_capacity: 海马体容量
        """
        self.hippocampal = HippocampalMemory(capacity=hippocampal_capacity)
        self.cortical = CorticalMemory()
        self.sleep_replay = SleepReplay(self.hippocampal, self.cortical)

        # 统计
        self.stats = {
            'total_encoded': 0,
            'total_consolidated': 0,
            'total_forgotten': 0,
            'sleep_cycles': 0,
        }

    def learn(self, content: Any, importance: float = 0.5) -> Memory:
        """
        学习新内容

        Args:
            content: 学习内容
            importance: 重要性 [0, 1]

        Returns:
            记忆对象
        """
        # 海马体快速编码
        memory = self.hippocampal.encode(content, importance)
        self.stats['total_encoded'] += 1

        return memory

    def sleep(self) -> ConsolidationResult:
        """
        睡眠巩固

        Returns:
            巩固结果
        """
        result = self.sleep_replay.replay_and_consolidate()
        self.stats['total_consolidated'] += result.memories_consolidated
        self.stats['total_forgotten'] += result.memories_forgotten
        self.stats['sleep_cycles'] += 1

        return result

    def recall(self, query: Any = None) -> List[Memory]:
        """
        回忆

        Args:
            query: 查询（可选）

        Returns:
            记忆列表
        """
        # 先从皮层检索
        cortical_memories = self.cortical.retrieve(query)

        # 再从海马体检索
        hippocampal_memories = self.hippocampal.retrieve(query)

        # 合并并去重
        all_memories = {}
        for memory in cortical_memories + hippocampal_memories:
            if memory.memory_id not in all_memories:
                all_memories[memory.memory_id] = memory

        # 按强度排序
        sorted_memories = sorted(
            all_memories.values(),
            key=lambda m: m.strength,
            reverse=True
        )

        return sorted_memories

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'hippocampal_count': len(self.hippocampal.memories),
            'cortical_count': len(self.cortical.memories),
        }
