"""通用知识单元 — 任何领域的概念表示

这是通用学习架构的核心数据结构。
一个 KnowledgeUnit 代表一个可学习的概念，可以是：
- 一个英语单词（domain='english'）
- 一个数学定理（domain='math'）
- 一个物理定律（domain='physics'）
- 一个编程概念（domain='cs'）
- 一个化学反应（domain='chemistry'）
- 任何其他知识领域的一个概念
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class MasteryLevel(Enum):
    """掌握度等级"""
    UNKNOWN = 0      # 完全未知
    EXPOSED = 1      # 接触过（见过/听过）
    RECOGNIZED = 2   # 能识别（看到能认出）
    UNDERSTOOD = 3   # 能理解（知道含义）
    APPLIED = 4      # 能应用（能使用）
    MASTERED = 5     # 精通（能教授）


@dataclass
class KnowledgeUnit:
    """通用知识单元

    代表一个可学习的概念，包含：
    - 基本信息（名称、定义、领域）
    - 学习状态（掌握度、练习次数）
    - 知识关系（前置知识、相关概念）
    - 学习资源（示例、解释、练习题）

    这是通用学习架构的核心，任何领域的知识都可以用这个结构表示。
    """

    # ── 基本信息 ──────────────────────────────────────────────
    id: str                           # 唯一标识（如 'math:calculus:derivative'）
    name: str                         # 概念名称（如 '导数'）
    domain: str                       # 知识领域（如 'math', 'physics', 'english'）
    definition: str = ''              # 定义/描述
    difficulty: float = 0.5           # 难度 [0, 1]（0=最简单，1=最难）

    # ── 学习资源 ──────────────────────────────────────────────
    examples: List[str] = field(default_factory=list)           # 示例
    explanations: List[str] = field(default_factory=list)       # 解释
    exercises: List[Dict[str, Any]] = field(default_factory=list)  # 练习题
    keywords: List[str] = field(default_factory=list)           # 关键词

    # ── 知识关系 ──────────────────────────────────────────────
    prerequisites: List[str] = field(default_factory=list)      # 前置知识（其他单元的 id）
    related: List[str] = field(default_factory=list)            # 相关概念
    is_a: List[str] = field(default_factory=list)               # 上位概念（is-a 关系）
    part_of: List[str] = field(default_factory=list)            # 所属整体（part-of 关系）
    has_part: List[str] = field(default_factory=list)           # 包含部分（has-part 关系）

    # ── 学习状态 ──────────────────────────────────────────────
    mastery: float = 0.0              # 掌握度 [0, 1]
    mastery_level: MasteryLevel = MasteryLevel.UNKNOWN  # 掌握等级
    exposure_count: int = 0           # 接触次数
    practice_count: int = 0           # 练习次数
    success_count: int = 0            # 成功次数
    last_practice: float = 0.0        # 上次练习时间
    error_history: List[float] = field(default_factory=list)    # 错误历史

    # ── 元数据 ──────────────────────────────────────────────
    source: str = ''                  # 来源（如 'oxford', 'textbook', 'api'）
    tags: List[str] = field(default_factory=list)               # 标签
    metadata: Dict[str, Any] = field(default_factory=dict)      # 额外元数据

    def update_mastery(self, success: bool, quality: float = 0.5) -> None:
        """更新掌握度

        Args:
            success: 是否成功
            quality: 质量评分 [0, 1]（0=完全错误，1=完美）
        """
        self.practice_count += 1
        if success:
            self.success_count += 1
            # 成功时增加掌握度
            delta = 0.1 * quality
            self.mastery = min(1.0, self.mastery + delta)
        else:
            # 失败时轻微降低掌握度
            self.mastery = max(0.0, self.mastery - 0.05)

        # 更新掌握等级
        self._update_mastery_level()

    def _update_mastery_level(self) -> None:
        """根据掌握度更新掌握等级"""
        if self.mastery < 0.1:
            self.mastery_level = MasteryLevel.UNKNOWN
        elif self.mastery < 0.3:
            self.mastery_level = MasteryLevel.EXPOSED
        elif self.mastery < 0.5:
            self.mastery_level = MasteryLevel.RECOGNIZED
        elif self.mastery < 0.7:
            self.mastery_level = MasteryLevel.UNDERSTOOD
        elif self.mastery < 0.9:
            self.mastery_level = MasteryLevel.APPLIED
        else:
            self.mastery_level = MasteryLevel.MASTERED

    def is_prerequisites_met(self, all_units: Dict[str, 'KnowledgeUnit'],
                             threshold: float = 0.5) -> bool:
        """检查前置知识是否满足

        Args:
            all_units: 所有知识单元的字典（id -> unit）
            threshold: 前置知识掌握度阈值

        Returns:
            前置知识是否满足
        """
        if not self.prerequisites:
            return True

        for prereq_id in self.prerequisites:
            prereq = all_units.get(prereq_id)
            if prereq is None:
                continue  # 前置知识不存在，跳过
            if prereq.mastery < threshold:
                return False

        return True

    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.practice_count == 0:
            return 0.0
        return self.success_count / self.practice_count

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'domain': self.domain,
            'definition': self.definition,
            'difficulty': self.difficulty,
            'examples': self.examples,
            'explanations': self.explanations,
            'exercises': self.exercises,
            'keywords': self.keywords,
            'prerequisites': self.prerequisites,
            'related': self.related,
            'is_a': self.is_a,
            'part_of': self.part_of,
            'has_part': self.has_part,
            'mastery': self.mastery,
            'mastery_level': self.mastery_level.name,
            'exposure_count': self.exposure_count,
            'practice_count': self.practice_count,
            'success_count': self.success_count,
            'last_practice': self.last_practice,
            'error_history': self.error_history,
            'source': self.source,
            'tags': self.tags,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'KnowledgeUnit':
        """从字典创建"""
        mastery_level = MasteryLevel.UNKNOWN
        if 'mastery_level' in d:
            try:
                mastery_level = MasteryLevel[d['mastery_level']]
            except KeyError:
                pass

        return cls(
            id=d.get('id', ''),
            name=d.get('name', ''),
            domain=d.get('domain', ''),
            definition=d.get('definition', ''),
            difficulty=d.get('difficulty', 0.5),
            examples=d.get('examples', []),
            explanations=d.get('explanations', []),
            exercises=d.get('exercises', []),
            keywords=d.get('keywords', []),
            prerequisites=d.get('prerequisites', []),
            related=d.get('related', []),
            is_a=d.get('is_a', []),
            part_of=d.get('part_of', []),
            has_part=d.get('has_part', []),
            mastery=d.get('mastery', 0.0),
            mastery_level=mastery_level,
            exposure_count=d.get('exposure_count', 0),
            practice_count=d.get('practice_count', 0),
            success_count=d.get('success_count', 0),
            last_practice=d.get('last_practice', 0.0),
            error_history=d.get('error_history', []),
            source=d.get('source', ''),
            tags=d.get('tags', []),
            metadata=d.get('metadata', {}),
        )
