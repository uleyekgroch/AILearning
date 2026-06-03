"""可学习文本单元"""

from dataclasses import dataclass, field
from typing import List, Set, Dict


@dataclass
class TextUnit:
    """一个可学习的文本块"""

    text: str
    sentences: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    unique_words: Set[str] = field(default_factory=set)
    new_words: Set[str] = field(default_factory=set)
    difficulty: float = 0.0
    level: str = 'elementary'
    source: str = ''
    metadata: Dict = field(default_factory=dict)
