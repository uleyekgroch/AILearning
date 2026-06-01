"""
创造性写作系统 - 像小朋友一样学习

不是简单的概念关联，而是：
1. 理解含义
2. 创造内容
3. 修改改进
4. 自我编码
"""

from .writer import CreativeWriter
from .editor import ArticleEditor
from .self_coder import SelfCoder

__all__ = [
    "CreativeWriter",
    "ArticleEditor",
    "SelfCoder",
]
