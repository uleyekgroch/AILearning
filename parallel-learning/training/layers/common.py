"""共享工具模块 — 统一实体提取等公共逻辑

解决：4个模块各自实现_extract_entities的问题
"""

import re
from typing import List, Set


# 统一停用词表
STOPWORDS = set(
    '的了是在我你他她它们这那个有不人大中上下来什么如何怎样'
    '可以能够应该已经正在将要一个这个那个一些很多所有'
    '因为所以如果那么但是而且或者而但虽然尽管不过而且'
    '为了对于关于通过使用进行实现'
)

# 关系指示词
RELATION_INDICATORS = {
    '是': 'is_a',
    '属于': 'belongs_to',
    '包括': 'includes',
    '位于': 'located_at',
    '使用': 'uses',
    '用于': 'used_for',
    '产生': 'produces',
    '导致': 'causes',
    '发明': 'invented',
    '发现': 'discovered',
    '创造': 'created',
    '提出': 'proposed',
    '开发': 'developed',
    '设计': 'designed',
    '称为': 'called',
    '叫做': 'called',
    '指的是': 'refers_to',
    '意味着': 'means',
}


def extract_entities(text: str, min_length: int = 2, max_length: int = 8) -> List[str]:
    """统一实体提取

    Args:
        text: 输入文本
        min_length: 最小实体长度
        max_length: 最大实体长度

    Returns:
        实体列表
    """
    entities = []

    # 分隔符
    separators = r'[，。！？；：、\s的了是在有位于属于包括使用产生导致引起为了因为所以如果那么但是而且或者而但]'

    # 分割
    parts = re.split(separators, text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # 中文实体
        zh_matches = re.findall(rf'([一-鿿]{{{min_length},{max_length}}})', part)
        for entity in zh_matches:
            if _validate_entity(entity):
                entities.append(entity)

        # 英文实体（大写开头）
        en_matches = re.findall(r'([A-Z][a-zA-Z]+)', part)
        for entity in en_matches:
            if _validate_entity(entity):
                entities.append(entity)

    return list(set(entities))


def _validate_entity(text: str) -> bool:
    """验证是否是有效的实体"""
    if len(text) < 2:
        return False
    if text in STOPWORDS:
        return False
    if re.match(r'^[\W\s]+$', text):
        return False
    # 过滤纯数字
    if text.isdigit():
        return False
    return True


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """提取关键词"""
    entities = extract_entities(text)

    # 按长度排序，优先保留长实体
    entities.sort(key=len, reverse=True)

    return entities[:max_keywords]
