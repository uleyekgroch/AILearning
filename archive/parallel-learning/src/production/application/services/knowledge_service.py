"""
知识管理应用服务

负责知识库的用例编排
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from src.production.domain.knowledge.knowledge_base import KnowledgeBase
from src.production.domain.knowledge.commonsense_fact import CommonsenseFact


class KnowledgeApplicationService:
    """知识管理应用服务

    职责：
    - 编排知识库的用例
    - 管理知识库的生命周期
    - 提供知识查询接口

    Attributes:
        knowledge_bases: 知识库映射
    """

    def __init__(self):
        """初始化知识管理应用服务"""
        self.knowledge_bases: Dict[str, KnowledgeBase] = {}

    def create_knowledge_base(self, name: str) -> KnowledgeBase:
        """创建知识库

        Args:
            name: 知识库名称

        Returns:
            创建的知识库
        """
        if name in self.knowledge_bases:
            raise ValueError(f"Knowledge base {name} already exists")

        kb = KnowledgeBase(name=name)
        self.knowledge_bases[name] = kb
        return kb

    def get_knowledge_base(self, name: str) -> Optional[KnowledgeBase]:
        """获取知识库

        Args:
            name: 知识库名称

        Returns:
            知识库实例，如果不存在返回None
        """
        return self.knowledge_bases.get(name)

    def add_fact(self, kb_name: str, fact_data: Dict[str, Any]) -> CommonsenseFact:
        """添加事实到知识库

        Args:
            kb_name: 知识库名称
            fact_data: 事实数据字典

        Returns:
            添加的事实

        Raises:
            ValueError: 如果知识库不存在
        """
        # 获取或创建知识库
        kb = self.knowledge_bases.get(kb_name)
        if kb is None:
            kb = self.create_knowledge_base(kb_name)

        # 创建事实对象
        fact = CommonsenseFact(
            fact_id=fact_data['fact_id'],
            statement=fact_data['statement'],
            subject=fact_data['subject'],
            predicate=fact_data['predicate'],
            object=fact_data['object'],
            confidence=fact_data['confidence'],
            fact_type=fact_data.get('fact_type', 'physical'),
            source=fact_data.get('source', 'manual'),
            metadata=fact_data.get('metadata', {})
        )

        # 添加到知识库
        kb.add_fact(fact)

        return fact

    def query_by_subject(self, kb_name: str, subject: str) -> List[CommonsenseFact]:
        """按主题查询事实

        Args:
            kb_name: 知识库名称
            subject: 主题

        Returns:
            匹配的事实列表

        Raises:
            ValueError: 如果知识库不存在
        """
        kb = self.knowledge_bases.get(kb_name)
        if kb is None:
            raise ValueError(f"Knowledge base {kb_name} not found")

        return kb.query_by_subject(subject)

    def query_by_predicate(self, kb_name: str, predicate: str) -> List[CommonsenseFact]:
        """按谓语查询事实

        Args:
            kb_name: 知识库名称
            predicate: 谓语

        Returns:
            匹配的事实列表

        Raises:
            ValueError: 如果知识库不存在
        """
        kb = self.knowledge_bases.get(kb_name)
        if kb is None:
            raise ValueError(f"Knowledge base {kb_name} not found")

        return kb.query_by_predicate(predicate)

    def query_by_triple(
        self,
        kb_name: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None
    ) -> List[CommonsenseFact]:
        """按三元组查询事实

        Args:
            kb_name: 知识库名称
            subject: 主题（可选）
            predicate: 谓语（可选）
            object_: 宾语（可选）

        Returns:
            匹配的事实列表

        Raises:
            ValueError: 如果知识库不存在
        """
        kb = self.knowledge_bases.get(kb_name)
        if kb is None:
            raise ValueError(f"Knowledge base {kb_name} not found")

        return kb.query_by_triple(subject, predicate, object_)

    def search_by_statement(self, kb_name: str, query: str, limit: int = 10) -> List[CommonsenseFact]:
        """按陈述搜索事实

        Args:
            kb_name: 知识库名称
            query: 搜索查询
            limit: 返回数量限制

        Returns:
            匹配的事实列表

        Raises:
            ValueError: 如果知识库不存在
        """
        kb = self.knowledge_bases.get(kb_name)
        if kb is None:
            raise ValueError(f"Knowledge base {kb_name} not found")

        return kb.search_by_statement(query, limit)

    def get_statistics(self, kb_name: str) -> Dict[str, Any]:
        """获取知识库统计信息

        Args:
            kb_name: 知识库名称

        Returns:
            统计信息字典

        Raises:
            ValueError: 如果知识库不存在
        """
        kb = self.knowledge_bases.get(kb_name)
        if kb is None:
            raise ValueError(f"Knowledge base {kb_name} not found")

        return kb.get_statistics()

    def list_knowledge_bases(self) -> List[str]:
        """列出所有知识库

        Returns:
            知识库名称列表
        """
        return list(self.knowledge_bases.keys())

    def delete_knowledge_base(self, name: str) -> bool:
        """删除知识库

        Args:
            name: 知识库名称

        Returns:
            是否删除成功
        """
        if name in self.knowledge_bases:
            del self.knowledge_bases[name]
            return True
        return False
