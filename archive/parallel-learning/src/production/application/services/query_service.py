"""
查询应用服务

负责查询的用例编排
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from src.production.domain.knowledge.commonsense_fact import CommonsenseFact
from .knowledge_service import KnowledgeApplicationService


class QueryApplicationService:
    """查询应用服务

    职责：
    - 编排查询的用例
    - 提供统一的查询接口
    - 支持多种查询模式

    Attributes:
        knowledge_service: 知识管理应用服务
    """

    def __init__(self, knowledge_service: KnowledgeApplicationService = None):
        """初始化查询应用服务

        Args:
            knowledge_service: 知识管理应用服务（可选）
        """
        self.knowledge_service = knowledge_service or KnowledgeApplicationService()

    def query_commonsense(self, query: str, kb_name: str = 'default') -> List[CommonsenseFact]:
        """常识查询

        Args:
            query: 查询文本
            kb_name: 知识库名称（默认为'default'）

        Returns:
            匹配的事实列表
        """
        # 尝试按主题查询
        results = self.knowledge_service.query_by_subject(kb_name, query)

        # 如果没有结果，尝试按陈述搜索
        if not results:
            results = self.knowledge_service.search_by_statement(kb_name, query)

        # 如果还是没有结果，尝试按谓语查询
        if not results:
            results = self.knowledge_service.query_by_predicate(kb_name, query)

        # 按置信度排序
        results.sort(key=lambda f: -f.confidence)

        return results

    def query_with_context(
        self,
        query: str,
        context: Dict[str, Any],
        kb_name: str = 'default'
    ) -> List[CommonsenseFact]:
        """带上下文的查询

        Args:
            query: 查询文本
            context: 上下文信息
            kb_name: 知识库名称（默认为'default'）

        Returns:
            匹配的事实列表
        """
        # 基础查询
        results = self.query_commonsense(query, kb_name)

        # 根据上下文过滤（简化实现，避免过度过滤）
        if context and results:
            precision = context.get('precision')
            if precision == 'high':
                # 只保留高置信度结果
                high_conf_results = [f for f in results if f.confidence > 0.8]
                # 如果有过滤结果，使用过滤后的；否则保持原结果
                if high_conf_results:
                    results = high_conf_results

        return results

    def query_by_triple_pattern(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
        kb_name: str = 'default'
    ) -> List[CommonsenseFact]:
        """按三元组模式查询

        Args:
            subject: 主题（可选）
            predicate: 谓语（可选）
            object_: 宾语（可选）
            kb_name: 知识库名称（默认为'default'）

        Returns:
            匹配的事实列表
        """
        return self.knowledge_service.query_by_triple(
            kb_name, subject, predicate, object_
        )

    def query_related_facts(
        self,
        concept: str,
        max_depth: int = 2,
        kb_name: str = 'default'
    ) -> List[CommonsenseFact]:
        """查询相关事实

        Args:
            concept: 概念
            max_depth: 最大深度
            kb_name: 知识库名称（默认为'default'）

        Returns:
            相关的事实列表
        """
        # 获取直接相关的事实
        direct_facts = self.knowledge_service.query_by_subject(kb_name, concept)

        # 如果需要更深层次的查询
        if max_depth > 1:
            # 获取宾语相关的事实
            indirect_facts = []
            for fact in direct_facts:
                related = self.knowledge_service.query_by_subject(kb_name, fact.object)
                indirect_facts.extend(related)

            # 合并结果，去重
            all_facts = direct_facts + indirect_facts
            seen_ids = set()
            unique_facts = []
            for fact in all_facts:
                if fact.fact_id not in seen_ids:
                    seen_ids.add(fact.fact_id)
                    unique_facts.append(fact)

            return unique_facts

        return direct_facts

    def get_statistics(self, kb_name: str = 'default') -> Dict[str, Any]:
        """获取查询统计信息

        Args:
            kb_name: 知识库名称（默认为'default'）

        Returns:
            统计信息字典
        """
        return self.knowledge_service.get_statistics(kb_name)
