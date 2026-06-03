"""
知识库CRUD操作层 - Knowledge Base CRUD Operations

整合：
- 图数据库（三元组存储）
- 向量数据库（相似度搜索）
- 数据模型（Schema验证）
- 提供统一的CRUD接口

功能：
- 知识条目增删改查
- 批量操作
- 查询和推理
- 验证和约束
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
import uuid
import numpy as np

from .schema import (
    Entity, Relation, KnowledgeEntry, Triple,
    EntityType, RelationType, EntitySubType,
    CommonsenseSchema, KnowledgeSchemaExporter
)
from .graph_db import GraphDatabase, Node, Relationship, get_graph_db
from .vector_db import VectorDatabase, VectorEmbedding, SearchResult, get_vector_db


@dataclass
class QueryResult:
    """查询结果"""
    success: bool
    results: List[Dict] = field(default_factory=list)
    total_count: int = 0
    execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeBaseCRUD:
    """
    知识库CRUD操作

    提供完整的知识管理功能：
    - 创建：添加实体、关系、知识条目
    - 读取：查询、搜索、推理
    - 更新：修改现有知识
    - 删除：移除知识
    """

    def __init__(self):
        # 初始化组件
        self.graph_db = get_graph_db()
        self.vector_db = get_vector_db(dimension=128)
        self.schema = CommonsenseSchema()

        # 元数据
        self.metadata = {
            'created_at': None,
            'last_modified': None,
            'total_entities': 0,
            'total_relations': 0,
            'total_triples': 0
        }

    # ========================================================================
    # 创建操作 (Create)
    # ========================================================================

    def create_entity(self, entity: Entity) -> Tuple[bool, List[str], str]:
        """创建实体

        Args:
            entity: 实体对象

        Returns:
            (is_success, errors, entity_id)
        """
        # 验证实体
        is_valid, errors = self.schema.validate_entity(entity)
        if not is_valid:
            return (False, errors, "")

        # 生成ID（如果没有）
        if not entity.id:
            entity.id = str(uuid.uuid4())

        # 创建图节点
        node = Node(
            id=entity.id,
            labels=[entity.entity_type.value],
            properties=entity.properties
        )

        success = self.graph_db.add_node(node)
        if success:
            self.metadata['total_entities'] += 1
            self._update_metadata('created')

        return (success, [] if success else ["添加节点失败"], entity.id)

    def create_relation(self, relation: Relation) -> Tuple[bool, List[str], str]:
        """创建关系

        Args:
            relation: 关系对象

        Returns:
            (is_success, errors, relation_id)
        """
        # 验证关系
        is_valid, errors = self.schema.validate_relation(relation)
        if not is_valid:
            return (False, errors, "")

        # 生成ID（如果没有）
        if not relation.id:
            relation.id = str(uuid.uuid4())

        # 创建图关系
        rel = Relationship(
            id=relation.id,
            type=relation.relation_type.value,
            source_node_id=relation.subject,
            target_node_id=relation.object,
            properties=relation.properties
        )

        success = self.graph_db.add_relationship(rel)
        if success:
            self.metadata['total_relations'] += 1
            self._update_metadata('created')

        return (success, [] if success else ["添加关系失败"], relation.id)

    def create_knowledge_entry(self, entry: KnowledgeEntry) -> Tuple[bool, List[str], str]:
        """创建知识条目

        Args:
            entry: 知识条目

        Returns:
            (is_success, errors, entry_id)
        """
        # 生成ID
        if not entry.id:
            entry.id = str(uuid.uuid4())

        # 存储三元组到图数据库
        triples = entry.to_triples()
        triple_success = True
        triple_errors = []

        for triple in triples:
            # 确保节点存在
            if triple.subject not in self.graph_db.nodes:
                subject_entity = Entity(
                    id=triple.subject,
                    text=triple.subject,
                    entity_type=EntityType.CONCEPT,
                    confidence=triple.confidence
                )
                self.create_entity(subject_entity)

            if triple.object not in self.graph_db.nodes:
                object_entity = Entity(
                    id=triple.object,
                    text=triple.object,
                    entity_type=EntityType.CONCEPT,
                    confidence=triple.confidence
                )
                self.create_entity(object_entity)

            # 添加三元组
            if not self.graph_db.add_triple(triple):
                triple_success = False
                triple_errors.append(f"添加三元组失败: {triple.subject} {triple.relation} {triple.object}")

        if triple_success:
            self.metadata['total_triples'] += len(triples)
            self._update_metadata('created')

        # 存储向量嵌入（如果有）
        if hasattr(entry, 'embedding') and entry.embedding is not None:
            embedding = VectorEmbedding(
                id=entry.id,
                vector=entry.embedding,
                metadata={'content': entry.content}
            )
            self.vector_db.insert(embedding)

        if triple_errors:
            return (False, triple_errors, entry.id)

        return (True, [], entry.id)

    def batch_create_entries(self, entries: List[KnowledgeEntry]) -> Dict:
        """批量创建知识条目

        Args:
            entries: 知识条目列表

        Returns:
            批量操作结果统计
        """
        results = {
            'total': len(entries),
            'success': 0,
            'failed': 0,
            'errors': []
        }

        for entry in entries:
            success, errors, entry_id = self.create_knowledge_entry(entry)
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
                results['errors'].extend(errors)

        return results

    # ========================================================================
    # 读取操作 (Read)
    # ========================================================================

    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """获取实体

        Args:
            entity_id: 实体ID

        Returns:
            实体信息字典
        """
        node = self.graph_db.get_node(entity_id)
        if node:
            return {
                'id': node.id,
                'labels': node.labels,
                'properties': node.properties
            }
        return None

    def get_relations(self, entity_id: str,
                      relation_type: Optional[str] = None,
                      direction: str = "outgoing") -> List[Dict]:
        """获取实体的关系

        Args:
            entity_id: 实体ID
            relation_type: 关系类型过滤
            direction: 方向 ('outgoing', 'incoming', 'both')

        Returns:
            关系列表
        """
        relationships = self.graph_db.get_relationships(entity_id, direction)

        results = []
        for rel in relationships:
            if relation_type is None or rel.type == relation_type:
                results.append({
                    'id': rel.id,
                    'type': rel.type,
                    'source': rel.source_node_id,
                    'target': rel.target_node_id,
                    'properties': rel.properties
                })

        return results

    def query_knowledge(self, query: str,
                        top_k: int = 10,
                        search_type: str = 'auto') -> QueryResult:
        """知识查询

        Args:
            query: 查询文本
            top_k: 返回top-k个结果
            search_type: 搜索类型 ('vector', 'graph', 'hybrid', 'auto')

        Returns:
            查询结果
        """
        import time
        start_time = time.time()

        results = []
        errors = []

        try:
            if search_type == 'auto':
                # 自动选择搜索类型
                search_type = self._determine_search_type(query)

            if search_type == 'vector':
                # 向量搜索（语义相似度）
                # 1. 将查询转换为向量（简化：使用词向量平均）
                query_vector = self._text_to_vector(query)
                if query_vector is not None:
                    search_results = self.vector_db.search(query_vector, top_k=top_k)

                    for result in search_results:
                        # 获取完整的知识条目
                        entry = self._get_entry_by_id(result.id)
                        if entry:
                            results.append({
                                'id': entry.id,
                                'content': entry.content,
                                'score': result.score,
                                'type': 'vector_search'
                            })

            elif search_type == 'graph':
                # 图搜索（多跳推理）
                # 1. 提取查询实体
                entities = self._extract_entities(query)
                if entities:
                    # 2. 多跳查询
                    for entity in entities:
                        if entity in self.graph_db.nodes:
                            # 获取邻居
                            neighbors = self.graph_db.get_neighbors(entity, max_count=top_k)

                            for rel, neighbor in neighbors:
                                results.append({
                                    'subject': entity,
                                    'relation': rel,
                                    'object': neighbor,
                                    'type': 'graph_search'
                                })

                            # 限制结果数量
                            if len(results) >= top_k:
                                break

            elif search_type == 'hybrid':
                # 混合搜索（向量+图）
                # 先图搜索获取候选，再向量排序
                # 实现较复杂，这里简化为图搜索
                search_type = 'graph'
                # 重新执行搜索

        except Exception as e:
            errors.append(str(e))

        # 按分数排序（如果有）
        if results and 'score' in results[0]:
            results.sort(key=lambda x: -x['score'])

        execution_time = time.time() - start_time

        return QueryResult(
            success=len(errors) == 0,
            results=results[:top_k],
            total_count=len(results),
            execution_time=execution_time,
            errors=errors,
            metadata={'search_type': search_type}
        )

    def find_path(self, source: str, target: str,
                   max_depth: int = 5) -> List[List[str]]:
        """查找两实体间的路径

        Args:
            source: 源实体
            target: 目标实体
            max_depth: 最大深度

        Returns:
            路径列表 [[实体1, 实体2, ...], ...]
        """
        return self.graph_db.find_path(source, target, max_depth)

    # ========================================================================
    # 更新操作 (Update)
    # ========================================================================

    def update_entity(self, entity_id: str,
                       updates: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """更新实体

        Args:
            entity_id: 实体ID
            updates: 更新内容

        Returns:
            (is_success, errors)
        """
        errors = []

        # 检查实体是否存在
        if entity_id not in self.graph_db.nodes:
            return (False, [f"实体不存在: {entity_id}"])

        try:
            node = self.graph_db.nodes[entity_id]

            # 更新属性
            for key, value in updates.items():
                if key == 'labels':
                    node.labels = value
                elif key == 'properties':
                    node.properties.update(value)
                else:
                    errors.append(f"无效的更新字段: {key}")

            self._update_metadata('modified')
            return (len(errors) == 0, errors)

        except Exception as e:
            errors.append(f"更新失败: {str(e)}")
            return (False, errors)

    def update_relation(self, relation_id: str,
                        updates: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """更新关系

        Args:
            relation_id: 关系ID
            updates: 更新内容

        Returns:
            (is_success, errors)
        """
        errors = []

        # 查找关系
        relation = None
        for node_id, rels in self.graph_db.relationships.items():
            for rel in rels:
                if rel.id == relation_id:
                    relation = rel
                    break
            if relation:
                # 更新属性
                for key, value in updates.items():
                    if key == 'type':
                        relation.type = value
                    elif key == 'properties':
                        relation.properties.update(value)
                    else:
                        errors.append(f"无效的更新字段: {key}")

                self._update_metadata('modified')
                return (len(errors) == 0, errors)
            else:
                return (False, [f"关系不存在: {relation_id}"])

    # ========================================================================
    # 删除操作 (Delete)
    # ========================================================================

    def delete_entity(self, entity_id: str) -> Tuple[bool, List[str]]:
        """删除实体

        Args:
            entity_id: 实体ID

        Returns:
            (is_success, errors)
        """
        errors = []

        if entity_id not in self.graph_db.nodes:
            return (False, [f"实体不存在: {entity_id}"])

        try:
            # 删除关系（简化：直接清空该实体的关系）
            if entity_id in self.graph_db.relationships:
                del self.graph_db.relationships[entity_id]

            # 删除节点
            del self.graph_db.nodes[entity_id]

            # 更新统计
            self.metadata['total_entities'] -= 1
            self._update_metadata('modified')

            return (True, [])
        except Exception as e:
            errors.append(f"删除失败: {str(e)}")
            return (False, errors)

    def delete_relation(self, relation_id: str) -> Tuple[bool, List[str]]:
        """删除关系

        Args:
            relation_id: 关系ID

        Returns:
            (is_success, errors)
        """
        errors = []

        try:
            # 查找并删除关系
            for node_id, rels in self.graph_db.relationships.items():
                for i, rel in enumerate(rels):
                    if rel.id == relation_id:
                        rels.pop(i)
                        break

            self.metadata['total_relations'] -= 1
            self._export_metadata('modified')
            return (True, [])
        except Exception as e:
            errors.append(f"删除失败: {str(e)}")
            return (False, errors)

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _determine_search_type(self, query: str) -> str:
        """确定最佳搜索类型"""
        # 简化：根据查询长度和关键词决定
        entities = self._extract_entities(query)

        if len(entities) > 0:
            return 'graph'  # 有实体，用图搜索
        else:
            return 'vector'  # 否则用向量搜索

    def _extract_entities(self, text: str) -> List[str]:
        """提取文本中的实体"""
        import re
        entities = re.findall(r'[一-鿿]{2,6}', text)

        # 过滤停用词
        stopwords = {'的', '了', '在', '是', '和', '有', '与', '被', '将', '把'}
        return [e for e in entities if e not in stopwords]

    def _text_to_vector(self, text: str) -> Optional[np.ndarray]:
        """文本转向量（简化实现）"""
        # 简化：使用随机向量（实际应使用真实的嵌入模型）
        import numpy as np

        # 这里应该调用实际的嵌入模型
        # 暂时使用哈希+随机方式生成伪向量
        hash_val = hash(text) % 1000
        np.random.seed(hash_val)
        vector = np.random.rand(128)

        # 归一化
        vector = vector / np.linalg.norm(vector)
        return vector

    def _get_entry_by_id(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """根据ID获取知识条目（简化实现）"""
        # 简化：从图数据库重建
        # 实际应该有专门的存储
        return None

    def _update_metadata(self, action: str):
        """更新元数据"""
        import datetime
        now = datetime.datetime.now()

        if action == 'created':
            if self.metadata['created_at'] is None:
                self.metadata['created_at'] = now
        elif action == 'modified':
            self.metadata['last_modified'] = now

    def _export_metadata(self, action: str):
        """导出元数据（持久化）"""
        # 实际应该保存到文件或数据库
        pass

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        graph_stats = self.graph_db.get_stats()
        vector_stats = self.vector_db.get_stats()

        return {
            'metadata': self.metadata,
            'graph': graph_stats,
            'vector': vector_stats,
            'total_entities': self.metadata['total_entities'],
            'total_relations': self.metadata['total_relations'],
            'total_triples': self.metadata['total_triples'],
        }

    def export_schema(self) -> Dict:
        """导出Schema定义"""
        exporter = KnowledgeSchemaExporter()
        return exporter.export_schema(self.schema)


# 便捷函数
def get_knowledge_base() -> KnowledgeBaseCRUD:
    """获取知识库CRUD实例"""
    return KnowledgeBaseCRUD()


if __name__ == '__main__':
    print("=== 知识库CRUD操作层 ===")
    print()
    print("功能:")
    print("- 实体/关系/知识条目CRUD")
    print("- 批量操作")
    print("- 查询和搜索")
    print("- 路径查找")
    print("- 验证和约束")
    print()
    print("设计特点:")
    print("- 统一接口")
    print("- Schema验证")
    print("- 错误处理")
    print("- 元数据管理")
