"""
图数据库抽象接口 - Graph Database Abstraction

支持：
- 三元组存储（主语-关系-宾语）
- 图查询（多跳、路径查找）
- 实体/关系索引
- 可替换实现（内存/Neo4j）

设计原则：
- 接口抽象，便于替换底层实现
- 高性能查询支持
- 可扩展到百万级节点
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class Node:
    """图节点"""
    id: str
    labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.labels, str):
            self.labels = [self.labels]


@dataclass
class Relationship:
    """图关系（边）"""
    id: str
    type: str
    source_node_id: str
    target_node_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    direction: str = "outgoing"  # outgoing, incoming, both


@dataclass
class Triple:
    """三元组（主语-关系-宾语）"""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> Tuple[str, str, str]:
        """转换为元组"""
        return (self.subject, self.relation, self.object)

    def __hash__(self):
        return hash((self.subject, self.relation, self.object))

    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return (self.subject == other.subject and
                self.relation == other.relation and
                self.object == other.object)


class GraphDatabase(ABC):
    """
    图数据库抽象接口

    定义所有图数据库必须实现的核心操作
    """

    @abstractmethod
    def add_node(self, node: Node) -> bool:
        """添加节点

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> bool:
        """添加关系

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def add_triple(self, triple: Triple) -> bool:
        """添加三元组（便捷方法）

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]:
        """获取节点"""
        pass

    @abstractmethod
    def get_relationships(self, node_id: str,
                         direction: str = "outgoing") -> List[Relationship]:
        """获取节点的关系"""
        pass

    @abstractmethod
    def find_path(self, source_id: str, target_id: str,
                   max_depth: int = 5) -> List[List[str]]:
        """查找两节点间的路径"""
        pass

    @abstractmethod
    def query(self, cypher: str) -> List[Dict]:
        """执行Cypher查询（如果支持）"""
        pass

    @abstractmethod
    def multi_hop_query(self, entity: str, relation_pattern: List[str],
                       max_hops: int = 3) -> List[Triple]:
        """多跳查询

        Args:
            entity: 起始实体
            relation_pattern: 关系模式（如 ["has_part", "located_in"]）
            max_hops: 最大跳数

        Returns:
            匹配的三元组列表
        """
        pass


class InMemoryGraphDB(GraphDatabase):
    """
    内存图数据库实现

    用于开发和小规模部署
    后续可替换为Neo4j实现
    """

    def __init__(self):
        # 存储节点
        self.nodes: Dict[str, Node] = {}

        # 存储关系
        self.relationships: Dict[str, List[Relationship]] = defaultdict(list)

        # 三元组索引
        self.triples: Set[Triple] = set()
        self.triple_index: Dict[Tuple[str, str, str], Triple] = {}

        # 高级索引
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # entity -> triples
        self.relation_index: Dict[str, Set[Triple]] = defaultdict(set)  # relation -> triples
        self.label_index: Dict[str, Set[str]] = defaultdict(set)  # label -> nodes

        # 统计信息
        self.stats = {
            'node_count': 0,
            'relationship_count': 0,
            'triple_count': 0
        }

    def add_node(self, node: Node) -> bool:
        """添加节点"""
        if node.id in self.nodes:
            return False

        self.nodes[node.id] = node

        # 更新标签索引
        for label in node.labels:
            self.label_index[label].add(node.id)

        self.stats['node_count'] += 1
        return True

    def add_relationship(self, relationship: Relationship) -> bool:
        """添加关系"""
        # 确保源节点和目标节点存在
        if relationship.source_node_id not in self.nodes:
            return False
        if relationship.target_node_id not in self.nodes:
            return False

        self.relationships[relationship.source_node_id].append(relationship)
        self.stats['relationship_count'] += 1
        return True

    def add_triple(self, triple: Triple) -> bool:
        """添加三元组"""
        # 检查是否已存在
        triple_key = triple.to_tuple()
        if triple_key in self.triple_index:
            return False

        # 添加到索引
        self.triple_index[triple_key] = triple
        self.triples.add(triple)

        # 更新实体索引
        self.entity_index[triple.subject].add(triple_key)
        self.entity_index[triple.object].add(triple_key)

        # 更新关系索引
        self.relation_index[triple.relation].add(triple)

        # 确保节点存在（自动创建）
        if triple.subject not in self.nodes:
            self.add_node(Node(id=triple.subject, labels=["Entity"]))
        if triple.object not in self.nodes:
            self.add_node(Node(id=triple.object, labels=["Entity"]))

        # 添加关系
        rel = Relationship(
            id=f"{triple.subject}_{triple.relation}_{triple.object}",
            type=triple.relation,
            source_node_id=triple.subject,
            target_node_id=triple.object,
            properties={'confidence': triple.confidence}
        )
        self.add_relationship(rel)

        self.stats['triple_count'] += 1
        return True

    def get_node(self, node_id: str) -> Optional[Node]:
        """获取节点"""
        return self.nodes.get(node_id)

    def get_relationships(self, node_id: str,
                         direction: str = "outgoing") -> List[Relationship]:
        """获取节点的关系"""
        if direction == "outgoing":
            return self.relationships.get(node_id, [])
        elif direction == "incoming":
            # 收集所有指向该节点的关系
            incoming = []
            for source_id, rels in self.relationships.items():
                for rel in rels:
                    if rel.target_node_id == node_id:
                        incoming.append(rel)
            return incoming
        else:  # both
            outgoing = self.relationships.get(node_id, [])
            incoming = []
            for source_id, rels in self.relationships.items():
                for rel in rels:
                    if rel.target_node_id == node_id:
                        incoming.append(rel)
            return outgoing + incoming

    def find_path(self, source_id: str, target_id: str,
                   max_depth: int = 5) -> List[List[str]]:
        """查找路径（BFS）"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return []

        # BFS搜索
        from collections import deque
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        paths = []

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_depth + 1:
                continue

            if current_id == target_id:
                paths.append(path)
                continue

            # 探索邻居
            for rel in self.get_relationships(current_id, "outgoing"):
                neighbor = rel.target_node_id
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return paths

    def query(self, cypher: str) -> List[Dict]:
        """执行简化查询（暂不支持完整Cypher）"""
        # 简化：只支持基本匹配
        # 实际生产应使用Neo4j
        raise NotImplementedError("完整Cypher查询需要Neo4j支持")

    def multi_hop_query(self, entity: str, relation_pattern: List[str],
                       max_hops: int = 3) -> List[Triple]:
        """多跳查询"""
        if entity not in self.nodes:
            return []

        # BFS多跳搜索
        from collections import deque
        queue = deque([(entity, 0, [])])  # (当前实体, 当前跳数, 路径)
        visited = {entity}
        results = []

        while queue:
            current_entity, hop_count, path = queue.popleft()

            if hop_count >= max_hops:
                continue

            # 获取所有出边
            for rel in self.get_relationships(current_entity, "outgoing"):
                # 检查关系是否匹配模式
                if hop_count < len(relation_pattern):
                    if rel.type != relation_pattern[hop_count]:
                        continue

                target = rel.target_node_id
                if target not in visited:
                    visited.add(target)

                    # 构造三元组
                    if path:
                        # 从路径重建三元组
                        for i, (s, r, o) in enumerate(path):
                            if i == hop_count - 1:
                                triple = Triple(s, r, o)
                                results.append(triple)

                    # 添加到队列
                    queue.append((target, hop_count + 1, path + [(current_entity, rel.type, target)]))

        return results

    def get_neighbors(self, entity: str, relation_type: Optional[str] = None,
                      max_count: int = 100) -> List[Tuple[str, str]]:
        """获取邻居节点

        Args:
            entity: 实体ID
            relation_type: 关系类型过滤
            max_count: 最大返回数量

        Returns:
            [(relation, object), ...]列表
        """
        neighbors = []

        for rel in self.get_relationships(entity, "outgoing"):
            if relation_type is None or rel.type == relation_type:
                neighbors.append((rel.type, rel.target_node_id))
                if len(neighbors) >= max_count:
                    break

        return neighbors

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'node_count': self.stats['node_count'],
            'relationship_count': self.stats['relationship_count'],
            'triple_count': self.stats['triple_count'],
            'entity_index_size': len(self.entity_index),
            'relation_index_size': len(self.relation_index),
        }


class GraphDBFactory:
    """图数据库工厂"""

    _instance = None

    @classmethod
    def get_instance(cls, backend: str = 'memory') -> GraphDatabase:
        """获取图数据库实例

        Args:
            backend: 后端类型 ('memory', 'neo4j')

        Returns:
            图数据库实例
        """
        if cls._instance is None:
            if backend == 'memory':
                cls._instance = InMemoryGraphDB()
            elif backend == 'neo4j':
                # 后续实现
                raise NotImplementedError("Neo4j backend尚未实现")
            else:
                raise ValueError(f"Unknown backend: {backend}")

        return cls._instance

    @classmethod
    def reset(cls):
        """重置实例（用于测试）"""
        cls._instance = None


# 便捷函数
def get_graph_db() -> GraphDatabase:
    """获取图数据库实例"""
    return GraphDBFactory.get_instance()


if __name__ == '__main__':
    print("=== 图数据库抽象接口 ===")
    print()
    print("功能:")
    print("- 节点和关系管理")
    print("- 三元组索引")
    print("- 多跳查询")
    print("- 路径查找")
    print("- 可扩展架构")
    print()
    print("实现:")
    print("- InMemoryGraphDB (当前)")
    print("- Neo4jGraphDB (后续)")
