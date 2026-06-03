"""
阶段1综合测试 - 基础架构搭建验收

测试：
1. 图数据库接口和操作
2. 向量数据库接口和操作
3. 数据模型和Schema验证
4. CRUD操作完整性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_graph_database():
    """测试图数据库"""
    print("=== 测试图数据库 ===")

    from src.knowledge.graph_db import InMemoryGraphDB, Node, Relationship, Triple

    # 创建图数据库
    graph = InMemoryGraphDB()

    # 添加节点
    node1 = Node(id="光合作用", labels=["Process", "Biological"])
    node2 = Node(id="植物", labels=["Organism", "LivingThing"])
    node3 = Node(id="阳光", labels=["Energy", "Physical"])

    assert graph.add_node(node1), "添加节点1失败"
    assert graph.add_node(node2), "添加节点2失败"
    assert graph.add_node(node3), "添加节点3失败"

    print(f"节点数: {graph.stats['node_count']}")

    # 添加关系
    rel1 = Relationship(
        id="rel1",
        type="需要",
        source_node_id="光合作用",
        target_node_id="阳光",
        properties={"essential": True}
    )

    assert graph.add_relationship(rel1), "添加关系失败"

    print(f"关系数: {graph.stats['relationship_count']}")

    # 添加三元组
    triple1 = Triple(subject="植物", relation="进行", object="光合作用", confidence=0.9)
    triple2 = Triple(subject="光合作用", relation="需要", object="阳光", confidence=1.0)

    assert graph.add_triple(triple1), "添加三元组1失败"
    assert graph.add_triple(triple2), "添加三元组2失败"

    print(f"三元组数: {graph.stats['triple_count']}")

    # 测试路径查找
    paths = graph.find_path("植物", "阳光", max_depth=3)
    print(f"路径数: {len(paths)}")
    for path in paths:
        print(f"  路径: {' -> '.join(path)}")

    # 测试多跳查询
    results = graph.multi_hop_query("植物", ["进行"], max_hops=2)
    print(f"多跳查询结果数: {len(results)}")
    for result in results:
        print(f"  {result.subject} {result.relation} {result.object}")

    print("[OK] 图数据库测试通过\n")


def test_vector_database():
    """测试向量数据库"""
    print("=== 测试向量数据库 ===")

    from src.knowledge.vector_db import InMemoryVectorDB, VectorEmbedding
    import numpy as np

    # 创建向量数据库
    vec_db = InMemoryVectorDB(dimension=128)

    # 创建测试向量
    vec1 = np.random.rand(128)
    vec1 = vec1 / np.linalg.norm(vec1)

    vec2 = np.random.rand(128)
    vec2 = vec2 / np.linalg.norm(vec2)

    vec3 = np.random.rand(128)
    vec3 = vec3 / np.linalg.norm(vec3)

    # 插入向量
    emb1 = VectorEmbedding(id="光合作用", vector=vec1, metadata={"type": "process"})
    emb2 = VectorEmbedding(id="植物", vector=vec2, metadata={"type": "organism"})
    emb3 = VectorEmbedding(id="阳光", vector=vec3, metadata={"type": "energy"})

    assert vec_db.insert(emb1), "插入向量1失败"
    assert vec_db.insert(emb2), "插入向量2失败"
    assert vec_db.insert(emb3), "插入向量3失败"

    stats = vec_db.get_stats()
    print(f"向量数: {stats['vector_count']}")

    # 测试相似度搜索
    query_vec = vec1.copy()
    results = vec_db.search(query_vec, top_k=3, metric='cosine')

    print(f"搜索结果数: {len(results)}")
    for res in results:
        print(f"  {res.id}: 相似度={res.score:.3f}")

    # 验证：查询自己应该返回最高相似度
    assert results[0].id == "光合作用", "最相似结果不正确"

    print("[OK] 向量数据库测试通过\n")


def test_schema_validation():
    """测试Schema验证"""
    print("=== 测试Schema验证 ===")

    from src.knowledge.schema import (
        Entity, Relation, KnowledgeEntry, CommonsenseSchema,
        EntityType, RelationType, Triple
    )

    # 创建Schema
    schema = CommonsenseSchema()

    # 测试实体验证
    entity = Entity(
        id="entity1",
        text="水",
        entity_type=EntityType.PHYSICAL_OBJECT,
        properties={"state": "liquid", "boiling_point": 100}
    )

    is_valid, errors = schema.validate_entity(entity)
    print(f"实体验证: {is_valid}")
    if not is_valid:
        print(f"  错误: {errors}")

    assert is_valid, "实体验证失败"

    # 测试关系验证
    relation = Relation(
        id="rel1",
        relation_type=RelationType.CAUSE,
        subject="加热",
        object="沸腾",
        properties={"confidence": 0.9}
    )

    is_valid, errors = schema.validate_relation(relation)
    print(f"关系验证: {is_valid}")
    if not is_valid:
        print(f"  错误: {errors}")

    assert is_valid, "关系验证失败"

    # 测试三元组验证
    is_valid, errors = schema.validate_triple(
        subject="加热",
        relation="导致",
        obj="沸腾",
        subject_type=EntityType.PHYSICAL_OBJECT,
        object_type=EntityType.PHYSICAL_OBJECT
    )

    print(f"三元组验证: {is_valid}")
    if not is_valid:
        print(f"  错误: {errors}")

    assert is_valid, "三元组验证失败"

    # 测试本体验证
    valid = schema.ontology.is_valid_triple(
        subject="水",
        relation="位于",
        obj="杯子",
        subject_type=EntityType.PHYSICAL_OBJECT,
        object_type=EntityType.ARTIFACT
    )

    print(f"本体验证: {valid}")

    # 测试类型推断
    types = schema.ontology.get_inherited_types(EntityType.ANIMAL)
    print(f"动物类型的父类型: {[t.value for t in types]}")
    assert EntityType.ORGANISM in types, "类型推断错误"

    print("[OK] Schema验证测试通过\n")


def test_crud_operations():
    """测试CRUD操作"""
    print("=== 测试CRUD操作 ===")

    from src.knowledge.crud import KnowledgeBaseCRUD
    from src.knowledge.schema import Entity, Relation, KnowledgeEntry, EntityType, RelationType, Triple

    # 创建知识库
    kb = KnowledgeBaseCRUD()

    # 测试创建实体
    entity1 = Entity(
        id="water",
        text="水",
        entity_type=EntityType.PHYSICAL_OBJECT,
        properties={"boiling_point": 100, "state": "liquid"}
    )

    success, errors, entity_id = kb.create_entity(entity1)
    assert success, f"创建实体失败: {errors}"
    print(f"创建实体: {entity_id}")

    # 创建关系的源和目标实体
    entity_heat = Entity(
        id="heat",
        text="加热",
        entity_type=EntityType.PHYSICAL_OBJECT,
        properties={"type": "process"}
    )
    kb.create_entity(entity_heat)

    entity_boil = Entity(
        id="boil",
        text="沸腾",
        entity_type=EntityType.PHYSICAL_OBJECT,
        properties={"type": "process"}
    )
    kb.create_entity(entity_boil)

    # 测试创建关系
    relation1 = Relation(
        id="rel1",
        relation_type=RelationType.CAUSE,
        subject="heat",
        object="boil",
        properties={"confidence": 0.9}
    )

    success, errors, rel_id = kb.create_relation(relation1)
    assert success, f"创建关系失败: {errors}"
    print(f"创建关系: {rel_id}")

    # 测试创建知识条目
    entry1 = KnowledgeEntry(
        id="fact1",
        content="加热导致水沸腾",
        entry_type="fact",
        entities=["heat", "water", "boil"],
        relations=[relation1],
        confidence=0.9
    )

    success, errors, entry_id = kb.create_knowledge_entry(entry1)
    assert success, f"创建知识条目失败: {errors}"
    print(f"创建知识条目: {entry_id}")

    # 测试查询
    result = kb.query_knowledge("水沸腾", top_k=3)
    print(f"查询结果数: {result.total_count}")
    for res in result.results[:3]:
        print(f"  - {res.get('content', res)[:50]}...")

    # 获取统计
    stats = kb.get_stats()
    print(f"\n知识库统计:")
    print(f"  实体数: {stats['total_entities']}")
    print(f"  关系数: {stats['total_relations']}")
    print(f"  三元组数: {stats['total_triples']}")

    print("[OK] CRUD操作测试通过\n")


def test_integration():
    """测试阶段1组件集成"""
    print("=== 测试阶段1组件集成 ===")

    from src.knowledge.crud import KnowledgeBaseCRUD
    import numpy as np

    # 创建知识库
    kb = KnowledgeBaseCRUD()

    # 批量导入常识知识（简化测试数据）
    test_facts = [
        ("水", "沸点", "100度", "physical"),
        ("人", "需要", "氧气", "biological"),
        ("杯子", "用于", "喝水", "functional"),
    ]

    created = 0
    for subject, relation, obj, fact_type in test_facts:
        from src.knowledge.schema import (
            Entity, Relation, KnowledgeEntry, EntityType, RelationType
        )

        # 创建实体
        subject_entity = Entity(
            id=subject,
            text=subject,
            entity_type=EntityType.PHYSICAL_OBJECT
        )
        kb.create_entity(subject_entity)

        object_entity = Entity(
            id=obj,
            text=obj,
            entity_type=EntityType.PHYSICAL_OBJECT
        )
        kb.create_entity(object_entity)

        # 创建关系
        relation_obj = Relation(
            id=f"{subject}_{relation}_{obj}",
            relation_type=RelationType.CAUSE if relation == "导致" else RelationType.USED_FOR,
            subject=subject,
            object=obj
        )
        kb.create_relation(relation_obj)

        # 创建知识条目
        entry = KnowledgeEntry(
            id=f"fact_{created}",
            content=f"{subject}{relation}{obj}",
            entry_type="fact",
            entities=[subject, obj],
            relations=[relation_obj],
            confidence=1.0
        )

        success, _, _ = kb.create_knowledge_entry(entry)
        if success:
            created += 1

    print(f"创建知识条目: {created}/{len(test_facts)}")

    # 测试查询
    result = kb.query_knowledge("水沸腾", top_k=2)
    print(f"查询结果数: {result.total_count}")
    for res in result.results:
        print(f"  - {res.get('content', res)[:50]}...")

    # 验证基本功能
    stats = kb.get_stats()
    assert stats['total_triples'] > 0, "应该有三元组"
    assert stats['total_entities'] > 0, "应该有实体"

    print("\n[OK] 阶段1集成测试通过")


if __name__ == '__main__':
    try:
        test_graph_database()
        test_vector_database()
        test_schema_validation()
        test_crud_operations()
        test_integration()

        print("=" * 50)
        print("[SUCCESS] 阶段1所有测试通过")
        print()
        print("阶段1完成总结:")
        print("- 图数据库接口: [OK]")
        print("- 向量数据库接口: [OK]")
        print("- 数据模型设计: [OK]")
        print("- CRUD操作: [OK]")
        print()
        print("下一步：阶段2 - 图神经推理引擎")
        print()
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
