"""
GNN推理引擎测试脚本

测试：
1. 图构建
2. 模型训练
3. 链接预测
4. 路径推理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_gnn_basic():
    """测试GNN基础功能"""
    print("=== 测试GNN基础功能 ===")

    from src.reasoning.gnn_engine import GNNReasoningEngine, GNNConfig

    # 创建配置（简化：少epoch）
    config = GNNConfig(
        input_dim=64,
        hidden_dim=128,
        output_dim=64,
        num_layers=2,
        num_heads=4,
        epochs=20,  # 简化测试
        learning_rate=0.01
    )

    # 创建引擎
    engine = GNNReasoningEngine(config)
    print(f"使用设备: {engine.device}")

    # 测试三元组
    triples = [
        ("人", "需要", "氧气"),
        ("人", "需要", "水"),
        ("人", "需要", "食物"),
        ("植物", "需要", "光合作用"),
        ("植物", "需要", "水分"),
        ("动物", "需要", "食物"),
        ("水", "沸点", "100度"),
        ("加热", "导致", "沸腾"),
        ("运动", "导致", "出汗"),
        ("下雨", "导致", "地面湿"),
    ]

    # 构建图
    graph = engine.build_graph(triples)
    print(f"图构建完成:")
    print(f"  节点数: {graph.num_nodes}")
    print(f"  边数: {graph.num_edges}")
    print(f"  节点: {list(graph.node_to_idx.keys())}")

    # 训练模型
    engine.train(graph)

    # 测试推理
    print("\n=== 测试链接预测推理 ===")
    queries = ["人", "植物", "加热"]

    for query in queries:
        print(f"\n查询: {query}")
        results = engine.reason(query, top_k=3)

        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. {result['target']} (概率: {result['probability']:.3f})")

    # 测试路径推理
    print("\n=== 测试路径推理 ===")
    paths = engine.find_reasoning_path("人", "水", max_hops=3)
    print(f"从'人'到'水'的路径 ({len(paths)}条):")

    for i, path in enumerate(paths[:3], 1):
        print(f"  {i}. {' -> '.join(path)}")

    print("\n[OK] GNN基础功能测试通过\n")


def test_gnn_integration():
    """测试GNN与知识库集成"""
    print("=== 测试GNN与知识库集成 ===")

    from src.knowledge.crud import KnowledgeBaseCRUD
    from src.reasoning.gnn_engine import GNNReasoningEngine, GNNConfig
    from src.knowledge.schema import Entity, Relation, KnowledgeEntry, EntityType, RelationType

    # 创建知识库
    kb = KnowledgeBaseCRUD()

    # 添加测试知识
    test_data = [
        ("水", "PHYSICAL_OBJECT", "沸点", "100度", "物理属性"),
        ("人", "PERSON", "需要", "氧气", "生物需求"),
        ("加热", "PHYSICAL_OBJECT", "导致", "沸腾", "因果关系"),
        ("运动", "PHYSICAL_OBJECT", "导致", "出汗", "因果关系"),
    ]

    created = 0
    for subj, subj_type, rel, obj, obj_type in test_data:
        # 创建实体
        subj_entity = Entity(
            id=subj, text=subj,
            entity_type=EntityType[subj_type]
        )
        kb.create_entity(subj_entity)

        obj_entity = Entity(
            id=obj, text=obj,
            entity_type=EntityType.PHYSICAL_OBJECT
        )
        kb.create_entity(obj_entity)

        # 创建关系
        relation_obj = Relation(
            id=f"{subj}_{rel}_{obj}",
            relation_type=RelationType.CAUSE if rel == "导致" else RelationType.USED_FOR,
            subject=subj,
            object=obj
        )
        kb.create_relation(relation_obj)

        created += 1

    print(f"创建知识条目: {created}")

    # 获取三元组
    triples = []
    for subj, _, rel, obj, _ in test_data:
        triples.append((subj, rel, obj))

    # 创建GNN引擎
    config = GNNConfig(
        input_dim=64,
        hidden_dim=128,
        output_dim=64,
        num_layers=2,
        epochs=15  # 简化测试
    )
    engine = GNNReasoningEngine(config)

    # 构建图并训练
    graph = engine.build_graph(triples)
    engine.train(graph)

    # 测试推理
    print("\n集成推理测试:")
    results = engine.reason("加热", top_k=3)
    for r in results[:3]:
        print(f"  {r['target']}: {r['probability']:.3f}")

    print("\n[OK] GNN集成测试通过\n")


if __name__ == '__main__':
    try:
        test_gnn_basic()
        test_gnn_integration()

        print("=" * 50)
        print("[SUCCESS] GNN引擎所有测试通过")
        print()
        print("GNN引擎功能总结:")
        print("- 图构建: [OK]")
        print("- 模型训练: [OK]")
        print("- 链接预测: [OK]")
        print("- 路径推理: [OK]")
        print("- 知识库集成: [OK]")
        print()
        print("下一步：阶段3 - 概率推理系统")
        print()
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
