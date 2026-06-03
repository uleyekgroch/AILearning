"""
调试种子数据查询
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.knowledge.commonsense import CommonsenseKB

# 加载种子数据
kb = CommonsenseKB()
kb.load_from_file('data/commonsense_seed.json')

print("=== 种子数据分析 ===")
stats = kb.get_stats()
print(f"总事实数: {stats['total_facts']}")
print(f"三元组数: {stats['total_triples']}")
print(f"概念覆盖: {stats['concept_coverage']}")

# 查看物理常识示例
physical_facts = kb.query_by_type('physical', top_k=5)
print(f"\n物理常识示例:")
for i, fact in enumerate(physical_facts[:5]):
    print(f"{i+1}. {fact.statement} (置信度: {fact.confidence})")

# 测试查询
question = "水在多少度沸腾"
print(f"\n查询: {question}")

# 检查概念索引
keywords = [w for w in question if len(w) >= 2 and w.strip()]
print(f"问题关键词: {keywords}")

# 检查事实中是否包含关键词
found_count = 0
for fact_id, fact in kb.facts_index.items():
    if '水' in fact.statement:
        found_count += 1
        if found_count <= 3:
            print(f"包含'水'的事实: {fact.statement}")

print(f"\n共{found_count}条事实包含'水'")

# 执行查询
results = kb.query(question, top_k=5)
print(f"\n查询结果: {len(results)}条")
for i, r in enumerate(results[:3]):
    print(f"{i+1}. {r.statement} (置信度: {r.confidence})")

# 测试另一个查询
question2 = "人需要什么呼吸"
print(f"\n查询: {question2}")
results2 = kb.query(question2, top_k=5)
print(f"查询结果: {len(results2)}条")
for i, r in enumerate(results2[:3]):
    print(f"{i+1}. {r.statement} (置信度: {r.confidence})")
