"""
调试常识知识库查询功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.knowledge.commonsense import CommonsenseKB

# 创建知识库
kb = CommonsenseKB()

# 添加测试事实
kb.add_triple("水", "沸点", "100度", 1.0)
kb.add_triple("人", "需要", "氧气", 1.0)
kb.add_triple("杯子", "用于", "喝水", 1.0)

print("=== 调试信息 ===")
print(f"总事实数: {len(kb.facts_index)}")
print(f"概念索引: {dict(list(kb.concept_facts.items())[:5])}")

# 测试概念提取
question = "水在多少度沸腾"
concepts = kb._extract_concepts(question)
print(f"\n问题: {question}")
print(f"提取的概念: {concepts}")

# 查看相关事实
for concept in concepts:
    fact_ids = kb.concept_facts.get(concept, set())
    print(f"概念 '{concept}' 关联的事实ID: {fact_ids}")
    for fid in fact_ids:
        fact = kb.facts_index.get(fid)
        if fact:
            print(f"  - {fact.statement} (类型: {fact.fact_type})")

# 执行查询
results = kb.query(question, top_k=5)
print(f"\n查询结果: {len(results)}条")
for i, r in enumerate(results):
    print(f"{i+1}. {r.statement} (置信度: {r.confidence})")

# 测试另一个问题
question2 = "人需要什么呼吸"
concepts2 = kb._extract_concepts(question2)
print(f"\n问题: {question2}")
print(f"提取的概念: {concepts2}")

results2 = kb.query(question2, top_k=5)
print(f"查询结果: {len(results2)}条")
for i, r in enumerate(results2):
    print(f"{i+1}. {r.statement} (置信度: {r.confidence})")
