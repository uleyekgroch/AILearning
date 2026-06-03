"""
调试类型层次问题
"""

from src.knowledge.schema import EntityType, Ontology

# 创建本体
ontology = Ontology()

# 打印类型层次
print("=== 类型层次结构 ===")
for type_name, parents in ontology.type_hierarchy.items():
    print(f"{type_name} -> {parents}")

print("\n=== 测试get_inherited_types ===")

# 测试ANIMAL
print(f"\n查询 EntityType.ANIMAL")
print(f"  - name: {EntityType.ANIMAL.name}")
print(f"  - value: {EntityType.ANIMAL.value}")

# 检查type_hierarchy中是否有"Animal"
print(f"\ntype_hierarchy.get('Animal', set()): {ontology.type_hierarchy.get('Animal', set())}")

# 手动检查parent查找
parent = "Organism"
print(f"\n查找parent: {parent}")
found = False
for et in EntityType:
    if et.value == parent:
        print(f"  找到匹配: {et.name} = {et.value}")
        found = True
        break
if not found:
    print(f"  没有找到value为'{parent}'的EntityType")

# 调用get_inherited_types
types = ontology.get_inherited_types(EntityType.ANIMAL)
print(f"\nget_inherited_types(EntityType.ANIMAL) 返回:")
for t in types:
    print(f"  - {t.name} ({t.value})")

# 检查断言
print(f"\nEntityType.ORGANISM in types: {EntityType.ORGANISM in types}")
