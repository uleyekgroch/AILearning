"""
简化的think()方法实现

将原有的7+1条路径简化为3条清晰路径：
1. 快速路径：常识库直接查询
2. 推理路径：集成推理引擎
3. 回退路径：概念空间激活扩散
"""

def think_simplified(self, question: str) -> str:
    """思考问题 — 简化版

    推理路径（优先级）：
    1. 快速路径：常识库直接查询（最快，置信度>0.7）
    2. 推理路径：集成推理引擎（多模式融合）
    3. 回退路径：概念空间激活扩散（兜底）

    Args:
        question: 问题文本

    Returns:
        答案文本
    """
    import re

    # 路径1：常识库查询（最快）
    if self.commonsense_kb:
        try:
            facts = self.commonsense_kb.query(question, top_k=1)
            if facts and facts[0].confidence > 0.7:
                return facts[0].statement
        except Exception:
            pass

    # 路径2：集成推理引擎
    try:
        # 延迟初始化集成推理引擎
        if not hasattr(self, '_integrated_reasoning'):
            from src.reasoning.integrated_engine import IntegratedReasoningEngine
            self._integrated_reasoning = IntegratedReasoningEngine(self)

        results = self._integrated_reasoning.reason(question, mode='auto')

        if results and results[0].confidence > 0.3:
            return results[0].answer
    except Exception:
        pass

    # 路径3：概念空间激活扩散（回退）
    return self._think_concept_space_fallback(question)


def _think_concept_space_fallback(self, question: str) -> str:
    """概念空间回退路径

    当常识库和推理引擎都无法回答时，使用概念空间激活扩散。

    Args:
        question: 问题文本

    Returns:
        基于概念激活扩散的答案
    """
    import re

    # 提取问题实体
    entities = re.findall(r'[一-鿿]{2,6}', question)

    if not entities:
        return "我还没有学习到这方面的知识"

    # 过滤停用词
    stopwords = {'的', '了', '在', '是', '和', '有', '与', '被', '将', '把'}
    entities = [e for e in entities if e not in stopwords]

    if not entities:
        return "我还没有学习到这方面的知识"

    # 尝试激活扩散
    try:
        cs = self._safe_registry_get('concept_space')
        if cs and hasattr(cs, 'activate'):
            activated = cs.activate(question, top_k=5, spread_depth=2)

            if activated and activated[0].activation > 0.1:
                # 获取最相关概念
                concepts = [a.concept_id for a in activated[:5] if a.concept_id not in entities[:3]]

                if concepts:
                    # 尝试通过知识图谱获取关系
                    kg = self.knowledge
                    if kg and hasattr(kg, 'get_relations'):
                        relations = []
                        for entity in entities[:2]:
                            rels = kg.get_relations(entity, limit=3)
                            relations.extend(rels)

                        if relations:
                            # 构建答案
                            rel_info = [f"{r.source} {r.relation} {r.target}" for r in relations[:3]]
                            return f"关于{entities[0]}，我了解到：{'、'.join(rel_info[:2])}"

                    # 简单回答
                    return f"关于{entities[0]}，相关的概念包括：{'、'.join(concepts[:3])}"

    except Exception:
        pass

    # 最终回退
    return "我还需要学习更多知识来回答这个问题"


# 使用说明：
# 将上述两个方法添加到 src/core/learner.py 的 Learner 类中
# 替换原有的 think() 方法
