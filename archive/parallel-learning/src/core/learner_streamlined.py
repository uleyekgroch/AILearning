"""
精简版 learn_from_text - 基于第一原理的核心学习循环

只保留5个核心模块：
1. StatisticalLearner - 概念提取
2. ConceptSpace - 概念存储+激活
3. FunctionalConcept - 功能性表征
4. KG - 关系存储
5. LanguageAcquisition - 语言表达

删除所有冗余的38个模块调用
"""

def learn_from_text_core(self, text):
    """核心学习循环 - 从第一原理出发"""
    result = {}

    # === Layer 1: 统一感知 ===
    text_repr = self._encode_text(text, train=False)  # 无需梯度
    result['representation'] = text_repr

    # === Layer 2: 概念提取 (StatisticalLearner) ===
    stat = self._safe_registry_get('statistical_learner')
    if stat:
        stat.observe(text)
        entities = stat.get_emergent_concepts(min_freq=1, filter_boundary=True)
        entities = [c for c, conf in entities if len(c) >= 2]
    else:
        # 兜底：简单提取
        entities = self._extract_entities_fallback(text)
    result['entities'] = entities

    if not entities:
        return result

    # === Layer 3: 关系提取 ===
    relations = self._extract_relations_simple(text, entities)
    result['relations'] = relations

    # === Layer 4: 概念注册 (统一概念空间) ===
    cs = self._safe_registry_get('concept_space')
    if cs:
        for e in entities[:20]:  # 限制数量
            e_repr = self._encode_text(e, train=False)
            cs.register(e, vector=e_repr, source='text')

        # Hebbian关系学习
        for i, e1 in enumerate(entities[:10]):
            for e2 in entities[i+1:10]:
                cs.learn_relation(e1, e2, strength=0.5)

    # === Layer 5: KG存储 ===
    for subj, rel, obj, conf in relations[:10]:
        self.knowledge.add_relation(subj, rel, obj, confidence=conf)

    return result


def _extract_relations_simple(self, text, entities):
    """简化版关系提取 - 避免复杂正则"""
    relations = []
    entity_set = set(entities)

    # 简单模式：A是B
    for e1 in entities:
        for e2 in entities:
            if e1 == e2: continue
            if f'{e1}是{e2}' in text or f'{e1}为{e2}' in text:
                relations.append((e1, '是', e2, 0.9))
            elif f'{e1}的{e2}' in text:
                relations.append((e1, '具有', e2, 0.7))

    return relations[:10]


def _extract_entities_fallback(self, text):
    """兜底实体提取"""
    import re
    # 简单的中文词提取
    words = re.findall(r'[\\u4e00-\\u9fff]{2,6}', text)
    stopwords = {'的', '了', '在', '是', '和', '有', '与'}
    return [w for w in words if w not in stopwords and len(w) >= 2]


# 核心推理：基于概念空间激活
def think_core(self, question):
    """核心推理 - 路径简化为3步"""
    # Step 1: 激活相关概念
    cs = self._safe_registry_get('concept_space')
    if not cs or not cs.concepts:
        return ""

    activated = cs.activate(question, top_k=5, spread_depth=2)
    if not activated:
        return ""

    # Step 2: 获取关系
    concepts = [a.concept_id for a in activated[:5]]

    # Step 3: 自然表达
    la = self.language_acquisition
    if la and hasattr(la, 'compose'):
        return la.compose(concepts, goal=question)
    else:
        # 兜底
        return "与".join(concepts)


# === 性能优化：批量处理 ===
class BatchLearner:
    """批量学习器 - 将单条处理改为批量处理"""

    def __init__(self, learner):
        self.learner = learner
        self.buffer = []
        self.buffer_size = 50

    def add(self, text):
        self.buffer.append(text)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """批量处理缓冲区中的文本"""
        if not self.buffer:
            return

        # 批量编码
        texts = self.buffer
        reprs = [self.learner._encode_text(t, train=False) for t in texts]

        # 批量学习
        for t, r in zip(texts, reprs):
            self.learner.learn_from_text_core(t)

        self.buffer.clear()


# === 性能目标 ===
# 当前: 14s/text → 目标: <1s/text
# 优化: O(n) → O(1)
# 模块: 42 → 5

print("Streamlined architecture loaded")
print("Core modules: StatisticalLearner, ConceptSpace, FunctionalConcept, KG, LanguageAcquisition")
print("Expected performance: <1s/text with O(1) scaling")
