"""替换 _extract_entities_from_repr 方法体"""

NEW_METHOD = '''    def _extract_entities_from_repr(self, text: str, repr: torch.Tensor) -> List[str]:
        """从文本中提取实体 — 统计学习为主通道，正则为兜底

        重构：不再依赖正则滑窗，而是用StatisticalLearner的
        转移概率+PMI检测词边界，产出涌现概念作为实体。
        """
        import re as _re

        entities = []
        source_text = text

        # === 主通道：统计学习涌现概念 ===
        stat = self._registry.get('statistical_learner')
        if stat:
            stat.observe(source_text)
            if hasattr(stat, 'get_emergent_concepts'):
                emergent = stat.get_emergent_concepts(min_freq=1, filter_boundary=True)
                entities = [c.text for c in emergent if hasattr(c, 'text') and len(c.text) >= 2]

        # === 补充通道：感知-预测循环的误差概念 ===
        pll = self._registry.get('perception_learning_loop')
        if pll and hasattr(pll, '_detect_concepts_from_error'):
            try:
                error_concepts = pll._detect_concepts_from_error(source_text)
                for ec in error_concepts:
                    label = ec.get('label', '') if isinstance(ec, dict) else ''
                    if label and label not in entities and len(label) >= 2:
                        entities.append(label)
            except Exception:
                pass

        # === 兜底：正则（仅当统计通道产出不足时）===
        if len(entities) < 2:
            stopwords = set(
                '的了是在我你他她它们这那个有不人大中上下来什么如何怎样'
                '而且还或者而但由于所以因为如果那么但是以为之一一个一些'
                '也能就要会可以被与及其对于到从向把给让比跟最更很已也并'
            )

            # 按标点分割后提取2-6字中文词组
            parts = _re.split(r'[，。！？；：、\\s]', source_text)
            for part in parts:
                part = part.strip()
                if not part or len(part) < 2:
                    continue
                short_words = _re.findall(r'[\\u4e00-\\u9fff]{2,6}', part)
                for w in short_words:
                    if w not in entities and not all(c in stopwords for c in w):
                        entities.append(w)

            # 英文实体
            en_words = _re.findall(r'[A-Z][a-zA-Z]{2,}', source_text)
            entities.extend([e for e in en_words if e not in entities])

        # 去重保序，限制数量
        seen = set()
        result = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                result.append(e)

        # 可选：向量语义过滤（保留原有逻辑作为质量提升）
        if hasattr(self, '_learnable_encoder') and repr is not None and len(result) > 15:
            try:
                with torch.no_grad():
                    candidates = result[:50]
                    ent_sims = []
                    for ent in candidates:
                        ent_repr = self._encode_text(ent)
                        sim = torch.cosine_similarity(repr.unsqueeze(0), ent_repr.unsqueeze(0)).item()
                        ent_sims.append((ent, sim))
                    ent_sims.sort(key=lambda x: x[1], reverse=True)
                    result = [e for e, s in ent_sims[:15]]
                    extra = [e for e, s in ent_sims[15:] if s > -0.5]
                    result.extend(extra[:5])
            except Exception:
                pass

        return result[:20]
'''

with open('src/core/learner.py', 'r', encoding='utf-8') as f:
    content = f.read()

marker_start = '    def _extract_entities_from_repr(self, text: str, repr: torch.Tensor) -> List[str]:'
marker_end = '\n    def _extract_relations_from_repr('

start_idx = content.find(marker_start)
end_idx = content.find(marker_end, start_idx)

if start_idx == -1 or end_idx == -1:
    print("ERROR: Could not find method boundaries")
    exit(1)

new_content = content[:start_idx] + NEW_METHOD + content[end_idx:]

with open('src/core/learner.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"SUCCESS: Replaced _extract_entities_from_repr ({end_idx - start_idx} chars -> {len(NEW_METHOD)} chars)")
