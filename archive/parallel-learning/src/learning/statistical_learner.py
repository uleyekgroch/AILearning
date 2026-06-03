"""统计学习器 — 从反复体验中让概念涌现

核心思想（认知科学基础）：
==========================

1. 不"提取"概念，概念从统计规律中"涌现"
   - Saffran(1996): 8个月婴儿仅听2分钟无意义音节流，
     就能通过统计相邻音节的转移概率发现"词"的边界
   - 系统观察N条文本后，反复出现的片段自动成为概念候选

2. 不"提取"关系，关系从共现统计中涌现
   - 人脑中没有预定义的"导致"、"属于"关系类型
   - 如果"下雨"和"地面湿"总是一起出现，它们之间就自然形成了关联
   - 这种关联不是三元组(subject, relation, object)，而是概念的共现强度

3. 学习方式：
   - 单条文本：只更新统计计数，不产生任何概念
   - 多次观察后：超过阈值的片段"涌现"为概念
   - 高频共现的概念对自动形成关联

4. 与正则提取的本质区别：
   - 正则：看到"数学是Y" → 立即提取三元组(数学, 是, Y)
   - 统计：看到10条包含"数学"的文本 → "数学"频率超过阈值 → 涌现为概念
   - 正则对噪音敏感（一次误匹配就出错），统计对噪音鲁棒（需要多次确认）

实现核心：
- n-gram 频率统计（2-gram, 3-gram, 4-gram）
- 点互信息(PMI)计算：衡量两个片段一起出现是否比随机更频繁
- 转移概率：给定上下文，预测下一个概念
- 概念涌现：频率+PMI超过阈值时，片段成为概念
- 关系涌现：概念间的高PMI共现自动形成关联
"""

import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict


@dataclass
class ConceptCandidate:
    """概念候选 — 尚未达到涌现阈值的片段"""
    text: str
    frequency: int = 0
    total_contexts: int = 0  # 出现在多少条不同文本中
    contexts: List[str] = field(default_factory=list)  # 上下文片段
    pmi: float = 0.0  # 点互信息
    last_seen_idx: int = 0  # 最后一次被观察到的文本序号

    @property
    def is_emergent(self) -> bool:
        """是否已达到涌现阈值"""
        return self.frequency >= 3 and self.total_contexts >= 2

    def confidence(self) -> float:
        """概念的可信度（基于频率和跨上下文出现）"""
        if self.frequency < 2:
            return 0.0
        # 频率的对数增长 + 跨上下文加成
        freq_score = min(math.log(self.frequency + 1) / math.log(10), 1.0)
        context_score = min(self.total_contexts / 5.0, 1.0)
        return (freq_score * 0.6 + context_score * 0.4)


@dataclass
class CooccurrenceRelation:
    """共现关系 — 两个概念一起出现的统计证据"""
    concept_a: str
    concept_b: str
    cooccurrence_count: int = 0
    pmi: float = 0.0  # 点互信息
    contexts: List[str] = field(default_factory=list)

    @property
    def strength(self) -> float:
        """关系强度"""
        if self.pmi <= 0:
            return 0.0
        return min(self.pmi / 5.0, 1.0)


class StatisticalLearner:
    """统计学习器 — 从反复体验中让概念涌现

    使用方式：
        learner = StatisticalLearner()

        # 反复观察文本
        for text in corpus:
            learner.observe(text)

        # 获取涌现的概念和关系
        concepts = learner.get_emergent_concepts()
        relations = learner.get_emergent_relations()

        # 预测
        predictions = learner.predict_next("数学是研究")
    """

    def __init__(self, max_ngram: int = 4, min_freq: int = 3,
                 min_pmi: float = 1.0, max_concepts: int = 5000):
        """
        Args:
            max_ngram: 最大n-gram长度
            min_freq: 概念涌现的最低频率阈值
            min_pmi: 关系涌现的最低PMI阈值
            max_concepts: 最多保留的概念候选数
        """
        self.max_ngram = max_ngram
        self.min_freq = min_freq
        self.min_pmi = min_pmi
        self.max_concepts = max_concepts

        # 核心统计
        self._char_freq: Dict[str, int] = defaultdict(int)          # 单字符频率
        self._ngram_freq: Dict[str, int] = defaultdict(int)          # n-gram频率
        self._ngram_contexts: Dict[str, Set[int]] = defaultdict(set) # n-gram出现在哪些文本中

        # 共现统计
        self._cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        # 转移概率（用于预测）
        self._transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._transition_totals: Dict[str, int] = defaultdict(int)

        # 已涌现的概念和关系
        self._concepts: Dict[str, ConceptCandidate] = {}
        self._relations: Dict[Tuple[str, str], CooccurrenceRelation] = {}

        # 文本计数
        self._text_count = 0
        self._total_chars = 0

        # 中文停用词（只保留真正的虚词，不像正则那样过滤内容词）
        self._stop_chars = set('的了是在我你他她它们这那个有不为上下来也就要会能被与及其到从向把给让比跟很已并')

    def observe(self, text: str) -> Dict:
        """观察一条文本，更新所有统计

        核心操作：
        1. 分词为字符序列
        2. 统计所有n-gram频率
        3. 更新共现统计
        4. 更新转移概率
        5. 检查是否有新概念涌现

        Returns:
            统计快照：新涌现的概念、新涌现的关系等
        """
        if not text or len(text.strip()) < 2:
            return {'new_concepts': [], 'new_relations': []}

        self._text_count += 1
        text_id = self._text_count

        # 1. 预处理：提取中文和英文片段
        segments = self._segment(text)
        self._total_chars += sum(len(s) for s in segments)

        # 2. 统计n-gram频率
        for segment in segments:
            self._count_ngrams(segment, text_id)

        # 3. 统计共现（同一文本中的n-gram互相共现）
        all_ngrams_in_text = set()
        for segment in segments:
            for n in range(2, self.max_ngram + 1):
                for i in range(len(segment) - n + 1):
                    gram = segment[i:i + n]
                    if self._is_valid_ngram(gram):
                        all_ngrams_in_text.add(gram)

        # 更新共现（限制对数避免O(n^2)膨胀）
        ngram_list = list(all_ngrams_in_text)
        if len(ngram_list) > 30:
            import random
            ngram_list = random.sample(ngram_list, 30)
        for i in range(len(ngram_list)):
            for j in range(i + 1, len(ngram_list)):
                a, b = ngram_list[i], ngram_list[j]
                # 避免包含关系的n-gram互相共现（如"数学"和"数学家"）
                if a in b or b in a:
                    continue
                key = (min(a, b), max(a, b))
                self._cooccurrence[key] += 1

        # 4. 更新转移概率
        for segment in segments:
            if len(segment) < 2:
                continue
            for i in range(len(segment) - 1):
                current = segment[i]
                next_char = segment[i + 1]
                self._transitions[current][next_char] += 1
                self._transition_totals[current] += 1

        # 5. 检查概念涌现
        new_concepts = self._check_emergence(all_ngrams_in_text)

        # Phase 7: 新概念涌现后，标记需要重新清理碎片
        if new_concepts:
            self._cleanup_done = False

        # 6. 检查关系涌现
        new_relations = self._check_relation_emergence()

        return {
            'new_concepts': new_concepts,
            'new_relations': new_relations,
            'total_texts': self._text_count,
            'total_ngrams': len(self._ngram_freq),
            'candidate_concepts': len(self._concepts),
        }

    def _post_emergence_cleanup(self):
        """涌现后碎片清理（Phase 7 — 替代原 Strategy E）

        核心原理：
        所有概念先自由涌现，然后一次性扫描清理碎片。
        关键改进（vs 原 Strategy E）：
        1. 在所有涌现完成后执行 → 无顺序依赖
        2. 从长到短处理 → 先清长碎片，再清短碎片（碎片不互为父概念）
        3. 多父概念计数只计有效父概念（排除已标记碎片和功能词碎片）
        4. 对无父概念的情况，检查 _ngram_freq 中的更长n-gram

        判断规则：
        1. 内部子串（非边缘）且有效父概念≤1个 → 碎片
           例："究人类" ⊂ "研究人类"，只此一个有效父 → 碎片
        2. 边缘子串 + 频率比重叠 + 余部无意义 → 碎片
           例："物理" ⊂ "物理学"，余部"学"是常见后缀 → 保留
           例："研究人" ⊂ "研究人类"，余部"类"非词非后缀 → 碎片
        3. 多有效父概念（≥2）→ 保留（有独立用法）
           例："理学" ⊂ "物理学"/"心理学"/"热力学" → 保留
        4. 无父概念但 _ngram_freq 中有更长n-gram → 检查频率比
           例："动规律" ⊂ "运动规律"(ngram_freq) → 频率比≈1.0 → 碎片
        """
        if not hasattr(self, '_concepts') or not self._concepts:
            return

        function_chars = set('是的有在了和与被把让给从到以也而')
        # 常见中文词缀（学科后缀、名词后缀等）
        common_suffixes = set('学论家者性化力度量')
        concept_list = list(self._concepts.keys())
        to_remove = set()

        def _is_valid_parent(text):
            """检查一个概念是否可作为有效的父概念（非碎片）"""
            if text in to_remove:
                return False
            if text[0] in function_chars or text[-1] in function_chars:
                return False
            if len(text) >= 3 and any(ch in function_chars for ch in text[1:-1]):
                return False
            return True

        # 从长到短处理：先处理长概念，短的碎片无法被已标记碎片保护
        sorted_concepts = sorted(concept_list, key=len, reverse=True)

        for gram in sorted_concepts:
            if len(gram) < 2:
                continue
            if gram[0] in function_chars or gram[-1] in function_chars:
                continue  # 已被其他策略过滤

            gram_freq = self._ngram_freq.get(gram, 0)
            if gram_freq == 0:
                continue

            # 收集有效父概念（排除已标记碎片和功能词碎片）
            valid_parents = []
            for other in concept_list:
                if len(other) <= len(gram) or gram not in other:
                    continue
                if not _is_valid_parent(other):
                    continue
                if self._concepts[other].frequency == 0:
                    continue
                valid_parents.append(other)

            # 路径A: 有有效父概念 → 频率比+词边界检查
            if valid_parents:
                self._check_parents(gram, gram_freq, valid_parents,
                                    concept_list, function_chars,
                                    common_suffixes, to_remove)
                continue

            # 路径B: 无有效父概念 → 检查 _ngram_freq 中的更长n-gram
            self._check_ngram_parents(gram, gram_freq, function_chars, to_remove)

        for f in to_remove:
            if f in self._concepts:
                del self._concepts[f]

    def _check_parents(self, gram, gram_freq, valid_parents,
                       concept_list, function_chars, common_suffixes,
                       to_remove):
        """路径A: 基于有效父概念的碎片检测

        两阶段决策：
        Phase 1 — 搜寻 KEEP 信号：任何父概念显示词边界 → 立即保留
        Phase 2 — 判定碎片：无 KEEP 信号时，多父概念(≥2)保留，单父概念删除
        """
        # Phase 1a: 构词法保护
        # 以常见学科/属性后缀结尾的概念（X学/X论/X家等）通常是合法复合词
        productive_suffixes = set('学论家者性化力度量')
        if len(gram) >= 2 and gram[-1] in productive_suffixes:
            return  # KEEP: 符合构词法规律（如"生物学"/"力学"/"科学"）

        # Phase 1b: 搜寻词边界 KEEP 信号
        for longer in valid_parents:
            longer_freq = self._concepts[longer].frequency
            ratio = gram_freq / longer_freq
            if not (0.6 <= ratio <= 1.4):
                continue

            pos = longer.index(gram)
            is_edge = (pos == 0 or pos + len(gram) == len(longer))

            if is_edge:
                if pos == 0:
                    suffix = longer[len(gram):]
                    if (suffix in concept_list or
                            (len(suffix) == 1 and suffix in common_suffixes)):
                        return  # KEEP: 词边界确认
                else:
                    prefix = longer[:pos]
                    if (prefix in concept_list or
                            (len(prefix) == 1 and prefix in common_suffixes)):
                        return  # KEEP: 词边界确认

        # Phase 2: 无 KEEP 信号 → 碎片判定
        # 多父概念(≥2)：出现在多个不同上下文中 → 有独立用法 → 保留
        # 例："理学" ⊂ "物理学"/"心理学"/"热力学" → 3个父 → 保留
        if len(valid_parents) >= 2:
            return  # KEEP: 多父概念暗示独立用法

        # 单父概念 + 频率重叠 → 碎片
        for longer in valid_parents:
            longer_freq = self._concepts[longer].frequency
            ratio = gram_freq / longer_freq
            if 0.6 <= ratio <= 1.4:
                to_remove.add(gram)
                return

    def _check_ngram_parents(self, gram, gram_freq, function_chars, to_remove):
        """路径B: 基于 _ngram_freq 的碎片检测（无 _concepts 父概念时使用）

        当概念没有更长的已涌现概念作为父概念时，检查 n-gram 频率表中的
        更长 n-gram，判断该概念是否只是某个更长 n-gram 的子串碎片。

        关键：使用所有有效父n-gram的频率之和计算ratio，
        因为gram可能分散在多个不同父n-gram中。
        例："动规律" ⊂ "运动规律"(freq=2) + "活动规律"(freq=1)
            → 总父freq=3, ratio=3/3=1.0 → 碎片
        """
        # 构词法保护：以常见学科/属性后缀结尾的概念通常是合法复合词
        # "科学"/"化学"/"力学"/"物理" 等不应仅因n-gram子串分析而被删除
        productive_suffixes = set('学论家者性化力度量')
        if len(gram) >= 2 and gram[-1] in productive_suffixes:
            return  # KEEP: 符合构词法规律

        # 收集有效的父n-gram（限制数量避免O(n)扫描）
        valid_ngram_parents = []
        checked = 0
        max_parents = 100  # 最多检查100个父候选
        for ngram_text, ngram_freq in self._ngram_freq.items():
            if len(ngram_text) <= len(gram) or gram not in ngram_text:
                continue
            if ngram_freq == 0:
                continue
            # 验证父n-gram本身不是碎片
            if ngram_text[0] in function_chars or ngram_text[-1] in function_chars:
                continue
            if len(ngram_text) >= 3 and any(ch in function_chars for ch in ngram_text[1:-1]):
                continue
            valid_ngram_parents.append((ngram_text, ngram_freq))
            checked += 1
            if checked >= max_parents:
                break

        if not valid_ngram_parents:
            return  # 没有父n-gram → 无法判定

        # 计算总父频率
        total_parent_freq = sum(freq for _, freq in valid_ngram_parents)
        if total_parent_freq == 0:
            return

        ratio = gram_freq / total_parent_freq

        # ratio ≈ 1.0 意味着 gram 几乎总是作为某个更长n-gram的子串出现
        if 0.7 <= ratio <= 1.3:
            # 检查位置：gram是否在所有父n-gram中都是内部位置
            all_internal = True
            for ngram_text, _ in valid_ngram_parents:
                pos = ngram_text.index(gram)
                is_edge = (pos == 0 or pos + len(gram) == len(ngram_text))
                if is_edge:
                    all_internal = False
                    break

            if all_internal:
                to_remove.add(gram)
                return

            # 边缘位置 + 频率完全重叠 → 也可能是碎片
            if 0.85 <= ratio <= 1.15:
                to_remove.add(gram)
                return

    def get_emergent_concepts(self, min_freq: int = None,
                              min_contexts: int = 2,
                              filter_boundary: bool = True) -> List[Tuple[str, float]]:
        """获取所有已涌现的概念

        Args:
            filter_boundary: 是否用词边界检测过滤跨词碎片

        Returns:
            List of (concept_text, confidence) 按confidence降序排列
        """
        # 懒执行cleanup：仅在概念数超过500时才执行（批量学习优先，碎片可接受）
        if not getattr(self, '_cleanup_done', False):
            if len(self._concepts) >= 500:
                self._post_emergence_cleanup()
                self._cleanup_done = True

        min_freq = min_freq or self.min_freq
        results = []
        for text, candidate in self._concepts.items():
            if candidate.frequency < min_freq:
                continue
            if candidate.total_contexts < min_contexts:
                continue
            # 词边界过滤：排除跨越语法边界的碎片
            if filter_boundary and not self._is_complete_word(text):
                continue
            results.append((text, candidate.confidence()))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_emergent_relations(self, min_pmi: float = None,
                               min_cooccurrence: int = 2) -> List[Tuple[str, str, float]]:
        """获取所有已涌现的关系

        Returns:
            List of (concept_a, concept_b, strength) 按strength降序排列
        """
        min_pmi = min_pmi or self.min_pmi
        results = []
        for (a, b), rel in self._relations.items():
            if rel.pmi >= min_pmi and rel.cooccurrence_count >= min_cooccurrence:
                results.append((a, b, rel.strength))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def predict_next(self, context: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """给定上下文，预测接下来最可能出现的字符

        基于转移概率的预测 — 这是"预期"的基础。
        人脑持续预测下一步会发生什么，预测误差驱动学习。

        Args:
            context: 上下文字符串
            top_k: 返回top-k预测

        Returns:
            List of (predicted_char, probability)
        """
        if not context:
            return []

        last_char = context[-1]
        if last_char not in self._transitions:
            return []

        total = self._transition_totals[last_char]
        if total == 0:
            return []

        probs = []
        for char, count in self._transitions[last_char].items():
            prob = count / total
            probs.append((char, prob))

        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_k]

    def get_concept_info(self, concept: str) -> Optional[ConceptCandidate]:
        """获取概念的详细信息"""
        return self._concepts.get(concept)

    def get_related(self, concept: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """获取与指定概念最相关的其他概念

        基于共现强度，不是基于正则模式匹配。
        """
        results = []
        for (a, b), rel in self._relations.items():
            if a == concept:
                results.append((b, rel.strength))
            elif b == concept:
                results.append((a, rel.strength))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_surprise(self, text: str) -> float:
        """计算文本的"意外度"（预测误差）

        如果文本中的字符转移概率都很高 → 低意外度（符合预期）
        如果文本中有很多低概率的转移 → 高意外度（出乎意料）

        高意外度的文本应该获得更多注意力（好奇心驱动学习）
        """
        if self._text_count < 5:
            return 1.0  # 观察太少，一切都是"意外的"

        segments = self._segment(text)
        total_surprise = 0.0
        count = 0

        for segment in segments:
            for i in range(len(segment) - 1):
                current = segment[i]
                next_char = segment[i + 1]

                # 计算转移概率
                total = self._transition_totals.get(current, 0)
                if total == 0:
                    surprise = 1.0  # 未见过的上下文 → 最大意外
                else:
                    count_trans = self._transitions.get(current, {}).get(next_char, 0)
                    prob = count_trans / total
                    if prob == 0:
                        surprise = 1.0
                    else:
                        surprise = -math.log2(prob + 1e-10)
                        surprise = min(surprise / 10.0, 1.0)  # 归一化

                total_surprise += surprise
                count += 1

        return total_surprise / max(count, 1)

    def get_stats(self) -> Dict:
        """获取学习统计"""
        emergent = self.get_emergent_concepts()
        relations = self.get_emergent_relations()
        return {
            'texts_observed': self._text_count,
            'total_chars': self._total_chars,
            'unique_ngrams': len(self._ngram_freq),
            'candidate_concepts': len(self._concepts),
            'emergent_concepts': len(emergent),
            'emergent_relations': len(relations),
            'top_concepts': [(c, round(s, 3)) for c, s in emergent[:10]],
            'top_relations': [(a, b, round(s, 3)) for a, b, s in relations[:10]],
        }

    # ====== 内部方法 ======

    def _segment(self, text: str) -> List[str]:
        """将文本分割为中文字符段和英文单词段

        不做分词！这是关键区别 — 我们统计的是原始字符片段，
        让"词"从统计中涌现，而不是预先分好。
        """
        segments = []
        current = ''

        for ch in text:
            if '一' <= ch <= '鿿':  # 中文
                current += ch
            elif ch.isascii() and (ch.isalpha() or ch.isdigit()):
                current += ch
            else:
                if current:
                    segments.append(current)
                    current = ''
                # 标点/空白作为分隔

        if current:
            segments.append(current)

        return segments

    def _count_ngrams(self, segment: str, text_id: int):
        """统计一个片段的所有n-gram频率"""
        if len(segment) < 2:
            return

        # 单字符频率
        for ch in segment:
            if '一' <= ch <= '鿿':
                self._char_freq[ch] += 1

        # n-gram频率 (2到max_ngram)
        for n in range(2, min(self.max_ngram + 1, len(segment) + 1)):
            for i in range(len(segment) - n + 1):
                gram = segment[i:i + n]
                if self._is_valid_ngram(gram):
                    self._ngram_freq[gram] += 1
                    self._ngram_contexts[gram].add(text_id)

    def _is_valid_ngram(self, gram: str) -> bool:
        """判断n-gram是否值得统计（过滤噪音）"""
        if len(gram) < 2:
            return False

        # 全是停用词 → 不值得
        if all(ch in self._stop_chars for ch in gram):
            return False

        # 全是数字 → 不作为概念候选
        if gram.isdigit():
            return False

        # 混合中英文 → 通常噪音
        has_cjk = any('一' <= ch <= '鿿' for ch in gram)
        has_ascii = any(ch.isascii() and ch.isalpha() for ch in gram)
        if has_cjk and has_ascii:
            return False

        return True

    def _compute_pmi(self, gram: str) -> float:
        """计算n-gram的点互信息(PMI)

        PMI(a,b) = log2(P(ab) / (P(a) * P(b)))

        PMI高 → a和b一起出现不是巧合 → 可能是一个有意义的片段
        PMI低或负 → a和b独立出现 → 可能是两个无关的字符碰巧相邻
        """
        if len(gram) < 2:
            return 0.0

        total = self._total_chars
        if total == 0:
            return 0.0

        # P(gram) = freq(gram) / total
        p_gram = self._ngram_freq.get(gram, 0) / total

        # 对于2-gram: PMI = log2(P(ab) / (P(a)*P(b)))
        # 对于3-gram以上: 用最好的二分计算
        if len(gram) == 2:
            a, b = gram[0], gram[1]
            p_a = self._char_freq.get(a, 0) / total
            p_b = self._char_freq.get(b, 0) / total
            if p_a == 0 or p_b == 0 or p_gram == 0:
                return 0.0
            return math.log2(p_gram / (p_a * p_b))
        else:
            # 对于长n-gram: 取所有二分中PMI最小的（瓶颈PMI）
            min_pmi = float('inf')
            for i in range(1, len(gram)):
                left = gram[:i]
                right = gram[i:]
                p_left = self._ngram_freq.get(left, self._char_freq.get(left[0], 0)) / total
                p_right = self._ngram_freq.get(right, self._char_freq.get(right[0], 0)) / total
                if p_left == 0 or p_right == 0 or p_gram == 0:
                    return 0.0
                pmi = math.log2(p_gram / (p_left * p_right))
                min_pmi = min(min_pmi, pmi)
            return min_pmi if min_pmi != float('inf') else 0.0

    def _check_emergence(self, new_ngrams: Set[str]) -> List[str]:
        """检查是否有新概念涌现

        概念涌现条件：
        1. 频率 >= min_freq（被观察到足够多次）
        2. 出现在 >= 2 个不同文本中（不是单条文本的特殊模式）
        3. PMI > 0（内部字符绑定紧密，不是随机拼接）
        """
        newly_emerged = []

        for gram in new_ngrams:
            freq = self._ngram_freq.get(gram, 0)
            contexts = len(self._ngram_contexts.get(gram, set()))

            if freq < self.min_freq:
                continue

            # 已存在 → 更新
            if gram in self._concepts:
                self._concepts[gram].frequency = freq
                self._concepts[gram].total_contexts = contexts
                self._concepts[gram].last_seen_idx = self._text_count
                continue

            # 新候选 → 检查PMI
            pmi = self._compute_pmi(gram)
            if pmi <= 0:
                continue  # PMI <= 0 意味着片段内部不紧密

            # 涌现！
            candidate = ConceptCandidate(
                text=gram,
                frequency=freq,
                total_contexts=contexts,
                pmi=pmi,
                last_seen_idx=self._text_count,
            )
            self._concepts[gram] = candidate
            newly_emerged.append(gram)

        # 容量限制：保留top-K个概念（按频率排序）
        if len(self._concepts) > self.max_concepts:
            sorted_concepts = sorted(
                self._concepts.items(),
                key=lambda x: (x[1].frequency, x[1].total_contexts),
                reverse=True
            )
            self._concepts = dict(sorted_concepts[:self.max_concepts])

        return newly_emerged

    def _check_relation_emergence(self) -> List[Tuple[str, str]]:
        """检查是否有新关系涌现

        关系涌现条件：
        1. 两个概念都已涌现
        2. 它们的共现次数 >= 2
        3. PMI >= min_pmi
        """
        newly_emerged = []
        emergent_set = set(self._concepts.keys())

        for (a, b), cooc_count in self._cooccurrence.items():
            if cooc_count < 2:
                continue
            if a not in emergent_set or b not in emergent_set:
                continue

            key = (min(a, b), max(a, b))

            # 计算PMI
            total = self._text_count
            if total == 0:
                continue
            p_a = self._ngram_freq.get(a, 0) / total
            p_b = self._ngram_freq.get(b, 0) / total
            p_ab = cooc_count / total

            if p_a == 0 or p_b == 0 or p_ab == 0:
                continue

            pmi = math.log2(p_ab / (p_a * p_b))

            if pmi < self.min_pmi:
                continue

            # 更新或创建关系
            if key not in self._relations:
                self._relations[key] = CooccurrenceRelation(
                    concept_a=key[0],
                    concept_b=key[1],
                    cooccurrence_count=cooc_count,
                    pmi=pmi,
                )
                newly_emerged.append(key)
            else:
                self._relations[key].cooccurrence_count = cooc_count
                self._relations[key].pmi = pmi

        return newly_emerged

    def _is_complete_word(self, gram: str) -> bool:
        """词边界检测 — 判断n-gram是否是一个完整的词，而非跨词碎片

        Saffran(1996)核心发现的实际应用：
        不看PMI绝对值（小语料中模板化的文本会让跨词对的PMI也很高），
        而看PMI曲线的"局部最小值" — 真正的词边界是PMI曲线的**谷底**。

        检测方法：
        1. 对gram中每对相邻字符计算2-gram PMI，形成PMI序列
        2. 检查PMI序列中是否存在"局部最小值"（比两边都低的位置）
        3. 局部最小值 = 词边界 → 跨词碎片

        例如 "学是研究"：PMI=[3.6, 4.6, 4.6]
        - 位置0(学是=3.6) < 位置1(是研=4.6) → 局部最小值 → 词边界
        - 但3.6不算特别低...

        更实际的方法：利用**互信息的可加性**
        如果"学是研究"是一个整体，那PMI(学是研究)应该>=所有内部对的PMI
        如果它是碎片（学|是|研究），那整体PMI会低于子成分的PMI

        最终策略（组合判断）：
        A. 如果存在PMI <= 0的位置 → 词边界（最强信号）
        B. 如果存在局部PMI最小值且比值<0.5 → 词边界
        C. 如果gram包含语法功能词的n-gram模式 → 可能是碎片
        """
        if len(gram) <= 1:
            return True  # 单字符无法判断边界

        # 计算gram中每对相邻字符的2-gram PMI
        pair_pmis = []
        for i in range(len(gram) - 1):
            pair = gram[i:i + 2]
            pair_pmi = self._compute_pair_pmi(pair)
            pair_pmis.append(pair_pmi)

        if not pair_pmis:
            return True

        # A. PMI <= 0 → 明确的词边界
        if any(pmi <= 0 for pmi in pair_pmis):
            return False

        # B. 局部最小值检测
        # PMI序列中的谷底 = 词边界
        if len(pair_pmis) >= 3:
            for i in range(1, len(pair_pmis) - 1):
                if pair_pmis[i] < pair_pmis[i-1] and pair_pmis[i] < pair_pmis[i+1]:
                    # 局部最小值，检查是否显著低于邻居
                    avg_neighbor = (pair_pmis[i-1] + pair_pmis[i+1]) / 2
                    if avg_neighbor > 0 and pair_pmis[i] < avg_neighbor * 0.5:
                        return False

        # C. 检测已知的跨词模式（语法功能词出现在n-gram中）
        # 中文字符中"是、的、了、在、和、与"等高频虚词 → 很可能跨越了词边界
        function_chars = set('是的有在了和与被把让给从到以也而')

        # C1. 对于3+gram：检查功能词位置
        if len(gram) >= 3:
            gram_freq = self._ngram_freq.get(gram, 0)

            # C1a. 首字符是功能词 → 几乎一定是碎片（如"是研究"、"的学科"）
            # 功能词（是、的、在、了等）几乎不会是真正词语的开头
            if gram[0] in function_chars:
                return False

            # C1b. 尾字符是功能词 → 碎片（如"规律的"、"研究的"）
            if gram[-1] in function_chars:
                return False

            # C1c. 中间位置出现功能词
            middle_chars = gram[1:-1]  # 去掉首尾
            for ch in middle_chars:
                if ch in function_chars:
                    # 找出功能词位置分割的两个子片段
                    idx = gram.index(ch)
                    left = gram[:idx]
                    right = gram[idx+1:]
                    if left and right:
                        # 获取子片段频率
                        if len(left) == 1:
                            left_freq = self._char_freq.get(left, 0)
                        else:
                            left_freq = self._ngram_freq.get(left, 0)
                        if len(right) == 1:
                            right_freq = self._char_freq.get(right, 0)
                        else:
                            right_freq = self._ngram_freq.get(right, 0)

                        # 如果两个子片段各自频率远高于整体 → 整体是碎片
                        if left_freq >= gram_freq and right_freq >= gram_freq:
                            return False

        # C2. 对于2-gram：如果其中一个字符是高频功能词 → 检查是否跨词碎片
        if len(gram) == 2:
            for i, ch in enumerate(gram):
                if ch in function_chars:
                    other = gram[1 - i]
                    char_freq = self._char_freq.get(ch, 0)
                    other_freq = self._char_freq.get(other, 0)
                    gram_freq = self._ngram_freq.get(gram, 0)

                    # C2a. 功能字符独立频率 >> 它在这个2-gram中的频率 → 跨词碎片
                    if char_freq > 0 and gram_freq > 0:
                        ratio = char_freq / gram_freq
                        if ratio >= 3.0:
                            return False

                    # C2b. 被更长的n-gram包含 → 碎片
                    # 如"学是"被"学是研"(3-gram)包含，且频率接近 → "学是"是跨词边界
                    # 真正独立的词不会总是出现在某个3-gram中
                    if gram_freq > 0:
                        found_longer = False
                        for longer, lfreq in self._ngram_freq.items():
                            if len(longer) == 3 and gram in longer and lfreq >= gram_freq * 0.4:
                                found_longer = True
                                break
                        if found_longer:
                            return False

        # D. 转移概率边界检测（Saffran 1996 核心发现）
        # 转移概率 P(next|current) 在词内部高，在词边界处骤降
        # "数学" 内部 P(学|数) 高, "学是" 跨边界 P(是|学) 低 → "学是" 是碎片
        if len(gram) >= 2 and self._transition_totals:
            trans_probs = []
            for i in range(len(gram) - 1):
                total = self._transition_totals.get(gram[i], 0)
                if total > 0:
                    tp = self._transitions[gram[i]].get(gram[i + 1], 0) / total
                else:
                    tp = 0.0
                trans_probs.append(tp)

            if trans_probs:
                avg_tp = sum(trans_probs) / len(trans_probs)
                min_tp = min(trans_probs)
                # 最低转移概率远低于平均 → 该位置是词边界
                if avg_tp > 0.02 and min_tp < avg_tp * 0.25:
                    return False
                # 接近0的转移概率但平均不低 → 强边界信号
                if min_tp < 0.005 and avg_tp > 0.03:
                    return False

        # 注意：子串反向验证（原 Strategy E）已移至 _post_emergence_cleanup()
        # 在 _is_complete_word() 中执行子串检查有顺序依赖问题：
        # 碎片（如"究人类"）可能在父概念（"研究人类"）之前涌现，
        # 导致 Strategy E 找不到父概念而漏过。
        # 现在改为：所有概念先涌现，然后一次性清理碎片。

        return True

    def _compute_pair_pmi(self, pair: str) -> float:
        """计算2-gram的精确PMI（高效版本）"""
        if len(pair) != 2:
            return 0.0

        total = self._total_chars
        if total == 0:
            return 0.0

        freq = self._ngram_freq.get(pair, 0)
        if freq == 0:
            return -1.0  # 未观察到的2-gram，PMI极低

        freq_a = self._char_freq.get(pair[0], 0)
        freq_b = self._char_freq.get(pair[1], 0)

        if freq_a == 0 or freq_b == 0:
            return 0.0

        p_pair = freq / total
        p_a = freq_a / total
        p_b = freq_b / total

        if p_a * p_b == 0:
            return 0.0

        return math.log2(p_pair / (p_a * p_b))
