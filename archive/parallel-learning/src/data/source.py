"""数据源接口 — 可插拔的知识来源

数据源是通用学习架构的输入层。
每个数据源负责从特定来源加载知识单元：
- 英语：从 Free Dictionary API + 牛津词汇表加载
- 数学：从数学概念库加载
- 物理：从物理概念库加载
- 计算机：从 CS 概念库加载
- 化学：从化学概念库加载

用户可以实现自己的 DataSource 来学习任何领域的知识。
"""

import json
import os
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pathlib import Path

from src.knowledge.unit import KnowledgeUnit
from src.data.oxford_words import (
    OXFORD_3000, WORD_PARTS, WORD_FAMILIES, CEFR_LEVELS,
    SEMANTIC_RELATIONS, COLLOCATIONS, WORD_EXAMPLES,
)


class DataSource(ABC):
    """数据源接口

    所有数据源必须实现这个接口。
    """

    @abstractmethod
    def get_domain(self) -> str:
        """返回知识领域名称"""
        pass

    @abstractmethod
    def load_units(self) -> List[KnowledgeUnit]:
        """加载所有知识单元"""
        pass

    def get_unit_by_id(self, unit_id: str) -> Optional[KnowledgeUnit]:
        """根据 ID 获取单个知识单元"""
        for unit in self.load_units():
            if unit.id == unit_id:
                return unit
        return None


class EnglishDataSource(DataSource):
    """英语数据源 — 从 Free Dictionary API + 牛津词汇表加载

    特性：
    - 覆盖牛津 3000 核心词汇（按 CEFR A1-C2 分级）
    - 从 Free Dictionary API 获取完整词典条目
    - 词根词缀分析（前缀、后缀、词根）
    - 词族关联（同一词根的不同词性）
    - 磁盘缓存（避免重复 API 调用）
    """

    def __init__(self, cache_dir: str = 'data/english_cache',
                 max_words: int = 1000,
                 use_api: bool = True):
        """
        Args:
            cache_dir: 缓存目录
            max_words: 最大加载词数
            use_api: 是否使用 API（False 则只用内置数据）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_words = max_words
        self.use_api = use_api
        self._cache: Dict[str, KnowledgeUnit] = {}

        # 从 oxford_words.py 加载词汇表
        self.word_list = list(OXFORD_3000.keys())

        # 构建反向词族查找：member_word → (root, siblings)
        self._reverse_family: Dict[str, tuple] = {}
        for root, members in WORD_FAMILIES.items():
            for m in members:
                if m != root:
                    siblings = [w for w in members if w != m]
                    self._reverse_family[m] = (root, siblings)

        # 构建前缀/后缀索引
        self._prefix_index: Dict[str, List[str]] = {}
        self._suffix_index: Dict[str, List[str]] = {}
        for w in self.word_list:
            for pfx in self._PREFIXES:
                if w.startswith(pfx) and len(w) > len(pfx) + 2:
                    self._prefix_index.setdefault(pfx, []).append(w)
            for sfx in self._SUFFIXES:
                if w.endswith(sfx) and len(w) > len(sfx) + 2:
                    self._suffix_index.setdefault(sfx, []).append(w)

    def get_domain(self) -> str:
        return 'english'

    def load_units(self) -> List[KnowledgeUnit]:
        """加载英语词汇知识单元"""
        units = []
        words_to_load = self.word_list[:self.max_words]

        for word in words_to_load:
            unit = self._load_word(word)
            if unit:
                units.append(unit)

        return units

    def _load_word(self, word: str) -> Optional[KnowledgeUnit]:
        """加载单个词汇"""
        # 检查内存缓存
        if word in self._cache:
            return self._cache[word]

        # 检查磁盘缓存
        cached = self._load_from_disk(word)
        if cached:
            self._cache[word] = cached
            return cached

        # 从 API 获取（如果启用）
        if self.use_api:
            entry = self._fetch_from_api(word)
            if entry:
                self._save_to_disk(word, entry)
                self._cache[word] = entry
                return entry

        # 使用内置数据创建
        entry = self._create_from_builtin(word)
        if entry:
            self._cache[word] = entry
            return entry

        return None

    def _fetch_from_api(self, word: str) -> Optional[KnowledgeUnit]:
        """从 Free Dictionary API 获取词汇"""
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'UniversalLearner/1.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            if not data or not isinstance(data, list):
                return None

            return self._parse_response(word, data[0])
        except Exception:
            return None

    def _parse_response(self, word: str, data: dict) -> KnowledgeUnit:
        """解析 API 响应为 KnowledgeUnit"""
        # 提取发音
        ipa = ''
        for phonetic in data.get('phonetics', []):
            if phonetic.get('text'):
                ipa = phonetic['text']
                break

        # 提取词义
        definitions = []
        synonyms = []
        antonyms = []
        examples = []
        pos_list = []

        for meaning in data.get('meanings', []):
            pos = meaning.get('partOfSpeech', '')
            if pos and pos not in pos_list:
                pos_list.append(pos)

            for defn in meaning.get('definitions', []):
                text = defn.get('definition', '')
                if text:
                    definitions.append(text)
                if defn.get('example'):
                    examples.append(defn['example'])
                for syn in defn.get('synonyms', []):
                    if syn and syn not in synonyms:
                        synonyms.append(syn)
                for ant in defn.get('antonyms', []):
                    if ant and ant not in antonyms:
                        antonyms.append(ant)

            for syn in meaning.get('synonyms', []):
                if syn and syn not in synonyms:
                    synonyms.append(syn)
            for ant in meaning.get('antonyms', []):
                if ant and ant not in antonyms:
                    antonyms.append(ant)

        # 构建定义文本
        definition = '; '.join(definitions[:3]) if definitions else f"The word '{word}'"

        # 构建相关词（同义词 + 反义词 + 词族）
        related = synonyms[:5] + antonyms[:3]

        # 添加词族关联
        word_family = WORD_FAMILIES.get(word, [])
        for family_word in word_family:
            if family_word != word and f"english:{family_word}" not in related:
                related.append(f"english:{family_word}")

        # 添加词根词缀信息
        explanations = []
        if ipa:
            explanations.append(f"IPA: {ipa}")

        # 分析词根词缀
        for prefix, info in WORD_PARTS.items():
            if prefix.endswith('-') and word.startswith(prefix[:-1]):
                explanations.append(f"Prefix {prefix}: {info['meaning']}")
            elif prefix.startswith('-') and word.endswith(prefix[1:]):
                explanations.append(f"Suffix {prefix}: {info['meaning']}")

        # CEFR 等级
        cefr = OXFORD_3000.get(word, 'B1')
        difficulty = CEFR_LEVELS.get(cefr, 0.4)

        return KnowledgeUnit(
            id=f"english:{word}",
            name=word,
            domain='english',
            definition=definition,
            difficulty=difficulty,
            examples=examples[:5],
            explanations=explanations,
            keywords=pos_list,
            prerequisites=[],
            related=[f"english:{w}" for w in related[:10]],
            source='dictionary_api',
            tags=pos_list + ['vocabulary', cefr],
            metadata={
                'cefr': cefr,
                'ipa': ipa,
                'synonyms': synonyms[:10],
                'antonyms': antonyms[:5],
                'word_family': word_family[:10],
            },
        )

    def _create_from_builtin(self, word: str) -> Optional[KnowledgeUnit]:
        """使用内置数据创建知识单元（不依赖 API）"""
        cefr = OXFORD_3000.get(word)
        if not cefr:
            return None

        difficulty = CEFR_LEVELS.get(cefr, 0.4)
        word_set = set(OXFORD_3000.keys())

        # ── 语义关系 ──
        related = []
        is_a = []
        prerequisites = []
        sem = SEMANTIC_RELATIONS.get(word, {})
        if sem:
            for syn in sem.get('synonyms', [])[:5]:
                if syn in word_set:
                    related.append(f"english:{syn}")
            for ant in sem.get('antonyms', [])[:3]:
                if ant in word_set:
                    related.append(f"english:{ant}")
            for hyper in sem.get('hypernyms', [])[:3]:
                is_a.append(f"english:{hyper}")

        # ── 词族（正向 + 反向查找）──
        word_family = WORD_FAMILIES.get(word, [])
        if not word_family and word in self._reverse_family:
            root, siblings = self._reverse_family[word]
            word_family = [root] + siblings

        for w in word_family:
            if w != word and f"english:{w}" not in related and w in word_set:
                related.append(f"english:{w}")

        if word_family and word_family[0] != word:
            root = word_family[0]
            if f"english:{root}" not in prerequisites:
                prerequisites.append(f"english:{root}")

        # ── 形态学自动关联（始终执行）──
        morph_siblings = self._find_morphological_siblings(word)
        for s in morph_siblings:
            if f"english:{s}" not in related:
                related.append(f"english:{s}")

        # ── 前缀族关联 ──
        prefix_siblings = self._find_prefix_siblings(word, word_set)
        for s in prefix_siblings:
            if f"english:{s}" not in related:
                related.append(f"english:{s}")

        # ── 后缀族关联 ──
        suffix_siblings = self._find_suffix_siblings(word, word_set)
        for s in suffix_siblings:
            if f"english:{s}" not in related:
                related.append(f"english:{s}")

        # ── CEFR 同级词关联（同级词互为参考）──
        same_cefr = [w for w, c in OXFORD_3000.items()
                     if c == cefr and w != word][:2]
        for w in same_cefr:
            if f"english:{w}" not in related:
                related.append(f"english:{w}")

        related = related[:15]

        # ── 词族元数据（用于词族覆盖率评估）──
        if not word_family:
            # 尝试从形态学推断词族
            for s in morph_siblings:
                word_family.append(s)
            if not word_family:
                word_family = [word]

        # ── 词根词缀 ──
        explanations = []
        morphology = []
        for prefix, info in WORD_PARTS.items():
            if prefix.endswith('-') and word.startswith(prefix[:-1]):
                explanations.append(f"Prefix {prefix}: {info['meaning']}")
                morphology.append(prefix)
            elif prefix.startswith('-') and word.endswith(prefix[1:]):
                explanations.append(f"Suffix {prefix}: {info['meaning']}")
                morphology.append(prefix)

        # ── 搭配 ──
        collocations = COLLOCATIONS.get(word, [])
        if collocations:
            explanations.append(f"Common collocations: {', '.join(collocations[:6])}")

        # ── 例句（有数据用数据，无数据自动生成）──
        examples = WORD_EXAMPLES.get(word, [])
        if not examples:
            examples = self._generate_examples(word)

        # ── 定义 ──
        definition = f"English word: {word}"
        if sem.get('hypernyms'):
            definition = f"A type of {sem['hypernyms'][0]}"

        return KnowledgeUnit(
            id=f"english:{word}",
            name=word,
            domain='english',
            definition=definition,
            difficulty=difficulty,
            examples=examples[:5],
            explanations=explanations,
            keywords=[],
            prerequisites=prerequisites,
            related=related,
            is_a=is_a,
            source='oxford_builtin',
            tags=['vocabulary', cefr],
            metadata={
                'cefr': cefr,
                'word_family': word_family[:10],
                'morphology': morphology,
                'collocations': collocations[:6],
                'synonyms': sem.get('synonyms', [])[:5],
                'antonyms': sem.get('antonyms', [])[:3],
            },
        )

    def _estimate_difficulty(self, word: str) -> float:
        """估算词汇难度"""
        cefr = OXFORD_3000.get(word, 'B1')
        return CEFR_LEVELS.get(cefr, 0.4)

    # 常见后缀变换模式
    _SUFFIX_PATTERNS = [
        ('tion', 'te'), ('sion', 'se'), ('ment', ''), ('ness', ''),
        ('able', ''), ('ible', ''), ('ful', ''), ('less', ''),
        ('ous', ''), ('ive', ''), ('al', ''), ('ly', ''),
        ('ize', ''), ('ify', ''), ('er', ''), ('or', ''),
        ('ist', ''), ('ance', ''), ('ence', ''),
    ]

    def _find_morphological_siblings(self, word: str) -> List[str]:
        """通过形态学模式查找可能的词族成员"""
        siblings = []
        word_set = set(OXFORD_3000.keys())

        for suffix, stem_suffix in self._SUFFIX_PATTERNS:
            # word 以 suffix 结尾 → 尝试去 suffix 加 stem_suffix
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                stem = word[:-len(suffix)] + stem_suffix
                if stem != word and stem in word_set:
                    siblings.append(stem)
                stem2 = word[:-len(suffix)]
                if stem2 != word and len(stem2) > 2 and stem2 in word_set:
                    siblings.append(stem2)

            # word 不以 suffix 结尾 → 尝试加 suffix
            if not word.endswith(suffix):
                candidate = word + suffix
                if candidate in word_set:
                    siblings.append(candidate)

        # 额外模式：-ing/-ed/-er/-est 形式
        for suffix in ['ing', 'ed', 'er', 'est', 's']:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                stem = word[:-len(suffix)]
                if stem in word_set:
                    siblings.append(stem)
            else:
                candidate = word + suffix
                if candidate in word_set:
                    siblings.append(candidate)

        return list(set(siblings))[:8]

    # 常见前缀
    _PREFIXES = ['un', 're', 'pre', 'dis', 'mis', 'over', 'under',
                 'out', 'inter', 'trans', 'super', 'sub', 'anti',
                 'co', 'auto', 'bi', 'multi', 'non', 'de', 'en']

    def _find_prefix_siblings(self, word: str, word_set: set) -> List[str]:
        """查找同前缀的词（使用索引）"""
        siblings = []
        for pfx in self._PREFIXES:
            if word.startswith(pfx) and len(word) > len(pfx) + 2:
                candidates = self._prefix_index.get(pfx, [])
                for w in candidates:
                    if w != word:
                        siblings.append(w)
                        if len(siblings) >= 3:
                            return siblings
        return siblings[:3]

    # 常见后缀
    _SUFFIXES = ['tion', 'sion', 'ment', 'ness', 'able', 'ible',
                 'ful', 'less', 'ous', 'ive', 'al', 'ly', 'ize',
                 'ify', 'er', 'or', 'ist', 'ance', 'ence', 'ity']

    def _find_suffix_siblings(self, word: str, word_set: set) -> List[str]:
        """查找同后缀的词（使用索引）"""
        siblings = []
        for sfx in self._SUFFIXES:
            if word.endswith(sfx) and len(word) > len(sfx) + 2:
                candidates = self._suffix_index.get(sfx, [])
                for w in candidates:
                    if w != word:
                        siblings.append(w)
                        if len(siblings) >= 3:
                            return siblings
        return siblings[:3]

    # 模板例句生成
    _SENTENCE_TEMPLATES = [
        'The {word} is very important.',
        'She likes the {word}.',
        'We need more {word}.',
        'He talked about the {word}.',
        'The {word} was good.',
    ]

    def _generate_examples(self, word: str) -> List[str]:
        """为没有例句的词生成模板例句"""
        return [
            f"The {word} is important.",
            f"She knows about {word}.",
            f"We discussed the {word}.",
        ]

    def _load_from_disk(self, word: str) -> Optional[KnowledgeUnit]:
        """从磁盘缓存加载"""
        path = self.cache_dir / f"{word}.json"
        if not path.exists():
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return KnowledgeUnit.from_dict(data)
        except Exception:
            return None

    def _save_to_disk(self, word: str, unit: KnowledgeUnit) -> None:
        """保存到磁盘缓存"""
        path = self.cache_dir / f"{word}.json"
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(unit.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass


class MathDataSource(DataSource):
    """数学数据源 — 从预定义数学概念加载"""

    def get_domain(self) -> str:
        return 'math'

    def load_units(self) -> List[KnowledgeUnit]:
        """加载数学概念"""
        return [
            # 基础算术
            KnowledgeUnit(
                id='math:arithmetic:addition',
                name='加法',
                domain='math',
                definition='将两个或多个数合并为一个数的运算',
                difficulty=0.1,
                examples=['2 + 3 = 5', '10 + 20 = 30'],
                keywords=['addition', 'sum', 'plus'],
                prerequisites=[],
                related=['math:arithmetic:subtraction'],
                tags=['arithmetic', 'basic'],
            ),
            KnowledgeUnit(
                id='math:arithmetic:subtraction',
                name='减法',
                domain='math',
                definition='从一个数中减去另一个数的运算',
                difficulty=0.1,
                examples=['5 - 2 = 3', '10 - 7 = 3'],
                keywords=['subtraction', 'minus', 'difference'],
                prerequisites=['math:arithmetic:addition'],
                related=['math:arithmetic:addition'],
                tags=['arithmetic', 'basic'],
            ),
            KnowledgeUnit(
                id='math:arithmetic:multiplication',
                name='乘法',
                domain='math',
                definition='将一个数重复相加多次的运算',
                difficulty=0.2,
                examples=['3 × 4 = 12', '5 × 5 = 25'],
                keywords=['multiplication', 'times', 'product'],
                prerequisites=['math:arithmetic:addition'],
                related=['math:arithmetic:division'],
                tags=['arithmetic', 'basic'],
            ),
            KnowledgeUnit(
                id='math:arithmetic:division',
                name='除法',
                domain='math',
                definition='将一个数分成若干等份的运算',
                difficulty=0.2,
                examples=['12 ÷ 3 = 4', '20 ÷ 5 = 4'],
                keywords=['division', 'divide', 'quotient'],
                prerequisites=['math:arithmetic:multiplication'],
                related=['math:arithmetic:multiplication'],
                tags=['arithmetic', 'basic'],
            ),
            # 代数
            KnowledgeUnit(
                id='math:algebra:variable',
                name='变量',
                domain='math',
                definition='用字母表示的未知数或可变数',
                difficulty=0.3,
                examples=['x + 5 = 10', 'y = 2x + 1'],
                keywords=['variable', 'unknown'],
                prerequisites=['math:arithmetic:addition'],
                related=['math:algebra:equation'],
                tags=['algebra'],
            ),
            KnowledgeUnit(
                id='math:algebra:equation',
                name='方程',
                domain='math',
                definition='含有未知数的等式',
                difficulty=0.4,
                examples=['x + 3 = 7', '2x - 1 = 5'],
                keywords=['equation', 'solve'],
                prerequisites=['math:algebra:variable'],
                related=['math:algebra:inequality'],
                tags=['algebra'],
            ),
            # 几何
            KnowledgeUnit(
                id='math:geometry:triangle',
                name='三角形',
                domain='math',
                definition='由三条线段围成的图形',
                difficulty=0.3,
                examples=['等边三角形', '直角三角形'],
                keywords=['triangle', 'polygon'],
                prerequisites=[],
                related=['math:geometry:circle'],
                tags=['geometry'],
            ),
            KnowledgeUnit(
                id='math:geometry:circle',
                name='圆',
                domain='math',
                definition='平面上到定点距离等于定长的点的集合',
                difficulty=0.3,
                examples=['圆的面积 = πr²'],
                keywords=['circle', 'radius', 'diameter'],
                prerequisites=[],
                related=['math:geometry:triangle'],
                tags=['geometry'],
            ),
            # 微积分
            KnowledgeUnit(
                id='math:calculus:limit',
                name='极限',
                domain='math',
                definition='函数在某点附近的行为',
                difficulty=0.6,
                examples=['lim(x→0) sin(x)/x = 1'],
                keywords=['limit', 'convergence'],
                prerequisites=['math:algebra:equation'],
                related=['math:calculus:derivative'],
                tags=['calculus'],
            ),
            KnowledgeUnit(
                id='math:calculus:derivative',
                name='导数',
                domain='math',
                definition='函数在某点的变化率',
                difficulty=0.7,
                examples=["f'(x) = lim(h→0) [f(x+h) - f(x)]/h"],
                keywords=['derivative', 'differentiation', 'rate of change'],
                prerequisites=['math:calculus:limit'],
                related=['math:calculus:integral'],
                tags=['calculus'],
            ),
            KnowledgeUnit(
                id='math:calculus:integral',
                name='积分',
                domain='math',
                definition='函数在区间上的累积量',
                difficulty=0.7,
                examples=['∫f(x)dx = F(x) + C'],
                keywords=['integral', 'integration', 'antiderivative'],
                prerequisites=['math:calculus:derivative'],
                related=['math:calculus:derivative'],
                tags=['calculus'],
            ),
        ]


class PhysicsDataSource(DataSource):
    """物理数据源 — 从预定义物理概念加载"""

    def get_domain(self) -> str:
        return 'physics'

    def load_units(self) -> List[KnowledgeUnit]:
        """加载物理概念"""
        return [
            # 力学
            KnowledgeUnit(
                id='physics:mechanics:force',
                name='力',
                domain='physics',
                definition='改变物体运动状态的作用',
                difficulty=0.3,
                examples=['重力', '摩擦力', '弹力'],
                keywords=['force', 'newton'],
                prerequisites=[],
                related=['physics:mechanics:acceleration'],
                tags=['mechanics'],
            ),
            KnowledgeUnit(
                id='physics:mechanics:acceleration',
                name='加速度',
                domain='physics',
                definition='速度变化的快慢',
                difficulty=0.4,
                examples=['a = F/m', '自由落体加速度 g = 9.8 m/s²'],
                keywords=['acceleration', 'velocity'],
                prerequisites=['physics:mechanics:force'],
                related=['physics:mechanics:force'],
                tags=['mechanics'],
            ),
            KnowledgeUnit(
                id='physics:mechanics:energy',
                name='能量',
                domain='physics',
                definition='物体做功的能力',
                difficulty=0.4,
                examples=['动能 = ½mv²', '势能 = mgh'],
                keywords=['energy', 'kinetic', 'potential'],
                prerequisites=['physics:mechanics:force'],
                related=['physics:mechanics:work'],
                tags=['mechanics'],
            ),
            KnowledgeUnit(
                id='physics:mechanics:work',
                name='功',
                domain='physics',
                definition='力在位移方向上的分量与位移的乘积',
                difficulty=0.4,
                examples=['W = F·d·cosθ'],
                keywords=['work', 'joule'],
                prerequisites=['physics:mechanics:force'],
                related=['physics:mechanics:energy'],
                tags=['mechanics'],
            ),
            # 电磁学
            KnowledgeUnit(
                id='physics:electromagnetism:charge',
                name='电荷',
                domain='physics',
                definition='物质的基本属性之一',
                difficulty=0.3,
                examples=['正电荷', '负电荷', '电荷守恒'],
                keywords=['charge', 'electric'],
                prerequisites=[],
                related=['physics:electromagnetism:field'],
                tags=['electromagnetism'],
            ),
            KnowledgeUnit(
                id='physics:electromagnetism:field',
                name='电场',
                domain='physics',
                definition='电荷周围空间的一种特殊物质',
                difficulty=0.5,
                examples=['E = kQ/r²'],
                keywords=['field', 'electric field'],
                prerequisites=['physics:electromagnetism:charge'],
                related=['physics:electromagnetism:potential'],
                tags=['electromagnetism'],
            ),
            # 热学
            KnowledgeUnit(
                id='physics:thermodynamics:temperature',
                name='温度',
                domain='physics',
                definition='物体冷热程度的量度',
                difficulty=0.2,
                examples=['摄氏度', '华氏度', '开尔文'],
                keywords=['temperature', 'heat'],
                prerequisites=[],
                related=['physics:thermodynamics:heat'],
                tags=['thermodynamics'],
            ),
            KnowledgeUnit(
                id='physics:thermodynamics:heat',
                name='热量',
                domain='physics',
                definition='由于温度差而传递的能量',
                difficulty=0.3,
                examples=['Q = mcΔT'],
                keywords=['heat', 'thermal'],
                prerequisites=['physics:thermodynamics:temperature'],
                related=['physics:thermodynamics:temperature'],
                tags=['thermodynamics'],
            ),
            # 光学
            KnowledgeUnit(
                id='physics:optics:light',
                name='光',
                domain='physics',
                definition='电磁波的一种，能引起视觉',
                difficulty=0.3,
                examples=['光的折射', '光的反射'],
                keywords=['light', 'optics'],
                prerequisites=[],
                related=['physics:optics:lens'],
                tags=['optics'],
            ),
            KnowledgeUnit(
                id='physics:optics:lens',
                name='透镜',
                domain='physics',
                definition='能使光线会聚或发散的光学元件',
                difficulty=0.4,
                examples=['凸透镜', '凹透镜', '1/f = 1/u + 1/v'],
                keywords=['lens', 'focus'],
                prerequisites=['physics:optics:light'],
                related=['physics:optics:light'],
                tags=['optics'],
            ),
        ]


class CSDataSource(DataSource):
    """计算机科学数据源 — 从预定义 CS 概念加载"""

    def get_domain(self) -> str:
        return 'cs'

    def load_units(self) -> List[KnowledgeUnit]:
        """加载 CS 概念"""
        return [
            # 编程基础
            KnowledgeUnit(
                id='cs:programming:variable',
                name='变量',
                domain='cs',
                definition='存储数据的命名容器',
                difficulty=0.2,
                examples=['int x = 5;', 'let name = "Alice";'],
                keywords=['variable', 'data'],
                prerequisites=[],
                related=['cs:programming:type'],
                tags=['programming', 'basic'],
            ),
            KnowledgeUnit(
                id='cs:programming:type',
                name='数据类型',
                domain='cs',
                definition='变量中存储的数据的种类',
                difficulty=0.2,
                examples=['int', 'float', 'string', 'boolean'],
                keywords=['type', 'data type'],
                prerequisites=['cs:programming:variable'],
                related=['cs:programming:variable'],
                tags=['programming', 'basic'],
            ),
            KnowledgeUnit(
                id='cs:programming:loop',
                name='循环',
                domain='cs',
                definition='重复执行代码块的结构',
                difficulty=0.3,
                examples=['for (int i=0; i<10; i++) {}', 'while (condition) {}'],
                keywords=['loop', 'iteration'],
                prerequisites=['cs:programming:variable'],
                related=['cs:programming:condition'],
                tags=['programming', 'control flow'],
            ),
            KnowledgeUnit(
                id='cs:programming:condition',
                name='条件语句',
                domain='cs',
                definition='根据条件执行不同代码的结构',
                difficulty=0.3,
                examples=['if (x > 0) {} else {}'],
                keywords=['condition', 'if', 'else'],
                prerequisites=['cs:programming:variable'],
                related=['cs:programming:loop'],
                tags=['programming', 'control flow'],
            ),
            KnowledgeUnit(
                id='cs:programming:function',
                name='函数',
                domain='cs',
                definition='可重用的代码块，接受输入并返回输出',
                difficulty=0.4,
                examples=['def add(a, b): return a + b'],
                keywords=['function', 'method', 'procedure'],
                prerequisites=['cs:programming:variable'],
                related=['cs:programming:recursion'],
                tags=['programming'],
            ),
            # 数据结构
            KnowledgeUnit(
                id='cs:ds:array',
                name='数组',
                domain='cs',
                definition='连续存储的同类型元素集合',
                difficulty=0.3,
                examples=['int[] arr = {1, 2, 3};'],
                keywords=['array', 'list'],
                prerequisites=['cs:programming:variable'],
                related=['cs:ds:linkedlist'],
                tags=['data structure'],
            ),
            KnowledgeUnit(
                id='cs:ds:linkedlist',
                name='链表',
                domain='cs',
                definition='通过指针连接的节点序列',
                difficulty=0.4,
                examples=['单链表', '双链表', '循环链表'],
                keywords=['linked list', 'node'],
                prerequisites=['cs:programming:variable'],
                related=['cs:ds:array'],
                tags=['data structure'],
            ),
            KnowledgeUnit(
                id='cs:ds:stack',
                name='栈',
                domain='cs',
                definition='后进先出（LIFO）的数据结构',
                difficulty=0.3,
                examples=['push', 'pop', 'peek'],
                keywords=['stack', 'LIFO'],
                prerequisites=['cs:ds:array'],
                related=['cs:ds:queue'],
                tags=['data structure'],
            ),
            KnowledgeUnit(
                id='cs:ds:queue',
                name='队列',
                domain='cs',
                definition='先进先出（FIFO）的数据结构',
                difficulty=0.3,
                examples=['enqueue', 'dequeue'],
                keywords=['queue', 'FIFO'],
                prerequisites=['cs:ds:array'],
                related=['cs:ds:stack'],
                tags=['data structure'],
            ),
            KnowledgeUnit(
                id='cs:ds:tree',
                name='树',
                domain='cs',
                definition='分层的数据结构，每个节点最多有一个父节点',
                difficulty=0.5,
                examples=['二叉树', '二叉搜索树', 'AVL树'],
                keywords=['tree', 'binary tree'],
                prerequisites=['cs:ds:linkedlist'],
                related=['cs:ds:graph'],
                tags=['data structure'],
            ),
            KnowledgeUnit(
                id='cs:ds:graph',
                name='图',
                domain='cs',
                definition='由顶点和边组成的非线性数据结构',
                difficulty=0.6,
                examples=['有向图', '无向图', '加权图'],
                keywords=['graph', 'vertex', 'edge'],
                prerequisites=['cs:ds:tree'],
                related=['cs:ds:tree'],
                tags=['data structure'],
            ),
            # 算法
            KnowledgeUnit(
                id='cs:algo:sort',
                name='排序算法',
                domain='cs',
                definition='将数据按特定顺序排列的算法',
                difficulty=0.4,
                examples=['冒泡排序', '快速排序', '归并排序'],
                keywords=['sort', 'algorithm'],
                prerequisites=['cs:ds:array'],
                related=['cs:algo:search'],
                tags=['algorithm'],
            ),
            KnowledgeUnit(
                id='cs:algo:search',
                name='搜索算法',
                domain='cs',
                definition='在数据中查找特定元素的算法',
                difficulty=0.4,
                examples=['线性搜索', '二分搜索'],
                keywords=['search', 'find'],
                prerequisites=['cs:ds:array'],
                related=['cs:algo:sort'],
                tags=['algorithm'],
            ),
            KnowledgeUnit(
                id='cs:algo:recursion',
                name='递归',
                domain='cs',
                definition='函数调用自身的编程技术',
                difficulty=0.5,
                examples=['阶乘', '斐波那契数列'],
                keywords=['recursion', 'recursive'],
                prerequisites=['cs:programming:function'],
                related=['cs:algo:dynamic'],
                tags=['algorithm'],
            ),
            KnowledgeUnit(
                id='cs:algo:dynamic',
                name='动态规划',
                domain='cs',
                definition='通过将问题分解为子问题来求解的算法思想',
                difficulty=0.7,
                examples=['背包问题', '最长公共子序列'],
                keywords=['dynamic programming', 'DP'],
                prerequisites=['cs:algo:recursion'],
                related=['cs:algo:greedy'],
                tags=['algorithm'],
            ),
        ]


class ChemistryDataSource(DataSource):
    """化学数据源 — 从预定义化学概念加载"""

    def get_domain(self) -> str:
        return 'chemistry'

    def load_units(self) -> List[KnowledgeUnit]:
        """加载化学概念"""
        return [
            # 基础概念
            KnowledgeUnit(
                id='chem:basic:atom',
                name='原子',
                domain='chemistry',
                definition='化学变化中的最小微粒',
                difficulty=0.2,
                examples=['氢原子 H', '氧原子 O', '碳原子 C'],
                keywords=['atom', 'element'],
                prerequisites=[],
                related=['chem:basic:molecule'],
                tags=['basic'],
            ),
            KnowledgeUnit(
                id='chem:basic:molecule',
                name='分子',
                domain='chemistry',
                definition='保持物质化学性质的最小微粒',
                difficulty=0.2,
                examples=['水分子 H₂O', '氧气分子 O₂'],
                keywords=['molecule', 'compound'],
                prerequisites=['chem:basic:atom'],
                related=['chem:basic:atom'],
                tags=['basic'],
            ),
            KnowledgeUnit(
                id='chem:basic:element',
                name='元素',
                domain='chemistry',
                definition='具有相同核电荷数的一类原子的总称',
                difficulty=0.2,
                examples=['氢 H', '氧 O', '碳 C', '铁 Fe'],
                keywords=['element', 'periodic table'],
                prerequisites=['chem:basic:atom'],
                related=['chem:basic:compound'],
                tags=['basic'],
            ),
            KnowledgeUnit(
                id='chem:basic:compound',
                name='化合物',
                domain='chemistry',
                definition='由两种或两种以上元素组成的纯净物',
                difficulty=0.3,
                examples=['水 H₂O', '二氧化碳 CO₂', '氯化钠 NaCl'],
                keywords=['compound', 'chemical formula'],
                prerequisites=['chem:basic:element'],
                related=['chem:basic:mixture'],
                tags=['basic'],
            ),
            KnowledgeUnit(
                id='chem:basic:mixture',
                name='混合物',
                domain='chemistry',
                definition='由两种或多种物质混合而成',
                difficulty=0.2,
                examples=['空气', '海水', '合金'],
                keywords=['mixture', 'solution'],
                prerequisites=['chem:basic:compound'],
                related=['chem:basic:compound'],
                tags=['basic'],
            ),
            # 化学反应
            KnowledgeUnit(
                id='chem:reaction:synthesis',
                name='化合反应',
                domain='chemistry',
                definition='两种或多种物质生成一种新物质的反应',
                difficulty=0.3,
                examples=['2H₂ + O₂ → 2H₂O'],
                keywords=['synthesis', 'combination'],
                prerequisites=['chem:basic:compound'],
                related=['chem:reaction:decomposition'],
                tags=['reaction'],
            ),
            KnowledgeUnit(
                id='chem:reaction:decomposition',
                name='分解反应',
                domain='chemistry',
                definition='一种物质分解成两种或多种新物质的反应',
                difficulty=0.3,
                examples=['2H₂O → 2H₂ + O₂'],
                keywords=['decomposition', 'breakdown'],
                prerequisites=['chem:basic:compound'],
                related=['chem:reaction:synthesis'],
                tags=['reaction'],
            ),
            KnowledgeUnit(
                id='chem:reaction:combustion',
                name='燃烧反应',
                domain='chemistry',
                definition='物质与氧气发生的剧烈氧化反应',
                difficulty=0.3,
                examples=['CH₄ + 2O₂ → CO₂ + 2H₂O'],
                keywords=['combustion', 'burn'],
                prerequisites=['chem:reaction:synthesis'],
                related=['chem:reaction:oxidation'],
                tags=['reaction'],
            ),
            # 化学键
            KnowledgeUnit(
                id='chem:bond:ionic',
                name='离子键',
                domain='chemistry',
                definition='阴阳离子之间通过静电作用形成的化学键',
                difficulty=0.4,
                examples=['NaCl', 'KCl'],
                keywords=['ionic bond', 'ion'],
                prerequisites=['chem:basic:atom'],
                related=['chem:bond:covalent'],
                tags=['bond'],
            ),
            KnowledgeUnit(
                id='chem:bond:covalent',
                name='共价键',
                domain='chemistry',
                definition='原子间通过共用电子对形成的化学键',
                difficulty=0.4,
                examples=['H₂', 'O₂', 'H₂O'],
                keywords=['covalent bond', 'shared electrons'],
                prerequisites=['chem:basic:atom'],
                related=['chem:bond:ionic'],
                tags=['bond'],
            ),
        ]


class ManualDataSource(DataSource):
    """手动数据源 — 用户手动添加知识单元"""

    def __init__(self, domain: str):
        self._domain = domain
        self._units: List[KnowledgeUnit] = []

    def get_domain(self) -> str:
        return self._domain

    def add_unit(self, unit: KnowledgeUnit) -> None:
        """添加知识单元"""
        self._units.append(unit)

    def load_units(self) -> List[KnowledgeUnit]:
        return self._units.copy()


class FileDataSource(DataSource):
    """文件数据源 — 从 JSON 文件加载知识单元"""

    def __init__(self, file_path: str, domain: str):
        self.file_path = file_path
        self._domain = domain

    def get_domain(self) -> str:
        return self._domain

    def load_units(self) -> List[KnowledgeUnit]:
        """从 JSON 文件加载"""
        if not os.path.exists(self.file_path):
            return []

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            units = []
            for item in data:
                unit = KnowledgeUnit.from_dict(item)
                units.append(unit)
            return units
        except Exception:
            return []
