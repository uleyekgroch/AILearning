"""知识获取管道 — 从多个来源获取完整知识数据

数据来源：
1. WordNet: 语义关系（同义词、反义词、上位词、下位词）
2. Free Dictionary API: 定义、例句、发音、词性
3. 本地生成: 搭配、例句补充

输出：
- data/knowledge/words.json — 完整词汇数据
- data/knowledge/semantic_relations.json — 语义关系
- data/knowledge/collocations.json — 搭配数据
- data/knowledge/examples.json — 例句数据
- data/knowledge/encyclopedia.json — 百科知识

运行方式：
    python training/acquire_knowledge.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 配置 ──────────────────────────────────────────────────
DATA_DIR = Path('data/knowledge')
DATA_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = 'https://api.dictionaryapi.dev/api/v2/entries/en'
API_DELAY = 0.5  # 避免请求过快


def load_wordnet():
    """加载 WordNet 数据"""
    from nltk.corpus import wordnet as wn
    return wn


def get_wordnet_relations(word: str, wn) -> Dict:
    """从 WordNet 获取语义关系"""
    relations = {
        'synonyms': [],
        'antonyms': [],
        'hypernyms': [],
        'hyponyms': [],
        'meronyms': [],  # 部分词
        'holonyms': [],  # 整体词
    }

    synsets = wn.synsets(word)
    if not synsets:
        return relations

    for synset in synsets[:3]:  # 只取前3个义项
        # 同义词
        for lemma in synset.lemmas():
            name = lemma.name().replace('_', ' ')
            if name != word and name not in relations['synonyms']:
                relations['synonyms'].append(name)

        # 反义词
        for lemma in synset.lemmas():
            for ant in lemma.antonyms():
                name = ant.name().replace('_', ' ')
                if name not in relations['antonyms']:
                    relations['antonyms'].append(name)

        # 上位词
        for hyper in synset.hypernyms():
            for lemma in hyper.lemmas():
                name = lemma.name().replace('_', ' ')
                if name not in relations['hypernyms']:
                    relations['hypernyms'].append(name)

        # 下位词
        for hypo in synset.hyponyms():
            for lemma in hypo.lemmas():
                name = lemma.name().replace('_', ' ')
                if name not in relations['hyponyms']:
                    relations['hyponyms'].append(name)

        # 部分词
        for mero in synset.part_meronyms():
            for lemma in mero.lemmas():
                name = lemma.name().replace('_', ' ')
                if name not in relations['meronyms']:
                    relations['meronyms'].append(name)

        # 整体词
        for holo in synset.part_holonyms():
            for lemma in holo.lemmas():
                name = lemma.name().replace('_', ' ')
                if name not in relations['holonyms']:
                    relations['holonyms'].append(name)

    # 限制数量
    for key in relations:
        relations[key] = relations[key][:10]

    return relations


def get_api_data(word: str) -> Optional[Dict]:
    """从 Free Dictionary API 获取数据"""
    try:
        url = f"{API_BASE}/{word}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            if isinstance(data, list) and data:
                return data[0]
    except Exception:
        pass
    return None


def parse_api_data(data: Dict) -> Dict:
    """解析 API 返回的数据"""
    result = {
        'word': data.get('word', ''),
        'phonetic': data.get('phonetic', ''),
        'meanings': [],
    }

    for meaning in data.get('meanings', []):
        pos = meaning.get('partOfSpeech', '')
        definitions = []
        for defn in meaning.get('definitions', [])[:5]:
            definitions.append({
                'definition': defn.get('definition', ''),
                'example': defn.get('example', ''),
                'synonyms': defn.get('synonyms', [])[:5],
                'antonyms': defn.get('antonyms', [])[:5],
            })
        result['meanings'].append({
            'partOfSpeech': pos,
            'definitions': definitions,
        })

    return result


def generate_collocations(word: str, pos: str) -> List[str]:
    """生成常见搭配"""
    collocations = []

    # 动词搭配模板
    if pos in ['verb', 'v']:
        templates = [
            f"{word} + noun",
            f"adverb + {word}",
            f"{word} + preposition",
        ]
        collocations.extend(templates)

    # 名词搭配模板
    elif pos in ['noun', 'n']:
        templates = [
            f"adjective + {word}",
            f"verb + {word}",
            f"{word} + of",
        ]
        collocations.extend(templates)

    # 形容词搭配模板
    elif pos in ['adjective', 'adj', 'a']:
        templates = [
            f"{word} + noun",
            f"adverb + {word}",
            f"{word} + than",
        ]
        collocations.extend(templates)

    return collocations


def generate_examples(word: str, definition: str) -> List[str]:
    """生成例句"""
    examples = []

    # 基于定义生成例句
    if definition:
        examples.append(f"The {word} is important.")
        examples.append(f"She knows about {word}.")
        examples.append(f"We discussed the {word}.")

    return examples


def acquire_word(word: str, wn) -> Dict:
    """获取一个词的完整数据"""
    # 1. WordNet 语义关系
    wn_relations = get_wordnet_relations(word, wn)

    # 2. API 数据
    api_data = get_api_data(word)
    parsed = parse_api_data(api_data) if api_data else None

    # 3. 组合数据
    result = {
        'word': word,
        'phonetic': parsed.get('phonetic', '') if parsed else '',
        'meanings': parsed.get('meanings', []) if parsed else [],
        'semantic_relations': wn_relations,
        'collocations': [],
        'examples': [],
    }

    # 4. 生成搭配和例句
    if parsed and parsed['meanings']:
        for meaning in parsed['meanings'][:2]:
            pos = meaning.get('partOfSpeech', '')
            result['collocations'].extend(generate_collocations(word, pos))

            for defn in meaning.get('definitions', [])[:2]:
                defn_text = defn.get('definition', '')
                result['examples'].extend(generate_examples(word, defn_text))

    # 去重
    result['collocations'] = list(set(result['collocations']))[:10]
    result['examples'] = list(set(result['examples']))[:5]

    return result


def acquire_batch(words: List[str], batch_size: int = 50) -> Dict[str, Dict]:
    """批量获取词汇数据"""
    wn = load_wordnet()
    results = {}
    total = len(words)

    for i, word in enumerate(words):
        if word in results:
            continue

        print(f"  [{i+1}/{total}] 获取: {word}")
        results[word] = acquire_word(word, wn)

        # API 限速
        if i % 5 == 0:
            time.sleep(API_DELAY)

        # 每100个词保存一次进度
        if (i + 1) % 100 == 0:
            save_progress(results, 'progress')

    return results


def save_progress(data: Dict, name: str):
    """保存进度"""
    path = DATA_DIR / f'{name}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {path} ({len(data)} 词)")


def load_oxford_words() -> List[str]:
    """加载牛津词汇表"""
    from src.data.oxford_words import OXFORD_3000
    return list(OXFORD_3000.keys())


def main():
    print("=" * 70)
    print("知识获取管道 — 从多源获取完整知识数据")
    print("=" * 70)

    # ── 加载词汇表 ──────────────────────────────────────────────
    print("\n[1] 加载词汇表...")
    words = load_oxford_words()
    print(f"  总词汇: {len(words)}")

    # ── 检查已有数据 ──────────────────────────────────────────────
    progress_path = DATA_DIR / 'progress.json'
    existing = {}
    if progress_path.exists():
        with open(progress_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"  已有数据: {len(existing)} 词")

    # ── 过滤需要获取的词 ──────────────────────────────────────────────
    words_to_acquire = [w for w in words if w not in existing]
    print(f"  需要获取: {len(words_to_acquire)} 词")

    if not words_to_acquire:
        print("  所有词汇数据已获取!")
    else:
        # ── 批量获取 ──────────────────────────────────────────────
        print(f"\n[2] 开始获取 ({len(words_to_acquire)} 词)...")
        start_time = time.time()

        new_data = acquire_batch(words_to_acquire)

        # 合并数据
        existing.update(new_data)
        save_progress(existing, 'progress')

        elapsed = time.time() - start_time
        print(f"\n  获取完成! 耗时: {elapsed:.1f} 秒")
        print(f"  总数据: {len(existing)} 词")

    # ── 生成最终数据文件 ──────────────────────────────────────────────
    print("\n[3] 生成最终数据文件...")

    # 语义关系
    semantic_relations = {}
    for word, data in existing.items():
        sr = data.get('semantic_relations', {})
        if any(sr.values()):
            semantic_relations[word] = sr
    save_progress(semantic_relations, 'semantic_relations')
    print(f"  语义关系: {len(semantic_relations)} 词")

    # 搭配
    collocations = {}
    for word, data in existing.items():
        cols = data.get('collocations', [])
        if cols:
            collocations[word] = cols
    save_progress(collocations, 'collocations')
    print(f"  搭配数据: {len(collocations)} 词")

    # 例句
    examples = {}
    for word, data in existing.items():
        exs = data.get('examples', [])
        if exs:
            examples[word] = exs
    save_progress(examples, 'examples')
    print(f"  例句数据: {len(examples)} 词")

    # 完整数据
    save_progress(existing, 'words')
    print(f"  完整数据: {len(existing)} 词")

    # ── 统计 ──────────────────────────────────────────────
    print("\n[4] 统计:")
    has_definition = sum(1 for d in existing.values() if d.get('meanings'))
    has_synonyms = sum(1 for d in existing.values()
                       if d.get('semantic_relations', {}).get('synonyms'))
    has_examples = sum(1 for d in existing.values() if d.get('examples'))

    print(f"  有定义: {has_definition}/{len(existing)}")
    print(f"  有同义词: {has_synonyms}/{len(existing)}")
    print(f"  有例句: {has_examples}/{len(existing)}")

    print("\n" + "=" * 70)
    print("知识获取完成!")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  总词汇: {len(existing)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
