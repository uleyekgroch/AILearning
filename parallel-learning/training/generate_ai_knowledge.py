"""AI 知识生成脚本 — 使用通义千问生成高质量词汇知识

运行方式：
    python training/generate_ai_knowledge.py

功能：
1. 使用通义千问 API 为每个词汇生成：
   - 清晰定义
   - 真实例句
   - 同义词/反义词
   - 常见搭配
   - 百科简述
   - 相关概念
2. 保存到 data/knowledge/ai_generated.json
3. 整合到学习系统
"""

import json
import os
import sys
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 配置 ──────────────────────────────────────────────────
API_KEY = 'sk-b68b1187aba542c3b2fec09cdc02634c'
BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
MODEL = 'qwen-turbo-latest'  # 最快，适合大批量
DATA_DIR = Path('data/knowledge')
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
}

PROMPT_TEMPLATE = '''Word: {word}. Output JSON with keys: definition(1 sentence), examples(2 short), synonyms(3), antonyms(2), collocations(2), pos, encyclo(1 sentence). Only JSON.'''


def generate_knowledge(word: str) -> Optional[Dict]:
    """使用通义千问生成词汇知识"""
    prompt = PROMPT_TEMPLATE.format(word=word)

    data = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 500,
        'temperature': 0.3,
    }

    try:
        resp = requests.post(
            f'{BASE_URL}/chat/completions',
            headers=HEADERS,
            json=data,
            timeout=30,
        )

        if resp.status_code == 200:
            result = resp.json()
            content = result['choices'][0]['message']['content']

            # 解析 JSON
            try:
                # 尝试直接解析
                knowledge = json.loads(content)
                knowledge['word'] = word
                knowledge['source'] = 'ai_generated'
                return knowledge
            except json.JSONDecodeError:
                # 尝试提取 JSON 部分
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    try:
                        knowledge = json.loads(content[start:end])
                        knowledge['word'] = word
                        knowledge['source'] = 'ai_generated'
                        return knowledge
                    except:
                        pass

        return None
    except Exception as e:
        print(f"  Error for {word}: {e}")
        return None


def generate_batch(words: List[str], batch_size: int = 10) -> Dict[str, Dict]:
    """批量生成词汇知识"""
    results = {}
    total = len(words)

    # 加载已有进度
    progress_path = DATA_DIR / 'ai_progress.json'
    if progress_path.exists():
        with open(progress_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"  已有进度: {len(results)} 词")

    for i, word in enumerate(words):
        if word in results:
            continue

        print(f"  [{i+1}/{total}] 生成: {word}")
        knowledge = generate_knowledge(word)

        if knowledge:
            results[word] = knowledge
        else:
            results[word] = {'word': word, 'source': 'ai_failed'}

        # 限速
        time.sleep(0.5)

        # 每50个词保存一次
        if (i + 1) % 50 == 0:
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  已保存: {len(results)} 词")

    # 最终保存
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    print("=" * 70)
    print("AI 知识生成 — 使用通义千问生成高质量词汇知识")
    print("=" * 70)

    # ── 加载词汇表 ──────────────────────────────────────────────
    print("\n[1] 加载词汇表...")
    from src.data.oxford_words import OXFORD_3000
    words = list(OXFORD_3000.keys())
    print(f"  总词汇: {len(words)}")

    # ── 过滤需要生成的词 ──────────────────────────────────────────────
    progress_path = DATA_DIR / 'ai_progress.json'
    existing = {}
    if progress_path.exists():
        with open(progress_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)

    words_to_generate = [w for w in words if w not in existing]
    print(f"  已有: {len(existing)} 词")
    print(f"  需要生成: {len(words_to_generate)} 词")

    if not words_to_generate:
        print("  所有词汇已生成!")
    else:
        # ── 批量生成 ──────────────────────────────────────────────
        print(f"\n[2] 开始生成 ({len(words_to_generate)} 词)...")
        print(f"  模型: {MODEL}")
        start_time = time.time()

        results = generate_batch(words_to_generate)

        elapsed = time.time() - start_time
        print(f"\n  生成完成! 耗时: {elapsed:.1f} 秒")

    # ── 统计 ──────────────────────────────────────────────
    print("\n[3] 统计:")
    with open(progress_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    success = sum(1 for d in all_data.values() if d.get('source') == 'ai_generated')
    has_examples = sum(1 for d in all_data.values() if d.get('examples'))
    has_synonyms = sum(1 for d in all_data.values() if d.get('synonyms'))

    print(f"  总词数: {len(all_data)}")
    print(f"  成功生成: {success}")
    print(f"  有例句: {has_examples}")
    print(f"  有同义词: {has_synonyms}")

    # ── 保存最终数据 ──────────────────────────────────────────────
    output_path = DATA_DIR / 'ai_generated.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到: {output_path}")

    print("\n" + "=" * 70)
    print("AI 知识生成完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
