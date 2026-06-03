"""用学习系统学习语料 — 真正的自主学习

流程：
1. 读取 Wikipedia 语料
2. 将每篇文章转化为学习系统的输入
3. 学习系统通过 WorldModel 发现规律
4. ConceptFormer 形成概念
5. LanguageGroundingSystem 将词汇接地
6. 知识存入学习系统的内部知识库

运行方式：
    python training/learn_corpus_with_system.py
"""

import json
import os
import sys
import re
import time
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.learning.world_model import WorldModel
from src.learning.concept_former import ConceptFormer
from src.learning.language_grounding_system import LanguageGroundingSystem


class CorpusEnvironment:
    """语料环境 — 将文本转化为学习系统的交互"""

    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.articles = []
        self.current_idx = 0
        self._load_articles()

    def _load_articles(self):
        """加载文章"""
        print(f"  加载语料: {self.zip_path}")
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            files = [f for f in z.namelist() if not f.endswith('/')]
            for file_path in files[:100]:  # 先加载100个文件
                try:
                    with z.open(file_path) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        for line in content.strip().split('\n'):
                            if not line.strip():
                                continue
                            try:
                                article = json.loads(line)
                                if article.get('title') and article.get('text'):
                                    self.articles.append(article)
                            except:
                                continue
                except:
                    continue
        print(f"  加载完成: {len(self.articles)} 篇文章")

    def get_state(self) -> dict:
        """获取当前状态"""
        if self.current_idx >= len(self.articles):
            return {}

        article = self.articles[self.current_idx]
        text = article.get('text', '')

        # 提取特征
        words = re.findall(r'[一-鿿]{2,4}', text)
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1

        # 取高频词作为特征
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        features = {w: min(1.0, c / 10) for w, c in top_words}

        return {
            'title': article.get('title', ''),
            'features': features,
            'text_length': len(text),
        }

    def get_available_actions(self) -> list:
        """获取可用动作"""
        return [
            {'type': 'read', 'article': self.current_idx},
            {'type': 'next'},
            {'type': 'analyze', 'depth': 'shallow'},
            {'type': 'analyze', 'depth': 'deep'},
        ]

    def step(self, action: dict) -> dict:
        """执行动作"""
        action_type = action.get('type', 'next')

        if action_type == 'next':
            self.current_idx = min(self.current_idx + 1, len(self.articles) - 1)

        state = self.get_state()

        # 构建下一状态
        next_state = dict(state.get('features', {}))

        return {
            'state': state.get('features', {}),
            'next_state': next_state,
            'context': {
                'word': state.get('title', ''),
                'text': self.articles[self.current_idx].get('text', '')[:500] if self.current_idx < len(self.articles) else '',
            },
        }

    def get_context(self) -> dict:
        """获取上下文"""
        if self.current_idx < len(self.articles):
            article = self.articles[self.current_idx]
            return {
                'word': article.get('title', ''),
                'text': article.get('text', '')[:500],
            }
        return {}


def main():
    print("=" * 70)
    print("用学习系统学习语料 — 真正的自主学习")
    print("=" * 70)

    # ── 初始化学习系统 ──────────────────────────────────────────────
    print("\n[1] 初始化学习系统...")
    world_model = WorldModel()
    concept_former = ConceptFormer()
    language_system = LanguageGroundingSystem()

    # ── 加载语料 ──────────────────────────────────────────────
    print("\n[2] 加载语料...")
    env = CorpusEnvironment('data/wiki_zh_2019.zip')

    # ── 学习循环 ──────────────────────────────────────────────
    print("\n[3] 开始学习...")
    total_steps = min(1000, len(env.articles))
    start_time = time.time()

    for step in range(total_steps):
        # 获取当前状态
        state = env.get_state()
        if not state:
            break

        # 执行动作（读取下一篇文章）
        action = {'type': 'next'}
        result = env.step(action)

        # 世界模型学习
        surprise = world_model.observe(
            result['state'],
            action,
            result['next_state'],
        )

        # 概念形成
        if result['next_state']:
            concept_id = concept_former.observe(result['next_state'])

        # 语言接地
        context = env.get_context()
        if context.get('word'):
            language_system.ground_word(
                context['word'],
                world_model,
                result['state'],
                action,
                result['next_state'],
            )

        # 进度输出
        if (step + 1) % 100 == 0:
            print(f"  [{step+1}/{total_steps}] "
                  f"surprise={surprise:.3f} "
                  f"rules={len(world_model.rules)} "
                  f"concepts={len(concept_former.concepts)} "
                  f"words={len(language_system.grounded_words)}")

        # 定期发现规律
        if (step + 1) % 50 == 0:
            world_model.discover_laws()

    elapsed = time.time() - start_time

    # ── 保存学习结果 ──────────────────────────────────────────────
    print("\n[4] 保存学习结果...")
    output_dir = Path('data/knowledge/learned')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存世界模型
    wm_data = world_model.get_knowledge_summary()
    with open(output_dir / 'world_model.json', 'w', encoding='utf-8') as f:
        json.dump(wm_data, f, ensure_ascii=False, indent=2)

    # 保存概念
    concepts_data = concept_former.get_stats()
    with open(output_dir / 'concepts.json', 'w', encoding='utf-8') as f:
        json.dump(concepts_data, f, ensure_ascii=False, indent=2)

    # 保存语言接地
    lang_data = language_system.get_stats()
    with open(output_dir / 'language.json', 'w', encoding='utf-8') as f:
        json.dump(lang_data, f, ensure_ascii=False, indent=2)

    print(f"  保存到: {output_dir}")

    # ── 统计 ──────────────────────────────────────────────
    print("\n[5] 学习统计:")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  学习步数: {total_steps}")
    print(f"  世界模型规则: {len(world_model.rules)}")
    print(f"  形成概念: {len(concept_former.concepts)}")
    print(f"  接地词汇: {len(language_system.grounded_words)}")
    print(f"  平均惊讶度: {world_model.avg_surprise:.3f}")

    # 显示发现的规律
    top_rules = wm_data.get('top_rules', [])
    if top_rules:
        print("\n  发现的规律:")
        for i, rule in enumerate(top_rules[:5], 1):
            print(f"    {i}. {rule['trigger']} → {rule['effect']} "
                  f"(置信度: {rule['confidence']:.3f})")

    print("\n" + "=" * 70)
    print("语料学习完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
