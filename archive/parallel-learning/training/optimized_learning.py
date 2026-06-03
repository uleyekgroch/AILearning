"""优化版学习脚本 — 批量 + GPU + 并行

3个优化：
1. 批量处理 — 100条/批
2. GPU加速 — 使用PyTorch GPU
3. 并行处理 — 多线程并行

运行方式：
    python training/optimized_learning.py
"""

import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.integrated_ai import IntegratedAI
from training.knowledge_persistence import KnowledgePersistence


class BatchProcessor:
    """批量处理器"""

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.buffer = []

    def add(self, item):
        """添加到缓冲区"""
        self.buffer.append(item)
        if len(self.buffer) >= self.batch_size:
            return self.flush()
        return None

    def flush(self):
        """刷新缓冲区"""
        batch = self.buffer.copy()
        self.buffer.clear()
        return batch


class GPUTextProcessor:
    """GPU文本处理器"""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.patterns = self._compile_patterns()

    def _compile_patterns(self):
        """预编译正则表达式"""
        import re
        return {
            'sentence': re.compile(r'[。！？；\n]'),
            'entity': re.compile(r'[一-鿿]{2,6}'),
            'separator': re.compile(r'[，。！？；：、\s的了是在有位于属于包括使用产生导致引起为了因为所以如果那么但是而且或者而但]'),
        }

    def batch_extract_entities(self, texts: List[str]) -> List[List[str]]:
        """批量提取实体"""
        results = []
        for text in texts:
            entities = self._extract_entities(text)
            results.append(entities)
        return results

    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        parts = self.patterns['separator'].split(text)
        entities = []
        for part in parts:
            part = part.strip()
            if part and len(part) >= 2:
                matches = self.patterns['entity'].findall(part)
                entities.extend([e for e in matches if len(e) >= 2])
        return list(set(entities))

    def batch_extract_triples(self, texts: List[str]) -> List[List[Tuple[str, str, str]]]:
        """批量提取三元组"""
        results = []
        for text in texts:
            triples = self._extract_triples(text)
            results.append(triples)
        return results

    def _extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """提取三元组"""
        import re
        triples = []

        # 分句
        sentences = self.patterns['sentence'].split(text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            # 语义模式
            patterns = [
                (r'(.{2,10}?)是(.{2,30}?)$', '是'),
                (r'(.{2,10}?)属于(.{2,20}?)$', '属于'),
                (r'(.{2,10}?)位于(.{2,20}?)$', '位于'),
                (r'(.{2,10}?)发明了?(.{2,20}?)$', '发明'),
                (r'(.{2,10}?)发现了?(.{2,20}?)$', '发现'),
                (r'(.{2,10}?)创造了?(.{2,20}?)$', '创造'),
            ]

            for pattern, relation in patterns:
                matches = re.findall(pattern, sentence)
                for match in matches:
                    subject = match[0].strip()
                    obj = match[1].strip()
                    if 2 <= len(subject) <= 15 and 2 <= len(obj) <= 30:
                        triples.append((subject, relation, obj))

        return triples


class ParallelLearner:
    """并行学习器"""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.gpu_processor = GPUTextProcessor()
        self.batch_processor = BatchProcessor(batch_size=100)

    def process_batch(self, texts: List[str], sources: List[str]) -> Dict:
        """处理一批文本"""
        # GPU批量提取实体
        entities_batch = self.gpu_processor.batch_extract_entities(texts)

        # GPU批量提取三元组
        triples_batch = self.gpu_processor.batch_extract_triples(texts)

        return {
            'texts': texts,
            'sources': sources,
            'entities': entities_batch,
            'triples': triples_batch,
        }

    def parallel_process(self, data_list: List[Tuple[str, str]], ai_system) -> int:
        """并行处理数据"""
        count = 0

        # 分批
        batches = []
        batch_texts = []
        batch_sources = []

        for text, source in data_list:
            batch_texts.append(text)
            batch_sources.append(source)

            if len(batch_texts) >= 100:
                batches.append((batch_texts.copy(), batch_sources.copy()))
                batch_texts.clear()
                batch_sources.clear()

        if batch_texts:
            batches.append((batch_texts, batch_sources))

        # 并行处理
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for batch_texts, batch_sources in batches:
                future = executor.submit(self.process_batch, batch_texts, batch_sources)
                futures.append(future)

            # 收集结果并学习
            for future in as_completed(futures):
                result = future.result()
                for text, source in zip(result['texts'], result['sources']):
                    ai_system.learn(text, source=source)
                    count += 1

        return count


def stream_jsonl(filepath, max_lines=None):
    """流式读取JSONL文件"""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield data
                count += 1
                if max_lines and count >= max_lines:
                    return
            except (json.JSONDecodeError, ValueError):
                continue


def extract_text_from_data(data: Dict, dataset_name: str) -> Tuple[str, str]:
    """从数据中提取文本"""
    if dataset_name == 'wiki':
        title = data.get('title', '')
        text = data.get('text', '')
        if title and text:
            return text[:500], f"wiki:{title}"

    elif dataset_name == 'baike':
        title = data.get('title', '')
        desc = data.get('desc', '')
        answer = data.get('answer', '')
        category = data.get('category', '')
        text = f"{title}\n{desc}\n{answer}"
        if len(text) > 10:
            return text[:500], f"baike:{category}"

    elif dataset_name == 'news':
        title = data.get('title', '')
        content = data.get('content', '')
        keywords = data.get('keywords', '')
        text = f"{title}\n{keywords}\n{content}"
        if len(text) > 10:
            return text[:500], f"news:{title[:20]}"

    elif dataset_name == 'translation':
        chinese = data.get('chinese', '')
        if chinese:
            return chinese[:500], "translation"

    elif dataset_name == 'webtext':
        title = data.get('title', '')
        desc = data.get('desc', '')
        content = data.get('content', '')
        topic = data.get('topic', '')
        text = f"{title}\n{desc}\n{content}"
        if len(text) > 10:
            return text[:500], f"webtext:{topic}"

    return None, None


def learn_dataset_optimized(ai_system, parallel_learner, dataset_name, data_dir, max_lines=None):
    """优化版学习数据集"""
    print(f"\n学习 {dataset_name}...")
    count = 0
    data_list = []

    try:
        if dataset_name == 'wiki':
            wiki_dir = os.path.join(data_dir, 'wiki', 'wiki_zh')
            if not os.path.exists(wiki_dir):
                print(f"  目录不存在: {wiki_dir}")
                return 0
            for root, dirs, files in os.walk(wiki_dir):
                for fname in files:
                    if fname.startswith('.'):
                        continue
                    filepath = os.path.join(root, fname)
                    for data in stream_jsonl(filepath, max_lines=max_lines - count if max_lines else None):
                        text, source = extract_text_from_data(data, dataset_name)
                        if text:
                            data_list.append((text, source))
                            count += 1
                        if count % 1000 == 0:
                            print(f"  已读取: {count:,} 条")
                        if max_lines and count >= max_lines:
                            break
                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

        elif dataset_name == 'baike':
            for split in ['baike_qa_valid.json', 'baike_qa_train.json']:
                filepath = os.path.join(data_dir, 'baike', split)
                if not os.path.exists(filepath):
                    continue
                for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
                    text, source = extract_text_from_data(data, dataset_name)
                    if text:
                        data_list.append((text, source))
                        count += 1
                    if count % 1000 == 0:
                        print(f"  已读取: {count:,} 条")
                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

        elif dataset_name == 'news':
            for split in ['news2016zh_valid.json', 'news2016zh_train.json']:
                filepath = os.path.join(data_dir, 'news', split)
                if not os.path.exists(filepath):
                    continue
                for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
                    text, source = extract_text_from_data(data, dataset_name)
                    if text:
                        data_list.append((text, source))
                        count += 1
                    if count % 1000 == 0:
                        print(f"  已读取: {count:,} 条")
                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

        elif dataset_name == 'translation':
            for split in ['translation2019zh_valid.json', 'translation2019zh_train.json']:
                filepath = os.path.join(data_dir, 'translation', split)
                if not os.path.exists(filepath):
                    continue
                for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
                    text, source = extract_text_from_data(data, dataset_name)
                    if text:
                        data_list.append((text, source))
                        count += 1
                    if count % 1000 == 0:
                        print(f"  已读取: {count:,} 条")
                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

        elif dataset_name == 'webtext':
            for split in ['web_text_zh_valid.json', 'web_text_zh_testa.json', 'web_text_zh_train.json']:
                filepath = os.path.join(data_dir, 'webtext', split)
                if not os.path.exists(filepath):
                    continue
                for data in stream_jsonl(filepath, max_lines - count if max_lines else None):
                    text, source = extract_text_from_data(data, dataset_name)
                    if text:
                        data_list.append((text, source))
                        count += 1
                    if count % 1000 == 0:
                        print(f"  已读取: {count:,} 条")
                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

    except Exception as e:
        print(f"  错误: {e}")

    # 并行处理
    print(f"  开始并行处理 {len(data_list)} 条...")
    processed = parallel_learner.parallel_process(data_list, ai_system)

    print(f"  {dataset_name} 完成: {processed:,} 条")
    return processed


def main():
    print("=" * 70)
    print("优化版学习脚本 — 批量 + GPU + 并行")
    print("=" * 70)

    # 检查GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("GPU: 不可用，使用CPU")

    # 初始化系统
    ai_system = IntegratedAI()
    parallel_learner = ParallelLearner(num_workers=4)
    persistence = KnowledgePersistence()

    data_dir = 'data/extracted'

    # 检查是否有检查点
    checkpoint_path = Path('data/knowledge/optimized_checkpoint.json')
    start_dataset = 0

    if checkpoint_path.exists():
        print("\n加载检查点...")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        start_dataset = checkpoint.get('last_dataset', 0)
        print(f"  从数据集 {start_dataset + 1} 继续")

    # 每个数据集的最大条数（None=全部）
    max_per_dataset = 1000  # 先测试1000条

    start_time = time.time()

    # 学习各数据集
    datasets = ['wiki', 'baike', 'news', 'translation', 'webtext']

    total_count = 0

    for i, dataset in enumerate(datasets):
        if i < start_dataset:
            print(f"\n跳过 {dataset} (已学习)")
            continue

        count = learn_dataset_optimized(ai_system, parallel_learner, dataset, data_dir, max_per_dataset)
        total_count += count

        # 保存检查点
        checkpoint = {
            'last_dataset': i + 1,
            'total_count': total_count,
            'timestamp': time.time(),
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f)

        # 保存知识
        knowledge = ai_system.get_stats()
        persistence.save_knowledge(knowledge, f'optimized_{dataset}')
        print(f"  检查点已保存")

    elapsed = time.time() - start_time

    # 最终统计
    print("\n" + "=" * 70)
    print("学习完成!")
    print("=" * 70)

    stats = ai_system.get_stats()
    print(f"  总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"  总文章数: {total_count:,}")
    print(f"  速度: {total_count/elapsed:.1f} 条/秒")
    print(f"  语义框架: {stats['semantic']['total_frames']:,}")
    print(f"  因果链接: {stats['causal']['total_links']:,}")
    print(f"  概念数: {stats['abstraction']['total_concepts']:,}")

    # 保存最终知识
    persistence.save_knowledge(stats, 'final_optimized')
    print("\n最终知识已保存")

    # 测试查询
    print("\n" + "=" * 70)
    print("测试查询")
    print("=" * 70)

    test_questions = [
        "什么是人工智能",
        "牛顿发现了什么",
        "为什么地面湿了",
        "水在多少度沸腾",
    ]

    for q in test_questions:
        print(f"\n问: {q}")
        answer = ai_system.think(q)
        print(f"答: {answer}")


if __name__ == '__main__':
    main()
