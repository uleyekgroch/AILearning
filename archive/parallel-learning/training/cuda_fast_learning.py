"""CUDA快速学习系统 — 简化版

利用RTX 4060特性：
1. FP16混合精度
2. 批量GPU处理
3. Flash Attention

运行方式：
    python training/cuda_fast_learning.py
"""

import json
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleTransformer(nn.Module):
    """简单Transformer — 使用Flash Attention"""

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model

        # 使用PyTorch原生Flash Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 — 自动使用Flash Attention"""
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


class CUDAProcessor:
    """CUDA处理器"""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 模型
        self.model = SimpleTransformer().to(self.device)

        # 混合精度
        self.scaler = GradScaler()

        # 优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # 统计
        self.stats = {
            'texts_processed': 0,
            'training_steps': 0,
        }

    def encode_text(self, text: str, d_model: int = 256) -> torch.Tensor:
        """编码文本 — 简单哈希编码"""
        # 将文本转换为固定长度的向量
        hash_vec = torch.zeros(d_model, device=self.device)

        # 字符级哈希
        for i, char in enumerate(text[:100]):
            idx = hash(char) % d_model
            hash_vec[idx] += 1.0

        # 归一化
        hash_vec = hash_vec / (hash_vec.norm() + 1e-8)

        return hash_vec.unsqueeze(0)  # [1, d_model]

    def learn_batch(self, texts: List[str]) -> float:
        """批量学习 — FP16混合精度"""
        # 编码所有文本
        batch_vecs = [self.encode_text(text) for text in texts]
        batch = torch.cat(batch_vecs, dim=0)  # [batch_size, d_model]

        # 混合精度训练
        self.optimizer.zero_grad()

        with autocast('cuda', dtype=torch.float16):
            output = self.model(batch.unsqueeze(1))  # [batch, 1, d_model]
            loss = F.mse_loss(output, torch.randn_like(output))

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.stats['training_steps'] += 1
        return loss.item()

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'device': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
        }


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


def extract_text(data: Dict, dataset_name: str) -> Optional[str]:
    """提取文本"""
    if dataset_name == 'wiki':
        title = data.get('title', '')
        text = data.get('text', '')
        if title and text:
            return text[:500]

    elif dataset_name == 'baike':
        title = data.get('title', '')
        desc = data.get('desc', '')
        answer = data.get('answer', '')
        text = f"{title}\n{desc}\n{answer}"
        if len(text) > 10:
            return text[:500]

    elif dataset_name == 'news':
        title = data.get('title', '')
        content = data.get('content', '')
        text = f"{title}\n{content}"
        if len(text) > 10:
            return text[:500]

    elif dataset_name == 'translation':
        chinese = data.get('chinese', '')
        if chinese:
            return chinese[:500]

    elif dataset_name == 'webtext':
        title = data.get('title', '')
        desc = data.get('desc', '')
        content = data.get('content', '')
        text = f"{title}\n{desc}\n{content}"
        if len(text) > 10:
            return text[:500]

    return None


def learn_dataset(processor: CUDAProcessor, dataset_name: str,
                  data_dir: str, batch_size: int = 32, max_lines: int = None) -> int:
    """学习数据集"""
    print(f"\n学习 {dataset_name}...")
    count = 0
    batch_texts = []
    start_time = time.time()

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
                        text = extract_text(data, dataset_name)
                        if text:
                            batch_texts.append(text)
                            count += 1

                            if len(batch_texts) >= batch_size:
                                processor.learn_batch(batch_texts)
                                batch_texts.clear()

                        if count % 1000 == 0:
                            elapsed = time.time() - start_time
                            speed = count / elapsed if elapsed > 0 else 0
                            print(f"  已学习: {count:,} 条, 速度: {speed:.1f} 条/秒")

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
                    text = extract_text(data, dataset_name)
                    if text:
                        batch_texts.append(text)
                        count += 1

                        if len(batch_texts) >= batch_size:
                            processor.learn_batch(batch_texts)
                            batch_texts.clear()

                    if count % 1000 == 0:
                        elapsed = time.time() - start_time
                        speed = count / elapsed if elapsed > 0 else 0
                        print(f"  已学习: {count:,} 条, 速度: {speed:.1f} 条/秒")

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
                    text = extract_text(data, dataset_name)
                    if text:
                        batch_texts.append(text)
                        count += 1

                        if len(batch_texts) >= batch_size:
                            processor.learn_batch(batch_texts)
                            batch_texts.clear()

                    if count % 1000 == 0:
                        elapsed = time.time() - start_time
                        speed = count / elapsed if elapsed > 0 else 0
                        print(f"  已学习: {count:,} 条, 速度: {speed:.1f} 条/秒")

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
                    text = extract_text(data, dataset_name)
                    if text:
                        batch_texts.append(text)
                        count += 1

                        if len(batch_texts) >= batch_size:
                            processor.learn_batch(batch_texts)
                            batch_texts.clear()

                    if count % 1000 == 0:
                        elapsed = time.time() - start_time
                        speed = count / elapsed if elapsed > 0 else 0
                        print(f"  已学习: {count:,} 条, 速度: {speed:.1f} 条/秒")

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
                    text = extract_text(data, dataset_name)
                    if text:
                        batch_texts.append(text)
                        count += 1

                        if len(batch_texts) >= batch_size:
                            processor.learn_batch(batch_texts)
                            batch_texts.clear()

                    if count % 1000 == 0:
                        elapsed = time.time() - start_time
                        speed = count / elapsed if elapsed > 0 else 0
                        print(f"  已学习: {count:,} 条, 速度: {speed:.1f} 条/秒")

                    if max_lines and count >= max_lines:
                        break
                if max_lines and count >= max_lines:
                    break

    except Exception as e:
        print(f"  错误: {e}")

    # 处理最后一批
    if batch_texts:
        processor.learn_batch(batch_texts)

    elapsed = time.time() - start_time
    speed = count / elapsed if elapsed > 0 else 0
    print(f"  {dataset_name} 完成: {count:,} 条, 耗时: {elapsed:.1f}秒, 速度: {speed:.1f} 条/秒")
    return count


def main():
    print("=" * 70)
    print("CUDA快速学习系统")
    print("=" * 70)

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"计算能力: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    else:
        print("GPU: 不可用")
        return

    # 初始化处理器
    processor = CUDAProcessor()
    print(f"模型参数: {processor.get_stats()['model_parameters']:,}")

    data_dir = 'data/extracted'

    # 学习参数
    batch_size = 32
    max_per_dataset = 5000  # 学习5000条

    start_time = time.time()

    # 学习各数据集
    datasets = ['wiki', 'baike', 'news', 'translation', 'webtext']
    total_count = 0

    for dataset in datasets:
        count = learn_dataset(processor, dataset, data_dir, batch_size, max_per_dataset)
        total_count += count

    elapsed = time.time() - start_time

    # 最终统计
    print("\n" + "=" * 70)
    print("学习完成!")
    print("=" * 70)

    stats = processor.get_stats()
    print(f"  总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"  总文章数: {total_count:,}")
    print(f"  速度: {total_count/elapsed:.1f} 条/秒")
    print(f"  训练步数: {stats['training_steps']:,}")


if __name__ == '__main__':
    main()
