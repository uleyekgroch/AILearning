"""知识持久化层

保存和加载学习到的知识，支持跨会话学习。

核心能力：
1. 保存知识 — 将学习结果保存到文件
2. 加载知识 — 从文件加载学习结果
3. 增量保存 — 只保存新增的知识
4. 知识合并 — 合并多个学习会话的知识

运行方式：
    python training/knowledge_persistence.py
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


class KnowledgePersistence:
    """知识持久化层

    核心能力：
    - 保存和加载学习结果
    - 支持增量保存
    - 支持知识合并
    """

    def __init__(self, save_dir: str = 'data/knowledge/persisted'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 统计
        self.stats = {
            'saves': 0,
            'loads': 0,
            'knowledge_items': 0,
        }

    def save_knowledge(self, knowledge: Dict, name: str = 'default') -> str:
        """保存知识"""
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}.pkl"
        filepath = self.save_dir / filename

        # 添加元数据
        knowledge['_metadata'] = {
            'saved_at': timestamp,
            'name': name,
            'version': '1.0',
        }

        # 保存
        with open(filepath, 'wb') as f:
            pickle.dump(knowledge, f)

        self.stats['saves'] += 1
        self.stats['knowledge_items'] = len(knowledge)

        return str(filepath)

    def load_knowledge(self, name: str = 'default') -> Optional[Dict]:
        """加载知识（最新的）"""
        # 查找最新的文件
        pattern = f"{name}_*.pkl"
        files = list(self.save_dir.glob(pattern))

        if not files:
            return None

        # 按时间戳排序，取最新的
        latest_file = max(files, key=lambda f: f.stat().st_mtime)

        # 加载
        with open(latest_file, 'rb') as f:
            knowledge = pickle.load(f)

        self.stats['loads'] += 1
        return knowledge

    def list_saves(self, name: str = None) -> List[Dict]:
        """列出所有保存"""
        if name:
            pattern = f"{name}_*.pkl"
        else:
            pattern = "*.pkl"

        files = list(self.save_dir.glob(pattern))

        saves = []
        for f in files:
            saves.append({
                'name': f.stem.rsplit('_', 1)[0],
                'timestamp': f.stat().st_mtime,
                'size': f.stat().st_size,
                'path': str(f),
            })

        return sorted(saves, key=lambda x: x['timestamp'], reverse=True)

    def merge_knowledge(self, knowledge1: Dict, knowledge2: Dict) -> Dict:
        """合并两个知识库"""
        merged = knowledge1.copy()

        # 合并语义框架
        if 'semantic' in knowledge2:
            if 'semantic' not in merged:
                merged['semantic'] = {}
            merged['semantic'].update(knowledge2['semantic'])

        # 合并因果链接
        if 'causal' in knowledge2:
            if 'causal' not in merged:
                merged['causal'] = {'links': []}
            if 'links' in knowledge2['causal']:
                merged['causal']['links'].extend(knowledge2['causal']['links'])

        # 合并概念
        if 'concepts' in knowledge2:
            if 'concepts' not in merged:
                merged['concepts'] = {}
            merged['concepts'].update(knowledge2['concepts'])

        # 合并数值事实
        if 'numerical' in knowledge2:
            if 'numerical' not in merged:
                merged['numerical'] = {'facts': []}
            if 'facts' in knowledge2['numerical']:
                merged['numerical']['facts'].extend(knowledge2['numerical']['facts'])

        return merged

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'save_dir': str(self.save_dir),
            'total_files': len(list(self.save_dir.glob('*.pkl'))),
        }


def test_knowledge_persistence():
    """测试知识持久化"""
    print("=" * 70)
    print("知识持久化测试")
    print("=" * 70)

    persistence = KnowledgePersistence()

    # 测试保存
    test_knowledge = {
        'semantic': {'frames': 10, 'entities': 20},
        'causal': {'links': [('下雨', '地面湿了')]},
        'concepts': {'Python': {'type': '编程语言'}},
    }

    with open('persistence_test.txt', 'w', encoding='utf-8') as f:
        f.write('知识持久化测试\n')
        f.write('=' * 70 + '\n\n')

        # 保存
        filepath = persistence.save_knowledge(test_knowledge, 'test')
        f.write(f'保存到: {filepath}\n')

        # 加载
        loaded = persistence.load_knowledge('test')
        f.write(f'加载成功: {loaded is not None}\n')
        f.write(f'语义框架: {loaded.get("semantic", {}).get("frames", 0)}\n')

        # 列出保存
        saves = persistence.list_saves()
        f.write(f'\n保存列表 ({len(saves)} 个):\n')
        for save in saves[:5]:
            f.write(f'  {save["name"]}: {save["size"]} bytes\n')

        # 统计
        f.write('\n统计:\n')
        stats = persistence.get_stats()
        for k, v in stats.items():
            f.write(f'  {k}: {v}\n')

    print('Written to persistence_test.txt')


if __name__ == '__main__':
    test_knowledge_persistence()
