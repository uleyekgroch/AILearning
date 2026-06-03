"""
批量学习Wiki语料库
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.learner import Learner
from src.core.config import LearnerConfig

def main():
    config = LearnerConfig(device='cpu')
    learner = Learner(config)

    wiki_dir = Path('data/extracted/wiki/wiki_zh')

    # 收集所有文件
    all_files = []
    for letter_dir in sorted(wiki_dir.iterdir()):
        if letter_dir.is_dir():
            for file_path in sorted(letter_dir.iterdir()):
                if file_path.is_file():
                    all_files.append(file_path)

    total_files = len(all_files)
    print(f"找到 {total_files} 个文件")

    # 限制学习的文件数量（避免一次性学习太多）
    max_files = 500  # 先学习500个文件
    files_to_learn = all_files[:max_files]

    start_time = time.time()
    total_texts = 0

    for i, file_path in enumerate(files_to_learn):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 按行分割，每行作为一个文本
            lines = [line.strip() for line in content.split('\n') if line.strip()]

            # 学习每行文本
            for line in lines[:50]:  # 每个文件最多学50行
                if len(line) >= 10:  # 过滤太短的文本
                    learner.learn_from_text(line)
                    total_texts += 1

            # 每10个文件报告进度
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                speed = total_texts / elapsed if elapsed > 0 else 0
                print(f"[{i+1}/{max_files}] 学了{total_texts}条, "
                      f"{elapsed:.1f}s, {speed:.1f}条/s, "
                      f"STDP连接:{len(learner._stdp_system['connections'])}, "
                      f"海马记忆:{len(learner._hippocampal_memory['episodes'])}")

        except Exception as e:
            print(f"处理文件 {file_path} 失败: {e}")
            continue

    total_time = time.time() - start_time
    print(f"\n=== 学习完成 ===")
    print(f"文件数: {len(files_to_learn)}")
    print(f"文本数: {total_texts}")
    print(f"总耗时: {total_time:.1f}s")
    print(f"平均速度: {total_texts/total_time:.1f}条/s")
    print(f"STDP连接数: {len(learner._stdp_system['connections'])}")
    print(f"海马记忆数: {len(learner._hippocampal_memory['episodes'])}")
    print(f"睡眠巩固次数: {learner._sleep_consolidation['consolidation_count']}")
    print(f"词向量数: {len(learner._perception_vectors)}")

    # 测试推理
    print(f"\n=== 测试推理 ===")
    test_questions = [
        "什么是光合作用",
        "地球绕什么公转",
        "水在多少度沸腾",
    ]

    for q in test_questions:
        answer = learner.think(q)
        print(f"Q: {q}")
        print(f"A: {answer[:100] if answer else '(无答案)'}")
        print()

if __name__ == '__main__':
    main()
