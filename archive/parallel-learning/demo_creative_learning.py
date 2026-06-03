"""
创造性学习演示 - 像小朋友一样学习

展示真正的学习能力：
1. 学习知识
2. 写文章
3. 修改文章
4. 改进代码
5. 自我编码
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.production.domain.creative_writing.writer import CreativeWriter, WritingPrompt
from src.production.domain.creative_writing.editor import ArticleEditor
from src.production.domain.creative_writing.self_coder import SelfCoder


def demo_writing():
    """演示写作能力"""
    print("=" * 60)
    print("1. 写作演示 - 学习后写作")
    print("=" * 60)

    writer = CreativeWriter()

    # 学习知识
    print("\n学习数学知识...")
    writer.learn_knowledge("数学", [
        "数学是研究数量、结构、变化的学科",
        "数学包括算术、代数、几何等分支",
        "数学是科学的基础",
        "数学帮助我们理解世界",
    ])

    writer.learn_knowledge("物理", [
        "物理研究物质和能量",
        "牛顿发现万有引力",
        "爱因斯坦提出相对论",
        "量子力学研究微观世界",
    ])

    # 写不同风格的文章
    styles = ['narrative', 'descriptive', 'persuasive', 'expository']

    for style in styles:
        prompt = WritingPrompt(topic="数学", style=style)
        work = writer.write(prompt)
        print(f"\n【{style}风格】")
        print(f"标题: {work.title}")
        print(f"内容: {work.content[:100]}...")
        print(f"字数: {work.word_count}")

    # 统计
    stats = writer.get_writing_stats()
    print(f"\n写作统计: {stats}")


def demo_editing():
    """演示编辑能力"""
    print("\n" + "=" * 60)
    print("2. 编辑演示 - 修改文章")
    print("=" * 60)

    editor = ArticleEditor()

    # 原文
    original = "这是一个美丽的公园。公园里有很多很多的树。树上有许多许多的鸟。"
    print(f"\n原文: {original}")

    # 分析
    analysis = editor.analyze(original)
    print(f"\n分析结果:")
    print(f"  字数: {analysis['word_count']}")
    print(f"  句数: {analysis['sentence_count']}")
    print(f"  问题: {analysis['problems']}")
    print(f"  质量: {analysis['quality_score']:.2f}")

    # 编辑
    result = editor.edit(original)
    print(f"\n编辑后: {result.edited_text}")
    print(f"建议: {result.suggestions}")
    print(f"改进: {result.improvements}")


def demo_coding():
    """演示编码能力"""
    print("\n" + "=" * 60)
    print("3. 编码演示 - 改进代码")
    print("=" * 60)

    coder = SelfCoder()

    # 原始代码
    code = '''
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total

def calculate_average(numbers):
    total = calculate_sum(numbers)
    count = len(numbers)
    average = total / count
    return average

def find_max(numbers):
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
'''
    print(f"\n原始代码:")
    print(code)

    # 分析代码
    analysis = coder.analyze_code(code, "math_utils.py")
    print(f"\n代码分析:")
    print(f"  文件: {analysis.file_path}")
    print(f"  函数: {analysis.functions}")
    print(f"  行数: {analysis.lines}")
    print(f"  复杂度: {analysis.complexity:.2f}")
    print(f"  问题: {analysis.issues}")

    # 改进代码
    improvements = coder.improve_code(code)
    print(f"\n改进建议:")
    for imp in improvements:
        print(f"  - {imp.reason} ({imp.improvement_type})")


def demo_full_cycle():
    """演示完整循环"""
    print("\n" + "=" * 60)
    print("4. 完整循环 - 学习 -> 写作 -> 编辑 -> 改进")
    print("=" * 60)

    # 1. 学习
    print("\n【步骤1】学习知识...")
    writer = CreativeWriter()
    writer.learn_knowledge("人工智能", [
        "人工智能是计算机科学的分支",
        "机器学习是AI的核心技术",
        "深度学习使用神经网络",
        "AI可以学习和推理",
    ])

    # 2. 写作
    print("\n【步骤2】写作...")
    prompt = WritingPrompt(topic="人工智能", style="expository")
    work = writer.write(prompt)
    print(f"作品: {work.content[:100]}...")

    # 3. 编辑
    print("\n【步骤3】编辑...")
    editor = ArticleEditor()
    edit_result = editor.edit(work.content)
    print(f"编辑后: {edit_result.edited_text[:100]}...")

    # 4. 改进代码
    print("\n【步骤4】改进代码...")
    coder = SelfCoder()
    sample_code = "def hello(): print('hello')"
    improvements = coder.improve_code(sample_code)
    print(f"代码改进建议: {len(improvements)} 条")

    print("\n【完成】学习 -> 写作 -> 编辑 -> 改进 循环完成！")


def main():
    """主函数"""
    print("=" * 60)
    print("创造性学习系统演示")
    print("像小朋友一样学习：理解、创造、改进")
    print("=" * 60)

    demo_writing()
    demo_editing()
    demo_coding()
    demo_full_cycle()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n总结：")
    print("1. 学习能力：学习知识并理解")
    print("2. 写作能力：根据知识创作文章")
    print("3. 编辑能力：发现问题并改进")
    print("4. 编码能力：分析代码并优化")
    print("5. 自我改进：持续学习和进化")


if __name__ == '__main__':
    main()
