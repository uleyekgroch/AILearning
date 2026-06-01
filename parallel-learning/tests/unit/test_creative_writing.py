"""
创造性写作系统测试

测试真正的学习能力：
1. 写作能力
2. 编辑能力
3. 自我编码能力
"""

import pytest


class TestCreativeWriter:
    """创造性写作测试"""

    def test_learn_and_write(self):
        """测试学习后写作"""
        # Arrange
        from src.production.domain.creative_writing.writer import CreativeWriter, WritingPrompt

        writer = CreativeWriter()

        # 学习知识
        writer.learn_knowledge("数学", [
            "数学是研究数量、结构、变化的学科",
            "数学包括算术、代数、几何等分支",
            "数学是科学的基础",
        ])

        # Act
        prompt = WritingPrompt(
            topic="数学",
            style="expository",
            length="short",
        )
        work = writer.write(prompt)

        # Assert
        assert work is not None
        assert "数学" in work.content
        assert work.word_count > 0

    def test_write_different_styles(self):
        """测试不同风格写作"""
        # Arrange
        from src.production.domain.creative_writing.writer import CreativeWriter, WritingPrompt

        writer = CreativeWriter()
        writer.learn_knowledge("物理", ["物理研究物质和能量", "牛顿发现万有引力"])

        # Act - 不同风格
        styles = ['narrative', 'descriptive', 'persuasive', 'expository']
        works = []

        for style in styles:
            prompt = WritingPrompt(topic="物理", style=style)
            work = writer.write(prompt)
            works.append(work)

        # Assert
        assert len(works) == 4
        for work in works:
            assert work.content is not None
            assert len(work.content) > 0

    def test_writing_stats(self):
        """测试写作统计"""
        # Arrange
        from src.production.domain.creative_writing.writer import CreativeWriter, WritingPrompt

        writer = CreativeWriter()
        writer.learn_knowledge("化学", ["化学研究物质变化"])

        # Act
        for i in range(3):
            prompt = WritingPrompt(topic="化学", style="narrative")
            writer.write(prompt)

        stats = writer.get_writing_stats()

        # Assert
        assert stats['total_works'] == 3
        assert '化学' in stats['topics_written']


class TestArticleEditor:
    """文章编辑测试"""

    def test_analyze_article(self):
        """测试分析文章"""
        # Arrange
        from src.production.domain.creative_writing.editor import ArticleEditor

        editor = ArticleEditor()
        text = "这是一个美丽的公园。公园里有很多很多的树。"

        # Act
        analysis = editor.analyze(text)

        # Assert
        assert analysis['word_count'] > 0
        assert analysis['sentence_count'] > 0
        assert len(analysis['problems']) > 0  # 应该发现冗余问题

    def test_edit_article(self):
        """测试编辑文章"""
        # Arrange
        from src.production.domain.creative_writing.editor import ArticleEditor

        editor = ArticleEditor()
        text = "这是一个美丽的公园。公园里有很多很多的树。"

        # Act
        result = editor.edit(text)

        # Assert
        assert result.original_text == text
        assert result.edited_text is not None
        assert len(result.suggestions) > 0

    def test_edit_stats(self):
        """测试编辑统计"""
        # Arrange
        from src.production.domain.creative_writing.editor import ArticleEditor

        editor = ArticleEditor()

        # Act
        editor.edit("测试文本1")
        editor.edit("测试文本2")

        stats = editor.get_edit_stats()

        # Assert
        assert stats['total_edits'] == 2


class TestSelfCoder:
    """自我编码测试"""

    def test_analyze_code(self):
        """测试分析代码"""
        # Arrange
        from src.production.domain.creative_writing.self_coder import SelfCoder

        coder = SelfCoder()
        code = '''
def hello():
    print("hello")

def world():
    print("world")

class MyClass:
    def method(self):
        pass
'''

        # Act
        analysis = coder.analyze_code(code, "test.py")

        # Assert
        assert analysis.file_path == "test.py"
        assert 'hello' in analysis.functions
        assert 'world' in analysis.functions
        assert 'MyClass' in analysis.classes

    def test_improve_code(self):
        """测试改进代码"""
        # Arrange
        from src.production.domain.creative_writing.self_coder import SelfCoder

        coder = SelfCoder()
        code = '''
def long_function():
    x = 1
    y = 2
    z = 3
    a = 1
    b = 2
    c = 3
    d = 1
    e = 2
    f = 3
    g = 1
    h = 2
    i = 3
    j = 1
    k = 2
    l = 3
    m = 1
    n = 2
    o = 3
    p = 1
    q = 2
    r = 3
    return x + y + z
'''

        # Act
        improvements = coder.improve_code(code)

        # Assert
        assert len(improvements) > 0

    def test_rewrite_function(self):
        """测试重写函数"""
        # Arrange
        from src.production.domain.creative_writing.self_coder import SelfCoder, CodeImprovement

        coder = SelfCoder()
        code = "x = 42"

        improvement = CodeImprovement(
            original="42",
            improved="ANSWER_TO_EVERYTHING",
            reason="避免魔法数字",
            improvement_type="readability",
            confidence=0.9,
        )

        # Act
        rewritten = coder.rewrite_function(code, [improvement])

        # Assert
        assert "ANSWER_TO_EVERYTHING" in rewritten
