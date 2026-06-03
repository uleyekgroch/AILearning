"""内容加载器 — txt / md / pdf"""

import os
from typing import List, Optional


class ContentLoader:
    """从文件系统加载学习内容"""

    def __init__(self, content_dir: str = 'content'):
        self.content_dir = content_dir

    def load_text(self, filepath: str) -> str:
        """加载 .txt 或 .md 文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def load_pdf(self, filepath: str) -> str:
        """用 PyMuPDF 提取 PDF 文本"""
        try:
            import fitz
        except ImportError:
            raise ImportError("需要 PyMuPDF: pip install PyMuPDF")

        doc = fitz.open(filepath)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return '\n'.join(text_parts)

    def load_content(self, filepath: str) -> str:
        """自动检测格式并加载"""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.pdf':
            return self.load_pdf(filepath)
        return self.load_text(filepath)

    def list_content(self, level: Optional[str] = None) -> List[str]:
        """列出可用内容文件"""
        results = []
        search_dirs = [self.content_dir]

        if level:
            level_dir = os.path.join(self.content_dir, level)
            if os.path.isdir(level_dir):
                search_dirs = [level_dir]
            else:
                return []

        for base_dir in search_dirs:
            for root, _, files in os.walk(base_dir):
                for fname in files:
                    if fname.endswith(('.txt', '.md', '.pdf')):
                        results.append(os.path.join(root, fname))

        return sorted(results)

    def load_by_difficulty(self, level: str) -> List[str]:
        """加载指定难度的所有内容"""
        paths = self.list_content(level)
        contents = []
        for path in paths:
            try:
                contents.append(self.load_content(path))
            except Exception as e:
                print(f"[跳过] {path}: {e}")
        return contents
