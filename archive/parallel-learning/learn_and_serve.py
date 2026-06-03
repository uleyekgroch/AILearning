"""
Wiki全量学习 + HTTP服务器

学习所有wiki数据，然后启动HTTP服务器供测试
"""

import json
import os
import sys
import time
from typing import List, Dict, Any
import numpy as np
import torch
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import threading

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.production.domain.true_learning.learner import TrueLearner
from src.production.domain.true_learning.understanding import UnderstandingEngine
from src.production.domain.true_learning.reasoning import ReasoningEngine
from src.production.domain.true_learning.creation import CreationEngine, CreationRequest


class WikiFullLearner:
    """Wiki全量学习器"""

    def __init__(self, embedding_dim: int = 64):
        """
        初始化学习器

        Args:
            embedding_dim: 嵌入维度
        """
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 真正的学习器
        self.learner = TrueLearner(embedding_dim)

        # 知识索引
        self.knowledge_index: Dict[str, Dict[str, Any]] = {}

        # 学习统计
        self.stats = {
            'total_articles': 0,
            'total_sentences': 0,
            'total_concepts': 0,
            'files_processed': 0,
            'learning_time': 0.0,
        }

    def load_wiki_file(self, filepath: str, limit: int = None) -> List[Dict[str, Any]]:
        """加载wiki文件"""
        articles = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break

                    try:
                        article = json.loads(line.strip())
                        articles.append(article)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"  错误: {e}")

        return articles

    def extract_sentences(self, text: str, max_sentences: int = 10) -> List[str]:
        """提取句子"""
        sentences = text.split('。')
        cleaned = []
        for s in sentences:
            s = s.strip()
            if len(s) > 5 and len(s) < 200:
                cleaned.append(s)
        return cleaned[:max_sentences]

    def learn_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """学习一篇文章"""
        title = article.get('title', '')
        text = article.get('text', '')

        # 提取句子
        sentences = self.extract_sentences(text)

        if not sentences:
            return {'title': title, 'sentences_learned': 0}

        # 学习主题
        result = self.learner.learn_topic(
            topic=title,
            facts=sentences,
            relations={}
        )

        # 索引知识
        self.knowledge_index[title] = {
            'facts': sentences,
            'concept': title,
            'score': result.overall_score,
        }

        # 更新统计
        self.stats['total_articles'] += 1
        self.stats['total_sentences'] += len(sentences)
        self.stats['total_concepts'] += 1

        return {
            'title': title,
            'sentences_learned': len(sentences),
            'score': result.overall_score,
        }

    def learn_all_wiki(self, wiki_dir: str, limit_per_file: int = 50) -> Dict[str, Any]:
        """学习所有wiki文件"""
        start_time = time.time()

        # 获取所有wiki文件
        wiki_files = sorted(Path(wiki_dir).rglob('wiki_*'))
        total_files = len(wiki_files)

        print(f"找到 {total_files} 个wiki文件")

        all_results = []
        for i, wiki_file in enumerate(wiki_files):
            print(f"\n处理文件 {i+1}/{total_files}: {wiki_file.name}")

            # 加载文章
            articles = self.load_wiki_file(str(wiki_file), limit=limit_per_file)
            print(f"  加载了 {len(articles)} 篇文章")

            # 学习每篇文章
            for j, article in enumerate(articles):
                result = self.learn_article(article)
                all_results.append(result)

                if (j + 1) == limit_per_file:
                    break

            self.stats['files_processed'] += 1

            # 每10个文件打印进度
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{total_files} 文件, {self.stats['total_articles']} 文章")

        # 计算统计
        self.stats['learning_time'] = time.time() - start_time

        return {
            'files_processed': self.stats['files_processed'],
            'articles_learned': self.stats['total_articles'],
            'sentences_learned': self.stats['total_sentences'],
            'concepts_formed': self.stats['total_concepts'],
            'learning_time': self.stats['learning_time'],
        }

    def query(self, question: str) -> Dict[str, Any]:
        """查询"""
        # 理解问题
        understanding = self.learner.understand(question)

        # 推理
        reasoning = self.learner.reason(question, 'deductive')

        # 查找相关知识
        related_topics = []
        for topic, data in self.knowledge_index.items():
            if any(char in topic for char in question):
                related_topics.append(topic)

        return {
            'question': question,
            'understanding': {
                'confidence': understanding.confidence,
                'reasoning_chain': understanding.reasoning_chain,
            },
            'reasoning': {
                'conclusion': reasoning.conclusion,
                'confidence': reasoning.confidence,
            },
            'related_topics': related_topics[:10],
            'total_knowledge': len(self.knowledge_index),
        }

    def create_content(self, topic: str, style: str = 'expository') -> Dict[str, Any]:
        """创造内容"""
        # 查找相关知识
        facts = []
        if topic in self.knowledge_index:
            facts = self.knowledge_index[topic]['facts']

        # 创造内容
        request = CreationRequest(
            topic=topic,
            style=style,
            length='medium',
            requirements=[],
            context={'facts': facts}
        )

        result = self.learner.creation_engine.create(request)

        return {
            'topic': topic,
            'style': style,
            'content': result.content,
            'quality_score': result.quality_score,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self.stats,
            'knowledge_count': len(self.knowledge_index),
            'device': str(self.device),
        }


class LearningHTTPHandler(BaseHTTPRequestHandler):
    """HTTP请求处理器"""

    learner = None

    def do_GET(self):
        """处理GET请求"""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        params = urllib.parse.parse_qs(parsed_path.query)

        if path == '/':
            self.send_index()
        elif path == '/query':
            self.send_query(params)
        elif path == '/create':
            self.send_create(params)
        elif path == '/stats':
            self.send_stats()
        elif path == '/topics':
            self.send_topics()
        else:
            self.send_error(404, "Not Found")

    def send_index(self):
        """发送首页"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>AI学习系统 - 测试界面</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .section { margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }
                input[type="text"] { width: 70%; padding: 10px; }
                button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
                button:hover { background: #0056b3; }
                #result { margin-top: 20px; padding: 15px; background: white; border: 1px solid #ddd; border-radius: 5px; min-height: 100px; }
            </style>
        </head>
        <body>
            <h1>🧠 AI学习系统 - 测试界面</h1>

            <div class="section">
                <h2>📝 查询知识</h2>
                <input type="text" id="queryInput" placeholder="输入问题，例如：数学是什么">
                <button onclick="queryKnowledge()">查询</button>
            </div>

            <div class="section">
                <h2>✍️ 创造内容</h2>
                <input type="text" id="topicInput" placeholder="输入主题，例如：人工智能">
                <select id="styleSelect">
                    <option value="expository">说明文</option>
                    <option value="narrative">叙述文</option>
                    <option value="descriptive">描写文</option>
                    <option value="argumentative">议论文</option>
                </select>
                <button onclick="createContent()">创造</button>
            </div>

            <div class="section">
                <h2>📊 系统统计</h2>
                <button onclick="getStats()">查看统计</button>
            </div>

            <div id="result"></div>

            <script>
                async function queryKnowledge() {
                    const query = document.getElementById('queryInput').value;
                    const response = await fetch('/query?q=' + encodeURIComponent(query));
                    const data = await response.json();
                    document.getElementById('result').innerHTML = '<h3>查询结果</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                }

                async function createContent() {
                    const topic = document.getElementById('topicInput').value;
                    const style = document.getElementById('styleSelect').value;
                    const response = await fetch('/create?topic=' + encodeURIComponent(topic) + '&style=' + style);
                    const data = await response.json();
                    document.getElementById('result').innerHTML = '<h3>创造结果</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                }

                async function getStats() {
                    const response = await fetch('/stats');
                    const data = await response.json();
                    document.getElementById('result').innerHTML = '<h3>系统统计</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                }
            </script>
        </body>
        </html>
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def send_query(self, params):
        """发送查询结果"""
        query = params.get('q', [''])[0]

        if not query:
            result = {'error': '请提供查询参数 q'}
        else:
            result = self.learner.query(query)

        self.send_json(result)

    def send_create(self, params):
        """发送创造结果"""
        topic = params.get('topic', [''])[0]
        style = params.get('style', ['expository'])[0]

        if not topic:
            result = {'error': '请提供主题参数 topic'}
        else:
            result = self.learner.create_content(topic, style)

        self.send_json(result)

    def send_stats(self):
        """发送统计信息"""
        result = self.learner.get_stats()
        self.send_json(result)

    def send_topics(self):
        """发送主题列表"""
        topics = list(self.learner.knowledge_index.keys())[:100]
        result = {'topics': topics, 'total': len(self.learner.knowledge_index)}
        self.send_json(result)

    def send_json(self, data):
        """发送JSON响应"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))

    def log_message(self, format, *args):
        """禁用日志"""
        pass


def main():
    """主函数"""
    print("=" * 60)
    print("Wiki全量学习 + HTTP服务器")
    print("=" * 60)

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU")

    # 创建学习器
    learner = WikiFullLearner(embedding_dim=32)

    # Wiki目录
    wiki_dir = "data/wiki_zh"

    # 检查目录
    if not os.path.exists(wiki_dir):
        print(f"错误: 目录不存在 {wiki_dir}")
        return

    # 学习所有wiki
    print("\n开始学习所有wiki文章...")
    stats = learner.learn_all_wiki(wiki_dir, limit_per_file=20)

    # 打印统计
    print("\n" + "=" * 60)
    print("学习完成!")
    print("=" * 60)
    print(f"处理文件数: {stats['files_processed']}")
    print(f"学习文章数: {stats['articles_learned']}")
    print(f"学习句子数: {stats['sentences_learned']}")
    print(f"形成概念数: {stats['concepts_formed']}")
    print(f"学习时间: {stats['learning_time']:.2f}秒")

    # 设置HTTP处理器
    LearningHTTPHandler.learner = learner

    # 启动HTTP服务器
    port = 8080
    server = HTTPServer(('0.0.0.0', port), LearningHTTPHandler)
    print(f"\nHTTP服务器启动在: http://localhost:{port}")
    print("按 Ctrl+C 停止服务器")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        server.shutdown()


if __name__ == '__main__':
    main()
