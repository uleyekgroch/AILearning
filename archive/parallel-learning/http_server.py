"""
HTTP服务器 - 测试界面

加载学习结果，提供查询接口供用户测试
"""

import json
import os
import sys
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.production.domain.true_learning.learner import TrueLearner
from src.production.domain.true_learning.creation import CreationRequest


class KnowledgeBase:
    """知识库 - 加载学习结果"""

    def __init__(self):
        """初始化知识库"""
        self.learner = TrueLearner(embedding_dim=32)
        self.knowledge_index: Dict[str, Dict[str, Any]] = {}
        self.stats = {}

    def load_learning_result(self, filepath: str):
        """加载学习结果"""
        if not os.path.exists(filepath):
            print(f"警告: 学习结果文件不存在 {filepath}")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.stats = data.get('stats', {})
        self.knowledge_index = data.get('knowledge_index', {})

        # 将知识加载到学习器
        for topic, info in self.knowledge_index.items():
            facts = info.get('facts', [])
            self.learner.learn_topic(topic, facts)

        print(f"加载了 {len(self.knowledge_index)} 个主题")

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
                related_topics.append({
                    'topic': topic,
                    'facts_count': len(data.get('facts', [])),
                    'score': data.get('score', 0),
                })

        # 排序
        related_topics.sort(key=lambda x: -x['score'])

        return {
            'question': question,
            'understanding': {
                'confidence': understanding.confidence,
                'reasoning_chain': understanding.reasoning_chain,
            },
            'reasoning': {
                'conclusion': reasoning.conclusion,
                'confidence': reasoning.confidence,
                'type': reasoning.reasoning_type,
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
            'creation_process': result.creation_process,
        }

    def get_topics(self, limit: int = 100) -> Dict[str, Any]:
        """获取主题列表"""
        topics = list(self.knowledge_index.keys())[:limit]
        return {
            'topics': topics,
            'total': len(self.knowledge_index),
            'showing': len(topics),
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self.stats,
            'knowledge_count': len(self.knowledge_index),
        }


class HTTPHandler(BaseHTTPRequestHandler):
    """HTTP请求处理器"""

    knowledge_base = None

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
                body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f0f0f0; }
                h1 { color: #333; text-align: center; }
                .section { margin: 20px 0; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                h2 { color: #007bff; margin-top: 0; }
                input[type="text"] { width: 60%; padding: 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
                select { padding: 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
                button { padding: 12px 24px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-left: 10px; }
                button:hover { background: #0056b3; }
                #result { margin-top: 20px; padding: 20px; background: #f8f9fa; border: 1px solid #ddd; border-radius: 5px; min-height: 150px; white-space: pre-wrap; font-family: monospace; }
                .example { color: #666; font-size: 14px; margin-top: 10px; }
            </style>
        </head>
        <body>
            <h1>🧠 AI学习系统 - 测试界面</h1>
            <p style="text-align: center; color: #666;">系统已学习Wiki知识库，您可以测试它的理解能力</p>

            <div class="section">
                <h2>📝 测试理解能力</h2>
                <input type="text" id="queryInput" placeholder="输入问题，测试系统的理解能力">
                <button onclick="queryKnowledge()">查询</button>
                <div class="example">
                    试试这些问题：
                    <a href="#" onclick="setQuery('数学是什么')">数学是什么</a> |
                    <a href="#" onclick="setQuery('物理定律')">物理定律</a> |
                    <a href="#" onclick="setQuery('化学反应')">化学反应</a> |
                    <a href="#" onclick="setQuery('历史事件')">历史事件</a> |
                    <a href="#" onclick="setQuery('人工智能')">人工智能</a>
                </div>
            </div>

            <div class="section">
                <h2>✍️ 测试创造能力</h2>
                <input type="text" id="topicInput" placeholder="输入主题，测试系统的创造能力">
                <select id="styleSelect">
                    <option value="expository">说明文</option>
                    <option value="narrative">叙述文</option>
                    <option value="descriptive">描写文</option>
                    <option value="argumentative">议论文</option>
                </select>
                <button onclick="createContent()">创造</button>
                <div class="example">
                    试试这些主题：
                    <a href="#" onclick="setTopic('数学')">数学</a> |
                    <a href="#" onclick="setTopic('物理')">物理</a> |
                    <a href="#" onclick="setTopic('人工智能')">人工智能</a>
                </div>
            </div>

            <div class="section">
                <h2>📊 系统信息</h2>
                <button onclick="getStats()">查看统计</button>
                <button onclick="getTopics()">查看主题</button>
            </div>

            <div id="result"></div>

            <script>
                function setQuery(q) {
                    document.getElementById('queryInput').value = q;
                    queryKnowledge();
                }

                function setTopic(t) {
                    document.getElementById('topicInput').value = t;
                }

                async function queryKnowledge() {
                    const query = document.getElementById('queryInput').value;
                    if (!query) return;

                    document.getElementById('result').innerHTML = '查询中...';
                    const response = await fetch('/query?q=' + encodeURIComponent(query));
                    const data = await response.json();

                    let html = '<h3>📊 查询结果</h3>';
                    html += '<p><strong>问题：</strong>' + data.question + '</p>';
                    html += '<p><strong>理解置信度：</strong>' + (data.understanding.confidence * 100).toFixed(1) + '%</p>';
                    html += '<p><strong>推理结论：</strong>' + data.reasoning.conclusion + '</p>';
                    html += '<p><strong>推理置信度：</strong>' + (data.reasoning.confidence * 100).toFixed(1) + '%</p>';

                    if (data.related_topics.length > 0) {
                        html += '<p><strong>相关主题：</strong></p><ul>';
                        data.related_topics.forEach(t => {
                            html += '<li>' + t.topic + ' (' + t.facts_count + '条知识)</li>';
                        });
                        html += '</ul>';
                    }

                    html += '<p><strong>知识库总量：</strong>' + data.total_knowledge + ' 个主题</p>';

                    document.getElementById('result').innerHTML = html;
                }

                async function createContent() {
                    const topic = document.getElementById('topicInput').value;
                    const style = document.getElementById('styleSelect').value;
                    if (!topic) return;

                    document.getElementById('result').innerHTML = '创造中...';
                    const response = await fetch('/create?topic=' + encodeURIComponent(topic) + '&style=' + style);
                    const data = await response.json();

                    let html = '<h3>✍️ 创造结果</h3>';
                    html += '<p><strong>主题：</strong>' + data.topic + '</p>';
                    html += '<p><strong>风格：</strong>' + data.style + '</p>';
                    html += '<p><strong>质量分数：</strong>' + (data.quality_score * 100).toFixed(1) + '%</p>';
                    html += '<h4>内容：</h4>';
                    html += '<div style="background: white; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">' + data.content + '</div>';

                    document.getElementById('result').innerHTML = html;
                }

                async function getStats() {
                    document.getElementById('result').innerHTML = '加载中...';
                    const response = await fetch('/stats');
                    const data = await response.json();

                    let html = '<h3>📊 系统统计</h3>';
                    html += '<ul>';
                    html += '<li>学习文章数：' + data.articles_learned + '</li>';
                    html += '<li>学习句子数：' + data.sentences_learned + '</li>';
                    html += '<li>形成概念数：' + data.concepts_formed + '</li>';
                    html += '<li>知识库主题数：' + data.knowledge_count + '</li>';
                    html += '<li>学习时间：' + data.learning_time?.toFixed(2) + '秒</li>';
                    html += '</ul>';

                    document.getElementById('result').innerHTML = html;
                }

                async function getTopics() {
                    document.getElementById('result').innerHTML = '加载中...';
                    const response = await fetch('/topics?limit=50');
                    const data = await response.json();

                    let html = '<h3>📚 知识主题（前50个）</h3>';
                    html += '<p>总共 ' + data.total + ' 个主题</p>';
                    html += '<div style="columns: 3;">';
                    data.topics.forEach(t => {
                        html += '<div style="margin: 5px 0;">• ' + t + '</div>';
                    });
                    html += '</div>';

                    document.getElementById('result').innerHTML = html;
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

        # URL解码
        query = urllib.parse.unquote(query)

        if not query:
            result = {'error': '请提供查询参数 q'}
        else:
            result = self.knowledge_base.query(query)

        self.send_json(result)

    def send_create(self, params):
        """发送创造结果"""
        topic = params.get('topic', [''])[0]
        style = params.get('style', ['expository'])[0]

        # URL解码
        topic = urllib.parse.unquote(topic)

        if not topic:
            result = {'error': '请提供主题参数 topic'}
        else:
            result = self.knowledge_base.create_content(topic, style)

        self.send_json(result)

    def send_stats(self):
        """发送统计信息"""
        result = self.knowledge_base.get_stats()
        self.send_json(result)

    def send_topics(self):
        """发送主题列表"""
        limit = int(self.path.split('limit=')[1].split('&')[0]) if 'limit=' in self.path else 100
        result = self.knowledge_base.get_topics(limit)
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
    print("AI学习系统 - HTTP测试服务器")
    print("=" * 60)

    # 创建知识库
    kb = KnowledgeBase()

    # 加载学习结果
    learning_result_file = "data/wiki_learning_result.json"
    if os.path.exists(learning_result_file):
        print(f"加载学习结果: {learning_result_file}")
        kb.load_learning_result(learning_result_file)
    else:
        print(f"警告: 学习结果文件不存在 {learning_result_file}")
        print("请先运行 learn_wiki_full.py 进行学习")

    # 设置HTTP处理器
    HTTPHandler.knowledge_base = kb

    # 启动HTTP服务器
    port = 8080
    server = HTTPServer(('0.0.0.0', port), HTTPHandler)
    print(f"\nHTTP服务器启动在: http://localhost:{port}")
    print("按 Ctrl+C 停止服务器")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        server.shutdown()


if __name__ == '__main__':
    main()
