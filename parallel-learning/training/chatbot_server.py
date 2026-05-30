"""知识问答服务器 — 验证学习效果

运行方式：
    python training/chatbot_server.py

功能：
1. 加载学习到的知识（词汇、语义关系、语料知识）
2. 提供 HTTP 问答接口
3. 用户可以提问，系统基于学习到的知识回答
4. 支持中英文问答

访问方式：
    浏览器: http://localhost:8080
    API: POST http://localhost:8080/ask {"question": "什么是重力？"}
"""

import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class KnowledgeBase:
    """知识库 — 整合所有学习到的知识"""

    def __init__(self):
        self.word_knowledge: Dict = {}
        self.semantic_relations: Dict = {}
        self.corpus_knowledge: Dict = {}
        self.loaded = False

    def load(self):
        """加载所有知识"""
        print("  加载词汇知识...")
        self._load_json('data/knowledge/words.json', 'word_knowledge')
        self._load_json('data/knowledge/ai_progress.json', 'ai_knowledge')
        self._load_json('data/knowledge/semantic_relations.json', 'semantic_relations')
        self._load_json('data/knowledge/corpus_knowledge.json', 'corpus_knowledge')
        self.loaded = True
        print(f"  词汇知识: {len(self.word_knowledge)} 词")
        print(f"  AI 知识: {len(getattr(self, 'ai_knowledge', {}))} 词")
        print(f"  语义关系: {len(self.semantic_relations)} 词")
        print(f"  语料知识: {len(self.corpus_knowledge.get('entities', {}))} 实体")

    def _load_json(self, path: str, attr: str):
        """加载 JSON 文件"""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    setattr(self, attr, json.load(f))
            except Exception as e:
                print(f"    加载 {path} 失败: {e}")

    def answer(self, question: str) -> Dict:
        """回答问题"""
        if not self.loaded:
            return {'answer': '知识库尚未加载', 'confidence': 0}

        # 分析问题
        keywords = self._extract_keywords(question)

        # 搜索知识
        results = []
        for keyword in keywords:
            # 搜索词汇知识
            word_result = self._search_word(keyword)
            if word_result:
                results.append(word_result)

            # 搜索语料知识
            corpus_result = self._search_corpus(keyword)
            if corpus_result:
                results.append(corpus_result)

        # 生成回答
        if results:
            answer = self._generate_answer(question, results)
            confidence = min(1.0, len(results) * 0.3)
        else:
            answer = "抱歉，我没有找到相关知识。"
            confidence = 0

        return {
            'answer': answer,
            'confidence': confidence,
            'keywords': keywords,
            'sources': len(results),
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        import re
        # 英文单词
        en_words = re.findall(r'[a-zA-Z]+', text.lower())
        # 中文词（2-4字）
        zh_words = re.findall(r'[一-鿿]{2,4}', text)
        return list(set(en_words + zh_words))

    def _search_word(self, keyword: str) -> Optional[Dict]:
        """搜索词汇知识"""
        # 精确匹配
        if keyword in self.word_knowledge:
            return {
                'type': 'word',
                'word': keyword,
                'data': self.word_knowledge[keyword],
            }

        # AI 知识匹配
        if hasattr(self, 'ai_knowledge') and keyword in self.ai_knowledge:
            return {
                'type': 'ai_word',
                'word': keyword,
                'data': self.ai_knowledge[keyword],
            }

        # 语义关系匹配
        if keyword in self.semantic_relations:
            return {
                'type': 'semantic',
                'word': keyword,
                'data': self.semantic_relations[keyword],
            }

        # 模糊匹配
        for word in self.word_knowledge:
            if keyword in word or word in keyword:
                return {
                    'type': 'word_fuzzy',
                    'word': word,
                    'data': self.word_knowledge[word],
                }

        return None

    def _search_corpus(self, keyword: str) -> Optional[Dict]:
        """搜索语料知识"""
        entities = self.corpus_knowledge.get('entities', {})

        if keyword in entities:
            return {
                'type': 'corpus',
                'entity': keyword,
                'data': entities[keyword],
            }

        # 模糊匹配
        for name, entity in entities.items():
            if keyword in name or name in keyword:
                return {
                    'type': 'corpus_fuzzy',
                    'entity': name,
                    'data': entity,
                }

        return None

    def _generate_answer(self, question: str, results: List[Dict]) -> str:
        """生成回答"""
        answer_parts = []

        for result in results:
            if result['type'] in ['word', 'ai_word']:
                word = result['word']
                data = result['data']

                # 定义
                if 'definition' in data:
                    answer_parts.append(f"**{word}**: {data['definition']}")

                # AI 知识
                if 'encyclo' in data:
                    answer_parts.append(f"百科: {data['encyclo']}")

                # 例句
                if 'examples' in data and data['examples']:
                    examples = data['examples'][:2]
                    answer_parts.append(f"例句: {'; '.join(examples)}")

                # 同义词
                if 'synonyms' in data and data['synonyms']:
                    synonyms = data['synonyms'][:3]
                    answer_parts.append(f"同义词: {', '.join(synonyms)}")

            elif result['type'] == 'semantic':
                word = result['word']
                data = result['data']

                if 'synonyms' in data and data['synonyms']:
                    answer_parts.append(f"{word} 的同义词: {', '.join(data['synonyms'][:5])}")

                if 'hypernyms' in data and data['hypernyms']:
                    answer_parts.append(f"{word} 的上位词: {', '.join(data['hypernyms'][:3])}")

            elif result['type'] in ['corpus', 'corpus_fuzzy']:
                entity = result['entity']
                data = result['data']

                if 'contexts' in data and data['contexts']:
                    context = data['contexts'][0]
                    answer_parts.append(f"关于 {entity}: {context}")

        if not answer_parts:
            return "抱歉，我没有找到相关知识。"

        return '\n'.join(answer_parts)


class ChatbotHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""

    knowledge_base = None

    def do_GET(self):
        """处理 GET 请求"""
        if self.path == '/' or self.path == '/index.html':
            self._serve_html()
        elif self.path == '/api/stats':
            self._serve_stats()
        else:
            self.send_error(404)

    def do_POST(self):
        """处理 POST 请求"""
        if self.path == '/api/ask':
            self._handle_ask()
        else:
            self.send_error(404)

    def _serve_html(self):
        """提供 HTML 页面"""
        html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>知识问答系统</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-box { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; margin: 10px 0; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background: #e3f2fd; text-align: right; }
        .bot { background: #f5f5f5; }
        .input-area { display: flex; gap: 10px; }
        input { flex: 1; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #2196F3; color: white; border: none; cursor: pointer; }
        .stats { background: #fff3e0; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>知识问答系统</h1>
    <p>基于自主学习的 AI 系统，使用 WordNet + 通义千问 + Wikipedia 语料学习</p>

    <div class="stats" id="stats">加载中...</div>

    <div class="chat-box" id="chat"></div>

    <div class="input-area">
        <input type="text" id="question" placeholder="输入问题..." onkeypress="if(event.key==='Enter')ask()">
        <button onclick="ask()">发送</button>
    </div>

    <script>
        // 加载统计信息
        fetch('/api/stats').then(r=>r.json()).then(data=>{
            document.getElementById('stats').innerHTML =
                `词汇: ${data.word_knowledge} | AI知识: ${data.ai_knowledge} | 语义关系: ${data.semantic_relations} | 语料实体: ${data.corpus_entities}`;
        });

        function ask() {
            const input = document.getElementById('question');
            const question = input.value.trim();
            if (!question) return;

            // 显示用户问题
            addMessage('user', question);
            input.value = '';

            // 发送请求
            fetch('/api/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            })
            .then(r => r.json())
            .then(data => {
                addMessage('bot', data.answer + '\\n\\n(置信度: ' + (data.confidence*100).toFixed(0) + '%, 来源: ' + data.sources + '个)');
            })
            .catch(e => {
                addMessage('bot', '错误: ' + e.message);
            });
        }

        function addMessage(type, text) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.innerText = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        // 初始消息
        addMessage('bot', '你好！我是知识问答系统。你可以问我任何问题，我会基于学到的知识回答。');
    </script>
</body>
</html>'''
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def _serve_stats(self):
        """提供统计信息"""
        kb = self.knowledge_base
        stats = {
            'word_knowledge': len(kb.word_knowledge),
            'ai_knowledge': len(getattr(kb, 'ai_knowledge', {})),
            'semantic_relations': len(kb.semantic_relations),
            'corpus_entities': len(kb.corpus_knowledge.get('entities', {})),
        }
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode('utf-8'))

    def _handle_ask(self):
        """处理问答请求"""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body)
            question = data.get('question', '')

            if not question:
                result = {'answer': '请输入问题', 'confidence': 0}
            else:
                result = self.knowledge_base.answer(question)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            self.send_error(400, str(e))

    def log_message(self, format, *args):
        """禁用日志"""
        pass


def main():
    print("=" * 70)
    print("知识问答服务器")
    print("=" * 70)

    # 加载知识库
    print("\n[1] 加载知识库...")
    kb = KnowledgeBase()
    kb.load()

    # 启动服务器
    print("\n[2] 启动服务器...")
    ChatbotHandler.knowledge_base = kb

    host = '0.0.0.0'
    port = 8080
    server = HTTPServer((host, port), ChatbotHandler)

    print(f"  服务器地址: http://localhost:{port}")
    print(f"  浏览器访问: http://localhost:{port}")
    print(f"  API 接口: POST http://localhost:{port}/api/ask")

    print("\n" + "=" * 70)
    print("服务器已启动! 按 Ctrl+C 停止")
    print("=" * 70)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        server.server_close()


if __name__ == '__main__':
    main()
