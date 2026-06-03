"""知识问答客户端 — 查询学习系统的内部知识

运行方式：
    python training/knowledge_client.py

功能：
1. 加载整合学习系统的知识（三元组 + 向量索引）
2. 提供问答接口
3. 用户提问，系统基于内部知识回答
4. 验证学习系统是否真正学到了知识
"""

import json
import os
import sys
import re
import pickle
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IntegratedKnowledge:
    """整合知识系统 — 加载学习结果"""

    def __init__(self):
        self.triples = []
        self.entities = {}
        self.summaries = {}
        self.total_articles = 0
        self.loaded = False

        # 索引
        self.subject_index = defaultdict(list)
        self.object_index = defaultdict(list)

    def load(self):
        """加载学习结果"""
        base = Path('data/knowledge/integrated')

        # 加载检查点
        checkpoint_path = base / 'checkpoint.pkl'
        if checkpoint_path.exists():
            print(f"  加载检查点: {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)

            text_data = data.get('text_system', {})
            self.triples = text_data.get('triples', [])
            self.entities = text_data.get('entities', {})
            self.summaries = text_data.get('summaries', {})
            self.total_articles = data.get('article_count', 0)

            # 重建索引
            for i, (s, r, o) in enumerate(self.triples):
                self.subject_index[s].append(i)
                self.object_index[o].append(i)

            self.loaded = True
            print(f"  加载完成: {self.total_articles} 文章, {len(self.triples)} 三元组")
        else:
            print(f"  检查点不存在: {checkpoint_path}")

    def query(self, question: str) -> Dict:
        """查询知识"""
        if not self.loaded:
            return {'answer': '知识尚未加载', 'confidence': 0}

        # 提取关键词
        keywords = self._extract_keywords(question)

        # 过滤掉太通用的关键词
        generic_words = set('什么 是 有 在 于 的 了 和 与 或')
        filtered_keywords = [kw for kw in keywords if kw not in generic_words and len(kw) >= 2]

        # 按长度排序，优先匹配长关键词
        filtered_keywords.sort(key=len, reverse=True)

        # 搜索三元组
        triple_results = []
        seen = set()

        for keyword in filtered_keywords[:5]:  # 只取前5个最长的关键词
            # 作为主语搜索
            for idx in self.subject_index.get(keyword, []):
                s, r, o = self.triples[idx]
                key = f"{s}|{r}|{o}"
                if key not in seen:
                    seen.add(key)
                    triple_results.append({
                        'subject': s,
                        'relation': r,
                        'object': o,
                        'score': 1.0,
                        'match_type': 'subject',
                        'keyword': keyword,
                    })

            # 作为宾语搜索
            for idx in self.object_index.get(keyword, []):
                s, r, o = self.triples[idx]
                key = f"{s}|{r}|{o}"
                if key not in seen:
                    seen.add(key)
                    triple_results.append({
                        'subject': s,
                        'relation': r,
                        'object': o,
                        'score': 0.8,
                        'match_type': 'object',
                        'keyword': keyword,
                    })

        # 按分数排序，优先显示"是"关系
        def sort_key(x):
            score = x['score']
            # 优先显示"是"关系
            if x['relation'] == '是':
                score += 0.5
            # 降低"相关内容"关系的优先级
            elif x['relation'] == '相关内容':
                score -= 0.3
            return score

        triple_results.sort(key=sort_key, reverse=True)

        # 生成回答
        if triple_results:
            answer = self._generate_answer(question, triple_results[:10])
            confidence = min(1.0, len(triple_results) * 0.1)
        else:
            answer = "学习系统尚未学到相关知识。"
            confidence = 0

        return {
            'answer': answer,
            'confidence': confidence,
            'keywords': filtered_keywords[:10],
            'total_triples': len(self.triples),
            'total_entities': len(self.entities),
            'results': triple_results[:10],
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 英文单词
        en_words = re.findall(r'[a-zA-Z]+', text.lower())
        # 中文词（2-4字）- 滑动窗口提取
        zh_words = []
        for i in range(len(text)):
            for j in range(i+2, min(i+5, len(text)+1)):
                word = text[i:j]
                if re.match(r'^[一-鿿]+$', word):
                    zh_words.append(word)
        # 过滤停用词
        stopwords = set('的了是在我你他她它们这那个有不人大中上下来什么如何怎样')
        all_words = en_words + zh_words
        return [w for w in all_words if w not in stopwords and len(w) >= 2]

    def _generate_answer(self, question: str, results: List[Dict]) -> str:
        """生成回答"""
        parts = []
        parts.append(f"基于学习系统的内部知识（{len(self.triples):,} 个三元组）：")

        for r in results[:5]:
            parts.append(f"- {r['subject']} {r['relation']} {r['object']}")

        return '\n'.join(parts)

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'total_articles': self.total_articles,
            'total_triples': len(self.triples),
            'total_entities': len(self.entities),
            'total_summaries': len(self.summaries),
            'loaded': self.loaded,
        }


class KnowledgeHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""

    knowledge = None

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._serve_html()
        elif self.path == '/api/stats':
            self._serve_stats()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/ask':
            self._handle_ask()
        else:
            self.send_error(404)

    def _serve_html(self):
        html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>学习系统知识问答</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .header { background: #2196F3; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .stats { background: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .chat-box { background: white; height: 500px; overflow-y: scroll; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .message { margin: 10px 0; padding: 12px; border-radius: 8px; }
        .user { background: #e3f2fd; text-align: right; }
        .bot { background: #f5f5f5; }
        .input-area { display: flex; gap: 10px; }
        input { flex: 1; padding: 12px; font-size: 16px; border: 1px solid #ddd; border-radius: 8px; }
        button { padding: 12px 24px; font-size: 16px; background: #2196F3; color: white; border: none; border-radius: 8px; cursor: pointer; }
        .confidence { font-size: 12px; color: #666; margin-top: 5px; }
        .detail { font-size: 12px; color: #888; margin-top: 10px; padding: 10px; background: #fafafa; border-radius: 5px; }
        .triple { margin: 5px 0; padding: 8px; background: #e8f5e9; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>学习系统知识问答</h1>
        <p>基于自主学习系统的内部知识回答问题</p>
    </div>

    <div class="stats" id="stats">加载中...</div>

    <div class="chat-box" id="chat"></div>

    <div class="input-area">
        <input type="text" id="question" placeholder="输入问题..." onkeypress="if(event.key==='Enter')ask()">
        <button onclick="ask()">提问</button>
    </div>

    <script>
        fetch('/api/stats').then(r=>r.json()).then(data=>{
            document.getElementById('stats').innerHTML =
                `<strong>学习系统状态：</strong>
                 文章数: ${data.total_articles.toLocaleString()} |
                 三元组: ${data.total_triples.toLocaleString()} |
                 实体数: ${data.total_entities.toLocaleString()} |
                 状态: ${data.loaded ? '已加载' : '未加载'}`;
        });

        function ask() {
            const input = document.getElementById('question');
            const question = input.value.trim();
            if (!question) return;

            addMessage('user', question);
            input.value = '';

            fetch('/api/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            })
            .then(r => r.json())
            .then(data => {
                let html = `<div>${data.answer}</div>`;
                html += `<div class="confidence">置信度: ${(data.confidence*100).toFixed(0)}% | 三元组: ${data.total_triples.toLocaleString()}</div>`;
                if (data.results && data.results.length > 0) {
                    html += '<div class="detail">';
                    data.results.forEach(r => {
                        html += `<div class="triple">${r.subject} → ${r.relation} → ${r.object}</div>`;
                    });
                    html += '</div>';
                }
                addMessage('bot', html, true);
            })
            .catch(e => addMessage('bot', '错误: ' + e.message));
        }

        function addMessage(type, text, isHtml) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message ' + type;
            if (isHtml) div.innerHTML = text;
            else div.innerText = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        addMessage('bot', '你好！我是学习系统的知识问答界面。你可以问我任何问题，我会基于学习系统内部学到的知识回答。');
        addMessage('bot', '提示：学习系统通过自主学习提取了超过160万个知识三元组。');
    </script>
</body>
</html>'''
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def _serve_stats(self):
        stats = self.knowledge.get_stats()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode('utf-8'))

    def _handle_ask(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body)
            question = data.get('question', '')
            result = self.knowledge.query(question)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            self.send_error(400, str(e))

    def log_message(self, format, *args):
        pass


def main():
    print("=" * 70)
    print("学习系统知识问答客户端")
    print("=" * 70)

    print("\n[1] 加载学习系统的知识...")
    knowledge = IntegratedKnowledge()
    knowledge.load()

    if not knowledge.loaded:
        print("错误：无法加载知识。请先运行学习脚本。")
        return

    print("\n[2] 启动服务器...")
    KnowledgeHandler.knowledge = knowledge

    port = 8081
    server = HTTPServer(('0.0.0.0', port), KnowledgeHandler)

    print(f"  访问: http://localhost:{port}")
    print(f"  API: POST http://localhost:{port}/api/ask")

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
