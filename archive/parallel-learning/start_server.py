#!/usr/bin/env python3
"""启动学习AI HTTP服务器"""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.learner import Learner, LearnerConfig
from src.knowledge.entity import Entity
from src.knowledge.relation import Relation
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


def quick_learn(learner, limit=200):
    """快速学习"""
    count = 0
    success = 0
    filepath = 'data/extracted/baike/baike_qa_train.json'

    print(f'[学习] 开始学习 {limit} 条...')

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= limit:
                break
            count += 1

            try:
                item = json.loads(line.strip())
                title = item.get('title', '').strip()
                answer = item.get('answer', '').strip().replace('\r\n', ' ')[:100]

                if len(title) < 4 or len(answer) < 4:
                    continue

                text = f"{title}。{answer}"
                text_repr = learner._encode_text(text, train=True)
                entities = learner._extract_entities_from_repr(text, text_repr)
                triples = learner._extract_relations_from_repr(text, entities, text_repr)

                for item in triples:
                    if len(item) >= 3:
                        subj, rel, obj = item[0], item[1], item[2]
                        learner.knowledge.add_entity(Entity(
                            id=subj, type='concept',
                            embedding=learner._encode_text(subj)
                        ))
                        learner.knowledge.add_entity(Entity(
                            id=obj, type='concept',
                            embedding=learner._encode_text(obj)
                        ))
                        learner.knowledge.add_relation(Relation(
                            source_id=subj, target_id=obj,
                            type=rel, confidence=0.8
                        ))

                if triples:
                    success += 1

                if count % 50 == 0:
                    print(f'[学习] 进度: {count}/{limit}, 成功: {success}')

            except Exception:
                continue

    print(f'[学习] 完成: {success}条, 实体: {len(learner.knowledge.entities)}')
    return success


class LearnerHandler(BaseHTTPRequestHandler):
    """HTTP请求处理器"""

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/ask':
            params = parse_qs(parsed.query)
            question = params.get('q', [''])[0]

            if not question:
                self.send_json({'error': '缺少参数 q'}, 400)
                return

            answer = self.server.learner.think(question)
            self.send_json({'question': question, 'answer': answer})

        elif parsed.path == '/status':
            self.send_json({
                'status': 'running',
                'entities': len(self.server.learner.knowledge.entities),
            })

        elif parsed.path == '/':
            self.send_html('''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>学习AI</title>
    <style>
        body { font-family: monospace; max-width: 800px; margin: 40px auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }
        h1 { color: #00d4ff; }
        input { width: 70%; padding: 10px; font-size: 16px; background: #16213e; color: #e0e0e0; border: 1px solid #0f3460; }
        button { padding: 10px 20px; font-size: 16px; background: #0f3460; color: #00d4ff; border: 1px solid #00d4ff; cursor: pointer; }
        button:hover { background: #00d4ff; color: #1a1a2e; }
        #answer { margin-top: 20px; padding: 15px; background: #16213e; border: 1px solid #0f3460; white-space: pre-wrap; min-height: 100px; }
    </style>
</head>
<body>
    <h1>学习AI</h1>
    <p>从学习的本源出发的人工智能系统</p>
    <div>
        <input id="q" placeholder="输入问题..." onkeydown="if(event.key==='Enter')ask()">
        <button onclick="ask()">提问</button>
    </div>
    <div id="answer"></div>
    <script>
        async function ask() {
            const q = document.getElementById('q').value;
            if (!q) return;
            document.getElementById('answer').textContent = '思考中...';
            const r = await fetch('/ask?q=' + encodeURIComponent(q));
            const d = await r.json();
            document.getElementById('answer').textContent = d.answer || d.error;
        }
    </script>
</body>
</html>''')

        else:
            self.send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == '/learn':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')

            if not body:
                self.send_json({'error': '缺少学习内容'}, 400)
                return

            result = self.server.learner.learn_from_text(body, source='http')
            self.send_json({
                'learned': body,
                'entities': result['entities'],
                'triples': len(result['triples']),
            })

        else:
            self.send_json({'error': 'Not found'}, 404)

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def send_html(self, html, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def log_message(self, format, *args):
        print(f'[HTTP] {args[0]}')


def main():
    print('=' * 50)
    print('学习AI HTTP服务器')
    print('=' * 50)

    # 初始化学习体
    config = LearnerConfig(obs_dim=128)
    learner = Learner(config)

    # 预初始化
    print('[初始化] 预初始化编码器...')
    _ = learner._encode_text('初始化', train=True)
    print('[初始化] 完成')

    # 快速学习
    quick_learn(learner, limit=200)

    # 启动HTTP服务器
    port = 8080
    server = HTTPServer(('0.0.0.0', port), LearnerHandler)
    server.learner = learner

    print(f'\n[服务器] 启动在 http://localhost:{port}')
    print(f'[服务器] 浏览器打开: http://localhost:{port}')
    print(f'[服务器] API提问: http://localhost:{port}/ask?q=什么是人工智能')
    print(f'[服务器] 按 Ctrl+C 停止')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[服务器] 已停止')
        server.shutdown()


if __name__ == '__main__':
    main()
