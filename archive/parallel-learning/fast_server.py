#!/usr/bin/env python3
"""快速学习AI HTTP服务器

优化版本：
1. 预学习知识库
2. 快速查询
3. 增量学习
"""

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


# 全局learner
learner = None


def init_and_learn():
    """初始化并学习"""
    global learner

    config = LearnerConfig(obs_dim=128)
    learner = Learner(config)

    # 预初始化
    print('[初始化] 预初始化编码器...')
    _ = learner._encode_text('初始化', train=True)
    print('[初始化] 完成')

    # 内置科学知识
    science_knowledge = [
        '牛顿发现了万有引力定律。',
        '水在100摄氏度沸腾。',
        '光速约为每秒30万公里。',
        '地球绕太阳公转一周需要365天。',
        '人类的DNA由四种碱基组成。',
        '声音在空气中的传播速度约为340米每秒。',
        '地球的重力加速度约为9.8米每二次方秒。',
        '化学元素周期表有118种元素。',
        '光合作用将二氧化碳和水转化为葡萄糖和氧气。',
        '热力学第二定律说熵总是增加的。',
        '圆周率π约等于3.14159。',
        '勾股定理说的是直角三角形的两直角边的平方和等于斜边的平方。',
        '人工智能是计算机科学的一个分支。',
        '机器学习是人工智能的一个子领域。',
        '深度学习使用多层神经网络来学习数据的表示。',
        '图灵机是计算理论的基础模型。',
        '秦始皇统一了中国。',
        '第二次世界大战结束于1945年。',
        '珠穆朗玛峰是世界上最高的山峰。',
        '太平洋是世界上最大的海洋。',
        '人类有23对染色体。',
        '大脑是人体最复杂的器官。',
        '力等于质量乘以加速度。',
        '能量守恒定律说能量不能被创造或消灭。',
        '水的化学式是H2O。',
        '氧气的化学式是O2。',
        '供给和需求决定了市场价格。',
        '巴甫洛夫发现了条件反射。',
    ]

    print(f'[学习] 学习 {len(science_knowledge)} 条科学知识...')
    for text in science_knowledge:
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

    print(f'[学习] 完成: {len(learner.knowledge.entities)} 个实体')

    # 从百科学习100条
    print('[学习] 从百科学习100条...')
    count = 0
    with open('data/extracted/baike/baike_qa_train.json', 'r', encoding='utf-8') as f:
        for line in f:
            if count >= 100:
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
            except Exception:
                continue

    print(f'[学习] 完成: {len(learner.knowledge.entities)} 个实体')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/ask':
            params = parse_qs(parsed.query)
            q = params.get('q', [''])[0]
            if q:
                answer = learner.think(q)
                self.send_json({'question': q, 'answer': answer})
            else:
                self.send_json({'error': '缺少参数 q'}, 400)

        elif parsed.path == '/status':
            self.send_json({
                'status': 'running',
                'entities': len(learner.knowledge.entities),
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
            if body:
                # 快速学习
                text_repr = learner._encode_text(body, train=True)
                entities = learner._extract_entities_from_repr(body, text_repr)
                triples = learner._extract_relations_from_repr(body, entities, text_repr)
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
                self.send_json({
                    'learned': body,
                    'entities': len(entities),
                    'triples': len(triples),
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
        pass  # 静默日志


def main():
    print('=' * 50)
    print('学习AI HTTP服务器')
    print('=' * 50)

    init_and_learn()

    port = 8080
    server = HTTPServer(('0.0.0.0', port), Handler)

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
