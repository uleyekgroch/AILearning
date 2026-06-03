#!/usr/bin/env python3
"""学习AI HTTP服务器

启动后通过HTTP提问：
  curl http://localhost:8080/ask?q=什么是人工智能
  或浏览器打开 http://localhost:8080/ask?q=什么是人工智能

学习语料：
  curl -X POST http://localhost:8080/learn -d "牛顿发现了万有引力定律"
"""

import sys
import os
import json
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.learner import Learner, LearnerConfig

# 全局learner实例
learner = None


def init_learner():
    """初始化学习体并加载语料"""
    global learner
    config = LearnerConfig(obs_dim=128)
    learner = Learner(config)

    # 内置知识语料
    corpus = [
        # 科学常识
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

        # 数学
        '圆周率π约等于3.14159。',
        '勾股定理说的是直角三角形的两直角边的平方和等于斜边的平方。',
        '一元二次方程的求根公式是x等于负b加减根号下b方减4ac除以2a。',
        '导数描述的是函数在某一点的变化率。',
        '积分是导数的逆运算。',

        # 计算机科学
        '人工智能是计算机科学的一个分支。',
        '机器学习是人工智能的一个子领域。',
        '深度学习使用多层神经网络来学习数据的表示。',
        '算法的时间复杂度描述了算法运行时间随输入规模增长的趋势。',
        '数据结构是计算机存储和组织数据的方式。',
        '图灵机是计算理论的基础模型。',
        'TCP/IP是互联网的基础协议。',
        '操作系统管理计算机的硬件和软件资源。',

        # 历史
        '秦始皇统一了中国。',
        '第二次世界大战结束于1945年。',
        '美国独立宣言签署于1776年。',
        '法国大革命开始于1789年。',
        '万里长城始建于春秋战国时期。',

        # 地理
        '珠穆朗玛峰是世界上最高的山峰。',
        '亚马逊河是世界上流量最大的河流。',
        '撒哈拉沙漠是世界上最大的热沙漠。',
        '太平洋是世界上最大的海洋。',

        # 生物
        '人类有23对染色体。',
        '大脑是人体最复杂的器官。',
        '心脏是人体的泵血器官。',
        '肺是人体的呼吸器官。',
        '肝脏是人体最大的内脏器官。',

        # 因果关系
        '下雨导致地面湿了。',
        '缺乏运动导致肥胖。',
        '学习导致知识增长。',
        '太阳照射导致温度升高。',
        '病毒入侵导致生病。',

        # 物理
        '力等于质量乘以加速度。',
        '能量守恒定律说能量不能被创造或消灭。',
        '动量守恒定律说在一个封闭系统中总动量保持不变。',
        '电磁波包括可见光、无线电波和X射线。',
        '量子力学描述了微观粒子的行为。',
        '相对论说时间和空间是相对的。',

        # 化学
        '水的化学式是H2O。',
        '氧气的化学式是O2。',
        '二氧化碳的化学式是CO2。',
        '酸和碱中和生成盐和水。',
        '氧化反应是物质与氧结合的反应。',

        # 经济
        '供给和需求决定了市场价格。',
        '通货膨胀是物价持续上涨的现象。',
        'GDP是衡量一个国家经济总量的指标。',

        # 心理学
        '巴甫洛夫发现了条件反射。',
        '马斯洛提出了需求层次理论。',
        '弗洛伊德是精神分析学派的创始人。',
        '认知失调理论说人们倾向于保持信念的一致性。',
    ]

    print(f'[学习] 开始学习 {len(corpus)} 条语料...')
    for i, text in enumerate(corpus):
        learner.learn_from_text(text, source='corpus')
        if (i + 1) % 10 == 0:
            print(f'[学习] 已学习 {i+1}/{len(corpus)} 条')

    print(f'[学习] 学习完成！')
    print(f'[学习] 知识图谱实体数: {len(learner.knowledge.entities)}')

    # 执行一次巩固
    consolidation = learner.consolidate()
    print(f'[巩固] 巩固完成: {consolidation}')


class LearnerHandler(BaseHTTPRequestHandler):
    """HTTP请求处理器"""

    def do_GET(self):
        """处理GET请求"""
        parsed = urlparse(self.path)

        if parsed.path == '/ask':
            # 提问接口
            params = parse_qs(parsed.query)
            question = params.get('q', [''])[0]

            if not question:
                self.send_json({'error': '缺少参数 q'}, 400)
                return

            # 思考并回答
            answer = learner.think(question)

            self.send_json({
                'question': question,
                'answer': answer,
            })

        elif parsed.path == '/status':
            # 状态接口
            self.send_json({
                'status': 'running',
                'entities': len(learner.knowledge.entities),
                'learning_stats': learner._learning_stats,
            })

        elif parsed.path == '/':
            # 首页
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
        .info { color: #888; font-size: 12px; margin-top: 10px; }
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
    <div class="info">
        <p>API: GET /ask?q=问题 | POST /learn -d "语料" | GET /status</p>
    </div>
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
        """处理POST请求"""
        parsed = urlparse(self.path)

        if parsed.path == '/learn':
            # 学习接口
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')

            if not body:
                self.send_json({'error': '缺少学习内容'}, 400)
                return

            result = learner.learn_from_text(body, source='http')
            self.send_json({
                'learned': body,
                'entities': result['entities'],
                'triples': len(result['triples']),
            })

        else:
            self.send_json({'error': 'Not found'}, 404)

    def send_json(self, data, status=200):
        """发送JSON响应"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def send_html(self, html, status=200):
        """发送HTML响应"""
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def log_message(self, format, *args):
        """简化日志"""
        print(f'[HTTP] {args[0]}')


def main():
    """启动服务器"""
    print('=' * 50)
    print('学习AI HTTP服务器')
    print('=' * 50)

    # 初始化学习体
    init_learner()

    # 启动HTTP服务器
    port = 8080
    server = HTTPServer(('0.0.0.0', port), LearnerHandler)
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
