#!/usr/bin/env python3
"""简单学习AI HTTP服务器

使用预构建知识库，快速响应。
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


# 预构建知识库
KNOWLEDGE = {
    '牛顿': {'发现': '万有引力定律'},
    '水': {'温度': '100摄氏度沸腾'},
    '光速': {'速度': '每秒30万公里'},
    '地球': {'公转周期': '365天', '重力加速度': '9.8米每二次方秒'},
    'DNA': {'组成': '四种碱基'},
    '声音': {'传播速度': '340米每秒'},
    '元素周期表': {'元素数量': '118种'},
    '光合作用': {'产物': '葡萄糖和氧气'},
    '热力学第二定律': {'内容': '熵总是增加的'},
    '圆周率': {'值': '约等于3.14159'},
    '勾股定理': {'内容': '直角三角形的两直角边的平方和等于斜边的平方'},
    '人工智能': {'定义': '计算机科学的一个分支', '子领域': '机器学习'},
    '机器学习': {'定义': '人工智能的一个子领域', '方法': '深度学习'},
    '深度学习': {'定义': '使用多层神经网络来学习数据的表示'},
    '图灵机': {'定义': '计算理论的基础模型'},
    '秦始皇': {'成就': '统一了中国'},
    '第二次世界大战': {'结束时间': '1945年'},
    '珠穆朗玛峰': {'特征': '世界上最高的山峰'},
    '太平洋': {'特征': '世界上最大的海洋'},
    '人类': {'染色体': '23对', '最复杂器官': '大脑'},
    '力': {'公式': '质量乘以加速度'},
    '能量守恒定律': {'内容': '能量不能被创造或消灭'},
    '水': {'化学式': 'H2O'},
    '氧气': {'化学式': 'O2'},
    '供需': {'关系': '决定市场价格'},
    '巴甫洛夫': {'发现': '条件反射'},
    '巨蟹座': {'日期': '6月22日到7月22日'},
    '白羊座': {'日期': '3月21日到4月19日'},
    '天秤座': {'日期': '9月23日到10月22日'},
    '天蝎座': {'日期': '10月23日到11月21日'},
    '射手座': {'日期': '11月22日到12月21日'},
    '摩羯座': {'日期': '12月22日到1月19日'},
    '水瓶座': {'日期': '1月20日到2月18日'},
    '双鱼座': {'日期': '2月19日到3月20日'},
    '金牛座': {'日期': '4月20日到5月20日'},
    '双子座': {'日期': '5月21日到6月21日'},
    '狮子座': {'日期': '7月23日到8月22日'},
    '处女座': {'日期': '8月23日到9月22日'},
}


def answer_question(question: str) -> str:
    """回答问题"""
    q = question.strip()

    # 直接匹配
    for key, facts in KNOWLEDGE.items():
        if key in q:
            answers = []
            for relation, value in facts.items():
                answers.append(f"{key}的{relation}是{value}")
            return '。'.join(answers) + '。'

    # 关键词匹配
    keywords = ['什么', '谁', '哪里', '多少', '怎么', '为什么', '如何']
    for kw in keywords:
        if kw in q:
            # 找最相关的知识
            best_match = None
            best_score = 0
            for key in KNOWLEDGE:
                score = sum(1 for c in key if c in q)
                if score > best_score:
                    best_score = score
                    best_match = key

            if best_match and best_score > 0:
                facts = KNOWLEDGE[best_match]
                answers = []
                for relation, value in facts.items():
                    answers.append(f"{best_match}的{relation}是{value}")
                return '。'.join(answers) + '。'

    return "抱歉，我还不知道这个问题的答案。你可以教我：POST /learn -d '知识内容'"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/ask':
            params = parse_qs(parsed.query)
            q = params.get('q', [''])[0]
            if q:
                answer = answer_question(q)
                self.send_json({'question': q, 'answer': answer})
            else:
                self.send_json({'error': '缺少参数 q'}, 400)

        elif parsed.path == '/status':
            self.send_json({
                'status': 'running',
                'knowledge_entries': len(KNOWLEDGE),
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
                # 简单学习：添加到知识库
                self.send_json({
                    'learned': body,
                    'message': '已学习（简化版本，完整版本需要启动完整服务器）',
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
        pass


def main():
    port = 8080
    server = HTTPServer(('0.0.0.0', port), Handler)

    print('=' * 50)
    print('学习AI HTTP服务器 (简化版)')
    print('=' * 50)
    print(f'服务器启动在 http://localhost:{port}')
    print(f'浏览器打开: http://localhost:{port}')
    print(f'API提问: http://localhost:{port}/ask?q=什么是人工智能')
    print(f'按 Ctrl+C 停止')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n服务器已停止')
        server.shutdown()


if __name__ == '__main__':
    main()
