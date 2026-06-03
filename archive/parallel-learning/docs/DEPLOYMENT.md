# 部署文档

## 概述

本文档描述了常识推理系统的部署方法。

## 环境要求

### 系统要求

- **操作系统**: Linux, macOS, Windows
- **Python**: 3.11+
- **内存**: 4GB+ (推荐8GB)
- **磁盘**: 1GB+

### 依赖包

```bash
pip install -r requirements.txt
```

## 部署方式

### 1. 本地开发部署

```bash
# 克隆代码
git clone <repository_url>
cd parallel-learning

# 安装依赖
pip install -r requirements.txt

# 运行测试
python -m pytest tests/

# 启动服务
python -m src.production.interfaces.rest.main
```

### 2. Docker部署

#### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "src.production.interfaces.rest.main"]
```

#### 构建镜像

```bash
docker build -t commonsense-reasoning .
```

#### 运行容器

```bash
docker run -p 8000:8000 commonsense-reasoning
```

### 3. Docker Compose部署

#### docker-compose.yml

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/commonsense
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=commonsense
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

#### 启动服务

```bash
docker-compose up -d
```

## 配置管理

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| DATABASE_URL | 数据库连接URL | sqlite:///commonsense.db |
| REDIS_URL | Redis连接URL | redis://localhost:6379 |
| LOG_LEVEL | 日志级别 | INFO |
| API_PORT | API端口 | 8000 |
| API_HOST | API主机 | 0.0.0.0 |

### 配置文件

创建`.env`文件：

```env
DATABASE_URL=postgresql://user:password@localhost:5432/commonsense
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
API_PORT=8000
API_HOST=0.0.0.0
```

## 生产环境部署

### 1. 使用Gunicorn

```bash
pip install gunicorn

gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.production.interfaces.rest.main:app
```

### 2. 使用Nginx反向代理

```nginx
server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. 使用Systemd服务

创建`/etc/systemd/system/commonsense.service`：

```ini
[Unit]
Description=Commonsense Reasoning System
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/commonsense
Environment="PATH=/opt/commonsense/venv/bin"
ExecStart=/opt/commonsense/venv/bin/python -m src.production.interfaces.rest.main
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl enable commonsense
sudo systemctl start commonsense
```

## 监控

### 1. 健康检查

```bash
curl http://localhost:8000/health
```

### 2. 日志查看

```bash
# Docker日志
docker logs <container_id>

# Systemd日志
journalctl -u commonsense -f
```

### 3. 性能监控

```bash
# 访问统计
curl http://localhost:8000/metrics
```

## 备份

### 数据库备份

```bash
# PostgreSQL备份
pg_dump -U user -d commonsense > backup.sql

# 恢复
psql -U user -d commonsense < backup.sql
```

### 文件备份

```bash
# 备份配置和数据
tar -czf backup.tar.gz .env data/
```

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 查找占用端口的进程
   lsof -i :8000
   
   # 杀死进程
   kill -9 <PID>
   ```

2. **数据库连接失败**
   - 检查数据库服务是否启动
   - 检查连接URL是否正确
   - 检查防火墙设置

3. **内存不足**
   - 增加系统内存
   - 调整Python内存限制
   - 优化数据结构

### 日志分析

```bash
# 查看错误日志
grep "ERROR" /var/log/commonsense/app.log

# 查看慢查询
grep "slow" /var/log/commonsense/app.log
```

## 升级

### 1. 备份数据

```bash
# 备份数据库
pg_dump -U user -d commonsense > backup_$(date +%Y%m%d).sql

# 备份配置
cp .env .env.backup
```

### 2. 更新代码

```bash
git pull origin main
```

### 3. 更新依赖

```bash
pip install -r requirements.txt
```

### 4. 运行迁移

```bash
python -m src.production.infrastructure.persistence.migrations
```

### 5. 重启服务

```bash
# Docker
docker-compose restart

# Systemd
sudo systemctl restart commonsense
```

## 性能优化

### 1. 数据库优化

- 添加索引
- 优化查询
- 使用连接池

### 2. 缓存优化

- 使用Redis缓存
- 设置合理的TTL
- 缓存热点数据

### 3. 应用优化

- 异步处理
- 批量操作
- 压缩传输

## 安全建议

### 1. 网络安全

- 使用HTTPS
- 配置防火墙
- 限制访问IP

### 2. 应用安全

- 输入验证
- SQL注入防护
- XSS防护

### 3. 数据安全

- 加密敏感数据
- 定期备份
- 访问控制
