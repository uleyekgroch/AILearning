# API文档

## 概述

本文档描述了常识推理系统的API接口。

## 基础信息

- **基础URL**: `http://localhost:8000`
- **API版本**: v1.0.0
- **内容类型**: application/json

## 错误处理

所有API响应都遵循以下格式：

```json
{
  "status": "success" | "error",
  "data": {...},
  "error_code": "ERROR_CODE",
  "message": "错误信息"
}
```

## API接口

### 1. 知识管理API

#### 1.1 创建知识库

**请求**
```
POST /api/knowledge/bases
```

**请求体**
```json
{
  "name": "knowledge_base_name"
}
```

**响应**
```json
{
  "status": "success",
  "data": {
    "name": "knowledge_base_name",
    "version": 1,
    "created_at": "2026-06-01T00:00:00"
  }
}
```

#### 1.2 添加事实

**请求**
```
POST /api/knowledge/bases/{kb_name}/facts
```

**请求体**
```json
{
  "fact_id": "fact_001",
  "statement": "水在100度沸腾",
  "subject": "水",
  "predicate": "沸点",
  "object": "100度",
  "confidence": 1.0
}
```

**响应**
```json
{
  "status": "success",
  "data": {
    "fact_id": "fact_001",
    "statement": "水在100度沸腾",
    "subject": "水",
    "predicate": "沸点",
    "object": "100度",
    "confidence": 1.0,
    "created_at": "2026-06-01T00:00:00"
  }
}
```

#### 1.3 查询事实

**请求**
```
GET /api/knowledge/bases/{kb_name}/facts?subject=水
```

**响应**
```json
{
  "status": "success",
  "data": [
    {
      "fact_id": "fact_001",
      "statement": "水在100度沸腾",
      "subject": "水",
      "predicate": "沸点",
      "object": "100度",
      "confidence": 1.0
    }
  ],
  "total": 1
}
```

#### 1.4 获取统计信息

**请求**
```
GET /api/knowledge/bases/{kb_name}/statistics
```

**响应**
```json
{
  "status": "success",
  "data": {
    "name": "knowledge_base_name",
    "total_facts": 100,
    "by_type": {
      "physical": 50,
      "biological": 30
    },
    "avg_confidence": 0.95
  }
}
```

### 2. 推理API

#### 2.1 创建推理会话

**请求**
```
POST /api/reasoning/sessions
```

**请求体**
```json
{
  "session_id": "session_001"
}
```

**响应**
```json
{
  "status": "success",
  "data": {
    "session_id": "session_001",
    "status": "created",
    "created_at": "2026-06-01T00:00:00"
  }
}
```

#### 2.2 添加推理任务

**请求**
```
POST /api/reasoning/sessions/{session_id}/tasks
```

**请求体**
```json
{
  "task_id": "task_001",
  "query": "水在多少度沸腾",
  "reasoning_type": "commonsense"
}
```

**响应**
```json
{
  "status": "success",
  "data": {
    "task_id": "task_001",
    "query": "水在多少度沸腾",
    "reasoning_type": "commonsense",
    "status": "pending",
    "created_at": "2026-06-01T00:00:00"
  }
}
```

#### 2.3 执行推理任务

**请求**
```
POST /api/reasoning/sessions/{session_id}/tasks/{task_id}/execute
```

**响应**
```json
{
  "status": "success",
  "data": {
    "task_id": "task_001",
    "answer": "水在100度沸腾",
    "confidence": 0.95,
    "status": "completed",
    "reasoning_chain": [
      "查询: 水在多少度沸腾",
      "找到相关事实: 水在100度沸腾",
      "返回结果"
    ],
    "created_at": "2026-06-01T00:00:00"
  }
}
```

#### 2.4 获取推理链

**请求**
```
GET /api/reasoning/sessions/{session_id}/tasks/{task_id}/chain
```

**响应**
```json
{
  "status": "success",
  "data": {
    "task_id": "task_001",
    "steps": [
      "查询: 水在多少度沸腾",
      "找到相关事实: 水在100度沸腾",
      "返回结果"
    ],
    "step_count": 3,
    "created_at": "2026-06-01T00:00:00"
  }
}
```

### 3. 查询API

#### 3.1 常识查询

**请求**
```
POST /api/query/commonsense
```

**请求体**
```json
{
  "query": "水",
  "kb_name": "physics_kb"
}
```

**响应**
```json
{
  "status": "success",
  "data": [
    {
      "fact_id": "fact_001",
      "statement": "水在100度沸腾",
      "subject": "水",
      "predicate": "沸点",
      "object": "100度",
      "confidence": 1.0
    }
  ],
  "total": 1
}
```

#### 3.2 带上下文的查询

**请求**
```
POST /api/query/contextual
```

**请求体**
```json
{
  "query": "水",
  "context": {
    "domain": "physics",
    "precision": "high"
  },
  "kb_name": "physics_kb"
}
```

**响应**
```json
{
  "status": "success",
  "data": [
    {
      "fact_id": "fact_001",
      "statement": "水在100度沸腾",
      "subject": "水",
      "predicate": "沸点",
      "object": "100度",
      "confidence": 1.0
    }
  ],
  "total": 1
}
```

## 错误码

| 错误码 | 说明 |
|--------|------|
| NOT_FOUND | 资源不存在 |
| VALIDATION_ERROR | 验证错误 |
| INTERNAL_ERROR | 内部错误 |
| CONFLICT | 资源冲突 |
| UNAUTHORIZED | 未授权 |
| FORBIDDEN | 禁止访问 |

## 示例代码

### Python示例

```python
import requests

# 创建知识库
response = requests.post(
    "http://localhost:8000/api/knowledge/bases",
    json={"name": "physics_kb"}
)
print(response.json())

# 添加事实
response = requests.post(
    "http://localhost:8000/api/knowledge/bases/physics_kb/facts",
    json={
        "fact_id": "fact_001",
        "statement": "水在100度沸腾",
        "subject": "水",
        "predicate": "沸点",
        "object": "100度",
        "confidence": 1.0
    }
)
print(response.json())

# 查询事实
response = requests.get(
    "http://localhost:8000/api/knowledge/bases/physics_kb/facts",
    params={"subject": "水"}
)
print(response.json())
```

### cURL示例

```bash
# 创建知识库
curl -X POST http://localhost:8000/api/knowledge/bases \
  -H "Content-Type: application/json" \
  -d '{"name": "physics_kb"}'

# 添加事实
curl -X POST http://localhost:8000/api/knowledge/bases/physics_kb/facts \
  -H "Content-Type: application/json" \
  -d '{
    "fact_id": "fact_001",
    "statement": "水在100度沸腾",
    "subject": "水",
    "predicate": "沸点",
    "object": "100度",
    "confidence": 1.0
  }'

# 查询事实
curl "http://localhost:8000/api/knowledge/bases/physics_kb/facts?subject=水"
```
