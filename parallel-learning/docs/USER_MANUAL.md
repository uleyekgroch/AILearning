# 用户手册

## 概述

常识推理系统是一个基于人工智能的知识推理系统，支持常识知识管理、因果推理、多模态输入等功能。

## 快速开始

### 1. 安装

```bash
# 克隆代码
git clone <repository_url>
cd parallel-learning

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动系统

```bash
python -m src.production.interfaces.rest.main
```

### 3. 访问系统

打开浏览器，访问 `http://localhost:8000`

## 功能模块

### 1. 知识管理

#### 创建知识库

```python
from src.production.application.services.knowledge_service import KnowledgeApplicationService

service = KnowledgeApplicationService()
kb = service.create_knowledge_base("physics_kb")
```

#### 添加事实

```python
fact_data = {
    'fact_id': 'fact_001',
    'statement': '水在100度沸腾',
    'subject': '水',
    'predicate': '沸点',
    'object': '100度',
    'confidence': 1.0
}

service.add_fact("physics_kb", fact_data)
```

#### 查询事实

```python
# 按主题查询
facts = service.query_by_subject("physics_kb", "水")

# 按谓语查询
facts = service.query_by_predicate("physics_kb", "沸点")

# 按宾语查询
facts = service.query_by_object("physics_kb", "100度")
```

### 2. 推理引擎

#### 创建推理会话

```python
from src.production.application.services.reasoning_service import ReasoningApplicationService

service = ReasoningApplicationService()
session = service.create_session("session_001")
```

#### 添加推理任务

```python
task_data = {
    'task_id': 'task_001',
    'query': '水在多少度沸腾',
    'reasoning_type': 'commonsense'
}

service.add_task("session_001", task_data)
```

#### 执行推理

```python
result = service.execute_task("session_001", "task_001")

print(f"答案: {result.answer}")
print(f"置信度: {result.confidence}")
print(f"推理链: {result.reasoning_chain}")
```

### 3. 多模态输入

#### 创建多模态输入

```python
from src.production.application.services.multimodal_service import MultimodalApplicationService
from src.production.domain.multimodal.modality import Modality, ModalityType

service = MultimodalApplicationService()

text_modality = Modality(
    modality_id="text_001",
    modality_type=ModalityType.TEXT.value,
    content="水在100度沸腾",
    metadata={}
)

result = service.process_input([text_modality])
```

### 4. 具身感知

#### 处理感觉输入

```python
from src.production.application.services.embodied_service import EmbodiedApplicationService
from src.production.domain.embodied.sensory_input import SensoryInput, SensoryType

service = EmbodiedApplicationService()

sensory_input = SensoryInput(
    input_id="visual_001",
    sensory_type=SensoryType.VISUAL.value,
    data={"image": "base64_data"},
    timestamp=datetime.now()
)

result = service.process_sensory_input(sensory_input)
```

### 5. 持续学习

#### 处理学习经验

```python
from src.production.application.services.continual_learning_service import ContinualLearningService
from src.production.domain.continual_learning.learning_experience import LearningExperience

service = ContinualLearningService()

experience = LearningExperience(
    experience_id="exp_001",
    input_data={"text": "水在100度沸腾"},
    output_data={"fact": "水沸点100度"},
    feedback={"correct": True, "confidence": 0.95},
    timestamp=datetime.now()
)

result = service.process_learning_experience(experience)
```

## API使用

### 1. 知识管理API

#### 创建知识库

```bash
curl -X POST http://localhost:8000/api/knowledge/bases \
  -H "Content-Type: application/json" \
  -d '{"name": "physics_kb"}'
```

#### 添加事实

```bash
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
```

#### 查询事实

```bash
curl "http://localhost:8000/api/knowledge/bases/physics_kb/facts?subject=水"
```

### 2. 推理API

#### 创建会话

```bash
curl -X POST http://localhost:8000/api/reasoning/sessions \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session_001"}'
```

#### 执行推理

```bash
curl -X POST http://localhost:8000/api/reasoning/sessions/session_001/tasks/task_001/execute
```

### 3. 查询API

#### 常识查询

```bash
curl -X POST http://localhost:8000/api/query/commonsense \
  -H "Content-Type: application/json" \
  -d '{"query": "水", "kb_name": "physics_kb"}'
```

## 配置说明

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

## 常见问题

### 1. 如何添加自定义知识？

```python
from src.production.application.services.knowledge_service import KnowledgeApplicationService

service = KnowledgeApplicationService()

# 创建知识库
service.create_knowledge_base("custom_kb")

# 添加事实
service.add_fact("custom_kb", {
    'fact_id': 'custom_001',
    'statement': '自定义事实',
    'subject': '主题',
    'predicate': '关系',
    'object': '对象',
    'confidence': 0.9
})
```

### 2. 如何提高推理准确率？

1. **增加知识库**: 添加更多相关事实
2. **提高置信度**: 使用更准确的知识
3. **优化查询**: 使用更精确的查询语句

### 3. 如何处理多模态输入？

```python
from src.production.application.services.multimodal_service import MultimodalApplicationService
from src.production.domain.multimodal.modality import Modality, ModalityType

service = MultimodalApplicationService()

# 创建文本模态
text = Modality(
    modality_id="text_001",
    modality_type=ModalityType.TEXT.value,
    content="文本内容",
    metadata={}
)

# 创建图像模态
image = Modality(
    modality_id="image_001",
    modality_type=ModalityType.IMAGE.value,
    content="base64_image_data",
    metadata={}
)

# 融合处理
result = service.fuse_modalities([text, image])
```

### 4. 如何监控系统性能？

```python
from src.production.application.services.knowledge_service import KnowledgeApplicationService

service = KnowledgeApplicationService()

# 获取统计信息
stats = service.get_statistics("kb_name")
print(f"总事实数: {stats['total_facts']}")
print(f"平均置信度: {stats['avg_confidence']}")
```

## 最佳实践

### 1. 知识组织

- 使用清晰的命名规范
- 合理组织知识库结构
- 定期清理过期知识

### 2. 推理优化

- 选择合适的推理类型
- 提供充足的上下文
- 验证推理结果

### 3. 性能优化

- 使用缓存减少查询
- 批量处理提高效率
- 监控系统资源

### 4. 安全建议

- 验证用户输入
- 限制访问权限
- 定期备份数据

## 技术支持

如有问题，请联系技术支持：

- **邮箱**: support@example.com
- **文档**: http://localhost:8000/docs
- **GitHub**: https://github.com/example/commonsense-reasoning
