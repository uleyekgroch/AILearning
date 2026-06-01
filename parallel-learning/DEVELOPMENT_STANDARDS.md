# 系统开发规范 - 必须遵守

## 🎯 核心开发原则

**本规范是系统开发的强制性标准，所有代码实现必须严格遵守。**

---

## 0. 核心设计原则（强制）

### 0.1 第一性原理（First Principles Thinking）

**分析问题和技术架构时必须使用第一性原理：**

#### 定义
第一性原理是一种思维方法，要求从问题的本质出发，而不是从现有方案或经验出发。它要求：
1. **回归本质**：找到问题的最基本组成部分
2. **分解问题**：将复杂问题分解为最基本的元素
3. **重新构建**：从基本元素重新构建解决方案
4. **质疑假设**：质疑所有现有假设，不接受"一直都是这样做的"

#### 应用场景

**架构设计：**
```
❌ 错误做法：照搬现有架构，因为"大家都这么做"
✅ 正确做法：从系统本质需求出发，设计最适合的架构

示例：
问题：需要一个知识推理系统
错误：直接使用现有的知识图谱方案
正确：
1. 分析知识推理的本质需求是什么？
2. 知识如何表示最有效？
3. 推理如何实现最高效？
4. 从这些基本需求出发设计架构
```

**技术选型：**
```
❌ 错误做法：选择最流行的技术，因为"大家都在用"
✅ 正确做法：从技术本质特性出发，选择最适合的技术

示例：
问题：需要存储百万级知识条目
错误：直接使用Neo4j，因为"知识图谱都用它"
正确：
1. 数据的本质特征是什么？（结构化/非结构化）
2. 查询的本质模式是什么？（图遍历/向量搜索/全文搜索）
3. 性能的本质要求是什么？（读多写少/写多读少）
4. 从这些本质需求选择技术
```

**问题分析：**
```
❌ 错误做法：直接修复表面症状
✅ 正确做法：从问题根本原因出发

示例：
问题：系统响应慢
错误：直接加缓存
正确：
1. 响应慢的根本原因是什么？（算法复杂度/IO瓶颈/资源不足）
2. 这个原因是否可以消除？
3. 如何从根本上解决？
4. 从根本原因出发设计解决方案
```

#### 检查清单
- [ ] 是否从问题本质出发？
- [ ] 是否质疑了所有假设？
- [ ] 是否分解到最基本元素？
- [ ] 是否从基本元素重新构建？
- [ ] 是否避免了"经验主义"陷阱？

---

### 0.2 DRY原则（Don't Repeat Yourself）

**禁止重复代码：**

#### 定义
DRY原则要求系统的每一个知识或逻辑都只在一个地方表达，避免重复。重复代码会导致：
- 修改困难：需要同时修改多处
- 不一致风险：可能漏改导致不一致
- 维护成本高：重复代码增加维护负担

#### 应用规则

**代码层面：**
```python
# ❌ 违反DRY：重复的验证逻辑
def create_user(name, email):
    if not name:
        raise ValueError("Name required")
    if not email:
        raise ValueError("Email required")
    # ...

def update_user(name, email):
    if not name:
        raise ValueError("Name required")
    if not email:
        raise ValueError("Email required")
    # ...

# ✅ 遵循DRY：提取公共验证
def validate_user_input(name, email):
    if not name:
        raise ValueError("Name required")
    if not email:
        raise ValueError("Email required")

def create_user(name, email):
    validate_user_input(name, email)
    # ...

def update_user(name, email):
    validate_user_input(name, email)
    # ...
```

**配置层面：**
```python
# ❌ 违反DRY：重复的配置
DB_HOST = "localhost"
DB_PORT = 5432

# 在多个文件中重复定义
# file1.py: DB_HOST = "localhost"
# file2.py: DB_HOST = "localhost"

# ✅ 遵循DRY：统一配置
# config.py
DATABASE = {
    'host': 'localhost',
    'port': 5432,
}

# 使用配置
from config import DATABASE
```

**业务规则层面：**
```python
# ❌ 违反DRY：重复的业务规则
def calculate_discount_vip(user):
    if user.level == 'gold':
        return 0.2
    elif user.level == 'silver':
        return 0.1

def calculate_discount_promotion(user):
    if user.level == 'gold':
        return 0.2
    elif user.level == 'silver':
        return 0.1

# ✅ 遵循DRY：统一业务规则
def get_discount_rate(user):
    if user.level == 'gold':
        return 0.2
    elif user.level == 'silver':
        return 0.1
    return 0.0

def calculate_discount_vip(user):
    return get_discount_rate(user)

def calculate_discount_promotion(user):
    return get_discount_rate(user)
```

#### 检查方法
- [ ] 是否有重复的代码块？
- [ ] 是否有重复的配置信息？
- [ ] 是否有重复的业务规则？
- [ ] 修改一处是否需要同步修改另一处？
- [ ] 是否可以提取为公共方法或类？

---

### 0.3 KISS原则（Keep It Simple, Stupid）

**保持简单：**

#### 定义
KISS原则要求系统设计和代码实现保持简单，避免不必要的复杂性。简单的设计和代码：
- 易于理解
- 易于维护
- 易于测试
- 易于扩展

#### 应用规则

**代码层面：**
```python
# ❌ 违反KISS：过度复杂的实现
def process_data(data):
    result = []
    for item in data:
        if item is not None:
            if isinstance(item, dict):
                if 'value' in item:
                    if item['value'] > 0:
                        result.append(item['value'] * 2)
    return result

# ✅ 遵循KISS：简单直接的实现
def process_data(data):
    return [
        item['value'] * 2
        for item in data
        if item and isinstance(item, dict) and item.get('value', 0) > 0
    ]
```

**设计层面：**
```python
# ❌ 违反KISS：过度设计
class AbstractFactory:
    @abstractmethod
    def create_builder(self): pass

class ConcreteFactory(AbstractFactory):
    def create_builder(self):
        return ConcreteBuilder()

class AbstractBuilder:
    @abstractmethod
    def build(self): pass

class ConcreteBuilder(AbstractBuilder):
    def build(self):
        return Product()

# ✅ 遵循KISS：简单设计
def create_product():
    return Product()
```

**函数设计：**
```python
# ❌ 违反KISS：函数太长，逻辑复杂
def process_order(order):
    # 验证订单（20行）
    # 计算价格（30行）
    # 处理支付（25行）
    # 更新库存（15行）
    # 发送通知（10行）
    # 总共100行

# ✅ 遵循KISS：拆分为小函数
def process_order(order):
    validate_order(order)
    price = calculate_price(order)
    process_payment(order, price)
    update_inventory(order)
    send_notification(order)
```

#### 检查方法
- [ ] 代码是否易于理解？
- [ ] 函数是否足够短（<30行）？
- [ ] 嵌套是否足够浅（<3层）？
- [ ] 是否需要注释才能理解？
- [ ] 是否有更简单的实现方式？

---

### 0.4 SOLID原则（面向对象设计原则）

**必须遵守的SOLID原则：**

#### S - 单一职责原则（Single Responsibility Principle）
**一个类只有一个职责：**

```python
# ❌ 违反SRP：一个类有多个职责
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def validate(self):
        # 验证逻辑
        pass

    def save_to_database(self):
        # 持久化逻辑
        pass

    def send_email(self):
        # 邮件发送逻辑
        pass

# ✅ 遵循SRP：每个类只有一个职责
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

class UserValidator:
    def validate(self, user):
        # 验证逻辑
        pass

class UserRepository:
    def save(self, user):
        # 持久化逻辑
        pass

class EmailService:
    def send(self, user):
        # 邮件发送逻辑
        pass
```

#### O - 开闭原则（Open/Closed Principle）
**对扩展开放，对修改关闭：**

```python
# ❌ 违反OCP：需要修改现有代码来添加新功能
class DiscountCalculator:
    def calculate(self, user, amount):
        if user.level == 'vip':
            return amount * 0.8
        elif user.level == 'svip':
            return amount * 0.7
        # 添加新等级需要修改这个方法

# ✅ 遵循OCP：通过扩展添加新功能
from abc import ABC, abstractmethod

class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, amount):
        pass

class VipDiscount(DiscountStrategy):
    def calculate(self, amount):
        return amount * 0.8

class SvipDiscount(DiscountStrategy):
    def calculate(self, amount):
        return amount * 0.7

# 添加新等级只需添加新类，无需修改现有代码
class SvvipDiscount(DiscountStrategy):
    def calculate(self, amount):
        return amount * 0.6

class DiscountCalculator:
    def __init__(self, strategy: DiscountStrategy):
        self.strategy = strategy

    def calculate(self, amount):
        return self.strategy.calculate(amount)
```

#### L - 里氏替换原则（Liskov Substitution Principle）
**子类必须能够替换父类：**

```python
# ❌ 违反LSP：子类改变了父类的预期行为
class Bird:
    def fly(self):
        return "Flying"

class Ostrich(Bird):
    def fly(self):
        raise Exception("Ostrich can't fly")  # 改变了父类行为

# ✅ 遵循LSP：子类保持父类的预期行为
from abc import ABC, abstractmethod

class Bird(ABC):
    @abstractmethod
    def move(self):
        pass

class FlyingBird(Bird):
    def move(self):
        return "Flying"

class Ostrich(Bird):
    def move(self):
        return "Running"
```

#### I - 接口隔离原则（Interface Segregation Principle）
**客户端不应该依赖它不需要的接口：**

```python
# ❌ 违反ISP：接口太大
class Worker(ABC):
    @abstractmethod
    def work(self):
        pass

    @abstractmethod
    def eat(self):
        pass

    @abstractmethod
    def sleep(self):
        pass

class Robot(Worker):
    def work(self):
        return "Working"

    def eat(self):
        raise Exception("Robot doesn't eat")  # 不需要这个方法

    def sleep(self):
        raise Exception("Robot doesn't sleep")  # 不需要这个方法

# ✅ 遵循ISP：接口小而专一
class Workable(ABC):
    @abstractmethod
    def work(self):
        pass

class Eatable(ABC):
    @abstractmethod
    def eat(self):
        pass

class Sleepable(ABC):
    @abstractmethod
    def sleep(self):
        pass

class Robot(Workable):
    def work(self):
        return "Working"

class Human(Workable, Eatable, Sleepable):
    def work(self):
        return "Working"

    def eat(self):
        return "Eating"

    def sleep(self):
        return "Sleeping"
```

#### D - 依赖倒置原则（Dependency Inversion Principle）
**高层模块不应该依赖低层模块，两者都应该依赖抽象：**

```python
# ❌ 违反DIP：高层模块依赖低层模块
class MySQLDatabase:
    def save(self, data):
        # MySQL specific implementation
        pass

class UserService:
    def __init__(self):
        self.db = MySQLDatabase()  # 直接依赖具体实现

    def save_user(self, user):
        self.db.save(user)

# ✅ 遵循DIP：依赖抽象
from abc import ABC, abstractmethod

class Database(ABC):
    @abstractmethod
    def save(self, data):
        pass

class MySQLDatabase(Database):
    def save(self, data):
        # MySQL specific implementation
        pass

class PostgreSQLDatabase(Database):
    def save(self, data):
        # PostgreSQL specific implementation
        pass

class UserService:
    def __init__(self, db: Database):  # 依赖抽象
        self.db = db

    def save_user(self, user):
        self.db.save(user)
```

---

### 0.5 YAGNI原则（You Aren't Gonna Need It）

**不要过度设计：**

#### 定义
YAGNI原则要求只实现当前需要的功能，不为未来可能的需求编写代码。过度设计会导致：
- 代码复杂度增加
- 维护成本增加
- 开发时间增加
- 可能引入不必要的bug

#### 应用规则

**功能层面：**
```python
# ❌ 违反YAGNI：为未来需求编写代码
class UserService:
    def get_user(self, user_id):
        # 当前需求：根据ID获取用户
        pass

    def get_user_by_email(self, email):
        # 当前不需要，但"以后可能需要"
        pass

    def get_user_by_phone(self, phone):
        # 当前不需要，但"以后可能需要"
        pass

    def get_user_by_username(self, username):
        # 当前不需要，但"以后可能需要"
        pass

# ✅ 遵循YAGNI：只实现当前需要的功能
class UserService:
    def get_user(self, user_id):
        # 当前需求：根据ID获取用户
        pass

    # 其他方法等到真正需要时再添加
```

**抽象层面：**
```python
# ❌ 违反YAGNI：过度抽象
from abc import ABC, abstractmethod

class Repository(ABC):
    @abstractmethod
    def get(self, id):
        pass

    @abstractmethod
    def save(self, entity):
        pass

    @abstractmethod
    def delete(self, id):
        pass

    @abstractmethod
    def find(self, criteria):
        pass

    # 当前只用到get和save，但"以后可能用到"其他方法

# ✅ 遵循YAGNI：只抽象当前需要的
class UserRepository:
    def get(self, user_id):
        # 当前需要
        pass

    def save(self, user):
        # 当前需要
        pass

    # 等到真正需要时再抽象
```

**配置层面：**
```python
# ❌ 违反YAGNI：过度配置化
class Config:
    def __init__(self):
        self.database_host = 'localhost'
        self.database_port = 5432
        self.database_name = 'mydb'
        self.database_user = 'user'
        self.database_password = 'pass'
        self.database_pool_size = 10
        self.database_max_overflow = 20
        self.database_pool_timeout = 30
        self.database_pool_recycle = 3600
        # 当前只用到host、port、name，但"以后可能用到"其他配置

# ✅ 遵循YAGNI：只配置当前需要的
class Config:
    def __init__(self):
        self.database_host = 'localhost'
        self.database_port = 5432
        self.database_name = 'mydb'
        # 等到真正需要时再添加其他配置
```

#### 检查方法
- [ ] 这个功能当前是否被使用？
- [ ] 这个抽象当前是否有多个实现？
- [ ] 这个配置当前是否被使用？
- [ ] 是否在为"以后可能需要"编写代码？
- [ ] 是否可以等到真正需要时再添加？

---

### 0.6 代码行数限制（强制）

**单个文件不超过800行：**

#### 限制标准

| 类型 | 限制 | 理想行数 |
|------|------|---------|
| **文件** | ≤800行 | ≤500行 |
| **类** | ≤500行 | ≤300行 |
| **函数** | ≤50行 | ≤20行 |
| **方法** | ≤30行 | ≤15行 |
| **嵌套层级** | ≤3层 | ≤2层 |

#### 超过限制时的处理流程

```
1. 识别职责
   ↓
2. 分解分离
   ↓
3. 遵循原则（DRY、KISS、SOLID、YAGNI）
   ↓
4. 保持内聚
   ↓
5. 降低耦合
```

#### 分解方法

**提取类：**
```python
# ❌ 一个类有多个职责（超过800行）
class UserManager:
    # 用户验证（200行）
    # 用户持久化（200行）
    # 用户权限（200行）
    # 用户通知（200行）
    # 总共800行

# ✅ 提取为多个类
class UserValidator:  # 200行
    pass

class UserRepository:  # 200行
    pass

class PermissionManager:  # 200行
    pass

class NotificationService:  # 200行
    pass
```

**提取接口：**
```python
# ❌ 一个类实现了多个不相关的功能
class DataProcessor:
    def process_csv(self): pass
    def process_json(self): pass
    def process_xml(self): pass
    def validate_csv(self): pass
    def validate_json(self): pass
    def validate_xml(self): pass

# ✅ 提取为接口和实现
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    @abstractmethod
    def process(self): pass

    @abstractmethod
    def validate(self): pass

class CSVProcessor(DataProcessor):
    def process(self): pass
    def validate(self): pass

class JSONProcessor(DataProcessor):
    def process(self): pass
    def validate(self): pass
```

**提取模块：**
```python
# ❌ 一个文件包含多个不相关的类
# models.py (800行)
class User: pass
class Order: pass
class Product: pass
class Category: pass
class Payment: pass

# ✅ 提取为多个模块
# models/user.py
class User: pass

# models/order.py
class Order: pass

# models/product.py
class Product: pass

# models/category.py
class Category: pass

# models/payment.py
class Payment: pass
```

**使用组合：**
```python
# ❌ 使用继承导致类层次过深
class Animal:
    pass

class Mammal(Animal):
    pass

class Dog(Mammal):
    pass

class PetDog(Dog):
    pass

class TrainedPetDog(PetDog):
    pass

# ✅ 使用组合组织代码
class Animal:
    pass

class Dog(Animal):
    def __init__(self, behavior, training):
        self.behavior = behavior  # 组合
        self.training = training  # 组合
```

#### 检查清单
- [ ] 文件是否超过800行？
- [ ] 类是否超过500行？
- [ ] 函数是否超过50行？
- [ ] 方法是否超过30行？
- [ ] 嵌套是否超过3层？
- [ ] 是否可以提取为新的类或模块？
- [ ] 是否可以使用组合替代继承？

---

## 1. 软件工程流程

### 1.1 开发流程（强制）

```
需求分析 → 领域建模 → 测试设计 → 代码实现 → 代码审查 → 持续集成
    ↓         ↓         ↓         ↓         ↓         ↓
 明确需求    DDD建模    TDD测试    红绿重构   同行评审    自动化测试
```

### 1.2 阶段门禁（Stage Gates）

每个阶段必须通过门禁才能进入下一阶段：

| 阶段 | 门禁条件 | 验收标准 |
|------|---------|---------|
| **需求分析** | 需求文档完整、无歧义 | 需求评审通过 |
| **领域建模** | 限界上下文清晰、聚合根明确 | 领域模型评审通过 |
| **测试设计** | 测试用例覆盖所有场景 | 测试评审通过 |
| **代码实现** | 所有测试通过、代码规范 | 代码审查通过 |
| **代码审查** | 无高危问题、性能达标 | 审查报告完成 |
| **持续集成** | CI/CD流水线通过 | 自动化部署成功 |

---

## 2. 领域驱动设计（DDD）- 强制执行

### 2.1 限界上下文（Bounded Context）

**必须识别和定义的上下文：**

```
┌─────────────────────────────────────────────────────────┐
│                    系统限界上下文                        │
├─────────────────────────────────────────────────────────┤
│ 1. 知识管理上下文 (Knowledge Management)                │
│    - 聚合根: KnowledgeBase                              │
│    - 实体: CommonsenseFact, CausalRule                  │
│    - 值对象: Triple, Confidence, FactType               │
│                                                         │
│ 2. 推理引擎上下文 (Reasoning Engine)                    │
│    - 聚合根: ReasoningSession                           │
│    - 实体: ReasoningTask, ReasoningStep                 │
│    - 值对象: Query, Result, Chain                       │
│                                                         │
│ 3. 查询服务上下文 (Query Service)                       │
│    - 领域服务: QueryService                             │
│    - 值对象: QueryRequest, QueryResponse                │
│    - 规格: QuerySpecification                           │
└─────────────────────────────────────────────────────────┘
```

### 2.2 聚合根设计原则（强制）

**每个聚合根必须包含：**

```python
class AggregateRoot(ABC):
    """聚合根基类 - 所有聚合根必须继承"""
    
    # 1. 唯一标识
    id: str
    
    # 2. 版本控制
    version: int
    
    # 3. 时间戳
    created_at: datetime
    updated_at: datetime
    
    # 4. 领域事件
    _events: List[DomainEvent]
    
    # 5. 不可变性保证
    def get_events(self) -> List[DomainEvent]:
        """获取领域事件（事件溯源）"""
        return self._events.copy()
    
    def clear_events(self):
        """清空已发布的事件"""
        self._events.clear()
```

**聚合根设计规则：**
- ✅ 聚合根是事务边界
- ✅ 聚合根内部对象只能通过聚合根访问
- ✅ 聚合根之间通过ID引用，不直接引用
- ✅ 聚合根发布领域事件
- ❌ 禁止跨聚合根的直接修改
- ❌ 禁止聚合根内部对象直接访问外部

### 2.3 领域事件（Domain Events）

**所有状态变更必须通过领域事件：**

```python
@dataclass(frozen=True)
class DomainEvent:
    """领域事件基类"""
    event_id: str
    occurred_on: datetime
    aggregate_id: str
    aggregate_version: int
    
    @abstractmethod
    def event_type(self) -> str:
        pass

class FactAddedEvent(DomainEvent):
    """事实添加事件"""
    fact_id: str
    statement: str
    subject: str
    predicate: str
    object: str
    confidence: float
```

### 2.4 值对象设计原则

**值对象必须是不可变的：**

```python
@dataclass(frozen=True)
class ValueObject:
    """值对象基类"""
    pass

class Triple(ValueObject):
    """三元组值对象"""
    subject: str
    predicate: str
    object: str
    
    def __post_init__(self):
        if not self.subject or not self.predicate or not self.object:
            raise ValueError("Triple cannot have empty fields")

class Confidence(ValueObject):
    """置信度值对象"""
    value: float
    
    def __post_init__(self):
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
```

---

## 3. 测试驱动开发（TDD）- 强制执行

### 3.1 TDD流程（红-绿-重构）

```
红灯 (Red)          绿灯 (Green)         重构 (Refactor)
    ↓                   ↓                    ↓
 写失败测试           写最小实现           优化代码
    ↓                   ↓                    ↓
 测试全部失败         测试全部通过         保持测试通过
    ↓                   ↓                    ↓
 定义接口契约         满足契约             提高质量
```

### 3.2 TDD规则（强制）

**必须遵守的规则：**

1. **先写测试，后写实现**
   - 测试定义接口契约
   - 实现满足契约
   - 禁止先写实现后补测试

2. **测试必须先失败（红灯）**
   - 写完测试后必须运行，确认失败
   - 失败原因必须是"模块不存在"或"方法不存在"
   - 禁止跳过红灯阶段

3. **最小实现让测试通过（绿灯）**
   - 只写让测试通过的最少代码
   - 禁止添加测试未覆盖的功能
   - 禁止过度设计

4. **重构保持测试通过**
   - 重构后所有测试必须仍然通过
   - 重构不能改变外部行为
   - 重构必须有明确目的

### 3.3 测试分层（强制）

```
┌─────────────────────────────────────────────────────────┐
│                    测试金字塔                            │
├─────────────────────────────────────────────────────────┤
│  E2E测试 (10%)                                          │
│  - 完整系统功能验证                                      │
│  - 用户场景测试                                         │
├─────────────────────────────────────────────────────────┤
│  集成测试 (20%)                                         │
│  - 组件间交互测试                                       │
│  - 外部依赖测试                                         │
├─────────────────────────────────────────────────────────┤
│  单元测试 (70%)                                         │
│  - 领域模型测试                                         │
│  - 业务逻辑测试                                         │
│  - 边界条件测试                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.4 测试命名规范（强制）

```python
class TestKnowledgeBase:
    """测试类命名：Test + 被测类名"""
    
    def test_should_create_knowledge_base_with_valid_name(self):
        """测试方法命名：test_should_ + 预期行为 + when + 条件"""
        pass
    
    def test_should_raise_error_when_name_is_empty(self):
        """异常场景命名：test_should_raise_error_when + 条件"""
        pass
    
    def test_should_return_facts_when_query_by_subject(self):
        """查询场景命名：test_should_return_xxx_when + 条件"""
        pass
```

### 3.5 测试覆盖率要求（强制）

| 测试类型 | 最低覆盖率 | 目标覆盖率 |
|---------|-----------|-----------|
| **单元测试** | 80% | 90%+ |
| **集成测试** | 60% | 80%+ |
| **E2E测试** | 40% | 60%+ |
| **总体覆盖率** | 70% | 85%+ |

---

## 4. 代码质量标准

### 4.1 代码规范（强制）

**Python代码规范：**
- 遵循 PEP 8
- 使用 type hints
- 使用 dataclass 或 Pydantic
- 文档字符串必须完整

**代码组织：**
```
src/
├── production/
│   ├── domain/                    # 领域层（核心业务逻辑）
│   │   ├── knowledge/            # 知识管理限界上下文
│   │   │   ├── __init__.py
│   │   │   ├── knowledge_base.py # 聚合根
│   │   │   ├── commonsense_fact.py # 实体
│   │   │   ├── causal_rule.py    # 实体
│   │   │   ├── knowledge_index.py # 值对象
│   │   │   └── events.py         # 领域事件
│   │   └── reasoning/            # 推理引擎限界上下文
│   │       ├── __init__.py
│   │       ├── reasoning_session.py # 聚合根
│   │       ├── reasoning_task.py # 实体
│   │       └── events.py         # 领域事件
│   ├── application/              # 应用层（用例编排）
│   │   ├── services/            # 应用服务
│   │   └── commands/            # 命令处理器
│   ├── infrastructure/          # 基础设施层（技术实现）
│   │   ├── persistence/        # 持久化
│   │   ├── messaging/          # 消息传递
│   │   └── external/           # 外部服务
│   └── interfaces/              # 接口层（API）
│       ├── rest/               # REST API
│       └── graphql/            # GraphQL API
```

### 4.2 设计原则（强制）

**SOLID原则：**
- **S** - 单一职责原则
- **O** - 开闭原则
- **L** - 里氏替换原则
- **I** - 接口隔离原则
- **D** - 依赖倒置原则

**DDD原则：**
- 领域模型反映业务语言
- 聚合根保护业务不变量
- 领域事件实现松耦合
- 限界上下文明确边界

### 4.3 代码审查清单（强制）

**每次代码提交必须检查：**

```markdown
## 代码审查清单

### 设计
- [ ] 是否遵循DDD原则？
- [ ] 聚合根设计是否合理？
- [ ] 领域事件是否正确使用？

### 测试
- [ ] 是否遵循TDD流程？
- [ ] 测试覆盖率是否达标？
- [ ] 测试命名是否规范？

### 代码质量
- [ ] 是否遵循SOLID原则？
- [ ] 是否有代码异味？
- [ ] 是否有安全漏洞？

### 文档
- [ ] 是否有完整的文档字符串？
- [ ] 是否有变更日志？
- [ ] 是否有API文档？
```

---

## 5. 持续集成/持续部署（CI/CD）

### 5.1 CI/CD流水线（强制）

```yaml
# CI/CD流水线阶段
stages:
  - name: 代码检查
    steps:
      - lint: 代码风格检查
      - type_check: 类型检查
      - security_scan: 安全扫描
  
  - name: 单元测试
    steps:
      - unit_tests: 运行单元测试
      - coverage: 检查覆盖率
  
  - name: 集成测试
    steps:
      - integration_tests: 运行集成测试
      - contract_tests: 契约测试
  
  - name: 构建部署
    steps:
      - build: 构建镜像
      - deploy_staging: 部署到预发布
      - smoke_tests: 冒烟测试
  
  - name: 生产部署
    steps:
      - deploy_production: 部署到生产
      - monitoring: 监控告警
```

### 5.2 质量门禁（强制）

**每次提交必须通过：**
- ✅ 所有单元测试通过
- ✅ 代码覆盖率 ≥ 70%
- ✅ 无高危安全漏洞
- ✅ 代码风格检查通过
- ✅ 类型检查通过

**每次部署必须通过：**
- ✅ 所有集成测试通过
- ✅ 性能测试达标
- ✅ 安全扫描通过
- ✅ 代码审查完成

---

## 6. 文档要求

### 6.1 必须文档（强制）

**每个模块必须包含：**
1. **README.md** - 模块说明、使用方法
2. **API文档** - 接口说明、参数、返回值
3. **设计文档** - 架构设计、设计决策
4. **变更日志** - 版本变更记录

**每个类必须包含：**
1. **类文档字符串** - 类的职责、使用场景
2. **方法文档字符串** - 方法功能、参数说明、返回值、异常
3. **示例代码** - 使用示例

### 6.2 文档模板

```python
class KnowledgeBase:
    """知识库聚合根
    
    知识库是知识管理限界上下文的核心聚合根，
    负责管理常识事实和因果规则的生命周期。
    
    职责：
    - 管理常识事实的添加、更新、删除
    - 提供事实查询和检索功能
    - 维护知识库的版本和一致性
    
    使用场景：
    - 创建和管理知识库
    - 添加和查询常识事实
    - 维护知识库的完整性
    
    Example:
        >>> kb = KnowledgeBase(name="my_kb")
        >>> fact = CommonsenseFact(
        ...     fact_id="fact_001",
        ...     statement="水在100度沸腾",
        ...     subject="水",
        ...     predicate="沸点",
        ...     object="100度",
        ...     confidence=1.0
        ... )
        >>> kb.add_fact(fact)
        >>> assert kb.fact_count == 1
    """
    
    def add_fact(self, fact: CommonsenseFact) -> None:
        """添加常识事实到知识库
        
        将新的常识事实添加到知识库中，并发布领域事件。
        如果事实ID已存在，将抛出异常。
        
        Args:
            fact: 要添加的常识事实，必须是有效的CommonsenseFact实例
        
        Raises:
            ValueError: 如果fact为None
            DuplicateFactError: 如果fact_id已存在
        
        Example:
            >>> kb = KnowledgeBase(name="my_kb")
            >>> fact = CommonsenseFact(fact_id="f1", ...)
            >>> kb.add_fact(fact)
        """
        pass
```

---

## 7. 违规处理

### 7.1 违规等级

**严重违规（必须立即修复）：**
- 跳过TDD流程
- 测试覆盖率低于最低要求
- 聚合根设计违反DDD原则
- 代码存在安全漏洞

**一般违规（必须在下个迭代修复）：**
- 文档不完整
- 代码风格不规范
- 测试命名不规范

### 7.2 审计机制

**定期审计：**
- 每周代码审查
- 每月质量报告
- 每季度架构评审

---

## 8. 总结

### 核心要求（强制）

1. **必须遵循DDD原则**
   - 识别限界上下文
   - 设计聚合根
   - 使用领域事件

2. **必须遵循TDD流程**
   - 先写测试，后写实现
   - 红-绿-重构循环
   - 测试覆盖率达标

3. **必须遵循软件工程流程**
   - 需求分析
   - 领域建模
   - 测试设计
   - 代码实现
   - 代码审查
   - 持续集成

### 口号

**"没有测试的代码是垃圾代码！"**
**"没有文档的代码是不存在的代码！"**
**"不遵循DDD的代码是无法维护的代码！"**

---

**本文档版本：v1.0.0**
**生效日期：2026年06月01日**
**审核人：技术负责人**
**批准人：项目经理**
