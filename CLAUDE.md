# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 5. 强制开发规范 - 必须遵守

**本规范是系统开发的强制性标准，所有代码实现必须严格遵守。**

### 5.0 核心设计原则（强制）

**分析和设计必须遵循的原则：**

#### 1. 第一性原理（First Principles Thinking）
**分析问题和技术架构时必须使用第一性原理：**

- **回归本质**：从问题的本质出发，而不是从现有方案或经验出发
- **分解问题**：将复杂问题分解为最基本的组成部分
- **重新构建**：从基本组成部分重新构建解决方案
- **质疑假设**：质疑所有现有假设，不接受"一直都是这样做的"

**应用场景：**
- 架构设计：从系统本质需求出发，不照搬现有架构
- 技术选型：从技术本质特性出发，不盲目跟风
- 问题分析：从问题根本原因出发，不治标不治本

#### 2. DRY原则（Don't Repeat Yourself）
**禁止重复代码：**

- **同一知识只在一个地方表达**
- **相同逻辑必须抽象为公共方法或类**
- **配置信息只定义一次**
- **业务规则只实现一次**

**检查方法：**
- 如果两段代码看起来相似，提取公共部分
- 如果修改一处需要同步修改另一处，说明违反了DRY
- 使用函数、类、模块消除重复

#### 3. KISS原则（Keep It Simple, Stupid）
**保持简单：**

- **简单优于复杂**
- **复杂优于晦涩**
- **可读性优于性能（除非性能是关键需求）**
- **明确优于隐式**

**检查方法：**
- 如果代码需要注释才能理解，考虑简化
- 如果函数超过20行，考虑拆分
- 如果嵌套超过3层，考虑重构

#### 4. SOLID原则（面向对象设计原则）
**必须遵守的SOLID原则：**

- **S - 单一职责原则（Single Responsibility Principle）**
  - 一个类只有一个职责
  - 一个类只有一个变化的原因
  - 如果类有多个职责，拆分为多个类

- **O - 开闭原则（Open/Closed Principle）**
  - 对扩展开放，对修改关闭
  - 新功能通过添加代码实现，而不是修改现有代码
  - 使用抽象和多态实现扩展

- **L - 里氏替换原则（Liskov Substitution Principle）**
  - 子类必须能够替换父类
  - 子类不能改变父类的预期行为
  - 子类只能扩展父类功能，不能限制

- **I - 接口隔离原则（Interface Segregation Principle）**
  - 客户端不应该依赖它不需要的接口
  - 接口应该小而专一
  - 如果接口太大，拆分为多个小接口

- **D - 依赖倒置原则（Dependency Inversion Principle）**
  - 高层模块不应该依赖低层模块，两者都应该依赖抽象
  - 抽象不应该依赖细节，细节应该依赖抽象
  - 依赖抽象，不依赖具体实现

#### 5. YAGNI原则（You Aren't Gonna Need It）
**不要过度设计：**

- **只实现当前需要的功能**
- **不为未来可能的需求编写代码**
- **不过度抽象和泛化**
- **不添加"以防万一"的功能**

**检查方法：**
- 如果代码没有被使用，删除它
- 如果功能没有明确需求，不实现它
- 如果抽象只有一个实现，考虑是否需要抽象

#### 6. 代码行数限制（强制）
**单个文件不超过800行：**

- **类文件**：不超过800行
- **函数**：不超过50行（理想20-30行）
- **方法**：不超过30行（理想10-20行）

**超过限制时的处理：**
1. **识别职责**：识别文件或类的多个职责
2. **分解分离**：将不同职责分离到不同文件或类
3. **遵循原则**：在分解过程中遵循DRY、KISS、SOLID、YAGNI原则
4. **保持内聚**：分解后的模块应该高度内聚
5. **降低耦合**：分解后的模块应该低耦合

**分解方法：**
- **提取类**：将独立职责提取为新类
- **提取接口**：将公共行为提取为接口
- **提取模块**：将相关功能提取为独立模块
- **使用组合**：通过组合而非继承组织代码

### 5.1 软件工程流程（强制）

**开发流程必须遵循：**
```
需求分析 → 领域建模 → 测试设计 → 代码实现 → 代码审查 → 持续集成
```

**每个阶段必须通过门禁才能进入下一阶段。**

### 5.2 领域驱动设计（DDD）- 强制执行

**必须遵守的DDD原则：**

1. **识别限界上下文（Bounded Context）**
   - 知识管理上下文
   - 推理引擎上下文
   - 查询服务上下文

2. **聚合根设计原则**
   - 聚合根是事务边界
   - 聚合根内部对象只能通过聚合根访问
   - 聚合根之间通过ID引用，不直接引用
   - 聚合根发布领域事件

3. **领域事件**
   - 所有状态变更必须通过领域事件
   - 事件必须是不可变的
   - 事件必须包含完整上下文

4. **值对象**
   - 值对象必须是不可变的
   - 值对象通过值比较相等性
   - 值对象不能有业务逻辑

### 5.3 测试驱动开发（TDD）- 强制执行

**必须遵守的TDD流程：**

1. **红灯阶段（Red）**
   - 先写失败的测试
   - 测试定义接口契约
   - 确认测试失败（模块不存在或方法不存在）

2. **绿灯阶段（Green）**
   - 写最小实现让测试通过
   - 只写让测试通过的最少代码
   - 禁止添加测试未覆盖的功能

3. **重构阶段（Refactor）**
   - 优化代码结构
   - 保持测试通过
   - 提高代码质量

**TDD规则：**
- ✅ 必须先写测试，后写实现
- ✅ 测试必须先失败（红灯）
- ✅ 最小实现让测试通过（绿灯）
- ✅ 重构保持测试通过
- ❌ 禁止先写实现后补测试
- ❌ 禁止跳过红灯阶段
- ❌ 禁止过度设计

### 5.4 测试覆盖率要求（强制）

| 测试类型 | 最低覆盖率 | 目标覆盖率 |
|---------|-----------|-----------|
| **单元测试** | 80% | 90%+ |
| **集成测试** | 60% | 80%+ |
| **E2E测试** | 40% | 60%+ |
| **总体覆盖率** | 70% | 85%+ |

### 5.5 代码质量标准（强制）

**必须遵守：**
- 遵循SOLID原则
- 遵循DDD原则
- 使用type hints
- 完整的文档字符串
- 规范的测试命名

**代码审查清单：**
- [ ] 是否遵循DDD原则？
- [ ] 是否遵循TDD流程？
- [ ] 测试覆盖率是否达标？
- [ ] 是否有代码异味？
- [ ] 是否有安全漏洞？

### 5.6 违规处理

**严重违规（必须立即修复）：**
- 跳过TDD流程
- 测试覆盖率低于最低要求
- 聚合根设计违反DDD原则
- 代码存在安全漏洞

**违规后果：**
- 代码审查不通过
- 不能合并到主分支
- 不能部署到生产环境

---

## 6. 开发规范文档

详细的开发规范请参考：
- **[DEVELOPMENT_STANDARDS.md](DEVELOPMENT_STANDARDS.md)** - 完整开发规范文档

**本文档是系统开发的强制性标准，所有代码实现必须严格遵守。违反规范的代码将被拒绝合并。**
