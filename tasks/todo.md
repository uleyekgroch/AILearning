# Phase 3 实现计划：三大核心认知能力

## 目标
实现让系统从"模式匹配"进化为"举一反三"的三大根本能力：
1. **跨领域类比迁移** — 学一个领域，自动迁移到新领域
2. **持续终身学习** — 学新不忘旧（防灾难性遗忘）
3. **抽象概念形成** — 从具体实例中涌现高层抽象概念

## 实现顺序
- [x] 1. 头文件设计（3个模块）
- [x] 2. TDD：先写测试
- [x] 3. 实现模块 1：AnalogicalTransferEngine（Gentner 结构映射）
- [x] 4. 实现模块 2：ContinualLearner（EWC 弹性约束 + 经验回放）
- [x] 5. 实现模块 3：AbstractConceptEngine（原型抽象 + 层次化 + 分化）
- [x] 6. 集成到 Learner 类（6 个新接口方法 + 3 个子系统）
- [x] 7. 编译验证：**215 测试全绿，1233 断言通过**

## 文件清单

### 新增头文件
| 文件 | 行数 | 说明 |
|------|------|------|
| `include/ai_learning/learning/analogical_transfer.hpp` | 191 | 类比迁移引擎 |
| `include/ai_learning/learning/continual_learner.hpp` | 229 | 持续学习引擎 |
| `include/ai_learning/learning/abstract_concept.hpp` | 249 | 抽象概念引擎 |

### 新增实现
| 文件 | 行数 | 说明 |
|------|------|------|
| `src/learning/analogical_transfer.cpp` | 440 | 类比迁移实现 |
| `src/learning/continual_learner.cpp` | 440 | 持续学习实现 |
| `src/learning/abstract_concept.cpp` | 650 | 抽象概念实现 |

### 新增测试
| 文件 | 行数 | 说明 |
|------|------|------|
| `tests/learning/test_phase3_advanced_cognition.cpp` | 611 | 29 个测试 + 4 集成测试 |

### 修改文件
| 文件 | 变更 |
|------|------|
| `include/ai_learning/core/learner.hpp` | +3 include, +6 接口方法, +3 成员变量 |
| `src/core/learner.cpp` | +3 初始化, +4 方法实现 |

## 核心算法

### 1. AnalogicalTransferEngine（Gentner 结构映射理论）
- `surface_similarity_`: Jaccard 系数计算属性重叠
- `relational_alignment_`: 关系动词匹配（"导致"、"限制"、"→"等）
- `match_attributes_`: 贪心属性匹配（精确匹配 + 子串匹配）
- `transfer`: 映射→迁移→验证→评估 四步流水线
- `learn_from_failure`: 从失败类比中提取教训

### 2. ContinualLearner（EWC + 经验回放）
- Fisher 重要性: `importance = confidence × (1 + log(1+usage)/5)`
- 保护等级: High(≥0.8) / Medium(≥0.5) / Low(≥0.2) / None
- 冲突解决: keep_old(High) / merge(Medium) / branch(Low) / override(None)
- 经验回放: 优先级采样 = importance × 1/(1+replay_count)
- 遗忘检测: current_confidence / original_confidence < threshold

### 3. AbstractConceptEngine（Rosch 原型理论）
- 核心属性提取: 在 ≥60% 实例中出现的属性
- 原型特征: 所在实例特征的平均值
- 概念强度: instance_factor(0.4) + consistency(0.4) + relation(0.2)
- 分化检测: 某属性仅在 20%-80% 实例出现 → 触发分化
- 抽象层次: Concrete → Basic → Superordinate → Relational → Meta

## 设计原则
- 每个模块独立可测，不依赖 Learner（通过引用注入依赖）
- C++20 `concept` 关键字避让（变量名使用 `cpt`、`prototype`）
- 值语义成员变量，与已有代码风格一致
