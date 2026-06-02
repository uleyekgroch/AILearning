# Phase 4 实现计划：三大增强智能模块

## 目标
实现让系统从"理性学习器"进化为"类人智能体"的三大增强能力：
1. **社会性/观察学习** — 从观察他人行为中学习（Bandura 社会学习理论）
2. **情感驱动学习** — 情绪调制记忆编码与学习策略（McGaugh 情绪-记忆理论）
3. **知识重组/顿悟** — 重新组织知识产生创造性突破（Ohlsson 顿悟理论）

## 文件清单

### 新增头文件
| 文件 | 行数 | 说明 |
|------|------|------|
| `include/ai_learning/learning/social_learning.hpp` | 218 | 社会性/观察学习引擎 |
| `include/ai_learning/learning/emotion_engine.hpp` | 198 | 情感驱动学习引擎 |
| `include/ai_learning/learning/insight_engine.hpp` | 221 | 知识重组/顿悟引擎 |

### 新增实现
| 文件 | 行数 | 说明 |
|------|------|------|
| `src/learning/social_learning.cpp` | 372 | 社会学习实现 |
| `src/learning/emotion_engine.cpp` | 300 | 情感引擎实现 |
| `src/learning/insight_engine.cpp` | 492 | 顿悟引擎实现 |

### 新增测试
| 文件 | 行数 | 说明 |
|------|------|------|
| `tests/learning/test_phase4_enhanced_intelligence.cpp` | 686 | 36 个测试 + 4 集成测试 |

### 修改文件
| 文件 | 变更 |
|------|------|
| `include/ai_learning/core/learner.hpp` | +3 include, +6 接口方法, +3 成员变量 |
| `src/core/learner.cpp` | +3 初始化, +3 方法实现 |

## 实现顺序
- [x] 1. 三个头文件设计
- [x] 2. TDD：先写测试
- [x] 3. 实现 SocialLearningEngine（Bandura 观察学习五阶段）
- [x] 4. 实现 EmotionEngine（McGaugh 情绪调制 + Yerkes-Dodson）
- [x] 5. 实现 InsightEngine（Ohlsson 约束释放 + Mednick 远程联想）
- [x] 6. 集成到 Learner（6 新接口 + 3 子系统）
- [x] 7. 编译验证：**251 测试全绿，1301 断言通过**

## 核心算法

### 1. SocialLearningEngine（Bandura 社会学习理论 1977）
- **观察建模**: observe() → 存储 + 更新榜样评估 + 尝试模式提取
- **模式提取**: extract_pattern() → 频率 ≥50% 的动作作为共同步骤
- **榜样评估**: 指数移动平均(α=0.3)更新专业度，可信度随观察次数增长
- **策略模仿**: 逐步复制 + 能力适配（过滤超出自身能力的步骤）
- **社会反馈**: correction/suggestion/praise 三类反馈，更新反馈者可信度

### 2. EmotionEngine（McGaugh + Yerkes-Dodson）
- **情绪维度**: Russell 环状模型（效价 × 唤醒度 × 支配度）
- **多巴胺 RPE**: prediction error = actual - expected（Schultz, 1997）
- **记忆调制**: encoding_boost = 1 + arousal×0.5 + |valence|×0.3
- **Yerkes-Dodson**: performance = -4(arousal - optimal)² + 1（倒U型）
- **情绪调节**: 认知重评（极端情绪向中心压缩）
- **离散标签**: 效价-唤醒度映射 → excited/confident/frustrated/bored/curious 等

### 3. InsightEngine（Ohlsson + Mednick + Gestalt）
- **远程联想**: 检测元素间共同特征 + 互补性 → 发现隐藏联系
- **约束释放**: 释放弱约束(strength < threshold) → 新视角
- **视角转换**: 相反/微观/宏观/类比/历史 五种视角
- **创造性评估**: novelty(0.3) + utility(0.3) + surprise(0.2) + elegance(0.2)
- **新颖度检测**: 与历史顿悟的 Jaccard 距离

## 跨 Phase 集成
- observe_behavior() 自动将学到的策略注册到 ContinualLearner 保护
- try_insight() 产生顿悟时自动触发 EmotionEngine 的兴奋事件
- 情感调制参数影响社会学习的记忆编码强度
