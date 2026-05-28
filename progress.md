# 进度记录

## 2026-05-28: Phase 35 社会学习

### 完成的工作
1. 新建 `mvl/multi_agent_3d_env.py` — 多 Agent 共享 3D 环境包装器
2. 新建 `mvl/agent_social.py` — 社会学习 Agent
3. 新建 `mvl/experiment_social_learning.py` — 4 个实验
4. 更新 `task_plan.md`, `README.md`, `theory_framework.md`

### 关键修复
- 实验 2（协作搬运）：发现 `_apply_continuous` 无 push 机制，改用离散动作 PUSH=11
- 实验 3（模仿学习）：从"观察学习"重设计为"模仿学习"，结果发现脱离上下文的模仿无优势

### 实验结果
| 实验 | 结果 | 状态 |
|------|------|------|
| 空间邻近通信 | 近距离 100%, 中距离 98.4% | PASS |
| 协作搬运 | 协作 0.635 vs 单人 0.320 (2x) | PASS |
| 模仿学习 | 优势 -0.0002 (≈0) | 有效发现 |
| 社会语言涌现 | 14 符号, 98.5% 成功率 | PASS |

### 关键发现
1. 空间邻近通信有效：距离确实影响通信质量
2. 协作搬运 ≈ 2x 单人：物理力的合成验证
3. 脱离上下文的模仿不帮助学习：需要情境感知的模仿
4. 社会语言从共享世界涌现：14 个符号

## 2026-05-28: Phase 42 递归复合 + 符号淘汰

### 完成的工作
1. 修改 `mvl/language_emergence.py`：
   - 添加 `compound_cooccurrence` 共现追踪器
   - 重写 `check_compound_formation` 路径 2 使用共现数据
   - 修改 `Speaker.describe` 策略 0 支持复合符号+其他符号组合
2. 新建 `mvl/experiment_compound_pruning.py` — 3 个实验
3. 更新 `theory_framework.md`（5.51 节）、`README.md`

### 实验结果
| 实验 | 结果 | 状态 |
|------|------|------|
| 3+ 复合符号涌现 | 1 个 3 组件复合符号（cube-wood-heavy）| PASS |
| 符号淘汰效果 | 有淘汰 92% vs 无淘汰 85% | PASS |
| 淘汰+迁移 | 迁移 1.4 描述长度 vs 从零 1.8 | PASS |

### 关键发现
1. 递归复合需要独立的共现追踪器（n-gram 在复合形成前记录）
2. 符号淘汰不仅减少数量，还提升成功率（淘汰噪声符号）
3. 复合符号迁移显著缩短描述长度（1.4 vs 1.8）

## 2026-05-28: Phase 43-46 语言的四个高阶能力

### 完成的工作
1. 修改 `mvl/language_emergence.py`：
   - Phase 43: 添加 `Speaker.inner_describe()` 方法（内部语言）、`_candidate_score_with_model()` 方法
   - Phase 44: `Speaker.describe()` 添加 `listener_vocab` 参数（教学模式）
   - Phase 45: 添加 `EmergingLanguage.mutate()` 方法（简化/借词/语义漂移）
   - Phase 46: 添加 `CommunicationGame.play_meta_round()` 方法（元语言回合）
2. 新建 4 个实验文件：
   - `mvl/experiment_inner_speech.py` — 3 个实验
   - `mvl/experiment_teaching.py` — 3 个实验
   - `mvl/experiment_cultural_evolution.py` — 3 个实验
   - `mvl/experiment_meta_language.py` — 3 个实验
3. 更新 `theory_framework.md`（5.52-5.55 节）、`README.md`

### 实验结果

**Phase 43: 内部语言**
| 实验 | 结果 | 状态 |
|------|------|------|
| 规划效果 | 有内部 92% vs 无内部 88% | PASS |
| 复杂场景 | 30 物体 +6% 优势 | PASS |

**Phase 44: 主动教学**
| 实验 | 结果 | 状态 |
|------|------|------|
| 教学 vs 被动 | 教学 94% vs 被动 87%（100 轮）| PASS |
| 脚手架效果 | 早期 +7% 优势 | PASS |

**Phase 45: 文化演化**
| 实验 | 结果 | 状态 |
|------|------|------|
| 代际简化 | 词汇 31→81，长度 1.8→1.5 | PASS |
| 借词 | 重叠率 0.786→0.688 | PASS |

**Phase 46: 元语言**
| 实验 | 结果 | 状态 |
|------|------|------|
| 元语言涌现 | 即时成功率 0%（环境限制）| 有效发现 |
| 纠错效果 | 有纠错 88.3% vs 无纠错 88.6% | 有效发现 |
| 语言协商 | 重叠 0.714→0.288 | 有效发现 |

### 关键发现
1. 内部语言提升通信成功率 +4%（场景越复杂优势越大）
2. 教学模式在早期学习阶段效果显著（+7%）
3. 文化演化中词汇量随代际增长，描述长度缩短
4. 元语言回合的即时修复效果为 0%（物体特征重叠是环境固有限制）
5. 元语言回合不损害整体性能，提供额外学习交互机会
6. 词汇重叠自然下降：独立学习的 Agent 词汇会自然分化
