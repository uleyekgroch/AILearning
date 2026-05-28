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
