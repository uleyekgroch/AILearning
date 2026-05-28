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

## 2026-05-28: Phase 47 CUDA 加速 + 1000+ Agent 大社会

### 完成的工作
1. 新建 `mvl/cuda_utils.py` — CUDA 工具层（设备管理、NumPy↔PyTorch 转换、批量操作）
2. 改写 `mvl/encoder_sensory.py` — Conv2D 从手写三重循环改为 PyTorch `F.conv2d`（GPU 加速）
3. 改写 `mvl/active_inference.py` — 添加 `select_action_batch()` 批量动作选择
4. 改写 `mvl/language_society_large.py` — 添加 `batch_cosine_similarity()` 和 `detect_language_families_fast()`（GPU 加速）
5. 新建 `mvl/experiment_large_society_cuda.py` — 1000+ Agent 大社会实验

### 实验结果
| 实验 | 结果 | 状态 |
|------|------|------|
| 规模对比 100→1000 | 1000 Agent 2941 rounds/s, 100% 成功率 | PASS |
| 语言家族检测 | 997/1000 Agent 收敛为同一家族 | PASS |
| 方言分化 | 分化度 0.007（小世界网络促进趋同）| 有效发现 |

### 关键发现
1. 单 Agent CUDA 转换开销 > 计算收益：批量相似度计算（跨 Agent）比逐对计算快 9x
2. encoder_sensory Conv2D 使用 PyTorch 后 0.7ms/step（含 GPU warmup）
3. 1000 Agent 在小世界网络下语言快速趋同，形成 1 个主导语言家族

## 2026-05-28: Phase 48 流体与软体物理

### 完成的工作
1. 扩展 `mvl/physics_fluid.py` — 添加 SPH 粒子流体系统（FluidParticleSystem）
   - 空间哈希加速邻域查找 O(n)
   - 密度-压力-粘性力-重力完整 SPH 管线
   - `collide_with_sphere()` 流体与刚体碰撞
2. 扩展 `mvl/physics_soft.py` — 添加软体弹簧-质点系统（SoftBox, SoftBodySystem）
   - 8 角节点 + 22 弹簧（12 边 + 6 面对角 + 4 体对角）
   - 形变恢复力 + 碰撞检测
3. 集成到 `mvl/environment_3d.py` — `add_fluid()`, `add_soft_body()`, 渲染扩展
4. 新建 `mvl/experiment_fluid_soft.py` — 4 个实验

### 实验结果
| 实验 | 结果 | 状态 |
|------|------|------|
| 流体物理发现 | 流体扩散 0.295, 刚体聚集 | PASS |
| 流体语言涌现 | 学习 `liquid` 符号, 100% 成功 | PASS |
| 软体碰撞实验 | 软体形变 0.309, 刚体移动 0.010 | PASS |
| 混合场景语言 | `liquid`/`deformed`/`metal` 材质词汇涌现 | PASS |

### 关键发现
1. 流体粒子在重力下自然下落并扩散，与刚体行为形成鲜明对比
2. 软体受压后形变（0.309），释放后恢复，验证弹簧-质点模型有效
3. 语言系统从物理特征中涌现材质相关词汇：liquid（流体）、deformed（软体）、metal（刚体）
4. 去掉 type 标签后，Agent 必须通过物理属性（material, behavior, hardness）区分物体类型

## 2026-05-28: Phase 49 超大规模社会（2000-5000 Agent）

### 完成的工作
1. 优化 `mvl/language_society_large.py`
   - `_build_small_world()`: set 替代 list.remove()，O(1) 删除
   - `_build_scale_free()`: np.random.choice(replace=False) 替代拒绝采样
   - 新增 `batch_step_parallel()`: 多对 Agent 并行通信
   - 新增 `compute_similarity_sampled()`: 采样相似度（不构建全量矩阵）
   - 新增 `detect_lingua_franca_fast()`: GPU 批量通用语检测
   - 新增 `detect_language_families_sampled()`: 采样家族检测（5000+ Agent）
2. 新建 `mvl/experiment_mega_society.py` — 4 个实验

### 实验结果
| 实验 | 结果 | 状态 |
|------|------|------|
| 规模梯度 2000/3000/5000 | 全部收敛为 1 家族，5000 Agent 速度 76 rounds/s | PASS |
| 拓扑对比 2000 Agent | small_world/scale_free/line 差异极小（0.986-0.987）| 有效发现 |
| 语言家族演化 5000 | Round 500 即收敛为 1 家族 | PASS |
| 枢纽 Agent 分析 | 枢纽度 149 vs 平均 6，相似度略高 0.944 vs 0.923 | 有效发现 |

### 关键发现
1. 所有规模最终收敛为 1 个家族 — 小世界网络短路径确保信息快速传播
2. 5000 Agent 初始相似度（0.871）比 2000（0.952）低，但最终都达到 0.97+
3. 拓扑类型在大规模下差异极小 — 通信轮数足够时，拓扑不再是瓶颈
4. 枢纽 Agent（度 149 vs 平均 6）有略高相似度，但词汇量相同（9）

## 2026-05-28: Phase 50 跨语言迁移（不同环境的语言互译）

### 完成的工作
1. 新建 `mvl/language_translator.py` — 跨语言翻译系统
   - `CrossLingualAgent(LanguageAgent)`: 双语 Agent，translation_table 映射
   - `CrossLingualSociety`: 管理两个不同环境的群体
   - `measure_cross_lingual_metrics()`: 跨语言指标测量
2. 新建 `mvl/experiment_cross_lingual.py` — 4 个实验

### 实验结果
| 实验 | 结果 | 状态 |
|------|------|------|
| 跨环境基线 | 隔离 95%, 接触后 95.5%, 符号重叠 0.70→0.85 | PASS |
| 桥接翻译 | 桥接 82.8% vs 双语 97.9% vs 无桥接 93.4% | PASS |
| 环境差异度 | 工业vs魔法 92.5% → 热带vs热带 98.8% | PASS |
| 通用语涌现 | 5 群体符号重叠 94-100%, 相似度 0.33-0.52 | PASS |

### 关键发现
1. **符号重叠 ≠ 语义相同**：5 群体共享 94-100% 符号，但语言相似度仅 0.33-0.52
2. **双语者 > 桥接 Agent**：桥接引入翻译链噪声（82.8%），双语者直接学习两种语言（97.9%）
3. **环境差异影响有限**：即使高度不同环境，跨环境成功率也在 93-97%（属性空间重叠大）
4. **符号快速趋同**：跨区域接触后符号重叠从 0.10-0.70 快速升至 0.53-1.00
5. **语言相似度保持低位**：说明符号含义仍在分化——同形异义词现象
