# 任务清单：复杂物理任务系统

## 目标
实现目标导向的物理任务（物体操控、流体导航、工具使用），让 agent 必须理解物理规律才能完成任务。

## 步骤

### Phase 1: 任务基础框架
- [x] 创建 `mvl/tasks_base.py`（PhysicsTask 基类、TaskCategory、Difficulty）
- [x] 修改 `mvl/environment_physics.py`（集成任务层）
- [x] 创建 `mvl/tasks_manipulation.py`（PushToGoalTask）

### Phase 2: Agent 集成
- [x] 修改 `mvl/agent.py`（奖励融合：70%好奇心+30%任务）
- [x] 修改 `mvl/agent_3d.py`（透传 extrinsic_reward）
- [x] 创建 `mvl/task_environments.py`（环境工厂函数）

### Phase 3: 更多任务
- [x] 完善 `mvl/tasks_manipulation.py`（StackObjectsTask, SortByMaterialTask）
- [x] 创建 `mvl/tasks_fluid.py`（NavigateInFluidTask, AvoidHazardousFluidTask）
- [x] 创建 `mvl/tasks_tool_use.py`（IndirectPushTask, BridgeGapTask）

### Phase 4: 对比实验
- [x] 创建 `mvl/experiment_tasks.py`（三条件对比：无任务 vs 任务+好奇心 vs 纯任务）
- [x] 运行实验，分析结果
- [x] 验证：任务驱动学习 vs 纯好奇心学习的差异

### 实验结果
- 任务成功率：0%（任务太难，agent 的好奇心驱动动作选择无法完成任务）
- 泛化能力：任务+好奇心（3.63）vs 纯好奇心（9.55），提升 62%
- 关键发现：任务环境中的学习产生了更好的物理理解，即使任务未完成

### Phase 5: 任务管理器
- [x] 在 `mvl/tasks_base.py` 中添加 TaskManager
- [ ] 根据发展阶段自动选择任务（待实现）

### Phase 8: 更长的发展（Piaget 四阶段）
- [x] 扩展 `mvl/agent.py` DevelopmentEngine 添加形式运算阶段
- [x] 添加新指标：抽象推理、假设验证、反事实多样性、元认知
- [x] 改进分类准确率计算（基于物体特征聚类一致性）
- [x] 添加 epsilon-greedy 探索（模拟幼儿随机试错）
- [x] 创建 `mvl/experiment_development.py`（四阶段发展对比实验）
- [x] 运行实验，分析结果

### 实验结果（发展阶段）
- 完整四阶段轨迹：感知运动(0) → 前运算(128) → 具体运算(1004) → 形式运算(1170)
- 有阶段限制泛化能力高 40.4%（泛化误差 2.33 vs 3.91）
- 具体运算阶段最长（876步），验证了"高阶能力依赖低阶充分发展"

### Phase 6: 更丰富的感知
- [x] 创建 `mvl/perception_visual.py`（VisualField：射线投射 8x8 视觉场）
- [x] 创建 `mvl/perception_auditory.py`（AuditorySystem：碰撞/流体/移动声音事件）
- [x] 修改 `mvl/environment_physics.py`（集成视觉场和听觉事件到观测中）
- [x] 创建 `mvl/agent_multimodal.py`（MultimodalAgent：视觉/听觉/触觉编码器+融合）
- [x] 创建 `mvl/experiment_multimodal.py`（三条件对比实验）
- [x] 运行实验，分析结果

### 实验结果（多模态感知）
- 结构化特征：训练误差 0.0000，泛化误差 4.5898（过度拟合）
- 纯视觉场：训练误差 0.1137，泛化误差 0.2574（最佳泛化）
- 视觉+听觉：训练误差 0.0124，泛化误差 0.3290
- 关键发现：**原始感官输入比预处理特征产生更好的泛化能力**（泛化误差降低 94%）

### Phase 7: 真正的社会交互
- [x] 修复 `mvl/agent.py`（social_interactions 统计 bug，添加模仿学习支持）
- [x] 修改 `mvl/teacher.py`（激活 JointAttentionManager，脚手架渐退，示范动作）
- [x] 创建 `mvl/peer_learning.py`（PeerAgent 同伴学习，PeerLearningEnvironment 双 agent 环境）
- [x] 创建 `mvl/experiment_social.py`（四条件对比实验）
- [x] 运行实验，分析结果

### 实验结果（社会交互）
- 无社会交互：泛化误差 8.5679
- 静态脚手架：泛化误差 7.9044
- 脚手架渐退：泛化误差 **5.0696**（最佳，降低 41%）
- 渐退+模仿学习：泛化误差 8.5899
- 关键发现：**脚手架渐退显著提升泛化能力**，但过多的支持（静态脚手架）反而阻碍自主学习

## 验证标准
- 任务成功率随时间提升
- 任务+好奇心 > 纯好奇心 > 纯任务（物理理解泛化测试）
- Agent 能在新场景中泛化物理知识

### Phase 8: 更长的发展（Piaget 四阶段）
- [x] 修改 `mvl/agent.py`（添加形式运算阶段，复杂晋升条件）
- [x] 修改 `mvl/agent_3d.py`（添加 epsilon-greedy 探索）
- [x] 创建 `mvl/experiment_development.py`（四阶段发展对比实验）
- [x] 运行实验，分析结果

### 实验结果（发展阶段）
- 感知运动 → 前运算：128 步
- 前运算 → 具体运算：876 步
- 具体运算 → 形式运算：166 步
- 有阶段限制泛化误差 2.3304 vs 无阶段限制 3.9086，**泛化能力高 40.4%**

### Phase 9: 自由能原理形式化
- [x] 创建 `mvl/free_energy.py`（概率生成模型，精度加权学习）
- [x] 创建 `mvl/variational_inference.py`（变分推断，贝叶斯信念更新）
- [x] 创建 `mvl/active_inference.py`（主动推理，期望自由能）
- [x] 创建 `mvl/agent_fep.py`（FEP Agent）
- [x] 创建 `mvl/experiment_fep.py`（MSE vs FEP 对比实验）
- [x] 运行实验，分析结果

### 实验结果（自由能原理）
- MSE 最终误差 0.0743 vs FEP 0.2335（MSE 更低）
- FEP 符号学习 3 个 vs MSE 1 个（FEP 更多）
- 泛化误差 10.9064 vs 10.9398（持平）
- ±2σ 覆盖率 60.8%（理想 95.4%，方差网络需改善）
- 关键发现：**FEP 的价值不在于预测精度，而在于不确定性估计和符号学习**

## 错误记录
