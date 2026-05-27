# 最小可行学习体 (Minimum Viable Learner)

一个从学习本源出发的AI系统原型。

## 核心理念

不是"更大的模型"，而是"不同的学习范式"：
- 学习信号 = 预测误差（不是标签）
- 驱动力 = 好奇心（不是损失函数）
- 数据来源 = 主动探索（不是被动接收）
- 发展 = 阶段性（不是一次性训练）

## 运行

```bash
cd mvl
python main.py
```

## 架构

- `environment.py` - 2D网格世界环境
- `agent.py` - 学习体（预测模型+好奇心+发展阶段）
- `teacher.py` - 教师Agent（社会交互）
- `main.py` - 主程序
