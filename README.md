# PJ-数学题.pdf 信息整理

## 一、任务背景
### 赛题信息
- **赛题名称**：小学数学应用题自动解题
- **比赛官网**：https://www.datafountain.cn/competitions/467
- **背景**：应用题结合文字表述与推理计算，是评估机器阅读理解能力的典型场景，也是K12教研核心内容，对AI在教育领域的发展有重要意义。
- **任务目标**：输入小学数学1-6年级校内应用题，输出对应数字答案（类似数学填空题）。

### 示例
| 问题 | 答案 |
|------|------|
| 商店有4框苹果，每框55千克，已经卖出135千克，还剩多少千克苹果? | 85 |
| 玩具厂生产了960个电子玩具，每3个装一盒，每5盒装一箱，一共装了多少箱? | 64 |

---

## 二、数据集与评估
### 数据说明
- 训练集：`train.json`（12000条，已清洗）
- 测试集：`test.json`（8000条）
- 提交模板：`submit.csv`（含`id`和`ret`两列）
- 整理资源：https://github.com/AI-FDU/Math_Solver

### 评估标准
- **指标**：正确率（预测值与真实值一致的样本占比）
- **提交要求**：
  - 格式：CSV文件
  - 频率：每日每账号3次
  - 入口：“作品提交”

---

## 三、Baseline方案
### 技术路线
- **基础模型**：Qwen2.5-0.5B（https://huggingface.co/Qwen/Qwen2.5-0.5B）
- **微调方法**：LoRA（Low-Rank Adaptation）+ PEFT（Parameter-Efficient Fine-Tuning）
- **核心代码**：

```python
from peft import get_peft_model, LoraConfig
from transformers import Trainer
model = ...
config = LoraConfig(...)
model = get_peft_model(model, config)
trainer = Trainer(model=model, ...)
trainer.train()
```

- **完整代码**：https://github.com/AI-FDU/Math_Solver/blob/main/qwen_ft.py

---

## 四、可扩展方向
| 方案 | 说明 |
|------|------|
| **思维链（CoT）** | 优化Prompt，无需微调：<br>• Few-shot CoT：加入带推理步骤的示例<br>• Zero-shot CoT：添加“Let's think step by step”等关键词 |
| **数据构建** | • 用高级模型补充训练集解题步骤<br>• 合成新数据（如替换数字）<br>• SFT训练含步骤的答案，用正则提取最终数字 |
| **RLHF+DPO** | 构建偏好数据（正确答案+错误答案），使用DPO直接优化策略：<br>参考代码：https://github.com/ShawhinT/YouTube-Blog/tree/main/LLMs/dpo |
| **GRPO** | 组相对策略优化（DeepSeek-R1采用）：生成多响应→奖励打分→组内归一化优势→PPO风格更新<br>参考代码：https://github.com/huggingface/open-r1 |

> 注：鼓励尝试其他大模型或小模型方案，需提供实现、结果与分析。

---

## 五、规则说明
### 参赛形式
| 类型 | 要求 |
|------|------|
| **组队** | • 最多5人，15周汇报进展，16周提交报告<br>• 禁止中途换队（特殊情况除外） |
| **个人** | • 15周前提交报告草稿，16周提交完整报告<br>• 要求略低于组队 |

### 时间与评分
| 节点 | 截止日期 | 要求 |
|------|----------|------|
| 比赛截止 | 2026-06-11 | 提交排名截图、CSV至eLearning |
| 报告截止 | 2026-06-19 | 提交代码、4页报告、PPT（可选）至eLearning |

#### 评分公式
- **总分**：`S = min(S1 + S2, 15)`（满分15分）
- `S1 = 比赛正确率 × 15`（数值分）
- `S2 = 方案数 × 单方案分`（组队3分/方案，个人5分/方案）

---

## 六、其他说明
- 禁止使用测试集训练，推理模型需≤0.5B（推荐Qwen系列）。
- 允许更换同难度比赛，需提前向助教报备。
- 摆烂组员经核实后扣分，创新尝试即使效果不佳也可获认可。

---
**人工智能助教团队**  
2026年5月22日