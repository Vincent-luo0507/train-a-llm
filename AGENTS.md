# AGENTS.md — 复刻 CS336 模型训练过程

> 目标：从零复刻 Stanford CS336 (Language Modeling from Scratch) 的模型训练全流程，
> 不依赖高级封装库，核心组件自己实现。AI 负责写代码，**思路由用户指导，AI 必须先讲清楚核心思路再动手**。

## 协作规则（强约束）

1. **核心思路先行**：每个模块动工前，AI 必须用文字明确指出——
   - 本模块要解决什么问题（为什么需要它）
   - 核心思想 / 数学原理（公式 + 直觉解释）
   - 实现路径与关键设计选择（如有分叉，列出选项 + 各自 trade-off，等用户拍板）
2. **用户确认后才写代码**：思路被用户认可或修改后，AI 才开始实现。
3. **代码即讲解**：代码中关键步骤加注释，标注对应讲义的公式/章节；用户可随时追问任意一行。
4. **验证环节内置**：每个模块自带测试或 sanity check（如与 `torch` 官方实现对比数值），交付前 AI 需额外自检一轮再宣告完成。
5. **术语双语**：讨论中保留英文术语（如 BPE、cross-entropy、learning rate warmup），解释用中文。

## 复刻路线图（按 CS336 Assignment 1-basics 顺序，可调整）

### 0. 工程基础
- [ ] uv 管理依赖（torch、numpy、pytest）
- [ ] 项目骨架：`model/`、`tokenizer/`、`optimizer/`、`data/`、`train/`、`tests/`

### 1. BPE Tokenizer
- 核心思路：字节级 BPE —— 先把文本当原始字节流，再迭代合并出现频率最高的相邻 pair，
  训练出 vocabulary；编码/解码均基于 merge 优先级。
- 待决策点：特殊 token 处理、预分词（pre-tokenization pattern）选择、并行化 merge 统计。

### 2. Transformer 语言模型（从零实现）
- 核心思路：pre-norm RMSNorm + Rotary Position Embedding (RoPE) + SwiGLU FFN + 多头自注意力（causal mask）。
  这是 LLaMA 系架构，也是 CS336 指定的实现目标。
- 待决策点：权重初始化策略（截断正态 × 缩放系数）、RoPE 应用在 query/key 的方式。

### 3. 手写 AdamW
- 核心思路：Adam + 解耦 weight decay（decay 项不经过一阶/二阶矩，直接作用在参数上），
  配合 gradient clipping、lr warmup + cosine decay 调度。
- 待决策点：bias correction 的处理、eps 放在哪里（² 内还是外）。

### 4. 数据管线
- 核心思路：语料 tokenize 后存成一个大 memmap 数组，训练时随机采样固定长度窗口的
  (input, target) 对 —— target 就是 input 右移一位，无需拉取整篇文档。
- 待决策点：采样方式（随机 vs 顺序）、batch 内序列是否跨文档。

### 5. 训练循环（Trainer）
- 核心思路：标准 loop —— 采样 batch → 前向 → cross-entropy loss → 反向 → AdamW step；
  记录 loss / lr / grad-norm 到日志，支持 checkpoint 保存与恢复。
- 待决策点：混合精度（bf16 autocast）是否引入、checkpoint 格式。

### 6. 训练实验与验证
- 在 TinyStories 或 OpenWebText 子集上跑通小模型（如 4 层 / d=256），
  验证 loss 正常下降、生成文本连贯。
- 对照实验（可选）：不同 lr / warmup / 架构选择的 ablation，对应讲义实验章节。

## 当前状态

- 阶段：**Step 0 尚未开始**
- 下一步：确认项目骨架与依赖，然后从 BPE Tokenizer 的核心思路讨论开始。
