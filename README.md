## 基于金字塔流（Pyramidal Flow）的 NLP 伪干扰生成系统（Python + 交互界面）

这个项目实现一个“金字塔流式（由粗到细、多粒度）”的文本伪干扰生成器，并提供带界面的多任务工作台：

- **伪干扰生成**：句/短语/子词/字符四级扰动，支持类型与强度控制，并输出可追踪的操作轨迹（trace）
- **增强训练**：用伪干扰对训练数据做增广，训练一个轻量分类器（TF‑IDF + Logistic Regression）
- **鲁棒评测**：对比 clean vs perturbed 的指标变化，输出脆弱点统计
- **对抗攻防（轻量）**：针对当前分类器做“重要词破坏”的贪心攻击（在语义约束与预算内）
- **数据清洗测试**：检测并修复常见脏文本（重复字符、全角半角、异常空白、标点乱序等）
- **多任务演示**：实体抽取（规则/启发式）、文本匹配、检索、问答（检索式）、生成（模板式）

> 说明：为了保证“全 Python、可在普通环境跑起来”，系统默认提供轻量可运行的基线实现；如果你希望接入更强的 Transformer 模型，可在此基础上扩展（README 不强制安装 torch/transformers）。

### 运行

1) 安装依赖

```bash
python -m pip install -r requirements.txt
```

2) 启动界面

```bash
streamlit run app.py
```

### 数据格式（示例）

- **分类**：CSV，至少包含 `text,label` 两列。示例见 `data/sample_cls.csv`
- **检索/问答知识库**：每行一个文档（txt/CSV 都可），界面内可直接粘贴或上传

### 项目结构

- `app.py`：Streamlit 交互界面入口
- `nlp_pyramidal_flow/`：核心库
  - `pyramidal_flow.py`：金字塔流式伪干扰 orchestrator + trace
  - `perturbations/`：各尺度扰动算子（sentence/word/subword/char）
  - `tasks/`：分类、抽取、匹配、检索、问答、生成的轻量实现
  - `cleaning.py`：脏文本检测与修复
- `data/`：示例数据


