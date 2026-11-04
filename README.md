> 当前运行流程与配置请以 [DESIGN.md](DESIGN.md) 为准。作者画像、跨论文记忆、文献推荐及旧 `eval.py` 已移除，下方相关旧版说明不再适用。

# Plagiarism Check Agent

> 基于 LLM Agent 架构的学术论文查重与降重辅助系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**不同于传统查重工具**，本系统不接入第三方查重 API（如知网、Turnitin），而是在用户提供的对照文献范围内，通过向量语义检索 + LLM 精判两阶段定位相似段落，溯源到具体论文，并给出可操作的改写建议。

## 核心特性

| 特性 | 说明 |
|------|------|
| **可控语料库查重** | 在用户上传的对照文献范围内查重，结果可解释、可复现 |
| **语义级相似度** | 基于 BGE-large-zh 向量检索，检测近义改写而不仅是逐字重复 |
| **LLM 精判** | 向量检索提供候选，LLM 判断重复类型并生成改写建议 |
| **四级重复分类** | 逐字重复 / 近义改写 / 观点重复 / 无问题 |
| **溯源标注** | 每个疑似重复段落标注来源论文、相似度分数 |
| **Agent 架构** | 完整的 LLM Agent 循环 + Function Call + 循环守护 + 轨迹记录 |
| **记忆系统** | 跨论文热点检测、领域常识注入、人工纠错积累 |
| **用户画像** | 查后分析：文献集中度、章节诊断、写作风格、引用规范、进步追踪 |
| **训练数据导出** | 轨迹可导出为 SFT / Tool-Supervision / DPO 偏好对格式 |

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                               │
│        check  │  inspect  │  export                             │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Layer                               │
│              PlagiarismCheckOrchestrator                        │
│   Phase 1: 解析文档    Phase 2: 文本分块 + 向量索引                 │
│   Phase 3: LLM 分析    Phase 4: 生成报告                          │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Layer                                 │
│  ┌─────────────┐  ┌──────────┐  ┌────────────┐  ┌───────────┐  │
│  │ AgentRunner │  │LLMClient │  │ToolRouter  │  │SkillLoader│  │
│  │ 主循环       │  │OpenAI API│  │ 工具分发    │  │ SKILL.md  │  │
│  │ 轨迹记录     │  │兼容接口   │  │ 结果截断    │  │ 系统提示   │  │
│  │ 循环守护     │  │          │  │            │  │            │  │
│  └─────────────┘  └──────────┘  └────────────┘  └───────────┘  │
│  ┌──────────┐                                                   │
│  │LoopGuard │  循环异常检测（重复调用/连续错误/迭代预警）        │
│  └──────────┘                                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Tools Layer                                 │
│  ┌──────────────────┐    ┌──────────────┐  ┌───────────────┐   │
│  │  DocumentTools   │    │ ReportTools  │  │ MemoryTools   │   │
│  │  - parse_pdf     │    │ - generate   │  │ - memory_get  │   │
│  │  - chunk_document│    │   _report    │  │   _summary    │   │
│  │  - search_similar│    └──────────────┘  └───────────────┘   │
│  │  - build_index   │                                           │
│  │  - get_chunk_ctx │    ┌──────────────────────────────────┐   │
│  └──────────────────┘    │  底层依赖                         │   │
│                          │  PyMuPDF  │  sentence-transformers  │   │
│                          │  ChromaDB │  BGE-large-zh-v1.5     │   │
│                          └──────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Skills Layer (纯 Prompt)                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐   │
│  │ plagiarism-check │  │ plagiarism-check │  │ plagiarism-  │   │
│  │ (主技能/编排)    │  │ -analyze (分析)  │  │ check-report │   │
│  └─────────────────┘  └──────────────────┘  └──────────────┘   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Memory Layer                                  │
│  ┌──────────────┐    ┌──────────────────────────────────┐       │
│  │  store.py    │    │  profile.py                      │       │
│  │  持久化存储   │    │  分析引擎                         │       │
│  │  - chunk     │    │  - 来源集中度分析                 │       │
│  │    hotspot   │    │  - 章节诊断                       │       │
│  │  - common    │    │  - 写作模式分析                   │       │
│  │    knowledge │    │  - 引用规范检查                   │       │
│  │  - false     │    │  - 进步追踪                       │       │
│  │    positive  │    │  - 文献推荐                       │       │
│  └──────────────┘    └──────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Agent 执行流程

```
用户输入: "帮我查重 thesis.pdf，对照文献在 ./refs/ 中"
    ↓
┌──────── Phase 1: Parse ────────────────────────┐
│ Tool: parse_pdf(thesis.pdf)                    │
│ Tool: parse_pdf(refs/paper1.pdf)               │
│ Tool: parse_pdf(refs/paper2.pdf)               │
│ ...                                            │
└────────────────────────────────────────────────┘
    ↓
┌──────── Phase 2: Index ────────────────────────┐
│ Tool: chunk_document(doc_id) × N               │
│ Tool: build_corpus_index()                     │
│   → ChromaDB + BGE-large-zh 向量库             │
│   → 首次构建后持久化，下次跳过重建             │
└────────────────────────────────────────────────┘
    ↓
┌──────── Phase 3: Analyze (Agent Loop) ─────────┐
│ for each thesis paragraph:                     │
│   Tool: search_similar_chunks(text, top_k=5)   │
│   if similarity >= threshold:                  │
│     Tool: get_chunk_context(chunk_id)          │
│     LLM 精判: verbatim/near_paraphrase/idea    │
│     LLM 生成: 改写建议                          │
│                                                 │
│   LoopGuard 实时检测：                          │
│     - 重复调用同一工具 → 注入纠正提示           │
│     - 连续3次错误 → 强制终止                    │
│     - 接近迭代上限 → 倒计时提醒                 │
└────────────────────────────────────────────────┘
    ↓
┌──────── Phase 4: Report ───────────────────────┐
│ Tool: generate_report(findings, output_dir)    │
│   → HTML 报告 (标红 + 来源 + 修改建议)          │
│   → JSON 结构化数据                             │
└────────────────────────────────────────────────┘
    ↓
┌──────── Post-check: Profile ───────────────────┐
│ - 来源集中度分析                                │
│ - 章节级诊断                                    │
│ - 写作风格分析                                  │
│ - 引用规范检查                                  │
│ - 进步追踪 (与历史检查对比)                     │
│ - 文献推荐 (Semantic Scholar)                   │
└────────────────────────────────────────────────┘
```

## 快速开始

### 1. 环境准备

```bash
cd plagiarism_check_agent

# 创建虚拟环境
conda create -n plagiarism_check python=3.10 -y
conda activate plagiarism_check

# 安装依赖
pip install -r requirements.txt
```

首次运行时，`sentence-transformers` 会自动下载 `BAAI/bge-large-zh-v1.5` 模型（约 1.3GB）。
如果 HuggingFace 访问较慢，可设置镜像：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

检查安装：

```bash
python3 -c "import fitz,chromadb,openai; from sentence_transformers import SentenceTransformer; print('OK')"
```

### 2. 配置 LLM API Key

```bash
# 方式一：环境变量
export OPENAI_API_KEY="your-api-key"

# 方式二：命令行参数
python cli.py check thesis.pdf --refs ./refs/ --api-key "your-key"
```

### 3. 准备文件

```
your_project/
├── thesis.pdf          # 你的论文
└── refs/               # 对照文献目录
    ├── paper1.pdf
    ├── paper2.pdf
    └── ...
```

### 4. 运行查重

```bash
# 基本用法
python cli.py check thesis.pdf --refs ./refs/

# 指定论文标题
python cli.py check thesis.pdf --refs ./refs/ --title "基于深度学习的文本分类研究"

# 指定模型
python cli.py check thesis.pdf --refs ./refs/ --model gpt-4o

# 试运行（不写入报告，写工具调用被拦截）
python cli.py check thesis.pdf --refs ./refs/ --dry-run
```

### 5. 查看结果

查重完成后会生成以下文件：

```
plagiarism_check_agent/
├── output/                          ← 查重报告
│   ├── plagiarism_report_*.html     ← 可视化 HTML 报告
│   └── plagiarism_data_*.json       ← 结构化数据
├── trajectories/                    ← 执行轨迹
│   └── trace_xxxxxxxxxxxx.jsonl     ← Agent 每一步的完整记录
└── vector_db/                       ← 向量检索库
    └── chroma.sqlite3 + 索引文件    ← 同一批文献下次跳过重建
└── memory_data/                     ← 记忆数据
    ├── chunk_hotspots.jsonl         ← 跨论文热点片段
    ├── common_knowledge.jsonl       ← 领域常识
    ├── false_positives.jsonl        ← 人工纠错
    └── author_profiles.jsonl        ← 用户画像
```

```bash
# 查看执行轨迹
python cli.py inspect ./trajectories/trace_abc123.jsonl

# 导出训练数据
python cli.py export --input ./trajectories --output ./exports --format sft tool-supervision
```

## 大模型推荐

| 模型 | 推荐度 | 中文能力 | 性价比 | API 获取 |
|------|--------|----------|--------|----------|
| **DeepSeek-chat** (默认) | ⭐⭐⭐⭐⭐ | 优秀 | 极高 | [platform.deepseek.com](https://platform.deepseek.com) |
| **Claude Sonnet 4.6** | ⭐⭐⭐⭐⭐ | 优秀 | 中 | [console.anthropic.com](https://console.anthropic.com) |
| **Qwen-Plus** | ⭐⭐⭐⭐ | 极佳 | 高 | [dashscope.aliyun.com](https://dashscope.aliyun.com) |
| **GPT-4o** | ⭐⭐⭐⭐ | 良好 | 中 | [platform.openai.com](https://platform.openai.com) |

### 配置示例

```bash
# DeepSeek (默认)
export OPENAI_API_KEY="sk-xxx"      # 需要在deepseek API平台获取API KEY
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export OPENAI_MODEL="deepseek-chat"

# Claude Sonnet 4.6 (via OpenAI-compatible proxy)
export OPENAI_API_KEY="sk-ant-xxx"
export OPENAI_BASE_URL="https://api.anthropic.com/v1"
export OPENAI_MODEL="claude-sonnet-4-6"

# 阿里 Qwen
export OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL="qwen-plus"

# GPT-4o
export OPENAI_API_KEY="sk-xxx"
export OPENAI_MODEL="gpt-4o"
```

## CLI 命令参考

```bash
# 查重
python cli.py check <论文.pdf> --refs <对照文献目录> [选项]
  选项:
    --title TEXT          论文标题 (默认使用文件名)
    --model TEXT          LLM 模型名
    --api-key TEXT        API Key
    --base-url TEXT       API Base URL
    --trajectories-dir   轨迹输出目录 (默认: ./trajectories)
    --output-dir         报告输出目录 (默认: ./output)
    --dry-run            试运行，不写入文件

# 查看轨迹
python cli.py inspect <trace_xxx.jsonl>

# 导出训练数据
python cli.py export --input ./trajectories --output ./exports
  选项:
    --format [sft] [tool-supervision]  导出格式
```

## 高级配置

### 调整查重敏感度

```bash
export SIMILARITY_THRESHOLD=0.80   # 提高阈值，更严格 (默认 0.75)
export TOP_K_RESULTS=3             # 每个段落最多返回 3 个候选 (默认 5)
export CHUNK_SIZE=768              # 增大分块大小 (默认 512)
export CHUNK_OVERLAP=128           # 增大重叠 (默认 64)
```

### 迭代与Token控制

```bash
export MAX_ITERATIONS=50           # Agent 最大迭代轮数 (默认 50)
export MAX_TOKENS=16000            # 单次 LLM 响应的最大 token (默认 16000)
export MAX_TOOL_RESULT_TOKENS=6000 # 工具返回结果的最大 token 预算 (默认 6000)
export THINKING_BUDGET=8000        # LLM 思考链最大 token (默认 8000)
```

### 记忆数据存储

```bash
# 默认: ./memory_data/ (项目目录下)
export MEMORY_DATA_DIR="./my_memory_data"
```

### 使用本地 Embedding 模型

```bash
# 默认使用 BGE-large-zh-v1.5 (中文最优)
export EMBEDDING_MODEL="BAAI/bge-large-zh-v1.5"

# 英文论文可使用
export EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"

# 轻量化选择 (更快但精度略低)
export EMBEDDING_MODEL="BAAI/bge-base-zh-v1.5"

# 使用 GPU 加速 embedding
export EMBEDDING_DEVICE="cuda"
```

## 项目结构

```
plagiarism_check_agent/
├── cli.py                  # CLI 入口 (check / inspect / export)
├── config.py               # 配置 (环境变量驱动)
├── eval.py                 # 评估脚本 (Embedding vs LLM vs Hybrid)
├── requirements.txt        # 依赖
│
├── agent/                  # Agent 框架
│   ├── runner.py           # Agent 主循环 (ReAct 范式)
│   ├── llm_client.py       # LLM 客户端 (OpenAI 兼容协议)
│   ├── tool_router.py      # 工具路由 + Token预算截断
│   ├── skill_loader.py     # SKILL.md 加载器 + 子技能自动发现
│   └── loop_guard.py       # 循环守护 (重复调用/连续错误检测)
│
├── tools/                  # 工具层
│   ├── registry.py         # 工具注册表 (解耦三层架构)
│   ├── document_tools.py   # PDF解析、文本分块、向量检索
│   ├── report_tools.py     # HTML 报告生成
│   └── memory_tools.py     # 记忆系统查询接口
│
├── skills/                 # 技能层 (纯 Prompt，无 Python 代码)
│   ├── plagiarism-check/           # 主技能 (编排 + 核心原则)
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── detection-taxonomy.md
│   ├── plagiarism-check-analyze/   # 分析子技能 (分级标准 + 精判格式)
│   │   └── SKILL.md
│   └── plagiarism-check-report/    # 报告子技能 (报告规范 + 改写指南)
│       └── SKILL.md
│
├── pipeline/               # 流程编排
│   └── orchestrator.py     # 四阶段查重流程 (Phase 1-2 代码执行, 3-4 Agent 驱动)
│
├── trajectory/             # 轨迹记录
│   ├── schemas.py          # 数据模型 (LLMResponse, AgentResult 等)
│   ├── logger.py           # JSONL 记录器 (上下文管理器)
│   └── exporter.py         # 训练数据导出 (SFT / Tool-Supervision / DPO)
│
├── memory/                 # 记忆系统
│   ├── store.py            # 持久化存储 (JSONL, 4种记录类型)
│   └── profile.py          # 分析引擎 (5维分析 + 文献推荐)
│
├── tests/                  # 测试
│   ├── test_tools.py       # 工具单元测试
│   └── eval_dataset.json   # 评估数据集 (27条标注样本)
│
├── output/                 # 查重报告输出 (自动生成)
├── trajectories/           # 执行轨迹记录 (自动生成)
├── vector_db/              # 向量检索库 (自动生成，支持复用)
└── memory_data/            # 记忆系统数据 (自动生成，4种类型分文件存储)
```

## 毕业设计建议

如果你的毕业设计基于此项目，以下是一些可展开的方向：

### 论文可写的技术点

1. **系统设计章节**
   - LLM Agent 架构的设计与实现（Agent Loop、Function Call、Tool Use）
   - Skills 作为可插拔 Prompt 模块的设计模式
   - 向量检索 + LLM 精判的两阶段混合架构
   - Memory 系统：从无状态 Agent 到有学习能力的系统
   - LoopGuard 循环守护机制

2. **实验章节**
   - 在标注数据集上对比不同方法的 F1 分数
   - 纯 Embedding vs 纯 LLM vs 混合方案的效果对比
   - 不同 LLM（DeepSeek vs GPT-4o vs Qwen）的查重准确率对比
   - 不同 Embedding 模型（BGE vs M3E vs text2vec）的检索召回率对比
   - 分块大小和重叠参数对效果的影响

3. **创新点**
   - 可控语料库查重的概念（vs 黑盒查重系统）
   - LLM 精判的四级分类体系
   - 查重 + 改写建议的一体化设计
   - Agent 执行轨迹的可观测性与训练数据导出
   - 跨论文热点检测的记忆系统

## 免责声明

本系统仅在用户提供的对照文献范围内进行文本相似度检测。它不是一个"全网查重"工具，也不声称能替代知网、Turnitin 等专业查重系统。改写建议仅供学术写作参考，用户应对自己的学术诚信负责。

## License

MIT License

---

**项目作者**: Songlei
