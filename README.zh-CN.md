# SciScout — 科研文献 Deep Research 多智能体系统

[English →](./README.md)

面向科研文献调研的多智能体(multi-agent) Agent 系统:给定一个研究问题,自动拆解子任务、
并行检索多个学术数据源、做反思式多轮迭代检索、生成带引用溯源的调研报告,并支持针对
报告继续多轮追问。核心检索路径用 LoRA 微调过一个本地开源小模型作为可选的低成本替代方案。

## 系统架构

```
用户问题
   │
   ▼
PlannerAgent ── 拆解成 3-5 个可检索的子任务(结构化 JSON,Pydantic 校验)
   │
   ▼
并发检索(ThreadPoolExecutor, 上限3并发)
   ├── ArxivSearchTool        ─┐
   └── SemanticScholarTool    ─┴─→ 按 URL + 标准化标题跨数据源去重
   │
   ▼
ReflectionAgent ── 判断当前检索结果是否覆盖问题各方面,不够则生成补充子任务,回到并发检索
   │ (最多 max_rounds 轮)
   ▼
ReportAgent ── 依据论文标题+摘要生成结构化报告,每个论点标注 [编号] 引用
   │
   ▼
CitationVerifier ── 独立 LLM 调用,逐条核实引用是否真的被摘要支撑(temperature=0)
   │
   ▼
FastAPI WebSocket 长连接,全过程进度事件实时推送给前端
   │
   ▼
FollowUpAgent ── 报告生成后连接不关闭,支持针对报告继续多轮追问(短期记忆,滑动窗口截断)
```

RAG 子系统(`app/tools/rag_tool.py`)已经实现了 dense(embedding) + BM25 的混合检索,
用 RRF(Reciprocal Rank Fusion)融合排序,但**尚未接入**上面报告生成的主链路(目前报告/追问
只用检索阶段拿到的标题+摘要,不查论文全文)——这是有意识的取舍,详见下方"已知限制"。

## 已实现的核心能力

- **多智能体循环编排**:Planner 拆解 → 并发检索 → Reflection 判断是否需要补充检索
  (由 LLM 在运行时动态决定要不要再来一轮,不是写死的固定流程,这是它区别于普通 pipeline 的地方)
- **多数据源并行检索**:arXiv + Semantic Scholar,`ThreadPoolExecutor` 并发查询,
  跨数据源按标准化标题去重(不同数据源给同一篇论文的 URL 不同,纯 URL 去重会漏)
- **RAG 混合检索**:dense(text-embedding-3-small)+ BM25 稀疏检索,RRF 融合排序,
  独立验证过能把关键词密集但语义没那么"漂亮"的段落从纯 dense 检索的第3名提升到第1名
- **带溯源的报告生成 + 引用校验**:每个论点标 `[编号]`,独立的 `CitationVerifier`
  用 `temperature=0` 二次核实引用是否真的被对应论文的摘要支撑
- **多轮追问(短期记忆)**:WebSocket 长连接在报告生成后不关闭,支持继续追问;
  对话历史拼进 prompt 实现短期记忆,`MAX_HISTORY_TURNS=6` 滑动窗口截断防止
  请求成本随对话轮数二次增长
- **LoRA 微调 + 量化评测**:用 GPT-4o-mini 的历史调用结果蒸馏训练数据,在
  `Qwen2.5-1.5B-Instruct` 上做 LoRA 微调(rank=16),替代 Planner 的任务分解环节;
  在 24 条留出验证集上跑了自动化评测(embedding 余弦相似度打分 + JSON 格式合法率 +
  风格失配检测),零 API 成本推理
- **实时进度推送**:FastAPI WebSocket,后端把拆解/检索/反思/生成/校验每一步进展
  实时推给前端,前端渲染成时间线动画

## LoRA 微调评测结果

在 24 条 held-out 验证集上,本地 LoRA 模型 vs. 线上 GPT-4o-mini(teacher):

| 指标 | 结果 |
|---|---|
| JSON 格式合法率 | 100% |
| 子任务数量落在合理范围的比例 | 100% |
| 平均语义对齐分数(embedding cosine,全部样本) | 0.833 |
| 平均语义对齐分数(排除风格失配样本后) | 0.854 |
| 平均推理延迟 | ~11.7s(本地 RTX 5080 Laptop GPU) |
| 推理阶段 API 成本 | $0(完全本地) |

详细方法见 `backend/scripts/22_eval_benchmark.py`。

## 技术栈

**后端**:Python, FastAPI + WebSocket, Pydantic(结构化输出校验), OpenAI SDK
(可切换 DashScope/DeepSeek 兼容接口), Qdrant(向量库,本地持久化模式), rank-bm25,
transformers + peft + trl(LoRA 训练), ThreadPoolExecutor(并发检索)

**前端**:React + Vite, 原生 WebSocket, marked(markdown 渲染)

## 项目结构

```
backend/
  app/
    agents/       # PlannerAgent, ReflectionAgent, ReportAgent, CitationVerifier, FollowUpAgent
    tools/        # ArxivSearchTool, SemanticScholarTool, RagTool(混合检索)
    core/         # LLMClient/LocalModelClient(统一 ChatClient 接口)、配置
    api/          # WebSocket 路由
    orchestrator.py   # 检索-反思循环编排
  scripts/        # 00-24: 从"跑通一个概念"的小脚本,到训练/评测脚本,按顺序编号
  data/           # 蒸馏训练数据、benchmark 结果
frontend/
  src/App.jsx     # 单页应用:发起调研 + 实时进度时间线 + 报告展示 + 多轮追问
```

`scripts/` 目录里每个脚本对应项目演进过程中的一个具体概念验证,编号大致按时间顺序,
可以按顺序读下来大致还原整个项目从"能跑通一次 LLM 调用"到"完整多智能体系统"的搭建过程。

## 快速开始

**后端**

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # 然后把 OPENAI_API_KEY 换成你自己的
uvicorn app.main:app --reload
```

**前端**

```bash
cd frontend
npm install
npm run dev
```

浏览器打开前端显示的地址,输入研究问题即可。

**(可选)本地 LoRA 微调模型**

```bash
cd backend
pip install -r requirements-train.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128  # 按你的 CUDA 版本调整
python -m scripts.18_format_training_data
python -m scripts.19_train_lora
python -m scripts.20_eval_lora
```

训练产物(`backend/models/`)体积较大(单个 checkpoint 超过 GitHub 单文件 100MB 限制),
不在版本库里,需要本地跑一遍上面的脚本重新生成。

## 已知限制

- **RAG 全文混合检索尚未接入报告生成主链路**:报告/追问目前只依据检索阶段拿到的
  标题+摘要,`rag_tool.py` 里已经验证可用的 dense+BM25 混合检索还没有接进
  `ReportAgent`/`FollowUpAgent`——有意保持两边信息颗粒度一致,避免用两套不一致的证据源
- **Semantic Scholar 匿名 API 受共享限流影响**:未注册 API key 时走匿名共享配额,
  容易被 429 限流(已加指数退避重试),注册免费 key 后有专属配额会更稳定
- **多轮追问的历史窗口固定为最近 6 轮**:超出窗口的更早历史会被直接丢弃,不做摘要压缩
- **短期记忆不跨会话**:对话历史只存在于当前 WebSocket 连接的生命周期内,刷新页面/
  断开连接后不保留
