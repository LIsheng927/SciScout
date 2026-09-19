"""Planner Agent:把一个开放式研究问题拆解为若干个可直接用于检索的子任务。

这是整个项目里"算法含量"最高的一环——后续 LoRA 微调就是专门针对这一步训练一个
小模型来替代这里的 prompt-only 实现,所以这里的 prompt / 输出 schema 设计要尽量
稳定,方便之后把 `PlannerAgent.plan()` 的实现换成本地微调模型而不改动调用方代码。
"""
from __future__ import annotations

import json
import logging

from app.agents.schemas import ResearchPlan, SubTask
from app.core.llm_client import ChatClient, LLMClient

logger = logging.getLogger("scires.planner")

SYSTEM_PROMPT = """你是一个科研文献调研助手中的"任务规划"模块。
给定一个研究问题,你需要把它拆解为 3-5 个具体的、可以直接拿去学术搜索引擎检索的子任务。

要求:
1. 每个子任务的 query 要具体、可检索,不要停留在原问题的重复改写
2. 子任务之间应尽量覆盖问题的不同侧面(如:方法本身、对比方法、应用场景、最新进展)
3. 用严格的 JSON 格式输出,不要任何多余文字,格式如下:
{
  "subtasks": [
    {"id": "t1", "query": "...", "rationale": "..."},
    {"id": "t2", "query": "...", "rationale": "..."}
  ]
}
"""


class PlannerAgent:
    def __init__(self, llm: ChatClient | None = None) -> None:
        # 类型标注用 ChatClient(接口),不是 LLMClient(具体实现)。
        # 默认还是创建线上的 LLMClient,但调用方也可以传一个 LocalModelClient 进来,
        # PlannerAgent 内部代码完全不用改——这是"依赖接口而不是依赖具体实现"。
        self.llm = llm or LLMClient()

    def plan(self, question: str) -> ResearchPlan:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"研究问题:{question}"},
        ]
        raw = self.llm.chat(messages, temperature=0.2, response_format_json=True)
        try:
            data = json.loads(raw)
            subtasks = [SubTask(**item) for item in data["subtasks"]]
        except Exception as exc:  # noqa: BLE001
            logger.error("Planner 输出解析失败: %s\n原始内容: %s", exc, raw)
            raise ValueError(f"Planner 返回内容不是合法的子任务 JSON: {exc}") from exc

        if not subtasks:
            raise ValueError("Planner 未产出任何子任务")

        return ResearchPlan(original_question=question, subtasks=subtasks)
