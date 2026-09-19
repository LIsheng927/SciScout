"""Reflection Agent:检查已经收集到的资料,判断是否足够回答原始问题;
不够的话就生成新的子任务,驱动检索再跑一轮——这是让整个系统从"一次性管道"
变成"能自我评估、自己决定要不要再查一轮"的 agentic 循环的关键一环。

写法上和 PlannerAgent 几乎是同一个模板(系统提示词 + LLMClient.chat() +
json.loads() 解析成 pydantic 结构体),这不是偷懒,而是因为它们本质上是
同一类任务:"给 LLM 一段材料,让它按固定 schema 输出判断结果"。
"""
from __future__ import annotations

import json
import logging

from app.agents.schemas import PaperHit, ReflectionResult, ResearchPlan
from app.core.llm_client import LLMClient

logger = logging.getLogger("scires.reflection")

SYSTEM_PROMPT = """你是一个科研文献调研助手中的"反思评估"模块。
给定用户的原始研究问题、已经拆解出的调研子任务、以及目前已经检索到的论文列表,
你需要判断:这些论文加起来,是否已经足够全面地支撑对原始问题的回答。

判断标准:
1. 是否覆盖了问题涉及的主要方面(不要求穷尽,但明显的空白要指出来)
2. 检索到的论文数量是否过少、或者内容是否明显跑题
3. 如果不够,给出具体缺失的方面,并生成1-3个新的、可直接检索的子任务来补充

用严格的 JSON 格式输出,不要任何多余文字,格式如下:
{
  "is_sufficient": true 或 false,
  "missing_aspects": ["缺失的方面1", "..."],
  "new_subtasks": [
    {"id": "t_new1", "query": "...", "rationale": "..."}
  ]
}
如果已经足够,missing_aspects 和 new_subtasks 都给空列表。
"""


class ReflectionAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def reflect(self, plan: ResearchPlan, gathered_papers: list[PaperHit]) -> ReflectionResult:
        """评估当前已收集到的论文是否足够回答原始问题。

        参数里传入完整的 plan(而不只是原始问题),是因为"已拆解的子任务"本身
        也是有用的上下文——能让 LLM 知道"我们已经打算从哪些角度去查了",
        从而更准确地判断"还缺什么角度",而不是重新猜一遍问题该怎么拆。
        """
        papers_block = self._format_papers(gathered_papers)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"原始研究问题:{plan.original_question}\n\n"
                    f"已拆解的子任务:\n{plan.as_prompt_block()}\n\n"
                    f"已检索到的论文:\n{papers_block}"
                ),
            },
        ]
        raw = self.llm.chat(messages, temperature=0.2, response_format_json=True)
        try:
            data = json.loads(raw)
            result = ReflectionResult(**data)
        except Exception as exc:  # noqa: BLE001 - 解析失败时把原始内容打出来方便调试
            logger.error("Reflection 输出解析失败: %s\n原始内容: %s", exc, raw)
            raise ValueError(f"Reflection 返回内容不是合法的 JSON: {exc}") from exc

        return result

    @staticmethod
    def _format_papers(papers: list[PaperHit]) -> str:
        if not papers:
            return "(暂无检索到的论文)"
        lines = [f"- [{p.subtask_id}] {p.title}: {p.summary}" for p in papers]
        return "\n".join(lines)
