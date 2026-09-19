"""Report Generation Agent:把收集到的论文,整理成一份带引用编号的调研报告。

第一版先只用检索阶段已经拿到的元数据(标题+摘要),不去下载全文——
先把"提问 -> 报告"这条完整链路跑通,以后如果发现摘要粒度不够细,
再考虑针对关键论文调 RagTool 读全文深挖。
"""
from __future__ import annotations

import json
import logging

from app.agents.schemas import Citation, PaperHit, Report, ResearchPlan
from app.core.llm_client import LLMClient

logger = logging.getLogger("scires.report")

SYSTEM_PROMPT = """你是一个科研文献调研助手中的"报告生成"模块。
给定用户的原始研究问题、以及已经检索到的论文列表(每篇论文前面标了编号),
你需要写一份结构化的调研报告来回答这个问题。

要求:
1. 只能依据下面提供的论文列表来写,不要使用你自己的知识编造论文之外的信息
2. 报告要有清晰的结构(比如分几个小节,分别对应问题的不同方面)
3. 每一个论点后面,必须用 [编号] 标注它是依据哪(几)篇论文得出的,编号要对应论文列表里的编号
4. 如果论文列表里的信息明显不足以覆盖某个方面,在报告里如实说明"现有资料未充分覆盖此方面",不要编造

用严格的 JSON 格式输出,不要任何多余文字,格式如下:
{
  "content": "markdown格式的报告正文,论点后面带 [1][2] 这样的引用编号",
  "used_refs": [1, 2, 3]
}
used_refs 里列出报告正文里实际引用到的编号(哪些论文真正被用上了,没用到的论文不用列)。
"""


class ReportAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def generate(self, plan: ResearchPlan, papers: list[PaperHit]) -> Report:
        """把 papers 编号后喂给 LLM,让它依据编号写报告并标注引用;
        再把 LLM 回报的编号,反查回真正的论文标题/链接,组装成 Citation 列表。
        """
        papers_block, numbered_papers = self._format_papers(papers)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"原始研究问题:{plan.original_question}\n\n论文列表:\n{papers_block}",
            },
        ]
        raw = self.llm.chat(messages, temperature=0.3, response_format_json=True)
        try:
            data = json.loads(raw)
            content = data["content"]
            used_refs = data.get("used_refs", [])
        except Exception as exc:  # noqa: BLE001 - 解析失败时打出原始内容方便调试
            logger.error("Report 输出解析失败: %s\n原始内容: %s", exc, raw)
            raise ValueError(f"Report 返回内容不是合法的 JSON: {exc}") from exc

        citations = [
            Citation(
                ref_id=ref_id,
                title=numbered_papers[ref_id].title,
                url=numbered_papers[ref_id].url,
            )
            for ref_id in used_refs
            if ref_id in numbered_papers
        ]

        return Report(question=plan.original_question, content=content, citations=citations)

    @staticmethod
    def _format_papers(papers: list[PaperHit]) -> tuple[str, dict[int, PaperHit]]:
        """给每篇论文编号(从1开始),同时返回"编号 -> PaperHit"的映射,
        方便后面把 LLM 回报的编号反查回具体是哪篇论文。
        """
        numbered_papers: dict[int, PaperHit] = {}
        lines = []
        for i, p in enumerate(papers, start=1):
            numbered_papers[i] = p
            lines.append(f"[{i}] {p.title}: {p.summary}")
        return "\n".join(lines), numbered_papers
