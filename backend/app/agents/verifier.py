"""Citation Verifier:检查 Report 正文里每处 [编号] 引用,是不是真的能在
对应论文的摘要里找到直接支撑——而不是"看起来有引用、实际上站不住脚"的幻觉。

设计上刻意让它是一个独立模块、单独一次 LLM 调用,而不是信任 ReportAgent
自己声称"我引用得没问题"——生成和校验分成两个角色、互相不信任对方,
这是防止"自己批改自己作业"最基本的做法。
"""
from __future__ import annotations

import json
import logging

from app.agents.schemas import PaperHit, Report, VerificationResult
from app.core.llm_client import LLMClient

logger = logging.getLogger("scires.verifier")

SYSTEM_PROMPT = """你是一个"引用校验"模块。给定一份调研报告(正文里带有 [1][2] 这样
的引用编号)和被引用的论文列表(编号对应论文标题+摘要),你需要逐条检查:
报告里每处引用某个编号的论点,是否真的能在对应论文的摘要内容里找到直接支撑。

判断标准要严格:
- 论文摘要里明确提到了论点相关的具体内容 -> 判定为支撑(supported = true)
- 论文摘要只是主题上跟论点沾点边、但没有直接提到论点说的具体内容,或者论文
  跟论点几乎完全不相关 -> 判定为不支撑(supported = false)
- 不要因为"报告写得像那么回事"就放宽标准,你的任务就是挑刺,尤其要警惕报告
  把不相关论文的内容强行关联到论点上的情况

用严格的 JSON 格式输出,不要任何多余文字,格式如下:
{
  "results": [
    {"ref_id": 1, "supported": true, "reason": "..."},
    {"ref_id": 2, "supported": false, "reason": "..."}
  ]
}
"""


class CitationVerifier:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def verify(self, report: Report, papers: list[PaperHit]) -> list[VerificationResult]:
        numbered_papers = {i: p for i, p in enumerate(papers, start=1)}
        papers_block = "\n".join(
            f"[{i}] {p.title}: {p.summary}" for i, p in numbered_papers.items()
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"报告正文:\n{report.content}\n\n"
                    f"被引用的论文列表:\n{papers_block}"
                ),
            },
        ]
        raw = self.llm.chat(messages, temperature=0.0, response_format_json=True)
        try:
            data = json.loads(raw)
            results = [VerificationResult(**item) for item in data["results"]]
        except Exception as exc:  # noqa: BLE001 - 解析失败时打出原始内容方便调试
            logger.error("Verifier 输出解析失败: %s\n原始内容: %s", exc, raw)
            raise ValueError(f"Verifier 返回内容不是合法的 JSON: {exc}") from exc

        return results
