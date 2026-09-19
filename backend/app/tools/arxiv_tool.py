"""arXiv 检索工具:免费、无需 API key,是 Retriever Agent 的核心数据源。

用官方推荐的 `arxiv` python 包(底层封装 export.arxiv.org 的 Atom API)。
"""
from __future__ import annotations

import logging

import arxiv

from app.agents.schemas import PaperHit, SubTask

logger = logging.getLogger("scires.tools.arxiv")


class ArxivSearchTool:
    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results
        self._client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=2)

    def search(self, subtask: SubTask) -> list[PaperHit]:
        """针对一个子任务的 query 去 arXiv 检索论文,失败时降级返回空列表而不是抛异常,
        保证 Planner 分解出的其他子任务不受影响(多智能体并行检索时的容错设计)。
        """
        search = arxiv.Search(
            query=subtask.query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        hits: list[PaperHit] = []
        try:
            for result in self._client.results(search):
                hits.append(
                    PaperHit(
                        subtask_id=subtask.id,
                        title=result.title.strip().replace("\n", " "),
                        authors=[a.name for a in result.authors],
                        summary=result.summary.strip().replace("\n", " ")[:500],
                        url=result.entry_id,
                        published=result.published.strftime("%Y-%m-%d") if result.published else "",
                    )
                )
        except Exception as exc:  # noqa: BLE001 - 网络/限流类问题降级处理,不让整个流程崩掉
            logger.warning("arXiv 检索失败 (subtask=%s): %s", subtask.id, exc)
            return []

        return hits
