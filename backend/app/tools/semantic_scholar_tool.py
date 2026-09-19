"""Semantic Scholar 检索工具:第二个论文数据源,跟 ArxivSearchTool 接口完全一致
(同样是 search(subtask) -> list[PaperHit]),这样编排层可以把两个工具当成
"同一种东西"并发调用,不用为了多一个数据源专门改上层逻辑——这是"面向接口"
的又一个实例,跟 ChatClient Protocol 是同一个设计思路。

为什么要加这个:实测发现 arXiv 的关键词搜索对一些细分/交叉领域主题,换着法子
改写query,搜出来的还是同一小撮论文(甚至部分不相关)——这是它自己检索能力的
天花板,不是我们代码的问题。多接一个独立的数据源,是缓解"单一数据源覆盖不足"
最直接的办法:两个源各自漏掉的,另一个源可能刚好覆盖到。
"""
from __future__ import annotations

import logging
import time

import httpx

from app.agents.schemas import PaperHit, SubTask
from app.core.config import get_settings

logger = logging.getLogger("scires.tools.semantic_scholar")


class SemanticScholarTool:
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,authors,abstract,url,publicationDate"
    MAX_RETRIES = 2
    RETRY_BACKOFF_SEC = 3.0  # 遇到429,等这么久再重试;每次重试翻倍(指数退避)

    def __init__(self, max_results: int = 5, timeout: float = 10.0) -> None:
        self.max_results = max_results
        self.timeout = timeout
        self._api_key = get_settings().semantic_scholar_api_key

    def search(self, subtask: SubTask) -> list[PaperHit]:
        """跟 ArxivSearchTool.search() 一样,失败时降级返回空列表而不是抛异常——
        Semantic Scholar 的匿名配额是"所有没用API key的用户共享一个池子",
        比 arXiv 更容易被限流(429),而且限流的是"全世界这一刻所有匿名请求"而不只是
        我们自己发得快不快,所以单纯"发得慢一点"不一定能完全避免429。
        应对方式两层:1) 429时做几次指数退避重试(等得越来越久再试),
        给拥堵的共享池子一点缓冲时间;2) 如果配置了免费申请的API key就带上,
        走的是"专属配额"而不是共享池子,更稳定——重试解决的是"运气不好偶尔被限",
        API key解决的是"从根上换一个不那么拥堵的配额池"。
        """
        params = {"query": subtask.query, "limit": self.max_results, "fields": self.FIELDS}
        headers = {"x-api-key": self._api_key} if self._api_key else {}

        delay = self.RETRY_BACKOFF_SEC
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = httpx.get(self.BASE_URL, params=params, headers=headers, timeout=self.timeout)
                if response.status_code == 429 and attempt < self.MAX_RETRIES:
                    logger.warning(
                        "Semantic Scholar 429限流 (subtask=%s), %.1f秒后重试 (第%d次)",
                        subtask.id, delay, attempt + 1,
                    )
                    time.sleep(delay)
                    delay *= 2  # 指数退避:3s -> 6s -> ...
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.MAX_RETRIES:
                    logger.warning("Semantic Scholar 检索失败 (subtask=%s): %s", subtask.id, exc)
                    return []
                time.sleep(delay)
                delay *= 2

        hits: list[PaperHit] = []
        for item in data.get("data", []):
            abstract = item.get("abstract")
            if not abstract:
                # 摘要为空的论文对后续RAG/报告生成没有用,直接跳过而不是硬塞进结果里
                continue
            hits.append(
                PaperHit(
                    subtask_id=subtask.id,
                    title=(item.get("title") or "").strip(),
                    authors=[a.get("name", "") for a in item.get("authors", []) or []],
                    summary=abstract.strip()[:500],
                    url=item.get("url") or "",
                    published=item.get("publicationDate") or "",
                )
            )
        return hits
