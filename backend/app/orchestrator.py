"""编排层:把 PlannerAgent / ArxivSearchTool / ReflectionAgent 串成一个真正的
"多轮检索循环"——单独测试这三个模块只能证明"每一步自己是对的",
但只有把 Reflection 的判断结果真正喂回去驱动下一轮检索,才算是一个完整的、
能自己决定"要不要再查一轮"的 agent 循环,而不是一条跑一次就结束的流水线。
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from app.agents.planner import PlannerAgent
from app.agents.reflection import ReflectionAgent
from app.agents.schemas import PaperHit, ResearchPlan, SubTask
from app.tools.arxiv_tool import ArxivSearchTool
from app.tools.semantic_scholar_tool import SemanticScholarTool

logger = logging.getLogger("scires.orchestrator")

# 事件回调的类型:一个函数,接收一个"事件字典"(比如 {"type": "plan_created", ...}),
# 不需要有任何返回值。之所以用 dict 而不是自己定义一个 EventClass,是因为这些事件
# 最终要经过 WebSocket 发给前端——dict 可以直接用 json.dumps() 序列化,不用额外转换。
EventCallback = Callable[[dict], None]


def run_research(
    question: str,
    max_rounds: int = 2,
    papers_per_subtask: int = 2,
    on_event: EventCallback | None = None,
) -> tuple[ResearchPlan, list[PaperHit]]:
    """完整跑一次"提问 -> 拆解 -> 检索 -> 反思 -> (不够就)补充检索"的循环。

    max_rounds: 循环最多跑几轮。和 LLMClient 里的调用预算是同一个道理——
    Reflection 理论上有可能一直判断"不够",不设上限的话程序会陷入死循环,
    把 API 余额耗光都不会自己停下来。

    on_event: 可选的"进度通知"回调。每次有阶段性进展(拆解完成、某个子任务
    搜完、某一轮反思结束),就调用一次这个回调,把进展作为一个 dict 传出去。
    这个函数自己完全不关心 on_event 具体拿这些消息去做什么(打印到终端?
    通过 WebSocket 发给前端?),这是调用方的事——默认是 None,不传就什么
    都不做,不影响已经写好的测试脚本继续正常工作。
    """

    def _emit(event: dict) -> None:
        if on_event is not None:
            on_event(event)

    planner = PlannerAgent()
    reflection_agent = ReflectionAgent()
    # 两个互相独立的论文数据源,接口一致(都实现 search(subtask)->list[PaperHit]),
    # 编排层把它们当"同一种东西"并发调用——多一个数据源不用改这里的循环逻辑。
    search_tools = [
        ArxivSearchTool(max_results=papers_per_subtask),
        SemanticScholarTool(max_results=papers_per_subtask),
    ]

    plan = planner.plan(question)
    _emit({"type": "plan_created", "subtasks": [t.model_dump() for t in plan.subtasks]})

    all_papers: list[PaperHit] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    def _normalize_title(title: str) -> str:
        """去掉标点、多余空格、统一小写,用来判断"两个数据源返回的是不是同一篇论文"。

        为什么单靠url去重不够:arXiv 和 Semantic Scholar 对同一篇论文给出的url
        完全不同(比如 arxiv.org/abs/xxxx vs semanticscholar.org/paper/xxxx),
        同一篇论文被两个数据源都搜到时,url去重完全抓不住这种重复——只能退回
        到"标题内容本身是不是同一篇"这个更本质的判断标准。
        """
        return re.sub(r"[^a-z0-9]+", "", title.lower())

    # 同时最多几个子任务并发去搜——不设成"有多少子任务就开多少线程",是因为
    # arXiv 是免费公共API,官方礼貌使用政策要求控制请求频率(ArxivSearchTool里
    # 已经用 delay_seconds=3 做了限速)。给线程池设一个较小的并发上限(3),
    # 是在"确实并发、比纯顺序快"和"不把请求量打得太猛"之间找的一个平衡点,
    # 不是并发数越大越好。
    MAX_CONCURRENT_SEARCHES = 3

    def _search_and_collect(subtasks: list[SubTask]) -> None:
        """对一批子任务做检索,并把结果去重后追加进 all_papers。

        每个子任务会被同时提交给全部数据源(现在是arXiv + Semantic Scholar)
        并发查询——用线程池,多个"子任务x数据源"的组合本质上互不依赖,没必要
        排队一个个等。去重用两道关卡:PaperHit.url(同一数据源内部去重最直接)
        + 标准化后的标题(跨数据源去重,因为不同源给同一篇论文的url是不一样的)。

        注意:线程池"谁先完成就先处理谁"，各个搜索任务完成的先后顺序不再
        保证和 subtasks/数据源的原始顺序一致——这是真并发的正常副作用,如果前端
        要求事件必须严格按子任务顺序展示,需要在前端自己做排序,不能假设
        WebSocket 收到事件的顺序就是提交顺序。
        """
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SEARCHES) as pool:
            future_to_job = {
                pool.submit(tool.search, t): (t, tool.__class__.__name__)
                for t in subtasks
                for tool in search_tools
            }
            for future in as_completed(future_to_job):
                t, source_name = future_to_job[future]
                hits = future.result()  # search() 内部已经把异常降级成空列表,这里不会再抛

                new_hits = []
                for h in hits:
                    title_key = _normalize_title(h.title)
                    if h.url in seen_urls or (title_key and title_key in seen_titles):
                        continue
                    seen_urls.add(h.url)
                    if title_key:
                        seen_titles.add(title_key)
                    new_hits.append(h)

                all_papers.extend(new_hits)
                logger.info(
                    "子任务 [%s] 数据源[%s] 搜到 %d 篇,去重后新增 %d 篇",
                    t.id, source_name, len(hits), len(new_hits),
                )
                _emit({
                    "type": "search_done",
                    "subtask_id": t.id,
                    "source": source_name,
                    "query": t.query,
                    "found": len(hits),
                    "new": len(new_hits),
                })

    _search_and_collect(plan.subtasks)

    round_num = 1
    while round_num <= max_rounds:
        result = reflection_agent.reflect(plan, all_papers)
        logger.info(
            "第 %d 轮反思: is_sufficient=%s, 缺失方面=%s",
            round_num, result.is_sufficient, result.missing_aspects,
        )
        _emit({
            "type": "reflection_done",
            "round": round_num,
            "is_sufficient": result.is_sufficient,
            "missing_aspects": result.missing_aspects,
        })
        if result.is_sufficient or not result.new_subtasks:
            break
        _search_and_collect(result.new_subtasks)
        round_num += 1

    return plan, all_papers
