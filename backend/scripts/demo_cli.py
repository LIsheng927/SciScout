"""Week 1 最小端到端 demo。

跑法(先激活虚拟环境、配好 .env):
    python scripts/demo_cli.py "大语言模型的检索增强生成(RAG)有哪些主流优化方法?"

流程: Planner 拆解问题 -> 对每个子任务并行调用 arXiv 检索 -> 打印汇总结果。
这一版本还没有 Writer/反思 agent,先把"规划 -> 检索"这条主链路跑通、可演示。
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.agents.planner import PlannerAgent
from app.agents.schemas import PaperHit
from app.core.config import get_settings
from app.tools.arxiv_tool import ArxivSearchTool


def run(question: str) -> None:
    settings = get_settings()
    print(f"\n=== 研究问题 ===\n{question}\n")

    planner = PlannerAgent()
    plan = planner.plan(question)
    print("=== Planner 拆解出的子任务 ===")
    print(plan.as_prompt_block())
    print()

    tool = ArxivSearchTool(max_results=settings.arxiv_max_results)
    all_hits: list[PaperHit] = []

    print("=== 并行检索 arXiv ===")
    with ThreadPoolExecutor(max_workers=len(plan.subtasks)) as pool:
        future_map = {pool.submit(tool.search, t): t for t in plan.subtasks}
        for future in as_completed(future_map):
            subtask = future_map[future]
            hits = future.result()
            all_hits.extend(hits)
            print(f"\n--- [{subtask.id}] {subtask.query} -> {len(hits)} 篇 ---")
            for h in hits:
                print(f"  * {h.title} ({h.published})  {h.url}")

    print(f"\n=== 汇总 ===\n共检索到 {len(all_hits)} 篇候选文献,覆盖 {len(plan.subtasks)} 个子任务。")
    print(f"本次 LLM 调用次数: {planner.llm.call_count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('用法: python scripts/demo_cli.py "你的研究问题"')
        sys.exit(1)
    run(sys.argv[1])
