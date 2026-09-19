"""测试完整的编排循环:一次性验证
Planner -> 检索 -> 反思 -> (不够就)补充检索 -> 再反思
这一整条链路能不能自动跑起来,而不用像之前那样自己手动看一轮结果再决定下一步。
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 把日志级别设成 INFO,这样 orchestrator.py 里 logger.info(...) 打的过程信息
# (比如"第几轮反思、去重后新增了几篇")才会显示在终端上,不然默认级别看不到。
logging.basicConfig(level=logging.INFO, format="%(message)s")

from app.orchestrator import run_research

question = input("请输入你的研究问题: ")
plan, papers = run_research(question, max_rounds=2)

print(f"\n=== 最终结果: 共收集到 {len(papers)} 篇不重复的论文 ===")
for p in papers:
    print(f"  - [{p.subtask_id}] {p.title}")
