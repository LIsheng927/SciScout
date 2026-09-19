"""测试 ReflectionAgent:完整跑一遍 Planner -> ArxivSearchTool -> ReflectionAgent,
看反思模块怎么判断"目前检索到的论文够不够回答问题",不够的话又会补充出什么新子任务。

这个脚本本身还不是"循环"——先看清楚反思这一步单独跑起来是什么样子,
理解了之后,下一步才是把它接进一个 while 循环里,变成真正的多轮检索。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.planner import PlannerAgent
from app.agents.reflection import ReflectionAgent
from app.tools.arxiv_tool import ArxivSearchTool

question = input("请输入你的研究问题: ")

# ---- 第一步: Planner 拆解 ----
planner = PlannerAgent()
plan = planner.plan(question)
print("\n=== Planner 拆解出的子任务 ===")
for t in plan.subtasks:
    print(f"  [{t.id}] {t.query}  (原因: {t.rationale})")

# ---- 第二步: 对每个子任务去 arXiv 检索 ----
arxiv_tool = ArxivSearchTool(max_results=2)  # 每个子任务只搜2篇,省时间也省钱
all_papers = []
print("\n=== 开始检索 ===")
for t in plan.subtasks:
    hits = arxiv_tool.search(t)
    print(f"  子任务 [{t.id}] 找到 {len(hits)} 篇")
    for h in hits:
        print(f"      - {h.title}")
    all_papers.extend(hits)

# ---- 第三步: Reflection 评估"这些论文够不够回答原始问题" ----
reflection = ReflectionAgent()
result = reflection.reflect(plan, all_papers)

print("\n=== Reflection 评估结果 ===")
print(f"是否足够: {result.is_sufficient}")
if not result.is_sufficient:
    print("缺失的方面:")
    for m in result.missing_aspects:
        print(f"  - {m}")
    print("补充的新子任务:")
    for t in result.new_subtasks:
        print(f"  [{t.id}] {t.query}  (原因: {t.rationale})")
