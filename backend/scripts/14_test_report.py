"""端到端测试:提问 -> Planner拆解 -> 检索 -> Reflection反思(不够就补充检索)
-> Report生成。这是目前为止第一次把整条链路完整跑通,拿到一份真正的、
带引用编号的调研报告。
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from app.agents.report import ReportAgent
from app.agents.verifier import CitationVerifier
from app.orchestrator import run_research

question = input("请输入你的研究问题: ")

plan, papers = run_research(question, max_rounds=2)
print(f"\n=== 检索完成: 共 {len(papers)} 篇不重复的论文 ===")

report_agent = ReportAgent()
report = report_agent.generate(plan, papers)

print("\n=== 生成的报告 ===")
print(report.content)

print("\n=== 引用来源 ===")
for c in report.citations:
    print(f"  [{c.ref_id}] {c.title}\n      {c.url}")

verifier = CitationVerifier()
verification_results = verifier.verify(report, papers)

print("\n=== 引用校验结果 ===")
for r in verification_results:
    tag = "通过" if r.supported else "!! 不支撑 !!"
    print(f"  [{r.ref_id}] {tag} -- {r.reason}")
