"""验证 RagTool 这个正式类:用一篇真实论文测试 ingest -> answer 全流程。"""
import sys
from pathlib import Path

# 把项目根目录(backend/)加入模块搜索路径,这样才能 import app.xxx
# __file__ 是"这个脚本自己的路径",.parent 两次就是从 scripts/ 退到 backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import arxiv

from app.tools.rag_tool import RagTool

# 搜一篇论文
search = arxiv.Search(query="LoRA low-rank adaptation fine-tuning", max_results=1)
paper = next(arxiv.Client().results(search))
print(f"找到论文: {paper.title}")

# 用 arXiv id 作为 paper_id(唯一标识,不同论文不会冲突)
paper_id = paper.get_short_id()

rag = RagTool()
n = rag.ingest_paper(
    paper_id=paper_id,
    pdf_url=paper.pdf_url,
    title=paper.title,
    source_url=paper.entry_id,
)
print(f"已存入 {n} 个片段(point_id按paper_id+序号算出来是固定的,重复跑这个脚本会覆盖更新,不会重复堆积)\n")

question = input("请输入关于这篇论文的问题: ")
result = rag.answer(question)

print(f"\n=== 回答 ===\n{result.answer}")
print(f"\n=== 支撑这个回答的原文片段(用于引用溯源) ===")
for i, chunk in enumerate(result.source_chunks, start=1):
    print(f"[{i}] {chunk}...")
