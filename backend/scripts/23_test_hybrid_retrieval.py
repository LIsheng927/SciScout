"""验证RagTool新加的混合检索(dense+BM25)有没有真的起作用——下载一篇真实论文
存进Qdrant,拿同一个问题分别用hybrid=True和hybrid=False查一遍,对比排名前几的
结果有没有差异。

运行(在 backend 目录下):
    python -m scripts.23_test_hybrid_retrieval

这里特意选了一个包含"缩写/专有名词"的问题去测——这种问题是BM25(关键词匹配)
相对dense检索(语义理解)的强项,应该能观察到hybrid模式下,包含这个缩写原文的
chunk排名比纯dense模式更靠前。
"""
from __future__ import annotations

import arxiv

from app.tools.rag_tool import RagTool

QUERY = "how does the sketching mechanism reduce communication overhead in federated LoRA fine-tuning"


def main() -> None:
    rag = RagTool()

    # 用真实的LoRA论文做测试数据(如果之前跑过10号脚本/已经ingest过,这里会重复
    # 存一次,不影响验证——Qdrant允许同一个point_id覆盖写入)
    print("搜索并下载LoRA论文...")
    search = arxiv.Search(query="LoRA low-rank adaptation fine-tuning", max_results=1)
    client = arxiv.Client()
    paper = next(client.results(search))
    print(f"找到: {paper.title}\n")

    n_chunks = rag.ingest_paper(
        paper_id=paper.entry_id, pdf_url=paper.pdf_url,
        title=paper.title, source_url=paper.entry_id,
    )
    print(f"已存入 {n_chunks} 个chunk\n")

    print(f"查询: {QUERY}\n")

    print("=== 纯dense检索(hybrid=False) ===")
    dense_results = rag.retrieve(QUERY, top_k=3, hybrid=False)
    for i, r in enumerate(dense_results, start=1):
        print(f"[{i}] score={r.score:.4f}")
        print(f"    {r.chunk.text[:150]}...")

    print("\n=== 混合检索(hybrid=True, dense+BM25融合) ===")
    hybrid_results = rag.retrieve(QUERY, top_k=3, hybrid=True)
    for i, r in enumerate(hybrid_results, start=1):
        print(f"[{i}] rrf_score={r.score:.4f}")
        print(f"    {r.chunk.text[:150]}...")


if __name__ == "__main__":
    main()
