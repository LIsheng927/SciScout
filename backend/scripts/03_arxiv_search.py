"""
把Planner拆解出的子任务,拿去arXiv搜索真实论文。
这一步不花钱(arXiv免费,不需要key),用来先验证"搜索"这部分逻辑。
"""
import arxiv

subtask = input("请输入一个检索关键词(比如: multi-agent task planning): ")

# 创建一个"搜索请求",告诉arXiv要搜什么、要几条结果
search = arxiv.Search(
    query=subtask,
    max_results=3,
    sort_by=arxiv.SortCriterion.Relevance
)

client = arxiv.Client()

print(f"\n搜索 '{subtask}' 的结果:\n")
for result in client.results(search):
    print(f"标题: {result.title}")
    print(f"作者: {', '.join(a.name for a in result.authors)}")
    print(f"发表日期: {result.published.strftime('%Y-%m-%d')}")
    print(f"链接: {result.entry_id}")
    print(f"摘要: {result.summary[:150]}...")
    print("-" * 60)
