"""
把Planner拆解 + arXiv检索拼到一起:
输入一个研究问题 -> AI拆成几个子任务 -> 对每个子任务自动去arXiv搜论文
"""
import json
import arxiv
from openai import OpenAI
from app.core.config import get_settings

client = OpenAI(api_key=get_settings().openai_api_key)  # 换成你自己的key
arxiv_client = arxiv.Client()


def search_arxiv(query, max_results=3):
    """把'搜索'这个动作包装成一个函数,输入关键词,返回论文列表。
    这样下面对每个子任务调用时,只需要写 search_arxiv(某个子任务) 一行,不用重复整段代码。
    """
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    return list(arxiv_client.results(search))


# ---- 第一步: Planner 拆解问题 ----
question = input("请输入你的研究问题: ")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": (
            "你是一个任务拆解助手。把用户的中文研究问题拆成3个具体的英文检索关键词(用于arXiv搜索,所以必须是英文)。"
            '必须用这个JSON格式回答: {"subtasks": ["英文关键词1", "英文关键词2", "英文关键词3"]}'
        )},
        {"role": "user", "content": question}
    ]
)
data = json.loads(response.choices[0].message.content)
subtasks = data["subtasks"]

print("\n=== Planner 拆解出的子任务(已翻译成英文关键词) ===")
for i, task in enumerate(subtasks, start=1):
    print(f"  {i}. {task}")

# ---- 第二步: 对每个子任务调用 search_arxiv 函数 ----
print("\n=== 开始检索 ===")
for task in subtasks:
    papers = search_arxiv(task)
    print(f"\n--- 子任务: {task} (找到 {len(papers)} 篇) ---")
    for p in papers:
        print(f"  * {p.title} ({p.published.strftime('%Y-%m-%d')})")
