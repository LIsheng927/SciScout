"""
把之前的函数版代码,改造成"类"的写法。
逻辑完全没变,只是组织方式变了——这是理解这一步的关键:不要以为类是什么新功能,
它只是把同一批东西换了个"打包方式"。
"""
import json
import arxiv
from openai import OpenAI
from app.core.config import get_settings


class ResearchAssistant:
    """一个"工具箱":把OpenAI连接、arXiv连接,以及围绕它们的操作,都装在这一个对象里。"""

    def __init__(self, api_key, max_results=3):
        """__init__ 是"开箱即用"的初始化动作:创建这个工具箱的那一刻,
        就把里面需要的东西(连接对象、设置)都准备好,存在 self 里。
        self 就是"这个工具箱自己",self.xxx = ... 就是"往这个工具箱里放东西"。
        """
        self.llm_client = OpenAI(api_key=api_key)
        self.arxiv_client = arxiv.Client()
        self.max_results = max_results

    def plan(self, question):
        """拆解问题。跟之前函数版一样,只是这次用 self.llm_client 而不是外部的 client 变量。"""
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": (
                    "你是一个任务拆解助手。把用户的中文研究问题拆成3个具体的英文检索关键词。"
                    '必须用这个JSON格式回答: {"subtasks": ["英文关键词1", "英文关键词2", "英文关键词3"]}'
                )},
                {"role": "user", "content": question}
            ]
        )
        data = json.loads(response.choices[0].message.content)
        return data["subtasks"]

    def search(self, query):
        """搜索arXiv。跟之前函数版一样,只是用 self.arxiv_client 和 self.max_results。"""
        search = arxiv.Search(query=query, max_results=self.max_results, sort_by=arxiv.SortCriterion.Relevance)
        return list(self.arxiv_client.results(search))


if __name__ == "__main__":
    # 用法:先"开箱"(创建一个ResearchAssistant实例),再调用它的方法
    assistant = ResearchAssistant(api_key=get_settings().openai_api_key)  # 换成你的key

    question = input("请输入你的研究问题: ")
    subtasks = assistant.plan(question)

    print("\n=== 拆解出的子任务 ===")
    for i, task in enumerate(subtasks, start=1):
        print(f"  {i}. {task}")

    print("\n=== 开始检索 ===")
    for task in subtasks:
        papers = assistant.search(task)
        print(f"\n--- {task} (找到 {len(papers)} 篇) ---")
        for p in papers:
            print(f"  * {p.title} ({p.published.strftime('%Y-%m-%d')})")
