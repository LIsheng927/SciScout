"""
让AI按JSON格式回答,然后用Python代码把它转换成真正能操作的数据(列表)。
这是Planner Agent(任务拆解)的核心原理。
"""
import json
from openai import OpenAI
from app.core.config import get_settings

client = OpenAI(api_key=get_settings().openai_api_key)  # 换成你自己的key

question = input("请输入你的研究问题: ")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},   # 关键:强制AI只能返回合法JSON
    messages=[
        {"role": "system", "content": (
            "你是一个任务拆解助手。把用户的问题拆成3个具体的子任务。"
            '必须用这个JSON格式回答,不要有任何多余文字: {"subtasks": ["子任务1", "子任务2", "子任务3"]}'
        )},
        {"role": "user", "content": question}
    ]
)

raw_text = response.choices[0].message.content
print("AI原始返回的文字:", raw_text)

data = json.loads(raw_text)          # 把文字转换成Python的字典
subtasks = data["subtasks"]           # 从字典里取出子任务列表

print("\n拆解出的子任务:")
for i, task in enumerate(subtasks, start=1):
    print(f"  {i}. {task}")
