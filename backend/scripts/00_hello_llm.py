"""
第一个最小例子:什么都不封装,就是最原始地"发一个问题给AI,把回答打印出来"。
运行前需要:
  1. pip install openai
  2. 把下面 YOUR_API_KEY 换成你自己的 OpenAI key(先跑这一个文件,不用管 .env 配置那些)
"""
from openai import OpenAI
from app.core.config import get_settings

client = OpenAI(api_key=get_settings().openai_api_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "你好,请用一句话介绍你自己"}
    ]
)

print(response.choices[0].message.content)
