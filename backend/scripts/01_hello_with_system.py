"""
在上一个例子基础上加两个东西:
1. system 消息:给AI设定一个"身份/行为规则"
2. 用 input() 让你自己输入问题,而不是写死在代码里
"""
from openai import OpenAI
from app.core.config import get_settings

client = OpenAI(api_key=get_settings().openai_api_key)  # 记得换成你自己的key

user_question = input("请输入你的问题: ")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "你是一个只会用中文回答、并且每次回答都不超过30个字的助手。"},
        {"role": "user", "content": user_question}
    ]
)

print(response.choices[0].message.content)
