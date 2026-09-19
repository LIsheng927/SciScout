"""
最小例子:让你亲眼看到"意思相近的句子,数字确实更接近"。
用OpenAI的embedding模型,把几句话转换成向量,然后算它们两两之间的"相似度"。
"""
import numpy as np
from openai import OpenAI
from app.core.config import get_settings

client = OpenAI(api_key=get_settings().openai_api_key)  # 换成你的key

# 故意放几句话:前两句意思接近(都是聊RAG检索增强),第三句完全不相关(聊做饭)
sentences = [
    "RAG通过检索外部知识来增强大语言模型的回答质量",
    "检索增强生成技术可以让AI引用外部文档来回答问题",
    "今天中午我打算做一份番茄炒蛋"
]


def get_embedding(text):
    """把一段文字变成一串数字(向量)。"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    """计算两个向量的'相似度',范围大约在-1到1之间,越接近1说明意思越像。"""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 把每句话都转换成向量
embeddings = [get_embedding(s) for s in sentences]

print("=== 三句话两两之间的相似度 ===\n")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"句子{i+1} vs 句子{j+1}: 相似度 = {sim:.4f}")
        print(f"  [{i+1}] {sentences[i]}")
        print(f"  [{j+1}] {sentences[j]}\n")
