"""
完整RAG流程:检索最相关内容 -> 把内容塞进prompt -> AI基于这段内容回答(而不是凭记忆瞎编)
"""
import numpy as np
from openai import OpenAI
from app.core.config import get_settings

client = OpenAI(api_key=get_settings().openai_api_key)  # 换成你的key

# 假装这是从论文里提取出的几个片段(真实项目里会是论文正文的分段内容)
chunks = [
    "本文提出的方法在ImageNet数据集上达到了92.3%的准确率,比之前最好的方法提升了1.8个百分点。",
    "实验部分使用了8张A100 GPU进行训练,总训练时长约为72小时。",
    "该方法的核心创新点在于引入了一种新的注意力机制,能够动态调整不同层之间的信息流动。",
]


def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(response.data[0].embedding)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def retrieve(query, chunks, top_k=1):
    """检索:返回跟query最相关的top_k个chunk"""
    query_emb = get_embedding(query)
    scored = [(c, cosine_similarity(query_emb, get_embedding(c))) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, score in scored[:top_k]]


def generate_answer(query, retrieved_chunks):
    """生成:把检索到的内容作为背景知识,让AI基于这些内容回答"""
    context = "\n".join(retrieved_chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "你是一个论文问答助手。只能根据下面提供的'背景资料'来回答问题,"
                "不要使用你自己的知识编造答案。如果背景资料里没有相关信息,就直说不知道。\n\n"
                f"背景资料:\n{context}"
            )},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content


question = input("请输入你的问题: ")
retrieved = retrieve(question, chunks, top_k=1)

print(f"\n=== 检索到的相关内容 ===\n{retrieved[0]}")

answer = generate_answer(question, retrieved)
print(f"\n=== AI基于检索内容的回答 ===\n{answer}")
