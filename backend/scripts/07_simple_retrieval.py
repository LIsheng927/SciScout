"""
从"算两句话相似度",升级成"给一个问题,从一堆候选句子里找出最相关的"。
这就是"检索"(Retrieval)本身,只是候选句子数量还很少,还没用到真正的向量数据库。
"""
import numpy as np
from openai import OpenAI
from app.core.config import get_settings

client = OpenAI(api_key=get_settings().openai_api_key)  # 换成你的key

# 假设这是从几篇论文摘要里拆出来的候选句子(实际项目里会是论文的真实片段)
candidates = [
    "LoRA通过在原始权重旁边加入低秩矩阵来实现高效微调,大幅减少可训练参数量",
    "多智能体系统中,智能体之间通过消息传递协议进行协作与信息共享",
    "检索增强生成利用外部知识库来提升大语言模型回答的准确性和时效性",
    "Transformer架构中的自注意力机制能够捕捉序列中任意两个位置之间的依赖关系",
]


def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(response.data[0].embedding)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 提前把所有候选句子转换成向量,存起来(实际项目里这一步会存进向量数据库)
candidate_embeddings = [get_embedding(c) for c in candidates]

query = input("请输入你的问题: ")
query_embedding = get_embedding(query)

# 算问题跟每个候选句子的相似度
scores = [cosine_similarity(query_embedding, emb) for emb in candidate_embeddings]

# 按相似度从高到低排序,把候选句子和分数打包在一起排
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

print("\n=== 按相关度排序的结果 ===")
for text, score in ranked:
    print(f"[{score:.4f}] {text}")
