"""
用真正的向量数据库 Qdrant,替代之前手动算相似度的方式。
这里用的是Qdrant的"内存模式"(:memory:),不用额外装数据库软件,方便先理解概念;
真实项目部署时会换成"本地文件模式"或者"服务器模式"来持久化保存。
"""
import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from app.core.config import get_settings

client = OpenAI(api_key=get_settings().openai_api_key)  # 从 .env 读取,不再写死在代码里


def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


# ---- 第一步: 创建一个Qdrant客户端和一个"集合"(collection,类似数据库里的"表") ----
qdrant = QdrantClient(":memory:")   # ":memory:" 表示数据存在内存里,程序结束就没了,先用来学习概念
qdrant.create_collection(
    collection_name="paper_chunks",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),  # 1536是text-embedding-3-small输出向量的长度
)

# ---- 第二步: 把文本片段转换成向量,存进Qdrant ----
chunks = [
    "本文提出的方法在ImageNet数据集上达到了92.3%的准确率,比之前最好的方法提升了1.8个百分点。",
    "实验部分使用了8张A100 GPU进行训练,总训练时长约为72小时。",
    "该方法的核心创新点在于引入了一种新的注意力机制,能够动态调整不同层之间的信息流动。",
    "LoRA通过在原始权重旁边加入低秩矩阵来实现高效微调,大幅减少可训练参数量。",
]

points = []
for i, chunk in enumerate(chunks):
    points.append(
        PointStruct(
            id=i,                          # 每条数据的编号
            vector=get_embedding(chunk),   # 这段文字对应的向量
            payload={"text": chunk}        # 附带存一下原文,方便查到向量后能取回文字内容
        )
    )

qdrant.upsert(collection_name="paper_chunks", points=points)
print(f"已存入 {len(points)} 条数据到 Qdrant\n")

# ---- 第三步: 真正的检索,一行代码搞定,不用自己写循环算相似度了 ----
query = input("请输入你的问题: ")
query_vector = get_embedding(query)

results = qdrant.query_points(
    collection_name="paper_chunks",
    query=query_vector,
    limit=2,   # 取最相关的2条
).points

print("\n=== Qdrant 检索结果 ===")
for r in results:
    print(f"[相似度分数 {r.score:.4f}] {r.payload['text']}")
