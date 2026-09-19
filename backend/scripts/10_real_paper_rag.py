"""
真实数据版RAG:下载一篇真实arXiv论文的PDF -> 提取文字 -> 切块(chunking) -> 存入Qdrant -> 检索问答。
"""
import arxiv
from pypdf import PdfReader
from urllib.request import urlretrieve
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from app.core.config import get_settings

client = OpenAI(api_key=get_settings().openai_api_key)  # 从 .env 读取,不再写死在代码里


def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def chunk_text(text, chunk_size=800, overlap=100):
    """把长文本切成一个个小块。
    chunk_size: 每块大概多少个字符
    overlap: 相邻两块之间重叠多少字符(避免一句话被硬生生切成两半,导致语义丢失)
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)   # 下一块从"当前块结尾往回退overlap"的地方开始,制造重叠
    return chunks


# ---- 第一步: 搜一篇论文,下载PDF ----
print("正在搜索论文...")
search = arxiv.Search(query="LoRA low-rank adaptation fine-tuning", max_results=1)
arxiv_client = arxiv.Client()
paper = next(arxiv_client.results(search))
print(f"找到论文: {paper.title}")

print("正在下载PDF...")
pdf_path = "temp_paper.pdf"
urlretrieve(paper.pdf_url, pdf_path)

# ---- 第二步: 从PDF提取纯文字 ----
reader = PdfReader(pdf_path)
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"
print(f"提取到 {len(full_text)} 个字符的文字\n")

# ---- 第三步: 切块 ----
chunks = chunk_text(full_text)
print(f"切成了 {len(chunks)} 个块\n")

# ---- 第四步: 存入Qdrant ----
print("正在生成向量并存入数据库(这一步会调用embedding API,有点耗时)...")
qdrant = QdrantClient(":memory:")
qdrant.create_collection(
    collection_name="real_paper",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
points = [
    PointStruct(id=i, vector=get_embedding(c), payload={"text": c})
    for i, c in enumerate(chunks)
]
qdrant.upsert(collection_name="real_paper", points=points)
print(f"已存入 {len(points)} 个块\n")

# ---- 第五步: 检索 + 生成回答 ----
question = input("请输入关于这篇论文的问题: ")
query_vector = get_embedding(question)
results = qdrant.query_points(collection_name="real_paper", query=query_vector, limit=3).points

context = "\n---\n".join(r.payload["text"] for r in results)
answer = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"只根据下面的论文片段回答问题,不知道就说不知道。\n\n{context}"},
        {"role": "user", "content": question}
    ]
)

print(f"\n=== 回答 ===\n{answer.choices[0].message.content}")
