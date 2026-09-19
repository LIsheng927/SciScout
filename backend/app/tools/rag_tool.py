"""RAG 子系统:下载论文PDF -> 提取文字 -> 切块 -> 存入Qdrant -> 检索 -> 接地生成回答。

这是今天在 scripts/06~10 里验证过的逻辑的"正式版",区别:
1. 用 Qdrant 的本地持久化模式(存到磁盘的 qdrant_path),而不是 :memory:
   —— 这样程序重启后,之前处理过的论文不用重新下载/切块/embedding,省时间也省钱。
2. 用 PaperChunk/RetrievalResult 这些强类型结构体,而不是裸字符串。
3. 包成一个类,方便和 PlannerAgent、ArxivSearchTool 放在同一套调用方式里。
"""
from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from app.agents.schemas import PaperChunk, RagAnswer, RetrievalResult
from app.core.config import get_settings
from app.core.llm_client import LLMClient

logger = logging.getLogger("scires.tools.rag")

GROUNDING_SYSTEM_PROMPT = (
    "你是一个论文问答助手。只能根据下面提供的'背景资料'来回答问题,"
    "不要使用你自己的知识编造答案。如果背景资料里没有相关信息,就直说不知道。\n\n"
    "背景资料:\n{context}"
)


class RagTool:
    COLLECTION = "paper_chunks"
    VECTOR_SIZE = 1536  # text-embedding-3-small 输出的向量长度

    def __init__(self, llm: LLMClient | None = None) -> None:
        settings = get_settings()
        self.settings = settings
        self.llm = llm or LLMClient()
        # path=本地磁盘路径 => 持久化存储,和之前demo脚本用的 ":memory:" 是关键区别
        self.qdrant = QdrantClient(path=settings.qdrant_path)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """集合可能已经存在(上次运行创建过),不存在才新建,避免重复运行时报错。"""
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.COLLECTION not in existing:
            self.qdrant.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE),
            )

    def _get_embedding(self, text: str) -> list[float]:
        return self.llm.embed(text)  # 走 LLMClient 的公开接口,不再伸手拿它的私有属性

    def _strip_references(self, text: str) -> str:
        """砍掉"References"标题之后的内容——那基本是引用列表,不是论文正文,
        留着只会污染检索结果(比如把目录/参考文献当成"相关内容"检索出来)。
        """
        match = re.search(r"\n\s*References\s*\n", text, re.IGNORECASE)
        if match:
            return text[:match.start()]
        return text

    _STOPWORDS = {
        "the", "a", "an", "of", "to", "and", "in", "is", "are", "for",
        "with", "this", "that", "we", "our", "on", "as", "by", "from",
        "which", "be", "can", "or", "at", "these", "such", "has", "have",
        "not", "it", "its", "their", "into", "based", "than", "also",
    }

    def _is_low_quality(self, chunk: str) -> bool:
        """判断一个块是不是"主要由公式/符号堆砌而成"。

        第一版只看"字母字符占比",结果漏掉了这种块:
            "+10(eLL)^(1/3) F0σρ/T + 4F0/T, whereσ2ρ =σ2 + 3(ρ+1)σ2h, ..."
        原因: Python 的 str.isalpha() 对希腊字母(σ、ρ...)、公式里的字母变量名
        同样返回 True,所以哪怕整段基本是公式,只要变量名/希腊符号够多,
        alpha_ratio 照样能冲到 0.5 以上,把这道过滤器骗过去。

        更可靠的信号是"虚词密度":正常英文陈述句里 the/of/and/is/with 这类
        虚词(stopword)大概占所有单词的 8%~15%;公式推导即使掺了不少字母,
        也几乎不会出现这些词——它不是在"说话",只是符号在排列。
        所以在字母占比过滤的基础上,再加一层虚词密度检查,双重把关。
        """
        if not chunk:
            return True
        alpha_ratio = sum(ch.isalpha() for ch in chunk) / len(chunk)
        if alpha_ratio < 0.5:
            return True

        words = re.findall(r"[A-Za-z]+", chunk.lower())
        if len(words) < 15:
            # 英文单词太少(比如整段几乎是希腊字母+符号),同样判定为低质量
            return True

        stopword_ratio = sum(w in self._STOPWORDS for w in words) / len(words)
        return stopword_ratio < 0.08

    def _chunk_text(self, text: str) -> list[str]:
        """按段落切块(而不是死板按固定字符数切),再过滤掉质量差的块。

        做法:先按空行拆成一个个"段落",然后把连续段落攒到接近chunk_size就切一刀,
        这样绝大多数情况下不会把一句话/一个公式硬切成两半;
        遇到单个段落本身就超过chunk_size(比如很长的公式推导),才不得已按字符数硬切。
        """
        text = self._strip_references(text)
        chunk_size = self.settings.chunk_size
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) <= chunk_size:
                current += ("\n\n" if current else "") + para
            else:
                if current:
                    chunks.append(current)
                if len(para) > chunk_size:
                    for i in range(0, len(para), chunk_size):
                        chunks.append(para[i:i + chunk_size])
                    current = ""
                else:
                    current = para
        if current:
            chunks.append(current)

        good_chunks = [c for c in chunks if not self._is_low_quality(c)]
        logger.info("切块: %d 个原始块, 过滤掉低质量块后剩 %d 个", len(chunks), len(good_chunks))
        return good_chunks

    def ingest_paper(self, paper_id: str, pdf_url: str, title: str, source_url: str) -> int:
        """下载一篇论文的PDF、切块、存入Qdrant。返回存入了多少个块。

        paper_id 用于给每个chunk生成唯一编号(不同论文的chunk id不会冲突)。
        """
        # tempfile.gettempdir() 会自动返回当前系统正确的临时文件夹
        # (Windows是类似 C:\\Users\\xxx\\AppData\\Local\\Temp,Mac/Linux是 /tmp),不用自己写死路径
        local_path = str(Path(tempfile.gettempdir()) / f"{hashlib.sha256(paper_id.encode()).hexdigest()[:16]}.pdf")
        urlretrieve(pdf_url, local_path)

        reader = PdfReader(local_path)
        full_text = "\n".join(page.extract_text() for page in reader.pages)
        chunks = self._chunk_text(full_text)

        points = []
        for i, chunk_text in enumerate(chunks):
            paper_chunk = PaperChunk(
                paper_id=paper_id, chunk_index=i, text=chunk_text,
                source_title=title, source_url=source_url,
            )
            point_id = int(hashlib.sha256(f"{paper_id}:{i}".encode()).hexdigest()[:12], 16)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=self._get_embedding(chunk_text),
                    payload=paper_chunk.model_dump(),
                )
            )

        self.qdrant.upsert(collection_name=self.COLLECTION, points=points)
        logger.info("已存入论文 %s 的 %d 个片段", title, len(points))
        return len(points)

    _RRF_K = 60  # RRF公式里的平滑常数,业界常用默认值,排名差距不会被无限放大
    _CANDIDATE_POOL = 20  # dense/BM25各自先取多少候选,融合之后再截到top_k

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """给BM25用的简单分词:小写化 + 按非字母数字字符切开,并丢弃单字母token。

        只处理英文场景(论文正文以英文为主),不是通用分词器,这是有意的简化,
        面试如果问"能不能处理中文",诚实回答:当前分词器按空白/标点切分,
        对中文这种没有天然空格分词的语言效果不好,真要支持得换jieba之类的分词库
        (这也是为什么这版RAG的问答查询建议用英文/包含英文关键词——中文查询过
        这个分词器,中文部分会被直接丢弃,只留下里面夹杂的英文/数字词)。

        单字母token(比如论文里超参数记号"r"、"d"、"n")信息量太低,不去掉的话,
        BM25会因为这类字母在图表坐标轴标注、公式变量名里出现频率极高,把这些
        噪声段落的分数顶得很高,排到真正讲解内容的正文前面——这是实测踩出来的坑,
        不是理论上的预防措施。
        """
        return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 2]

    def _load_all_chunks(self) -> list[tuple[int, PaperChunk]]:
        """把collection里所有chunk都拉出来,给BM25建索引用。

        用scroll而不是query_points,是因为这里不是"按相似度查top_k",
        是要"拿到全部数据"——scroll就是Qdrant提供的、专门用来遍历整个集合的接口。
        当前项目规模(几十篇论文、几百个chunk)这样做没问题;如果论文数量涨到
        几万篇级别,每次retrieve都重新扫全量再建BM25索引会变慢,到时候需要把
        BM25索引单独持久化、增量更新,而不是每次现场重建——这是这个简化实现
        明确的可扩展性上限,不是没考虑到。
        """
        points, _ = self.qdrant.scroll(
            collection_name=self.COLLECTION, limit=10_000, with_payload=True, with_vectors=False
        )
        return [(p.id, PaperChunk(**p.payload)) for p in points]

    @staticmethod
    def _rrf_fuse(*ranked_lists: list[int], k: int) -> dict[int, float]:
        """倒数排名融合(Reciprocal Rank Fusion)。

        每个ranked_lists是一份"按相关度从高到低排好的point_id列表"(不管来自
        dense还是BM25)。同一个point_id在不同列表里排第几名,就贡献 1/(k+排名)
        分,把它在所有列表里贡献的分数加起来,就是融合后的最终得分——分数越高
        排名越靠前。只在一种方式的候选里出现过的chunk,也能拿到那一份贡献,
        不会被直接排除。
        """
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, point_id in enumerate(ranked, start=1):
                scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank)
        return scores

    def retrieve(self, query: str, top_k: int | None = None, hybrid: bool = True) -> list[RetrievalResult]:
        """检索最相关的chunk。hybrid=True(默认)时融合dense向量检索 + BM25关键词检索;
        hybrid=False时退化成纯dense检索(保留这个开关方便做A/B对比,验证混合检索
        到底有没有比单独dense更好)。
        """
        top_k = top_k or self.settings.rag_top_k

        # --- dense检索:embedding + 余弦相似度,Qdrant原生支持 ---
        query_vector = self._get_embedding(query)
        dense_hits = self.qdrant.query_points(
            collection_name=self.COLLECTION, query=query_vector, limit=self._CANDIDATE_POOL
        ).points
        dense_ranked_ids = [h.id for h in dense_hits]
        chunk_by_id: dict[int, PaperChunk] = {h.id: PaperChunk(**h.payload) for h in dense_hits}

        if not hybrid:
            return [
                RetrievalResult(chunk=chunk_by_id[pid], score=h.score)
                for pid, h in zip(dense_ranked_ids, dense_hits)
            ][:top_k]

        # --- BM25检索:现场从全量chunk建一次索引(见_load_all_chunks的说明) ---
        all_chunks = self._load_all_chunks()
        for pid, chunk in all_chunks:
            chunk_by_id.setdefault(pid, chunk)  # BM25独有、dense候选里没有的chunk也要收进来

        corpus_ids = [pid for pid, _ in all_chunks]
        tokenized_corpus = [self._tokenize(chunk.text) for _, chunk in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(self._tokenize(query))
        bm25_ranked_ids = [
            corpus_ids[i] for i in sorted(range(len(corpus_ids)), key=lambda i: bm25_scores[i], reverse=True)
        ][: self._CANDIDATE_POOL]

        # --- 融合两份排名,取最终top_k ---
        fused_scores = self._rrf_fuse(dense_ranked_ids, bm25_ranked_ids, k=self._RRF_K)
        top_ids = sorted(fused_scores, key=lambda pid: fused_scores[pid], reverse=True)[:top_k]

        return [RetrievalResult(chunk=chunk_by_id[pid], score=fused_scores[pid]) for pid in top_ids]

    def answer(self, question: str, top_k: int | None = None) -> RagAnswer:
        """检索 + 接地生成:完整的RAG问答入口。"""
        results = self.retrieve(question, top_k=top_k)
        context = "\n---\n".join(r.chunk.text for r in results)

        raw_answer = self.llm.chat(
            messages=[
                {"role": "system", "content": GROUNDING_SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
        )
        return RagAnswer(
            question=question,
            answer=raw_answer,
            source_chunks=[r.chunk.text[:200] for r in results],
        )
