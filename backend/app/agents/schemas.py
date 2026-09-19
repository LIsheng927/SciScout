"""Agent 之间传递的结构化数据。全部用 pydantic 强约束,方便:
1. LLM 输出做 json_object 强约束后直接校验,校验失败可以重试或降级
2. 后续做 benchmark 评测时,直接对这些结构化字段打分,而不是对着一段自然语言文本猜
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    id: str = Field(description="子任务编号,如 t1")
    query: str = Field(description="用于检索的具体查询语句,应尽量具体、可直接拿去搜索")
    rationale: str = Field(description="为什么需要这个子任务,与原始问题的关系")


class ResearchPlan(BaseModel):
    original_question: str
    subtasks: list[SubTask]

    def as_prompt_block(self) -> str:
        lines = [f"- [{t.id}] {t.query}  (原因: {t.rationale})" for t in self.subtasks]
        return "\n".join(lines)


class PaperHit(BaseModel):
    subtask_id: str
    title: str
    authors: list[str]
    summary: str
    url: str
    published: str


class ReflectionResult(BaseModel):
    is_sufficient: bool
    missing_aspects: list[str] = Field(default_factory=list)
    new_subtasks: list[SubTask] = Field(default_factory=list)


class PaperChunk(BaseModel):
    """RAG 子系统里"一个可检索片段"的最小单位。

    paper_id 用来把同一篇论文的所有 chunk 关联起来(比如后续要展示"这个论点来自
    第几篇论文"时用得上),chunk_index 记录它在原文里的顺序,方便调试时定位。
    """

    paper_id: str = Field(description="论文标识,如 arXiv id 或标题的hash")
    chunk_index: int = Field(description="这个片段是该论文切出的第几块,从0开始")
    text: str = Field(description="片段正文")
    source_title: str = Field(description="来源论文标题")
    source_url: str = Field(description="来源论文链接,用于引用溯源")


class RetrievalResult(BaseModel):
    """一次检索命中:命中的片段 + 相似度分数。

    单独用一个类包装而不是直接返回 PaperChunk 列表,是因为 score 是"这次检索"的
    属性,不是 chunk 本身的属性——同一个 chunk 在不同查询下分数不一样。
    """

    chunk: PaperChunk
    score: float = Field(description="与查询的相似度分数,越高越相关")


class RagAnswer(BaseModel):
    """RAG问答的结果:答案 + 用来支撑这个答案的原文片段(方便做引用溯源)。"""
    question: str
    answer: str
    source_chunks: list[str]


class Citation(BaseModel):
    """报告里的一条引用来源——对应 PaperHit 列表里的某一篇论文。"""
    ref_id: int = Field(description="引用编号,报告正文里用 [编号] 标注,从1开始")
    title: str
    url: str


class Report(BaseModel):
    """Report Agent 的最终产出:一份带引用编号的调研报告。

    content 是 markdown 格式的正文,论点后面会带 [1][2] 这种编号;
    citations 把每个编号具体对应哪篇论文(标题+链接)列出来,
    方便前端渲染成"点击编号跳转到原文链接"这种效果。
    """
    question: str
    content: str
    citations: list[Citation]


class VerificationResult(BaseModel):
    """针对报告里某一条引用的校验结果:这个编号对应的论文,
    是不是真的支撑了报告里用它标注的那个论点。
    """
    ref_id: int = Field(description="对应报告里的引用编号")
    supported: bool = Field(description="论文摘要是否真的支撑了这个引用标注的论点")
    reason: str = Field(description="判断理由,方便人工复查")
