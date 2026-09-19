"""Follow-up Agent:处理用户在拿到报告之后的追问。

为什么需要单独一个 Agent,而不是直接把追问塞给 ReportAgent 复用:
ReportAgent 的职责是"从零生成一份完整报告",prompt 是按"写报告"设计的
(要求分节、要求覆盖全部子任务)。追问的场景不一样——用户往往只是想深挖
某一个点("第二篇论文具体怎么做的"),不需要每次都重新组织成一整篇报告结构,
而且追问必须"记得"之前已经问过什么、答过什么(多轮对话),ReportAgent
完全没有"对话历史"这个概念。职责不同,prompt 和输入结构也该不同,所以
拆成两个类,而不是硬塞一个参数去分支复用。

信息颗粒度和 ReportAgent 保持一致(有意为之): 现在 production 的报告生成
只用检索阶段拿到的"标题+摘要",没有接入 rag_tool.py 那套全文混合检索
(hybrid dense+BM25)。追问先保持同样的颗粒度,不引入不一致的两套"证据源"
——等以后把 RAG 全文接入 ReportAgent,追问可以顺带升级。
"""
from __future__ import annotations

import logging

from app.agents.schemas import PaperHit
from app.core.llm_client import ChatClient, LLMClient

logger = logging.getLogger("scires.followup")

# 历史滑动窗口:每次追问只把"最近 N 轮"问答拼进 prompt,而不是从头到现在的
# 全部历史。不这样做的话,历史长度和请求成本是二次增长关系——第k轮追问要
# 把前面k-1轮全部重新当输入token付费一遍,轮数一多,费用、延迟都会跟着爆炸,
# 而且过长的上下文反而会让模型更难注意到关键信息(业内叫 "lost in the
# middle": 模型对长输入"中间部分"的注意力,普遍不如开头和结尾部分)。
# 6 是一个经验值:对于"深挖一份报告"这种场景,用户很少会一直追问超过6轮
# 还高度依赖更早的上下文,超过这个窗口的历史被截掉,是"记住最近聊了什么"
# 和"不让成本失控"之间的一个取舍,不是越大越好。
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """你是一个科研文献调研助手中的"追问答疑"模块。
用户已经拿到过一份基于若干篇论文写的调研报告,现在针对报告内容或者论文本身继续提问。

你会收到:
1. 原始研究问题
2. 检索到的论文列表(每篇论文前面标了编号,格式和之前生成报告时一样)
3. 之前的追问历史(如果有):用户问过什么、你之前怎么答的(只包含最近几轮)
4. 用户这一轮的新问题

要求:
1. 只能依据提供的论文列表(标题+摘要)来回答,不要使用你自己的知识编造论文之外的信息
2. 如果论文列表里的信息不足以回答这个追问,如实说"现有检索到的论文摘要没有覆盖这一点",
   不要编造细节
3. 回答里提到具体论点时,同样要用 [编号] 标注依据哪篇论文,编号对应论文列表里的编号
4. 回答保持简洁,针对用户这一句追问来回答,不需要重新写一整份报告结构
5. 如果新问题看起来是在延续之前追问历史里的某个话题,要结合历史来理解用户在问什么

直接输出回答正文(markdown 格式,可以带 [1][2] 这样的引用编号),不需要 JSON,不需要多余的前后缀。
"""


class FollowUpAgent:
    def __init__(self, llm: ChatClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def answer(
        self,
        original_question: str,
        papers: list[PaperHit],
        history: list[dict],
        new_question: str,
    ) -> str:
        """papers 的编号规则和 ReportAgent._format_papers 完全一致(从1开始,
        按 papers 列表顺序编号),这样追问回答里的 [编号] 才能对应上用户已经
        在报告里看到过的那一套编号,不会出现"报告里的[3]"和"追问回答里的[3]"
        实际指向不同论文的错位。

        注意: 这里传进来的 history 是"这次连接从头到现在的完整历史"
        (research_ws.py 里维护的那个列表,只增不减),真正会被截断成
        最近 MAX_HISTORY_TURNS 轮的,是下面 _format_history 拼 prompt 的
        那一步——调用方(research_ws.py)不需要关心截断这件事,截断是
        FollowUpAgent 自己的实现细节,职责边界很清楚。
        """
        papers_block = self._format_papers(papers)
        history_block = self._format_history(history)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"原始研究问题:{original_question}\n\n"
                    f"论文列表:\n{papers_block}\n\n"
                    f"{history_block}"
                    f"用户这一轮的新问题:{new_question}"
                ),
            },
        ]
        answer = self.llm.chat(messages, temperature=0.3)
        return answer

    @staticmethod
    def _format_papers(papers: list[PaperHit]) -> str:
        lines = [f"[{i}] {p.title}: {p.summary}" for i, p in enumerate(papers, start=1)]
        return "\n".join(lines)

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        """没有历史时返回空字符串(第一次追问,还没有"之前问过什么")。

        用 history[-MAX_HISTORY_TURNS:] 做滑动窗口截断——Python 切片的负数
        索引意思是"从倒数第N个开始取到末尾",list比MAX_HISTORY_TURNS短时
        这个写法不会报错(切片天然容忍越界),直接返回整个列表,不需要
        额外判断长度。这是最简单的"短期记忆"实现方式:不需要专门的记忆
        数据库或者摘要压缩,就是把最近几轮历史原样拼进这一次请求的 prompt
        里。更早的历史会被彻底丢弃(不是压缩,是完全不进这次的输入)——
        如果用户真的在第8轮追问里提到第2轮讨论过的细节,模型是"记不住"的,
        这是当前设计明确接受的取舍,不是bug。
        """
        if not history:
            return ""
        recent = history[-MAX_HISTORY_TURNS:]
        lines = ["之前的追问历史(最近几轮):"]
        for turn in recent:
            lines.append(f"问: {turn['question']}")
            lines.append(f"答: {turn['answer']}")
        return "\n".join(lines) + "\n\n"
