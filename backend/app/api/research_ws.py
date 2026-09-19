"""WebSocket 接口:把 orchestrator + ReportAgent + CitationVerifier 这条同步、
阻塞式的 pipeline,接进一个异步的 FastAPI WebSocket 连接里,并且把中间的
每一步进展实时推送给前端。

核心矛盾: run_research() 等函数内部调 openai/arxiv 这些库都是同步阻塞的,
执行到"等API返回"就会卡住整个线程;但 WebSocket 处理函数是 async def,
直接在里面调用会把整个服务器都卡住。解决办法: 把这条 pipeline 丢到一个
独立的工作线程里跑,工作线程通过一个线程安全的 queue.Queue,把进度事件
传回主线程,主线程再把这些事件一条条转发给前端。

多轮追问(本次新增): 第一轮 研究->报告->校验 跑完之后,连接不再直接关闭,
而是继续 receive_text() 监听后续消息——每条后续消息被当成"追问",
交给 FollowUpAgent 处理,复用同一次连接里已经检索到的论文列表和
report,不用把整个 pipeline 重新跑一遍(便宜、也快很多)。
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agents.followup import FollowUpAgent
from app.agents.report import ReportAgent
from app.agents.schemas import PaperHit, ResearchPlan
from app.agents.verifier import CitationVerifier
from app.orchestrator import run_research

logger = logging.getLogger("scires.api.research_ws")

router = APIRouter()

# 工作线程用这个特殊对象告诉主线程"我跑完了,别再等新消息了"。
# 用一个专属的 object() 实例而不是 None/字符串,是为了保证不会跟任何
# 真实的事件数据混淆(真实事件永远是 dict,不可能等于这个哨兵对象)。
_DONE = object()


def _run_pipeline_sync(question: str, on_event, result_holder: dict) -> None:
    """真正干活的同步函数,会被丢进独立线程里执行。

    依次跑完 检索循环 -> 生成报告 -> 校验引用,每一步都通过 on_event
    往外报告进展。这个函数完全不知道 WebSocket 或者 queue 的存在——
    它只管调用 on_event(一个普通函数),不关心调用者拿这些事件去做什么。

    result_holder: 一个普通的 dict,worker 线程跑完之后把 plan/papers 存进去。
    之所以不能直接 return 把结果带出去——这个函数是被丢进线程池执行的
    (loop.run_in_executor),执行结果不会自动"传回"外层的 async 函数,
    要么用 Future.result() 拿(但这里外层是靠 queue 轮询,不是直接 await
    这个 future),要么就用这种"传一个可变对象进去,worker 自己往里写"
    的办法,让外层在 worker 跑完之后可以从这个共享的 dict 里读到结果。
    """
    plan, papers = run_research(question, on_event=on_event)

    on_event({"type": "generating_report"})
    report_agent = ReportAgent()
    report = report_agent.generate(plan, papers)
    on_event({
        "type": "report_ready",
        "content": report.content,
        "citations": [c.model_dump() for c in report.citations],
    })

    on_event({"type": "verifying_citations"})
    verifier = CitationVerifier()
    results = verifier.verify(report, papers)
    on_event({
        "type": "verification_done",
        "results": [r.model_dump() for r in results],
    })

    result_holder["plan"] = plan
    result_holder["papers"] = papers


async def _handle_followup_turns(
    websocket: WebSocket,
    original_question: str,
    plan: ResearchPlan,
    papers: list[PaperHit],
) -> None:
    """第一轮 pipeline 跑完之后进入的"追问模式"。

    history 是这个函数自己的局部变量,只在这一次 WebSocket 连接的生命周期内
    存在——这就是"短期记忆":连接一断开(用户关掉页面/刷新),这段历史就
    随着这个函数结束而消失,不会持久化到数据库。这对应之前讨论过的取舍:
    我们要的是"这一次调研会话内能追问",不是跨会话、跨设备都记得的长期记忆,
    后者是完全不同量级的工程(需要用户账号体系、持久化存储),超出这个项目
    的范围。
    """
    history: list[dict] = []

    while True:
        try:
            raw = await websocket.receive_text()
        except WebSocketDisconnect:
            logger.info("客户端断开连接,结束追问模式")
            return

        try:
            data = json.loads(raw)
            followup_question = data["question"]
        except Exception:
            await websocket.send_json(
                {"type": "error", "message": '追问格式不对,需要发送 {"question": "..."}'}
            )
            continue

        await websocket.send_json({"type": "followup_thinking"})

        loop = asyncio.get_event_loop()
        try:
            followup_agent = FollowUpAgent()
            answer = await loop.run_in_executor(
                None,
                followup_agent.answer,
                original_question,
                papers,
                history,
                followup_question,
            )
        except Exception as exc:  # noqa: BLE001 - 追问失败也要报告给前端,不能让连接卡死
            logger.exception("追问处理失败")
            await websocket.send_json({"type": "error", "message": str(exc)})
            continue

        history.append({"question": followup_question, "answer": answer})
        await websocket.send_json({
            "type": "followup_answer",
            "question": followup_question,
            "answer": answer,
        })


@router.websocket("/ws/research")
async def research_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        question = data["question"]
    except Exception:
        await websocket.send_json(
            {"type": "error", "message": '请求格式不对,需要发送 {"question": "..."}'}
        )
        await websocket.close()
        return

    # 线程安全的"传送带":工作线程用 put() 往里塞事件,
    # 主线程(下面的 while 循环)用 get() 从里面取事件。
    event_queue: queue.Queue = queue.Queue()
    result_holder: dict = {}

    def on_event(event: dict) -> None:
        """会在工作线程里被调用——把事件放进传送带,不直接碰 WebSocket
        (WebSocket 的收发必须在主线程的事件循环里做,工作线程不能直接调)。
        """
        event_queue.put(event)

    def worker() -> None:
        try:
            _run_pipeline_sync(question, on_event, result_holder)
        except Exception as exc:  # noqa: BLE001 - 把工作线程里的异常也报告给前端,而不是让线程静默失败
            logger.exception("pipeline 执行失败")
            event_queue.put({"type": "error", "message": str(exc)})
        finally:
            event_queue.put(_DONE)

    loop = asyncio.get_event_loop()
    # run_in_executor(None, ...) = 丢给线程池去跑,不阻塞当前这个 async 函数。
    loop.run_in_executor(None, worker)

    while True:
        # event_queue.get() 本身也是阻塞的(队列空的时候会一直等),
        # 所以同样要用 run_in_executor,不能直接在 async 函数里调用它。
        event = await loop.run_in_executor(None, event_queue.get)
        if event is _DONE:
            break
        await websocket.send_json(event)

    # 第一轮 pipeline 顺利跑完(result_holder 里有 plan/papers)才进入追问模式;
    # 如果 pipeline 中途报错(比如上面发过 error 事件),result_holder 是空的,
    # 没有论文列表可以追问,直接关闭连接,和原来的行为一致。
    if "plan" in result_holder and "papers" in result_holder:
        await _handle_followup_turns(
            websocket, question, result_holder["plan"], result_holder["papers"]
        )

    await websocket.close()
