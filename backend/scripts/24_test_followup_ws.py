"""测试追问功能:先跑一次完整调研,拿到报告之后,连接不断开,
可以在终端里一直输入追问,实时看后端怎么回答。

和 15_test_websocket_client.py 的区别: 15 用 `async for raw in ws` 一直等到
连接关闭为止,现在连接在第一轮跑完之后不会自动关闭了,所以要改成
"先专门等到 verification_done 事件,再切换成读用户输入、发追问"这种
更明确的状态机写法,不能再简单地"收到什么就打印什么直到断开"。

运行前,先在**另一个终端**里把后端服务跑起来:
    uvicorn app.main:app --reload
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets


async def main() -> None:
    question = input("请输入你的研究问题: ")
    uri = "ws://127.0.0.1:8000/ws/research"

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"question": question}))

        # 第一阶段: 一直打印事件,直到看到 verification_done(第一轮 pipeline
        # 的最后一个事件)或者 error,才说明可以开始追问了。
        while True:
            raw = await ws.recv()
            event = json.loads(raw)
            event_type = event.get("type")
            print(f"\n[事件: {event_type}]")
            print(json.dumps(event, ensure_ascii=False, indent=2))
            if event_type in ("verification_done", "error"):
                break

        if event_type == "error":
            print("\n第一轮调研出错,没有论文列表可以追问,退出。")
            return

        print("\n========== 可以开始追问了,直接输入问题;输入 exit 退出 ==========")

        # 第二阶段: 用户输入一条,发一条,等回答;重复,直到用户输入 exit。
        while True:
            followup = input("\n追问: ").strip()
            if followup.lower() == "exit":
                break
            await ws.send(json.dumps({"question": followup}))

            while True:
                raw = await ws.recv()
                event = json.loads(raw)
                event_type = event.get("type")
                if event_type == "followup_thinking":
                    print("(思考中...)")
                    continue
                print(f"\n[事件: {event_type}]")
                print(json.dumps(event, ensure_ascii=False, indent=2))
                break


if __name__ == "__main__":
    asyncio.run(main())
