"""测试 WebSocket 接口:模拟前端连接后端,发一个问题过去,
实时打印后端推送回来的每一条进度事件(还没有前端界面,先用这个脚本
验证"后端能不能正确地边跑边推送消息"这件事本身有没有问题)。

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
        async for raw in ws:
            event = json.loads(raw)
            event_type = event.get("type")
            print(f"\n[事件: {event_type}]")
            print(json.dumps(event, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
