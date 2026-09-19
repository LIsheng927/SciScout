"""FastAPI 应用入口。用 `uvicorn app.main:app --reload` 启动。"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.research_ws import router as research_router

app = FastAPI(title="SciScout API")

# CORS: 允许前端(跑在另一个端口上,比如 React 开发服务器的 5173/3000)
# 跨源访问这个后端。开发阶段先全部放开,方便调试,后续部署时应该收紧成
# 只允许真正的前端域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research_router)


@app.get("/health")
def health() -> dict:
    """最简单的健康检查接口,用来确认服务是不是跑起来了。"""
    return {"status": "ok"}
