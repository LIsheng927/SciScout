"""统一 LLM 调用层。

设计目的(面试可以讲的点):
1. 通过 base_url 切换即可在 OpenAI / 通义千问(DashScope兼容模式) / DeepSeek 之间切换,不改动上层 agent 代码
2. 内置调用计数器 + 硬上限,防止 agent 在死循环/多轮反思时意外打爆 API 余额
3. 内置简单的内存缓存(按 prompt 内容做 key),同一次进程内重复问题不重复付费调用
4. 之后接入本地 LoRA 微调模型时,只需新增一个 LocalModelClient 实现同样的 .chat() 接口即可无缝替换
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from openai import OpenAI

from app.core.config import get_settings

logger = logging.getLogger("scires.llm")


@runtime_checkable
class ChatClient(Protocol):
    """PlannerAgent(以及其它 agent)真正依赖的接口,不是某个具体类。

    这是"面向接口编程"的一个具体例子:PlannerAgent 只在乎"给我一个有 .chat() 方法、
    签名长这样的对象",不关心这个对象内部是调 OpenAI 的 HTTP 接口,还是在本机 GPU 上
    跑一个 LoRA 微调过的小模型。Python 是动态类型语言,其实不写这个 Protocol,
    LLMClient 和 LocalModelClient "刚好都有 .chat() 方法"也能直接互换用(这叫鸭子类型,
    duck typing:一个对象只要长得像鸭子、叫起来像鸭子,就当它是鸭子,不看它的血统)。
    但显式声明一个 Protocol,能让类型检查器(比如 mypy)和 IDE 帮你检查"新写的
    LocalModelClient 是不是真的实现对了这个接口",而不是等到运行时才因为少传了个
    参数报错。
    """

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        response_format_json: bool = False,
    ) -> str: ...


class LLMBudgetExceeded(RuntimeError):
    """单次运行的 LLM 调用次数超过了配置的上限。"""


@dataclass
class LLMClient:
    call_count: int = field(default=0, init=False)
    _cache: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key or "no-key-set",
            base_url=settings.llm_base_url or None,
        )

    def _cache_key(self, messages: list[dict], temperature: float) -> str:
        raw = json.dumps(messages, ensure_ascii=False, sort_keys=True) + f"|t={temperature}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        response_format_json: bool = False,
    ) -> str:
        """发起一次对话补全,返回纯文本内容。带缓存与预算保护。"""
        key = self._cache_key(messages, temperature)
        if key in self._cache:
            logger.info("LLM cache hit, 跳过真实调用")
            return self._cache[key]

        if self.call_count >= self._settings.max_llm_calls_per_run:
            raise LLMBudgetExceeded(
                f"已达到单次运行的 LLM 调用上限 ({self._settings.max_llm_calls_per_run} 次)。"
                " 如需继续测试请提高 MAX_LLM_CALLS_PER_RUN 或确认是否出现了意外循环。"
            )

        kwargs: dict = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}

        self.call_count += 1
        logger.info("LLM call #%d, model=%s", self.call_count, self._settings.llm_model)
        completion = self._client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content or ""
        self._cache[key] = content
        return content

    def embed(self, text: str) -> list[float]:
        """把一段文字转换成向量。跟 chat() 一样是对外公开的正式接口,
        调用方(比如 RagTool)不需要、也不应该知道内部用的是 self._client 这个细节。
        """
        settings = self._settings
        response = self._client.embeddings.create(model=settings.embedding_model, input=text)
        return response.data[0].embedding
