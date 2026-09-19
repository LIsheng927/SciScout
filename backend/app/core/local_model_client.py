"""本地 LoRA 微调模型的调用层,实现和 LLMClient 一样的 .chat() 接口
(见 llm_client.py 里的 ChatClient Protocol),这样 PlannerAgent 完全不用改代码,
只需要在构造时传一个 LocalModelClient() 而不是 LLMClient() 进去。

跟线上 LLMClient 最大的不同:
1. LLMClient 每次 .chat() 都是一次网络请求,模型在 OpenAI 的服务器上;
   LocalModelClient 是把模型(底座 + LoRA adapter)一次性加载进本机 GPU 显存,
   之后每次 .chat() 都是本地推理,不联网、不花钱,但要占着显存。
2. 所以 LocalModelClient 应该被"复用"——只创建一次、常驻,而不是像 LLMClient
   那样每次都 new 一个(参考 16 号脚本里为什么要给每个问题都 new 一个新的
   PlannerAgent/LLMClient:是为了绕开单进程调用次数上限,而不是真的推荐这么用)。
   如果每次调用 chat() 都重新加载模型,光加载模型本身就要好几秒,比真正的推理还慢。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "planner-lora"


@dataclass
class LocalModelClient:
    base_model_name: str = DEFAULT_BASE_MODEL
    adapter_dir: Path = DEFAULT_ADAPTER_DIR
    call_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        self._model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
        self._model.eval()

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        response_format_json: bool = False,
    ) -> str:
        """和 LLMClient.chat() 完全一样的签名。

        response_format_json 这个参数在 LLMClient 里是让 OpenAI API 强制返回合法
        JSON;本地模型没有这种服务端能力,这里保留这个参数只是为了接口一致
        (调用方不需要知道/关心两边的实现差异),实际靠的是训练数据本身已经让模型
        学会了"稳定输出 JSON"这个习惯——第20号脚本的评测已经验证过这一点。
        """
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        # temperature=0 在 OpenAI 那边意味着"尽量确定性输出";本地生成对应的做法
        # 是直接关掉采样,用贪心解码(do_sample=False),效果类似但不是同一套机制。
        do_sample = temperature > 0
        gen_kwargs: dict = {
            "max_new_tokens": 512,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        self.call_count += 1
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
