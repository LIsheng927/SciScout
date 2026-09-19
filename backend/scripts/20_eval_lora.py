"""评测脚本:拿训练时没见过的问题,同时跑一遍"微调后的本地小模型"和"线上 GPT-4o-mini
(PlannerAgent 原实现)",把两边的任务拆解结果摆在一起对比,直观看差距在哪。

运行(在 backend 目录下):
    python -m scripts.20_eval_lora

=== 这里在做什么 ===

1) 加载本地模型的方式和训练时不一样:
   训练时是"底座模型 + LoraConfig -> get_peft_model()"现场挂载。
   评测/以后线上用的时候,是"底座模型 + 已经训练好、存在磁盘上的 adapter"这样加载:
   AutoModelForCausalLM 先加载原始 Qwen2.5-1.5B,再用
   PeftModel.from_pretrained(base_model, adapter_path) 把训练好的 LoRA 权重"接"上去。
   这就是 LoRA 最大的实用优势:adapter 文件很小,底座模型只需要下载/保存一份,
   不同任务的 adapter 可以像"插件"一样按需换着用。

2) 生成参数为什么用贪心解码(do_sample=False):
   Planner 这个任务要求输出严格的 JSON,不需要"创造性"——我们想要的是模型每次都
   给出它认为概率最高的那个答案,方便公平地比较训练效果,而不是像聊天场景那样需要
   一些随机性让回答更有变化。这和之前 CitationVerifier 用 temperature=0 的思路是
   一致的。

3) 怎么算"及格":
   先看输出是不是合法 JSON、字段是否符合 SubTask schema(能不能被 Pydantic 解析)
   ——这是最基本的"格式对不对"。格式之上,再人工/主观地看子任务内容的质量
   (是否覆盖了问题的不同侧面、query 是否可检索)——这部分暂时没做自动化打分,
   现阶段用肉眼对比,后续如果要更严谨,可以再引入一个 LLM-as-judge 来打分。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.agents.planner import PlannerAgent, SYSTEM_PROMPT
from app.agents.schemas import SubTask

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "models" / "planner-lora"
VAL_PATH = Path(__file__).resolve().parent.parent / "data" / "planner_sft_val.jsonl"

# 从验证集里挑几个问题(训练时用 assistant_only_loss,这些问题本身模型没直接学过
# "该输出什么",只在 eval 阶段算过 loss,没有被用来更新参数)
N_EXAMPLES = 5


def load_eval_questions(n: int) -> list[str]:
    questions = []
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            # messages[1] 是 user 消息,内容形如 "研究问题:xxx"
            user_content = record["messages"][1]["content"]
            question = user_content.removeprefix("研究问题:")
            questions.append(question)
            if len(questions) >= n:
                break
    return questions


def load_local_model():
    print(f"加载底座模型 {BASE_MODEL} + LoRA adapter {ADAPTER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    return tokenizer, model


def run_local_model(tokenizer, model, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"研究问题:{question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # 贪心解码,方便公平对比,原因见文件头注释
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只取新生成的部分,去掉输入的 prompt
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def try_parse(raw: str) -> tuple[bool, str]:
    """尝试把模型输出解析成合法的 subtasks 列表,返回(是否成功, 说明文字)。"""
    try:
        data = json.loads(raw)
        subtasks = [SubTask(**item) for item in data["subtasks"]]
        return True, f"合法,{len(subtasks)} 个子任务"
    except (json.JSONDecodeError, KeyError, ValidationError, TypeError) as exc:
        return False, f"解析失败: {exc}"


def main() -> None:
    questions = load_eval_questions(N_EXAMPLES)
    print(f"从验证集抽取 {len(questions)} 个问题做对比评测\n")

    tokenizer, local_model = load_local_model()
    gpt_planner = PlannerAgent()

    for i, question in enumerate(questions, start=1):
        print(f"{'=' * 70}\n[{i}] 问题: {question}\n{'=' * 70}")

        local_raw = run_local_model(tokenizer, local_model, question)
        local_ok, local_msg = try_parse(local_raw)
        print(f"\n--- 本地微调模型 (格式: {local_msg}) ---")
        print(local_raw)

        try:
            gpt_plan = gpt_planner.plan(question)
            print(f"\n--- GPT-4o-mini (PlannerAgent,格式:合法,{len(gpt_plan.subtasks)} 个子任务) ---")
            for t in gpt_plan.subtasks:
                print(f"  [{t.id}] {t.query}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n--- GPT-4o-mini 调用失败: {exc} ---")

        print()


if __name__ == "__main__":
    main()
