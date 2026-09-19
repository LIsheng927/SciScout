"""把 finetune_planner.jsonl(question + 解析后的 subtasks)转换成真正能喂给
SFTTrainer 的"对话格式"训练数据。

为什么需要这一步:
LLM 微调本质上还是在学"看到这样的输入,应该续写出这样的输出"。GPT-4o-mini 这类
线上模型内部会自动把 system/user/assistant 三种角色的消息,按照它自己的模板拼成
一个大字符串再喂进模型——我们看不到这个拼接过程。但本地开源模型(Qwen 等)训练/
推理时必须由我们自己(或者训练框架)显式地把 messages 列表转换成模型认识的那个
带特殊 token 的字符串,这一步叫 chat template。

所以这里没有直接手写拼好的字符串,而是把每条数据整理成和 PlannerAgent.plan()
实际发给 LLM 的消息结构完全一样的 {"messages": [system, user, assistant]} 格式:
- system / user 两条消息,内容和 PlannerAgent 在线上调用 GPT-4o-mini 时发送的
  完全一致(直接从 app.agents.planner 里 import SYSTEM_PROMPT,而不是复制粘贴一份,
  避免以后 Planner 的 prompt 改了,训练数据却没跟着改,导致训练目标和真实推理目标不一致)
- assistant 消息的 content,就是当初 GPT-4o-mini 真实吐出来、我们解析成功的那个
  JSON 字符串本身——也就是"给这个 system+user,理想情况下模型应该输出什么"

后续训练脚本(下一步)会调用 tokenizer.apply_chat_template(messages, ...) 把这个
messages 列表转换成 Qwen2.5-Instruct 认识的字符串格式,而不需要我们手工拼接
<|im_start|>之类的特殊 token——这一步交给 tokenizer 自己做,不容易出错。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from app.agents.planner import SYSTEM_PROMPT

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SRC_PATH = DATA_DIR / "finetune_planner.jsonl"
TRAIN_PATH = DATA_DIR / "planner_sft_train.jsonl"
VAL_PATH = DATA_DIR / "planner_sft_val.jsonl"

VAL_RATIO = 0.08  # 306 条 -> 大约 24 条留作验证/评测,其余用于训练
SEED = 42


def load_records() -> list[dict]:
    records = []
    with open(SRC_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_chat_example(record: dict) -> dict:
    question = record["question"]
    subtasks = record["subtasks"]

    assistant_content = json.dumps({"subtasks": subtasks}, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"研究问题:{question}"},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main() -> None:
    records = load_records()
    print(f"读取到 {len(records)} 条原始数据: {SRC_PATH}")

    examples = [to_chat_example(r) for r in records]

    rng = random.Random(SEED)
    rng.shuffle(examples)

    n_val = max(1, round(len(examples) * VAL_RATIO))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(VAL_PATH, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"训练集: {len(train_examples)} 条 -> {TRAIN_PATH}")
    print(f"验证集: {len(val_examples)} 条 -> {VAL_PATH}")
    print("\n示例(训练集第1条):")
    print(json.dumps(train_examples[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
