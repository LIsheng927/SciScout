"""LoRA 微调 Planner 任务:在 Qwen2.5-1.5B-Instruct 上,用我们自己生成的
282条(question -> subtasks JSON)数据,训练一个能在本地跑的"任务分解"小模型。

跑之前需要装训练专用依赖(见 backend/requirements-train.txt):
    pip install torch --index-url https://download.pytorch.org/whl/cu128   # 按你的 CUDA 版本换
    pip install -r requirements-train.txt

运行(在 backend 目录下):
    python -m scripts.19_train_lora

=== 这个脚本在干什么,分几块讲 ===

1) 为什么只想对 assistant 部分算 loss(assistant_only_loss)?
   数据集里每条样本是 {"messages": [system, user, assistant]} 三段对话。
   如果把这一整段文本都丢给模型学,模型会"每一个 token 都学着去预测",包括
   system prompt 和 user 问题本身——但我们根本不需要模型学会"怎么把问题复述一遍",
   只想让它学会"看到这个 system+user,该输出什么样的 assistant JSON"。
   新版 trl 把这件事直接内置到了 SFTConfig 里:assistant_only_loss=True 之后,
   trainer 会自动识别 messages 里哪些 token 属于 assistant 回合,只在那部分
   计算 loss、反向传播,system/user 部分的 token 完全不参与训练信号。
   (更早期的 trl 版本要手动用 DataCollatorForCompletionOnlyLM 去找"assistant 回合
   开始的特殊标记"来做这件事,新版本把这一步自动化、和模型自带的 chat template
   绑定得更紧,不容易配错。)

2) chat template 谁来套?
   数据集里每条样本已经是 {"messages": [...]} 格式(和 PlannerAgent 实际调用 LLM
   时发送的消息结构完全一样)。trl 的 SFTTrainer 发现数据集有 messages 列,会自动
   调用模型自己的 tokenizer.chat_template 把它转换成带特殊 token 的文本,不需要
   我们手工拼接。

3) LoRA 具体怎么"挂"到模型上?
   LoraConfig 里的 target_modules 指定"在哪些权重矩阵上加旁路"。Qwen2.5 这种
   Transformer 结构里,每一层 attention 有 q_proj/k_proj/v_proj/o_proj 四个矩阵,
   feed-forward 部分有 gate_proj/up_proj/down_proj 三个矩阵——LoRA 通常挂在这 7 个
   地方,r=16 表示旁路矩阵的秩(rank),越大表达能力越强但参数也越多;alpha=32
   是缩放系数,常见经验值是 alpha = 2 * r。get_peft_model() 之后,原模型的参数会
   被冻结(requires_grad=False),只有新插入的 LoRA 旁路参数是可训练的——
   print_trainable_parameters() 会直接告诉你可训练参数占比,通常 1% 都不到。

4) 训练完之后产出什么?
   不是一整个新模型,而是一个几十 MB 的 adapter 文件夹(只存 LoRA 的 A/B 矩阵),
   保存在 backend/models/planner-lora/。以后加载的时候是"原始 Qwen2.5-1.5B 底座 +
   这个 adapter"组合着用,不需要重新下载/占用一份完整模型的磁盘空间。
"""
from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_PATH = DATA_DIR / "planner_sft_train.jsonl"
VAL_PATH = DATA_DIR / "planner_sft_val.jsonl"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "planner-lora"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")
    if device == "cpu":
        print("!! 没检测到可用 GPU,CPU 上跑 1.5B 模型的微调会非常慢,建议先排查 CUDA 环境 !!")

    print(f"加载底座模型: {BASE_MODEL} (第一次会从 HuggingFace 下载,存到本机缓存)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype, device_map=device
    )

    print("加载训练/验证数据(messages 格式,交给 SFTTrainer 自动套 chat template)")
    dataset = load_dataset(
        "json",
        data_files={"train": str(TRAIN_PATH), "validation": str(VAL_PATH)},
    )
    print("样例(原始 messages,未套模板):")
    print(dataset["train"][0]["messages"])

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=(device == "cuda"),
        report_to=[],
        max_length=1024,
        packing=False,
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\n训练完成,LoRA adapter 已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
