"""LoRA微调Planner的量化评测benchmark——把20号脚本那种"肉眼对比"升级成有具体
数字支撑的报告,可以直接写进简历/面试话术里。

运行(在 backend 目录下):
    python -m scripts.22_eval_benchmark

=== 设计思路 ===

1) 标准答案(reference)不用重新花钱调用GPT-4o-mini拿——用
   `data/planner_sft_val.jsonl` 里现成的24条验证集。这24条里的 assistant
   消息,就是当初生成训练数据时GPT-4o-mini真实输出的拆解结果,而且因为训练时
   做了train/val切分,这些数据从没被用来更新过模型权重,可以放心当"标准答案"用。
   评测阶段能省的API调用尽量省,这是个成本意识上的工程习惯。

2) 拆得"好不好"怎么量化——用embedding算语义相似度。
   把本地模型输出的每个子任务query,和GPT-4o-mini(标准答案)的每个子任务query
   都转成向量,两两算余弦相似度;本地模型的每个子任务,在标准答案里找相似度最高
   的那个当"最佳匹配",取这个分数;一道题里所有子任务的最佳匹配分数取平均,就是
   这道题的对齐分数(alignment score);24道题再取平均,就是模型整体得分。
   分数越接近1,说明本地模型的拆解思路跟GPT-4o-mini越接近。

3) 另外两个更直接的硬指标:格式合法率(JSON能不能解析成SubTask列表)、
   平均子任务数是否落在Planner prompt要求的3-5个区间内。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from app.agents.schemas import SubTask
from app.core.llm_client import LLMClient
from app.core.local_model_client import LocalModelClient


def cjk_ratio(text: str) -> float:
    """一段文字里,中日韩(CJK)字符占的比例。用来粗略判断这句话是"中文关键词风格"
    还是"英文完整短语风格"——不需要真的做语言检测,字符编码范围就够用了:
    中文汉字在Unicode里落在 U+4E00-U+9FFF 这个区间。
    """
    if not text:
        return 0.0
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk_count / len(text)

VAL_PATH = Path(__file__).resolve().parent.parent / "data" / "planner_sft_val.jsonl"
RESULT_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark_results.json"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def load_val_examples() -> list[dict]:
    examples = []
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            messages = record["messages"]
            question = messages[1]["content"].removeprefix("研究问题:")
            teacher_subtasks = json.loads(messages[2]["content"])["subtasks"]
            examples.append({
                "question": question,
                "system": messages[0]["content"],
                "teacher_subtasks": teacher_subtasks,
            })
    return examples


def parse_local_output(raw: str) -> list[dict] | None:
    try:
        data = json.loads(raw)
        subtasks = [SubTask(**item).model_dump() for item in data["subtasks"]]
        return subtasks
    except Exception:  # noqa: BLE001
        return None


def alignment_score(local_subtasks: list[dict], teacher_subtasks: list[dict], embedder: LLMClient) -> float:
    """本地模型每个子任务query,在teacher子任务里找最相似的,取平均相似度。"""
    teacher_vecs = [embedder.embed(t["query"]) for t in teacher_subtasks]
    local_vecs = [embedder.embed(t["query"]) for t in local_subtasks]

    best_matches = []
    for lv in local_vecs:
        sims = [cosine_similarity(lv, tv) for tv in teacher_vecs]
        best_matches.append(max(sims))
    return sum(best_matches) / len(best_matches)


def main() -> None:
    examples = load_val_examples()
    print(f"加载 {len(examples)} 条held-out验证集(标准答案来自已记录的GPT-4o-mini输出)\n")

    print("加载本地微调模型...")
    local_client = LocalModelClient()
    embedder = LLMClient()  # 只用它的 embed() 方法,不产生chat调用费用

    per_example_results = []

    for i, ex in enumerate(examples, start=1):
        messages = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": f"研究问题:{ex['question']}"},
        ]
        start = time.time()
        raw = local_client.chat(messages, temperature=0.0)
        latency = time.time() - start

        local_subtasks = parse_local_output(raw)
        format_valid = local_subtasks is not None

        local_queries = [t["query"] for t in local_subtasks] if local_subtasks else []
        teacher_queries = [t["query"] for t in ex["teacher_subtasks"]]

        # 疑似"跨语言/风格不一致"检测:本地模型习惯输出英文完整短语,如果teacher
        # 这道题恰好写成了中文关键词堆砌风格,两边字符层面差异就会很大,
        # embedding相似度会被拉低,但不代表语义角度真的没覆盖到——这种样本要单独标记,
        # 不能直接算进"模型质量差"的证据里。
        local_cjk = sum(cjk_ratio(q) for q in local_queries) / len(local_queries) if local_queries else 0.0
        teacher_cjk = sum(cjk_ratio(q) for q in teacher_queries) / len(teacher_queries) if teacher_queries else 0.0
        style_mismatch = (teacher_cjk - local_cjk) > 0.15  # teacher明显更"中文关键词化"

        result = {
            "question": ex["question"],
            "format_valid": format_valid,
            "subtask_count": len(local_subtasks) if local_subtasks else 0,
            "count_in_range": format_valid and 3 <= len(local_subtasks) <= 5,
            "latency_sec": round(latency, 2),
            "alignment_score": None,
            "style_mismatch_suspected": style_mismatch,
            # 把双方原始子任务也存下来,不然分数出了异常没法debug,只能干瞪眼
            "local_subtasks": local_queries,
            "teacher_subtasks": teacher_queries,
            "raw_output": raw if not format_valid else None,
        }

        if format_valid:
            result["alignment_score"] = round(
                alignment_score(local_subtasks, ex["teacher_subtasks"], embedder), 4
            )

        per_example_results.append(result)
        status = "OK" if format_valid else "解析失败"
        score_str = f"align={result['alignment_score']}" if result["alignment_score"] is not None else ""
        flag_str = " [疑似风格不一致]" if result["style_mismatch_suspected"] else ""
        print(f"[{i}/{len(examples)}] {status} | 子任务数={result['subtask_count']} | "
              f"耗时={result['latency_sec']}s | {score_str}{flag_str}")

    # 汇总统计
    n = len(per_example_results)
    format_valid_rate = sum(r["format_valid"] for r in per_example_results) / n
    count_in_range_rate = sum(r["count_in_range"] for r in per_example_results) / n
    avg_latency = sum(r["latency_sec"] for r in per_example_results) / n
    valid_scores = [r["alignment_score"] for r in per_example_results if r["alignment_score"] is not None]
    avg_alignment = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    n_flagged = sum(r["style_mismatch_suspected"] for r in per_example_results)
    clean_scores = [
        r["alignment_score"] for r in per_example_results
        if r["alignment_score"] is not None and not r["style_mismatch_suspected"]
    ]
    avg_alignment_clean = sum(clean_scores) / len(clean_scores) if clean_scores else 0.0

    summary = {
        "n_examples": n,
        "format_valid_rate": round(format_valid_rate, 4),
        "subtask_count_in_range_rate": round(count_in_range_rate, 4),
        "avg_alignment_score_all": round(avg_alignment, 4),
        "n_style_mismatch_flagged": n_flagged,
        "avg_alignment_score_excl_style_mismatch": round(avg_alignment_clean, 4),
        "avg_latency_sec": round(avg_latency, 3),
        "local_model_call_count": local_client.call_count,
        "estimated_cost_usd": 0.0,  # 本地推理,不计费(相对于GPT-4o-mini每次调用的费用)
    }

    print("\n" + "=" * 50)
    print("汇总结果:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_example": per_example_results}, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存到: {RESULT_PATH}")


if __name__ == "__main__":
    main()
