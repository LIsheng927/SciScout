"""用 PlannerAgent(背后是 GPT-4o-mini)批量生成"问题 -> 拆解结果"的样本对,
存成 JSONL 格式,作为后续 LoRA 微调的训练数据。

这是"知识蒸馏"的第一步:用一个聪明但贵的大模型,批量产出高质量的示范样本,
再拿这些样本去训练一个小模型模仿它的行为。

注意: 问题特意覆盖了好几个不同的研究领域(不只是LoRA+联邦学习那一个窄话题),
这样训练出来的小模型才能学到"怎么拆解一个研究问题"这个通用能力,
而不是死记硬背某个特定领域的套路。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.planner import PlannerAgent

# 覆盖多个领域的研究问题,数量先控制在40个左右,快速跑通"数据生成->后续训练"
# 这条流水线;跑通之后如果发现效果不错,再考虑要不要扩大数量。
QUESTIONS_POOL_PATH = Path(__file__).resolve().parent.parent / "data" / "questions_pool.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "finetune_planner.jsonl"

# 保留几个手写的问题做兜底(比如还没跑过 17 号脚本生成问题池的时候,
# 这个脚本依然能跑,不会因为文件不存在直接报错退出)。
FALLBACK_QUESTIONS = [
    "LoRA 微调在联邦学习场景下有哪些最新进展",
    "扩散模型在图像生成中的最新进展是什么",
    "大语言模型的幻觉问题有哪些缓解方法",
]


def load_questions() -> list[str]:
    if QUESTIONS_POOL_PATH.exists():
        with open(QUESTIONS_POOL_PATH, "r", encoding="utf-8") as f:
            questions = json.load(f)
        print(f"从问题池加载了 {len(questions)} 个问题: {QUESTIONS_POOL_PATH}")
        return questions
    print(f"未找到问题池文件 {QUESTIONS_POOL_PATH},使用内置的少量兜底问题")
    return FALLBACK_QUESTIONS


def main() -> None:
    questions = load_questions()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    success, failed = 0, 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, question in enumerate(questions, start=1):
            # 每道题都新建一个 PlannerAgent(内部会新建一个 LLMClient),
            # 这样每个 LLMClient 自己的调用计数器都是从0开始,
            # 不会因为跑了40道题就触发 LLMBudgetExceeded。
            planner = PlannerAgent()
            try:
                plan = planner.plan(question)
            except Exception as exc:  # noqa: BLE001 - 单条失败不影响整体收集
                print(f"[{i}/{len(questions)}] 失败: {question}  ({exc})")
                failed += 1
                continue

            record = {
                "question": plan.original_question,
                "subtasks": [t.model_dump() for t in plan.subtasks],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            success += 1
            print(f"[{i}/{len(questions)}] 完成: {question} -> {len(plan.subtasks)} 个子任务")

            time.sleep(0.5)  # 给API一点缓冲,避免短时间内请求太密集触发限流

    print(f"\n全部完成: 成功 {success} 条, 失败 {failed} 条")
    print(f"数据已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
