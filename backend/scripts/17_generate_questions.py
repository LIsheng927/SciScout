"""用 LLM 批量生成多样化的研究问题,覆盖不同领域,为后面的训练数据生成
(scripts/16)提供更大规模的问题池——40个问题手写还行,300个手写就不现实了,
所以让 LLM 自己去想"某个领域里,人们会问哪些研究问题"。

做法: 列一批AI/CS的子领域,针对每个领域单独问一次LLM"给我20个这个领域的
研究问题",分开问而不是一次性问300个,是为了避免"一次性要求生成太多"时
LLM容易开始重复、敷衍(这是实际用LLM做批量生成时的一个常见坑)。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm_client import LLMClient

DOMAINS = [
    "自然语言处理与大语言模型",
    "计算机视觉",
    "强化学习",
    "图神经网络与知识图谱",
    "多模态学习",
    "AI 安全与对齐",
    "联邦学习与隐私计算",
    "生成式模型(扩散模型/GAN)",
    "推荐系统",
    "语音处理",
    "AI for Science(生物/化学/气候)",
    "机器人学与具身智能",
    "模型压缩与推理加速",
    "分布式机器学习系统",
    "时序数据与预测",
]

QUESTIONS_PER_DOMAIN = 20

SYSTEM_PROMPT = """你是一个AI/计算机科学领域的研究选题助手。给定一个研究方向,
生成{n}个具体、有区分度的研究问题,这些问题应该像是一个准备做文献调研的人
会问的问题(即"这个领域最近有什么进展/有哪些方法/面临什么挑战"这种类型)。

要求:
1. 问题之间要有明显区分度,不要只是换几个字的重复表述
2. 用中文提问
3. 严格用JSON格式输出: {{"questions": ["问题1", "问题2", ...]}}
"""

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "questions_pool.json"


def main() -> None:
    all_questions: list[str] = []

    for i, domain in enumerate(DOMAINS, start=1):
        llm = LLMClient()  # 每个领域用一个新的LLMClient,避免累积触发调用预算上限
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(n=QUESTIONS_PER_DOMAIN)},
            {"role": "user", "content": f"研究方向: {domain}"},
        ]
        try:
            raw = llm.chat(messages, temperature=0.8, response_format_json=True)
            data = json.loads(raw)
            questions = data["questions"]
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(DOMAINS)}] {domain} 生成失败: {exc}")
            continue

        all_questions.extend(questions)
        print(f"[{i}/{len(DOMAINS)}] {domain}: 生成 {len(questions)} 个问题")
        time.sleep(0.5)

    # 去重(理论上不同领域生成的问题不会撞,但保险起见去一下重)
    unique_questions = list(dict.fromkeys(all_questions))

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_questions, f, ensure_ascii=False, indent=2)

    print(f"\n共生成 {len(unique_questions)} 个不重复的问题,已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
