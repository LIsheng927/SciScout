"""验证 LocalModelClient 真的可以无缝替换 LLMClient——不改 PlannerAgent 一行代码,
只是构造的时候传入不同的对象。

运行(在 backend 目录下):
    python -m scripts.21_test_local_planner

这是整个"LoRA 微调支线"最后要证明的一件事:第16-20号脚本做的是"造出一个能用的
本地模型",这个脚本做的是"把它接进项目原有的架构里,而不是另起一套代码"。能这样
无缝替换,靠的是 llm_client.py 里定义的 ChatClient 这个 Protocol——LLMClient 和
LocalModelClient 都实现了同样签名的 .chat() 方法,PlannerAgent 自己不知道、也不需要
知道自己拿到的到底是线上 API 还是本地模型。
"""
from __future__ import annotations

from app.agents.planner import PlannerAgent
from app.core.local_model_client import LocalModelClient


def main() -> None:
    print("加载本地微调模型(底座 + LoRA adapter)...")
    local_client = LocalModelClient()

    # 关键的一行:PlannerAgent 的代码完全没变,只是这里传了个不一样的 llm 进去
    planner = PlannerAgent(llm=local_client)

    question = "多模态大模型在医学影像诊断中的应用有哪些最新进展?"
    print(f"\n问题: {question}\n")

    plan = planner.plan(question)
    print(f"拆解出 {len(plan.subtasks)} 个子任务(全程零 API 调用,零费用):")
    for t in plan.subtasks:
        print(f"  [{t.id}] {t.query}")
        print(f"      理由: {t.rationale}")

    print(f"\nLocalModelClient.call_count = {local_client.call_count}")


if __name__ == "__main__":
    main()
