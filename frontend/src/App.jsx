import { useRef, useState } from "react";
import { marked } from "marked";
import "./App.css";

const EVENT_LABELS = {
  plan_created: "问题拆解完成",
  search_done: "检索完成",
  reflection_done: "反思完成",
  generating_report: "正在生成报告...",
  report_ready: "报告生成完成",
  verifying_citations: "正在校验引用...",
  verification_done: "引用校验完成",
  error: "出错了",
};

function describeEvent(event) {
  switch (event.type) {
    case "plan_created":
      return `拆解出 ${event.subtasks.length} 个子任务`;
    case "search_done":
      return `子任务 [${event.subtask_id}] 搜到 ${event.found} 篇,新增 ${event.new} 篇`;
    case "reflection_done":
      return `第 ${event.round} 轮反思: ${event.is_sufficient ? "信息足够" : "信息不够,准备补充检索"}`;
    case "verification_done": {
      const supportedCount = event.results.filter((r) => r.supported).length;
      return `${event.results.length} 条引用里,${supportedCount} 条通过校验`;
    }
    case "error":
      return event.message;
    default:
      return "";
  }
}

// 给每种事件配一个"点"的颜色,让时间线一眼能看出"现在大概在哪个阶段"。
// reflection_done 比较特殊:同一种事件类型,根据 is_sufficient 的值显示不同颜色
// (绿色=够了,黄色=还要继续),所以单独判断一下,不能简单查表。
function dotClass(event) {
  if (event.type === "reflection_done") {
    return event.is_sufficient ? "dot dot-reflection-ok" : "dot dot-reflection-more";
  }
  const map = {
    plan_created: "dot-plan",
    search_done: "dot-search",
    generating_report: "dot-report",
    report_ready: "dot-report",
    verifying_citations: "dot-verify",
    verification_done: "dot-verify",
    error: "dot-error",
  };
  return `dot ${map[event.type] ?? ""}`;
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [events, setEvents] = useState([]);
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState(null);
  const [verification, setVerification] = useState(null);

  // 追问相关的状态:
  // canFollowup: 第一轮 pipeline 跑完(verification_done)之后才允许追问,
  //   之前追问框都是禁用的——因为后端在跑完第一轮之前根本不会进入"追问模式"
  //   的 receive_text() 循环,这时候发消息过去后端根本收不到。
  // conversation: 追问的问答历史,每一项是 {question, answer}——answer
  //   一开始是 null(还没收到回复,用来渲染"思考中..."那一条),收到
  //   followup_answer 事件之后再填上真正的回答文本。
  // wsRef: 用 useRef 而不是 useState 存 WebSocket 实例——因为发追问的时候
  //   要立刻同步地调用 ws.send(),不需要、也不应该因为存了这个值而触发
  //   页面重新渲染(WebSocket 对象本身不是"要显示在界面上的数据")。
  const wsRef = useRef(null);
  const [canFollowup, setCanFollowup] = useState(false);
  const [conversation, setConversation] = useState([]);
  const [followupInput, setFollowupInput] = useState("");
  const [followupLoading, setFollowupLoading] = useState(false);

  function startResearch() {
    if (!question.trim() || running) return;

    // 如果上一次调研还开着一个追问连接,先关掉它,不然会同时存在两条连接,
    // 追问答案不知道该显示在哪一轮结果下面,状态会乱掉。
    if (wsRef.current) {
      wsRef.current.close();
    }

    setEvents([]);
    setReport(null);
    setVerification(null);
    setRunning(true);
    setCanFollowup(false);
    setConversation([]);
    setFollowupInput("");
    setFollowupLoading(false);

    const ws = new WebSocket("ws://127.0.0.1:8000/ws/research");
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ question }));
    };

    ws.onmessage = (message) => {
      const event = JSON.parse(message.data);

      // 追问阶段的事件,和第一轮 pipeline 的事件分开处理,不往 events 时间线里塞——
      // 追问是另一种交互,混进"执行过程"时间线里反而会让人看不懂。
      if (event.type === "followup_thinking") {
        setFollowupLoading(true);
        return;
      }
      if (event.type === "followup_answer") {
        setFollowupLoading(false);
        setConversation((prev) => {
          // 把最后一条"还没收到回复"(answer === null)的记录填上答案;
          // 用 map 而不是直接改数组,是 React 里"不要直接修改 state,
          // 而是每次产出一个新的数组/对象"这个规则的具体体现。
          const lastIdx = prev.length - 1;
          return prev.map((item, i) =>
            i === lastIdx && item.answer === null
              ? { ...item, answer: event.answer }
              : item
          );
        });
        return;
      }

      setEvents((prev) => [...prev, event]);

      if (event.type === "report_ready") {
        setReport(event);
      }
      if (event.type === "verification_done") {
        setVerification(event.results);
        setCanFollowup(true);
      }
      if (event.type === "error") {
        setFollowupLoading(false);
      }
    };

    ws.onclose = () => {
      setRunning(false);
      setCanFollowup(false);
    };

    ws.onerror = () => {
      setEvents((prev) => [...prev, { type: "error", message: "WebSocket 连接出错" }]);
      setRunning(false);
    };
  }

  function sendFollowup() {
    const q = followupInput.trim();
    if (!q || followupLoading || !canFollowup || !wsRef.current) return;

    wsRef.current.send(JSON.stringify({ question: q }));
    setConversation((prev) => [...prev, { question: q, answer: null }]);
    setFollowupInput("");
  }

  return (
    <div className="page">
      <div className="header">
        <h1>SciScout</h1>
        <p>科研文献 Deep Research 多智能体系统</p>
      </div>

      <div className="input-row">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && startResearch()}
          placeholder="输入你的研究问题,比如: LoRA 微调在联邦学习场景下有哪些最新进展"
        />
        <button onClick={startResearch} disabled={running}>
          {running && <span className="spinner" />}
          {running ? "调研中..." : "开始调研"}
        </button>
      </div>

      {events.length > 0 && (
        <div className="card">
          <h2>执行过程</h2>
          <ul className="timeline">
            {events.map((event, i) => (
              <li key={i}>
                <span className={dotClass(event)} />
                <div>
                  <span className="label">{EVENT_LABELS[event.type] ?? event.type}</span>
                  <span className="desc">{describeEvent(event)}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {report && (
        <div className="card">
          <h2>报告</h2>
          {/* marked.parse() 把 markdown 文本转换成一段 HTML 字符串;
              dangerouslySetInnerHTML 是 React 里"直接把一段HTML字符串塞进页面"
              的写法——名字里特意带"dangerously"(危险地),是在提醒你:
              如果这段HTML是不可信来源(比如网站用户自己输入的内容),
              直接这样塞进去有被注入恶意脚本的风险(XSS攻击)。
              这里之所以能接受,是因为这段内容来自我们自己的后端LLM生成结果,
              不是任意用户能直接控制的输入——但这是个值得记住的安全边界,
              换成别的场景(比如显示用户上传的内容)就不能这么写了。 */}
          <div
            className="report-body"
            dangerouslySetInnerHTML={{ __html: marked.parse(report.content) }}
          />
        </div>
      )}

      {report && (
        <div className="card">
          <h2>引用来源</h2>
          <ul className="citation-list">
            {report.citations.map((c) => {
              const v = verification?.find((r) => r.ref_id === c.ref_id);
              return (
                <li key={c.ref_id}>
                  <span>[{c.ref_id}]</span>
                  <a href={c.url} target="_blank" rel="noreferrer">
                    {c.title}
                  </a>
                  {v && (
                    <span className={`badge ${v.supported ? "badge-ok" : "badge-fail"}`}>
                      {v.supported ? "校验通过" : "校验未通过"}
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* 报告出来、引用校验也跑完之后才显示追问区——呼应后端 canFollowup
          的时机:后端也是同一时刻(verification_done)才真正进入追问模式。 */}
      {canFollowup && (
        <div className="card">
          <h2>追问</h2>

          {conversation.length > 0 && (
            <ul className="followup-list">
              {conversation.map((turn, i) => (
                <li key={i} className="followup-item">
                  <div className="followup-question">{turn.question}</div>
                  {turn.answer === null ? (
                    <div className="followup-answer followup-thinking">
                      <span className="spinner spinner-dark" />
                      思考中...
                    </div>
                  ) : (
                    <div
                      className="followup-answer"
                      dangerouslySetInnerHTML={{ __html: marked.parse(turn.answer) }}
                    />
                  )}
                </li>
              ))}
            </ul>
          )}

          <div className="input-row">
            <input
              type="text"
              value={followupInput}
              onChange={(e) => setFollowupInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendFollowup()}
              placeholder="针对上面的报告继续追问,比如: 第二篇论文具体怎么做的"
              disabled={followupLoading}
            />
            <button onClick={sendFollowup} disabled={followupLoading || !followupInput.trim()}>
              发送
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
