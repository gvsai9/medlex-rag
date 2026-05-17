import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Brain,
  Database,
  FileText,
  Network,
  Send,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Loader2,
  Server,
  Pill,
  Settings2,
  Copy,
  CheckCircle2,
  AlertTriangle
} from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";

const DEFAULT_DRUGS = [
  "ibuprofen",
  "aspirin",
  "metformin",
  "lisinopril",
  "atorvastatin",
  "omeprazole",
  "amoxicillin",
  "levothyroxine",
  "acetaminophen",
  "naproxen"
];

const SAMPLE_QUESTIONS = [
  "What are the warnings for ibuprofen?",
  "What should I ask a doctor before using ibuprofen?",
  "What are the stomach bleeding warnings for ibuprofen?",
  "What is the dosage for ibuprofen?"
];

function classNames(...classes) {
  return classes.filter(Boolean).join(" ");
}

function CitationCard({ citation }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-blue-700">
            <FileText className="h-4 w-4" />
            Source [{citation.id}]
          </div>
          <h4 className="mt-2 text-sm font-semibold text-slate-900">
            {citation.doc_title || "Unknown document"}
          </h4>
        </div>
        <span className="rounded-full bg-blue-50 px-3 py-1 text-xs font-medium text-blue-700">
          score {citation.score}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-600">
        <div>
          <span className="font-semibold text-slate-800">Drug:</span>{" "}
          {citation.drug || "—"}
        </div>
        <div>
          <span className="font-semibold text-slate-800">Page:</span>{" "}
          {citation.page_num ?? "—"}
        </div>
        <div className="col-span-2">
          <span className="font-semibold text-slate-800">Section:</span>{" "}
          {citation.section_title || "—"}
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={classNames("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div
        className={classNames(
          "max-w-[88%] rounded-3xl px-5 py-4 shadow-sm",
          isUser
            ? "bg-blue-600 text-white"
            : "border border-slate-200 bg-white text-slate-900"
        )}
      >
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold opacity-80">
          {isUser ? (
            <>
              <Sparkles className="h-4 w-4" />
              You
            </>
          ) : (
            <>
              <Stethoscope className="h-4 w-4 text-blue-600" />
              MedLex RAG
            </>
          )}
        </div>

        <div className="prose-answer whitespace-pre-wrap text-sm">
          {message.content}
          {message.streaming && (
            <span className="ml-1 inline-block h-4 w-2 animate-pulse rounded-sm bg-blue-500 align-middle" />
          )}
        </div>

        {!isUser && message.citations?.length > 0 && (
          <div className="mt-4 grid gap-3">
            {message.citations.map((c) => (
              <CitationCard key={`${message.id}-${c.id}`} citation={c} />
            ))}
          </div>
        )}

        {!isUser && message.latency && (
          <div className="mt-3 text-xs text-slate-500">
            Model: {message.model || "qwen2.5:3b"} · Latency: {message.latency} ms
          </div>
        )}
      </div>
    </div>
  );
}

function StatusPill({ status }) {
  const isOk = status === "online";

  return (
    <div
      className={classNames(
        "inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold",
        isOk
          ? "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200"
          : "bg-amber-50 text-amber-700 ring-1 ring-amber-200"
      )}
    >
      {isOk ? <CheckCircle2 className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
      API {isOk ? "Online" : "Checking"}
    </div>
  );
}

export default function App() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [question, setQuestion] = useState("");
  const [drug, setDrug] = useState("ibuprofen");
  const [customDrug, setCustomDrug] = useState("");
  const [topK, setTopK] = useState(3);
  const [isStreaming, setIsStreaming] = useState(false);
  const [copied, setCopied] = useState(false);

  const [messages, setMessages] = useState([
    {
      id: crypto.randomUUID(),
      role: "assistant",
      content:
        "Welcome to MedLex RAG. Ask a question about an ingested FDA drug label. I will answer only from retrieved documents and show citations.",
      citations: []
    }
  ]);

  const scrollRef = useRef(null);

  const activeDrug = useMemo(() => {
    return customDrug.trim() || drug;
  }, [customDrug, drug]);

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) setApiStatus("online");
        else setApiStatus("offline");
      } catch {
        setApiStatus("offline");
      }
    }

    checkHealth();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [messages]);

  async function askStream(q = question) {
    const finalQuestion = q.trim();
    if (!finalQuestion || isStreaming) return;

    setQuestion("");
    setIsStreaming(true);

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: finalQuestion
    };

    const assistantId = crypto.randomUUID();

    const assistantMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      citations: [],
      streaming: true,
      model: "",
      latency: null
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);

    try {
      const response = await fetch(`${API_BASE}/ask/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question: finalQuestion,
          drug: activeDrug,
          domain: "fda",
          top_k: Number(topK)
        })
      });

      if (!response.ok || !response.body) {
        throw new Error(`API error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          const event = JSON.parse(line);

          if (event.type === "meta") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      citations: event.citations || [],
                      model: event.model
                    }
                  : m
              )
            );
          }

          if (event.type === "token") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      content: m.content + event.content
                    }
                  : m
              )
            );
          }

          if (event.type === "done") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      streaming: false,
                      model: event.model,
                      latency: event.latency_ms
                    }
                  : m
              )
            );
          }
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                streaming: false,
                content:
                  "Could not get a response from the RAG API. Make sure FastAPI is running on http://127.0.0.1:8000.",
                citations: []
              }
            : m
        )
      );
    } finally {
      setIsStreaming(false);
    }
  }

  async function copyLastAnswer() {
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
    if (!lastAssistant?.content) return;

    await navigator.clipboard.writeText(lastAssistant.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,#dbeafe,transparent_35%),linear-gradient(135deg,#f8fafc,#eef2ff)]">
      <header className="border-b border-white/60 bg-white/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-600 shadow-soft">
              <Stethoscope className="h-7 w-7 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-950">
                MedLex RAG
              </h1>
              <p className="text-sm text-slate-500">
                FDA drug-label assistant with Pinecone, Neo4j, MySQL & Ollama
              </p>
            </div>
          </div>

          <StatusPill status={apiStatus} />
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-6 py-6 lg:grid-cols-[320px_1fr]">
        <aside className="space-y-5">
          <section className="rounded-3xl border border-white/70 bg-white/80 p-5 shadow-soft backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-2">
              <Settings2 className="h-5 w-5 text-blue-600" />
              <h2 className="font-semibold text-slate-950">RAG Controls</h2>
            </div>

            <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Drug filter
            </label>
            <select
              value={drug}
              onChange={(e) => setDrug(e.target.value)}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none ring-blue-500 focus:ring-2"
            >
              {DEFAULT_DRUGS.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>

            <input
              value={customDrug}
              onChange={(e) => setCustomDrug(e.target.value)}
              placeholder="Or type custom drug"
              className="mt-3 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none ring-blue-500 focus:ring-2"
            />

            <label className="mt-5 block text-xs font-semibold uppercase tracking-wide text-slate-500">
              Retrieved chunks
            </label>
            <select
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none ring-blue-500 focus:ring-2"
            >
              <option value={2}>2 chunks</option>
              <option value={3}>3 chunks</option>
              <option value={5}>5 chunks</option>
            </select>

            <div className="mt-5 rounded-2xl bg-slate-50 p-4 text-xs text-slate-600">
              Active drug:{" "}
              <span className="font-semibold text-slate-900">{activeDrug}</span>
            </div>
          </section>

          <section className="rounded-3xl border border-white/70 bg-white/80 p-5 shadow-soft backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-2">
              <Activity className="h-5 w-5 text-blue-600" />
              <h2 className="font-semibold text-slate-950">Pipeline</h2>
            </div>

            <div className="space-y-3 text-sm">
              <div className="flex items-center gap-3 rounded-2xl bg-blue-50 p-3 text-blue-800">
                <Database className="h-5 w-5" />
                Pinecone Vector Search
              </div>
              <div className="flex items-center gap-3 rounded-2xl bg-indigo-50 p-3 text-indigo-800">
                <Network className="h-5 w-5" />
                Neo4j Graph Retrieval
              </div>
              <div className="flex items-center gap-3 rounded-2xl bg-emerald-50 p-3 text-emerald-800">
                <FileText className="h-5 w-5" />
                MySQL BM25 Chunks
              </div>
              <div className="flex items-center gap-3 rounded-2xl bg-purple-50 p-3 text-purple-800">
                <Brain className="h-5 w-5" />
                Ollama qwen2.5:3b
              </div>
            </div>
          </section>

          <section className="rounded-3xl border border-white/70 bg-white/80 p-5 shadow-soft backdrop-blur-xl">
            <div className="mb-3 flex items-center gap-2">
              <ShieldCheck className="h-5 w-5 text-blue-600" />
              <h2 className="font-semibold text-slate-950">Safety</h2>
            </div>
            <p className="text-sm leading-6 text-slate-600">
              Answers are generated only from retrieved FDA-label context. This is
              for document Q&A, not medical advice.
            </p>
          </section>
        </aside>

        <section className="flex min-h-[calc(100vh-140px)] flex-col rounded-3xl border border-white/70 bg-white/75 shadow-soft backdrop-blur-xl">
          <div className="flex items-center justify-between border-b border-slate-200/70 px-6 py-4">
            <div>
              <h2 className="text-lg font-bold text-slate-950">Document Chat</h2>
              <p className="text-sm text-slate-500">
                Ask FDA-label questions and get cited RAG answers.
              </p>
            </div>

            <button
              onClick={copyLastAnswer}
              className="inline-flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50"
            >
              {copied ? <CheckCircle2 className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              {copied ? "Copied" : "Copy answer"}
            </button>
          </div>

          <div
            ref={scrollRef}
            className="scrollbar-thin flex-1 space-y-5 overflow-y-auto px-6 py-6"
          >
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}
          </div>

          <div className="border-t border-slate-200/70 px-6 py-4">
            <div className="mb-3 flex flex-wrap gap-2">
              {SAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => askStream(q)}
                  disabled={isStreaming}
                  className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:border-blue-300 hover:text-blue-700 disabled:opacity-50"
                >
                  {q}
                </button>
              ))}
            </div>

            <div className="flex items-end gap-3">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    askStream();
                  }
                }}
                placeholder={`Ask about ${activeDrug}...`}
                rows={2}
                className="max-h-40 flex-1 resize-none rounded-3xl border border-slate-200 bg-white px-5 py-4 text-sm outline-none ring-blue-500 focus:ring-2"
              />

              <button
                onClick={() => askStream()}
                disabled={isStreaming || !question.trim()}
                className="inline-flex h-14 items-center gap-2 rounded-3xl bg-blue-600 px-6 text-sm font-bold text-white shadow-soft transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                {isStreaming ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Send className="h-5 w-5" />
                )}
                Ask
              </button>
            </div>

            <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
              <Server className="h-4 w-4" />
              Backend: {API_BASE} · Streaming endpoint: /ask/stream
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}