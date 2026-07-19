---
title: "Building a Production-Grade AI Agent PoC — From Blueprint to Working System"
date: 2026-07-19T12:30:00Z
draft: false
description: "A walkthrough of building an observable, testable AI agent with RAG, web search, and code execution — including every twist, turn, and debugging session along the way."
tags: ["ai", "agents", "langchain", "gemini", "fastapi", "rag", "agents-poc"]
categories: ["Engineering"]
toc: true
cover:
  image: ""
  alt: "AI Agent Architecture"
---

## The Blueprint

It started with a battle-tested blueprint I'd distilled from real-world AI team
implementations — a minimal yet production-grade agent PoC that covers the
*entire* AI-native development lifecycle: PRD → Code → Test → Deploy → Observe.

The requirements were clear:

- Single-agent system with **RAG + web search + code execution**
- **Observability-first**: every step must emit logs and traces
- **Testable**: 100% critical path coverage
- **Deployable**: runs in Docker with zero config

The tech stack was simple on paper:

| Component      | Choice                         |
|----------------|--------------------------------|
| LLM            | Google Gemini                  |
| Vector Store   | FAISS (local)                  |
| Web Search     | DuckDuckGo (free, no API key)  |
| Code Executor  | Python subprocess sandbox      |
| API Server     | FastAPI + Uvicorn              |
| Tracing        | LangSmith                      |
| Logging        | Structured JSON                |
| Container      | Docker + Docker Compose        |

## Building It

The architecture followed a clean layered structure:

```
agent-poc/
├── prd/spec.md              # PRD — the "why" before code
├── src/
│   ├── config.py            # Centralized env-based config
│   ├── agent/
│   │   ├── core.py          # ReAct agent via langchain.agents.create_agent
│   │   ├── tools.py         # DuckDuckGo search + Python executor
│   │   └── memory.py        # FAISS vector store + Gemini embeddings
│   ├── api/
│   │   ├── main.py          # FastAPI with lifespan
│   │   └── routes.py        # GET /, GET /health, POST /chat
│   └── eval/
│       └── harness.py       # LangSmith eval (with local fallback)
├── tests/
│   ├── unit/                # 8 tests — mocked LLM/tools
│   └── integration/         # 5 tests — FastAPI TestClient
├── docs/knowledge-base.md   # RAG source
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

### Core Agent (ReAct Pattern)

The agent uses LangChain's `create_agent` — the modern replacement for the old
`AgentExecutor`. It's a compiled state graph from LangGraph under the hood:

```python
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=...)
agent = create_agent(model=llm, tools=tools, system_prompt=...)
result = agent.invoke({"messages": [HumanMessage(content=query)]})
```

Each tool is a simple decorated function:

```python
@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    return DuckDuckGoSearchRun().run(query)

@tool
def code_executor(code: str) -> str:
    """Execute Python code in a sandboxed subprocess."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout or result.stderr
```

### RAG Pipeline

The knowledge base is a directory of markdown files. On startup, they're split
into chunks, embedded via `gemini-embedding-001`, and indexed into FAISS:

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("./data/faiss_index")
```

### API Layer

FastAPI with structured JSON logging and LangSmith tracing:

```python
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    result = run_agent(req.query, vectorstore=get_index())
    return ChatResponse(
        response=result["response"],
        trace=result.get("trace", []),
    )
```

## The Slog — What Actually Happened When Building

No blog post is honest without the debugging journey. Here's the real timeline:

### Block 1: Vertex AI Auth Wall

The original plan was Google Vertex AI. First error:

```
DefaultCredentialsError: Your default credentials were not found.
```

The user had `gcloud auth login` but not `gcloud auth application-default login`.
We found the ADC credentials at `~/.config/gcloud/legacy_credentials/.../adc.json`
and set `GOOGLE_APPLICATION_CREDENTIALS`.

Next: `Vertex AI API not enabled`. Opened the GCP console, clicked **ENABLE**.

Then: `model textembedding-gecko@003 not found`. Tried `text-embedding-004` —
also not found. The Vertex AI project was too fresh.

**Pivot**: Switched from `langchain-google-vertexai` to `langchain-google-genai`.
The Gemini API just needs an API key — no project setup, no IAM, no enabling APIs.

### Block 2: Model Name Roulette

```
gemini-2.0-flash → "no longer available"
gemini-1.5-flash-001 → "not found"
```

Listed available models via the API — `gemini-2.5-flash` was the stable
option. Also learned the embedding model is `models/gemini-embedding-001`
(not `models/embedding-001` or `text-embedding-004`).

### Block 3: The New LangChain API

The original blueprint used the old API:

```python
from langchain.agents import AgentExecutor, create_react_agent  # gone
```

In LangChain 1.3+, `create_react_agent` moved to `langgraph.prebuilt`, then
was replaced by `create_agent` from `langchain.agents`. The return type is a
`CompiledStateGraph`, not an `AgentExecutor`. Input schema is `{"messages": [...]}`,
output is `{"messages": [...]}` with the full conversation history.

### Block 4: Response Parsing

The Gemini API returns content as a list of parts:

```python
# Raw response:
[{'type': 'text', 'text': 'Paris is the capital of France.'}]

# Fix: extract text from parts
if isinstance(content, list):
    response = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
```

### Block 5: Missing DuckDuckGo dependency

```
Could not import ddgs python package. Please install it with pip install -U ddgs.
```

A quick `pip install ddgs` fixed it — the newer `duckduckgo_search` library
needs `ddgs` as a separate package.

## Testing & Validation

The test suite runs at three levels:

### Unit Tests (8 tests, ~10s)

```
tests/unit/test_tools.py
  test_code_executor_simple_math  — print(2+2) returns "4"
  test_code_executor_multiline    — for loop works
  test_code_executor_timeout      — sleep(30) → timeout message
  test_code_executor_syntax_error — malformed code → error message
  test_code_executor_no_output    — x=42 → "(no output)"

tests/unit/test_memory.py
  test_retrieve_returns_documents
  test_retrieve_empty_results
  test_retrieve_calls_similarity_search
```

### Integration Tests (5 tests, instant)

```
tests/integration/test_api.py
  test_health_endpoint
  test_health_returns_json
  test_chat_endpoint_missing_query  — 422 validation
  test_chat_endpoint_invalid_method — 405 on GET

tests/integration/test_agent.py
  test_agent_returns_dict
```

### Eval Harness (LangSmith + local fallback)

The eval harness runs 5 test cases and reports success rate + P95 latency.
With LangSmith configured, it creates an experiment with full trace visibility:

```
https://smith.langchain.com/o/.../datasets/.../compare
```

Without LangSmith, it falls back to local logging:

```
Success rate: 80.0% (4/5)
P95 latency:  4.60s
```

## The Examples — Seeing It Work

Here's what the agent actually did with real queries. Each response includes
a full `trace` array showing every step of the agent's reasoning.

### 1 — Code Execution

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Calculate 7 * 8 using Python"}'
```

```json
{
  "response": "7 * 8 = 56",
  "trace": [
    {"role": "user", "content": "Calculate 7 * 8 using Python"},
    {"role": "assistant", "tool_calls": [
      {"name": "code_executor", "args": {"code": "print(7 * 8)"}}
    ]},
    {"role": "tool", "name": "code_executor", "content": "56"},
    {"role": "assistant", "content": "7 * 8 = 56"}
  ]
}
```

### 2 — Web Search

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the current population of Japan?"}'
```

```json
{
  "response": "The current population of Japan is 125,694,302 as of Tuesday, July 14, 2026.",
  "trace": [
    {"role": "user", "content": "What is the current population of Japan?"},
    {"role": "assistant", "tool_calls": [
      {"name": "web_search", "args": {"query": "current population of Japan"}}
    ]},
    {"role": "tool", "name": "web_search", "content": "Japan population data..."},
    {"role": "assistant", "content": "The current population of Japan is 125,694,302..."}
  ]
}
```

### 3 — RAG (Local Knowledge Base)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the Production Agent use for its vector store?"}'
```

```json
{
  "response": "The Production Agent uses FAISS (local, persisted to disk) for its vector store.",
  "trace": [
    {"role": "user", "content": "What does the Production Agent use for its vector store?"},
    {"role": "assistant", "tool_calls": [
      {"name": "retrieve_docs", "args": {"q": "Production Agent vector store"}}
    ]},
    {"role": "tool", "name": "retrieve_docs",
     "content": "# Production Agent Documentation\n\n## Overview..."},
    {"role": "assistant", "content": "The Production Agent uses FAISS..."}
  ]
}
```

### 4 — Multi-Tool Orchestration

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "population ratio of EU over USA"}'
```

```json
{
  "response": "The population ratio of the EU over the USA is approximately 1.295.",
  "trace": [
    {"role": "user", "content": "population ratio of EU over USA"},
    {"role": "assistant", "tool_calls": [
      {"name": "web_search", "args": {"query": "population of EU"}},
      {"name": "web_search", "args": {"query": "population of USA"}}
    ]},
    {"role": "tool", "name": "web_search", "content": "EU population ~452,000,000..."},
    {"role": "tool", "name": "web_search", "content": "USA population ~349,035,494..."},
    {"role": "assistant", "tool_calls": [
      {"name": "code_executor", "args": {
        "code": "eu_population = 452000000\nusa_population = 349035494\nratio = eu_population / usa_population\nprint(ratio)"
      }}
    ]},
    {"role": "tool", "name": "code_executor", "content": "1.294997236011762"},
    {"role": "assistant", "content": "The population ratio of the EU over the USA is approximately 1.295."}
  ]
}
```

This is the ReAct pattern working at its best — the agent realized it needed
two independent data points before it could compute, gathered both
simultaneously via parallel web searches, then used code execution to derive
the answer. 9.3s total, 7 steps, 2 tool types.

## What We Learned

### What Worked
- **Structured JSON logging**: Every request, response, tool call, and agent
  step is logged as JSON. Correlating via `trace_id` makes debugging trivial.
- **The trace endpoint**: Returning the full message chain in the API response
  lets you inspect exactly what the agent did without external tooling.
- **DuckDuckGo**: Despite some Wikipedia TLS errors, it returned useful results
  with zero API key cost.
- **LangSmith integration**: The eval harness pushed results to LangSmith with
  zero additional code — LangChain handles tracing transparently.

### What Surprised Us
- **Vertex AI is heavy for a PoC**: GCP project setup, API enabling, IAM —
  the Gemini API (just an API key) was far simpler for getting started.
- **LangChain's API churn**: Between 1.0 and 1.3, the agent API changed
  completely (`AgentExecutor` → `create_agent` → `CompiledStateGraph`).
  If you're reading this in 2027, check the current API.
- **Gemini model naming**: Models have names like `gemini-2.5-flash`,
  `gemini-2.0-flash-001`, `gemini-1.5-flash-001`, and they get deprecated
  surprisingly fast. Always list available models at runtime.

### What We'd Do Next
- **Sandboxed code execution**: The current subprocess approach works for a
  PoC but needs proper containerization with gVisor or Docker-in-Docker for
  production.
- **Multi-agent orchestration**: The blueprint explicitly scoped this out,
  but adding a supervisor agent would handle complex queries better.
- **Persistent memory**: Add conversation history and a proper message store
  for stateful interactions.

## Try It Yourself

```bash
git clone <repo>
cd agent-poc
cp .env.example .env   # Add your GEMINI_API_KEY and LANGSMITH_API_KEY
pip install .[dev]
python -c "from src.agent.memory import build_index; build_index()"
uvicorn src.api.main:app --reload

# In another terminal:
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here"}' | python3 -m json.tool
```

The full source code, including tests, Dockerfile, CI pipeline, and eval
harness, is available in the [repository](https://github.com/YidaWeng/Production-Agent).
