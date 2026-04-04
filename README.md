# DJ Marc Edward — AI Booking Chatbot

A production-ready conversational AI assistant for a professional DJ's booking and inquiry workflow. Built as a portfolio project demonstrating **Retrieval-Augmented Generation (RAG)**, multi-surface deployment, provider-agnostic LLM design, and real-world conversational state management.

---

## What It Does

- Answers general inquiries (genres, services, experience) using a knowledge base — no scripted responses
- Guides interested clients through a booking flow: collects event type, location, date, and duration
- Resolves relative date expressions ("next Friday", "this weekend", "sa Sabado") against today's actual date
- Maintains short-term conversational memory so follow-up questions and references resolve naturally
- Generates a personalized price estimate once all details are gathered
- Handles budget negotiation, negative closes, and conversation wrap-up gracefully
- Understands English, Tagalog, and Taglish

---

## Architecture

```
knowledge_base.txt          Plain-text knowledge source (editable without touching code)
      │
      ▼
kb_loader.py                Parses [SECTION] blocks → chunks → loads into ChromaDB
      │
      ▼
ChromaDB (in-memory)        Vector store — semantic retrieval at query time
      │
      ▼
bot.py                      Core logic: intent classification, extraction, flow, quote generation
      │
      ├── llm()                   Single LLM call site — swap providers here only
      ├── llm_json()              JSON-mode call with centralized fence cleaning
      ├── chat_cli()              CLI / VS Code terminal
      ├── get_bot_response()      Streamlit adapter
      └── handle_webhook_message()  FB / IG Messenger adapter
```

### Why RAG?

Instead of scripting every possible answer, the bot retrieves the most relevant sections from `knowledge_base.txt` at query time and passes them to the LLM as context. This means:

- The knowledge base can be updated (new venues, pricing changes, add-ons) without modifying bot logic
- Answers are grounded in real data — the model can't hallucinate services that aren't listed
- The same pattern scales to any domain: lawyers, photographers, caterers, venues

### Conversational Memory

Each session maintains a `history` list of the last N turns (default: 6). This is injected into LLM prompts so the model can resolve follow-up questions, pronouns, and references naturally — without the user having to repeat themselves. Memory is stored in-session state only and is wiped automatically when the session ends or `init_state()` is called again. No external storage required.

### Time Grounding

`datetime.now()` is injected into all prompts that reason about dates. This lets the LLM correctly interpret expressions like "next Friday", "this weekend", or "end of the month" relative to the actual current date — a critical requirement for any booking workflow.

### Provider-Agnostic LLM Layer

All LLM calls go through two thin wrapper functions — `llm()` for plain text and `llm_json()` for structured output. Switching from Gemini to OpenAI (or any other provider) means changing only these two functions. Every prompt, flow branch, and adapter remains untouched.

---

## Stack

| Layer | Tool |
|---|---|
| LLM | Google Gemini 2.5 Flash (via `google-genai`) |
| Vector Store | ChromaDB (in-memory; swap to persistent for production) |
| Language | Python 3.10+ |
| Surfaces | CLI, Streamlit, FB/IG Messenger webhook |

---

## Project Structure

```
dj_marc_bot/
├── bot.py               Core bot logic + surface adapters
├── kb_loader.py         KB parser and ChromaDB loader
├── knowledge_base.txt   DJ's services, pricing, genres, venues, contact info
└── README.md            This file
```

---

## Quickstart

**Install dependencies**
```bash
pip install google-genai chromadb
```

**Run in terminal**
```bash
python bot.py
```

**Test a different surface** — open `bot.py`, find `chat_cli()`, change:
```python
state = init_state(source="cli")   # → "streamlit" | "fb" | "ig"
```

**Update the knowledge base** — edit `knowledge_base.txt` directly. Section headers use `[BRACKETS]`. The loader picks up changes on the next startup.

**Adjust memory window** — change `MEMORY_WINDOW` at the top of `bot.py` (default: 6 turns).

---

## Surfaces

### CLI / VS Code
Runs `chat_cli()` directly. Bot speaks first, then loops on `input()`.

### Streamlit
Use `get_bot_response(user_input, state)` in your `app.py`.

```python
from bot import init_state, get_greeting, get_bot_response
from kb_loader import load_knowledge_base

if "state" not in st.session_state:
    load_knowledge_base()
    st.session_state.state = init_state(source="streamlit")
    greeting = get_greeting(st.session_state.state)
    st.session_state.state["greeted"] = True
    # render greeting as the first chat bubble

response, st.session_state.state = get_bot_response(user_input, st.session_state.state)
```

### FB / IG Messenger Webhook
Use `handle_webhook_message(user_text, state)`. State — including conversation history — must be persisted externally (Redis, Firestore, etc.) keyed on `sender_id`.

```python
from bot import init_state, get_greeting, handle_webhook_message

# On incoming message:
state = load_state_from_db(sender_id) or init_state(source="fb")

if not state["greeted"]:
    send_to_messenger(sender_id, get_greeting(state))
    state["greeted"] = True

response, state = handle_webhook_message(user_text, state)
send_to_messenger(sender_id, response)
save_state_to_db(sender_id, state)   # history is inside state — persists automatically
```

---

## Conversation Flow

```
User messages
      │
      ├─ Not greeted yet?           → Send greeting, ask for name
      ├─ New contact info?          → Acknowledge it
      ├─ Negative close?            → Graceful exit
      ├─ Budget negotiation?        → Flag to Marc, ask for contact (CLI/Streamlit only)
      ├─ General inquiry?           → RAG answer from KB, no interrogation
      └─ Booking intent?
            ├─ Name missing?        → Ask for name
            ├─ Event details?       → Ask for missing fields (time-grounded extraction)
            ├─ Contact missing?     → Ask (CLI/Streamlit only)
            └─ All present?         → Generate quote + CTA
```

---

## Key Design Decisions

**Single LLM call site** — `llm()` and `llm_json()` are the only places that touch the provider SDK. Switching from Gemini to OpenAI is a two-function change, not a codebase refactor.

**LLM intent classifier instead of keyword lists** — `classify_intent()` uses the LLM to decide `"general"` vs `"booking"`. Handles natural phrasing across languages without brittle phrase maintenance.

**Time grounding via `datetime.now()`** — Injected into extraction and date-reasoning prompts. "Next Friday" always resolves to the correct calendar date regardless of when the bot is running.

**In-session memory, no infrastructure** — History lives in the state dict. On CLI/Streamlit it's naturally scoped to the session. On webhooks it persists as long as state is saved to your DB — no separate memory store needed.

**Source-aware contact ask** — On FB/IG the bot never asks for a phone number or email. On CLI/Streamlit it's asked clearly once before the quote.

**KB separated from logic** — `knowledge_base.txt` is the single source of truth for all factual content. A non-technical person (the DJ or his manager) can update pricing, add venues, or change contact details without reading Python.

**Greeting-first design** — The bot always introduces itself before the user speaks, mirroring how a real customer service agent handles a call or chat.

---

## Switching LLM Providers

Only `llm()` and `llm_json()` need to change. Example OpenAI replacement:

```python
from openai import OpenAI
_oa = OpenAI(api_key=OPENAI_API_KEY)

def llm(prompt: str) -> str:
    r = _oa.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content.strip()

def llm_json(prompt: str) -> dict | None:
    # Use OpenAI structured output or keep the same fence-cleaning approach
    from pydantic import BaseModel
    class Extraction(BaseModel):
        name: str | None
        contact: str | None
        event_type: str | None
        location: str | None
        date: str | None
        duration: float | None
    r = _oa.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format=Extraction
    )
    return r.choices[0].message.parsed.model_dump()
```

Everything else — prompts, flow logic, state, adapters — remains unchanged.

---

## Production Notes

- Swap `chromadb.Client()` (in-memory) for `chromadb.PersistentClient(path="./chroma_db")` to persist the vector store across restarts
- Store conversation state in Redis or Firestore for the webhook surface
- Move `API_KEY` to an environment variable (`os.environ.get("GEMINI_API_KEY")`) before deploying — never commit a live key to a public repo
- Add a rate limiter on the webhook handler to prevent abuse
- `MEMORY_WINDOW = 6` is a sensible default; increase for longer, more complex conversations or decrease to reduce token usage

---

## About

Built by an AI automation engineer as a real deployment for DJ Marc Edward (Philippines). Demonstrates end-to-end RAG chatbot architecture, provider-agnostic LLM design, in-session conversational memory, time-grounded date extraction, and multi-surface adapter pattern — in a clean, maintainable codebase.
