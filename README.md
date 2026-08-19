# Tech Support Agent 👩🏻‍💻

A RAG (Retrieval-Augmented Generation) chatbot that answers technical support questions by
referencing your own product manuals and documentation — with **advanced conversational memory**
so it stays coherent across long chat sessions without the prompt growing unbounded.

Document embedding runs locally (no document content is sent to any embedding API); the chat
LLM runs on [Groq](https://groq.com)'s hosted API for fast inference and easy cloud deployment.
An earlier fully-local variant (Ollama for both embeddings and chat) is also supported — see
[Running fully local instead](#-running-fully-local-instead-optional).

---

## ✨ Features

- **Document-grounded answers** — upload PDF/TXT manuals; the bot only answers from what's indexed.
- **HyDE-boosted retrieval** — generates a hypothetical answer first to improve semantic search
  matches against your document chunks, before doing the real similarity search.
- **Hand-rolled advanced memory** — keeps the last 6 conversational turns verbatim and
  automatically condenses older turns into a running LLM-generated summary, so the model
  remembers context from many turns ago without the prompt size exploding.
- **Persistent local vector DB** — documents are chunked, embedded, and stored in a local
  Chroma database that survives app restarts.
- **In-character support persona** — "Truptishree," a friendly, on-brand tech support agent,
  implemented via a system prompt + strict response-formatting rules.
- **Source attribution** — every answer shows which uploaded document(s) it was grounded in.
- **One-click reset** — wipe the vector DB, uploaded docs, and chat/memory state.

---

## 🏗️ Architecture

```
┌─────────────┐      ┌────────────────────┐      ┌───────────────────┐
│  app.py     │──────▶ data_processing.py │──────▶  Chroma vector DB  │
│ (Streamlit  │      │ (extract, chunk,   │      │  (./tech_db)       │
│  UI layer)  │      │  embed & store)    │      └───────────────────┘
└─────┬───────┘                                            ▲
      │                                                     │ similarity_search
      ▼                                                     │
┌─────────────┐      ┌────────────────────┐      ┌──────────┴─────────┐
│  memory.py  │◀─────▶│    techbot.py     │──────▶│  Groq API           │
│ (turn window│      │ (RAG orchestration,│      │  (openai/gpt-oss-20b)│
│ + summary)  │      │  HyDE, prompting)  │      └────────────────────┘
└─────────────┘      └────────────────────┘
      ▲
      │
┌─────┴───────┐
│  helper.py  │
│ (reset app) │
└─────────────┘
```

### File responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI: document upload/sync sidebar, chat interface, session-state orchestration. |
| `data_processing.py` | PDF/TXT text extraction, recursive chunking, and building/persisting the Chroma vector store. |
| `techbot.py` | Core RAG logic: embeddings, vector search, HyDE query expansion, LLM prompting (`ChatGroq`), and response parsing. |
| `memory.py` | Hand-rolled advanced memory manager — windowed recent turns + LLM-summarized older history. |
| `helper.py` | Full application reset (clears cache, session state, uploaded docs, and vector DB on disk). |

---

## 🧠 How the memory works

Most basic chatbot demos just dump the entire message history into every prompt. That doesn't
scale — long conversations blow past context limits and slow down every call. This project
instead implements a **summary-buffer** pattern by hand (no deprecated LangChain memory classes):

1. Every completed turn (`user` + `assistant`) is appended to an in-memory **window**
   (`turn_buffer`), kept in **chronological** order.
2. Once the window exceeds `MAX_TURNS` (default: **6** turns), the oldest turn(s) are popped out
   and folded into a single **running summary** via one extra LLM call.
3. Each new question is answered using `chat_summary` (condensed older context) +
   `recent_turns_text` (last 6 turns, verbatim) — bounding prompt size regardless of how long the
   conversation runs.
4. If the summarization LLM call fails, the overflow turns are safely re-queued instead of being
   silently dropped.

This logic lives entirely in `memory.py` and is UI-framework agnostic (it only touches
`st.session_state` as a dict-like store, so it could be swapped for another session backend).

---

## 🛠️ Tech Stack

- **UI:** [Streamlit](https://streamlit.io)
- **LLM:** [Groq](https://groq.com) hosted API running `openai/gpt-oss-20b`, via `langchain-groq`'s `ChatGroq`
- **Embeddings:** `all-MiniLM-L6-v2`, run locally (via `langchain-huggingface`)
- **Vector store:** [Chroma](https://www.trychroma.com/) (`langchain-chroma`), persisted locally
- **Text splitting:** `langchain-text-splitters` (`RecursiveCharacterTextSplitter`)
- **PDF parsing:** `pypdf`

---

## 📋 Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com) (sign up → API Keys → Create)

---

## ⚙️ Installation

1. Clone/download the project files into a folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide your Groq API key. For local development, copy the template and fill in your key:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # then edit .streamlit/secrets.toml and paste your real key
   ```
   `.streamlit/secrets.toml` is git-ignored — never commit it.

---

## ▶️ Running the app

```bash
streamlit run app.py
```

The app will open in your browser (default: `http://localhost:8501`).

---

## ☁️ Deploying online (Streamlit Community Cloud)

1. Push this project to a GitHub repo — include `app.py`, `data_processing.py`, `helper.py`,
   `techbot.py`, `memory.py`, `requirements.txt`, and `.gitignore`. **Do not commit
   `.streamlit/secrets.toml`.**
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **New app**,
   and point it at your repo/branch and `app.py`.
3. In the app's dashboard, go to **Settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```
4. Deploy. First load will download the local embedding model (~90MB), so expect a slightly slow
   cold start.

**Note on storage:** Streamlit Community Cloud's filesystem is ephemeral — `./document` and
`./tech_db` reset whenever the app restarts or redeploys. That's fine for demo purposes (just
re-upload a sample doc after a restart), but isn't a persistent production knowledge base. See
[Known limitations](#️-known-limitations).

---

## 💡 Usage

1. **Upload documents** — In the sidebar, upload one or more `.pdf` or `.txt` tech manuals and
   click **Upload files**. The app extracts, chunks, embeds, and indexes them into the local
   vector DB (`./tech_db`).
2. **Ask questions** — Once documents are indexed, use the chat box to ask technology-related
   questions. Truptishree retrieves the most relevant document chunks and answers grounded in
   them, citing the source file(s) in an expandable panel.
3. **Continue the conversation** — Ask follow-ups; the bot remembers recent context (and a
   summary of older context) automatically — no need to re-explain earlier messages.
4. **Reset** — Click **🔄 Clear All & Reset** in the sidebar to wipe uploaded documents, the
   vector database, chat history, and memory state, and start fresh.

> If no documents have been uploaded/indexed yet, the app prompts you to add some before the
> chat interface becomes available.

---

## 🖥️ Running fully local instead (optional)

If you'd rather keep everything on-device (no data leaves your machine, no API key needed), an
earlier version of `techbot.py` used [Ollama](https://ollama.com)'s `ChatOllama` in place of
`ChatGroq`:

1. Install Ollama and pull a model: `ollama pull llama3.1`, then run `ollama serve`.
2. In `techbot.py`, swap the `ChatGroq` import/instantiation back to:
   ```python
   from langchain_ollama import ChatOllama
   # ...
   return ChatOllama(model="llama3.1", temperature=0.0, stop=["</response>"])
   ```
3. Swap `langchain-groq` for `langchain-ollama` in `requirements.txt`, and drop the
   `GROQ_API_KEY` secret requirement.

This trades cloud deployability for full data locality — good for a "fully local LLM" portfolio
demo, but requires a VPS with Ollama running to deploy online (rather than free platforms like
Streamlit Community Cloud, which can't run Ollama).

---

## 📁 Data & storage

| Path | Contents |
|---|---|
| `./document/` | Raw uploaded PDF/TXT files. |
| `./tech_db/` | Persisted Chroma vector database (chunk embeddings + metadata). |

Both are deleted on **Clear All & Reset**, and `./tech_db` is auto-cleared if `./document` ever
ends up empty (e.g. files were removed manually), to avoid a stale/orphaned index.

---

## ⚠️ Known limitations

- Chat LLM calls go to Groq's hosted API (not fully local) — document content included in prompts
  (retrieved chunks, questions, summaries) leaves your machine on each request. Embeddings and the
  vector DB itself remain local.
- On free hosting (e.g. Streamlit Community Cloud), the filesystem is ephemeral — uploaded
  documents and the vector DB reset on app restarts/redeploys. Fine for a demo, not for a
  persistent production knowledge base.
- Retrieval quality depends on chunking strategy (`chunk_size=1500`, `overlap=500`) and may need
  tuning for very large or densely-formatted manuals.
- Summarization adds one extra LLM call whenever the turn window overflows — a minor latency
  cost in exchange for bounded prompt size on long conversations.
- Single-session memory only (per Streamlit session state) — not persisted across app restarts
  or shared across users.
- Groq's free tier has rate limits (requests/tokens per minute) — heavy concurrent demo traffic
  may hit them.

---

## 🚀 Possible future improvements

- Swap local Chroma for a hosted vector DB (e.g. Chroma Cloud, Pinecone) so indexed documents
  survive redeploys on ephemeral hosting.
- Persist chat memory to disk/DB so context survives app restarts.
- Add automated tests around chunking, retrieval, and memory windowing.
- Support additional document types (`.docx`, `.md`, etc.).
- Add a "regenerate answer" button, and streaming token output for a more responsive feel.
- Configurable `MAX_TURNS` and chunking parameters from the UI.