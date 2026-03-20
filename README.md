# 🔮 NexusChat — RAG + Web Search AI Chatbot

A production-ready Streamlit chatbot featuring:
- **Multi-provider LLM** — Groq, OpenAI, Gemini
- **RAG (Retrieval-Augmented Generation)** — upload PDFs, DOCX, TXT, Markdown; get answers grounded in your documents
- **Live Web Search** — real-time Google results via Serper.dev
- **Response Modes** — ⚡ Concise or 📖 Detailed

---

## Project Structure

```
project/
├── config/
│   └── config.py          ← All API keys & settings (env vars / Streamlit secrets)
├── models/
│   ├── llm.py             ← OpenAI / Groq / Gemini adapters
│   └── embeddings.py      ← Sentence-transformer + OpenAI embedding wrappers
├── utils/
│   ├── rag.py             ← Text extraction, chunking, FAISS vector store, retrieval
│   ├── web_search.py      ← Serper.dev integration
│   └── prompt_builder.py  ← Dynamic system-prompt assembly
├── app.py                 ← Main Streamlit UI
├── requirements.txt
└── .streamlit/
    └── secrets_template.toml
```

---

## Local Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API keys (choose one method)

# Method A — .env file
cp .env.example .env
# edit .env with your keys

# Method B — export directly
export GROQ_API_KEY="gsk_..."
export SERPER_API_KEY="..."

# 5. Run
streamlit run app.py
```

---

## API Keys Required

| Key | Where to get it | Required for |
|-----|----------------|--------------|
| `GROQ_API_KEY` | https://console.groq.com | Groq LLM (default) |
| `OPENAI_API_KEY` | https://platform.openai.com | OpenAI LLM / embeddings |
| `GEMINI_API_KEY` | https://aistudio.google.com | Gemini LLM |
| `SERPER_API_KEY` | https://serper.dev | Live web search |

---

## Streamlit Cloud Deployment

1. Push to GitHub (ensure `config/config.py` and `.env` are in `.gitignore`)
2. Go to https://streamlit.io/cloud → New app → connect your repo
3. Set secrets in **App Settings → Secrets** (TOML format):
   ```toml
   GROQ_API_KEY = "gsk_..."
   SERPER_API_KEY = "your-key"
   ```
4. Deploy!

---

## Features Deep-Dive

### RAG Pipeline
1. Upload documents via sidebar (PDF, DOCX, TXT, MD)
2. Text is extracted → split into 800-char overlapping chunks
3. Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` (local, free)
4. Stored in a FAISS flat index (cosine similarity)
5. At query time, top-4 chunks are retrieved and injected into the system prompt

### Web Search
- Queries Serper.dev (Google Search API) in real time
- Top 5 organic results injected as context before the LLM call
- Gracefully degrades if key is missing

### Response Modes
- **Concise**: 2–4 sentence limit, single key insight
- **Detailed**: Headers, bullets, examples, depth over brevity
"# NexusChat" 
