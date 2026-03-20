# NexusChat — Setup Guide

A step-by-step guide to get NexusChat running locally and deployed to Streamlit Cloud.

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10 or 3.11 | 3.11 recommended for Streamlit Cloud |
| pip | latest | `pip install --upgrade pip` |
| Git | any | for cloning and pushing to GitHub |

---

## Step 1 — Clone or Unzip the Project

**From zip:**
```bash
unzip NexusChat_Project.zip
cd project
```

**From GitHub:**
```bash
git clone https://github.com/your-username/nexuschat.git
cd nexuschat
```

---

## Step 2 — Create a Virtual Environment

```bash
# Create
python -m venv .venv

# Activate — macOS / Linux
source .venv/bin/activate

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate — Windows (CMD)
.venv\Scripts\activate.bat
```

You should see `(.venv)` in your terminal prompt.

---

## Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80 MB)
> on first run. This is cached after the initial download.

---

## Step 4 — Configure API Keys

```bash
cp .env.example .env
```

Open `.env` in any editor and fill in your keys:

```env
GROQ_API_KEY=gsk_...         # Required for default Groq provider
SERPER_API_KEY=...            # Required for Live Web Search feature
OPENAI_API_KEY=sk-...         # Optional — only if using OpenAI provider
GEMINI_API_KEY=AIza...        # Optional — only if using Gemini provider
```

### Where to get each key

| Key | URL | Free Tier |
|-----|-----|-----------|
| `GROQ_API_KEY` | https://console.groq.com | Yes — generous free tier |
| `SERPER_API_KEY` | https://serper.dev | Yes — 2,500 searches/month |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys | Pay-as-you-go |
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey | Yes — free quota |

> **Minimum setup:** Only `GROQ_API_KEY` is required to run the chatbot.
> `SERPER_API_KEY` is only needed if you enable the Web Search toggle.

---

## Step 5 — Run the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## Step 6 — Using the Features

### RAG (Document Q&A)
1. Click **Enable RAG** toggle in the sidebar
2. Upload one or more files using the file uploader (PDF, DOCX, TXT, Markdown)
3. Wait for the "✅ embedded!" confirmation
4. Ask questions — the chatbot will retrieve relevant passages from your documents

### Live Web Search
1. Click **Enable web search** toggle in the sidebar
2. Ask any question — the app queries Google via Serper.dev before calling the LLM
3. The answer will be grounded in current web results

### Response Modes
- Select **⚡ Concise** for short, 2–4 sentence answers
- Select **📖 Detailed** for comprehensive answers with structure and examples

### Switching LLM Providers
- Use the **Provider** dropdown to switch between Groq, OpenAI, and Gemini
- Use the **Model** dropdown to pick a specific model for that provider

---

## Deploying to Streamlit Cloud

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial NexusChat commit"
git remote add origin https://github.com/your-username/nexuschat.git
git push -u origin main
```

Make sure `.gitignore` excludes `.env` and `.streamlit/secrets.toml`.

### 2. Create the app on Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Click **New app**
3. Select your GitHub repo, branch (`main`), and set **Main file path** to `app.py`
4. Click **Advanced settings** → set Python version to `3.11`

### 3. Add Secrets
In **App Settings → Secrets**, paste the following (TOML format):

```toml
GROQ_API_KEY   = "gsk_..."
SERPER_API_KEY = "your-serper-key"
OPENAI_API_KEY = "sk-..."       # optional
GEMINI_API_KEY = "AIza..."      # optional
```

### 4. Deploy
Click **Deploy!** — Streamlit Cloud installs `requirements.txt` automatically.
Your app will be live at `https://your-app-name.streamlit.app`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: faiss` | Run `pip install faiss-cpu` — on Apple Silicon use `pip install faiss-cpu --no-binary :all:` |
| `GROQ_API_KEY not set` warning | Make sure `.env` exists and `python-dotenv` is installed |
| Web search returns no results | Check `SERPER_API_KEY` is valid; verify at https://serper.dev/dashboard |
| Slow first response | SentenceTransformer downloads model on cold start — subsequent requests are fast |
| Streamlit Cloud deploy fails | Check Python version is 3.11 in Advanced Settings; check secrets are set correctly |
| Gemini `role` error | Ensure you're using `gemini-1.5-flash` or `gemini-1.5-pro` — older models differ |

---

## Project Structure Quick Reference

```
project/
├── config/
│   └── config.py          ← All settings; reads from .env or Streamlit secrets
├── models/
│   ├── llm.py             ← get_llm_response() — unified LLM entry point
│   └── embeddings.py      ← embed_texts() / embed_query() — local or OpenAI
├── utils/
│   ├── rag.py             ← VectorStore, chunk_text(), build_rag_context()
│   ├── web_search.py      ← web_search(), search_and_format()
│   └── prompt_builder.py  ← build_system_prompt()
├── app.py                 ← Streamlit UI
├── requirements.txt
├── runtime.txt            ← Python 3.11 for Streamlit Cloud
├── .env.example           ← Template — copy to .env
├── .gitignore
└── README.md
```
