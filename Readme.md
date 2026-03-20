# 📄 DocuMind AI — RAG Application

> Upload PDFs · Ask questions · Get answers grounded in your documents — no hallucinations.

---

## Architecture Overview

```
User uploads PDF(s)
        │
        ▼
┌──────────────────┐
│   PDF Parser     │  PyMuPDF → raw text
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Text Splitter   │  RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Embedding Model  │  SentenceTransformers (all-MiniLM-L6-v2, 384-dim, local)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  FAISS Index     │  IndexFlatIP (exact inner-product search)
└────────┬─────────┘
         │  ◄──── User question (also embedded)
         ▼
┌──────────────────┐
│  Top-K Retrieval │  Returns most relevant chunks (default k=5)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Prompt Builder  │  System prompt + numbered context excerpts + question
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Google Gemini   │  Generates grounded answer (temperature=0.2)
└────────┬─────────┘
         │
         ▼
     Answer + source excerpts displayed in chat UI
```

---

## Project Structure

```
smart_pdf_chat/
├── app.py             ← Streamlit UI (chat interface, sidebar, source display)
├── rag_pipeline.py    ← Core RAG engine (embed, index, retrieve, generate)
├── utils.py           ← Pure helpers (PDF parse, chunking, sanitization)
├── config.py          ← All settings read from environment variables
├── requirements.txt   ← Pinned Python dependencies
├── .env.example       ← Template for your API key & tunable parameters
└── README.md          ← This file
```

---

## Prerequisites

| Tool    | Minimum version | Notes                         |
|---------|----------------|-------------------------------|
| Python  | 3.10+          | 3.11 recommended              |
| pip     | 23+            | `pip install --upgrade pip`   |
| Git     | any            | optional, for cloning         |

---

## Step-by-Step Setup

### Step 1 — Clone / download the project

```bash
git clone <your-repo-url> smart_pdf_chat
cd smart_pdf_chat
```

Or simply copy the files into a folder named `smart_pdf_chat`.

---

### Step 2 — Create a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your terminal prompt.

---

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `streamlit` – UI framework
- `PyMuPDF` – PDF text extraction
- `langchain` – text splitting utilities
- `sentence-transformers` – local embeddings (downloads ~90 MB model on first run)
- `faiss-cpu` – vector similarity search
- `google-generativeai` – Gemini API client
- `python-dotenv` – loads `.env` file

> **Note:** The first run downloads the SentenceTransformer model (~90 MB). Subsequent runs are instant.

---

### Step 4 — Get a Google Gemini API key

1. Go to → https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click **Create API Key**
4. Copy the key (starts with `AIza…`)

> The free tier gives you generous quota — plenty for development and testing.

---

### Step 5 — Configure your environment

```bash
cp .env.example .env
```

Open `.env` in any text editor and paste your key:

```dotenv
GOOGLE_API_KEY=AIzaSy...your_actual_key_here...
```

You can also tweak other parameters in `.env`:

| Variable         | Default            | What it does                              |
|------------------|--------------------|-------------------------------------------|
| `GEMINI_MODEL`   | `gemini-1.5-flash` | Gemini model (flash = fast & cheap)       |
| `EMBEDDING_MODEL`| `all-MiniLM-L6-v2` | Local embedding model                     |
| `CHUNK_SIZE`     | `800`              | Characters per chunk                      |
| `CHUNK_OVERLAP`  | `150`              | Overlap between consecutive chunks        |
| `TOP_K_CHUNKS`   | `5`                | Context chunks sent to LLM per question   |

---

### Step 6 — Run the application

```bash
streamlit run app.py
```

Your browser should open automatically at:

```
http://localhost:8501
```

If it doesn't, open that URL manually.

---

## How to Use

1. **Upload PDFs** using the sidebar file uploader (multiple files supported).
2. Wait for the green ✅ confirmation — chunks are now embedded and indexed.
3. **Type your question** in the chat input box at the bottom.
4. The answer appears with a "📌 Source excerpts" expander below — click it to see which parts of the document were used.
5. Continue the conversation — full chat history is preserved during the session.
6. Use **🗑️ Clear Chat** to wipe history without re-processing PDFs.
7. Use **♻️ Reset All** to clear everything and start fresh.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `GOOGLE_API_KEY is not set` | `.env` missing or key not set | Check Step 4 & 5 |
| `No extractable text` | Scanned/image PDF | Run OCR first (e.g. Adobe, Tesseract) |
| `password-protected` | Encrypted PDF | Decrypt with Adobe or `qpdf` |
| Slow first startup | Embedding model downloading | Wait ~1 min; only happens once |
| `faiss-cpu` install fails | Missing build tools | `pip install faiss-cpu --no-build-isolation` |
| Generic API error | Gemini rate limit | Wait 60 s and retry; or use a paid tier |

---

## Production Hardening (Next Steps)

- **Persistence**: Save the FAISS index to disk with `faiss.write_index` so re-uploads aren't needed after restarts.
- **Authentication**: Add Streamlit's built-in `st.secrets` and a simple login page.
- **OCR support**: Integrate `pytesseract` for scanned PDFs.
- **Async**: Move embedding and LLM calls to `asyncio` workers to avoid blocking the UI.
- **Observability**: Add LangSmith or Weights & Biases tracing for production monitoring.
- **Containerisation**: Wrap in a `Dockerfile` and deploy on Cloud Run / Fly.io.

---

## License

MIT — use freely, attribute kindly.