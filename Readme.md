# 🧠 DocuMind AI — Smart PDF Chat using RAG + LLM

> Upload PDFs · Ask questions · Get accurate answers grounded in your documents — powered by Retrieval-Augmented Generation (RAG).

---

## ✨ Key Features

* 📄 Chat with multiple PDFs in real-time
* 🧠 RAG-based architecture (reduces hallucinations)
* ⚡ Ultra-fast responses using Groq LLM
* 🔍 Source-based answers with context excerpts
* 💬 Chat history maintained within session
* 🎯 Clean and interactive Streamlit UI

---

## 🏗️ Architecture Overview

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
│  Text Splitter   │  Chunking (size=1000, overlap=150)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Embedding Model  │  SentenceTransformers (all-MiniLM-L6-v2)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  FAISS Index     │  Vector similarity search
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Top-K Retrieval │  Relevant chunks (k=8)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Prompt Builder  │  Context + Question
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Groq LLM       │  llama-3.1-8b-instant
└────────┬─────────┘
         │
         ▼
 Answer + Source Excerpts in Chat UI
```

---

## 📁 Project Structure

```
DocuMind_AI/
├── app.py              # Streamlit UI (chat interface)
├── rag_pipeline.py     # Core RAG engine
├── utils.py            # PDF parsing + chunking
├── config.py           # Environment-based configuration
├── requirements.txt    # Dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python
* **LLM**: Groq (LLaMA 3.1)
* **Embeddings**: Sentence Transformers
* **Vector DB**: FAISS
* **PDF Processing**: PyMuPDF

---

## 🚀 Step-by-Step Setup

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd DocuMind_AI
```

---

### 2️⃣ Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4️⃣ Configure Environment

Create `.env` file in root folder:

```env
GROQ_API_KEY=groq_api_key_here
```

👉 Get free API key: https://console.groq.com/keys

---

### 5️⃣ Run Application

```bash
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

## 🌐 Deployment (Streamlit Cloud)

1. Push project to GitHub
2. Go to https://share.streamlit.io/
3. Select your repo → `app.py`
4. Add Secrets:

```toml
GROQ_API_KEY = "----"
```

5. Click **Deploy** 

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

## 🔮 Future Improvements

* Multi-document memory
* Highlight exact answer spans
* Export chat as PDF
* Docker deployment
* OCR for scanned PDFs

---

## 💼 Project Highlights

* Built a real-world **RAG pipeline**
* Integrated **Groq LLM for fast inference**
* Implemented **vector search using FAISS**
* Designed an **end-to-end AI application**
* Deployed using **Streamlit Cloud**

---

## 📜 License

MIT License — free to use and modify.

##  Built with ❤️ by Bhavya Sri Pasileti
