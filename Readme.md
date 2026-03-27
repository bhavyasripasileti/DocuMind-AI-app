<div align="center">

# 🧠 DocuMind AI — Smart PDF Chat using RAG + LLM

### Turn your documents into intelligent conversations — instantly.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-VectorDB-blue?style=for-the-badge)](https://faiss.ai)
[![SentenceTransformers](https://img.shields.io/badge/Embeddings-MiniLM-green?style=for-the-badge)](https://www.sbert.net/)
[![Groq](https://img.shields.io/badge/Groq-LLM-black?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

> An AI-powered document assistant that allows users to upload PDFs and ask questions — delivering accurate, context-aware answers using **Retrieval-Augmented Generation (RAG)**.

<br/>

**[Live Demo](#-live-demo) · [🧠 Architecture](#️-architecture-overview) · [📦 Setup](#-step-by-step-setup) · [✨ Features](#-key-features)**

</div>

---

## 🌐 Live App

> 🔗 **[Click here to explore the dashboard »](https://twtqy2ewdvbfszhb39seck.streamlit.app)**

---
## 🌐 Overview

Understanding long PDFs manually is time-consuming and inefficient.

**DocuMind AI** transforms static documents into an **interactive Q&A system**, enabling users to ask questions and receive precise, context-grounded answers.

Unlike traditional chatbots, it uses **RAG (Retrieval-Augmented Generation)** to minimize hallucinations and ensure reliable responses.

---

## ✨ Key Features

| Feature               | Description                                   |
| --------------------- | --------------------------------------------- |
| 📄 Multi-PDF Chat     | Upload and query multiple PDFs simultaneously |
| 🧠 RAG Architecture   | Ensures context-aware and accurate responses  |
| ⚡ Fast LLM Inference  | Powered by Groq (LLaMA 3.1) for low latency   |
| 🔍 Source Attribution | Displays exact document excerpts used         |
| 💬 Session Memory     | Maintains chat history within session         |
| 🎯 Interactive UI     | Clean and intuitive Streamlit interface       |

---

## 🧠 Architecture Overview

```
User uploads PDF(s)
        │
        ▼
PDF Parsing → Text Chunking → Embeddings → FAISS Index
        │
        ▼
Top-K Retrieval → Prompt Construction → LLM (Groq)
        │
        ▼
Context-Aware Answer + Source Excerpts
```

### 🔄 Pipeline Breakdown

* **PDF Parsing** → Extract text using PyMuPDF
* **Chunking** → Split text into manageable segments
* **Embedding** → Convert text into vector representations
* **Vector Search** → Retrieve relevant chunks using FAISS
* **LLM Generation** → Generate grounded answers using Groq

---

## 📂 Project Structure

```
DocuMind_AI/
│
├── app.py              # Streamlit UI (chat interface)
├── rag_pipeline.py     # Core RAG pipeline
├── utils.py            # PDF parsing & chunking
├── config.py           # Configuration settings
├── requirements.txt    # Dependencies
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Category        | Technology                              |
| --------------- | --------------------------------------- |
| Language        | Python 3.9+                             |
| Frontend        | Streamlit                               |
| Backend         | Python                                  |
| LLM             | Groq (LLaMA 3.1-8B)                     |
| Embeddings      | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS                                   |
| PDF Processing  | PyMuPDF                                 |

---

## 🚀 Step-by-Step Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/DocuMind_AI.git
cd DocuMind_AI
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Configure Environment

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_api_key_here
```

Get your API key: https://console.groq.com/keys

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🧪 How to Use

1. Upload one or more PDFs using the sidebar
2. Wait for processing confirmation ✅
3. Ask questions in natural language
4. View answers along with **source excerpts**
5. Continue chatting — session memory is preserved
6. Use **Clear Chat** to reset conversation
7. Use **Reset All** to clear documents and restart

---

## 🌐 Deployment (Streamlit Cloud)

1. Push project to GitHub
2. Go to https://share.streamlit.io/
3. Select repository → choose `app.py`
4. Add secrets:

```toml
GROQ_API_KEY = "your_api_key"
```

5. Click **Deploy** 🚀

---

## 🔮 Roadmap

* [ ] Multi-document persistent memory
* [ ] Highlight exact answer spans
* [ ] Export chat as PDF
* [ ] Docker deployment
* [ ] OCR for scanned PDFs

---

## 💼 Project Highlights

* Built a complete **end-to-end RAG pipeline**
* Integrated **LLM + vector search architecture**
* Optimized for **fast inference using Groq**
* Developed a **real-world AI application**
* Deployed using **Streamlit Cloud**

---

## 📜 License

MIT License — free to use and modify.

---

## 👤 Author

**Bhavya Sri Pasileti**

> Data Science & AI Enthusiast
> Passionate about building AI-powered applications and intelligent systems.

[LinkedIn](https://www.linkedin.com/in/bhavya-sri-pasileti-16565a2a1)

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

*Built with ❤️ by Bhavya Sri Pasileti*

</div>
