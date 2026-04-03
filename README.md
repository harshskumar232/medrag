# 🏥 MedRAG — Full Stack Clinical Intelligence Suite

A production-grade local RAG system for healthcare documents with:
- **FastAPI** backend with real REST APIs
- **ChromaDB** persistent vector database
- **Sentence Transformers** for real semantic embeddings
- **PyMuPDF** for proper PDF parsing
- **2 AI Agents**: Medical Summary Agent + Symptom Checker
- **Full frontend** (the HTML file you already have, now connected to real backend)

---

## 📁 Project Structure

```
medrag/
├── backend/
│   ├── main.py              # FastAPI app — all routes
│   ├── rag.py               # RAG pipeline (embed, store, retrieve)
│   ├── agents.py            # AI Agents (Summary + Symptom Checker)
│   ├── parser.py            # Document parsers (PDF, CSV, JSON, TXT)
│   └── config.py            # Settings & environment variables
├── frontend/
│   └── index.html           # Full UI (connects to backend API)
├── data/
│   ├── uploads/             # Uploaded files stored here
│   └── chroma/              # ChromaDB persistent vector store
├── .env                     # Your API key goes here
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🚀 Setup (One Time)

### 1. Install Python dependencies
```bash
cd medrag
pip install -r requirements.txt
```

### 2. Add your Anthropic API key
Edit the `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Run the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Open the frontend
Open your browser and go to:
```
http://localhost:8000
```
That's it! The backend serves the frontend automatically.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/documents/upload` | Upload & index a file |
| `POST` | `/api/documents/paste` | Index pasted text |
| `GET`  | `/api/documents` | List all indexed documents |
| `DELETE` | `/api/documents/{id}` | Remove a document |
| `POST` | `/api/chat` | RAG chat query |
| `POST` | `/api/agents/summary` | Medical Summary Agent |
| `POST` | `/api/agents/symptom` | Symptom Checker Agent |
| `GET`  | `/api/stats` | System statistics |
| `DELETE` | `/api/reset` | Clear everything |

---

## 🧠 How the RAG Pipeline Works

```
Document Upload
      │
      ▼
  Parser (PDF/TXT/CSV/JSON)
      │
      ▼
  Text Chunker (configurable size + overlap)
      │
      ▼
  Sentence Transformer Embeddings
  (all-MiniLM-L6-v2 — runs 100% locally)
      │
      ▼
  ChromaDB Vector Store (persisted to disk)
      │
      ▼
  On Query: Semantic Search (cosine similarity)
      │
      ▼
  Top-K Chunks → Claude API → Response
```

---

## 💡 Tips

- First run downloads the embedding model (~90MB) — subsequent runs are instant
- ChromaDB persists to `data/chroma/` — your docs survive restarts
- Uploaded files are saved to `data/uploads/`
- The embedding model runs **entirely on your CPU** — no GPU needed
