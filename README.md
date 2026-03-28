# 🏗️ Construction AI Assistant — Mini RAG System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://indecimal-assignment-sgeqcybgjijnu59ytdwwrv.streamlit.app/)

> A production-ready Retrieval-Augmented Generation (RAG) system for a construction marketplace, built with FAISS, SentenceTransformers, and OpenRouter LLM.

---

## 🚀 Live Demo

🔗 [https://indecimal-assignment-sgeqcybgjijnu59ytdwwrv.streamlit.app/](https://indecimal-assignment-sgeqcybgjijnu59ytdwwrv.streamlit.app/)

---

## 🎯 What It Does

This AI assistant answers user questions **strictly from internal construction documents** — no hallucinations, no external knowledge. It:

- ✅ Retrieves the most relevant document chunks using semantic search
- ✅ Generates grounded answers using an LLM
- ✅ Clearly shows **which chunks were used** for every answer
- ✅ Supports **dual-model comparison** (Cloud API vs. Local LLM)
- ✅ **Robust Environment Detection**: Automatically unloads local models seamlessly when hosted remotely on cloud platforms

---

## 🧠 System Architecture

```
User Query
    ↓
Embedding (SentenceTransformer: all-MiniLM-L6-v2)
    ↓
FAISS Vector Search
    ↓
Top-K Relevant Chunks
    ↓
LLM (OpenRouter / Ollama)
    ↓
Grounded Answer + Retrieved Context
```

---

## ⚙️ Tech Stack

| Component    | Tool                                         |
|-------------|----------------------------------------------|
| Embeddings  | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB   | FAISS (local, in-memory)                    |
| Cloud LLM   | OpenRouter (`meta-llama/llama-3-8b-instruct`)|
| Local LLM   | Ollama (`phi`) — optional, local only        |
| UI          | Streamlit                                    |
| Language    | Python 3.10+                                 |

### Why these choices?
- **`all-MiniLM-L6-v2`**: Fast, lightweight, 384-dim embeddings — ideal for local inference without a GPU
- **FAISS**: No external service needed; fast cosine/L2 similarity search in memory
- **OpenRouter**: Free-tier access to LLaMA-3 with no local GPU required for deployment
- **Ollama (optional)**: Demonstrates local LLM capability and model comparison

---

## 📂 Project Structure

```
project/
│
├── app.py                # Main Streamlit application
├── rag_pipeline.ipynb    # Jupyter notebook: chunking, embedding, indexing
├── faiss.index           # Pre-built FAISS vector index
├── chunks.pkl            # Pre-computed document chunks
├── doc1.md               # Source document 1 (Construction Policies)
├── doc2.md               # Source document 2 (FAQs & Pricing)
├── doc3.md               # Source document 3 (Project Specifications)
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .env.example          # Template for environment variables
└── .gitignore            # Ignored files (secrets, cache)
```

---

## 🔧 How Chunking & Retrieval Work

### 1. Document Processing (see `rag_pipeline.ipynb`)
- Each `.md` document is split into paragraph-level chunks
- Each chunk is embedded using `SentenceTransformer('all-MiniLM-L6-v2')`
- Embeddings are stored in a FAISS index and chunks saved to `chunks.pkl`

### 2. Semantic Retrieval
- At query time, the user question is embedded with the same model
- FAISS performs an L2 similarity search and returns the Top-K closest chunks
- Chunks shorter than 50 characters are filtered out

### 3. Grounded Generation
- Retrieved chunks are concatenated as context in the LLM prompt
- The prompt explicitly instructs the model to **only use the provided context**:
  - `"Answer ONLY from the context"`
  - `"Do NOT add external knowledge"`
  - `"If answer is not clearly present, say: 'Not available in context'"`

---

## ⚖️ LLM Comparison

| Aspect        | OpenRouter (Cloud) | Ollama (Local)  |
|--------------|-------------------|-----------------|
| Speed        | Fast ⚡            | Slower 🐢       |
| Cost         | Free tier         | Fully free      |
| Groundedness | Strong ✅          | Medium ⚠️       |
| Hallucination| Low               | Moderate        |
| Dependency   | Internet + API Key | Local daemon    |

---

## 🏃 Running Locally

### Prerequisites
- Python 3.10+
- (Optional) [Ollama](https://ollama.com/) installed and running for local LLM

### 1. Clone the repository
```bash
git clone https://github.com/Shivam123-Kumar/Indecimal-Assignment.git
cd Indecimal-Assignment
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
```bash
# Copy the template
cp .env.example .env

# Edit .env and add your key
# OPENROUTER_API_KEY=your_actual_key_here
```
Get a free API key at [https://openrouter.ai](https://openrouter.ai)

### 4. (Optional) Start Ollama for local LLM
```bash
ollama run phi
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push the repository to GitHub (`.env` is in `.gitignore` — it will NOT be pushed)
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New App** → Select your GitHub repo → Set entry point to `app.py`
4. In **App Settings → Secrets**, add:
   ```toml
   OPENROUTER_API_KEY = "your_actual_key_here"
   ```
5. Click **Deploy** ✅

> **Note:** Ollama cannot run on Streamlit Cloud (no local daemon). The app will automatically detect and warn the user on cloud deployments.

---

## 🔑 Environment Variables

| Variable           | Description                        | Required |
|-------------------|------------------------------------|----------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key         | Yes (for LLM) |

Copy `.env.example` → `.env` and fill in the value.

---

## 🧪 Quality Evaluation

### Sample Test Queries
1. What is escrow payment?
2. How are construction delays handled?
3. What are the pricing packages?
4. What is the quality assurance process?
5. How does the customer journey work?
6. What are the contractor responsibilities?
7. How are disputes resolved?
8. What are the project timeline guarantees?

### Observations
- ✅ Retrieval is accurate for domain-specific queries
- ✅ Increasing Top-K improves completeness for broad questions
- ✅ API model produces precise, well-grounded responses
- ⚠️ Local model (Ollama) may occasionally add unsupported claims — mitigated by strict prompt
- ⚠️ Very short queries may return less relevant chunks — solved by increasing Top-K

---

## ✅ Pre-Deployment Checklist

- [x] App runs locally (`streamlit run app.py`)
- [x] No secrets in code or committed files
- [x] `.env` is in `.gitignore`
- [x] `.env.example` committed with placeholder
- [x] `requirements.txt` is complete and minimal
- [x] README is complete with setup + deploy instructions
- [x] GitHub repo is clean (no zip, no cache files)
- [x] Streamlit Secrets set for cloud deployment

---

## ⚠️ Limitations

- Basic paragraph-level chunking (no sliding window overlap)
- No reranking of retrieved results
- Ollama only works locally — not on cloud platforms
- FAISS index must be pre-built before deployment

---

## 🔮 Future Improvements

- Chunking with overlap for better context continuity
- Cross-encoder reranking for better chunk quality
- Hybrid search (BM25 keyword + FAISS semantic)
- Chat history and follow-up question support
- Automatic index rebuild from uploaded documents

---

## 🏁 Conclusion

This project demonstrates a complete, production-oriented RAG pipeline with accurate document retrieval, grounded answer generation, transparent context display, and real-world deployability. It highlights practical trade-offs between cloud-based and local LLMs in a real AI assistant use case.
