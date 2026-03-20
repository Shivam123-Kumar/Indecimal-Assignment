# 🏗️ Construction AI Assistant — Mini RAG System

## 🚀 Overview

This project implements a Retrieval-Augmented Generation (RAG) based AI assistant for a construction marketplace.  
The system answers user queries strictly using internal documents such as policies, pricing specifications, and customer journey information.

It ensures:

- ✅ Grounded responses (no hallucination)
- ✅ Transparency (shows retrieved context)
- ✅ Explainability (clear reasoning source)

---

## 🎯 Objective

To build a system that:

- Retrieves relevant information from internal documents  
- Generates answers **only from retrieved content**  
- Demonstrates clarity, correctness, and explainability  

---

## 🧠 System Architecture
User Query
    ↓
Embedding (SentenceTransformer)
    ↓
FAISS Vector Search
    ↓
Top-K Relevant Chunks
    ↓
LLM (OpenRouter / Ollama)
    ↓
Grounded Answer


---

## ⚙️ Tech Stack

| Component   | Tool Used |
|------------|----------|
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector DB  | FAISS |
| API LLM    | OpenRouter (LLaMA-3 / GPT models) |
| Local LLM  | Ollama (`phi`) |
| UI         | Streamlit |
| Language   | Python |

---

## 📂 Project Structure
mini-rag/
│
├── app.py
├── faiss.index
├── untitled.ipynb
├── chunks.pkl
├── doc1.md
├── doc2.md
├── doc3.md
├── requirements.txt
└── README.md


---

## 🔧 How It Works

### 1. Document Processing
- Documents are loaded and split into smaller chunks  
- Each chunk is converted into embeddings  

### 2. Vector Indexing
- FAISS is used to store embeddings  
- Enables fast similarity search  

### 3. Retrieval
- User query is embedded  
- Top-K most relevant chunks are retrieved  

### 4. Grounded Generation
- LLM is prompted with:
  - Retrieved chunks  
  - Strict rules to avoid hallucination  

---

## 🔒 Grounding Strategy

The system enforces strict constraints:
 - Answer ONLY from context
 - Do NOT add external knowledge
 - If not found → "Not available in context"

 
This ensures high reliability and prevents hallucinations.

---

## 🖥️ Features

### ✅ Core Features
- Semantic search using FAISS  
- Context-aware answer generation  
- Transparent display of retrieved chunks  

### ⭐ Advanced Features
- Dual LLM support:
  - API-based (OpenRouter)  
  - Local (Ollama)  
- Model comparison mode  
- Latency measurement  
- Highlighted keywords in context  
- Adjustable Top-K retrieval  

---

## ⚖️ Model Comparison

| Aspect         | OpenRouter (API) | Ollama (Local) |
|---------------|-----------------|---------------|
| Speed         | Fast ⚡         | Slow 🐢       |
| Cost          | API-based       | Free          |
| Groundedness  | Strong ✅       | Medium ⚠️     |
| Hallucination | Low             | Moderate      |
| Dependency    | Internet        | Local         |

---

## 🧪 Evaluation

### Test Queries
- What is escrow payment?  
- How are construction delays handled?  
- What are pricing packages?  
- What is quality assurance system?  
- What is customer journey?  

### Observations
- Retrieval is generally accurate  
- Increasing Top-K improves completeness  
- API model produces more precise answers  
- Local model may introduce hallucinations  

---

## 📊 Key Findings

### OpenRouter
- Faster and more reliable  
- Better grounded responses  

### Ollama
- Fully local and cost-free  
- Requires stronger prompt control  
- Slightly higher latency  

---

## ⚠️ Limitations

- Basic chunking (no overlap)  
- No reranking of retrieved results  
- Some irrelevant chunks may appear  
- Local models may hallucinate  

---

## 🚀 Future Improvements

- Better chunking with overlap  
- Reranking models (e.g., cross-encoder)  
- Hybrid search (keyword + semantic)  
- Chat history support  
- Deployment on cloud (Streamlit Cloud / AWS)  

---

## ▶️ How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run Ollama (optional)
ollama run phi

### 3. Run app
streamlit run app.py


---

## 🔑 Environment Variables

Create `.env` file:
OPENROUTER_API_KEY=your_api_key_here
---

## 🏁 Conclusion

This project demonstrates a complete RAG pipeline with:

- Accurate retrieval  
- Grounded generation  
- Model comparison  
- Real-world usability via UI  

It highlights trade-offs between cloud-based and local LLMs, making it a practical and scalable solution.
