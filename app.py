import streamlit as st
import faiss
import numpy as np
import pickle
import time
import os
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------
# LOAD ENV VARIABLES
# Supports both local .env and Streamlit Cloud secrets
# -------------------------------
load_dotenv()

def get_api_key():
    """Load API key from Streamlit secrets (cloud) or .env (local)."""
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return os.getenv("OPENROUTER_API_KEY")

api_key = get_api_key()

# Detect if running on Streamlit Cloud (no local Ollama available)
IS_CLOUD = os.getenv("STREAMLIT_SHARING_MODE") is not None or "STREAMLIT_SERVER_ADDRESS" in os.environ

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("🏗️ Construction AI Assistant")

# -------------------------------
# CUSTOM CSS (DARK UI)
# -------------------------------
st.markdown("""
<style>
.chunk-box {
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 12px;
    border: 1px solid #333;
    color: #e6e6e6;
    font-size: 14px;
    line-height: 1.6;
}

.chunk-title {
    color: #00c6ff;
    font-weight: bold;
    margin-bottom: 8px;
}

mark {
    background-color: #ffeb3b;
    color: black;
    padding: 2px 4px;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODELS & DATA
# -------------------------------
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("faiss.index")
    chunks = pickle.load(open("chunks.pkl", "rb"))
    return embedding_model, index, chunks

embedding_model, index, all_chunks = load_models()

# -------------------------------
# INIT API CLIENT
# -------------------------------
client = None
if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
else:
    st.warning("⚠️ OpenRouter API key not found. Set it in `.env` locally or in Streamlit Cloud Secrets.")

# -------------------------------
# RETRIEVAL
# -------------------------------
def retrieve(query, k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results, scores = [], []

    for dist, i in zip(distances[0], indices[0]):
        chunk = all_chunks[i]
        if len(chunk.strip()) > 50:
            results.append(chunk)
            scores.append(float(round(float(dist), 4)))

    return results, scores

# -------------------------------
# HIGHLIGHT FUNCTION
# -------------------------------
def highlight_text(text, query):
    words = query.split()
    for w in words:
        pattern = re.compile(re.escape(w), re.IGNORECASE)
        text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
    return text

# -------------------------------
# PROMPT
# -------------------------------
def build_prompt(query, context):
    return f"""
You are a strict AI assistant.

RULES:
1. Answer ONLY from the context
2. Do NOT add external knowledge
3. Do NOT guess
4. If answer is not clearly present, say: "Not available in context"
5. Keep answer short and precise

Context:
{context}

Question:
{query}

Answer:
"""

# -------------------------------
# API MODEL (OpenRouter)
# -------------------------------
def generate_api_answer(query, chunks):
    if not client:
        return "❌ API key missing. Cannot use OpenRouter.", 0

    context = "\n\n".join(chunks)
    prompt = build_prompt(query, context)

    start = time.time()

    response = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    latency = float(round(time.time() - start, 2))
    return response.choices[0].message.content, latency

# -------------------------------
# LOCAL LLM MODEL (Ollama)
# Lazy import — safe for cloud deployments where Ollama is not running
# -------------------------------
def generate_ollama_answer(query, chunks):
    if IS_CLOUD:
        return (
            "⚠️ Ollama is a local-only model. It cannot run on Streamlit Cloud. "
            "Please run the app locally with Ollama installed to use this feature.",
            0.0
        )

    try:
        import ollama  # lazy import — only needed locally
    except ImportError:
        return "❌ Ollama package not installed. Run `pip install ollama` and try again.", 0.0

    context = "\n\n".join(chunks)
    prompt = build_prompt(query, context)
    start = time.time()

    try:
        response = ollama.chat(
            model='phi',
            messages=[{"role": "user", "content": prompt}]
        )
        latency = float(round(time.time() - start, 2))
        return response['message']['content'], latency
    except Exception as e:
        latency = float(round(time.time() - start, 2))
        return (
            f"⚠️ Ollama Error: Could not connect to local model. "
            f"Make sure Ollama is installed and running (`ollama run phi`).\n\nDetails: {e}",
            latency
        )

# -------------------------------
# SIDEBAR INFO
# -------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Mini RAG System** for a Construction Marketplace.
    
    - 🔍 **Retrieval**: FAISS semantic search  
    - 📝 **Embedding**: `all-MiniLM-L6-v2`  
    - 🤖 **LLM**: OpenRouter API (cloud)  
    - 🧠 **Local LLM**: Ollama `phi` (local only)  
    """)
    if IS_CLOUD:
        st.info("☁️ Running on cloud — Ollama unavailable. Use OpenRouter.")

# -------------------------------
# UI CONTROLS
# -------------------------------
st.caption("Single Model → One model answer | Compare Models → Side-by-side comparison")

mode = st.radio("Mode:", ["Single Model", "Compare Models"])
model_choice = st.selectbox("Model:", ["OpenRouter", "Ollama (local only)"])
top_k = st.slider("Top K Chunks", 1, 10, 5)

query = st.text_input("🔍 Ask your question:")

# -------------------------------
# MAIN
# -------------------------------
if query:

    chunks, scores = retrieve(query, top_k)

    st.subheader("📄 Retrieved Context")

    for i, (c, s) in enumerate(zip(chunks, scores)):
        highlighted = highlight_text(c[:400], query)

        st.markdown(f"""
        <div class="chunk-box">
            <div class="chunk-title">
                Chunk {i+1} | Score: {s}
            </div>
            {highlighted}...
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------
    # SINGLE MODE
    # -------------------------------
    if mode == "Single Model":

        st.subheader("💡 Answer")

        with st.spinner("Generating..."):
            if model_choice == "OpenRouter":
                answer, latency = generate_api_answer(query, chunks)
            else:
                answer, latency = generate_ollama_answer(query, chunks)

        if "Not available in context" in answer or answer.startswith("❌") or answer.startswith("⚠️"):
            st.warning(answer)
        else:
            st.success(answer)

        st.info(f"⏱ Latency: {latency} sec")
        st.markdown(f"**Model Used:** {model_choice}")

    # -------------------------------
    # COMPARE MODE
    # -------------------------------
    else:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🚀 OpenRouter")
            with st.spinner("Running..."):
                api_answer, api_time = generate_api_answer(query, chunks)

            if api_answer.startswith("❌") or api_answer.startswith("⚠️"):
                st.warning(api_answer)
            else:
                st.success(api_answer)
            st.info(f"⏱ {api_time} sec")

        with col2:
            st.subheader("🧠 Ollama (local only)")
            with st.spinner("Running..."):
                ollama_answer, ollama_time = generate_ollama_answer(query, chunks)

            if ollama_answer.startswith("❌") or ollama_answer.startswith("⚠️"):
                st.warning(ollama_answer)
            else:
                st.success(ollama_answer)
            st.info(f"⏱ {ollama_time} sec")