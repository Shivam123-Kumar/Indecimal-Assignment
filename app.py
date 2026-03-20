import streamlit as st
import faiss
import numpy as np
import pickle
import time
import os
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import ollama
from dotenv import load_dotenv

# -------------------------------
# LOAD ENV VARIABLES
# -------------------------------
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("Construction AI Assistant")

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
    st.warning("⚠️ OpenRouter API key not found. API mode disabled.")

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
            scores.append(round(float(dist), 4))

    return results, scores

# -------------------------------
# HIGHLIGHT FUNCTION (FIXED)
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
# API MODEL
# -------------------------------
def generate_api_answer(query, chunks):
    if not client:
        return "API key missing. Cannot use OpenRouter.", 0

    context = "\n\n".join(chunks)
    prompt = build_prompt(query, context)

    start = time.time()

    response = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    latency = round(time.time() - start, 2)
    return response.choices[0].message.content, latency

# -------------------------------
# OLLAMA MODEL
# -------------------------------
def generate_ollama_answer(query, chunks):
    context = "\n\n".join(chunks)
    prompt = build_prompt(query, context)

    start = time.time()

    response = ollama.chat(
        model='phi',
        messages=[{"role": "user", "content": prompt}]
    )

    latency = round(time.time() - start, 2)
    return response['message']['content'], latency

# -------------------------------
# UI CONTROLS
# -------------------------------
st.caption("Single Model → One model | Compare Models → Side-by-side comparison")

mode = st.radio("Mode:", ["Single Model", "Compare Models"])
model_choice = st.selectbox("Model:", ["OpenRouter", "Ollama"])
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

        if "Not available in context" in answer:
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

            st.success(api_answer)
            st.info(f"⏱ {api_time} sec")

        with col2:
            st.subheader("🧠 Ollama")
            with st.spinner("Running..."):
                ollama_answer, ollama_time = generate_ollama_answer(query, chunks)

            st.success(ollama_answer)
            st.info(f"⏱ {ollama_time} sec")