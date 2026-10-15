import os
import re
from pathlib import Path

import gradio as gr
import chromadb
import pypdf
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

from utils.extraction import extract_medical_values
from utils.risk import compute_risk


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-3.5-turbo"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHROMA_PATH = "./chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVE_K = 10
RERANK_TOP_N = 5  #This controls how many chunks are retrieved initially from the vector database.
COMPRESS_TOP_N = 3

# ─────────────────────────────────────────
# INIT MODELS
# ─────────────────────────────────────────
embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

Path(CHROMA_PATH).mkdir(exist_ok=True)
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(name="rag_docs")

client = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────
RAG_PROMPT = """\
You are a medical report analysis assistant.

Explain the report in simple terms.

Rules:
- Do NOT give medical diagnosis
- Highlight abnormal values
- Be clear and simple
- Add disclaimer: Consult a doctor

Format:
1. Summary
2. Abnormal Findings
3. Explanation
4. Advice

Context:
{context}

Question: {question}

Answer:
"""

# ─────────────────────────────────────────
# PDF → TEXT
# ─────────────────────────────────────────
def extract_text(file_path):
    reader = pypdf.PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return pages

# ─────────────────────────────────────────
# CHUNK
# ─────────────────────────────────────────
def chunk_text(text):
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE):
        chunks.append(text[i:i+CHUNK_SIZE])
    return chunks

# ─────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────
def ingest(file):
    pages = extract_text(file.name)
    all_chunks = []

    for p in pages:
        all_chunks.extend(chunk_text(p))

    embeddings = embedder.encode(all_chunks).tolist()
    ids = [str(i) for i in range(len(all_chunks))]

    collection.upsert(ids=ids, documents=all_chunks, embeddings=embeddings)

    return f"✅ {len(all_chunks)} chunks stored"

# ─────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────
def search(query):
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=RETRIEVE_K)

    chunks = []
    for doc in results["documents"][0]:
        chunks.append({"text": doc})

    return chunks

# ─────────────────────────────────────────
# RERANK
# ─────────────────────────────────────────
def rerank(query, chunks):
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)

    for c, s in zip(chunks, scores):
        c["score"] = float(s)

    return sorted(chunks, key=lambda x: x["score"], reverse=True)[:RERANK_TOP_N]

# ─────────────────────────────────────────
# COMPRESS
# ─────────────────────────────────────────
def compress(query, chunks):
    return chunks[:COMPRESS_TOP_N]

# ─────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────
def run_pipeline(question):
    retrieved = search(question)
    reranked = rerank(question, retrieved)
    compressed = compress(question, reranked)

    # 🧬 MEDICAL EXTRACTION
    medical_data = []
    for chunk in compressed:
        medical_data.extend(extract_medical_values(chunk["text"]))

    # 🚨 RISK
    risk_level = compute_risk(medical_data)

    # CONTEXT
    context = "\n\n".join([c["text"] for c in compressed])
    prompt = RAG_PROMPT.format(context=context, question=question)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content

    # ADD RISK
    answer += f"\n\n🔴 Risk Level: {risk_level}"

    return answer, compressed, medical_data

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
def ui_upload(file):
    return ingest(file)

def ui_ask(question):
    answer, sources, medical_data = run_pipeline(question)

    src_text = ""
    for i, s in enumerate(sources):
        src_text += f"[{i+1}] {s['text'][:200]}...\n\n"

    med_text = ""
    for m in medical_data:
        med_text += f"- {m['test']}: {m['value']} {m['unit']} ({m['status']})\n"

    return answer, src_text, med_text

# ─────────────────────────────────────────
# GRADIO
# ─────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 AI Medical Report Analyzer")

    file = gr.File()
    upload_btn = gr.Button("Upload")
    upload_output = gr.Textbox()

    upload_btn.click(ui_upload, inputs=file, outputs=upload_output)

    question = gr.Textbox(label="Ask Question")
    ask_btn = gr.Button("Ask")

    answer_output = gr.Textbox(label="Answer")
    sources_output = gr.Textbox(label="Sources")
    medical_output = gr.Textbox(label="Extracted Medical Values")

    ask_btn.click(
        ui_ask,
        inputs=question,
        outputs=[answer_output, sources_output, medical_output],
    )

if __name__ == "__main__":
    demo.launch()