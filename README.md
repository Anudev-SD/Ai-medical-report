# 🏥 AI Medical Report Analyzer

A Retrieval-Augmented Generation (RAG) app that reads a medical report (PDF) and explains it in simple language. Upload a report, ask a question, and get a plain-English summary, abnormal findings, extracted lab values, and a risk level.

> ⚠️ **Disclaimer:** This project is for learning and demonstration only. It does **not** provide medical diagnosis. Always consult a qualified doctor.

## ✨ Features

- **PDF ingestion**: extracts text from medical report PDFs
- **Semantic search**: embeds chunks with `all-MiniLM-L6-v2` and stores them in ChromaDB
- **Reranking**: improves relevance using a cross-encoder (`ms-marco-MiniLM-L-6-v2`)
- **Medical value extraction**: pulls out test names, values, units, and status (normal/abnormal)
- **Risk scoring**: computes an overall risk level from the extracted values
- **LLM explanation**: OpenAI generates a simple summary, abnormal findings, explanation, and advice
- **Gradio UI**: easy web interface in the browser

## 🧠 How it works

```
PDF → Extract text → Chunk → Embed → ChromaDB
                                        │
Question → Embed → Retrieve top 10 → Rerank (top 5) → Compress (top 3)
                                        │
                      ┌─────────────────┴─────────────────┐
              Extract medical values + Risk         Build prompt
                                        └─────────┬─────────┘
                                              OpenAI LLM
                                                  │
                          Answer + Sources + Extracted values + Risk level
```

## 🛠️ Tech stack

- Python
- Gradio (UI)
- ChromaDB (vector database)
- sentence-transformers (embeddings + reranker)
- OpenAI API (answer generation)
- pypdf (PDF text extraction)

## 📁 Project structure

```
medical-rag-app/
├── app.py              # Main app: ingestion, retrieval, reranking, UI
├── utils/
│   ├── extraction.py   # Extracts medical values from text
│   └── risk.py         # Computes risk level
├── requirements.txt    # Dependencies
├── .env.example        # Example environment variables
└── .gitignore
```

`chroma_db/` is created automatically when you upload your first report.

## 🚀 Getting started

### 1. Clone the repo

```bash
git clone https://github.com/Anudev-SD/Ai-medical-report.git
cd Ai-medical-report
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Copy the example file and add your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
OPENAI_API_KEY=your_key_here
```

### 5. Run the app

```bash
python app.py
```

Open the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

## 💡 Usage

1. Upload a medical report PDF and click **Upload**
2. Wait for the "chunks stored" confirmation
3. Type a question, e.g. *"Explain my blood test results"*
4. Click **Ask** to see the answer, source chunks, extracted values, and risk level

## ⚙️ Configuration

Settings are at the top of `app.py`:

| Setting | Default | Meaning |
|---|---|---|
| `OPENAI_MODEL` | `gpt-3.5-turbo` | LLM used for answers |
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `RETRIEVE_K` | 10 | Chunks fetched from the vector DB |
| `RERANK_TOP_N` | 5 | Chunks kept after reranking |
| `COMPRESS_TOP_N` | 3 | Chunks sent to the LLM |

## 🔮 Future improvements

- Smarter chunking with real overlap
- LLM-based context compression
- Support for multiple reports and per-user collections
- Better handling of scanned PDFs (OCR)

## 📄 License

MIT. Feel free to use this project for learning.
