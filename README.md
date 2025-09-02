
# Agentic RAG System

A lightweight, production-ready **Retrieval-Augmented Generation (RAG) pipeline** built with **Ollama + LangChain**, supporting **hybrid retrieval (FAISS + BM25 + Reciprocal Rank Fusion)** and an **agentic reasoning loop** for more accurate answers.  

This version guarantees **persistent FAISS indexes**, robust document loaders (TXT, MD, PDF with fallback), and an iterative self-check mechanism for better reliability.  

---

## Features  
- Robust loaders: Recursively loads `.pdf`, `.txt`, `.md` files from `./data`  
- Chunking: Smart text splitting with overlap  
- Hybrid retrieval: Dense (FAISS + OllamaEmbeddings) + Sparse (BM25) + Reciprocal Rank Fusion  
- Agentic reasoning loop (rewrite → decompose → retrieve → synthesize → self-check → re-retrieval)  
- Guaranteed FAISS save/load (`./rag_index`)  
- Graceful fallbacks for empty data/index  
- No deprecation warnings (`langchain-ollama`)  

---

## Installation  

### Prerequisites  
- Python **3.10+**  
- [Ollama](https://ollama.ai) installed and running  

### Pull required models  
```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### Install dependencies  
```bash
pip install -U \
  langchain langchain-community langchain-text-splitters langchain-ollama \
  faiss-cpu rank-bm25 tqdm pydantic==2.* unstructured
```

(Optional, for better PDF parsing)  
```bash
pip install unstructured[local-inference] pdfminer-six
```

---

##  Usage  

### 1. Build index  
```bash
python agentic_rag.py --build
```

### 2. Ask a question  
```bash
python agentic_rag.py --ask "What is retrieval-augmented generation?"
```

---

## ⚙️ Environment Variables  

| Variable            | Default        | Description |
|---------------------|----------------|-------------|
| `DATA_DIR`          | `data`         | Path to documents |
| `INDEX_DIR`         | `rag_index`    | Index storage dir |
| `LLM_MODEL`         | `llama3.2:1b`  | Ollama LLM to use |
| `EMBED_MODEL`       | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE`        | `1200`         | Chunk size |
| `CHUNK_OVERLAP`     | `200`          | Overlap size |
| `K_DENSE`           | `12`           | Dense retrieval top-k |
| `K_BM25`            | `12`           | BM25 retrieval top-k |
| `K_FUSED`           | `12`           | Final fused top-k |
| `K_CONTEXT`         | `8`            | Context snippets passed to LLM |
| `SELF_CHECK_ITERS`  | `2`            | Self-check iterations |
| `UNCERTAINTY_TRIGGER` | `0.35`       | Threshold to trigger re-retrieval |

---

## Project Structure  

```
.
├── agentic_rag.py        # Main script
├── data/                 # Place documents here (.pdf, .txt, .md)
├── rag_index/            # Auto-created index storage
│   ├── faiss.index
│   ├── dense_meta.json
│   └── bm25_meta.json
└── README.md
```

---

## 📝 License  
MIT License. Free to use and modify.  
