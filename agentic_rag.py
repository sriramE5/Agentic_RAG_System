from __future__ import annotations
import os, re, json, argparse, glob, sys
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Progress
from tqdm import tqdm

# LangChain bits
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, TextLoader
)
# Optional fallback for tricky PDFs (scanned/complex)
try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
    HAVE_UNSTRUCTURED = True
except Exception:
    HAVE_UNSTRUCTURED = False

from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Dense store
import faiss
import numpy as np

# Sparse BM25
from rank_bm25 import BM25Okapi

#   Config 
DATA_DIR = os.environ.get("DATA_DIR", "data")
INDEX_DIR = os.environ.get("INDEX_DIR", "rag_index")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2:1b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))

K_DENSE = int(os.environ.get("K_DENSE", 12))
K_BM25 = int(os.environ.get("K_BM25", 12))
K_FUSED = int(os.environ.get("K_FUSED", 12))
K_CONTEXT = int(os.environ.get("K_CONTEXT", 8))

SELF_CHECK_ITERS = int(os.environ.get("SELF_CHECK_ITERS", 2))
UNCERTAINTY_TRIGGER = float(os.environ.get("UNCERTAINTY_TRIGGER", 0.35))

#  Data IO

def load_documents() -> List[Any]:
    """Load PDFs (with fallback), TXT, MD from DATA_DIR recursively."""
    docs = []
    # Use DirectoryLoader with glob patterns (recursive)
    loaders = [
        DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
        DirectoryLoader(DATA_DIR, glob="**/*.md", loader_cls=TextLoader, show_progress=True),
        DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
    ]
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"[warn] loader failed: {e}")

    # Fallback path for PDFs with parsing issues
    if HAVE_UNSTRUCTURED:
        pdf_paths = glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)
        # Only try fallback for PDFs that might not have yielded text
        # (We can’t easily detect per-file coverage here, so we parse all as a safety net)
        if pdf_paths:
            print("[info] Running UnstructuredPDFLoader fallback for PDFs (if installed)...")
            for path in tqdm(pdf_paths, desc="unstructured pdfs"):
                try:
                    docs.extend(UnstructuredPDFLoader(path).load())
                except Exception:
                    pass
    return docs


def chunk_documents(docs: List[Any]) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_documents(docs)

# Dense (FAISS)
@dataclass
class DenseIndex:
    embed_model: str
    texts: List[str] = None
    metadatas: List[Dict[str, Any]] = None
    index: Any = None

    def build(self, chunks: List[Any]):
        self.texts = [c.page_content for c in chunks]
        self.metadatas = [c.metadata for c in chunks]
        if not self.texts:
            raise RuntimeError("No chunks to index (dense).")
        embed = OllamaEmbeddings(model=self.embed_model)
        embs = embed.embed_documents(self.texts)
        vecs = np.array(embs, dtype="float32")
        if vecs.ndim != 2 or vecs.shape[0] == 0:
            raise RuntimeError("Embedding returned empty vectors.")
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        embed = OllamaEmbeddings(model=self.embed_model)
        q = np.array([embed.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        return list(zip(I[0].tolist(), D[0].tolist()))

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "dense_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "metadatas": self.metadatas, "embed_model": self.embed_model}, f)
        print(f"[success] Saved FAISS + dense_meta to {path}")

    def load(self, path: str):
        with open(os.path.join(path, "dense_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.texts = meta["texts"]
        self.metadatas = meta["metadatas"]
        self.embed_model = meta.get("embed_model", self.embed_model)
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))

# Sparse (BM25)   
@dataclass
class BM25Index:
    texts: List[str] = None
    metadatas: List[Dict[str, Any]] = None
    tokenized: List[List[str]] = None
    _bm25: Any = None

    @staticmethod
    def _tok(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def build(self, chunks: List[Any]):
        self.texts = [c.page_content for c in chunks]
        self.metadatas = [c.metadata for c in chunks]
        if not self.texts:
            raise RuntimeError("No chunks to index (bm25).")
        self.tokenized = [self._tok(t) for t in self.texts]
        self._bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        q = self._tok(query)
        scores = self._bm25.get_scores(q)
        idxs = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in idxs]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "bm25_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "metadatas": self.metadatas}, f)
        print(f"[success] Saved bm25_meta to {path}")

    def load(self, path: str):
        with open(os.path.join(path, "bm25_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.texts = meta["texts"]
        self.metadatas = meta["metadatas"]
        self.tokenized = [self._tok(t) for t in self.texts]
        self._bm25 = BM25Okapi(self.tokenized)

# Fusion   

def reciprocal_rank_fusion(results_lists: List[List[Tuple[int, float]]], k_out: int) -> List[int]:
    RRF_K = 60.0
    scores: Dict[int, float] = {}
    for res in results_lists:
        for rank, (idx, _) in enumerate(res, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:k_out]]

#   - LLM helpers   
_LLM = None

def llm() -> OllamaLLM:
    global _LLM
    if _LLM is None:
        _LLM = OllamaLLM(model=LLM_MODEL, temperature=0.2)
    return _LLM

# Agentic Prompts   
SYS_REWRITE = (
    "Rewrite the user question into a single, specific search query. Avoid pronouns and ambiguity. Return only the rewritten query."
)

SYS_DECOMPOSE = (
    "If the question needs multiple facts, break it into up to 4 atomic sub-questions. Otherwise return a one-element list with the original. Return JSON list only."
)

SYS_SYNTH = (
    "Answer strictly using the provided CONTEXT snippets. If missing info, write 'INSUFFICIENT_CONTEXT' and list what's missing. Include [S#] citations matching the context items."
)

SYS_SELF_CHECK = (
    "Given the question and draft answer, return JSON: {\"confidence\": 0..1, \"followups\": [up to 2 short targeted search queries]}."
)

#  Agentic Steps   

def rewrite_query(user_q: str) -> str:
    prompt = f"System: {SYS_REWRITE}\n\nUser question: {user_q}\nRewritten:"
    out = llm().invoke(prompt).strip()
    return out.split("\n")[0].strip()


def decompose_q(q: str) -> List[str]:
    prompt = f"System: {SYS_DECOMPOSE}\n\nQuestion: {q}\nJSON:"
    raw = llm().invoke(prompt).strip()
    try:
        arr = json.loads(raw)
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr[:4]
    except Exception:
        pass
    return [q]


def retrieve(query: str, d: DenseIndex, s: BM25Index) -> List[int]:
    dense = d.search(query, k=K_DENSE)
    sparse = s.search(query, k=K_BM25)
    fused_ids = reciprocal_rank_fusion([dense, sparse], k_out=max(K_FUSED, 1))
    return fused_ids


def synthesize(user_q: str, d: DenseIndex, ctx_ids: List[int]) -> Tuple[str, List[str]]:
    if not ctx_ids:
        return "INSUFFICIENT_CONTEXT", []
    ctx_lines = []
    used_sources = []
    for i, idx in enumerate(ctx_ids[:K_CONTEXT], start=1):
        text = d.texts[idx]
        meta = d.metadatas[idx]
        src = meta.get("source", meta.get("file_path", "unknown"))
        ctx_lines.append(f"[S{i}] {text}\n(Source: {src})")
        used_sources.append(src)
    ctx_block = "\n\n".join(ctx_lines)
    prompt = (
        f"System: {SYS_SYNTH}\n\nQuestion: {user_q}\n\nCONTEXT:\n{ctx_block}\n\nAnswer:"
    )
    out = llm().invoke(prompt).strip()
    return out, list(dict.fromkeys(used_sources))


def self_check(user_q: str, draft: str) -> Tuple[float, List[str]]:
    prompt = f"System: {SYS_SELF_CHECK}\n\nQuestion: {user_q}\nDraft: {draft}\nJSON:"
    raw = llm().invoke(prompt).strip()
    try:
        obj = json.loads(raw)
        conf = float(obj.get("confidence", 0.0))
        fups = [str(x) for x in obj.get("followups", []) if str(x).strip()]
        return max(0.0, min(1.0, conf)), fups[:2]
    except Exception:
        return 0.0, []

#    Build/Load   -

def build_indexes() -> Tuple[DenseIndex, BM25Index]:
    if not os.path.isdir(DATA_DIR):
        print(f"[error] DATA_DIR not found: {DATA_DIR}")
        return None, None

    print("[info] Loading documents from", DATA_DIR)
    docs = load_documents()
    if not docs:
        print("[warn] No documents found. Add .pdf/.txt/.md under ./data")
        return None, None

    print(f"[info] Loaded {len(docs)} docs. Chunking...")
    chunks = chunk_documents(docs)
    print(f"[info] Created {len(chunks)} chunks.")

    print("[info] Building dense + sparse indexes...")
    d = DenseIndex(EMBED_MODEL)
    d.build(chunks)

    s = BM25Index()
    s.build(chunks)

    os.makedirs(INDEX_DIR, exist_ok=True)
    d.save(INDEX_DIR)
    s.save(INDEX_DIR)
    print("[success] Indexes saved to", INDEX_DIR)
    return d, s


def load_indexes() -> Tuple[DenseIndex, BM25Index]:
    dense_path = os.path.join(INDEX_DIR, "faiss.index")
    bm25_path = os.path.join(INDEX_DIR, "bm25_meta.json")
    if not (os.path.exists(dense_path) and os.path.exists(bm25_path)):
        raise SystemExit("[error] No index found. Run with --build first.")

    d = DenseIndex(EMBED_MODEL)
    d.load(INDEX_DIR)
    s = BM25Index()
    s.load(INDEX_DIR)
    return d, s

#   Ask

def ask(user_q: str) -> Dict[str, Any]:
    d, s = load_indexes()

    q1 = rewrite_query(user_q)
    subs = decompose_q(q1)

    gathered: List[int] = []
    seen = set()
    for sq in subs:
        ids = retrieve(sq, d, s)
        for idx in ids:
            if idx not in seen:
                gathered.append(idx)
                seen.add(idx)

    draft, srcs = synthesize(user_q, d, gathered)

    final = draft
    all_srcs = list(srcs)
    for _ in range(SELF_CHECK_ITERS):
        conf, followups = self_check(user_q, final)
        if conf >= 0.95:
            break
        if conf < UNCERTAINTY_TRIGGER and followups:
            extra_ids: List[int] = []
            for fq in followups:
                extra_ids.extend(retrieve(fq, d, s))
            for idx in extra_ids:
                if idx not in seen:
                    gathered.append(idx)
                    seen.add(idx)
            final, add_srcs = synthesize(user_q, d, gathered)
            for ssrc in add_srcs:
                if ssrc not in all_srcs:
                    all_srcs.append(ssrc)

    return {
        "question": user_q,
        "rewritten": q1,
        "sub_questions": subs,
        "answer": final,
        "sources": all_srcs,
    }

#   CLI 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="(re)build hybrid indexes from ./data")
    ap.add_argument("--ask", type=str, default=None, help="Ask a question")
    args = ap.parse_args()

    if args.build:
        build_indexes()

    if args.ask:
        out = ask(args.ask)
        print(json.dumps(out, ensure_ascii=False, indent=2))
