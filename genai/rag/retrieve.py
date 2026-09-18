import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as _cfg

RAG_DOCS_DIR = _cfg.PATHS['rag_docs_dir']
DEFAULT_FAISS_PATH = _cfg.PATHS['faiss_index']
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120


def _extract_title_from_markdown(text: str, source_file: str) -> str:
    m = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    stem = Path(source_file).stem
    return stem.replace("_", " ").title()


def _split_markdown(text: str, source_file: str,
                    chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    title = _extract_title_from_markdown(text, source_file)
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks: List[Dict[str, Any]] = []
    buffer = ""
    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue
        candidate = (buffer + "\n\n" + para_clean).strip() if buffer else para_clean
        if len(candidate) <= chunk_size or not buffer:
            buffer = candidate
        else:
            chunks.append({"text": buffer, "title": title, "source_file": Path(source_file).name})
            buffer_text = buffer
            if overlap > 0 and len(buffer_text) > overlap:
                buffer = buffer_text[-overlap:] + "\n\n" + para_clean
            else:
                buffer = para_clean
    if buffer.strip():
        chunks.append({"text": buffer.strip(), "title": title, "source_file": Path(source_file).name})
    return chunks


def load_knowledge_documents(docs_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    docs_dir = Path(docs_dir) if docs_dir else Path(RAG_DOCS_DIR)
    if not docs_dir.exists():
        return []
    all_chunks: List[Dict[str, Any]] = []
    for md_path in sorted(docs_dir.glob("*.md")):
        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception:
            continue
        chunks = _split_markdown(text, md_path.name)
        for c in chunks:
            c["source_path"] = str(md_path.resolve())
        all_chunks.extend(chunks)
    return all_chunks


def _normalize(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


@dataclass
class _TFIDFBundle:
    available: bool
    backend: str
    vectorizer: Any
    matrix: Any
    docs: List[Dict[str, Any]]
    reason: Optional[str] = None


def _try_sentence_transformers_bundle(docs: List[Dict[str, Any]], index_path: Path, model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        return None, f"sentence-transformers not installed: {e}"
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        return None, f"model load failed: {e}"
    try:
        texts = [d["text"] for d in docs]
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        return None, f"encode failed: {e}"
    embs = np.asarray(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    embs = embs / norms
    try:
        import faiss
        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embs)
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(index_path))
        except Exception:
            pass
    except Exception as e:
        return None, f"faiss build failed: {e}"
    return {
        "available": True,
        "backend": "sentence-transformers+faiss",
        "model": model,
        "index": index,
        "docs": docs,
    }, None


def _build_tfidf_bundle(docs: List[Dict[str, Any]]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [d["text"] for d in docs]
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
        lowercase=True,
        stop_words="english",
    )
    matrix = vec.fit_transform(texts)
    return _TFIDFBundle(
        available=True,
        backend="tfidf+cosine",
        vectorizer=vec,
        matrix=matrix,
        docs=docs,
    )


def load_faiss_index(model_name: str = "all-MiniLM-L6-v2",
                     index_path: Optional[Path] = None,
                     docs_dir: Optional[Path] = None) -> Dict[str, Any]:
    docs = load_knowledge_documents(docs_dir=docs_dir)
    if not docs:
        return {"available": False, "docs": [], "reason": "No knowledge base Markdown files found."}

    index_path = Path(index_path) if index_path else Path(DEFAULT_FAISS_PATH)

    bundle, reason = _try_sentence_transformers_bundle(docs, index_path, model_name)
    if bundle is not None:
        return bundle

    tfidf = _build_tfidf_bundle(docs)
    return {
        "available": True,
        "backend": "tfidf+cosine",
        "vectorizer": tfidf.vectorizer,
        "matrix": tfidf.matrix,
        "docs": docs,
        "note": f"Fallback TF-IDF used; sentence-transformers unavailable: {reason}",
    }


def retrieve(query: str, rag_bundle: Dict[str, Any], top_k: int = 4) -> List[Dict[str, Any]]:
    if not rag_bundle.get("available"):
        return []
    docs = rag_bundle["docs"]
    backend = rag_bundle.get("backend", "unknown")

    if backend.startswith("sentence-transformers"):
        model = rag_bundle["model"]
        index = rag_bundle["index"]
        q = model.encode([query], convert_to_numpy=True)
        q = np.asarray(q, dtype=np.float32)
        n = np.linalg.norm(q, axis=1, keepdims=True)
        n[n < 1e-12] = 1.0
        q = q / n
        k = min(top_k, index.ntotal)
        scores, indices = index.search(q, k)
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            doc = dict(docs[idx])
            doc["score"] = float(scores[0][i])
            results.append(doc)
        return results

    if backend == "tfidf+cosine":
        from sklearn.metrics.pairwise import cosine_similarity
        vec = rag_bundle["vectorizer"]
        matrix = rag_bundle["matrix"]
        qvec = vec.transform([query])
        sims = cosine_similarity(qvec, matrix).ravel()
        order = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in order:
            doc = dict(docs[idx])
            doc["score"] = float(sims[idx])
            results.append(doc)
        return results

    return []
