import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from genai.rag.retrieve import (
    load_knowledge_documents,
    load_faiss_index,
    retrieve,
)


def main():
    docs = load_knowledge_documents()
    n_files = len(set(d["source_file"] for d in docs))
    print(f"Loaded {len(docs)} chunks from {n_files} Markdown files")
    sources = set(d["source_file"] for d in docs)
    for s in sorted(sources):
        count = sum(1 for d in docs if d["source_file"] == s)
        print(f"  {s}: {count} chunks")

    print()
    print("Building FAISS index...")
    bundle = load_faiss_index(
        model_name="all-MiniLM-L6-v2",
        index_path=PROJECT_ROOT / "genai" / "rag" / "faiss_index.bin",
    )
    print(f"Index available: {bundle.get('available')}")
    if not bundle.get("available"):
        reason = bundle.get("reason", "unknown")
        print(f"  reason: {reason}")
        return 1

    queries = [
        "What follower to following ratio indicates a fake account?",
        "How does account age help detect bot accounts?",
        "What spam keywords appear in fake bios?",
        "What steps should I take to investigate a suspicious account?",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        results = retrieve(q, bundle, top_k=2)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['source_file']} - {r['title']}")
            snippet = r["text"][:160].replace("\n", " ")
            print(f"       ...{snippet}...")

    print("\nRAG verification DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
