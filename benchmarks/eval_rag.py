"""
benchmarks/eval_rag.py — Retrieval quality evaluation for APEX in-process RAG.

Metrics:
    precision@k  — fraction of top-k retrieved chunks whose doc_id appears
                   in the ground-truth relevant set for that query
    MRR          — Mean Reciprocal Rank; 1/rank of first relevant chunk;
                   measures how early a relevant chunk appears

Usage:
    # Evaluate against a live ChromaDB collection (requires GEMINI_API_KEY)
    python benchmarks/eval_rag.py --dataset benchmarks/eval_rag_dataset.json

    # Specify a non-default collection or chroma path
    python benchmarks/eval_rag.py \
        --dataset benchmarks/eval_rag_dataset.json \
        --chroma-path ~/.rag/chroma \
        --collection documents \
        --top-k 5

    # Output results as JSON
    python benchmarks/eval_rag.py --dataset benchmarks/eval_rag_dataset.json --json

Output (default):
    Query                                  P@5     RR
    -----------------------------------------------
    what is the refund policy?             0.600   1.000
    how do I cancel my subscription?       0.400   0.500
    ...
    -----------------------------------------------
    Mean Precision@5: 0.500
    MRR:              0.750

Exit codes:
    0  — eval completed
    1  — dataset not found or malformed
    2  — ChromaDB / embedding error
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Metric functions — pure, no I/O
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved chunk doc_ids that are relevant."""
    if not retrieved_doc_ids or k == 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / k


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    """1 / rank of the first relevant chunk; 0.0 if none found."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def mean_precision_at_k(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def mean_reciprocal_rank(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Retrieval — thin wrapper over apex.core.rag.store / embedder
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int,
    chroma_path: str,
    collection: str,
    api_key: str,
) -> list[str]:
    """
    Embed query and retrieve top_k chunks from ChromaDB.
    Returns list of doc_ids (one per chunk, in rank order).
    """
    try:
        from apex.core.rag.config import load_config
        from apex.core.rag import embedder, store
    except ImportError as exc:
        print(f"error: could not import apex.core.rag modules — install project dependencies: {exc}", file=sys.stderr)
        sys.exit(2)

    cfg = load_config(
        gemini_api_key=api_key,
        chroma_path=chroma_path,
        collection_name=collection,
        top_k=top_k,
        score_threshold=0.0,  # eval uses full ranking; threshold applied post-hoc by pipeline
    )

    query_vector = embedder.embed_query(query, cfg)
    results = store.query(query_vector, cfg)

    # results: list of dicts with keys 'text', 'metadata', 'score'
    return [r["metadata"]["doc_id"] for r in results]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict[str, Any]]:
    """
    Load eval dataset from JSON.

    Expected schema:
        [
          {
            "query": "what is the refund policy?",
            "relevant_doc_ids": ["policy-v2", "policy-v1"]
          },
          ...
        ]
    """
    p = Path(path)
    if not p.exists():
        print(f"error: dataset not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        print(f"error: malformed JSON in dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list) or not all("query" in d and "relevant_doc_ids" in d for d in data):
        print("error: dataset must be a list of {query, relevant_doc_ids} objects", file=sys.stderr)
        sys.exit(1)

    return data


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_table(results: list[dict], k: int, mp: float, mrr: float) -> None:
    col = 42
    header = f"{'Query':<{col}}  {'P@'+str(k):<8}  RR"
    print(header)
    print("-" * len(header))
    for r in results:
        q = r["query"][:col - 1].ljust(col)
        print(f"{q}  {r['precision']:<8.3f}  {r['rr']:.3f}")
    print("-" * len(header))
    print(f"Mean Precision@{k}: {mp:.3f}")
    print(f"MRR:               {mrr:.3f}")


def render_json(results: list[dict], k: int, mp: float, mrr: float) -> None:
    output = {
        "top_k": k,
        f"mean_precision_at_{k}": round(mp, 4),
        "mrr": round(mrr, 4),
        "queries": results,
    }
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate retrieval quality (precision@k, MRR) against a ground-truth dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="Path to eval dataset JSON")
    p.add_argument("--chroma-path", default=os.path.expanduser("~/.rag/chroma"), help="ChromaDB path")
    p.add_argument("--collection", default="documents", help="ChromaDB collection name")
    p.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per query (default: 5)")
    p.add_argument("--json", action="store_true", help="Output results as JSON")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        print("error: --top-k must be greater than 0", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("error: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    dataset = load_dataset(args.dataset)

    per_query: list[dict] = []
    for entry in dataset:
        query = entry["query"]
        relevant = set(entry["relevant_doc_ids"])

        retrieved = retrieve(
            query=query,
            top_k=args.top_k,
            chroma_path=args.chroma_path,
            collection=args.collection,
            api_key=api_key,
        )

        p_at_k = precision_at_k(retrieved, relevant, args.top_k)
        rr = reciprocal_rank(retrieved, relevant)

        per_query.append({
            "query": query,
            "relevant_doc_ids": sorted(relevant),
            "retrieved_doc_ids": retrieved,
            "precision": round(p_at_k, 4),
            "rr": round(rr, 4),
        })

    mp = mean_precision_at_k([r["precision"] for r in per_query])
    mrr = mean_reciprocal_rank([r["rr"] for r in per_query])

    if args.json:
        render_json(per_query, args.top_k, mp, mrr)
    else:
        render_table(per_query, args.top_k, mp, mrr)


if __name__ == "__main__":
    main()
