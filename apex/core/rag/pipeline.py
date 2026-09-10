"""Public ingest and query interface for the in-process RAG pipeline."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from apex.core.rag import chunker, embedder, generator, store
from apex.core.rag.config import Config

_STRATEGIES = {"fixed", "sentences"}


def _chunks(text: str, config: Config, strategy: str) -> list[str]:
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of: {', '.join(sorted(_STRATEGIES))}")
    if strategy == "sentences":
        return chunker.chunk_sentences(text, config.chunk_size)
    return chunker.chunk_fixed(text, config.chunk_size, config.chunk_overlap)


def ingest(
    text: str,
    doc_id: str,
    config: Config,
    metadata: dict | None = None,
    strategy: str = "fixed",
) -> dict:
    """Chunk, embed, and replace one document in the vector store."""
    chunks = _chunks(text, config, strategy)
    if not chunks:
        store.delete_document(doc_id, config)
        return {"doc_id": doc_id, "chunks_stored": 0}

    embeddings = embedder.embed_texts(chunks, config)
    store.upsert(chunks, embeddings, doc_id, config, metadata)
    return {"doc_id": doc_id, "chunks_stored": len(chunks)}


def pipeline_query(question: str, config: Config) -> dict:
    """Retrieve matching chunks and generate a grounded answer."""
    query_embedding = embedder.embed_query(question, config)
    chunks = store.store_query(query_embedding, config)
    result = generator.generate_answer(question, chunks, config)
    return {**result, "chunks": chunks}


def ingest_file(
    path: str,
    config: Config,
    metadata: dict | None = None,
    strategy: str = "fixed",
    doc_id: str | None = None,
) -> dict:
    """Read and ingest one UTF-8 text file."""
    file_path = Path(path).expanduser()
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return ingest(
        text,
        doc_id=doc_id or file_path.name,
        config=config,
        metadata=metadata,
        strategy=strategy,
    )


def ingest_directory(
    directory: str,
    config: Config,
    extensions: list[str] | None = None,
    strategy: str = "fixed",
    max_workers: int = 4,
) -> list[dict]:
    """Recursively ingest matching files concurrently and return deterministic results."""
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than 0")
    _chunks("", config, strategy)  # Validate strategy before scanning.

    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ValueError(f"directory not found: {root}")
    suffixes = set(extensions or [".txt", ".md"])
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        return []

    store._get_client(str(Path(config.chroma_path).expanduser()))
    indexed_results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                ingest_file,
                str(path),
                config,
                None,
                strategy,
                path.relative_to(root).as_posix(),
            ): index
            for index, path in enumerate(paths)
        }
        for future in as_completed(futures):
            indexed_results[futures[future]] = future.result()
    return [indexed_results[index] for index in range(len(paths))]


def query(target: str | list[float], config: Config) -> dict | list[dict]:
    """Query by natural-language question or an already computed embedding."""
    if isinstance(target, list):
        return store.store_query(target, config)
    return pipeline_query(target, config)
