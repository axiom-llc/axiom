"""Generate answers constrained to retrieved RAG context."""
from google import genai
from google.genai import types

from apex.core.rag.config import Config

_DEFAULT_SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the question using ONLY the provided context passages.
If the context does not contain enough information to answer, say so clearly.
Do not use prior knowledge. Do not speculate beyond the context.
Cite the source document (doc_id) inline when referencing specific facts."""


def generate_answer(
    query: str,
    context_chunks: list[dict],
    config: Config,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> dict:
    """Generate a grounded answer from retrieved chunks."""
    if not context_chunks:
        return {
            "answer": "No relevant documents found for this query.",
            "sources": [],
            "chunk_count": 0,
        }

    config.requires_api_key()
    context_block = "\n\n".join(
        f"[{chunk['metadata'].get('doc_id', 'unknown')}] {chunk['text']}"
        for chunk in context_chunks
    )
    prompt = f"Context:\n{context_block}\n\nQuestion: {query}"
    client = genai.Client(api_key=config.gemini_api_key)
    response = client.models.generate_content(
        model=config.generation_model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    sources = sorted(
        {chunk["metadata"].get("doc_id", "unknown") for chunk in context_chunks}
    )
    return {
        "answer": response.text or "",
        "sources": sources,
        "chunk_count": len(context_chunks),
    }
