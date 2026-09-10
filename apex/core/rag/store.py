"""APEX import compatibility for the canonical axiom-rag store."""
from rag.store import (
    _get_client,
    _get_collection,
    upsert,
    store_query,
    query,
    delete_document,
    list_documents,
    collection_stats,
)
