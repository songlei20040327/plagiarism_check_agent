"""
Document processing tools: PDF parsing, text chunking, embedding, and vector search.

Core pipeline:
  1. Parse PDFs into structured text blocks
  2. Split into overlapping chunks
  3. Embed with sentence-transformers
  4. Store in ChromaDB for similarity retrieval
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF

from ..config import Config

# ─── Tool definitions ─────────────────────────────────────────────────────────
# Tools are registered via ToolRegistry at module level (see bottom of file).
# This decouples tool definitions from the Agent layer — adding a new tool
# only requires a new method + a register() call; no changes needed in agent/.

# ─── In-memory document store ─────────────────────────────────────────────────

_document_store: dict[str, dict] = {}
_chunk_store: dict[str, dict] = {}
_embedder = None
_chroma_collection = None


def _get_embedder():
    """Lazy-load the sentence-transformers model."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        model_name = Config.EMBEDDING_MODEL
        device = Config.EMBEDDING_DEVICE
        print(f"  [Embed] Loading model: {model_name} (device={device}) ...")
        _embedder = SentenceTransformer(model_name, device=device)
        print(f"  [Embed] Model loaded. dim={_embedder.get_sentence_embedding_dimension()}")
    return _embedder


# ─── Tool implementations ─────────────────────────────────────────────────────


class DocumentTools:
    """Handler for document processing and vector search operations."""

    def parse_pdf(self, file_path: str) -> dict:
        """Parse a PDF and extract structured text."""
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        if path.suffix.lower() != ".pdf":
            return {"error": f"Not a PDF file: {file_path}"}

        try:
            doc = fitz.open(str(path))
            pages = []
            full_text_parts = []

            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                pages.append({
                    "page": page_num + 1,
                    "char_count": len(text),
                    "text": text,
                })
                full_text_parts.append(text)

            full_text = "\n".join(full_text_parts)

            # Generate a stable document ID
            doc_id = hashlib.md5(
                (path.name + full_text[:200]).encode()
            ).hexdigest()[:12]

            metadata = {
                "doc_id": doc_id,
                "file_path": str(path.absolute()),
                "file_name": path.name,
                "title": doc.metadata.get("title", path.stem),
                "author": doc.metadata.get("author", "unknown"),
                "page_count": len(pages),
                "word_count": len(full_text.replace("\n", " ").split()),
                "char_count": len(full_text),
            }

            # Store in memory
            _document_store[doc_id] = {
                "metadata": metadata,
                "pages": pages,
                "full_text": full_text,
            }

            doc.close()
            return metadata

        except Exception as e:
            return {"error": f"PDF parse error: {e}"}

    def chunk_document(self, document_id: str) -> dict:
        """Split a parsed document into overlapping chunks."""
        if document_id not in _document_store:
            return {"error": f"Document not found: {document_id}. Parse it first."}

        doc = _document_store[document_id]
        full_text = doc["full_text"]
        chunk_size = Config.CHUNK_SIZE
        overlap = Config.CHUNK_OVERLAP

        chunks = []
        # Split into paragraphs first, then merge into chunks
        paragraphs = [p.strip() for p in full_text.split("\n") if p.strip()]

        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n"
            else:
                if current_chunk.strip():
                    chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "index": chunk_index,
                        "text": current_chunk.strip(),
                        "char_count": len(current_chunk.strip()),
                    })
                    chunk_index += 1
                # Start new chunk with overlap
                if len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + para + "\n"
                else:
                    current_chunk = para + "\n"

        # Don't forget the last chunk
        if current_chunk.strip():
            chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "index": chunk_index,
                "text": current_chunk.strip(),
                "char_count": len(current_chunk.strip()),
            })

        # Store chunks
        for c in chunks:
            _chunk_store[c["chunk_id"]] = {
                **c,
                "file_name": doc["metadata"]["file_name"],
                "title": doc["metadata"]["title"],
                "author": doc["metadata"]["author"],
            }

        return {
            "document_id": document_id,
            "file_name": doc["metadata"]["file_name"],
            "total_chunks": len(chunks),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "avg_chunk_chars": sum(c["char_count"] for c in chunks) // max(len(chunks), 1),
        }

    def build_corpus_index(self, persist_dir: Optional[str] = None) -> dict:
        """Build vector index from all chunked documents using ChromaDB."""
        import chromadb
        from chromadb.utils import embedding_functions

        global _chroma_collection

        persist_dir = persist_dir or str(Config.VECTOR_DB_DIR)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        if not _chunk_store:
            return {"error": "No chunked documents found. Parse and chunk documents first."}

        embedder = _get_embedder()

        # Create a custom embedding function wrapper
        class STransformerEF(embedding_functions.EmbeddingFunction):
            def __call__(self, input):
                return embedder.encode(input, normalize_embeddings=True).tolist()

        client = chromadb.PersistentClient(path=persist_dir)

        # Only rebuild if collection doesn't exist or if explicitly requested
        try:
            existing = client.get_collection("reference_corpus")
            existing_count = existing.count()
            if existing_count > 0:
                _chroma_collection = existing
                unique_docs = len(set(
                    c.get("document_id") for c in _chunk_store.values()
                )) if _chunk_store else existing_count
                return {
                    "status": "ok",
                    "total_chunks_indexed": existing_count,
                    "unique_documents": unique_docs,
                    "message": "Using existing index (skip rebuild)",
                    "persist_dir": persist_dir,
                }
        except Exception:
            pass

        # No usable existing index — build from scratch
        try:
            client.delete_collection("reference_corpus")
        except Exception:
            pass

        collection = client.create_collection(
            name="reference_corpus",
            embedding_function=STransformerEF(),
            metadata={"hnsw:space": "cosine"},
        )

        # Batch insert
        chunk_ids = []
        documents = []
        metadatas = []
        batch_size = 100

        all_chunks = list(_chunk_store.values())
        total = len(all_chunks)

        for i, chunk in enumerate(all_chunks):
            chunk_ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])
            metadatas.append({
                "document_id": chunk["document_id"],
                "file_name": chunk.get("file_name", ""),
                "title": chunk.get("title", ""),
                "author": chunk.get("author", ""),
                "chunk_index": chunk["index"],
            })

            if len(chunk_ids) >= batch_size or i == total - 1:
                collection.add(
                    ids=chunk_ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                chunk_ids.clear()
                documents.clear()
                metadatas.clear()

        _chroma_collection = collection

        return {
            "status": "ok",
            "total_chunks_indexed": total,
            "unique_documents": len(set(c["document_id"] for c in all_chunks)),
            "persist_dir": persist_dir,
        }

    def search_similar_chunks(
        self,
        query_text: str,
        top_k: int = 5,
        threshold: float = 0.75,
    ) -> dict:
        """Search for similar chunks in the reference corpus."""
        global _chroma_collection

        if _chroma_collection is None:
            # Try to load existing index
            try:
                import chromadb

                persist_dir = str(Config.VECTOR_DB_DIR)
                client = chromadb.PersistentClient(path=persist_dir)
                try:
                    _chroma_collection = client.get_collection("reference_corpus")
                except Exception:
                    pass
            except Exception:
                pass

        if _chroma_collection is None:
            return {
                "error": (
                    "No vector index available. "
                    "Parse and chunk reference documents first, then call build_corpus_index."
                )
            }

        embedder = _get_embedder()
        query_embedding = embedder.encode(
            [query_text], normalize_embeddings=True
        ).tolist()

        results = _chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1 - distance  # cosine distance → similarity
                if similarity >= threshold:
                    meta = results["metadatas"][0][i]
                    matches.append({
                        "chunk_id": chunk_id,
                        "similarity": round(similarity, 4),
                        "matched_text": results["documents"][0][i][:500],
                        "source_paper": meta.get("title", meta.get("file_name", "unknown")),
                        "source_author": meta.get("author", "unknown"),
                        "document_id": meta.get("document_id", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                    })

        return {
            "query_length": len(query_text),
            "threshold": threshold,
            "matches_found": len(matches),
            "matches": matches,
        }

    def get_chunk_context(
        self, chunk_id: str, context_size: int = 2
    ) -> dict:
        """Get surrounding context for a matched chunk."""
        if chunk_id not in _chunk_store:
            # Try to reconstruct from document store
            doc_id = "_".join(chunk_id.split("_")[:-1])
            if doc_id in _document_store:
                return {
                    "chunk_id": chunk_id,
                    "error": "Chunk index was rebuilt. Re-chunk the document.",
                }
            return {"error": f"Chunk not found: {chunk_id}"}

        chunk = _chunk_store[chunk_id]
        doc_id = chunk["document_id"]
        chunk_idx = chunk["index"]

        # Find surrounding chunks from the same document
        same_doc_chunks = sorted(
            [
                c
                for c in _chunk_store.values()
                if c["document_id"] == doc_id
            ],
            key=lambda x: x["index"],
        )

        context_chunks = []
        for c in same_doc_chunks:
            if abs(c["index"] - chunk_idx) <= context_size:
                context_chunks.append({
                    "chunk_id": c["chunk_id"],
                    "index": c["index"],
                    "is_match": c["chunk_id"] == chunk_id,
                    "text": c["text"],
                })

        return {
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "source_paper": chunk.get("title", chunk.get("file_name", "")),
            "context_chunks": context_chunks,
        }

    def get_corpus_stats(self) -> dict:
        """Return statistics about the current corpus."""
        return {
            "documents_parsed": len(_document_store),
            "total_chunks": len(_chunk_store),
            "document_names": [
                d["metadata"]["file_name"]
                for d in _document_store.values()
            ],
            "index_built": _chroma_collection is not None,
        }

    def get_document_chunks(self, document_id: str) -> list[dict]:
        """Return all chunks for a document, sorted by chunk index.

        Public API — used by the orchestrator to extract thesis paragraphs
        without accessing the private _chunk_store directly.
        """
        chunks = [
            c for c in _chunk_store.values()
            if c.get("document_id") == document_id
        ]
        return sorted(chunks, key=lambda x: x.get("index", 0))


# ── Register all tools with the ToolRegistry ──────────────────────────────────
# This is the only place tool definitions live. Adding a new tool = new method
# + one register() call. No changes needed in agent/ or skills/.

from .registry import ToolRegistry  # noqa: E402

ToolRegistry.register(
    name="parse_pdf",
    description=(
        "Parse a PDF file and extract structured text content. "
        "Returns metadata (title, pages, word count) and the full text "
        "split into logical sections by page."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the PDF file",
            },
        },
        "required": ["file_path"],
    },
    handler_class=DocumentTools,
)

ToolRegistry.register(
    name="chunk_document",
    description=(
        "Split document text into overlapping chunks suitable for "
        "embedding and similarity search."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "Unique document identifier returned by parse_pdf",
            },
        },
        "required": ["document_id"],
    },
    handler_class=DocumentTools,
)

ToolRegistry.register(
    name="build_corpus_index",
    description=(
        "Build or rebuild the vector search index from all chunked "
        "reference documents. Must be called before search_similar_chunks."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "persist_dir": {
                "type": "string",
                "description": "Directory to persist the ChromaDB index",
            },
        },
    },
    handler_class=DocumentTools,
)

ToolRegistry.register(
    name="search_similar_chunks",
    description=(
        "Search the reference corpus for text chunks most similar to "
        "a query text. Returns top-K matches with source paper info, "
        "similarity scores, and the matched text for comparison."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query_text": {
                "type": "string",
                "description": "The text to search for (a paragraph from the thesis)",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top matches to return (default 5)",
            },
            "threshold": {
                "type": "number",
                "description": "Minimum similarity score 0-1 (default 0.75)",
            },
        },
        "required": ["query_text"],
    },
    handler_class=DocumentTools,
)

ToolRegistry.register(
    name="get_chunk_context",
    description=(
        "Get the surrounding context (adjacent paragraphs) for a matched "
        "chunk in a reference paper, to help the LLM make better judgments."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "chunk_id": {
                "type": "string",
                "description": "Chunk ID returned by search_similar_chunks",
            },
            "context_size": {
                "type": "integer",
                "description": "Number of surrounding chunks to include (default 2)",
            },
        },
        "required": ["chunk_id"],
    },
    handler_class=DocumentTools,
)

ToolRegistry.register(
    name="get_corpus_stats",
    description=(
        "Get statistics about the current reference corpus: total documents, "
        "total chunks, indexed status."
    ),
    input_schema={"type": "object", "properties": {}},
    handler_class=DocumentTools,
)
