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

_embedder = None


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

    def __init__(self, embedder=None):
        self._document_store: dict[str, dict] = {}
        self._chunk_store: dict[str, dict] = {}
        self._chroma_collection = None
        self._embedder = embedder
        self.reference_document_ids: set[str] = set()
        self.excluded_document_ids: set[str] = set()
        self.snapshot_id = ""
        self._indexed_chunks: dict[str, dict] = {}
        self._lexical_index = None

    def _encode(self, texts):
        model = self._embedder if self._embedder is not None else _get_embedder()
        return model.encode(texts, normalize_embeddings=True).tolist()

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
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            doc_id = file_hash[:24]

            metadata = {
                "doc_id": doc_id,
                "sha256": file_hash,
                "file_path": str(path.absolute()),
                "file_name": path.name,
                "title": doc.metadata.get("title", path.stem),
                "author": doc.metadata.get("author", "unknown"),
                "page_count": len(pages),
                "word_count": len(full_text.replace("\n", " ").split()),
                "char_count": len(full_text),
            }

            # Store in memory
            self._document_store[doc_id] = {
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
        if document_id not in self._document_store:
            return {"error": f"Document not found: {document_id}. Parse it first."}

        doc = self._document_store[document_id]
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

        # Re-chunking replaces the previous version, including removed tail chunks.
        self._chunk_store = {k: v for k, v in self._chunk_store.items()
                             if v["document_id"] != document_id}
        for c in chunks:
            self._chunk_store[c["chunk_id"]] = {
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

    def build_corpus_index(
        self, reference_document_ids: Optional[list[str]] = None,
        persist_dir: Optional[str] = None,
        excluded_document_ids: Optional[list[str]] = None,
    ) -> dict:
        """Build an immutable snapshot from an explicit reference allowlist."""
        import chromadb

        self._chroma_collection = None
        self.snapshot_id = ""
        self._indexed_chunks = {}
        self._lexical_index = None
        self.reference_document_ids = set(reference_document_ids or [])
        self.excluded_document_ids = set(excluded_document_ids or [])
        refs = self.reference_document_ids
        if not refs:
            return {"error": "An explicit, non-empty reference_document_ids list is required."}
        if refs & self.excluded_document_ids:
            return {"error": "The thesis must not be included in the reference corpus."}
        if not refs <= self._document_store.keys():
            return {"error": "Reference documents must be parsed in this run."}
        all_chunks = sorted(
            (c for c in self._chunk_store.values() if c["document_id"] in refs),
            key=lambda c: c["chunk_id"],
        )
        if {c["document_id"] for c in all_chunks} != refs:
            return {"error": "Every reference document must contain non-empty chunks."}
        manifest = {
            "version": 1,
            "embedding_model": Config.EMBEDDING_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "documents": [self._document_store[k]["metadata"] for k in sorted(refs)],
            "chunks": [{"id": c["chunk_id"],
                        "sha256": hashlib.sha256(c["text"].encode()).hexdigest()}
                       for c in all_chunks],
        }
        # Absolute file locations do not affect corpus identity.
        manifest["documents"] = [
            {k: v for k, v in meta.items() if k != "file_path"}
            for meta in manifest["documents"]
        ]
        snapshot_id = hashlib.sha256(json.dumps(
            manifest, ensure_ascii=False, sort_keys=True
        ).encode()).hexdigest()
        persist_dir = persist_dir or str(Config.VECTOR_DB_DIR)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(
            name="refs_" + snapshot_id[:48],
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "snapshot_id": snapshot_id},
        )
        expected_ids = {c["chunk_id"] for c in all_chunks}
        existing_ids = set(collection.get(include=[])["ids"])
        reused = existing_ids == expected_ids
        if not reused:
            if existing_ids - expected_ids:
                return {"error": "Snapshot index contains unexpected chunks; use a fresh cache directory."}
            for i in range(0, len(all_chunks), 100):
                batch = all_chunks[i:i + 100]
                collection.upsert(
                    ids=[c["chunk_id"] for c in batch],
                    documents=[c["text"] for c in batch],
                    embeddings=self._encode([c["text"] for c in batch]),
                    metadatas=[{
                        "document_id": c["document_id"], "file_name": c["file_name"],
                        "title": c.get("title") or c["file_name"],
                        "author": c.get("author", ""), "chunk_index": c["index"],
                    } for c in batch],
                )
        # Each snapshot has its own manifest; another task cannot replace it.
        manifest_path = Path(persist_dir) / ("manifest_" + snapshot_id + ".json")
        manifest_path.write_text(json.dumps(
            {"snapshot_id": snapshot_id, **manifest}, ensure_ascii=False, indent=2
        ), encoding="utf-8")
        self._chroma_collection = collection
        self.snapshot_id = snapshot_id
        self._indexed_chunks = {c["chunk_id"]: c.copy() for c in all_chunks}
        return {"status": "ok", "snapshot_id": snapshot_id,
                "total_chunks_indexed": len(all_chunks), "unique_documents": len(refs),
                "reused": reused, "persist_dir": persist_dir}

    def retrieval_config(self) -> dict:
        return {
            "mode": Config.RETRIEVAL_MODE, "top_k": Config.TOP_K_RESULTS,
            "pool_size": (max(Config.TOP_K_RESULTS, Config.RETRIEVAL_POOL_SIZE)
                          if Config.RETRIEVAL_MODE == "hybrid" else Config.TOP_K_RESULTS),
            "dense_threshold": Config.SIMILARITY_THRESHOLD,
            "literal_threshold": Config.LITERAL_RECALL_THRESHOLD,
            "literal_strong_threshold": Config.LITERAL_STRONG_THRESHOLD,
            "rrf_rank_constant": Config.RRF_RANK_CONSTANT,
            "lexical_version": "character-bigram-bm25+5gram+13gram-v1",
        }

    def search_similar_chunks(
        self, query_text: str, top_k: int = 5, threshold: float = 0.75,
    ) -> dict:
        """Recall independently by dense, BM25 and literal signals, then fuse."""
        if self._chroma_collection is None:
            return {"error": "No reference snapshot is available. Build the index first."}
        if Config.RETRIEVAL_MODE not in {"dense", "hybrid"}:
            return {"error": "RETRIEVAL_MODE must be dense or hybrid."}
        if top_k < 1 or not 0 <= threshold <= 1:
            return {"error": "Invalid top_k or dense threshold."}
        if (Config.RETRIEVAL_POOL_SIZE < 1 or Config.RRF_RANK_CONSTANT < 1
                or not 0 < Config.LITERAL_RECALL_THRESHOLD <= 1
                or not 0 < Config.LITERAL_STRONG_THRESHOLD <= 1):
            return {"error": "Invalid hybrid retrieval configuration."}
        from .hybrid_retrieval import LexicalIndex, reciprocal_rank_fusion
        hybrid = Config.RETRIEVAL_MODE == "hybrid"
        pool_size = max(top_k, Config.RETRIEVAL_POOL_SIZE) if hybrid else top_k
        query_embedding = self._encode([query_text])
        results = self._chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=min(pool_size, self._chroma_collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        dense_scores = {}
        for i, chunk_id in enumerate(results["ids"][0]):
            if results["metadatas"][0][i]["document_id"] not in self.reference_document_ids:
                raise ValueError("Dense index returned a document outside this snapshot.")
            dense_scores[chunk_id] = max(-1.0, min(1.0, 1 - results["distances"][0][i]))
        rankings = {"dense": [key for key, score in dense_scores.items() if score >= threshold]}
        bm25_scores, literal_scores = {}, {}
        if hybrid:
            if self._lexical_index is None:
                self._lexical_index = LexicalIndex({key: c["text"] for key, c in self._indexed_chunks.items()})
            bm25_scores = dict(self._lexical_index.bm25(query_text, pool_size))
            literal_scores = dict(self._lexical_index.literal(
                query_text, pool_size, Config.LITERAL_RECALL_THRESHOLD,
            ))
            rankings.update({"bm25": list(bm25_scores), "literal": list(literal_scores)})
        fused = reciprocal_rank_fusion(rankings, Config.RRF_RANK_CONSTANT)
        long_scores = {
            key: self._lexical_index.long_literal_score(query_text, key) if hybrid else 0.0
            for key in fused
        }
        # Preserve strong literal evidence before filling the remaining RRF slots.
        chosen = sorted(fused, key=lambda key: (
            -(long_scores[key] >= Config.LITERAL_STRONG_THRESHOLD), -fused[key], key,
        ))[:top_k]
        # Lexical candidates are not required to pass the dense recall gate.
        # Compute their dense score for explanation, without applying that gate again.
        extra = [key for key in chosen if key not in dense_scores]
        if extra:
            stored = self._chroma_collection.get(ids=extra, include=["embeddings"])
            query = query_embedding[0]
            qnorm = sum(v * v for v in query) ** .5
            for key, vector in zip(stored["ids"], stored["embeddings"]):
                norm = qnorm * sum(v * v for v in vector) ** .5
                if not norm or len(vector) != len(query):
                    raise ValueError("Invalid reference embedding.")
                dense_scores[key] = max(-1.0, min(1.0, float(sum(a * b for a, b in zip(query, vector)) / norm)))
        matches = []
        for key in chosen:
            chunk = self._indexed_chunks[key]
            matches.append({
                "chunk_id": key, "document_id": chunk["document_id"],
                "chunk_index": chunk["index"],
                "source_paper": chunk.get("title") or chunk["file_name"],
                "source_author": chunk.get("author", "unknown"),
                "matched_text": chunk["text"][:500],
                "similarity": round(dense_scores[key], 4),
                "retrieval_channels": [name for name, ranking in rankings.items() if key in ranking],
                "scores": {"dense": round(dense_scores[key], 4),
                           "bm25": bm25_scores.get(key, 0.0),
                           "literal": literal_scores.get(key, 0.0),
                           "literal_13": long_scores[key], "rrf": fused[key]},
            })
        return {
            "snapshot_id": self.snapshot_id, "query_length": len(query_text),
            "threshold": threshold, "threshold_scope": "dense_only",
            "retrieval_config": {**self.retrieval_config(), "top_k": top_k,
                                 "dense_threshold": threshold, "pool_size": pool_size},
            "matches_found": len(matches), "matches": matches,
        }

    def get_chunk_context(
        self, chunk_id: str, context_size: int = 2
    ) -> dict:
        """Get surrounding context for a matched chunk."""
        if chunk_id not in self._chunk_store:
            # Try to reconstruct from document store
            doc_id = "_".join(chunk_id.split("_")[:-1])
            if doc_id in self._document_store:
                return {
                    "chunk_id": chunk_id,
                    "error": "Chunk index was rebuilt. Re-chunk the document.",
                }
            return {"error": f"Chunk not found: {chunk_id}"}

        chunk = self._chunk_store[chunk_id]
        doc_id = chunk["document_id"]
        chunk_idx = chunk["index"]

        # Find surrounding chunks from the same document
        same_doc_chunks = sorted(
            [
                c
                for c in self._chunk_store.values()
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
            "documents_parsed": len(self._document_store),
            "total_chunks": len(self._chunk_store),
            "document_names": [
                d["metadata"]["file_name"]
                for d in self._document_store.values()
            ],
            "index_built": self._chroma_collection is not None,
        }

    def get_document_chunks(self, document_id: str) -> list[dict]:
        """Return all chunks for a document, sorted by chunk index.

        Public API — used by the orchestrator to extract thesis paragraphs
        without accessing the private self._chunk_store directly.
        """
        chunks = [
            c for c in self._chunk_store.values()
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
            "reference_document_ids": {
                "type": "array", "items": {"type": "string"}, "minItems": 1,
                "description": "Explicit reference document IDs; never include the thesis",
            },
            "excluded_document_ids": {
                "type": "array", "items": {"type": "string"},
            },
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
        "Retrieve reference candidates using dense, BM25 and literal recall (or dense baseline). "
        "Candidates may have a low dense similarity and strong lexical evidence. "
        "Review every returned match. similarity is cosine similarity, never duplicate probability."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "thesis_chunk_id": {
                "type": "string", "description": "Exact thesis chunk ID from the current task",
            },
            "query_text": {
                "type": "string",
                "description": "The text to search for (a paragraph from the thesis)",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1, "maximum": 50,
                "description": "Number of top matches to return (default 5)",
            },
            "threshold": {
                "type": "number",
                "minimum": 0, "maximum": 1,
                "description": "Dense-route recall gate only; lexical candidates bypass this gate (default 0.75)",
            },
        },
        "anyOf": [{"required": ["query_text"]}, {"required": ["thesis_chunk_id"]}],
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
                "minimum": 0,
                "maximum": 5,
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
