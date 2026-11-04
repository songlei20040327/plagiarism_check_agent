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
from pathlib import Path
from typing import Optional
from threading import Lock

import fitz  # PyMuPDF

from ..config import Config

# ─── Tool definitions ─────────────────────────────────────────────────────────
# Tools are registered via ToolRegistry at module level (see bottom of file).
# This decouples tool definitions from the Agent layer — adding a new tool
# only requires a new method + a register() call; no changes needed in agent/.

# ─── In-memory document store ─────────────────────────────────────────────────

_embedder = None
_EMBEDDER_LOCK = Lock()
_INDEX_LOCK = Lock()


def _get_embedder():
    """Lazy-load the sentence-transformers model."""
    global _embedder
    with _EMBEDDER_LOCK:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(Config.EMBEDDING_MODEL, revision=Config.EMBEDDING_REVISION,
                                           device=Config.EMBEDDING_DEVICE)
    return _embedder


# ─── Tool implementations ─────────────────────────────────────────────────────


class DocumentTools:
    """Handler for document processing and vector search operations."""

    def __init__(self, embedder=None, ranker=None):
        self._document_store: dict[str, dict] = {}
        self._chunk_store: dict[str, dict] = {}
        self._chroma_collection = None
        self._embedder = embedder
        self.reference_document_ids: set[str] = set()
        self.excluded_document_ids: set[str] = set()
        self.snapshot_id = ""
        self._indexed_chunks: dict[str, dict] = {}
        self._candidate_retriever = None
        self._ranker = ranker

    def _encode(self, texts):
        model = self._embedder if self._embedder is not None else _get_embedder()
        return model.encode(texts, normalize_embeddings=True).tolist()

    def close(self):
        """Release a task's temporary lexical index; retain parsed evidence."""
        if self._candidate_retriever is not None:
            self._candidate_retriever.close()
            self._candidate_retriever = None

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
        if self._candidate_retriever is not None:
            self._candidate_retriever.close()
        self._candidate_retriever = None
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
            "embedding_revision": Config.EMBEDDING_REVISION,
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
        # Chroma initialization and the check/upsert transaction must be
        # serialized for workers sharing a local persistence directory.
        with _INDEX_LOCK:
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
        mode = Config.RETRIEVAL_MODE
        result = {
            "mode": mode, "top_k": Config.TOP_K_RESULTS,
            "pool_size": max(Config.TOP_K_RESULTS, Config.RETRIEVAL_POOL_SIZE) if mode != "dense" else Config.TOP_K_RESULTS,
            "dense_baseline_threshold": Config.SIMILARITY_THRESHOLD,
            "bm25_implementation": "haystack-ai-2.31.1/BM25L",
            "tokenizer": "jieba-0.42.1/HMM=False+NFKC+casefold",
            "merge": "concatenate" if mode == "hybrid_rerank" else "rrf" if mode == "hybrid" else "none",
        }
        if mode == "hybrid_rerank":
            result.update({"reranker_model": Config.RERANKER_MODEL,
                           "reranker_revision": Config.RERANKER_REVISION,
                           "reranker_max_length": Config.RERANKER_MAX_LENGTH,
                           "reranker_score_type": "raw_logit_not_probability",
                           "reranker_strategy": "complete_windows_max_v1",
                           "reranker_window_overlap": Config.RERANKER_WINDOW_OVERLAP,
                           "auto_review_gates": Config.AUTO_REVIEW_GATES,
                           "reranker_pass_threshold": Config.RERANKER_PASS_THRESHOLD})
        return result

    def search_similar_chunks(
        self, query_text: str, top_k: int = 5, threshold: float = 0.75,
    ) -> dict:
        """Haystack pattern: independent recall -> full union -> Cross-Encoder."""
        if self._chroma_collection is None:
            return {"error": "No reference snapshot is available. Build the index first."}
        mode = Config.RETRIEVAL_MODE
        if mode not in {"dense", "hybrid", "hybrid_rerank"}:
            return {"error": "RETRIEVAL_MODE must be dense, hybrid, or hybrid_rerank."}
        if top_k < 1 or not 0 <= threshold <= 1 or Config.RETRIEVAL_POOL_SIZE < 1:
            return {"error": "Invalid retrieval configuration."}
        if mode == "hybrid_rerank" and (Config.RERANKER_BATCH_SIZE < 1 or Config.RERANKER_MAX_LENGTH < 1):
            return {"error": "Invalid reranker configuration."}
        pool_size = max(top_k, Config.RETRIEVAL_POOL_SIZE) if mode != "dense" else top_k
        query_embedding = self._encode([query_text])
        results = self._chroma_collection.query(
            query_embeddings=query_embedding, n_results=min(pool_size, self._chroma_collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        dense_scores = {}
        for i, chunk_id in enumerate(results["ids"][0]):
            if results["metadatas"][0][i]["document_id"] not in self.reference_document_ids:
                raise ValueError("Dense index returned a document outside this snapshot.")
            dense_scores[chunk_id] = max(-1.0, min(1.0, 1 - results["distances"][0][i]))
        # The old threshold is retained for the dense baseline only. Hybrid
        # candidates are selected by top-N, then jointly reranked, like Haystack.
        dense_ids = [key for key, score in dense_scores.items() if mode != "dense" or score >= threshold]
        bm25_scores, reranker_scores, fusion_scores, reranker_metadata = {}, {}, {}, {}
        diagnostics = {"dense_ids": dense_ids}
        if mode != "dense":
            from .haystack_retrieval import HaystackCandidateRetriever, get_reranker
            if self._candidate_retriever is None:
                self._candidate_retriever = HaystackCandidateRetriever(self._indexed_chunks)
            retriever = self._candidate_retriever
            bm25_scores = retriever.bm25(query_text, pool_size)
            candidates = retriever.join(dense_ids, list(bm25_scores))
            diagnostics.update({"bm25_ids": list(bm25_scores), "union_ids": [doc.id for doc in candidates]})
            if mode == "hybrid_rerank":
                # Rank ALL union candidates before taking top-K. Errors propagate;
                # never silently claim the unreranked result is reranked.
                ranker = self._ranker if self._ranker is not None else get_reranker()
                ranked = retriever.rank(query_text, candidates, ranker) if candidates else []
                reranker_scores = {doc.id: float(doc.score) for doc in ranked}
                reranker_metadata = {doc.id: doc.meta for doc in ranked}
                diagnostics["reranked_ids"] = [doc.id for doc in ranked]
            else:
                ranked = retriever.fuse_baseline(dense_ids, bm25_scores)
                fusion_scores = {doc.id: float(doc.score) for doc in ranked}
            chosen = [doc.id for doc in ranked[:top_k]]
        else:
            chosen = dense_ids[:top_k]
            diagnostics["union_ids"] = dense_ids
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
            scores = {"dense": round(dense_scores[key], 4)}
            if mode != "dense": scores["bm25"] = bm25_scores.get(key, 0.0)
            if key in reranker_scores: scores["reranker"] = reranker_scores[key]
            if key in fusion_scores: scores["rrf"] = fusion_scores[key]
            matches.append({
                "chunk_id": key, "document_id": chunk["document_id"], "chunk_index": chunk["index"],
                "source_paper": chunk.get("title") or chunk["file_name"],
                "source_author": chunk.get("author", "unknown"), "matched_text": chunk["text"][:500],
                "similarity": scores["dense"],
                "retrieval_channels": (["dense"] if key in dense_ids else []) + (["bm25"] if key in bm25_scores else []),
                "scores": scores,
                **({"reranker_window_pairs": reranker_metadata[key]['reranker_window_pairs'],
                    "reranker_complete_input": reranker_metadata[key]['reranker_complete_input']}
                   if key in reranker_metadata else {}),
            })
        diagnostics["selected_ids"] = chosen
        return {
            "snapshot_id": self.snapshot_id, "query_length": len(query_text),
            "threshold": threshold, "threshold_scope": "dense_baseline_only",
            "retrieval_config": {**self.retrieval_config(), "top_k": top_k, "pool_size": pool_size,
                                 "dense_baseline_threshold": threshold},
            "matches_found": len(matches), "matches": matches,
            "retrieval_diagnostics": diagnostics,
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
        "Retrieve reference candidates using BM25+dense recall followed by Cross-Encoder reranking. "
        "Candidates may have low cosine scores; reranker scores measure relevance, not plagiarism. "
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
                "description": "Only used in dense baseline mode; hybrid modes recall top-N then rank (default 0.75)",
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
