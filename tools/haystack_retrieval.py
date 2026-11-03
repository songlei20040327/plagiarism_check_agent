"""Haystack's two-stage retrieval pattern adapted to our corpus snapshots.

Reference: https://haystack.deepset.ai/tutorials/33_hybrid_retrieval
1. Retrieve independently with BM25 and dense vectors.
2. Concatenate/deduplicate the full candidate pool, without mixing raw scores.
3. Score the original (query, passage) pairs with a Cross-Encoder, then take top-K.

Chroma remains the dense store. Haystack supplies BM25, DocumentJoiner and
SentenceTransformersSimilarityRanker; jieba supplies explicit Chinese tokenization.
"""
import logging
import math
import os
import re
import unicodedata
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from threading import Lock

# Local papers and retrieval queries should not trigger framework telemetry.
os.environ.setdefault('HAYSTACK_TELEMETRY_ENABLED', 'False')

import jieba
from haystack import Document
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import ComponentDevice
from huggingface_hub import list_repo_files, snapshot_download, try_to_load_from_cache

from ..config import Config


@lru_cache(maxsize=1)
def _tokenizer():
    jieba.setLogLevel(logging.WARNING)
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    return tokenizer


def tokenize(text: str) -> list[str]:
    """Keep Chinese words and English terms; do not join English words together."""
    text = unicodedata.normalize('NFKC', text).casefold()
    return [part.strip() for part in _tokenizer().cut(text, HMM=False)
            if re.search(r'\w', part, re.UNICODE)]


_RANKER_LOCK = Lock()


@lru_cache(maxsize=2)
def _load_ranker(model, revision, device, batch_size, max_length, cache_dir):
    # Haystack 2.x does not expose CrossEncoder.cache_folder. Resolve a pinned
    # local snapshot first, rather than forwarding duplicate cache_dir kwargs
    # through AutoConfig/AutoModel in newer sentence-transformers releases.
    config_path = try_to_load_from_cache(model, 'config.json', revision=revision, cache_dir=cache_dir)
    weights = [try_to_load_from_cache(model, name, revision=revision, cache_dir=cache_dir)
               for name in ('model.safetensors', 'pytorch_model.bin')]
    if isinstance(config_path, str) and any(isinstance(path, str) for path in weights):
        model_path = str(Path(config_path).parent)
    else:
        files = list_repo_files(model, revision=revision)
        weight_patterns = ['*.safetensors'] if any(name.endswith('.safetensors') for name in files) else ['pytorch_model*.bin']
        model_path = snapshot_download(
            model, revision=revision, cache_dir=cache_dir,
            allow_patterns=['*.json', '*.model', 'vocab.txt', 'merges.txt', *weight_patterns],
            ignore_patterns=['onnx/*', 'openvino/*'],
        )
    ranker = SentenceTransformersSimilarityRanker(
        model=model_path, device=ComponentDevice.from_str(device),
        scale_score=False, score_threshold=None, batch_size=batch_size,
        tokenizer_kwargs={'model_max_length': max_length},
    )
    ranker.warm_up()
    return ranker


def get_reranker():
    # Avoid two batch workers loading the same large model simultaneously.
    with _RANKER_LOCK:
        return _load_ranker(
            Config.RERANKER_MODEL, Config.RERANKER_REVISION, Config.RERANKER_DEVICE,
            Config.RERANKER_BATCH_SIZE, Config.RERANKER_MAX_LENGTH,
            str(Config.VECTOR_DB_DIR / '.models'),
        )


class HaystackCandidateRetriever:
    def __init__(self, chunks: dict[str, dict]):
        self.documents = {key: Document(id=key, content=chunk['text']) for key, chunk in chunks.items()}
        self.tokens = {key: set(tokenize(doc.content)) for key, doc in self.documents.items()}
        self.store = InMemoryDocumentStore(
            bm25_tokenization_regex=r'(?u)\S+', bm25_algorithm='BM25L', return_embedding=False,
        )
        self.store.write_documents([
            Document(id=key, content=' '.join(tokenize(doc.content)))
            for key, doc in self.documents.items()
        ])
        self.bm25_retriever = InMemoryBM25Retriever(self.store, scale_score=False)
        self.joiner = DocumentJoiner(join_mode='concatenate', sort_by_score=False)

    def bm25(self, query: str, top_k: int) -> dict[str, float]:
        tokens = tokenize(query)
        if not tokens:
            return {}
        result = self.bm25_retriever.run(query=' '.join(tokens), top_k=top_k)
        return {doc.id: float(doc.score) for doc in result['documents']
                if set(tokens) & self.tokens[doc.id] and doc.score is not None and doc.score > 0}

    def join(self, dense_ids: list[str], bm25_ids: list[str]) -> list[Document]:
        # BM25's tokenized text is never fed to the reranker: use original text.
        # No raw BM25/cosine score is used to truncate this union.
        return self.joiner.run(documents=[
            [self.documents[key] for key in dense_ids],
            [self.documents[key] for key in bm25_ids],
        ])['documents']

    def rank(self, query: str, candidates: list[Document], ranker) -> list[Document]:
        if not candidates:
            return []
        original = {doc.id: doc.content for doc in candidates}
        result = ranker.run(query=query, documents=candidates, top_k=len(candidates))
        ranked = result.get('documents')
        if not isinstance(ranked, list):
            raise ValueError('Reranker must return a document list.')
        seen = set()
        for doc in ranked:
            if doc.id not in original or doc.id in seen or doc.content != original[doc.id]:
                raise ValueError('Reranker changed candidate identity or source text.')
            if isinstance(doc.score, bool) or not isinstance(doc.score, (int, float)) or not math.isfinite(doc.score):
                raise ValueError('Reranker returned an invalid score.')
            seen.add(doc.id)
        if seen != original.keys():
            raise ValueError('Reranker omitted candidates before final top-K selection.')
        return sorted(ranked, key=lambda doc: (-doc.score, doc.id))

    def fuse_baseline(self, dense_ids, bm25_scores):
        """Optional non-neural baseline, using Haystack's RRF implementation."""
        joiner = DocumentJoiner(join_mode='reciprocal_rank_fusion')
        return joiner.run(documents=[
            [replace(self.documents[key], score=float(len(dense_ids) - i)) for i, key in enumerate(dense_ids)],
            [replace(self.documents[key], score=score) for key, score in bm25_scores.items()],
        ])['documents']

    def close(self):
        # Haystack's in-memory stores live in a process registry; release each run.
        self.store.delete_documents(list(self.documents))
        self.store.shutdown()
