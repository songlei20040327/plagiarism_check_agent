"""Exercise real Chroma persistence with deterministic, offline embeddings."""
import hashlib
import json

import fitz
import numpy as np
import pytest

from plagiarism_check_agent.agent.tool_router import ToolRouter
from plagiarism_check_agent.config import Config
from plagiarism_check_agent.tools.document_tools import DocumentTools


class TinyEmbedder:
    def encode(self, texts, **kwargs):
        vectors = np.array([
            [1.0] + [b / 255 for b in hashlib.sha256(t.encode()).digest()[:7]]
            for t in texts
        ])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def documents(tmp_path):
    tools = DocumentTools(embedder=TinyEmbedder())

    def add(name, text):
        path = tmp_path / name
        with fitz.open() as pdf:
            page = pdf.new_page()
            page.insert_text((50, 50), text)
            pdf.save(path)
        metadata = tools.parse_pdf(str(path))
        assert "error" not in metadata
        assert tools.chunk_document(metadata["doc_id"])["total_chunks"] > 0
        return metadata["doc_id"]

    return tools, add


def test_reference_allowlist_excludes_thesis(documents, tmp_path):
    tools, add = documents
    thesis = add("thesis.pdf", "A unique thesis paragraph about pressure control.")
    ref = add("ref.pdf", "A reference paragraph about portable sensing devices.")
    result = tools.build_corpus_index([ref], str(tmp_path / "db"), [thesis])
    assert result["unique_documents"] == 1
    assert {m["document_id"] for m in tools._chroma_collection.get()["metadatas"]} == {ref}
    query = tools.get_document_chunks(thesis)[0]["text"]
    matches = tools.search_similar_chunks(query, threshold=0)["matches"]
    assert matches and all(m["document_id"] != thesis for m in matches)
    assert "error" in tools.build_corpus_index([ref, thesis], str(tmp_path / "db"), [thesis])


def test_snapshots_follow_corpus_and_configuration(documents, tmp_path, monkeypatch):
    tools, add = documents
    first = add("a.pdf", "First source document about pressure.")
    db = str(tmp_path / "db")
    initial = tools.build_corpus_index([first], db)
    assert tools.build_corpus_index([first], db)["reused"]
    second = add("b.pdf", "Second source document about sensors.")
    updated = tools.build_corpus_index([first, second], db)
    assert updated["snapshot_id"] != initial["snapshot_id"]
    assert updated["unique_documents"] == 2
    assert {m["document_id"] for m in tools._chroma_collection.get()["metadatas"]} == {first, second}
    manifest = json.loads((tmp_path / "db" / ("manifest_" + updated["snapshot_id"] + ".json")).read_text())
    assert manifest["snapshot_id"] == updated["snapshot_id"]
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "another-model-version")
    assert tools.build_corpus_index([first, second], db)["snapshot_id"] != updated["snapshot_id"]


def test_document_and_router_instances_do_not_share_state(documents):
    tools, add = documents
    add("a.pdf", "One task must not leak into another task.")
    other = DocumentTools()
    assert other.get_corpus_stats()["documents_parsed"] == 0
    router = ToolRouter(document_tools=tools)
    assert router.execute("get_corpus_stats", {})["documents_parsed"] == 1
    assert ToolRouter().execute("get_corpus_stats", {})["documents_parsed"] == 0


def test_document_identity_uses_entire_file(documents):
    tools, add = documents
    first = add("same.pdf", "A common prefix " * 20 + "old ending")
    second = add("same.pdf", "A common prefix " * 20 + "new ending")
    assert first != second
