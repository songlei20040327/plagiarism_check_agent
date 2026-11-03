"""Run actual Haystack BM25/join/ranker components with controlled model outputs."""
from dataclasses import replace

import fitz
import numpy as np
import pytest
from haystack import Document
from haystack.components.rankers import SentenceTransformersSimilarityRanker

from plagiarism_check_agent.config import Config
from plagiarism_check_agent.tools.document_tools import DocumentTools
from plagiarism_check_agent.tools.haystack_retrieval import HaystackCandidateRetriever, tokenize
from plagiarism_check_agent.agent.tool_router import ToolRouter


QUERY = 'Portable ventilator pressure control algorithm verification.'


class ControlledEmbedder:
    def encode(self, texts, **kwargs):
        return np.array([[0., 1., 0.] if text.startswith('SOURCE:') else
                         [0., 0., 1.] if text.startswith('OTHER:') else
                         [1., 0., 0.] for text in texts])


class ControlledCrossEncoder:
    def __init__(self):
        self.seen = []

    def rank(self, query, documents, **kwargs):
        self.seen = list(documents)
        scores = [4.0 if text.startswith('SOURCE:') else -2.0 for text in documents]
        return [{'corpus_id': i, 'score': scores[i]}
                for i in sorted(range(len(documents)), key=lambda i: -scores[i])]


def offline_ranker():
    # Only the neural computation is replaced. Haystack.run and its score,
    # document and deduplication handling are exercised unchanged.
    ranker = SentenceTransformersSimilarityRanker(model='offline-test', scale_score=False)
    ranker._cross_encoder = ControlledCrossEncoder()
    return ranker


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, 'RETRIEVAL_MODE', 'hybrid_rerank')
    monkeypatch.setattr(Config, 'RETRIEVAL_POOL_SIZE', 20)
    tools = DocumentTools(embedder=ControlledEmbedder(), ranker=offline_ranker())

    def add(name, text):
        path = tmp_path / name
        with fitz.open() as document:
            page = document.new_page()
            page.insert_text((50, 50), text)
            document.save(path)
        meta = tools.parse_pdf(str(path))
        tools.chunk_document(meta['doc_id'])
        return meta['doc_id']

    thesis = add('thesis.pdf', QUERY)
    source = add('source.pdf', 'SOURCE: ' + QUERY)
    semantic = add('semantic.pdf', 'SEMANTIC: An intentionally close vector but weak evidence.')
    other = add('other.pdf', 'OTHER: Zebra mango jewel quickbox.')
    assert tools.build_corpus_index([source, semantic, other], str(tmp_path / 'db'), [thesis])['status'] == 'ok'
    yield tools, thesis, source, semantic, other, tmp_path
    tools.close()


def test_tokenizer_handles_chinese_and_english():
    tokens = tokenize('呼吸机压力控制，ＰＩＤ controller')
    assert '呼吸机' in tokens and '压力' in tokens
    assert 'pid' in tokens and 'controller' in tokens
    assert 'pidcontroller' not in tokens
    assert tokenize('  ，。!? ') == []


def test_haystack_bm25_and_joiner_use_original_content():
    chunks = {'a': {'text': '呼吸机压力控制 PID controller'},
              'b': {'text': '海洋生物种群演化'}}
    retriever = HaystackCandidateRetriever(chunks)
    try:
        scores = retriever.bm25('呼吸机 PID', 5)
        assert list(scores) == ['a']
        assert retriever.bm25('totally absent words', 5) == {}
        docs = retriever.join(['a', 'b'], list(scores))
        assert len(docs) == 2
        assert next(doc.content for doc in docs if doc.id == 'a') == chunks['a']['text']
    finally:
        retriever.close()


def test_reranker_scores_entire_union_before_final_topk(corpus):
    tools, thesis, source, semantic, other, _ = corpus
    result = tools.search_similar_chunks(QUERY, top_k=1)
    diagnostics = result['retrieval_diagnostics']
    assert len(diagnostics['union_ids']) == 3
    assert len(diagnostics['reranked_ids']) == 3
    assert len(tools._ranker._cross_encoder.seen) == 3
    assert result['matches'][0]['document_id'] == source
    assert result['matches'][0]['scores']['reranker'] == 4
    assert result['matches'][0]['similarity'] == 0
    assert thesis not in {c['document_id'] for c in tools._indexed_chunks.values()}


def test_bm25_candidate_outside_dense_pool_is_reranked(corpus, monkeypatch):
    tools, _, source, _, _, _ = corpus
    monkeypatch.setattr(Config, 'RETRIEVAL_POOL_SIZE', 1)
    result = tools.search_similar_chunks(QUERY, top_k=1)
    match = result['matches'][0]
    assert match['document_id'] == source
    assert match['retrieval_channels'] == ['bm25']
    assert match['similarity'] == 0
    assert len(result['retrieval_diagnostics']['union_ids']) == 2
    assert 'SOURCE: ' + QUERY in tools._ranker._cross_encoder.seen


def test_hybrid_has_no_early_dense_threshold(corpus):
    tools, _, _, _, _, _ = corpus
    result = tools.search_similar_chunks(QUERY, threshold=1)
    assert len(result['retrieval_diagnostics']['dense_ids']) == 3
    assert result['threshold_scope'] == 'dense_baseline_only'
    assert result['retrieval_config']['merge'] == 'concatenate'


def test_dense_and_unreranked_hybrid_are_explicit_baselines(corpus, monkeypatch):
    tools, _, source, semantic, _, _ = corpus
    monkeypatch.setattr(Config, 'RETRIEVAL_MODE', 'dense')
    result = tools.search_similar_chunks(QUERY)
    assert {m['document_id'] for m in result['matches']} == {semantic}
    assert all('reranker' not in m['scores'] for m in result['matches'])
    monkeypatch.setattr(Config, 'RETRIEVAL_MODE', 'hybrid')
    result = tools.search_similar_chunks(QUERY)
    assert source in {m['document_id'] for m in result['matches']}
    assert all('reranker' not in m['scores'] for m in result['matches'])
    assert result['retrieval_config']['merge'] == 'rrf'


def test_rerank_is_deterministic_and_candidates_are_unique(corpus):
    tools, _, _, _, _, _ = corpus
    first = tools.search_similar_chunks(QUERY)
    second = tools.search_similar_chunks(QUERY)
    assert first == second
    ids = first['retrieval_diagnostics']['union_ids']
    assert len(ids) == len(set(ids))


def test_reranker_failure_is_not_silently_downgraded(corpus):
    tools, thesis, _, _, _, _ = corpus
    class Broken:
        def run(self, **kwargs):
            raise RuntimeError('simulated inference failure')
    tools._ranker = Broken()
    router = ToolRouter(document_tools=tools)
    query_id = tools.get_document_chunks(thesis)[0]['chunk_id']
    result = router.execute('search_similar_chunks', {'query_text': QUERY})
    assert 'error' in result and 'simulated inference failure' in result['error']


@pytest.mark.parametrize('failure', ['missing', 'foreign', 'nan', 'changed_text'])
def test_invalid_ranker_output_is_rejected(corpus, failure):
    tools, _, _, _, _, _ = corpus
    class Invalid:
        def run(self, query, documents, **kwargs):
            docs = [replace(doc, score=1.0) for doc in documents]
            if failure == 'missing': docs.pop()
            if failure == 'foreign': docs[0] = Document(id='outside', content='bad', score=1.)
            if failure == 'nan': docs[0] = replace(docs[0], score=float('nan'))
            if failure == 'changed_text': docs[0] = replace(docs[0], content='fabricated')
            return {'documents': docs}
    tools._ranker = Invalid()
    with pytest.raises(ValueError):
        tools.search_similar_chunks(QUERY)


def test_scores_and_config_are_preserved_in_verified_report(corpus):
    tools, thesis, source, _, _, tmp = corpus
    router = ToolRouter(document_tools=tools)
    router.bind_task(thesis, str(tmp / 'report'))
    thesis_chunk = tools.get_document_chunks(thesis)[0]['chunk_id']
    result = router.execute('search_similar_chunks', {'thesis_chunk_id': thesis_chunk})
    assert 'error' not in result
    findings = [{'thesis_chunk_id': thesis_chunk, 'source_chunk_id': m['chunk_id'],
                 'duplicate_type': 'no_issue', 'suggestion': ''} for m in result['matches']]
    report = router.execute('generate_report', {'thesis_title': 'Haystack test', 'findings': findings,
                                               'output_dir': str(tmp / 'report')})
    assert report['status'] == 'ok'
    assert router.completion_state() == ('completed', None)
    import json
    from pathlib import Path
    data = json.loads(Path(report['json_data']).read_text())
    assert data['retrieval_config']['mode'] == 'hybrid_rerank'
    assert data['findings'][0]['retrieval_scores']['reranker'] == 4
    html = Path(report['html_report']).read_text()
    assert '重排相关性分数' in html and '非重复概率' in html
