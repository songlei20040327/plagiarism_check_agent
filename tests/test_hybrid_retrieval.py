"""Validate route complementarity with real persistence and controlled vectors.

These are correctness regressions, not measurements of a real embedding model.
"""
import fitz
import numpy as np
import pytest

from plagiarism_check_agent.config import Config
from plagiarism_check_agent.tools.document_tools import DocumentTools
from plagiarism_check_agent.tools.hybrid_retrieval import (
    LexicalIndex, normalize_text, reciprocal_rank_fusion,
)
from plagiarism_check_agent.agent.tool_router import ToolRouter


QUERY = 'Portable ventilator pressure control algorithm verification.'


class ControlledEmbedder:
    def encode(self, texts, **kwargs):
        # Force literal evidence to have a low dense score to expose bad gating.
        return np.array([[0., 1., 0.] if text.startswith('SOURCE:') else
                         [0., 0., 1.] if text.startswith('OTHER:') else
                         [1., 0., 0.] for text in texts])


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, 'RETRIEVAL_MODE', 'hybrid')
    monkeypatch.setattr(Config, 'RETRIEVAL_POOL_SIZE', 20)
    tools = DocumentTools(embedder=ControlledEmbedder())

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
    literal = add('literal.pdf', 'SOURCE: ' + QUERY)
    semantic = add('semantic.pdf', 'SEMANTIC: An unrelated sentence with an intentionally close vector.')
    other = add('other.pdf', 'OTHER: Zebra mango jewel quickbox.')
    refs = [literal, semantic, other]
    result = tools.build_corpus_index(refs, str(tmp_path / 'db'), [thesis])
    assert result['status'] == 'ok'
    return tools, thesis, literal, semantic, other, tmp_path


def test_rank_fusion_uses_rank_and_deduplicates_each_route():
    scores = reciprocal_rank_fusion({'dense': ['a', 'b'], 'bm25': ['b', 'b', 'c']}, 60)
    assert scores['b'] == pytest.approx(1 / 62 + 1 / 61)
    assert scores['a'] == pytest.approx(1 / 61)
    assert scores['c'] == pytest.approx(1 / 62)
    assert reciprocal_rank_fusion({'empty': []}) == {}


def test_normalization_and_literal_containment():
    index = LexicalIndex({'source': '基 于 呼吸机控制的压力检测方案。', 'other': '海洋生物种群演化模型。'})
    assert normalize_text('ＰＩＤ 控制') == 'pid控制'
    assert index.literal('基于呼吸机控制的压力检测方案。', 5, .35) == [('source', 1.0)]
    assert index.literal('短句', 5, .35) == []
    assert index.bm25('完全无关联字符', 5) == []
    assert index.long_literal_score('基于呼吸机控制的压力检测方案。', 'source') == 1


def test_hybrid_recovers_literal_source_below_dense_gate(corpus, monkeypatch):
    tools, thesis, literal, semantic, other, _ = corpus
    hybrid = tools.search_similar_chunks(QUERY, top_k=5, threshold=.75)
    matches = {m['document_id']: m for m in hybrid['matches']}
    assert literal in matches and semantic in matches
    assert thesis not in matches
    source = matches[literal]
    assert source['similarity'] == 0
    assert 'literal' in source['retrieval_channels'] and 'bm25' in source['retrieval_channels']
    assert 'dense' not in source['retrieval_channels']
    assert source['scores']['literal'] == 1
    assert source['scores']['literal_13'] == 1
    assert source['scores']['bm25'] > 0
    assert hybrid['threshold_scope'] == 'dense_only'
    monkeypatch.setattr(Config, 'RETRIEVAL_MODE', 'dense')
    baseline = tools.search_similar_chunks(QUERY, top_k=5, threshold=.75)
    assert {m['document_id'] for m in baseline['matches']} == {semantic}


def test_literal_source_outside_dense_pool_gets_explanatory_score(corpus, monkeypatch):
    tools, _, literal, _, _, _ = corpus
    monkeypatch.setattr(Config, 'RETRIEVAL_POOL_SIZE', 1)
    result = tools.search_similar_chunks(QUERY, top_k=1, threshold=.75)
    assert len(result['matches']) == 1
    assert result['matches'][0]['document_id'] == literal
    assert result['matches'][0]['similarity'] == 0


def test_same_candidate_is_merged_and_search_is_deterministic(corpus):
    tools, _, _, _, _, _ = corpus
    first = tools.search_similar_chunks(QUERY, threshold=0)
    second = tools.search_similar_chunks(QUERY, threshold=0)
    assert first == second
    ids = [m['chunk_id'] for m in first['matches']]
    assert len(ids) == len(set(ids))
    assert any(len(m['retrieval_channels']) > 1 for m in first['matches'])


def test_zero_evidence_returns_no_candidates(corpus, monkeypatch):
    tools, _, _, _, _, _ = corpus
    # Dense score is at most 1; forcing query to the unused third direction
    # and requesting a completely separate script prevents lexical matches.
    tools._embedder = type('Unrelated', (), {
        'encode': lambda self, texts, **kwargs: np.array([[-1., 0., 0.] for _ in texts]),
    })()
    result = tools.search_similar_chunks('完全不同文字且没有交集', threshold=.75)
    assert result['matches'] == []


def test_low_dense_candidate_remains_in_verified_report(corpus):
    tools, thesis, literal, _, _, tmp = corpus
    router = ToolRouter(document_tools=tools)
    router.bind_task(thesis, str(tmp / 'report'))
    thesis_chunk = tools.get_document_chunks(thesis)[0]['chunk_id']
    result = router.execute('search_similar_chunks', {'thesis_chunk_id': thesis_chunk})
    assert 'error' not in result
    findings = [{'thesis_chunk_id': thesis_chunk, 'source_chunk_id': m['chunk_id'],
                 'duplicate_type': 'no_issue', 'suggestion': ''} for m in result['matches']]
    report = router.execute('generate_report', {'thesis_title': 'Hybrid test', 'findings': findings,
                                               'output_dir': str(tmp / 'report')})
    assert report['status'] == 'ok'
    assert router.completion_state() == ('completed', None)
    match = next(f for f in router.session.report_findings
                 if f['source_chunk_id'].startswith(literal))
    assert match['similarity'] == 0
    assert 'literal' in match['retrieval_channels']
    assert match['retrieval_scores']['literal'] == 1
    import json
    from pathlib import Path
    data = json.loads(Path(report['json_data']).read_text())
    assert data['retrieval_config']['mode'] == 'hybrid'
    html = Path(report['html_report']).read_text()
    assert '字面重合' in html and '非重复概率' in html
