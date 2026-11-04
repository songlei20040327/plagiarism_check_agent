from dataclasses import replace
from types import SimpleNamespace

import pytest
from haystack import Document

from plagiarism_check_agent.config import Config
from plagiarism_check_agent.tools.review_gate import route_pair
from plagiarism_check_agent.tools.window_reranker import rank_windows, windows
from .test_hybrid_retrieval import CharacterTokenizer, corpus
from plagiarism_check_agent.agent.tool_router import ToolRouter


def test_windows_cover_the_tail_and_preserve_original_evidence(monkeypatch):
    monkeypatch.setattr(Config, 'RERANKER_MAX_LENGTH', 512)
    text = ''.join(chr(0x5000 + i) for i in range(700))
    tokenizer = CharacterTokenizer()
    pieces = windows(text, tokenizer, 200, 32)
    assert set(''.join(pieces)) == set(text)
    assert all(len(piece) <= 200 for piece in pieces)
    assert pieces[-1].endswith(text[-20:])
    query = 'query background ' * 35 + 'TAIL_EVIDENCE'
    source = 'source background ' * 35 + 'TAIL_EVIDENCE'

    class Ranker:
        _cross_encoder = SimpleNamespace(tokenizer=tokenizer)
        def run(self, query, documents, **kwargs):
            assert all(len(tokenizer(query, d.content)['input_ids']) <= 512 for d in documents)
            return {'documents': [replace(d, score=8. if 'TAIL_EVIDENCE' in query and 'TAIL_EVIDENCE' in d.content else -4.)
                                  for d in documents]}

    ranked = rank_windows(query, [Document(id='original', content=source)], Ranker())
    assert ranked[0].id == 'original' and ranked[0].content == source
    assert ranked[0].score == 8.
    assert ranked[0].meta['reranker_complete_input'] is True
    assert ranked[0].meta['reranker_window_pairs'] > 1


def test_negative_overlap_is_rejected(monkeypatch):
    monkeypatch.setattr(Config, 'RERANKER_WINDOW_OVERLAP', -1)
    with pytest.raises(ValueError, match='non-negative'):
        rank_windows('q', [Document(id='d', content='d')], None)


def test_gate_requires_complete_input_and_protects_citations(monkeypatch):
    monkeypatch.setattr(Config, 'AUTO_REVIEW_GATES', True)
    monkeypatch.setattr(Config, 'RERANKER_PASS_THRESHOLD', 0.)
    match = {'scores': {'reranker': -3.}, 'reranker_complete_input': True}
    assert route_pair('无线通信数据帧的分包处理。', '海洋生态种群的季节性迁徙。', match)['action'] == 'auto_pass'
    assert route_pair('无线通信数据帧的分包处理。', '海洋生态种群的季节性迁徙。', {**match, 'reranker_complete_input': False})['action'] == 'llm'
    copied = '本装置通过读取不同传感器的数据来协调各个模块的输出，在完成初始化之后执行连续的采集处理。' * 5
    assert route_pair(copied, copied, match)['action'] == 'auto_verbatim'
    assert route_pair('引文[1]：“' + copied + '”', copied, match)['action'] == 'llm'
    assert route_pair('该公式定义为：' + copied, '该公式定义为：' + copied, match)['action'] == 'llm'
    boundary_a = 'A part of the system is isolated from the others. ' * 6
    boundary_b = 'Apart of the system is isolated from the others. ' * 6
    assert route_pair(boundary_a, boundary_b, {**match, 'scores': {'reranker': 2.}})['action'] == 'llm'


def test_prepared_session_merges_script_judgments_and_protects_them(corpus, monkeypatch):
    tools, thesis, source, _, _, directory = corpus
    monkeypatch.setattr(Config, 'AUTO_REVIEW_GATES', True)
    router = ToolRouter(document_tools=tools)
    router.bind_task(thesis, str(directory / 'prepared-report'))
    stats = router.prepare_reviews()
    assert stats['candidate_pairs'] == 3 and stats['automatic_pairs'] == 2 and stats['llm_pairs'] == 1
    assert 'search_similar_chunks' not in {d['name'] for d in router.tool_definitions}
    forbidden = router.execute('search_similar_chunks', {'thesis_chunk_id': next(iter(router.session.chunks))})
    assert 'error' in forbidden and forbidden['retryable'] is False
    assert router.prepare_reviews() == stats
    key = next(iter(router.session.pending_candidates()))
    findings = [{'thesis_chunk_id': key[0], 'source_chunk_id': key[1], 'duplicate_type': 'no_issue', 'suggestion': ''}]
    auto = next(iter(router.session.automatic.values()))
    bad = router.execute('generate_report', {'thesis_title': 'Prepared', 'findings': findings + [auto],
                                            'output_dir': str(directory / 'prepared-report')})
    assert 'error' in bad
    report = router.execute('generate_report', {'thesis_title': 'Prepared', 'findings': findings,
                                               'output_dir': str(directory / 'prepared-report')})
    assert report['status'] == 'ok'
    assert len(router.session.report_findings) == 3
    assert sum(f['decision_source'] == 'script_auto_pass' for f in router.session.report_findings) == 2
