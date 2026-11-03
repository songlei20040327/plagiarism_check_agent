"""Run the entire pipeline locally, replacing only the remote model and embeddings."""
import json
import re
from pathlib import Path

import fitz
import pytest

from plagiarism_check_agent.config import Config
from plagiarism_check_agent.pipeline import orchestrator as pipeline
from plagiarism_check_agent.tools import document_tools
from plagiarism_check_agent.trajectory.schemas import LLMResponse
from .test_corpus_isolation import TinyEmbedder
from .test_hybrid_retrieval import offline_ranker
from plagiarism_check_agent.tools import haystack_retrieval


class TaskLLM:
    def __init__(self, **kwargs):
        pass

    def complete(self, messages, system, **kwargs):
        context = json.loads(system.rsplit('## 任务上下文\n', 1)[1])
        call_names = [b['name'] for m in messages if m['role'] == 'assistant'
                      for b in m['content'] if b['type'] == 'tool_use']
        if 'generate_report' in call_names:
            return LLMResponse([{'type': 'text', 'text': 'Done'}], 'end_turn')
        if not call_names:
            chunk_ids = re.findall(r'\[thesis_chunk_id=([^\]]+)\]', messages[0]['content'])
            blocks = [{'type': 'tool_use', 'id': 'search_' + str(i), 'name': 'search_similar_chunks',
                       'input': {'thesis_chunk_id': cid}} for i, cid in enumerate(chunk_ids)]
        else:
            findings = []
            for message in messages:
                if message['role'] != 'user' or not isinstance(message['content'], list):
                    continue
                for block in message['content']:
                    data = json.loads(block['content'][0]['text'])
                    for match in data.get('matches', []):
                        findings.append({'thesis_chunk_id': data['thesis_chunk_id'],
                                         'source_chunk_id': match['chunk_id'],
                                         'duplicate_type': 'no_issue', 'suggestion': ''})
            blocks = [{'type': 'tool_use', 'id': 'report', 'name': 'generate_report', 'input': {
                'thesis_title': 'Same title', 'findings': findings, 'output_dir': context['output_dir'],
            }}]
        return LLMResponse(blocks, 'tool_use')


def pdf(path, text):
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((50, 50), text)
        document.save(path)


@pytest.fixture
def pipeline_case(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, 'VECTOR_DB_DIR', tmp_path / 'db')
    monkeypatch.setattr(document_tools, '_get_embedder', lambda: TinyEmbedder())
    monkeypatch.setattr(haystack_retrieval, 'get_reranker', offline_ranker)
    monkeypatch.setattr(pipeline, 'LLMClient', TaskLLM)
    monkeypatch.setattr(pipeline, 'analyze_check_result', lambda **kwargs: {
        'insights': [], 'reading_suggestions': [], 'field': '', 'topics': [],
    })
    monkeypatch.setattr(pipeline, 'record_chunk_matches', lambda **kwargs: {'new_hotspots': 0})
    thesis = tmp_path / 'thesis.pdf'
    refs = tmp_path / 'refs'
    refs.mkdir()
    pdf(thesis, 'A thesis about a portable controller and pressure sensors.')
    pdf(refs / 'reference.pdf', 'Reference evidence about portable control systems.')
    orchestrator = pipeline.PlagiarismCheckOrchestrator(
        trajectories_dir=tmp_path / 'traces', output_dir=tmp_path / 'reports',
    )
    return orchestrator, thesis, refs


def test_full_pipeline_and_repeat_run_have_isolated_reports(pipeline_case):
    orchestrator, thesis, refs = pipeline_case
    first = orchestrator.check(str(thesis), str(refs))
    second = orchestrator.check(str(thesis), str(refs))
    assert first.status == second.status == 'completed'
    assert first.error is second.error is None
    assert first.report_path != second.report_path
    assert Path(first.report_path).is_file() and Path(second.report_path).is_file()
    assert first.stats['snapshot_id'] == second.stats['snapshot_id']
    assert orchestrator.doc_tools.reference_document_ids.isdisjoint({
        c['document_id'] for c in orchestrator.doc_tools.get_document_chunks(
            next(c['document_id'] for c in orchestrator.doc_tools._chunk_store.values()
                 if c['document_id'] not in orchestrator.doc_tools.reference_document_ids)
        )
    })


def test_early_model_end_never_reaches_profile_analysis(pipeline_case, monkeypatch):
    orchestrator, thesis, refs = pipeline_case
    monkeypatch.setattr(orchestrator.runner.llm_client, 'complete', lambda **kwargs: LLMResponse([], 'end_turn'))
    monkeypatch.setattr(pipeline, 'analyze_check_result', lambda **kwargs: pytest.fail('Failed task wrote a profile'))
    result = orchestrator.check(str(thesis), str(refs))
    assert result.status == 'failed' and result.error
    assert result.report_path == ''


def test_parallel_batch_does_not_share_task_state(pipeline_case):
    orchestrator, thesis, refs = pipeline_case
    results = orchestrator.batch_check([
        {'thesis': str(thesis), 'references': str(refs)},
        {'thesis': str(thesis), 'references': str(refs)},
    ], max_workers=2)
    assert all(r.status == 'completed' for r in results)
    assert len({r.report_path for r in results}) == 2
