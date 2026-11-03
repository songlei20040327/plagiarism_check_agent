"""Offline contract, evidence and agent-loop regression tests."""
import json
from types import SimpleNamespace

import pytest

from plagiarism_check_agent.agent.loop_guard import LoopGuard
from plagiarism_check_agent.agent.runner import AgentRunner
from plagiarism_check_agent.agent.tool_router import ToolRouter
from plagiarism_check_agent.tools.document_tools import DocumentTools
from plagiarism_check_agent.trajectory.schemas import LLMResponse


@pytest.fixture
def review(tmp_path):
    docs = DocumentTools()
    docs.snapshot_id = 'snapshot-one'
    docs.reference_document_ids = {'reference'}
    docs._chunk_store = {
        'thesis_0': {'chunk_id': 'thesis_0', 'document_id': 'thesis', 'index': 0, 'text': 'Thesis original text.'},
        'reference_0': {'chunk_id': 'reference_0', 'document_id': 'reference', 'index': 0,
                        'text': 'Authentic reference evidence.', 'title': 'Reference', 'file_name': 'ref.pdf'},
    }
    router = ToolRouter(document_tools=docs)
    output = str(tmp_path / 'reports')
    router.bind_task('thesis', output)
    match = {'chunk_id': 'reference_0', 'document_id': 'reference', 'similarity': .91,
             'matched_text': 'Authentic reference evidence.', 'source_paper': 'Reference'}
    router._dispatch['search_similar_chunks'] = lambda **kwargs: {
        'snapshot_id': docs.snapshot_id, 'matches_found': 1, 'matches': [match.copy()],
    }
    finding = {'thesis_chunk_id': 'thesis_0', 'source_chunk_id': 'reference_0',
               'duplicate_type': 'near_paraphrase', 'suggestion': 'Cite the source.', 'analysis': 'Evidence reviewed.'}
    report = {'thesis_title': 'A thesis', 'findings': [finding], 'output_dir': output}
    return SimpleNamespace(router=router, docs=docs, report=report, match=match, tmp=tmp_path)


def search(review):
    result = review.router.execute('search_similar_chunks', {'thesis_chunk_id': 'thesis_0'})
    assert 'error' not in result
    return result


def tool(cid, name='search_similar_chunks', arguments=None):
    return {'type': 'tool_use', 'id': cid, 'name': name,
            'input': arguments if arguments is not None else {'thesis_chunk_id': 'thesis_0'}}


def response(*blocks):
    return LLMResponse(list(blocks), 'tool_use')


def end():
    return LLMResponse([{'type': 'text', 'text': 'Done'}], 'end_turn')


class ReplayLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, **kwargs):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def runner(review, responses):
    agent = AgentRunner.__new__(AgentRunner)
    agent.system_prompt = 'Synthetic review task'
    agent.tool_router = review.router
    agent.llm_client = ReplayLLM(responses)
    agent.trajectories_dir = review.tmp / 'traces'
    agent.guard = LoopGuard(50)
    return agent


@pytest.mark.parametrize('arguments', [
    {'thesis_chunk_id': 'thesis_0', 'top_k': 0},
    {'thesis_chunk_id': 'thesis_0', 'top_k': '5'},
    {'thesis_chunk_id': 'thesis_0', 'threshold': float('nan')},
    {'thesis_chunk_id': 'thesis_0', 'unexpected': True},
    {'thesis_chunk_id': 'foreign'},
    {'thesis_chunk_id': 'thesis_0', 'query_text': 'Made-up query'},
])
def test_bad_arguments_do_not_execute(review, arguments):
    review.router._dispatch['search_similar_chunks'] = lambda **kwargs: pytest.fail('Invalid call executed')
    assert 'error' in review.router.execute('search_similar_chunks', arguments)


def test_rejects_invalid_tool_observations(review):
    review.router._dispatch['search_similar_chunks'] = lambda **kwargs: {'matches': []}
    assert 'error' in review.router.execute('search_similar_chunks', {'thesis_chunk_id': 'thesis_0'})
    assert not review.router.session.searched


def test_rejects_foreign_retrieval_evidence(review):
    review.match['document_id'] = 'foreign'
    review.match['chunk_id'] = 'foreign_0'
    assert 'error' in review.router.execute('search_similar_chunks', {'thesis_chunk_id': 'thesis_0'})
    assert not review.router.session.searched


def test_coverage_and_all_candidate_judgments_required(review):
    assert 'error' in review.router.execute('generate_report', review.report)
    search(review)
    assert 'error' in review.router.execute('generate_report', {**review.report, 'findings': []})
    assert not (review.tmp / 'reports').exists()


@pytest.mark.parametrize('field,value', [
    ('duplicate_type', 'invalid'), ('similarity', 1.8), ('similarity', .82),
    ('source_chunk_id', 'fabricated'), ('source_paper', 'fabricated.pdf'),
    ('thesis_chunk', 'invented text'), ('overlapping_text', 'invented quote'),
])
def test_fabricated_findings_are_rejected(review, field, value):
    search(review)
    finding = {**review.report['findings'][0], field: value}
    result = review.router.execute('generate_report', {**review.report, 'findings': [finding]})
    assert 'error' in result
    assert not (review.tmp / 'reports').exists()


def test_bound_output_directory_and_context_scope(review):
    search(review)
    assert 'error' in review.router.execute('generate_report', {**review.report, 'output_dir': str(review.tmp)})
    assert 'error' in review.router.execute('get_chunk_context', {'chunk_id': 'thesis_0'})
    assert 'error' in review.router.execute('parse_pdf', {'file_path': '/outside.pdf'})


def test_canonical_report_and_completion_state(review):
    search(review)
    result = review.router.execute('generate_report', review.report)
    assert result['status'] == 'ok'
    assert review.router.completion_state() == ('completed', None)
    data = json.loads(open(result['json_data']).read())
    assert data['reference_count'] == 1
    assert data['coverage'] == {'total': 1, 'searched': 1}
    assert data['findings'][0]['overlapping_text'] == 'Authentic reference evidence.'
    assert data['findings'][0]['similarity'] == .91
    from pathlib import Path
    Path(result['html_report']).unlink()
    assert review.router.completion_state()[0] == 'failed'


def test_zero_matches_can_complete_with_empty_report(review):
    review.router._dispatch['search_similar_chunks'] = lambda **kwargs: {
        'snapshot_id': review.docs.snapshot_id, 'matches_found': 0, 'matches': [],
    }
    search(review)
    assert review.router.execute('generate_report', {**review.report, 'findings': []})['status'] == 'ok'
    assert review.router.completion_state()[0] == 'completed'


def test_early_end_is_failure(review):
    result = runner(review, [end()]).run('Review every chunk')
    assert result.status == 'failed' and result.error
    assert not result.report_result


def test_agent_success_counts_findings_once(review):
    agent = runner(review, [response(tool('s')), response(tool('r', 'generate_report', review.report)), end()])
    result = agent.run('Review every chunk')
    assert result.status == 'completed'
    assert result.stats['findings_count'] == 1
    assert result.stats['searched_paragraphs'] == 1
    assert result.stats['classification_distribution']['near_paraphrase'] == 1
    from plagiarism_check_agent.agent.llm_client import _convert_messages
    from plagiarism_check_agent.trajectory.exporter import export_sft
    trace = review.tmp / 'traces' / ('trace_' + result.session_id + '.jsonl')
    sample = export_sft(trace)
    assert sample['messages'][1:] == _convert_messages(result.messages)


def test_guard_completes_all_tool_call_ids(review):
    agent = runner(review, [response(tool('a')), response(tool('b')),
                            response(tool('c'), tool('d')), end()])
    result = agent.run('Review')
    results = [b for m in result.messages if m['role'] == 'user' and isinstance(m['content'], list)
               for b in m['content']]
    assert [b['tool_use_id'] for b in results] == ['a', 'b', 'c', 'd']
    assert json.loads(results[-1]['content'][0]['text'])['error_code'] == 'cancelled'
    assert 'matches' in json.loads(results[-2]['content'][0]['text'])
    from plagiarism_check_agent.agent.llm_client import _convert_messages
    from plagiarism_check_agent.trajectory.exporter import export_sft
    trace = review.tmp / 'traces' / ('trace_' + result.session_id + '.jsonl')
    assert export_sft(trace, include_failed=True)['messages'][1:] == _convert_messages(result.messages)


def test_guard_stop_is_not_mislabeled_as_budget_exhaustion(review):
    agent = runner(review, [response(tool('a'), tool('b'), tool('c'), tool('d'))])
    agent.guard.MAX_CORRECTIONS = 0
    result = agent.run('Review')
    assert result.status == 'failed'
    assert result.error == 'Loop guard terminated the task.'


def test_guard_reset_includes_correction_count():
    guard = LoopGuard(50)
    guard._correction_count = 2
    guard.reset()
    assert guard._correction_count == 0


def test_llm_failure_records_terminal_status(review):
    result = runner(review, [RuntimeError('simulated timeout')]).run('Review')
    assert result.status == 'failed' and 'simulated timeout' in result.error
    trace = review.tmp / 'traces' / ('trace_' + result.session_id + '.jsonl')
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert records[-1]['type'] == 'session_end'
    assert records[-1]['stats']['status'] == 'failed'


def test_iteration_budget_has_explicit_status(review):
    agent = runner(review, [LLMResponse([], 'tool_use')])
    agent.MAX_ITERATIONS = 1
    assert agent.run('Review').status == 'budget_exceeded'


def test_valid_report_on_last_iteration_is_completed(review):
    agent = runner(review, [response(tool('s')),
                            response(tool('r', 'generate_report', review.report))])
    agent.MAX_ITERATIONS = 2
    result = agent.run('Review')
    assert result.status == 'completed'
    assert result.error is None
    assert result.report_result['status'] == 'ok'


def test_optional_final_reply_failure_does_not_invalidate_report(review):
    agent = runner(review, [response(tool('s')),
                            response(tool('r', 'generate_report', review.report)),
                            RuntimeError('Final summary timeout')])
    result = agent.run('Review')
    assert result.status == 'completed'
    assert result.error is None
    assert result.report_result['status'] == 'ok'


def test_serialized_truncation_obeys_budget(review):
    review.router.MAX_TOOL_RESULT_TOKENS = 80
    result = review.router._truncate_result({f'field_{i}': 'x' * 100 for i in range(100)})
    assert review.router._count_tokens(json.dumps(result, ensure_ascii=False)) <= 80


def test_report_escapes_untrusted_text(review):
    review.docs._chunk_store['thesis_0']['text'] = '<script>bad()</script>'
    review.report['thesis_title'] = '<script>title()</script>'
    review.report['findings'][0]['suggestion'] = '<img src=x onerror=bad()>'
    search(review)
    result = review.router.execute('generate_report', review.report)
    from pathlib import Path
    html = Path(result['html_report']).read_text()
    assert '<script>' not in html and '<img src=x' not in html
    assert '&lt;script&gt;' in html


def test_dry_run_is_not_claimed_as_completed(review):
    review.router._dry_run = True
    search(review)
    assert review.router.execute('generate_report', review.report)['dry_run']
    assert review.router.completion_state() == ('dry_run', None)
    assert not (review.tmp / 'reports').exists()
