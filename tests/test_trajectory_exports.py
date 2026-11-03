"""Training exports must preserve observed tool interactions and task identity."""
import json
from pathlib import Path

import pytest

from plagiarism_check_agent.trajectory.exporter import (
    export_sft, export_tool_supervision, export_preference_pairs, quality_score,
)
from plagiarism_check_agent.trajectory.logger import TrajectoryLogger
from plagiarism_check_agent.trajectory.schemas import LLMResponse, ToolUseBlock


TOOL = {'name': 'get_corpus_stats', 'description': 'Inspect corpus',
        'input_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False}}


def trace(tmp_path, sid='session', prompt='same task', status='completed', score=None,
          snapshot='snapshot-one', observation=None, include_result=True, tools=None):
    with TrajectoryLogger(tmp_path, sid, thesis_path='thesis.pdf') as logger:
        logger.log_system_input('Exact system context', tools if tools is not None else [TOOL])
        logger.log_user_input(prompt)
        logger.log_assistant(LLMResponse([{
            'type': 'tool_use', 'id': 'call-one', 'name': 'get_corpus_stats', 'input': {},
        }], 'tool_use'))
        if include_result:
            logger.log_tool_result(ToolUseBlock('call-one', 'get_corpus_stats', {}),
                                   {'documents_parsed': 1, 'private_full_result': 'raw evidence'},
                                   observation=observation)
        logger.log_assistant(LLMResponse([{'type': 'text', 'text': 'Final answer ' + sid}], 'end_turn'))
        stats = {'status': status, 'snapshot_id': snapshot}
        if score is not None:
            stats['evaluation'] = {'verified': True, 'score': score}
        logger.close(stats)
    return tmp_path / ('trace_' + sid + '.jsonl')


def test_native_sft_keeps_results_system_and_tools(tmp_path):
    path = trace(tmp_path)
    sample = export_sft(path, system_prompt='A stale fallback prompt')
    assert [m['role'] for m in sample['messages']] == ['system', 'user', 'assistant', 'tool', 'assistant']
    assert sample['messages'][0]['content'] == 'Exact system context'
    assert sample['messages'][3]['tool_call_id'] == sample['messages'][2]['tool_calls'][0]['id']
    assert sample['tools'][0]['function']['name'] == 'get_corpus_stats'
    supervision = export_tool_supervision(path)
    assert len(supervision) == 1
    assert json.loads(supervision[0]['tool_result'])['documents_parsed'] == 1


def test_export_uses_observation_actually_seen_by_model(tmp_path):
    path = trace(tmp_path, observation={'error': 'Result too large'})
    sample = export_sft(path)
    observed = sample['messages'][3]['content']
    assert json.loads(observed) == {'error': 'Result too large'}
    assert export_tool_supervision(path)[0]['tool_result'] == observed


def test_missing_tool_result_is_rejected(tmp_path):
    path = trace(tmp_path, include_result=False)
    assert export_sft(path) is None
    assert export_tool_supervision(path) == []


def test_failed_task_is_excluded_from_positive_sft(tmp_path):
    path = trace(tmp_path, status='partial')
    assert export_sft(path) is None
    assert export_sft(path, include_failed=True) is not None


def test_legacy_user_wrapped_tool_results_are_supported(tmp_path):
    path = trace(tmp_path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for i, record in enumerate(records):
        if record['type'] == 'tool_result':
            records[i] = {'type': 'user', 'message': {'role': 'user', 'content': [{
                'type': 'tool_result', 'tool_use_id': record['tool_use_id'],
                'content': [{'type': 'text', 'text': json.dumps(record['result'])}],
            }]}}
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))
    assert export_sft(path) is not None
    assert len(export_tool_supervision(path)) == 1


def test_preference_pairs_require_explicit_grades(tmp_path):
    a = trace(tmp_path, 'a')
    b = trace(tmp_path, 'b')
    assert export_preference_pairs([a, b]) == []
    assert quality_score({'classification_distribution': {'verbatim': 99},
                          'avg_suggestion_length': 10000}) == 0
    assert quality_score({'evaluation': None}) == 0


def test_preference_pairs_have_shared_prompt_and_separate_completions(tmp_path):
    a = trace(tmp_path, 'a', score=.9)
    b = trace(tmp_path, 'b', score=.2, status='failed')
    pairs = export_preference_pairs([a, b])
    assert len(pairs) == 1
    pair = pairs[0]
    assert [m['role'] for m in pair['prompt']] == ['system', 'user']
    assert pair['chosen'][0]['role'] == pair['rejected'][0]['role'] == 'assistant'
    assert pair['metadata']['chosen_session'] == 'a'
    assert pair['metadata']['rejected_session'] == 'b'


@pytest.mark.parametrize('variation', ['prompt', 'snapshot', 'tools'])
def test_preference_pairs_do_not_cross_task_contexts(tmp_path, variation):
    a = trace(tmp_path, 'a', score=.9)
    args = {'prompt': 'another task'} if variation == 'prompt' else (
        {'snapshot': 'another snapshot'} if variation == 'snapshot' else {'tools': []})
    b = trace(tmp_path, 'b', score=.1, **args)
    assert export_preference_pairs([a, b]) == []


def test_duplicate_results_are_rejected(tmp_path):
    path = trace(tmp_path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    record = next(r for r in records if r['type'] == 'tool_result')
    records.insert(-2, record)
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    assert export_sft(path) is None
