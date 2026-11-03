"""Validate tool observations before they enter the agent conversation."""
import math
from pathlib import Path

from jsonschema import validate


def validate_result(name: str, result):
    if not isinstance(result, dict):
        raise ValueError('Tool results must be objects.')
    if 'error' in result:
        if not isinstance(result['error'], str) or not result['error']:
            raise ValueError('Tool errors must contain a non-empty message.')
        return
    required = {
        'parse_pdf': ['doc_id', 'file_name', 'page_count'],
        'chunk_document': ['document_id', 'total_chunks'],
        'build_corpus_index': ['snapshot_id', 'total_chunks_indexed', 'unique_documents'],
        'search_similar_chunks': ['snapshot_id', 'matches_found', 'matches'],
        'get_chunk_context': ['chunk_id', 'document_id', 'context_chunks'],
        'get_corpus_stats': ['documents_parsed', 'total_chunks', 'index_built'],
        'generate_report': ['status', 'html_report', 'json_data', 'total_findings'],
    }
    properties = {
        field: {'type': 'integer', 'minimum': 0}
        for field in ('page_count', 'total_chunks', 'total_chunks_indexed',
                      'unique_documents', 'matches_found', 'documents_parsed', 'total_findings')
    }
    properties.update({field: {'type': 'string'} for field in (
        'doc_id', 'document_id', 'file_name', 'snapshot_id', 'chunk_id', 'html_report', 'json_data'
    )})
    properties['index_built'] = {'type': 'boolean'}
    validate(result, {'type': 'object', 'required': required.get(name, []), 'properties': properties})
    if name == 'search_similar_chunks':
        validate(result['matches'], {'type': 'array', 'items': {
            'type': 'object', 'required': ['chunk_id', 'document_id', 'similarity', 'matched_text', 'source_paper'],
            'properties': {'chunk_id': {'type': 'string'}, 'document_id': {'type': 'string'},
                           'source_paper': {'type': 'string'}, 'matched_text': {'type': 'string'},
                           'similarity': {'type': 'number', 'minimum': -1, 'maximum': 1}},
        }})
        if result['matches_found'] != len(result['matches']):
            raise ValueError('matches_found disagrees with matches.')
        if any(not math.isfinite(m['similarity']) for m in result['matches']):
            raise ValueError('Non-finite retrieval score.')
        for match in result['matches']:
            if 'retrieval_channels' in match:
                validate(match['retrieval_channels'], {
                    'type': 'array', 'minItems': 1, 'uniqueItems': True,
                    'items': {'enum': ['dense', 'bm25']},
                })
            if 'scores' in match:
                validate(match['scores'], {
                    'type': 'object', 'required': ['dense'],
                    'additionalProperties': False,
                    'properties': {
                        'dense': {'type': 'number', 'minimum': -1, 'maximum': 1},
                        'bm25': {'type': 'number', 'minimum': 0},
                        'reranker': {'type': 'number'},
                        'rrf': {'type': 'number', 'minimum': 0},
                    },
                })
                if any(not math.isfinite(value) for value in match['scores'].values()):
                    raise ValueError('Non-finite hybrid retrieval score.')
                if match['scores']['dense'] != match['similarity']:
                    raise ValueError('Dense score and similarity disagree.')
                if result.get('retrieval_config', {}).get('mode') == 'hybrid_rerank' and 'reranker' not in match['scores']:
                    raise ValueError('Reranked results must include their Cross-Encoder score.')
    if name == 'get_chunk_context':
        validate(result['context_chunks'], {'type': 'array', 'items': {
            'type': 'object', 'required': ['chunk_id', 'text', 'is_match'],
            'properties': {'text': {'type': 'string'}, 'is_match': {'type': 'boolean'}},
        }})
        if not any(c['chunk_id'] == result['chunk_id'] and c['is_match'] for c in result['context_chunks']):
            raise ValueError('Requested match is missing from its context.')
    if name == 'generate_report':
        if result['status'] != 'ok' or any(not Path(result[p]).is_file() for p in ('html_report', 'json_data')):
            raise ValueError('Report artifacts were not successfully written.')
