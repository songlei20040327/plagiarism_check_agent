"""Task-scoped evidence and completion checks, independent of model claims."""
from pathlib import Path

from ..config import Config


class ReviewSession:
    def __init__(self, documents, thesis_document_id: str, output_dir: str):
        self.documents = documents
        self.chunks = {c['chunk_id']: c for c in documents.get_document_chunks(thesis_document_id)}
        if not self.chunks:
            raise ValueError('The thesis has no reviewable text chunks.')
        if thesis_document_id in documents.reference_document_ids:
            raise ValueError('The thesis is present in the reference corpus.')
        if not documents.snapshot_id:
            raise ValueError('A reference snapshot must be built before review.')
        self.output_dir = Path(output_dir).resolve()
        self.searched: set[str] = set()
        self.candidates: dict[tuple[str, str], dict] = {}
        self.report_result = None
        self.report_findings = []
        self.prepared = False
        self.automatic = {}
        self.gate_audit = {}

    def search_arguments(self, arguments: dict) -> dict:
        if self.prepared:
            raise ValueError('Retrieval is already complete; review the supplied candidates.')
        chunk_id = arguments.get('thesis_chunk_id')
        if chunk_id not in self.chunks:
            raise ValueError('Use a thesis_chunk_id from this task.')
        chunk = self.chunks[chunk_id]
        if 'query_text' in arguments and arguments['query_text'] != chunk['text']:
            raise ValueError('query_text must equal the selected thesis chunk.')
        if 'threshold' in arguments and arguments['threshold'] != Config.SIMILARITY_THRESHOLD:
            raise ValueError('The retrieval threshold is fixed by the task configuration.')
        if 'top_k' in arguments and arguments['top_k'] != Config.TOP_K_RESULTS:
            raise ValueError('top_k is fixed by the task configuration.')
        return {'query_text': chunk['text'], 'top_k': Config.TOP_K_RESULTS,
                'threshold': Config.SIMILARITY_THRESHOLD}

    def record_search(self, chunk_id: str, result: dict):
        if result.get('snapshot_id') != self.documents.snapshot_id:
            raise ValueError('Retrieval returned a different corpus snapshot.')
        for match in result['matches']:
            source = self.documents._chunk_store.get(match['chunk_id'])
            if not source or source['document_id'] not in self.documents.reference_document_ids:
                raise ValueError('Retrieval returned evidence outside the reference allowlist.')
            if match['document_id'] != source['document_id'] or match['matched_text'] not in source['text']:
                raise ValueError('Retrieval metadata or quote does not match the source chunk.')
            if match['source_paper'] != (source.get('title') or source['file_name']):
                raise ValueError('Retrieval source name does not match the source chunk.')
        self.searched.add(chunk_id)
        # The latest successful retrieval for a chunk replaces its candidate set.
        self.candidates = {k: v for k, v in self.candidates.items() if k[0] != chunk_id}
        for match in result['matches']:
            self.candidates[(chunk_id, match['chunk_id'])] = match.copy()
        self.report_result = None
        self.report_findings = []

    def pending_candidates(self):
        return {key: value for key, value in self.candidates.items() if key not in self.automatic}

    def finish_preparation(self):
        if self.searched != self.chunks.keys():
            raise ValueError('Cannot prepare LLM review before every chunk was searched.')
        from .review_gate import route_pair
        for key, match in self.candidates.items():
            query = self.chunks[key[0]]['text']
            source = self.documents._chunk_store[key[1]]['text']
            gate = route_pair(query, source, match)
            self.gate_audit[key] = gate
            if gate['action'] == 'llm':
                continue
            positive = gate['action'] == 'auto_verbatim'
            self.automatic[key] = {
                'thesis_chunk_id': key[0], 'source_chunk_id': key[1],
                'duplicate_type': 'verbatim' if positive else 'no_issue',
                'suggestion': '请核对来源并明确标引，必要时基于理解独立重述。' if positive else '',
                'analysis': gate['reason'],
            }
        self.prepared = True

    def preparation_stats(self):
        pending = self.pending_candidates()
        llm_chunks = {key[0] for key in pending}
        return {'total': len(self.chunks), 'searched': len(self.searched),
                'candidate_pairs': len(self.candidates), 'automatic_pairs': len(self.automatic),
                'llm_pairs': len(pending), 'llm_chunks': len(llm_chunks),
                'script_completed_chunks': len(self.chunks) - len(llm_chunks)}

    def llm_payload(self):
        groups = {}
        for key, match in self.pending_candidates().items():
            group = groups.setdefault(key[0], {'thesis_chunk_id': key[0],
                                               'text': self.chunks[key[0]]['text'], 'matches': []})
            group['matches'].append({'source_chunk_id': key[1], 'source_paper': match['source_paper'],
                                     'scores': match.get('scores', {'dense': match['similarity']})})
        return list(groups.values())

    def report_arguments(self, arguments: dict) -> dict:
        if self.searched != self.chunks.keys():
            missing = sorted(self.chunks.keys() - self.searched)
            raise ValueError(f'Coverage incomplete: {len(missing)} chunks remain: {missing[:5]}')
        if Path(arguments['output_dir']).resolve() != self.output_dir:
            raise ValueError('output_dir must be the task output directory.')
        normalized, seen = [], set()
        required = self.pending_candidates()
        for finding in arguments['findings']:
            key = (finding.get('thesis_chunk_id'), finding.get('source_chunk_id'))
            if key not in required or key in seen:
                raise ValueError('Every finding must reference a unique pending candidate; automatic judgments are protected.')
            seen.add(key)
        if seen != required.keys():
            raise ValueError('Every pending candidate needs a judgment, including no_issue.')
        findings = list(self.automatic.values()) + arguments['findings']
        for finding in findings:
            key = (finding['thesis_chunk_id'], finding['source_chunk_id'])
            match = self.candidates[key]
            source = self.documents._chunk_store[key[1]]
            canonical = {
                'thesis_chunk_id': key[0], 'source_chunk_id': key[1],
                'thesis_chunk': self.chunks[key[0]]['text'],
                'source_paper': match['source_paper'], 'similarity': match['similarity'],
                'overlapping_text': source['text'],
                'duplicate_type': finding['duplicate_type'], 'suggestion': finding['suggestion'],
                'analysis': finding.get('analysis', ''),
                'retrieval_channels': match.get('retrieval_channels', ['dense']),
                'retrieval_scores': match.get('scores', {'dense': match['similarity']}),
                'decision_source': ('script_' + self.gate_audit[key]['action'] if key in self.automatic else 'llm'),
                'gate_evidence': self.gate_audit.get(key, {}),
            }
            for field in ('thesis_chunk', 'source_paper', 'similarity'):
                if field in finding and finding[field] != canonical[field]:
                    raise ValueError(f'{field} does not match recorded retrieval evidence.')
            if 'overlapping_text' in finding and (
                not finding['overlapping_text'] or finding['overlapping_text'] not in source['text']
            ):
                raise ValueError('Quoted evidence is not an exact source substring.')
            normalized.append(canonical)
        return {**arguments, 'findings': normalized,
                'snapshot_id': self.documents.snapshot_id,
                'reference_count': len(self.documents.reference_document_ids),
                'retrieval_config': self.documents.retrieval_config(),
                'coverage': self.preparation_stats()}
