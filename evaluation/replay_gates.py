"""Replay frozen corpus candidates with complete-window scoring and routing.

Source-pair checks and the selected-query workload sample are reported
separately. Assistant adjudications never overwrite the human export.
"""
import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plagiarism_check_agent.evaluation.review_pack import dump, validate_annotations
from plagiarism_check_agent.evaluation.calibrate_review import target


def run(args):
    import torch
    from haystack import Document
    from plagiarism_check_agent.config import Config
    from plagiarism_check_agent.tools.haystack_retrieval import get_reranker
    from plagiarism_check_agent.tools.window_reranker import rank_windows
    from plagiarism_check_agent.tools.review_gate import route_pair
    torch.set_num_threads(4)
    pack = json.loads(args.pack.read_text())
    pool = json.loads(args.pool.read_text())
    submission = json.loads(args.annotations.read_text())
    if validate_annotations(pack, submission) != len(pack['cases']):
        raise ValueError('Human review is incomplete.')
    annotations = {row['case_id']: row for row in submission['annotations']}
    adjudications = json.loads(args.adjudications.read_text()) if args.adjudications else {}
    ranker = get_reranker()
    refs = pool['all_reference_chunks']
    cache, workloads = {}, []
    for group in pool['queries']:
        query = group['query']['text']
        ids = group['retrieval_diagnostics']['union_ids']
        ranked = rank_windows(query, [Document(id=key, content=refs[key]['text']) for key in ids], ranker)
        for doc in ranked:
            cache[(query, doc.id)] = doc
        actions = []
        for doc in ranked[:Config.TOP_K_RESULTS]:
            gate = route_pair(query, doc.content, {'scores': {'reranker': doc.score}, **doc.meta})
            actions.append({'source_chunk_id': doc.id, 'reranker': doc.score, **gate})
        workloads.append({'thesis_chunk_id': group['query']['chunk_id'],
                          'baseline_pairs': len(group['matches']), 'window_candidate_count': len(ranked),
                          'llm_pairs': sum(row['action'] == 'llm' for row in actions),
                          'automatic_pairs': sum(row['action'] != 'llm' for row in actions), 'actions': actions})
        print(f"Replayed query {group['query']['chunk_index']}: {workloads[-1]['llm_pairs']} LLM pairs", flush=True)
    rows = []
    for case in pack['cases']:
        query, source = case['query']['text'], case['source']['text']
        doc = cache.get((query, case['source'].get('chunk_id')))
        if doc is None:
            doc = rank_windows(query, [Document(id=case['id'], content=source)], ranker)[0]
        match = {'scores': {'reranker': doc.score}, **doc.meta}
        gate = route_pair(query, source, match)
        label, reason = target(annotations[case['id']])
        label_source = 'human_review'
        if label is None and case['id'] in adjudications:
            adjudication = adjudications[case['id']]
            if adjudication['case_hash'] != case['case_hash'] or adjudication['target'] not in (0, 1):
                raise ValueError('Adjudication does not match the reviewed evidence.')
            label, reason, label_source = adjudication['target'], adjudication['reason'], 'assistant_adjudication'
        rows.append({'id': case['id'], 'origin': case['origin'], 'case_hash': case['case_hash'],
                     'target': label, 'target_reason': reason, 'label_source': label_source,
                     'reranker': float(doc.score), 'window_pairs': doc.meta['reranker_window_pairs'], **gate})
    known = [row for row in rows if row['target'] is not None]
    result = {
        'batch_id': pack['batch_id'], 'strategy': 'complete_windows_max_v1',
        'pass_threshold': Config.RERANKER_PASS_THRESHOLD, 'model': Config.RERANKER_MODEL,
        'model_revision': Config.RERANKER_REVISION, 'max_pair_tokens': Config.RERANKER_MAX_LENGTH,
        'window_overlap': Config.RERANKER_WINDOW_OVERLAP,
        'review_pair_results': rows,
        'review_pair_summary': {'cases': len(rows), 'actions': dict(Counter(r['action'] for r in rows)),
                               'positive_pairs_auto_passed': [r['id'] for r in known if r['target'] == 1 and r['action'] == 'auto_pass'],
                               'negative_pairs_auto_verbatim': [r['id'] for r in known if r['target'] == 0 and r['action'] == 'auto_verbatim']},
        'selected_query_workload': workloads,
        'workload_summary': {'queries': len(workloads), 'baseline_pairs': sum(r['baseline_pairs'] for r in workloads),
                             'llm_pairs': sum(r['llm_pairs'] for r in workloads),
                             'queries_requiring_no_llm': sum(r['llm_pairs'] == 0 for r in workloads)},
        'scope': 'Selected development examples with assistant adjudications; not independent test accuracy or API token costs.',
        'provenance': {
            'input_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for name, path in [('pack', args.pack), ('pool', args.pool),
                                                ('annotations', args.annotations), ('adjudications', args.adjudications)]
                             if path is not None},
            'code_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                            for name in ['tools/review_gate.py', 'tools/window_reranker.py']},
            'runtime': {'python': platform.python_version(), 'haystack': version('haystack-ai'),
                        'sentence_transformers': version('sentence-transformers')},
        },
    }
    dump(args.output, result)
    print(json.dumps({key: result[key] for key in ['review_pair_summary', 'workload_summary']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['pack', 'pool', 'annotations', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--adjudications', type=Path)
    run(parser.parse_args())
