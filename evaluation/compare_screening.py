"""Compare screening policies using real BGE embeddings and verified frozen reranker scores.

This is a development experiment, not a held-out benchmark. It calls no chat
model. Pair decisions, retrieval retention and LLM workload are separate checks.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plagiarism_check_agent.config import Config
from plagiarism_check_agent.evaluation.review_pack import dump
from plagiarism_check_agent.tools.review_gate import route_pair


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def gate_settings(threshold):
    previous = Config.AUTO_REVIEW_GATES, Config.RERANKER_PASS_THRESHOLD
    Config.AUTO_REVIEW_GATES, Config.RERANKER_PASS_THRESHOLD = True, threshold
    try:
        yield
    finally:
        Config.AUTO_REVIEW_GATES, Config.RERANKER_PASS_THRESHOLD = previous


def prefilter(dense, bm25, similarities, threshold=.6):
    """A dense threshold applies only to the dense route; BM25 stays independent."""
    return {key for key in dense if similarities[key] >= threshold} | set(bm25)


def pair_summary(rows):
    return {
        'cases': len(rows), 'actions': dict(Counter(r['action'] for r in rows)),
        'positive_pairs_auto_passed': [r['id'] for r in rows if r['target'] == 1 and r['action'] == 'auto_pass'],
        'negative_pairs_auto_positive': [r['id'] for r in rows if r['target'] == 0 and r['action'] in {'auto_verbatim', 'auto_dense_positive'}],
        'pending_label_ids': [r['id'] for r in rows if r['target'] is None],
    }


def validate_replay(root, pack, pool, replay):
    if replay['batch_id'] != pack['batch_id'] or pack['snapshot_id'] != pool['snapshot_id']:
        raise ValueError('Review batch or corpus snapshot mismatch.')
    paths = {'pack': root / 'review_pack.json', 'pool': root / 'pool.json',
             'annotations': root / 'analysis/reviewed_annotations.json',
             'adjudications': root / 'analysis/assistant_adjudications.json'}
    for name, digest in replay['provenance']['input_sha256'].items():
        if sha(paths[name]) != digest:
            raise ValueError(f'Frozen replay input changed: {name}')
    for name, digest in replay['provenance']['code_sha256'].items():
        if sha(Path(__file__).resolve().parents[1] / name) != digest:
            raise ValueError(f'Frozen scoring or gate code changed: {name}')
    if (replay['strategy'] != 'complete_windows_max_v1'
            or replay['model'] != Config.RERANKER_MODEL
            or replay['model_revision'] != Config.RERANKER_REVISION
            or replay['max_pair_tokens'] != Config.RERANKER_MAX_LENGTH
            or replay['window_overlap'] != Config.RERANKER_WINDOW_OVERLAP):
        raise ValueError('Frozen reranker scores use a different model or window policy.')


def run(args):
    import chromadb
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from plagiarism_check_agent.tools.haystack_retrieval import HaystackCandidateRetriever

    root = args.review_dir.resolve()
    pack = json.loads((root / 'review_pack.json').read_text())
    pool = json.loads((root / 'pool.json').read_text())
    replay_path = root / 'analysis/window_gate_replay.json'
    replay = json.loads(replay_path.read_text())
    validate_replay(root, pack, pool, replay)
    refs = pool['all_reference_chunks']
    index = root / 'vector_index'
    manifest_path = index / ('manifest_' + pool['snapshot_id'] + '.json')
    manifest = json.loads(manifest_path.read_text())
    if manifest['embedding_model'] != Config.EMBEDDING_MODEL:
        raise ValueError('Embedding model does not match the frozen reference index.')
    client = chromadb.PersistentClient(path=str(index))
    collection = client.get_collection('refs_' + pool['snapshot_id'][:48], embedding_function=None)
    stored = collection.get(include=['embeddings', 'documents'])
    if set(stored['ids']) != refs.keys():
        raise ValueError('Reference index IDs differ from the frozen corpus.')
    vectors = {}
    for key, vector, text in zip(stored['ids'], stored['embeddings'], stored['documents']):
        if text != refs[key]['text']:
            raise ValueError('Reference text changed in the vector index.')
        vector = np.asarray(vector, dtype=np.float32)
        if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1., atol=1e-3):
            raise ValueError('Reference embeddings are not finite normalized vectors.')
        vectors[key] = vector
    source_ids = {}
    for case in pack['cases']:
        key = case['source'].get('chunk_id')
        if key not in refs or refs[key]['text'] != case['source']['text']:
            raise ValueError('This experiment requires every reviewed source to match the indexed corpus.')
        source_ids[case['id']] = key
    texts = list(dict.fromkeys([c['query']['text'] for c in pack['cases']]
                              + [g['query']['text'] for g in pool['queries']]))
    torch.set_num_threads(args.threads)
    model = SentenceTransformer(str(args.embedding_snapshot.resolve()), device='cpu', local_files_only=True)
    encoded = model.encode(texts, normalize_embeddings=True, batch_size=4, show_progress_bar=False)
    if not np.isfinite(encoded).all():
        raise ValueError('Non-finite query embeddings.')
    queries = dict(zip(texts, encoded))
    found = collection.query(query_embeddings=encoded.tolist(), n_results=20, include=['distances'])
    dense = {text: dict(zip(ids, [max(-1., min(1., 1. - d)) for d in distances]))
             for text, ids, distances in zip(texts, found['ids'], found['distances'])}
    similarity = lambda text, key: max(-1., min(1., float(np.dot(queries[text], vectors[key]))))
    # Confirm that the model/cache reproduces the earlier production embeddings.
    differences = [abs(similarity(g['query']['text'], m['source']['chunk_id']) - m['scores']['dense'])
                   for g in pool['queries'] for m in g['matches']]
    if max(differences) > 1e-3:
        raise ValueError('BGE scores do not reproduce the frozen runtime scores.')
    lexical = HaystackCandidateRetriever(refs)
    try:
        bm25 = {text: lexical.bm25(text, 20) for text in texts}
    finally:
        lexical.close()
    print(f'Encoded {len(texts)} query texts; reused {len(vectors)} reference vectors.', flush=True)
    scored = {r['id']: r for r in replay['review_pair_results']}
    pair_results = {}
    for name, threshold in [('rerank_t0', 0.), ('rerank_t2', 2.), ('rerank_t4', 4.), ('dense_pair_06_09_then_t2', 2.)]:
        rows = []
        with gate_settings(threshold):
            for case in pack['cases']:
                row = scored[case['id']]
                if row['case_hash'] != case['case_hash']:
                    raise ValueError('Reviewed case changed after reranker scoring.')
                query, source = case['query']['text'], case['source']['text']
                cosine = similarity(query, source_ids[case['id']])
                action = route_pair(query, source, {'scores': {'reranker': row['reranker']},
                                                    'reranker_complete_input': True})['action']
                if name == 'dense_pair_06_09_then_t2':
                    action = 'auto_pass' if cosine < .6 else 'auto_dense_positive' if cosine > .9 else action
                rows.append({'id': case['id'], 'case_hash': case['case_hash'], 'target': row['target'],
                             'label_source': row['label_source'], 'origin': case['origin'],
                             'dense': cosine, 'reranker': row['reranker'], 'action': action,
                             'bge_top1': max(dense[query].values()),
                             'source_dense_rank': (list(dense[query]).index(source_ids[case['id']]) + 1
                                                   if source_ids[case['id']] in dense[query] else None),
                             'source_bm25_rank': (list(bm25[query]).index(source_ids[case['id']]) + 1
                                                  if source_ids[case['id']] in bm25[query] else None)})
        pair_results[name] = {'summary': pair_summary(rows), 'rows': rows}
    workload = []
    groups = {g['query']['chunk_id']: g for g in pool['queries']}
    for item in replay['selected_query_workload']:
        group = groups[item['thesis_chunk_id']]
        query = group['query']['text']
        old = group['retrieval_diagnostics']
        # Use the original runtime union so that this isolates prefiltering.
        sims = {key: similarity(query, key) for key in old['union_ids']}
        kept = prefilter(old['dense_ids'], old['bm25_ids'], sims)
        top = item['actions']
        removed = [m['source_chunk_id'] for m in top if m['source_chunk_id'] not in kept]
        policies = {}
        for name, threshold in [('rerank_t0', 0.), ('rerank_t2', 2.), ('rerank_t4', 4.)]:
            with gate_settings(threshold):
                policies[name] = sum(route_pair(query, refs[m['source_chunk_id']]['text'],
                    {'scores': {'reranker': m['reranker']}, 'reranker_complete_input': True})['action'] == 'llm' for m in top)
        # If every previous Top5 survives a pure filter, the same Top5 remains.
        # Otherwise do not invent scores for candidates that might refill it.
        policies['dense_prefilter_06_then_t2'] = None if removed else policies['rerank_t2']
        workload.append({'thesis_chunk_index': group['query']['chunk_index'], 'baseline_pairs': len(top),
                         'union_candidates': len(old['union_ids']), 'prefilter_candidates': len(kept),
                         'removed_reranked_top5_ids': removed, 'llm_pairs': policies,
                         'bge_top1': max(dense[query].values()),
                         'union_dense_scores': sims})
    summaries = {}
    for name in workload[0]['llm_pairs']:
        values = [w['llm_pairs'][name] for w in workload]
        summaries[name] = {'llm_pairs': None if None in values else sum(values),
                           'llm_thesis_chunks': None if None in values else sum(n > 0 for n in values),
                           'reranker_candidate_pairs': sum(w['prefilter_candidates'] if name == 'dense_prefilter_06_then_t2'
                                                          else w['union_candidates'] for w in workload),
                           'needs_rerun_query_indices': [w['thesis_chunk_index'] for w in workload if w['llm_pairs'][name] is None]}
    retention = []
    for case in pack['cases']:
        row = scored[case['id']]
        if row['target'] != 1:
            continue
        query, key = case['query']['text'], source_ids[case['id']]
        scores = {k: similarity(query, k) for k in set(dense[query]) | set(bm25[query])}
        retention.append({'id': case['id'], 'dense_top20': key in dense[query],
                          'hybrid_union': key in scores,
                          'hybrid_after_dense_prefilter_06': key in prefilter(dense[query], bm25[query], scores),
                          'dense': similarity(query, key)})
    result = {
        'batch_id': pack['batch_id'], 'pair_policy_results': pair_results,
        'workload_summary': summaries, 'workload_rows': workload,
        'positive_source_candidate_retention': retention,
        'query_level_bge_early_exit': {'queries': len(workload),
                                     'top1_below_06': sum(w['bge_top1'] < .6 for w in workload),
                                     'top1_above_09': sum(w['bge_top1'] > .9 for w in workload)},
        'provenance': {'embedding_model': Config.EMBEDDING_MODEL,
                       'embedding_snapshot': args.embedding_snapshot.name,
                       'embedding_max_tokens': model.max_seq_length,
                       'replay_sha256': sha(replay_path), 'manifest_sha256': sha(manifest_path),
                       'experiment_code_sha256': sha(Path(__file__)),
                       'runtime': {name: version(name) for name in
                                   ['sentence-transformers', 'torch', 'haystack-ai', 'chromadb']},
                       'maximum_frozen_dense_score_difference': max(differences)},
        'scope': ('Selected development examples, including 8 constructed positives and 7 assistant adjudications. '
                  'Pair 0.6/0.9 classification is a pair-level diagnostic, not the query-level early-exit policy. '
                  'Candidate retention is before reranking and does not measure final pipeline recall. '
                  'Workload is candidate/chunk counts, not actual LLM tokens, fees or latency. '
                  'Frozen full-window scores are reused, not a new reranker inference run.'),
    }
    dump(args.output, result)
    print(json.dumps({'pairs': {n: v['summary'] for n, v in pair_results.items()},
                      'workload': summaries, 'positive_source_candidate_retention': retention,
                      'query_level_bge_early_exit': result['query_level_bge_early_exit']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--embedding-snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    run(parser.parse_args())
