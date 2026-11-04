"""Score complete text pairs through overlapping, token-bounded windows."""
import math
from dataclasses import replace

from ..config import Config


def windows(text, tokenizer, budget, overlap):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = encoded['offset_mapping']
    if len(offsets) <= budget:
        return [text]
    result = []
    step = budget - overlap
    if step < 1:
        raise ValueError('Window overlap must be smaller than the token budget.')
    for start in range(0, len(offsets), step):
        end = min(start + budget, len(offsets))
        left = offsets[start][0]
        right = offsets[end - 1][1]
        result.append(text[left:right])
        if end == len(offsets):
            break
    return result


def checked_run(query, candidates, ranker):
    expected = {doc.id: doc.content for doc in candidates}
    result = ranker.run(query=query, documents=candidates, top_k=len(candidates))
    ranked = result.get('documents')
    if not isinstance(ranked, list):
        raise ValueError('Reranker must return a document list.')
    seen = set()
    for doc in ranked:
        if doc.id not in expected or doc.id in seen or doc.content != expected[doc.id]:
            raise ValueError('Reranker changed candidate identity or source text.')
        if isinstance(doc.score, bool) or not isinstance(doc.score, (int, float)) or not math.isfinite(doc.score):
            raise ValueError('Reranker returned an invalid score.')
        seen.add(doc.id)
    if seen != expected.keys():
        raise ValueError('Reranker omitted candidates before final top-K selection.')
    return ranked


def rank_windows(query, candidates, ranker):
    if not candidates:
        return []
    if Config.RERANKER_WINDOW_OVERLAP < 0:
        raise ValueError('Window overlap must be non-negative.')
    tokenizer = ranker._cross_encoder.tokenizer
    special = tokenizer.num_special_tokens_to_add(pair=True)
    budget = (Config.RERANKER_MAX_LENGTH - special) // 2 - 8
    if budget < 8:
        raise ValueError('The reranker pair budget is too small for safe windows.')
    overlap = min(Config.RERANKER_WINDOW_OVERLAP, budget // 4)
    # Offsets preserve source substrings. Validate joint token lengths because
    # retokenizing a word fragment can use slightly more tokens than its slice.
    while True:
        queries = windows(query, tokenizer, budget, overlap)
        source_windows = []
        owners = {}
        for index, doc in enumerate(candidates):
            for number, text in enumerate(windows(doc.content, tokenizer, budget, overlap)):
                key = f'window_{index}_{number}'
                source_windows.append(replace(doc, id=key, content=text))
                owners[key] = doc.id
        if all(len(tokenizer(q, d.content, truncation=False)['input_ids']) <= Config.RERANKER_MAX_LENGTH
               for q in queries for d in source_windows):
            break
        budget //= 2
        overlap = min(overlap, budget // 4)
        if budget < 8:
            raise ValueError('Could not form complete reranker inputs without truncation.')
    scores = {doc.id: [] for doc in candidates}
    for q in queries:
        for doc in checked_run(q, source_windows, ranker):
            scores[owners[doc.id]].append(float(doc.score))
    return sorted([
        replace(doc, score=max(scores[doc.id]), meta={
            **doc.meta, 'reranker_window_pairs': len(scores[doc.id]),
            'reranker_complete_input': True,
        }) for doc in candidates
    ], key=lambda doc: (-doc.score, doc.id))
