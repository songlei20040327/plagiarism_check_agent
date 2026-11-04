"""Deterministic routing of evidence; high relevance alone never labels copying."""
import difflib
import math
import re
import unicodedata

from ..config import Config


def evidence_features(query, source):
    def clean(text):
        text = re.sub(r'[\uE000-\uF8FF]', '', unicodedata.normalize('NFKC', text)).casefold()
        text = re.sub(r'\s+', ' ', text).strip()
        # Repair Chinese PDF spacing while retaining English word boundaries.
        return re.sub(r'(?<=[\u4e00-\u9fff]) | (?=[\u4e00-\u9fff])', '', text)
    q, s = clean(query), clean(source)
    block = difflib.SequenceMatcher(None, q, s, autojunk=False).find_longest_match()
    grams = lambda text: {text[i:i + 13] for i in range(max(0, len(text) - 12))}
    qg, sg = grams(q), grams(s)
    return {
        'longest_match_chars': block.size,
        'query_13gram_coverage': len(qg & sg) / len(qg) if qg else 0.,
        'citation_or_quote_cue': bool(re.search(r'[“”「」"]|\[\s*\d|引用|引自|参考文献', unicodedata.normalize('NFKC', query))),
        'standard_definition_cue': bool(re.search(r'公式|定律|定义', query)),
    }


def route_pair(query, source, match):
    features = evidence_features(query, source)
    result = {'action': 'llm', 'reason': 'Evidence requires contextual judgment.',
              'version': 'review-gate-v1', **features}
    if not Config.AUTO_REVIEW_GATES:
        return result
    if not math.isfinite(Config.RERANKER_PASS_THRESHOLD):
        raise ValueError('The pass threshold must be finite.')
    if (features['longest_match_chars'] >= 120 and features['query_13gram_coverage'] >= .85
            and not features['citation_or_quote_cue'] and not features['standard_definition_cue']):
        return {**result, 'action': 'auto_verbatim',
                'reason': 'Long contiguous copied wording covers at least 85% of query 13-grams; no citation or definition cue.'}
    score = match.get('scores', {}).get('reranker')
    if (score is not None and match.get('reranker_complete_input') is True
            and score < Config.RERANKER_PASS_THRESHOLD
            and features['longest_match_chars'] < 40 and features['query_13gram_coverage'] < .15):
        return {**result, 'action': 'auto_pass',
                'reason': 'Complete windowed relevance is below the pass threshold, with weak literal overlap.'}
    return result
