"""Ensure threshold experiments respect attribution and preserve unknown labels."""
import pytest

from plagiarism_check_agent.evaluation.calibrate_review import lexical, sweep, target, workload


@pytest.mark.parametrize(('decision', 'attribution', 'expected'), [
    ('near_paraphrase', 'adequate', 0),
    ('idea_overlap', 'adequate', 0),
    ('verbatim', 'missing_or_unclear', 1),
    ('near_paraphrase', 'missing_or_unclear', 1),
    ('no_issue', 'not_applicable', 0),
    ('uncertain', 'adequate', None),
    ('idea_overlap', 'missing_or_unclear', None),
    ('idea_overlap', 'not_applicable', None),
    ('near_paraphrase', 'uncertain', None),
])
def test_target_policy_does_not_treat_cited_similarity_or_uncertainty_as_plagiarism(decision, attribution, expected):
    annotation = {'decision': decision, 'attribution': attribution}
    assert target(annotation)[0] == expected
    assert annotation == {'decision': decision, 'attribution': attribution}


def test_sweep_excludes_unknown_labels_but_reports_their_filtering():
    rows = [{'id': 'positive', 'target': 1, 'reranker_score': 2.},
            {'id': 'negative', 'target': 0, 'reranker_score': 5.},
            {'id': 'unknown', 'target': None, 'reranker_score': -1.}]
    result = next(row for row in sweep(rows) if row['threshold'] == 2.)
    assert result['known_total'] == 2
    assert result['positive_retention'] == 1.
    assert result['high_gate_precision'] == .5
    assert result['unresolved_filtered_ids'] == ['unknown']
    assert result['false_positive_ids_if_auto_positive'] == ['negative']
    # A stricter threshold can keep an irrelevant pair while losing evidence.
    strict = next(row for row in sweep(rows) if row['threshold'] == 5.)
    assert strict['positive_pairs_filtered'] == 1
    assert strict['false_negative_ids_if_auto_pass'] == ['positive']


def test_workload_counts_pairs_and_empty_queries_not_llm_calls():
    pool = {'queries': [{'matches': [{'scores': {'reranker': -2.}}, {'scores': {'reranker': 2.}}]},
                        {'matches': [{'scores': {'reranker': -1.}}]}]}
    row = workload(pool, [0.])[0]
    assert row['baseline_pairs'] == 3
    assert row['retained_pairs'] == 1
    assert row['queries_with_zero_candidates'] == 1
    assert 'token-cost' in row['scope']


def test_explicit_quotation_can_have_identical_lexical_evidence():
    source = '这是一个具有充分长度的完整来源段落，用于检测直接引用与无引用复制的区别。'
    copy = lexical(source, source)
    quote = lexical('明确引文[1]：“' + source + '”', source)
    assert copy['longest_contiguous_chars'] == quote['longest_contiguous_chars']
    assert not copy['query_has_quote_marker']
    assert quote['query_has_quote_marker'] and quote['query_has_citation_marker']
