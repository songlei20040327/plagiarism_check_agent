import pytest

from plagiarism_check_agent.config import Config
from plagiarism_check_agent.evaluation.compare_screening import gate_settings, pair_summary, prefilter


def test_dense_prefilter_preserves_independent_keyword_evidence():
    kept = prefilter(['semantic_low', 'semantic_high'], ['keyword_only', 'semantic_low'],
                     {'semantic_low': .2, 'semantic_high': .8, 'keyword_only': .1})
    assert kept == {'semantic_low', 'semantic_high', 'keyword_only'}
    assert prefilter(['weak'], [], {'weak': .3}) == set()


def test_experiment_restores_production_settings_after_failure():
    previous = Config.AUTO_REVIEW_GATES, Config.RERANKER_PASS_THRESHOLD
    with pytest.raises(RuntimeError):
        with gate_settings(2.):
            assert Config.AUTO_REVIEW_GATES and Config.RERANKER_PASS_THRESHOLD == 2.
            raise RuntimeError('Interrupted experiment')
    assert (Config.AUTO_REVIEW_GATES, Config.RERANKER_PASS_THRESHOLD) == previous


def test_report_distinguishes_known_errors_and_pending_labels():
    result = pair_summary([{'id': 'copy', 'target': 1, 'action': 'auto_pass'},
                           {'id': 'quote', 'target': 0, 'action': 'auto_dense_positive'},
                           {'id': 'unknown', 'target': None, 'action': 'auto_pass'}])
    assert result['positive_pairs_auto_passed'] == ['copy']
    assert result['negative_pairs_auto_positive'] == ['quote']
    assert result['pending_label_ids'] == ['unknown']
