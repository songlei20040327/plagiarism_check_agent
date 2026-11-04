"""Protect human labels from mismatched batches, stale evidence and HTML injection."""
import argparse
import json
from copy import deepcopy

import fitz
import pytest

from plagiarism_check_agent.evaluation.review_pack import (
    assemble, location, validate_annotations,
)
from plagiarism_check_agent.tools.document_tools import DocumentTools


@pytest.fixture
def submission():
    pack = {'batch_id': 'review-fixture', 'cases': [{'id': 'R01', 'case_hash': 'abc'}]}
    data = {'batch_id': 'review-fixture', 'annotations': [
        {'case_id': 'R01', 'case_hash': 'abc', 'confirmed': True,
         'decision': 'no_issue', 'attribution': 'adequate', 'note': 'Reviewed source context.'},
    ]}
    return pack, data


def test_annotation_accepts_explicit_human_confirmation(submission):
    assert validate_annotations(*submission) == 1


@pytest.mark.parametrize('problem', ['batch', 'hash', 'duplicate', 'label', 'missing_attribution', 'boolean', 'note'])
def test_annotation_rejects_invalid_provenance_and_labels(submission, problem):
    pack, data = deepcopy(submission)
    row = data['annotations'][0]
    if problem == 'batch': data['batch_id'] = 'another-batch'
    if problem == 'hash': row['case_hash'] = 'changed-source'
    if problem == 'duplicate': data['annotations'].append(deepcopy(row))
    if problem == 'label': row['decision'] = 'arbitrary-positive'
    if problem == 'missing_attribution': row['attribution'] = ''
    if problem == 'boolean': row['confirmed'] = 1
    if problem == 'note': row['note'] = ['not text']
    with pytest.raises(ValueError):
        validate_annotations(pack, data)


def test_source_locator_uses_physical_pages_and_rejects_fabrication(tmp_path):
    path = tmp_path / 'source.pdf'
    with fitz.open() as pdf:
        for text in ['First page content.', 'Second page evidence.']:
            page = pdf.new_page()
            page.insert_text((50, 50), text)
        pdf.save(path)
    tools = DocumentTools()
    meta = tools.parse_pdf(str(path))
    tools.chunk_document(meta['doc_id'])
    chunk = tools.get_document_chunks(meta['doc_id'])[0]
    loc = location(tools, chunk, tmp_path)
    assert loc['pdf_pages'] == [1, 2]
    assert loc['pdf_relative_path'] == 'source.pdf'
    with pytest.raises(ValueError, match='Cannot locate'):
        location(tools, {**chunk, 'text': 'invented evidence'}, tmp_path)


def test_assembled_page_escapes_paper_script_and_does_not_prefill_labels(tmp_path):
    source = {'text': '</script><script>alert(1)</script>'}
    pool = {'snapshot_id': 'fixture', 'retrieval_config': {}, 'queries': [], 'intended_use': 'dev only'}
    spec = {'controls': [{'origin': 'constructed', 'query': source, 'source': source}],
            'advice': {'R01': {'decision': 'verbatim'}}, 'policy': {'mode': 'review'}}
    (tmp_path / 'pool.json').write_text(json.dumps(pool))
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    assemble(argparse.Namespace(output=tmp_path, spec=tmp_path / 'spec.json'))
    html = (tmp_path / 'view' / 'review.html').read_text()
    assert '</script><script>alert(1)' not in html
    assert '\\u003c/script>' in html
    labels = json.loads((tmp_path / 'annotations_blank.json').read_text())
    assert labels['annotations'][0]['decision'] == ''
    assert labels['annotations'][0]['confirmed'] is False
