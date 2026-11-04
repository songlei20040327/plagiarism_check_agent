"""Score a frozen human review and inspect gates without changing runtime policy.

The reviewed pairs are a selected development sample, not an end-to-end test.
Citation annotations define evaluation targets only, never prediction features.
"""
import argparse
import difflib
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from html import escape
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plagiarism_check_agent.evaluation.review_pack import digest, dump, validate_annotations


def target(annotation):
    """Keep content relation and attribution separate from the final target."""
    decision, attribution = annotation['decision'], annotation['attribution']
    if decision == 'uncertain':
        return None, 'reviewer_uncertain'
    if decision == 'no_issue':
        return 0, 'reviewer_no_issue'
    if attribution == 'adequate':
        return 0, 'adequate_attribution_accepted_policy'
    if decision == 'idea_overlap':
        return None, 'idea_overlap_requires_adjudication'
    if attribution != 'missing_or_unclear':
        return None, 'attribution_requires_adjudication'
    if decision in {'verbatim', 'near_paraphrase'}:
        return 1, 'reviewed_substantial_repeat_without_adequate_credit'
    raise ValueError('Unsupported annotation combination.')


def lexical(query, source):
    from plagiarism_check_agent.evaluation.review_pack import normalize
    clean = lambda text: re.sub(r'[\uE000-\uF8FF]', '', normalize(text))
    q, s = clean(query), clean(source)
    block = difflib.SequenceMatcher(None, q, s, autojunk=False).find_longest_match()
    grams = lambda text: {text[i:i + 13] for i in range(max(0, len(text) - 12))}
    qg, sg = grams(q), grams(s)
    return {'longest_contiguous_chars': block.size,
            'query_13gram_coverage': len(qg & sg) / len(qg) if qg else 0.0,
            'query_has_quote_marker': bool(re.search(r'[“”「」"]', query)),
            'query_has_citation_marker': bool(re.search(r'\[\d+(?:[\s,，\-–]+\d+)*\]', query))}


def sweep(rows):
    known = [r for r in rows if r['target'] is not None]
    positives = sum(r['target'] == 1 for r in known)
    negatives = sum(r['target'] == 0 for r in known)
    scores = sorted({r['reranker_score'] for r in rows})
    thresholds = sorted({scores[0] - 1, scores[-1] + 1, *scores, -5., -2., 0., 2., 5., 6., 8., 10.})
    result = []
    for value in thresholds:
        kept = [r for r in known if r['reranker_score'] >= value]
        tp = sum(r['target'] == 1 for r in kept)
        fp = sum(r['target'] == 0 for r in kept)
        unresolved = [r['id'] for r in rows if r['target'] is None and r['reranker_score'] < value]
        result.append({'threshold': value, 'known_total': len(known),
                       'known_pairs_kept': len(kept), 'known_pairs_filtered': len(known) - len(kept),
                       'positive_retention': tp / positives if positives else None,
                       'positive_pairs_filtered': positives - tp,
                       'negative_pairs_kept': fp,
                       'high_gate_precision': tp / len(kept) if kept else None,
                       'negative_count': negatives, 'unresolved_filtered_ids': unresolved,
                       'false_positive_ids_if_auto_positive': [r['id'] for r in kept if r['target'] == 0],
                       'false_negative_ids_if_auto_pass': [r['id'] for r in known if r['target'] == 1 and r['reranker_score'] < value]})
    return result


def workload(pool, thresholds):
    """Estimate pair filtering for 16 selected queries, without assigning truth."""
    queries = pool['queries']
    baseline = sum(len(q['matches']) for q in queries)
    return [{'threshold': t, 'queries': len(queries), 'baseline_pairs': baseline,
             'retained_pairs': sum(m['scores']['reranker'] >= t for q in queries for m in q['matches']),
             'queries_with_zero_candidates': sum(all(m['scores']['reranker'] < t for m in q['matches']) for q in queries),
             'scope': 'selected query sample only; no end-to-end accuracy or token-cost claim'}
            for t in thresholds]


def render_report(result, destination):
    names = {'verbatim': '逐字重复', 'near_paraphrase': '近义改写', 'idea_overlap': '观点重合',
             'no_issue': '无问题', 'uncertain': '不确定'}
    rows = result['rows']
    positives = [row for row in rows if row['target'] == 1]
    constructed_positive_count = sum(row['origin'] == 'constructed' for row in positives)
    sample = result['workload_sample'][0] if result['workload_sample'] else {'queries': 0, 'baseline_pairs': 0}
    workload_strategy = result.get('workload_score_strategy', 'legacy_single_pair_truncate')
    fmt = lambda value: '—' if value is None else f'{value:.1%}'
    table = ''.join(f"<tr><td>{escape(r['id'])}</td><td>{escape(r['purpose'])}</td>"
                    f"<td>{escape(names[r['annotation']['decision']])}</td><td>{escape(r['annotation']['attribution'])}</td>"
                    f"<td>{'待核对' if r['target'] is None else '重复正例' if r['target'] else '最终无问题'}</td>"
                    f"<td>{r['reranker_score']:.3f}</td><td>{r['lexical']['longest_contiguous_chars']}</td>"
                    f"<td>{r['pair_tokens_before_truncation']}</td><td>{escape(r['origin'])}</td></tr>" for r in rows)
    gates = ''.join(f"<tr><td>{r['threshold']:.3f}</td><td>{r['known_pairs_filtered']}/{r['known_total']}</td>"
                    f"<td>{fmt(r['positive_retention'])}</td><td>{r['positive_pairs_filtered']}</td>"
                    f"<td>{r['negative_pairs_kept']}</td><td>{fmt(r['high_gate_precision'])}</td>"
                    f"<td>{escape(', '.join(r['unresolved_filtered_ids']))}</td></tr>"
                    for r in result['threshold_sweep'] if r['threshold'] in [-5., -2., 0., 2., 5., 6., 8., 10.])
    usage = ''.join(f"<tr><td>{r['threshold']:.3f}</td><td>{r['retained_pairs']}/{r['baseline_pairs']}</td>"
                    f"<td>{r['queries_with_zero_candidates']}/{r['queries']}</td></tr>" for r in result['workload_sample'])
    pending = ', '.join(r['id'] for r in rows if r['target'] is None)
    adjudication = ''
    for row in rows:
        if row['target'] is not None:
            continue
        evidence = row.get('evidence', {})
        advice = row.get('initial_advice', {})
        adjudication += (f"<details><summary>{escape(row['id'])} · {escape(row['purpose'])}</summary>"
                         f"<p>你的原判：{escape(names[row['annotation']['decision']])}；引用：{escape(row['annotation']['attribution'])}。"
                         f"备注：{escape(row['annotation'].get('note') or '未填写')}。</p>"
                         f"<p>待测文字：</p><blockquote>{escape(evidence.get('query', ''))}</blockquote>"
                         f"<p>参考文字：</p><blockquote>{escape(evidence.get('source', ''))}</blockquote>"
                         f"<p>我的初步意见：{escape(advice.get('reason', ''))}</p>"
                         f"<p>需核对：{escape(advice.get('check', ''))}</p></details>")
    truncated = sum(row['input_truncated'] for row in rows)
    strategy = result['model'].get('strategy', 'legacy_single_pair_truncate')
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>第一轮审核与阈值实验</title>
<style>body{{font:15px/1.7 -apple-system,"PingFang SC",sans-serif;color:#20313d;max-width:1350px;margin:35px auto;padding:0 24px;background:#f7f9fa}}h1{{font-size:27px}}h2{{font-size:20px;margin-top:30px}}table{{border-collapse:collapse;width:100%;background:white;font-size:13px}}td,th{{padding:9px 11px;border:1px solid #d8e2e8;text-align:left}}th{{background:#eaf1f5}}.note{{background:#fff4d8;padding:16px;border-radius:8px}}.scroll{{overflow-x:auto}}code{{background:#e5edf1;padding:2px 5px}}</style>
<h1>第一轮审核与阈值实验</h1><p>已核验 {result['confirmed']} 组人工确认记录。批次 <code>{escape(result['batch_id'])}</code>。</p>
<p class="note">本批为人工选择的开发样本。确定正例 {len(positives)} 组，其中构造对照 {constructed_positive_count} 组；同来源的复制与改写不能视为独立样本。表中的比例只描述这批样本，不能当作线上准确率。当前未选择或启用正式阈值。</p>
<h2>标签口径</h2><p>原始判定和引用意见原样保留。依据你确认的口径，引用恰当的相似内容最终按无问题处理。观点重合、引用情况不确定及显式“不确定”均列为待核对，不强行作为抄袭真值。</p>
<p>最终无问题 {sum(r['target']==0 for r in rows)} 组；重复正例 {sum(r['target']==1 for r in rows)} 组；待核对 {sum(r['target'] is None for r in rows)} 组：{pending}。</p>
<p class="note">本报告策略：{escape(strategy)}；实际截断 {truncated}/{len(rows)} 组。完整窗口策略会分别比较有重叠的短文本，并取最高相关性分数；不能将阈值直接迁移到不同切块、模型或窗口策略。</p>
<h2>阈值扫描</h2><p>同一阈值分别模拟“低于它直接通过”和“高于它直接判重复”，用于检查风险。高分精确率表示保留样本中有多少确为重复。待核对样本不参与正负统计，但列出它们是否被过滤。</p>
<table><tr><th>原始分数阈值</th><th>确定样本过滤数</th><th>构造正例保留率</th><th>漏掉正例数</th><th>无问题仍保留数</th><th>若直接判重复的精确率</th><th>被放行的待核对样本</th></tr>{gates}</table>
<h2>工作量样本</h2><p>来自选取的 {sample['queries']} 个真实待测 chunk、原始 Top5 共 {sample['baseline_pairs']} 对，使用冻结候选池的原评分策略 {escape(workload_strategy)}。未审核的其余候选没有真值；这里只估计候选数量变化，不声称节省了相同比例的总调用或费用。更换重排策略后应使用 replay_gates.py 重算候选排序。</p>
<table><tr><th>阈值</th><th>保留候选 / 原候选</th><th>零候选 chunk / 样本 chunk</th></tr>{usage}</table>
<h2>需要补充核对的样本</h2><p>无需重审全部 40 组。这里保留你的原始意见供核对，没有自动修改标签。若确认有实质重复，请指出对应语句或步骤；若只是公共知识或同主题，可以注明。判定“引用不明”时也可继续保留待定。</p>{adjudication}
<h2>逐组记录</h2><p>模型分数与引用是否恰当是不同信息。引用标注只用作评测标签，未作为预测特征。表中的 token 数是分窗前的总量，完整窗口策略不会仅因超过512而丢弃后文。</p>
<div class="scroll"><table><tr><th>编号</th><th>主题</th><th>原始判定</th><th>引用意见</th><th>校准标签</th><th>重排分数</th><th>最长连续重合字符</th><th>截断前 tokens</th><th>来源类型</th></tr>{table}</table></div>
<p>原始人工意见备份、模型分数与完整阈值表保存在同目录 JSON 文件。人工标签不随重新评分而改变。</p></html>'''
    destination.write_text(html, encoding='utf-8')


def run(args):
    pack_bytes, annotation_bytes = args.pack.read_bytes(), args.annotations.read_bytes()
    pack, submission = json.loads(pack_bytes), json.loads(annotation_bytes)
    confirmed = validate_annotations(pack, submission)
    if confirmed != len(pack['cases']) or len(submission['annotations']) != len(pack['cases']):
        raise ValueError('Complete this review batch before calibrating.')
    for case in pack['cases']:
        if digest({k: v for k, v in case.items() if k not in {'advice', 'case_hash'}}) != case['case_hash']:
            raise ValueError('Review evidence changed after annotation.')
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {'batch_id': pack['batch_id'], 'confirmed': confirmed,
               'source_annotation_sha256': hashlib.sha256(annotation_bytes).hexdigest(),
               'source_pack_sha256': hashlib.sha256(pack_bytes).hexdigest(),
               'imported_at': datetime.now(timezone.utc).isoformat(),
               'accepted_policy': 'adequate attribution -> final no_issue; preserve original relation',
               'source_annotations': str(args.annotations.resolve())}
    backup = args.output / 'reviewed_annotations.json'
    if backup.exists() and backup.read_bytes() != annotation_bytes:
        raise ValueError('A different review backup exists; choose a new output directory.')
    backup.write_bytes(annotation_bytes)
    dump(args.output / 'import_receipt.json', receipt)
    import torch
    from haystack import Document
    from plagiarism_check_agent.config import Config
    from plagiarism_check_agent.tools.haystack_retrieval import get_reranker
    from plagiarism_check_agent.tools.window_reranker import rank_windows
    torch.set_num_threads(args.threads)
    ranker = get_reranker()
    tokenizer = ranker._cross_encoder.tokenizer
    annotations = {row['case_id']: row for row in submission['annotations']}
    rows = []
    for case in pack['cases']:
        query, source = case['query']['text'], case['source']['text']
        document = Document(id=case['id'], content=source)
        legacy = getattr(args, 'legacy_truncated', False)
        scored = (ranker.run(query=query, documents=[document], top_k=1)['documents'][0] if legacy
                  else rank_windows(query, [document], ranker)[0])
        score = scored.score
        if not math.isfinite(score):
            raise ValueError('Non-finite model score.')
        annotation = annotations[case['id']]
        label, reason = target(annotation)
        token_count = len(tokenizer(query, source, truncation=False)['input_ids'])
        rows.append({'id': case['id'], 'case_hash': case['case_hash'], 'purpose': case['purpose'],
                     'origin': case['origin'], 'annotation': annotation, 'target': label, 'target_reason': reason,
                     'reranker_score': float(score), 'previous_score': case.get('scores', {}).get('reranker'),
                     'pair_tokens_before_truncation': token_count, 'input_truncated': legacy and token_count > Config.RERANKER_MAX_LENGTH,
                     'window_pairs': scored.meta.get('reranker_window_pairs', 1),
                     'lexical': lexical(query, source), 'suggestion_disagrees': case['advice']['decision'] != annotation['decision'],
                     'initial_advice': case['advice'], 'evidence': {'query': query, 'source': source}})
        print(f"Scored {case['id']}: {score:.3f}; target={label}", flush=True)
    thresholds = [-5., -2., 0., 2., 5., 6., 8., 10.]
    result = {'schema_version': 1, **receipt,
              'model': {'name': Config.RERANKER_MODEL, 'revision': Config.RERANKER_REVISION,
                        'max_pair_tokens': Config.RERANKER_MAX_LENGTH, 'scale_score': False,
                        'strategy': 'legacy_single_pair_truncate' if getattr(args, 'legacy_truncated', False) else 'complete_windows_max_v1'},
              'runtime': {'python': platform.python_version(), 'haystack': version('haystack-ai'),
                          'sentence_transformers': version('sentence-transformers')},
              'target_counts': dict(Counter('pending' if r['target'] is None else str(r['target']) for r in rows)),
              'rows': rows, 'threshold_sweep': sweep(rows),
              'workload_sample': workload(json.loads(args.pool.read_text()), thresholds),
              'workload_score_strategy': json.loads(args.pool.read_text()).get('retrieval_config', {}).get('reranker_strategy', 'legacy_single_pair_truncate'),
              'production_threshold': None, 'intended_use': 'selected development sample; not independent validation'}
    dump(args.output / 'calibration_analysis.json', result)
    render_report(result, args.output / 'review_analysis.html')
    print(json.dumps({'target_counts': result['target_counts'], 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--legacy-truncated', action='store_true', help='Reproduce the previous truncated-pair baseline only.')
    run(parser.parse_args())
