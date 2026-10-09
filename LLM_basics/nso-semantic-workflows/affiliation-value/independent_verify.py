"""Independently verify saved affiliation scores and grouped uncertainty.

Uses committed compact evidence only: no fitting, downloads, source cache,
model deserialization or imports from the implementation being checked.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 20261009
REPLICATES = 10000
# Fixed descriptive exclusions identified by a normalized-text source audit.
# Preserve the all-644 primary result; never refit or change a threshold here.
OVERLAP = {
    's2aff-1191': ['s2aff-1236'],
    's2aff-1220': ['s2aff-1207', 's2aff-1208', 's2aff-1211'],
    's2aff-1277': ['s2aff-1266'],
    's2aff-1278': ['s2aff-1266'],
}
ANNOTATION_SHA256 = '2385f28c6c1275f0114cf15c33d164672af7a2a6a88d0c03178e04236c3de8c3'


def read(rel):
    return json.loads((ROOT / rel).read_text())


def normalize(value):
    value = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower()
    return ' '.join(re.findall('[a-z0-9]+', value))


def validate_saved_llm(row, job):
    if row['status'] != 'valid':
        return False
    answer = row.get('answer')
    if not isinstance(answer, dict) or set(answer) != {'target_ids', 'evidence', 'review_required'}:
        return False
    ids = answer['target_ids']
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        return False
    if len(ids) != len(set(ids)) or not set(ids).issubset({x['id'] for x in job['user']['candidates']}):
        return False
    if not isinstance(answer['review_required'], bool) or not isinstance(answer['evidence'], list):
        return False
    evidence = answer['evidence']
    if not all(isinstance(e, dict) for e in evidence):
        return False
    if {e.get('id') for e in evidence} != set(ids):
        return False
    for e in evidence:
        if set(e) != {'id', 'source_span'}:
            return False
        span = e['source_span']
        if not isinstance(span, str) or not span or span not in job['user']['affiliation']:
            return False
    return True


def ratio(a, b):
    return float(a / b) if b else None


def components(ids, inputs, gold):
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    shared_key_counts = Counter()
    for i, case_id in enumerate(ids):
        keys = [('target', x) for x in gold[case_id]['gold_ids']]
        text = normalize(inputs[case_id]['text'])
        if text:
            keys.append(('text', text))
        keys += [('historical_non_ror_label', normalize(x)) for x in gold[case_id]['non_ror_labels'] if normalize(x)]
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
                shared_key_counts[key[0]] += 1
            else:
                seen[key] = i
    roots = [find(i) for i in range(len(ids))]
    numbering = {value: i for i, value in enumerate(sorted(set(roots)))}
    labels = np.array([numbering[x] for x in roots], dtype=int)
    membership = [[ids[i] for i in range(len(ids)) if labels[i] == g] for g in range(len(numbering))]
    return labels, membership, dict(shared_key_counts)


def score_rows(ids, gold, predictions):
    rows = []
    for case_id in ids:
        truth = set(gold[case_id]['gold_ids'])
        p = predictions[case_id]
        valid = bool(p['valid'])
        selected = set(p['target_ids']) if valid else set()
        review = valid and bool(p['review_required'])
        automatic = valid and not review
        exact = valid and selected == truth
        rows.append({
            'case_id': case_id,
            'n': 1,
            'exact_sets': int(exact),
            'tp': len(truth & selected),
            'fp': len(selected - truth),
            'fn': len(truth - selected),
            'failed': int(not valid),
            'review': int(review),
            'review_correct': int(review and exact),
            'automatic': int(automatic),
            'automatic_correct': int(automatic and exact),
            'automatic_wrong': int(automatic and not exact),
            'automatic_tp': len(truth & selected) if automatic else 0,
            'automatic_fp': len(selected - truth) if automatic else 0,
            'automatic_fn': len(truth - selected) if automatic else len(truth),
            'reference_nil': int(not truth),
            'correct_nil': int(valid and not truth and not selected),
            'false_nil': int(valid and bool(truth) and not selected),
            'false_assignment_queries': int(bool(selected - truth)),
            'linked_on_nil': int(not truth and bool(selected)),
            'automatic_reference_nil': int(automatic and not truth),
            'automatic_reference_nil_wrong': int(automatic and not truth and not exact),
            'predicted_nil': int(valid and not selected),
            'automatic_predicted_nil': int(automatic and not selected),
            'cardinality': len(truth),
        })
    return rows


COUNT_KEYS = ['n', 'exact_sets', 'tp', 'fp', 'fn', 'failed', 'review', 'review_correct',
              'automatic', 'automatic_correct', 'automatic_wrong', 'automatic_tp',
              'automatic_fp', 'automatic_fn', 'reference_nil',
              'correct_nil', 'false_nil', 'false_assignment_queries', 'linked_on_nil',
              'automatic_reference_nil', 'automatic_reference_nil_wrong', 'predicted_nil',
              'automatic_predicted_nil']


def summarize(rows):
    out = {k: sum(r[k] for r in rows) for k in COUNT_KEYS}
    out.update({
        'exact_set_accuracy': ratio(out['exact_sets'], out['n']),
        'micro_precision': ratio(out['tp'], out['tp'] + out['fp']),
        'micro_recall': ratio(out['tp'], out['tp'] + out['fn']),
        'automatic_coverage': ratio(out['automatic'], out['n']),
        'correct_automatic_per_input': ratio(out['automatic_correct'], out['n']),
        'error_among_automatic': ratio(out['automatic_wrong'], out['automatic']),
    })
    return out


def paired(a, b, labels, weights):
    # Positive exact difference favors a, positive false-edge difference disfavors a.
    groups = weights.shape[1]
    sizes = np.bincount(labels, minlength=groups)
    denominators = weights @ sizes
    out = {
        'corrected': sum(x['exact_sets'] > y['exact_sets'] for x, y in zip(a, b)),
        'regressed': sum(x['exact_sets'] < y['exact_sets'] for x, y in zip(a, b)),
        'both_correct': sum(x['exact_sets'] and y['exact_sets'] for x, y in zip(a, b)),
        'both_wrong': sum(not x['exact_sets'] and not y['exact_sets'] for x, y in zip(a, b)),
    }
    for key in ['exact_sets', 'automatic_correct', 'fp', 'fn']:
        delta = np.array([x[key] - y[key] for x, y in zip(a, b)], dtype=float)
        total_by_group = np.bincount(labels, weights=delta, minlength=groups)
        boot = (weights @ total_by_group) / denominators
        out[key + '_difference_per_input'] = {
            'point': float(delta.mean()),
            'paired_cluster_percentile_95': [float(x) for x in np.quantile(boot, [.025, .975])],
        }
    return out


def expected_view(ids, gold, predictions, view):
    if view == 'automatic_conditional':
        ids = [k for k in ids if predictions[k]['valid'] and not predictions[k]['review_required']]
    answer = dict.fromkeys(['rows', 'exact_sets', 'tp', 'fp', 'fn', 'nil_rows', 'nil_correct',
                            'false_nil_rows', 'false_assignment_on_nil_rows', 'automatic_rows',
                            'review_rows', 'invalid_rows', 'unresolved_rows_in_this_view'], 0)
    for k in ids:
        truth = set(gold[k]['gold_ids'])
        p = predictions[k]
        auto = p['valid'] and not p['review_required']
        resolved = p['valid'] and (view != 'automatic_resolved' or auto)
        selected = set(p['target_ids']) if resolved else set()
        answer['rows'] += 1
        answer['exact_sets'] += int(resolved and selected == truth)
        answer['tp'] += len(truth & selected)
        answer['fp'] += len(selected - truth)
        answer['fn'] += len(truth - selected)
        answer['nil_rows'] += int(not truth)
        answer['nil_correct'] += int(resolved and not truth and not selected)
        answer['false_nil_rows'] += int(resolved and bool(truth) and not selected)
        answer['false_assignment_on_nil_rows'] += int(not truth and bool(selected))
        answer['automatic_rows'] += int(auto)
        answer['review_rows'] += int(p['valid'] and p['review_required'])
        answer['invalid_rows'] += int(not p['valid'])
        answer['unresolved_rows_in_this_view'] += int(not resolved)
    for name, numerator, denominator in [
        ('exact_set_accuracy', answer['exact_sets'], answer['rows']),
        ('micro_precision', answer['tp'], answer['tp'] + answer['fp']),
        ('micro_recall', answer['tp'], answer['tp'] + answer['fn']),
        ('automatic_coverage', answer['automatic_rows'], answer['rows']),
    ]:
        answer[name] = numerator / denominator if denominator else None
    return answer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def group_ids(ids, gold, jobs):
    return {
        'all': ids,
        'single': [k for k in ids if len(gold[k]['gold_ids']) == 1],
        'nil': [k for k in ids if not gold[k]['gold_ids']],
        'multi': [k for k in ids if len(gold[k]['gold_ids']) > 1],
        'candidate_complete_positive': [k for k in ids if gold[k]['gold_ids'] and
            set(gold[k]['gold_ids']) <= {x['id'] for x in jobs[k]['user']['candidates']}],
        'candidate_missing_positive': [k for k in ids if gold[k]['gold_ids'] and not
            set(gold[k]['gold_ids']) <= {x['id'] for x in jobs[k]['user']['candidates']}],
    }


def analyze(ids, methods, inputs, gold, jobs, replicates, seed):
    groups = group_ids(ids, gold, jobs)
    scores = {name: {group: {view: expected_view(members, gold, predictions, view)
              for view in ['valid_set', 'automatic_resolved', 'automatic_conditional']}
              for group, members in groups.items()} for name, predictions in methods.items()}
    rows = {name: score_rows(ids, gold, predictions) for name, predictions in methods.items()}
    labels, membership, shared = components(ids, inputs, gold)
    g = len(membership)
    weights = np.random.default_rng(seed).multinomial(g, np.full(g, 1 / g), size=replicates)
    pairs = {}
    for name in methods:
        if name != 'saved_llm':
            pairs['saved_llm_minus_' + name] = paired(rows['saved_llm'], rows[name], labels, weights)
        if name not in ['lexical', 'saved_llm']:
            pairs[name + '_minus_lexical'] = paired(rows[name], rows['lexical'], labels, weights)
    return {
        'group_sizes': {name: len(members) for name, members in groups.items()},
        'scores': scores,
        'paired_comparisons': pairs,
        'grouping': {
            'components': g, 'largest_component': max(map(len, membership)),
            'component_sizes': sorted(map(len, membership), reverse=True),
            'shared_key_occurrences': shared, 'members': membership,
        },
    }


def verify_seal(prediction_path):
    seal = read('affiliation-value/results/prediction-seal.json')
    compact_sources = []
    for name, expected in seal['artifact_sha256'].items():
        assert sha(HERE / name) == expected, name
    for name, expected in seal['source_sha256'].items():
        if name.startswith('cache/'):
            continue
        assert sha(ROOT / name) == expected, name
        compact_sources.append(name)
    assert sha(prediction_path) == seal['artifact_sha256']['results/test-predictions.json']
    assert seal['source_sha256']['cache/linkage/gold_affiliation_annotations.csv'] == ANNOTATION_SHA256
    return {
        'artifact_files_checked': len(seal['artifact_sha256']),
        'compact_source_files_checked': len(compact_sources),
        'raw_source_archives_checked': False,
        'scope': 'Committed artifacts and compact sources are verified; full raw-source verification is the separate run.py verify command.',
    }


def verify_selection_and_scores(new, jobs):
    selection = read('affiliation-value/results/validation-selection.json')
    protocol = read('affiliation-value/protocol.json')
    sweep = selection['complete_sweep']
    assert len(sweep) == len(protocol['thresholds']) * len(new['arms']) == 80
    ordering = lambda r: (-r['exact_sets'], r['fp'], r['fn'], -r['threshold'],
                          0 if r['model'] == 'logistic' else 1)
    best = min(sweep, key=ordering)
    assert best['model'] == new['primary_model'] == selection['primary_model']
    reconstructed = 0
    for name, predictions in new['arms'].items():
        selected = min([r for r in sweep if r['model'] == name], key=ordering)
        assert selected == selection['selected_by_model'][name]
        scores = load_gz(HERE / 'results' / f'{name}-scores.json.gz')
        assert len(scores['val']) == 588
        assert {r['case_id'] for r in scores['test']} == set(predictions)
        assert len(scores['test']) == len(predictions) == 644
        for row in scores['test']:
            k = row['case_id']
            assert row['candidate_ids'] == [c['id'] for c in jobs[k]['user']['candidates']]
            assert len(row['candidate_ids']) == len(row['scores']) == 25
            assert all(np.isfinite(s) for s in row['scores'])
            targets = {c for c, score in zip(row['candidate_ids'], row['scores'])
                       if score >= selected['threshold']}
            assert targets == set(predictions[k]), (name, k)
            reconstructed += 1
    return {
        'validation_rows': 588, 'frozen_threshold_model_combinations': len(sweep),
        'primary_model': best['model'], 'primary_threshold': best['threshold'],
        'primary_validation_exact_sets': best['exact_sets'],
        'test_prediction_rows_reconstructed': reconstructed,
        'test_candidate_lists_checked_exactly': reconstructed,
        'scope': 'Selection is checked against the complete sealed validation sweep; committed test scores reproduce every prediction. Full validation labels are not loaded by this cache-free verifier.',
    }


def reconcile(full, methods, ids, gold, jobs):
    actual = read('affiliation-value/results/evaluation.json')
    assert actual['group_sizes'] == full['group_sizes']
    scalars = 0
    for name, groups in full['scores'].items():
        for group, views in groups.items():
            for view, metrics in views.items():
                observed = actual['scores'][name][group][view]
                for metric, value in metrics.items():
                    scalars += 1
                    same = observed[metric] == value if value is None or observed[metric] is None else abs(observed[metric] - value) <= 1e-12
                    assert same, (name, group, view, metric, value, observed[metric])

    def correct(k, name, automatic):
        p = methods[name][k]
        return p['valid'] and (not automatic or not p['review_required']) and set(p['target_ids']) == set(gold[k]['gold_ids'])

    pair_sets = 0
    for name in [n for n in methods if n not in ['lexical', 'saved_llm']]:
        for group, members in group_ids(ids, gold, jobs).items():
            for label, reference, automatic in [('versus_lexical', 'lexical', False),
                    ('versus_llm_valid_set', 'saved_llm', False),
                    ('versus_llm_automatic_resolved', 'saved_llm', True)]:
                observed = actual['paired_comparisons'][name][group][label]
                corrected = sorted(k for k in members if correct(k, name, automatic) and not correct(k, reference, automatic))
                regressed = sorted(k for k in members if not correct(k, name, automatic) and correct(k, reference, automatic))
                assert corrected == sorted(observed['corrected_case_ids'])
                assert regressed == sorted(observed['regressed_case_ids'])
                assert observed['corrections'] == len(corrected)
                assert observed['regressions'] == len(regressed)
                assert observed['net_correct_change'] == len(corrected) - len(regressed)
                pair_sets += 1
    return {'score_scalars_checked': scalars, 'paired_case_sets_checked': pair_sets, 'mismatches': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=HERE / 'results/test-predictions.json')
    parser.add_argument('--output', type=Path, default=HERE / 'results/independent-verification.json')
    parser.add_argument('--replicates', type=int, default=REPLICATES)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    assert args.replicates > 0
    inputs = {r['case_id']: r for r in read('data/linkage/s2aff-inputs.json')}
    gold = {r['case_id']: r for r in read('data/linkage/s2aff-gold.json')}
    jobs = {r['case_id']: r for r in read('data/linkage/s2aff-model-jobs.json')}
    ids = [k for k, row in inputs.items() if row['split'] == 'test']
    assert len(ids) == len(set(ids)) == 644
    assert Counter(min(2, len(gold[k]['gold_ids'])) for k in ids) == {0: 38, 1: 590, 2: 16}
    new = json.loads(args.predictions.read_text())
    seal_checks = verify_seal(args.predictions)
    selection_checks = verify_selection_and_scores(new, jobs)
    baseline = read('results/linkage/s2aff-baseline-predictions.json')['fused_selective_set']
    methods = {'lexical': {k: {'target_ids': baseline[k], 'valid': True, 'review_required': False} for k in ids},
               'saved_llm': {}}
    llm_rows = read('results/linkage/model-test-v1.json')['rows']
    assert len(llm_rows) == 644 and {r['case_id'] for r in llm_rows} == set(ids)
    for row in llm_rows:
        k = row['case_id']
        valid = validate_saved_llm(row, jobs[k])
        methods['saved_llm'][k] = {'target_ids': row['answer']['target_ids'] if valid else [],
                                  'valid': valid, 'review_required': valid and row['answer']['review_required']}
    for name, predictions in new['arms'].items():
        assert len(predictions) == 644 and set(predictions) == set(ids)
        for k, targets in predictions.items():
            assert isinstance(targets, list) and all(isinstance(t, str) for t in targets)
            assert len(targets) == len(set(targets))
            assert set(targets) <= {c['id'] for c in jobs[k]['user']['candidates']}
        methods[name] = {k: {'target_ids': targets, 'valid': True, 'review_required': False}
                         for k, targets in predictions.items()}
    full = analyze(ids, methods, inputs, gold, jobs, args.replicates, args.seed)
    assert full['scores']['saved_llm']['all']['valid_set']['exact_sets'] == 593
    assert full['scores']['saved_llm']['all']['automatic_resolved']['exact_sets'] == 546
    assert full['scores']['saved_llm']['all']['valid_set']['review_rows'] == 65
    assert full['scores']['saved_llm']['all']['valid_set']['invalid_rows'] == 3
    assert full['scores']['lexical']['all']['valid_set']['exact_sets'] == 456
    reconciliation = reconcile(full, methods, ids, gold, jobs)
    assert set(OVERLAP) <= set(ids)
    sensitivity_ids = [k for k in ids if k not in OVERLAP]
    assert len(sensitivity_ids) == 640
    sensitivity = analyze(sensitivity_ids, methods, inputs, gold, jobs, args.replicates, args.seed)
    sensitivity['excluded_test_ids_and_validation_matches'] = OVERLAP
    sensitivity['observed_normalized_test_text'] = {k: normalize(inputs[k]['text']) for k in OVERLAP}
    sensitivity['scope'] = ('Fixed descriptive four-row exclusion from reporting only; all models, thresholds and predictions unchanged. '
                            'The complete overlap list was established by a separate audit of the pinned annotation CSV, '
                            'whose SHA-256 is recorded below; this verifier replays the fixed sensitivity without loading that source cache.')
    files = ['data/linkage/s2aff-inputs.json', 'data/linkage/s2aff-gold.json',
             'data/linkage/s2aff-model-jobs.json', 'results/linkage/model-test-v1.json',
             'results/linkage/s2aff-baseline-predictions.json',
             'affiliation-value/results/prediction-seal.json',
             'affiliation-value/results/evaluation.json',
             'affiliation-value/results/validation-selection.json']
    output = {
        'status': 'independently_verified', 'schema': 'affiliation-independent-verification-v1',
        'implementation_sha256': sha(Path(__file__)),
        'input_sha256': {name: sha(ROOT / name) for name in files},
        'predictions_sha256': sha(args.predictions), 'annotation_source_sha256': ANNOTATION_SHA256,
        'seal_checks': seal_checks, 'selection_checks': selection_checks,
        'evaluation_reconciliation': reconciliation, 'primary_model': new['primary_model'],
        'full_644': full, 'fixed_overlap_sensitivity_640': sensitivity,
        'bootstrap': {
            'replicates': args.replicates, 'seed': args.seed, 'numpy_version': np.__version__,
            'generator': 'numpy.random.default_rng / PCG64',
            'group_rule': 'Transitive connected components of shared gold ROR IDs, normalized exact text and normalized historical non-ROR labels; ASCII NFKD alphanumeric normalization.',
            'sampling': 'Resample the observed number of components uniformly with replacement; retain every row in sampled components.',
            'estimand': 'Row-weighted paired difference; each resample divides summed differences by its summed row count.',
            'interval': 'Pointwise 2.5% and 97.5% percentile bounds; fixed models and policies; fitting and selection uncertainty excluded.',
            'interpretation': 'Descriptive sensitivity to source-organization composition under an exchangeable-component working assumption, not a population guarantee or fresh-holdout validation.',
        },
        'boundaries': [
            'Same already-inspected historical test: retrospective comparison, not an untouched holdout.',
            'Multiple ROR labels are historical annotated sets, not newly adjudicated simultaneous affiliations.',
            'Valid reviewed selections count in annotation agreement but not automatic resolved agreement.',
            'The new conventional learners are not the official S2AFF specialist or a best-classical-pipeline claim.',
            'Thresholds select exact-set agreement; model scores are not asserted to be calibrated probabilities.',
        ],
    }
    payload = (json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if args.output.exists():
        assert args.output.read_bytes() == payload, 'Existing derived evidence differs; choose a new output path.'
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(json.dumps({'status': output['status'], 'output': args.output.name,
                      'score_scalars_checked': reconciliation['score_scalars_checked'],
                      'paired_case_sets_checked': reconciliation['paired_case_sets_checked'],
                      'groups': full['grouping']['components'],
                      'primary_exact_sets': full['scores'][new['primary_model']]['all']['valid_set']['exact_sets'],
                      'saved_llm_exact_sets': full['scores']['saved_llm']['all']['valid_set']['exact_sets']}))


if __name__ == '__main__':
    main()
