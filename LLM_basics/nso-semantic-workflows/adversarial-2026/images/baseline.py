"""Offline OCR, lexical, unlabeled Splink and image/text CLIP comparators.

This module never reads evaluator gold/source identity. Test execution requires
an explicit policy freeze. Predictions retain failures and uncalibrated scores.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import time
import unicodedata

import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
RESULTS = HERE / 'results'
CACHE = Path(os.environ.get('NSO_IMAGE_CACHE', Path.home() / '.cache/nso-adversarial-2026/abo'))
P = json.loads((HERE / 'protocol.json').read_text())
STOP = set('a an and the of with for from in on by brand amazon market whole foods everyday value organic ounce ounces oz gram grams g lb pound pounds fl fluid ml milliliter liter count pack ct net wt weight fresh'.split())


def read(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n' for x in rows))
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(s):
    s = unicodedata.normalize('NFKD', str(s or '')).encode('ascii', 'ignore').decode().lower()
    return ' '.join(re.sub(r'[^a-z0-9.]+', ' ', s).split())
def brand_norm(s):
    n = norm(s)
    return '365' if n in {'365', '365 everyday value', '365 by whole foods market', '365 by wfm'} else n

def resolve_image(query):
    """Return the exact prepared JPEG used for both OCR and model pixel input."""
    key = query['image_cache_key']
    path = (CACHE / key).resolve()
    if not path.is_relative_to(CACHE.resolve()): raise ValueError('Invalid cache key')
    return path


def freeze_policies():
    path = HERE / 'policy-freeze.json'
    frozen = {'files': {name: digest(HERE / name) for name in ['protocol.json', 'schema.json', 'baseline.py', 'prepare.py', 'evaluate.py']},
              'development_subset_sha256': digest(HERE.parent / 'contracts/image-development-subset.json'),
              'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'test_outcomes_opened': False}
    if path.exists() and json.loads(path.read_text())['files'] != frozen['files']:
        raise RuntimeError('Existing policy freeze differs; record a versioned outcome-blind amendment explicitly.')
    if not path.exists(): path.write_text(json.dumps(frozen, indent=2) + '\n')
    print('Policies frozen; no test labels were read.')


def check_policy(split):
    if split == 'test':
        f = json.loads((HERE / 'policy-freeze.json').read_text())
        for name, expected in f['files'].items():
            if digest(HERE / name) != expected: raise RuntimeError('Policy hash changed: ' + name)


def inputs(split, view='closed'):
    source_split = 'dev' if split == 'dev80' else split
    queries = read(DATA / f'queries-{source_split}.jsonl')
    if split == 'dev80':
        subset = json.loads((HERE.parent / 'contracts/image-development-subset.json').read_text())
        keep = set(subset['queries'])
        queries = [q for q in queries if q['record_id'] in keep]
        if len(queries) != 80: raise ValueError('Expected frozen 80-query DEV subset')
    suffix = '-nil' if split == 'test' and view == 'nil' else ''
    return queries, read(DATA / f'catalog-{source_split}{suffix}.jsonl')


def ocr_one(q):
    path = resolve_image(q)
    cache = CACHE / 'ocr' / (q['record_id'] + '.json')
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists(): return json.loads(cache.read_text())
    result = {'record_id': q['record_id'], 'status': 'failed', 'text': '', 'modes': [], 'image_sha256': digest(path) if path.exists() else None}
    if path.exists():
        lines, seen = [], set()
        for mode in P['ocr']['psm_modes']:
            start = time.monotonic()
            try:
                cp = subprocess.run(['tesseract', str(path), 'stdout', '-l', 'eng', '--psm', str(mode)],
                                    capture_output=True, text=True, timeout=P['ocr']['timeout_seconds_per_mode'],
                                    env={**os.environ, 'OMP_THREAD_LIMIT': '1'})
                result['modes'].append({'psm': mode, 'returncode': cp.returncode, 'seconds': round(time.monotonic()-start, 3), 'stderr': cp.stderr[:500]})
                if cp.returncode == 0:
                    for line in cp.stdout.splitlines():
                        k = norm(line)
                        if k and k not in seen:
                            seen.add(k); lines.append(line.strip())
            except Exception as exc:
                result['modes'].append({'psm': mode, 'error': repr(exc), 'seconds': round(time.monotonic()-start, 3)})
        result['text'] = '\n'.join(lines)
        result['status'] = 'ok' if lines else 'empty_or_failed'
    else: result['error'] = 'prepared image unavailable'
    cache.write_text(json.dumps(result, ensure_ascii=False) + '\n')
    return result


def run_ocr(split):
    queries, _ = inputs(split)
    with concurrent.futures.ThreadPoolExecutor(max_workers=P['ocr']['workers']) as pool:
        rows = []
        for n, r in enumerate(pool.map(ocr_one, queries), 1):
            rows.append(r)
            if n % 50 == 0: print(f'OCR {split}: {n}/{len(queries)}', flush=True)
    write(RESULTS / f'ocr-{split}.jsonl', rows)
    lengths = [len(r['text']) for r in rows]
    summary = {'rows': len(rows), 'statuses': dict(collections.Counter(r['status'] for r in rows)), 'characters': sum(lengths),
               'median_characters': float(np.median(lengths)), 'p95_characters': float(np.quantile(lengths, .95)), 'max_characters': max(lengths, default=0),
               'mode_seconds': sum(m.get('seconds', 0) for r in rows for m in r['modes']), 'engine': subprocess.run(['tesseract', '--version'], capture_output=True, text=True).stdout.splitlines()[0]}
    (RESULTS / f'ocr-{split}-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


UNITS = {'g': ('mass', 1), 'gram': ('mass', 1), 'grams': ('mass', 1), 'kg': ('mass', 1000),
         'oz': ('mass', 28.349523125), 'ounce': ('mass', 28.349523125), 'ounces': ('mass', 28.349523125),
         'lb': ('mass', 453.59237), 'lbs': ('mass', 453.59237), 'pound': ('mass', 453.59237), 'pounds': ('mass', 453.59237),
         'ml': ('volume', 1), 'milliliter': ('volume', 1), 'milliliters': ('volume', 1), 'l': ('volume', 1000), 'liter': ('volume', 1000), 'liters': ('volume', 1000),
         'fl oz': ('volume', 29.5735295625), 'fluid ounce': ('volume', 29.5735295625), 'fluid ounces': ('volume', 29.5735295625),
         'pint': ('volume', 473.176473), 'quart': ('volume', 946.352946), 'gallon': ('volume', 3785.411784)}
UNIT_PATTERN = '|'.join(sorted((re.escape(x) for x in UNITS), key=len, reverse=True))
QUANTITY = re.compile(r'(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*(' + UNIT_PATTERN + r')\b', re.I)
PACK = re.compile(r'(?:pack\s+of\s+(\d+))|(?:(\d+)\s*(?:count|ct|slices|eggs|waffles|pieces)\b)', re.I)


def quantities(text):
    return [(UNITS[u.lower()][0], float(v) * UNITS[u.lower()][1]) for v, u in QUANTITY.findall(norm(text)) if float(v) > 0]


def canonical_quantity(value, unit):
    unit = str(unit or '').replace('_', ' ')
    if value is None or unit not in UNITS: return None
    try: return UNITS[unit][0], float(value) * UNITS[unit][1]
    except (TypeError, ValueError): return None


def features(queries, catalogs, extraction_path=None, split='dev'):
    brands = sorted({str(c['brand']) for c in catalogs}, key=lambda x: (-len(norm(x)), x))
    model_values = sorted({str(c['model_number']) for c in catalogs if c.get('model_number')})
    if extraction_path:
        source = {r['record_id']: r for r in read(extraction_path)}
    else: source = {r['record_id']: r for r in read(RESULTS / f'ocr-{split}.jsonl')}
    output = []
    for q in queries:
        r = source.get(q['record_id'], {})
        if extraction_path:
            f = r.get('fields', r.get('extracted', r.get('output', r)))
            if not isinstance(f, dict): f = {}
            product = ' '.join(str(f.get(k) or '') for k in ['product_name', 'variant']).strip()
            text = ' '.join(str(f.get(k) or '') for k in ['brand', 'product_name', 'variant', 'evidence_text', 'model_number']).strip()
            brand = brand_norm(f.get('brand')) or None
            quant = canonical_quantity(f.get('net_quantity_value'), f.get('net_quantity_unit'))
            qs = [quant] if quant else []
            pack = f.get('pack_count')
            model = norm(f.get('model_number')) or None
            status = r.get('status', 'ok') if r else 'missing'
        else:
            text = r.get('text', ''); product = text
            padded = ' ' + norm(text) + ' '
            b = next((b for b in brands if ' ' + norm(b) + ' ' in padded), None)
            # Common native 365 aliases are a source-text normalization, not an ID lookup.
            if not b and re.search(r'\b365\b', text): b = '365'
            if not b and 'whole foods' in norm(text): b = 'Whole Foods Market'
            brand = brand_norm(b) if b else None
            # Prefer explicit net-weight lines. Other OCR quantities remain candidate evidence;
            # they are never asserted to be annotated net contents.
            net_lines = [line for line in text.splitlines() if re.search(r'\b(net|wt|weight)\b', line, re.I)]
            qs = quantities(' '.join(net_lines)) if net_lines else quantities(text)
            m = PACK.search(norm(text)); pack = int(next(v for v in m.groups() if v)) if m else None
            model = next((norm(m) for m in model_values if ' ' + norm(m) + ' ' in padded), None)
            status = r.get('status', 'missing')
        tokens = sorted({t for t in norm(product).split() if t not in STOP and len(t) >= 2})
        output.append({'record_id': q['record_id'], 'text': text, 'title_tokens': tokens, 'brand': brand, 'quantities': qs,
                       'pack_count': pack, 'model': model, 'status': status, 'kind': 'query'})
    cats = []
    for c in catalogs:
        m = PACK.search(norm(c['title']))
        cats.append({'record_id': c['record_id'], 'text': c['catalog_text'], 'title': c['title'],
                     'title_tokens': sorted({t for t in norm(c['title']).split() if t not in STOP and len(t) >= 2}),
                     'brand': brand_norm(c['brand']), 'quantities': quantities(c['title']),
                     'pack_count': int(next(v for v in m.groups() if v)) if m else None,
                     'model': norm(c.get('model_number')) or None, 'status': 'native_catalog', 'kind': 'catalog'})
    return output, cats


def decision_status(input_status, has_evidence, view, score, margin, min_score, min_margin, fit_valid=True):
    if not fit_valid or input_status in {'failed','missing','decode_failed','empty_or_failed','baseline_failed'}:
        return 'failed'
    if not has_evidence:
        return 'review'
    if view == 'closed':
        return 'linked'
    if score < min_score:
        return 'nil'
    if margin < min_margin:
        return 'review'
    return 'linked'


def ranking_rows(queries, catalogs, matrix, method, view, min_score, min_margin, tie_matrix=None):
    rows = []
    ids = np.array([c['record_id'] for c in catalogs])
    for qi, (q, scores) in enumerate(zip(queries, matrix)):
        tie = tie_matrix[qi] if tie_matrix is not None else np.zeros(len(catalogs))
        idx = np.lexsort((ids, -tie, -scores))[:20]
        top = [{'catalog_id': str(ids[j]), 'score': float(scores[j])} for j in idx]
        margin = top[0]['score'] - top[1]['score'] if len(top) > 1 else 1.0
        has_evidence = bool(str(q.get('text', True)).strip())
        status = decision_status(q.get('status', 'ok'), has_evidence, view, top[0]['score'], margin, min_score, min_margin)
        prediction = top[0]['catalog_id'] if status == 'linked' else None
        rows.append({'query_id': q['record_id'], 'method': method, 'top_candidates': top, 'margin': margin,
                     'prediction': prediction, 'abstained': status in {'review','failed'}, 'input_status': q.get('status', 'ok'),
                     'decision_status': status, 'status': status, 'target_ids': [prediction] if prediction else []})
    return rows


def lexical_matrix(qf, cf):
    texts = [norm(c['title']) for c in cf]
    qtexts = [norm(q['text']) for q in qf]
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    wc = word.fit_transform(texts); wq = word.transform(qtexts)
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True)
    cc = char.fit_transform(texts); cq = char.transform(qtexts)
    score = .15 * (wq @ wc.T).toarray() + .30 * (cq @ cc.T).toarray()
    idf = dict(zip(word.get_feature_names_out(), word.idf_))
    for i, q in enumerate(qf):
        qt = set(norm(q['text']).split())
        for j, c in enumerate(cf):
            ct = set(c['title_tokens'])
            denominator = sum(idf.get(t, 1.0) for t in ct)
            coverage = sum(idf.get(t, 1.0) for t in qt & ct) / denominator if denominator else 0
            brand = q['brand'] is not None and q['brand'] == c['brand']
            quantity = any(ta == tb and abs(a-b) <= .05 * max(a,b) for ta,a in q['quantities'] for tb,b in c['quantities'])
            # A visible matching count supports the same bounded quantity feature.
            quantity = quantity or q['pack_count'] is not None and q['pack_count'] == c['pack_count']
            model = q['model'] is not None and q['model'] == c['model']
            score[i,j] += .35*coverage + .10*brand + .06*quantity + .04*model
    return score


def lexical(split, view, extraction_path=None, tag='ocr'):
    queries, catalogs = inputs(split, view)
    qf, cf = features(queries, catalogs, extraction_path, split)
    score = lexical_matrix(qf, cf)
    method = tag + '-lexical'
    rows = ranking_rows(qf, cf, score, method, view, .30, .03)
    write(RESULTS / f'{method}-{split}-{view}.jsonl', rows)
    write(RESULTS / f'{tag}-features-{split}-{view}.jsonl', qf + cf)
    out = CACHE / 'matrices'; out.mkdir(exist_ok=True)
    np.savez_compressed(out / f'{method}-{split}-{view}.npz', scores=score, query_ids=[q['record_id'] for q in qf], catalog_ids=[c['record_id'] for c in cf])
    print(f'Saved {len(rows)} {method} predictions; no outcomes read.', flush=True)


def prior_sensitivity(rows, prior, catalog_size, path):
    """Decision-prior sensitivity, holding the fitted likelihood ratios fixed."""
    output = []
    for fraction in [.25, .5, .8, 1.0]:
        alternative = fraction / catalog_size
        ratio = (alternative / (1-alternative)) / (prior / (1-prior))
        for row in rows:
            best = row['top_candidates'][0]
            probability = best['score']
            updated = ratio * probability / (1-probability + ratio*probability)
            valid = row.get('fit_valid') is True and row.get('input_status') not in {'failed','missing','decode_failed','empty_or_failed','baseline_failed'}
            output.append({'query_id':row['query_id'],'assumed_overlap_fraction':fraction,'pair_prior':alternative,
                'primary_pair_prior':prior,'catalog_size':catalog_size,'top_catalog_id':best['catalog_id'],
                'reweighted_top_probability':updated,'likelihood_ratios_fixed':True,'em_refitted':False,
                'threshold_decisions':{str(t):{'status':'failed' if not valid else ('review' if row.get('decision_status')=='review' else ('linked' if updated>=t else 'nil')),'target_ids':[best['catalog_id']] if valid and row.get('decision_status')!='review' and updated>=t else []} for t in [.8,.9,.95,.99]}})
    write(path,output)


def splink_run(split, view, extraction_path=None, tag='ocr'):
    import pandas as pd
    from splink import DuckDBAPI, Linker, SettingsCreator
    import splink.comparison_library as cl
    queries, catalogs = inputs(split, view)
    qf, cf = features(queries, catalogs, extraction_path, split)
    def frame(fs):
        result=[]
        for f in fs:
            z=f['quantities'][0] if f['quantities'] else (None,None)
            result.append({'unique_id':f['record_id'],'title_tokens':f['title_tokens'], 'first_token':f['title_tokens'][0] if f['title_tokens'] else None,
                           'brand':f['brand'],'quantity_type':z[0],'quantity':z[1],'model':f['model']})
        return pd.DataFrame(result)
    qdf, cdf = frame(qf), frame(cf)
    levels=[{'sql_condition':'list_count(title_tokens_l) = 0 OR list_count(title_tokens_r) = 0','label_for_charts':'Missing text','is_null_level':True}]
    ratio='list_count(list_intersect(title_tokens_l, title_tokens_r)) * 1.0 / least(list_count(title_tokens_l), list_count(title_tokens_r))'
    for t in [.9,.7,.4]: levels.append({'sql_condition':f'{ratio} >= {t}','label_for_charts':f'Token containment >= {t}'})
    levels.append({'sql_condition':'ELSE','label_for_charts':'Lower token containment'})
    comps=[cl.CustomComparison(levels, output_column_name='title_tokens'),cl.ExactMatch('brand')]
    if qdf['quantity'].notna().sum()>=10 and cdf['quantity'].notna().sum()>=10:
        comps.append(cl.CustomComparison([
            {'sql_condition':'quantity_l IS NULL OR quantity_r IS NULL','label_for_charts':'Missing quantity','is_null_level':True},
            {'sql_condition':'quantity_type_l = quantity_type_r AND abs(quantity_l - quantity_r) <= 0.05 * greatest(quantity_l, quantity_r)','label_for_charts':'Quantity agrees within 5%'},
            {'sql_condition':'ELSE','label_for_charts':'Quantity differs'}],output_column_name='quantity'))
    if qdf['model'].notna().sum()>=10 and cdf['model'].notna().sum()>=10: comps.append(cl.ExactMatch('model'))
    prior=(.8 if view=='nil' else 1.0)/len(catalogs)
    settings=SettingsCreator(link_type='link_only',comparisons=comps,blocking_rules_to_generate_predictions=['1=1'],
        probability_two_random_records_match=prior,max_iterations=200,retain_intermediate_calculation_columns=True)
    log=io.StringIO(); handler=logging.StreamHandler(log); logging.getLogger('splink').addHandler(handler)
    diagnostics={'method':tag+'-splink','prior':prior,'input_missingness':{s:{k:int(df[k].isna().sum()) for k in ['brand','quantity','model']} for s,df in [('query',qdf),('catalog',cdf)]},
                 'quantity_comparison_included':qdf['quantity'].notna().sum()>=10 and cdf['quantity'].notna().sum()>=10,
                 'model_comparison_included':qdf['model'].notna().sum()>=10 and cdf['model'].notna().sum()>=10,'em_sessions':[]}
    try:
        api=DuckDBAPI()
        linker=Linker([api.register(qdf,dataset_display_name='query'),api.register(cdf,dataset_display_name='catalog')],settings)
        linker.training.estimate_u_using_random_sampling(max_pairs=len(queries)*len(catalogs),seed=20261007,min_count_per_level=1)
        for rule in ['1=1']:
            try:
                session=linker.training.estimate_parameters_using_expectation_maximisation(rule,fix_u_probabilities=True,fix_probability_two_random_records_match=True,
                    populate_probability_two_random_records_match_from_trained_values=False)
                diagnostics['em_sessions'].append({'rule':rule,'status':'returned'})
            except Exception as exc: diagnostics['em_sessions'].append({'rule':rule,'status':'failed','error':repr(exc)})
        model=linker.misc.save_model_to_json()
        (RESULTS/f'{tag}-splink-{split}-{view}-model.json').write_text(json.dumps(model,indent=2)+'\n')
        changes = re.findall(r'Iteration (\d+): Largest change in params was ([+-]?[0-9.eE-]+)', log.getvalue())
        last_delta = abs(float(changes[-1][1])) if changes else None
        checks = {}
        for c in model['comparisons']:
            levels = [l for l in c['comparison_levels'] if not l.get('is_null_level')]
            checks[c['output_column_name']] = {}
            for kind in ['m_probability', 'u_probability']:
                vals = [l.get(kind) for l in levels]
                valid = all(isinstance(v, (int,float)) and math.isfinite(v) and v > 0 for v in vals)
                total = sum(vals) if valid else None
                checks[c['output_column_name']][kind] = {'positive_finite': valid, 'sum': total, 'normalized': valid and abs(total-1) <= 1e-6}
        fit_valid = last_delta is not None and last_delta <= 1e-4 and all(v['normalized'] for c in checks.values() for v in c.values())
        diagnostics.update(fit_valid=fit_valid, parameter_validity=checks, last_parameter_change=last_delta)
        if not fit_valid:
            raise RuntimeError('Numerically invalid fit: iteration tolerance or positive normalized m/u criterion failed; no valid matching/NIL decisions emitted.')
        pred=linker.inference.predict(threshold_match_probability=0).as_pandas_dataframe()
        qindex={q['record_id']:i for i,q in enumerate(qf)}; cindex={c['record_id']:j for j,c in enumerate(cf)}
        matrix=np.zeros((len(qf),len(cf)))
        for r in pred[['unique_id_l','unique_id_r','match_probability']].itertuples(index=False,name=None):
            left,right,p=r
            qi,ci=(left,right) if left in qindex else (right,left)
            matrix[qindex[qi],cindex[ci]]=float(p)
        rows=ranking_rows(qf,cf,matrix,tag+'-splink',view,.95,0,tie_matrix=lexical_matrix(qf,cf))
        diagnostics['tie_break'] = 'Frozen lexical score within exactly equal posterior values; posteriors unchanged'
        diagnostics['probability_sums'] = {c['output_column_name']: {k: sum(float(l.get(k,0)) for l in c['comparison_levels'] if not l.get('is_null_level')) for k in ['m_probability','u_probability']} for c in model['comparisons']}
        # A closed catalog ranks every evidence-bearing query; retain the predeclared
        # posterior threshold as a separate decision for calibrated-precision review.
        for r in rows: r['fit_valid']=True; r['threshold_095_prediction']=r['top_candidates'][0]['catalog_id'] if r['top_candidates'][0]['score']>=.95 and not r['abstained'] else None
        write(RESULTS/f'{tag}-splink-{split}-{view}.jsonl',rows)
        prior_sensitivity(rows,prior,len(catalogs),RESULTS/f'{tag}-splink-{split}-{view}-prior-sensitivity.jsonl')
        diagnostics['prior_sensitivity']={'overlap_fractions':[.25,.5,.8,1.0],'pair_priors':[f/len(catalogs) for f in [.25,.5,.8,1.0]],'mode':'Fitted likelihood ratios held fixed; EM not refitted; no outcome selection.'}
        diagnostics.update(status='predictions_saved',pair_count=len(pred),ranking_ties=sum(r['margin']==0 for r in rows),
            parameter_boundary_warning='Posterior values are model outputs; conditional independence and EM identifiability are not established.')
        for col in pred:
            if col.startswith('gamma_'): diagnostics.setdefault('comparison_level_counts',{})[col]={str(k):int(v) for k,v in pred[col].value_counts(dropna=False).items()}
        out=CACHE/'matrices';out.mkdir(exist_ok=True)
        np.savez_compressed(out/f'{tag}-splink-{split}-{view}.npz',scores=matrix,query_ids=[q['record_id'] for q in qf],catalog_ids=[c['record_id'] for c in cf])
    except Exception as exc:
        diagnostics.update(status='failed_fit',fit_valid=False,error=repr(exc))
        write(RESULTS/f'{tag}-splink-{split}-{view}.jsonl',[{'query_id':q['record_id'],'method':tag+'-splink','prediction':None,'top_candidates':[], 'abstained':True,'input_status':'baseline_failed','fit_valid':False,'decision_status':'failed','status':'failed','target_ids':[]} for q in qf])
    finally:
        logging.getLogger('splink').removeHandler(handler)
        changes = re.findall(r'Iteration (\d+): Largest change in params was ([+-]?[0-9.eE-]+)', log.getvalue())
        diagnostics['last_iteration'] = int(changes[-1][0]) if changes else None
        diagnostics['last_parameter_change'] = abs(float(changes[-1][1])) if changes else None
        diagnostics['convergence_tolerance_met'] = abs(float(changes[-1][1])) < .0001 if changes else None
        (RESULTS/f'{tag}-splink-{split}-{view}-diagnostics.json').write_text(json.dumps(diagnostics,indent=2,default=lambda x:bool(x) if isinstance(x,np.bool_) else str(x))+'\n')
        (RESULTS/f'{tag}-splink-{split}-{view}.log').write_text(log.getvalue())
    print('Splink',diagnostics['status'],'; no outcomes read.',flush=True)


def clip_run(split, view):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    queries,catalogs=inputs(split,view)
    cp=P['clip']; kwargs={'revision':cp['revision'],'cache_dir':str(CACHE/'huggingface')}
    processor=CLIPProcessor.from_pretrained(cp['model_id'],use_fast=False,**kwargs)
    model=CLIPModel.from_pretrained(cp['model_id'],**kwargs)
    device='mps' if torch.backends.mps.is_available() else 'cpu';model=model.to(device).eval()
    text=[cp['catalog_prompt'].format(title=c['title'],brand=c['brand']) for c in catalogs]
    lengths=[len(processor.tokenizer(t)['input_ids']) for t in text]
    text_vectors=[];image_vectors=[];status=[]
    with torch.inference_mode():
        for start in range(0,len(text),32):
            x=processor(text=text[start:start+32],return_tensors='pt',padding=True,truncation=True,max_length=77).to(device)
            y=model.get_text_features(**x);y=y/y.norm(dim=-1,keepdim=True)
            text_vectors.append(y.cpu().numpy())
        for start in range(0,len(queries),16):
            batch=[];valid=[]
            for q in queries[start:start+16]:
                try: batch.append(Image.open(resolve_image(q)).convert('RGB'));valid.append(True);status.append('ok')
                except Exception: batch.append(Image.new('RGB',(224,224)));valid.append(False);status.append('failed')
            x=processor(images=batch,return_tensors='pt').to(device)
            y=model.get_image_features(**x);y=y/y.norm(dim=-1,keepdim=True)
            arr=y.cpu().numpy();arr[np.logical_not(valid)]=0;image_vectors.append(arr)
            if start%160==0:print(f'CLIP {split}: {min(start+16,len(queries))}/{len(queries)}',flush=True)
    iv=np.concatenate(image_vectors);tv=np.concatenate(text_vectors);scores=iv@tv.T
    qf=[{**q,'status':s} for q,s in zip(queries,status)]
    write(RESULTS/f'clip-{split}-{view}.jsonl',ranking_rows(qf,catalogs,scores,'clip',view,.25,.02))
    diagnostics={'model':cp,'device':device,'queries':len(queries),'catalog_rows':len(catalogs),'image_failures':status.count('failed'),
                 'catalog_text_token_lengths':{'median':float(np.median(lengths)),'max':max(lengths),'truncated_rows':sum(n>77 for n in lengths)},
                 'input_image_size':224,'image_preprocessing':processor.image_processor.to_dict(),'outcomes_read':False}
    (RESULTS/f'clip-{split}-{view}-diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    out=CACHE/'matrices';out.mkdir(exist_ok=True)
    np.savez_compressed(out/f'clip-{split}-{view}.npz',scores=scores,query_ids=[q['record_id'] for q in queries],catalog_ids=[c['record_id'] for c in catalogs])
    print('CLIP predictions saved; no outcomes read.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['ocr','lexical','splink','clip','freeze-policies'])
    p.add_argument('--split',choices=['dev','dev80','test'],default='dev')
    p.add_argument('--view',choices=['closed','nil'],default='closed')
    p.add_argument('--extracted',type=Path);p.add_argument('--tag',default='ocr')
    a=p.parse_args();RESULTS.mkdir(exist_ok=True)
    if a.command=='freeze-policies':freeze_policies();return
    check_policy(a.split)
    if a.command=='ocr':run_ocr(a.split)
    elif a.command=='lexical':lexical(a.split,a.view,a.extracted,a.tag)
    elif a.command=='splink':splink_run(a.split,a.view,a.extracted,a.tag)
    else:clip_run(a.split,a.view)


if __name__=='__main__':main()
