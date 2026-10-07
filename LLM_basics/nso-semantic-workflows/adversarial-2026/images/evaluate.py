"""Score saved predictions. Test outcome access requires an explicit switch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]

def score(prediction_path,split='dev80',view='closed',allow_test=False):
    if split=='test' and not allow_test:
        raise RuntimeError('TEST outcomes remain sealed; pass --allow-test-outcomes only after all policies are frozen.')
    if split=='test' and not (HERE/'policy-freeze.json').exists():raise RuntimeError('Missing policy freeze')
    suffix='-nil' if split=='test' and view=='nil' else ''
    gold=read(HERE/'evaluator'/f'gold-{"dev" if split=="dev80" else split}{suffix}.jsonl')
    if split=='dev80':
        keep=set(json.loads((HERE.parent/'contracts/image-development-subset.json').read_text())['queries'])
        gold=[g for g in gold if g['query_id'] in keep]
    pred={p['query_id']:p for p in read(prediction_path)}
    outcomes=[]
    for g in gold:
        p=pred.get(g['query_id'],{})
        ranked=[x['catalog_id'] for x in p.get('top_candidates',[])]
        target=g['catalog_id'];chosen=p.get('prediction')
        found=target is not None and target in ranked
        rank=ranked.index(target)+1 if found else None
        status = p.get('decision_status', 'failed')
        valid = bool(p) and p.get('fit_valid') is not False and p.get('input_status') != 'baseline_failed'
        automatic = valid and status in {'linked','nil'}
        outcomes.append({**g,'prediction':chosen,'correct':automatic and chosen==target,'valid_fit':valid,'rank':rank,
                         'decision_status':status,'accepted':automatic,'missing_prediction_row':not bool(p)})
    n=len(outcomes);links=[r for r in outcomes if r['catalog_id'] is not None];nil=[r for r in outcomes if r['catalog_id'] is None]
    accepted=[r for r in outcomes if r['accepted']]
    result={'predictions':str(Path(prediction_path).name),'split':split,'view':view,'queries':n,
            'correct':sum(r['correct'] for r in outcomes),'accuracy':sum(r['correct'] for r in outcomes)/n,
            'coverage':len(accepted)/n,'selective_precision':sum(r['correct'] for r in accepted)/len(accepted) if accepted else None,
            'recall_at_5':sum(r['rank'] is not None and r['rank']<=5 for r in links)/len(links) if links else None,
            'recall_at_20':sum(r['rank'] is not None and r['rank']<=20 for r in links)/len(links) if links else None,
            'mrr_at_20_lower_bound':sum(1/r['rank'] if r['rank'] else 0 for r in links)/len(links) if links else None,
            'nil_queries':len(nil),'nil_false_links':sum(r['decision_status']=='linked' for r in nil),'link_abstentions':sum(r['decision_status'] in {'review','failed'} for r in links),'review_queries':sum(r['decision_status']=='review' for r in outcomes),'failed_queries':sum(r['decision_status']=='failed' for r in outcomes),
            'missing_prediction_rows':sum(r['missing_prediction_row'] for r in outcomes),'invalid_fit_rows':sum(not r['valid_fit'] for r in outcomes),
            'families':len({r['family_id'] for r in outcomes}),'target':'native catalog association, not physical SKU identity'}
    by={}
    for r in outcomes:by.setdefault(r['family_id'],[]).append(r)
    clusters=list(by.values());rng=np.random.default_rng(20261007);boot=[]
    for _ in range(2000):
        sample=[r for i in rng.integers(0,len(clusters),len(clusters)) for r in clusters[i]]
        boot.append(sum(r['correct'] for r in sample)/len(sample))
    result['family_bootstrap_accuracy_95pct']=list(np.quantile(boot,[.025,.975]))
    result['input_family_strata']={}
    for label,flag in [('source_similar_family',True),('source_singleton_family',False)]:
        subset=[r for r in outcomes if r['input_ambiguous_family']==flag]
        result['input_family_strata'][label]={'queries':len(subset),'correct':sum(r['correct'] for r in subset),'accuracy':sum(r['correct'] for r in subset)/len(subset) if subset else None}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('predictions',type=Path);p.add_argument('--split',choices=['dev','dev80','test'],default='dev80');p.add_argument('--view',choices=['closed','nil'],default='closed');p.add_argument('--allow-test-outcomes',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();r=score(a.predictions,a.split,a.view,a.allow_test_outcomes)
    text=json.dumps(r,indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text)
