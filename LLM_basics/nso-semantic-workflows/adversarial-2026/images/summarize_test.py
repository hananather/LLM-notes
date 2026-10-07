"""Score frozen free TEST baselines after development gate closure.

No model execution, matching, tuning, source selection or image replacement
belongs here. All strata follow the saved input-only duplicate audit.
"""
from pathlib import Path
import importlib.util
import json
import time
import numpy as np

HERE=Path(__file__).resolve().parent

def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x]

def cluster_accuracy_interval(rows,groups):
    stats={}
    for r in rows:
        z=stats.setdefault(groups[r['query_id']],[0,0]);z[0]+=int(r['correct']);z[1]+=1
    if not stats:return None
    values=np.asarray(list(stats.values()),float);rng=np.random.default_rng(20261007);boot=[]
    for _ in range(2000):
        sample=values[rng.integers(0,len(values),len(values))].sum(axis=0);boot.append(sample[0]/sample[1])
    return {'source_families':len(stats),'accuracy_95_percent_interval':np.quantile(boot,[.025,.975]).tolist(),
            'bootstrap_replicates':2000,'seed':20261007,'scope':'This source sample and frozen input-similarity family definition, not NSO population uncertainty.'}

def main():
    spec=importlib.util.spec_from_file_location('common_evaluation',HERE.parent/'evaluation.py')
    e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    test_ids={q['record_id'] for q in read(HERE/'data/queries-test.jsonl')}
    flags=read(HERE/'evaluator/duplicate-image-flags.jsonl')
    any_flag={x[k] for x in flags for k in ['a','b']} & test_ids
    cross_flag={x[k] for x in flags if x['cross_split'] for k in ['a','b']} & test_ids
    result={'evaluated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'scope':'Free baseline scoring after image-model development pair gate failed; no image-model TEST calls.',
            'methods':['ocr_lexical','ocr_splink','clip'],'views':{},
            'input_duplicate_audit':{'flag_pairs':len(flags),'cross_split_flag_pairs':sum(x['cross_split'] for x in flags),
              'exact_duplicate_pairs':sum(x['byte_equal'] or x['pixel_equal'] for x in flags),
              'test_queries_in_any_flag':len(any_flag),'test_queries_in_cross_split_flag':len(cross_flag),
              'status':'Unadjudicated pHash<=4 screen. False positives/negatives possible. No primary records dropped.'}}
    decisions={}
    for view in ['closed','nil']:
        suffix='-nil' if view=='nil' else ''
        gold=read(HERE/'evaluator'/f'gold-test{suffix}.jsonl')
        truth={g['query_id']:[g['catalog_id']] if g['catalog_id'] else [] for g in gold}
        groups={g['query_id']:g['family_id'] for g in gold};ambiguous={g['query_id'] for g in gold if g['input_ambiguous_family']}
        methods={};outcomes={};decisions[view]={}
        for method,stem in [('ocr_lexical','ocr-lexical'),('ocr_splink','ocr-splink'),('clip','clip')]:
            pred=read(HERE/'results'/f'{stem}-test-{view}.jsonl')
            pd={p['query_id']:{'status':p['decision_status'],'target_ids':p['target_ids']} for p in pred}
            decisions[view][method]=pd;out=e.outcomes(truth,pd);outcomes[method]=out
            summary=e.summarize(out);summary['family_bootstrap']=cluster_accuracy_interval(out,groups)
            strata={}
            for label,ids in [('cross_split_phash_flagged',cross_flag),('cross_split_phash_unflagged',test_ids-cross_flag),
                              ('any_phash_flagged',any_flag),('any_phash_unflagged',test_ids-any_flag),
                              ('source_similar_family',ambiguous),('source_singleton_family',test_ids-ambiguous)]:
                xs=[x for x in out if x['query_id'] in ids]
                strata[label]={'summary':e.summarize(xs),'family_bootstrap':cluster_accuracy_interval(xs,groups)} if xs else {'queries':0}
            methods[method]={'summary':summary,'input_only_strata':strata}
        paired={method:e.paired_comparison(outcomes['clip'],outcomes[method],groups) for method in ['ocr_lexical','ocr_splink']}
        result['views'][view]={'methods':methods,'paired_vs_clip':paired}
    (HERE/'results/free-test-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'results/common-test-decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    print(json.dumps({'duplicate_audit':result['input_duplicate_audit'],'views':{v:{m:{k:entry['summary'][k] for k in ['queries','correct_complete_decisions','automatic_decisions','false_links','review_queries','failed_queries']} for m,entry in x['methods'].items()} for v,x in result['views'].items()}},indent=2))

if __name__=='__main__':main()
