"""Score frozen 80-query development arms with the shared complete-decision contract."""
from pathlib import Path
import importlib.util
import json

HERE=Path(__file__).resolve().parent

def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x]

def main():
    spec=importlib.util.spec_from_file_location('common_evaluation',HERE.parent/'evaluation.py')
    e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    keep=set(json.loads((HERE.parent/'contracts/image-development-subset.json').read_text())['queries'])
    gs=[g for g in read(HERE/'evaluator/gold-dev.jsonl') if g['query_id'] in keep]
    truth={g['query_id']:[g['catalog_id']] for g in gs};groups={g['query_id']:g['family_id'] for g in gs}
    methods={'ocr_lexical':'ocr-lexical','ocr_splink':'ocr-splink','clip':'clip',
             'luna_ocr_lexical':'luna-ocr-lexical','luna_ocr_splink':'luna-ocr-splink',
             'luna_pixel_lexical':'luna-pixel-lexical','luna_pixel_splink':'luna-pixel-splink'}
    results={};outcomes={};decisions={}
    for method,stem in methods.items():
        rows=read(HERE/'results'/f'{stem}-dev80-closed.jsonl')
        decisions[method]={r['query_id']:{'status':r['decision_status'],'target_ids':r['target_ids']} for r in rows}
        outcomes[method]=e.outcomes(truth,decisions[method]);results[method]=e.summarize(outcomes[method])
    gates={};paired={};eligible=['ocr_lexical','ocr_splink','clip']
    for method in [m for m in methods if m.startswith('luna_')]:
        gates[method]=e.development_gate(results,results[method],eligible_methods=eligible)
        paired[method]={reference:e.paired_comparison(outcomes[reference],outcomes[method],groups) for reference in eligible}
    failures={}
    for arm in ['ocr','pixel']:
        rows=read(HERE/'results/model-development-v1'/f'{arm}-extraction.jsonl')
        failures[arm]={'queries':len(rows),'valid_extraction_rows':sum(r['status']=='ok' for r in rows),'failed_extraction_rows':sum(r['status']!='ok' for r in rows),
                       'failed_query_ids':[r['record_id'] for r in rows if r['status']!='ok']}
    payload={'queries':80,'catalog_records':96,'source_families':len(set(groups.values())),'summaries':results,'gates':gates,
             'paired_comparisons':paired,'extraction_failures':failures,
             'primary_semantic_fits':['luna_ocr_splink','luna_pixel_splink'],
             'normalization_comparisons':['luna_ocr_lexical','luna_pixel_lexical'],
             'test_outcomes_read':False,'policy_changed_from_results':False}
    out=HERE/'results/model-development-v1'
    (out/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
    (out/'common-decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    print(json.dumps({'summaries':{k:{x:v[x] for x in ['correct_complete_decisions','automatic_decisions','false_links','failed_queries']} for k,v in results.items()},'gates':{k:v['pass'] for k,v in gates.items()}},indent=2))

if __name__=='__main__':main()
