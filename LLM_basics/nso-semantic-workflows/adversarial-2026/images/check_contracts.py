"""Offline failure/NIL fixtures and frozen interface checks; no real outcomes."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import baseline as b

s=importlib.util.spec_from_file_location('shared_evaluation',b.HERE.parent/'evaluation.py')
e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
cases=[('empty OCR','empty_or_failed','',True,.1,.1,'failed'),
       ('failed fit','ok','text',False,.9,.1,'failed'),
       ('ambiguous margin','ok','text',True,.6,.001,'review'),
       ('successful null extraction','ok','',True,.1,.1,'review'),
       ('valid low score','ok','text',True,.1,.1,'nil'),
       ('valid link','ok','text',True,.6,.1,'linked')]
rows=[]
for label,input_status,text,fit_valid,score,margin,expected in cases:
 status=b.decision_status(input_status,bool(text),'nil',score,margin,.3,.03,fit_valid)
 assert status==expected,(label,status,expected)
 decision={'status':status,'target_ids':['c1'] if status=='linked' else []}
 o=e.outcomes({'q':[]},{'q':decision})[0]
 assert o['correct']==(expected=='nil'),(label,o)
 rows.append({'case':label,'decision_status':status,'correct_on_nil_truth':o['correct'],'passed':True})
assert b.ranking_rows([{'record_id':'q','status':'ok','text':''}],[{'record_id':'c1'},{'record_id':'c2'}],np.array([[0.,0.]]),'fixture','closed',.3,.03)[0]['decision_status']=='review'
queries,catalogs=b.inputs('dev80');assert len(queries)==80 and len(catalogs)==96
assert len({q['record_id'] for q in queries})==80
report={'fixtures':rows,'dev80_queries':80,'catalog_records':96,'test_outcomes_read':False}
(b.RESULTS/'contract-fixtures.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
