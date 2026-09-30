from pathlib import Path
import sys,importlib.util
import pandas as pd,numpy as np
B=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('legacy',B/'main_state_analysis.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d,_=m.load_source_records();d['day']=pd.to_datetime(d.sample_date).dt.normalize();d=d.dropna(subset=['patient_id','day']);labs=['WBC','HGB','PLT','NEUT'];labels={0:'AML',1:'ALL',2:'CML',3:'CLL'}
pat=d.groupby(['center','patient_id']).agg(subtypes=('level2_type','nunique'),rows=('day','size'));day_conf=d.groupby(['center','patient_id','day']).level2_type.nunique()
print('patient keys with >1 subtype:',int((pat.subtypes>1).sum()),'of',len(pat));print('same-day subtype conflicts:',int((day_conf>1).sum()))
print('subtype raw counts',d.level2_type.value_counts(dropna=False).sort_index().to_dict());print('APL-like flag counts',d.level3_aml_m3.value_counts(dropna=False).to_dict())
summary=pat.reset_index().groupby('center').agg(keys=('patient_id','size'),multi_subtype_keys=('subtypes',lambda x:int((x>1).sum()))).reset_index();summary.to_csv(O/'12_subtype_label_audit.csv',index=False,encoding='utf-8-sig');print(summary.to_string(index=False))
