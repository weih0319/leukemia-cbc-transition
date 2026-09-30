from pathlib import Path
import sys, importlib.util
import pandas as pd, numpy as np
B=Path(__file__).resolve().parent.parent
sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('legacy',B/'main_state_analysis.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d,extract=m.load_source_records(); labs=['WBC','HGB','PLT','NEUT'];o=Path(__file__).resolve().parent
rows=[];links=[]
for center,g in d.groupby('center'):
 valid=g.dropna(subset=['patient_id','sample_date']).copy();valid['date']=pd.to_datetime(valid.sample_date).dt.normalize()
 daily=valid.groupby(['patient_id','date'])[labs].median().reset_index().sort_values(['patient_id','date'])
 gp=daily.groupby('patient_id');daily['next_date']=gp.date.shift(-1);daily['days']=(daily.next_date-daily.date).dt.days
 for x in labs:daily['next_'+x]=gp[x].shift(-1)
 adj=daily[daily.next_date.notna()];win=adj[adj.days.between(1,3)]
 complete=win[win[labs+['next_'+x for x in labs]].notna().all(axis=1)].copy()
 c0=complete[(complete.WBC<100)&(complete.HGB>=60)&(complete.PLT>=20)&(complete.NEUT>=.5)]
 rows.append(dict(center=center,retained_source_rows=len(g),source_patient_keys=g.patient_id.nunique(),missing_patient_key_rows=int(g.patient_id.isna().sum()),missing_date_rows=int(g.sample_date.isna().sum()),valid_rows=len(valid),natural_day_rows=len(daily),same_day_extra_rows=len(valid)-len(daily),patients_with_dates=daily.patient_id.nunique(),complete_daily_rows=int(daily[labs].notna().all(axis=1).sum()),adjacent_all_gaps=len(adj),adjacent_1to3_days=len(win),complete_1to3_pairs=len(complete),patients_complete_1to3=complete.patient_id.nunique(),c0_pairs=len(c0),c0_patients=c0.patient_id.nunique(),first_date=valid.date.min().date(),last_date=valid.date.max().date(),median_gap_days=adj.days.median()))
 # Potential merging: distinct nonmissing source IDs assigned to the same name-derived key.
 v=g[['internal_key','patient_id_raw','name_raw','sex_raw']].copy();v['raw_id']=m.clean_text(v.patient_id_raw);v['name']=m.clean_text(v.name_raw)
 named=v[v.name.notna() & v.raw_id.notna()]
 nids=named.groupby('internal_key').raw_id.nunique()
 idkeys=named.groupby('raw_id').internal_key.nunique()
 links.append(dict(center=center,name_linked_keys=int(nids.size),name_linked_keys_multiple_raw_ids=int((nids>1).sum()),name_linked_rows_in_multiple_id_keys=int(named.internal_key.isin(nids[nids>1].index).sum()),distinct_raw_ids_multiple_name_keys=int((idkeys>1).sum()),rows_missing_name=int(v.name.isna().sum()),rows_missing_source_id=int(v.raw_id.isna().sum())))
pd.DataFrame(rows).to_csv(o/'07_corrected_cohort_flow.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(links).to_csv(o/'08_linkage_aggregate_audit.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(links).to_string(index=False))
