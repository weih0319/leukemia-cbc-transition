from pathlib import Path
import sys,importlib.util
import pandas as pd,numpy as np
from statsmodels.stats.proportion import proportion_confint
B=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('legacy',B/'main_state_analysis.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d,_=m.load_source_records();labs=['WBC','HGB','PLT','NEUT'];out=[]
rawid=m.clean_text(d.patient_id_raw);sex=m.clean_text(d.sex_raw).fillna('UNK')
d['id_key']=d.center.astype(str)+'|ID:'+rawid.astype(str)+'|SEX:'+sex.astype(str)
for keycol,method in [('patient_id','name_sex'),('id_key','source_id_sex')]:
 for center,g in d.groupby('center'):
  v=g.dropna(subset=[keycol,'sample_date']).copy();v['date']=pd.to_datetime(v.sample_date).dt.normalize()
  a=v.groupby([keycol,'date'])[labs].median().reset_index().sort_values([keycol,'date']);gp=a.groupby(keycol)
  a['next_date']=gp.date.shift(-1);a['days']=(a.next_date-a.date).dt.days
  for x in labs:a['next_'+x]=gp[x].shift(-1)
  a=a[a.days.between(1,3)&a[labs+['next_'+x for x in labs]].notna().all(axis=1)]
  c=a[(a.WBC<100)&(a.HGB>=60)&(a.PLT>=20)&(a.NEUT>=.5)].copy()
  c['event']=(c.next_WBC>=100)|(c.next_HGB<60)|(c.next_PLT<20)|(c.next_NEUT<.5)
  f=c.sort_values('date').drop_duplicates(keycol)
  low,hi=proportion_confint(int(f.event.sum()),len(f),method='wilson')
  out.append(dict(center=center,linkage_method=method,daily_rows=len(v.groupby([keycol,'date'])),c0_pairs=len(c),pair_events=int(c.event.sum()),pair_rate=c.event.mean(),eligible_keys=len(f),first_events=int(f.event.sum()),first_rate=f.event.mean(),first_wilson_low=low,first_wilson_high=hi))
r=pd.DataFrame(out);r.to_csv(O/'09_linkage_sensitivity.csv',index=False,encoding='utf-8-sig');print(r.to_string(index=False))
