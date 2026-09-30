from pathlib import Path
import sys,importlib.util
import pandas as pd
B=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('legacy',B/'main_state_analysis.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d,_=m.load_source_records();r=m.clean_text(d.patient_id_raw);sex=m.clean_text(d.sex_raw).fillna('UNK');d['id_key']=d.center.astype(str)+'|ID:'+r.astype(str)+'|SEX:'+sex.astype(str);labs=['WBC','HGB','PLT','NEUT'];out=[]
for key,method in [('patient_id','name_sex'),('id_key','source_id_sex')]:
 for center,g in d.groupby('center'):
  v=g.dropna(subset=[key,'sample_date']).copy();v['date']=pd.to_datetime(v.sample_date).dt.normalize();a=v.groupby([key,'date'])[labs].median().reset_index().sort_values([key,'date']);gp=a.groupby(key);a['ndate']=gp.date.shift(-1);a['gap']=(a.ndate-a.date).dt.days
  for x in labs:a['next_'+x]=gp[x].shift(-1)
  a=a[a.gap.between(1,3)&a[labs+['next_'+x for x in labs]].notna().all(axis=1)].copy()
  for prefix,col in [('', 'base'),('next_','next')]:
   n=(a[prefix+'WBC']>=100).astype(int)+(a[prefix+'HGB']<60).astype(int)+(a[prefix+'PLT']<20).astype(int)+(a[prefix+'NEUT']<.5).astype(int);a[col]=n.clip(upper=2).map({0:'C0',1:'C1',2:'C2+'})
  for b,z in a.groupby('base'):
   for dest in ['C0','C1','C2+']:
    out.append(dict(center=center,linkage_method=method,base=b,next=dest,pairs=len(z),events=int((z.next==dest).sum()),rate=float((z.next==dest).mean())))
pd.DataFrame(out).to_csv(O/'11_linkage_state_matrix_sensitivity.csv',index=False,encoding='utf-8-sig');print(pd.DataFrame(out).query("center=='铜陵'").to_string(index=False))
