from pathlib import Path
import sys,importlib.util
import pandas as pd,numpy as np
B=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('legacy',B/'main_state_analysis.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d,_=m.load_source_records();d['day']=pd.to_datetime(d.sample_date).dt.normalize();d=d.dropna(subset=['patient_id','day']).copy();labs=['WBC','HGB','PLT','NEUT'];keys=['center','patient_id','day'];labels={0:'AML',1:'ALL',2:'CML',3:'CLL'}
stable=d.groupby(['center','patient_id']).level2_type.nunique().eq(1).rename('stable_subtype').reset_index()
daily=d.groupby(keys)[labs].median().reset_index().sort_values(keys)
sub=d.groupby(keys).level2_type.first().reset_index();daily=daily.merge(sub,on=keys,validate='one_to_one').merge(stable,on=['center','patient_id'],validate='many_to_one');daily['subtype']=daily.level2_type.map(labels);daily['group']=daily.center+'|'+daily.patient_id
by=daily.groupby('group',sort=False);daily['next_day']=by.day.shift(-1);daily['gap']=(daily.next_day-daily.day).dt.days;daily['next_subtype']=by.subtype.shift(-1)
for x in labs:daily['next_'+x]=by[x].shift(-1)
p=daily[daily.gap.between(1,3)&daily[labs+['next_'+x for x in labs]].notna().all(axis=1)].copy();p['base_state']=((p.WBC>=100).astype(int)+(p.HGB<60).astype(int)+(p.PLT<20).astype(int)+(p.NEUT<.5).astype(int)).clip(upper=2);p['next_state']=((p.next_WBC>=100).astype(int)+(p.next_HGB<60).astype(int)+(p.next_PLT<20).astype(int)+(p.next_NEUT<.5).astype(int)).clip(upper=2);p['event']=p.base_state.eq(0)&p.next_state.gt(0)
print('complete pairs',len(p),'stable-subtype pairs',int(p.stable_subtype.sum()),'cross-subtype adjacent pairs',int((p.subtype!=p.next_subtype).sum()))
rows=[];mat=[];rng=np.random.default_rng(20260917)
for subtype in ['AML','ALL','CML','CLL']:
 z=p[p.stable_subtype&p.subtype.eq(subtype)];c=z[z.base_state.eq(0)];first=c.sort_values('day').drop_duplicates('group')
 ps=c.groupby('group').event.agg(['size','sum']).reset_index();arr=ps[['size','sum']].to_numpy(float)
 if len(arr)>0:
  ii=rng.integers(0,len(arr),size=(2000,len(arr)));q=arr[ii];v=q[:,:,1].sum(axis=1)/q[:,:,0].sum(axis=1);lo,hi=np.quantile(v,[.025,.975])
 else:lo=hi=np.nan
 rows.append(dict(subtype=subtype,stable_linked_patients=z.group.nunique(),all_complete_pairs=len(z),c0_pairs=len(c),c0_pair_events=int(c.event.sum()),c0_pair_rate=c.event.mean(),c0_pair_cluster_low=lo,c0_pair_cluster_high=hi,first_c0_keys=len(first),first_c0_events=int(first.event.sum()),first_c0_rate=first.event.mean()))
 for base in [0,1,2]:
  x=z[z.base_state.eq(base)]
  for nexts in [0,1,2]:mat.append(dict(subtype=subtype,base_state=base,next_state=nexts,pairs=len(x),events=int(x.next_state.eq(nexts).sum()),rate=x.next_state.eq(nexts).mean() if len(x) else np.nan))
bycentre=[]
for (centre,subtype),z in p[p.stable_subtype].groupby(['center','subtype']):
 c=z[z.base_state.eq(0)];bycentre.append(dict(center=centre,subtype=subtype,c0_pairs=len(c),c0_events=int(c.event.sum()),c0_rate=c.event.mean(),linked_keys=c.group.nunique()))
pd.DataFrame(rows).to_csv(O/'13_stable_subtype_c0_transition.csv',index=False,encoding='utf-8-sig');pd.DataFrame(mat).to_csv(O/'14_stable_subtype_state_matrix.csv',index=False,encoding='utf-8-sig');pd.DataFrame(bycentre).to_csv(O/'15_stable_subtype_by_centre.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(bycentre).to_string(index=False))
