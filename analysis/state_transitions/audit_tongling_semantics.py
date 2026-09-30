from pathlib import Path
import importlib.util,sys,numpy as np,pandas as pd
B=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('legacy',B/'main_state_analysis.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d,_=m.load_source_records(); d=d[d.center.eq('铜陵')].copy();d['rid']=m.clean_text(d.patient_id_raw);d['day']=pd.to_datetime(d.sample_date).dt.normalize();
by=d.groupby('internal_key').agg(raw_ids=('rid','nunique'),rows=('rid','size'),ages=('age','nunique'),min_age=('age','min'),max_age=('age','max'),types=('level2_type','nunique'),days=('day','nunique'))
multi=by[by.raw_ids>1];overlap=d.groupby(['internal_key','day']).rid.nunique();same_day_keys=overlap[overlap>1].reset_index().internal_key.nunique()
# Distinct raw IDs on same day may indicate true collision, or specimen/visit ID changes.
first=d.groupby(['internal_key','rid']).day.min().reset_index();last=d.groupby(['internal_key','rid']).day.max().reset_index();spans=first.merge(last,on=['internal_key','rid']);overlap_period_keys=0
for k,g in spans.groupby('internal_key'):
 if len(g)<2:continue
 x=g[['day_x','day_y']].to_numpy();flag=any(max(x[i,0],x[j,0])<=min(x[i,1],x[j,1]) for i in range(len(x)) for j in range(i+1,len(x)))
 overlap_period_keys+=bool(flag)
out={'multi_id_name_keys':len(multi),'multi_id_keys_age_span_gt_2y':int(((multi.max_age-multi.min_age)>2).sum()),'multi_id_keys_age_span_gt_10y':int(((multi.max_age-multi.min_age)>10).sum()),'multi_id_keys_multiple_leukemia_subtypes':int((multi.types>1).sum()),'multi_id_keys_same_day_multiple_ids':int(same_day_keys),'multi_id_keys_overlapping_id_date_ranges':int(overlap_period_keys),'median_ids_per_multi_key':float(multi.raw_ids.median()),'max_ids_per_multi_key':int(multi.raw_ids.max())}
pd.DataFrame([out]).to_csv(O/'10_tongling_id_semantics_aggregate.csv',index=False,encoding='utf-8-sig');print(out)
