from pathlib import Path
import sys,importlib.util,json,hashlib,platform
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
from statsmodels.stats.proportion import proportion_confint
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;ML=BASE/'ml'
LAB=['WBC','HGB','PLT','NEUT'];SEED=20260917
THR={'primary':(100,60,20,.5),'WBC_50':(50,60,20,.5),'HGB_70':(100,70,20,.5),'HGB_80':(100,80,20,.5),'PLT_10':(100,60,10,.5),'PLT_30':(100,60,30,.5),'NEUT_1':(100,60,20,1.0),'broad':(50,80,30,1.0)}
def save(d,n):d.to_csv(ROOT/n,index=False,encoding='utf-8-sig')
def state(d,threshold,prefix=''):
 w,h,p,n=threshold
 return d[prefix+'WBC'].ge(w).astype(int)+d[prefix+'HGB'].lt(h).astype(int)+d[prefix+'PLT'].lt(p).astype(int)+d[prefix+'NEUT'].lt(n).astype(int)
def ci(e,n):return proportion_confint(e,n,method='wilson') if n else (np.nan,np.nan)
def pct(a):return float(a.mean()) if len(a) else np.nan
def summary(pairs,sc):
 t=[]
 for center,group in [('Overall',pairs)]+list(pairs.groupby('center')):
  c0=group[group.base_state.eq(0)].copy();first=c0.sort_values('sample_date').drop_duplicates('group');y=first.new_critical.to_numpy(int)
  n=len(c0);ev=int(c0.new_critical.sum());pf=c0.groupby('group').new_critical.mean()
  for estimand,nn,ee,rate in [('first_patient',len(first),int(y.sum()),pct(y)),('all_pairs',n,ev,ev/n if n else np.nan),('patient_equal_pairs',len(pf),np.nan,pct(pf))]:
   low,high=ci(ee,nn) if estimand=='first_patient' else (np.nan,np.nan)
   t.append({'scenario':sc,'center':center,'estimand':estimand,'patients':len(first),'pairs':n,'denominator':nn,'events':ee,'rate':rate,'wilson_low_first_only':low,'wilson_high_first_only':high})
 return t

def main():
 spec=importlib.util.spec_from_file_location('legacy',BASE/'main_state_analysis.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 raw,_=legacy.load_source_records()
 src=raw[['center','patient_id','sample_date']+LAB].dropna(subset=['patient_id','sample_date']).copy();del raw
 src.sample_date=pd.to_datetime(src.sample_date).dt.normalize()
 daily=src.groupby(['center','patient_id','sample_date'],observed=True)[LAB].median().reset_index().sort_values(['center','patient_id','sample_date']).reset_index(drop=True)
 assert not daily.duplicated(['center','patient_id','sample_date']).any()
 daily['group']=daily.center+'|'+daily.patient_id;gp=daily.groupby('group',sort=False)
 daily['next_date']=gp.sample_date.shift(-1);daily['days']=(daily.next_date-daily.sample_date).dt.days
 for x in LAB:daily['next_'+x]=gp[x].shift(-1)
 complete=daily[LAB+['next_'+x for x in LAB]].notna().all(axis=1)
 paired=daily[complete&daily.days.between(1,3)].copy().reset_index(drop=True)
 # Check C0 landmark identity against independently generated ML cohort.
 chk=paired[state(paired,THR['primary']).eq(0)]
 old=pd.read_csv(ML/'round2/deidentified_landmarks.csv.gz',usecols=['center','patient_id','sample_date'])
 ids=lambda d:set(zip(d.center,d.patient_id,pd.to_datetime(d.sample_date).dt.normalize()))
 assert ids(chk)==ids(old) and len(chk)==len(old)==3909
 print('C0 cohort cross-check passed:',len(chk),flush=True)
 results=[];matrix=[];components=[];near=[];bootstrap_summary=[];bootstrap_matrix=[]
 rng=np.random.default_rng(SEED)
 for label,threshold in THR.items():
  p=paired.copy();p['base_state']=state(p,threshold);p['next_state']=state(p,threshold,'next_');p['base_cat']=p.base_state.clip(upper=2).map({0:'C0',1:'C1',2:'C2+'});p['next_cat']=p.next_state.clip(upper=2).map({0:'C0',1:'C1',2:'C2+'});p['new_critical']=p.base_state.eq(0)&p.next_state.gt(0)
  results+=summary(p,label)
  if label!='primary':continue
  for center,g in [('Overall',p)]+list(p.groupby('center')):
   for interval,z in [('1-3 days',g),('1 day',g[g.days.eq(1)]),('2-3 days',g[g.days.between(2,3)])]:
    for b in ['C0','C1','C2+']:
     v=z[z.base_cat.eq(b)];n=len(v)
     for dest in ['C0','C1','C2+']:
      count=int(v.next_cat.eq(dest).sum());lo,hi=ci(count,n)
      matrix.append({'center':center,'interval':interval,'base':b,'next':dest,'pairs':n,'events':count,'pair_rate':count/n if n else np.nan,'naive_wilson_low':lo,'naive_wilson_high':hi,'unique_patients':v.group.nunique()})
  c0=p[p.base_state.eq(0)].copy();first=c0.sort_values('sample_date').drop_duplicates('group')
  for name,col,limit,op in [('WBC','next_WBC',threshold[0],'ge'),('HGB','next_HGB',threshold[1],'lt'),('PLT','next_PLT',threshold[2],'lt'),('NEUT','next_NEUT',threshold[3],'lt')]:
   for frame,estimand in [(first,'first_patient'),(c0,'all_pairs')]:
    ev=int(getattr(frame[col],op)(limit).sum());lo,hi=ci(ev,len(frame))
    components.append({'estimand':estimand,'component':name,'observations':len(frame),'events':ev,'rate':ev/len(frame),'naive_wilson_low':lo,'naive_wilson_high':hi})
  components.append({'estimand':'first_patient','component':'C0_to_C1','observations':len(first),'events':int(first.next_state.eq(1).sum()),'rate':first.next_state.eq(1).mean()})
  components.append({'estimand':'first_patient','component':'C0_to_C2plus','observations':len(first),'events':int(first.next_state.ge(2).sum()),'rate':first.next_state.ge(2).mean()})
  # 2019 plan exploratory ranges, kept separate from threshold sensitivity.
  first['near_n']=(first.WBC.ge(50)&first.WBC.lt(100)).astype(int)+(first.HGB.ge(60)&first.HGB.lt(80)).astype(int)+(first.PLT.ge(20)&first.PLT.lt(50)).astype(int)+(first.NEUT.ge(.5)&first.NEUT.lt(1)).astype(int)
  first['near_cat']=first.near_n.clip(upper=2).map({0:'0',1:'1',2:'2+'})
  for k,z in first.groupby('near_cat'):
   e=int(z.new_critical.sum());lo,hi=ci(e,len(z));near.append({'near_critical_count':k,'patients':len(z),'events':e,'rate':e/len(z),'wilson_low':lo,'wilson_high':hi})
  # Patient-cluster bootstrap: centre-specific first, all-pair and patient-equal proportions.
  sample={}
  for center,z in c0.groupby('center'):
   ps=z.groupby('group').new_critical.agg(['size','sum']).reset_index();ps['first']=z.sort_values('sample_date').drop_duplicates('group').set_index('group').loc[ps.group,'new_critical'].to_numpy(int)
   arr=ps[['size','sum','first']].to_numpy(float);ii=rng.integers(0,len(arr),size=(3000,len(arr)));draw=arr[ii]
   out={'first_patient':draw[:,:,2].mean(axis=1),'all_pairs':draw[:,:,1].sum(axis=1)/draw[:,:,0].sum(axis=1),'patient_equal_pairs':(draw[:,:,1]/draw[:,:,0]).mean(axis=1)}
   sample[center]=out
   for est,val in out.items():
    bootstrap_summary.append({'center':center,'estimand':est,'low':np.quantile(val,.025),'high':np.quantile(val,.975),'replicates':len(val)})
   delta=out['first_patient']-out['all_pairs'];bootstrap_summary.append({'center':center,'estimand':'first_minus_all_pairs','low':np.quantile(delta,.025),'high':np.quantile(delta,.975),'replicates':len(delta),'estimate':float(ps['first'].mean()-ps['sum'].sum()/ps['size'].sum())})
  for est in ['first_patient','all_pairs']:
   a=sample['安医大'][est];c=sample['铜陵'][est];diff=a-c
   bootstrap_summary.append({'center':'安医大 minus 铜陵','estimand':est,'low':np.quantile(diff,.025),'high':np.quantile(diff,.975),'replicates':len(diff),'estimate':float(np.mean(diff))})
  # For full matrix, cluster-bootstrap each row with patients as units.
  for interval,z in [('1 day',p[p.days.eq(1)]),('2-3 days',p[p.days.between(2,3)]),('1-3 days',p)]:
   for b in ['C0','C1','C2+']:
    q=z[z.base_cat.eq(b)];tab=q.pivot_table(index='group',columns='next_cat',values='days',aggfunc='size',fill_value=0).reindex(columns=['C0','C1','C2+'],fill_value=0)
    vals=tab.to_numpy(float);ii=rng.integers(0,len(vals),size=(2000,len(vals)));rr=vals[ii].sum(axis=1);den=rr.sum(axis=1)
    for j,dest in enumerate(['C0','C1','C2+']):
     rate=rr[:,j]/den;bootstrap_matrix.append({'interval':interval,'base':b,'next':dest,'patient_cluster_low':np.quantile(rate,.025),'patient_cluster_high':np.quantile(rate,.975),'replicates':len(rate)})
 save(pd.DataFrame(results),'01_estimand_and_threshold_sensitivity.csv');save(pd.DataFrame(matrix),'02_state_transition_matrix.csv');save(pd.DataFrame(components),'03_first_transition_components.csv');save(pd.DataFrame(near),'04_near_critical_gradient.csv');save(pd.DataFrame(bootstrap_summary),'05_patient_cluster_bootstrap.csv');save(pd.DataFrame(bootstrap_matrix),'06_matrix_cluster_intervals.csv')
 manifest={'seed':SEED,'python':platform.python_version(),'thresholds':THR,'patient_cluster_bootstrap_replicates':3000,'matrix_cluster_bootstrap_replicates':2000,'source_loader_sha256':hashlib.sha256((BASE/'main_state_analysis.py').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'eligible_c0_identity_verified_against_ml_round2':True,'interpretation':'Observed complete adjacent retests on calendar dates; not unobserved incidence or causally comparable centre quality.'}
 (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 print(pd.DataFrame(results).query("scenario=='primary'").to_string(index=False),flush=True)
 print('DONE',flush=True)
if __name__=='__main__':main()
