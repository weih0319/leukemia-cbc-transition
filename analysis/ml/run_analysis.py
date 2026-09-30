from pathlib import Path
import sys, json, hashlib, importlib.util, warnings, platform
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
SEED=20260917
LABS=['WBC','HGB','PLT','NEUT']
DEV='安医大'
CENTERS=[DEV,'淮北煤炭','铜陵']
A=['age','sex2','sex_missing','log_WBC','HGB_margin','log_PLT_margin','log_NEUT_margin']
B=A+['trend_'+x for x in LABS]+['has_prior','log_prior_gap']+['missing_prior_'+x for x in LABS]
FEATURES={'A_current':A,'B_dynamic':B}
CONFIGS={'ridge':[{'C':c} for c in [.03,.3,3]],'hist_boost':[{'max_leaf_nodes':l,'l2_regularization':10.0} for l in [3,7]],'random_forest':[{'max_depth':d,'min_samples_leaf':30} for d in [3,5]]}

def save(df,name): df.to_csv(OUT/name,index=False,encoding='utf-8-sig')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def weights(f):
    w=1/f.groupby('group')['group'].transform('size').to_numpy(float)
    return w/w.mean()
def crit(d,prefix=''):
    return d[prefix+'WBC'].ge(100)|d[prefix+'HGB'].lt(60)|d[prefix+'PLT'].lt(20)|d[prefix+'NEUT'].lt(.5)

def prepare():
    spec=importlib.util.spec_from_file_location('legacy_loader',BASE/'main_state_analysis.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    raw,extraction=mod.load_source_records()
    # Never serialize raw names, identifiers, or internal linkage keys.
    cols=['center','patient_id','sample_date','age','sex']+LABS
    source=raw[cols].copy(); del raw
    source['sample_date']=pd.to_datetime(source.sample_date,errors='coerce')
    old=pd.read_csv(BASE/'formal_analysis_three_centers/deidentified_daily_cbc.csv.gz')
    old['sample_date']=pd.to_datetime(old.sample_date,format='mixed')
    audit=[]
    for center in CENTERS:
        r=source[source.center.eq(center)]; o=old[old.center.eq(center)].copy()
        o['day']=o.sample_date.dt.normalize()
        audit.append({'center':center,'raw_records':len(r),'patients':r.patient_id.nunique(),'start':r.sample_date.min(),'end':r.sample_date.max(),'old_rows':len(o),'old_nonmidnight_rows':int(o.sample_date.ne(o.day).sum()),'old_duplicate_calendar_day_rows':int(o.duplicated(['patient_id','day']).sum()),'missing_date_records':int(r.sample_date.isna().sum()),'missing_patient_records':int(r.patient_id.isna().sum())})
    source=source.dropna(subset=['patient_id','sample_date'])
    source['sample_date']=source.sample_date.dt.normalize()
    keys=['center','patient_id','sample_date']
    g=source.groupby(keys,observed=True)
    daily=g[LABS+['age']].median().reset_index()
    sex=g.sex.agg(lambda s: s.dropna().iloc[0] if s.dropna().nunique()==1 else np.nan).rename('sex').reset_index()
    daily=daily.merge(sex,on=keys,validate='one_to_one').sort_values(keys).reset_index(drop=True)
    assert not daily.duplicated(keys).any()
    assert daily.sample_date.eq(daily.sample_date.dt.normalize()).all()
    daily['group']=daily.center+'|'+daily.patient_id
    gp=daily.groupby('group',sort=False)
    daily['prior_date']=gp.sample_date.shift(1); daily['next_date']=gp.sample_date.shift(-1)
    daily['prior_gap']=(daily.sample_date-daily.prior_date).dt.days
    daily['next_gap']=(daily.next_date-daily.sample_date).dt.days
    for x in LABS:
        daily['prior_'+x]=gp[x].shift(1); daily['next_'+x]=gp[x].shift(-1)
    current=daily[LABS].notna().all(axis=1)&~crit(daily)
    future=daily[['next_'+x for x in LABS]].notna().all(axis=1)&daily.next_gap.between(1,3)
    daily['eligible']=current&future
    flow=[]
    for center in CENTERS:
        ix=daily.center.eq(center); e=daily.loc[ix&daily.eligible]
        flow.append({'center':center,'calendar_days':int(ix.sum()),'patients':daily.loc[ix,'group'].nunique(),'current_complete_noncritical':int((ix&current).sum()),'eligible_landmarks':len(e),'eligible_patients':e.group.nunique(),'event_landmarks':int(crit(e,'next_').sum()),'event_patients':e.loc[crit(e,'next_'),'group'].nunique(),'max_landmarks_per_patient':e.groupby('group').size().max(),'median_landmarks_per_patient':e.groupby('group').size().median()})
    save(pd.DataFrame(audit),'01_source_time_audit.csv'); save(pd.DataFrame(flow),'02_corrected_cohort_flow.csv')
    miss=[]
    for center in CENTERS:
        for variable in LABS+['age','sex']:
            s=daily.loc[daily.center.eq(center),variable]
            miss.append({'center':center,'variable':variable,'n':len(s),'missing':int(s.isna().sum()),'missing_rate':s.isna().mean()})
    save(pd.DataFrame(miss),'03_daily_missingness.csv')
    f=daily.loc[daily.eligible].copy().reset_index(drop=True)
    f['outcome']=crit(f,'next_').astype(int)
    s=pd.to_numeric(f.sex,errors='coerce'); f['sex2']=s.eq(2).astype(float).where(s.isin([1,2])); f['sex_missing']=~s.isin([1,2])
    f['log_WBC']=np.log1p(f.WBC); f['HGB_margin']=f.HGB-60
    f['log_PLT_margin']=np.log1p(f.PLT-20); f['log_NEUT_margin']=np.log1p(f.NEUT-.5)
    f['has_prior']=f.prior_gap.gt(0).astype(int); f['log_prior_gap']=np.log1p(f.prior_gap)
    for x in LABS:
        valid=f.prior_gap.gt(0)&f['prior_'+x].notna()
        f['trend_'+x]=((f[x]-f['prior_'+x])/f.prior_gap*(1 if x=='WBC' else -1)).where(valid)
        f['missing_prior_'+x]=(~valid).astype(int)
    for features in FEATURES.values():
        assert not any(x.startswith('next_') or x=='outcome' for x in features)
        assert not np.isinf(f[features].to_numpy(float)).any()
    # Explicit whitelist; anonymous patient IDs retained locally for reproducibility.
    keep=['center','patient_id','group','sample_date','outcome']+B
    f[keep].to_csv(OUT/'deidentified_landmarks.csv.gz',index=False,compression='gzip')
    summary=pd.DataFrame(flow)
    print('Corrected cohort:',summary.to_json(orient='records',force_ascii=False),flush=True)
    manifest={'seed':SEED,'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__,'features':FEATURES,'candidate_parameters':CONFIGS,'outer_folds':5,'inner_folds':3,'selection_metric':'patient-equal log-loss','evaluation_unit':'patient-equal calendar-day landmarks','date_convention':'calendar days; not exact 72 hours; daily summary available after day ends','external_status':'previously inspected centres; exploratory transportability assessment','raw_sha256':{p.name:digest(p) for p in [mod.RAW/x for x in mod.FILES_AND_SHEETS]},'legacy_loader_sha256':digest(BASE/'main_state_analysis.py'),'script_sha256':digest(Path(__file__))}
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return f

class ClipTrends(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.columns_=[c for c in X.columns if c.startswith('trend_')]
        self.lo_=X[self.columns_].quantile(.01); self.hi_=X[self.columns_].quantile(.99)
        return self
    def transform(self,X):
        X=X.copy(); X[self.columns_]=X[self.columns_].clip(self.lo_,self.hi_,axis=1); return X

def pipeline(kind,params):
    if kind=='ridge': m=LogisticRegression(**params,solver='lbfgs',max_iter=3000,random_state=SEED)
    elif kind=='hist_boost': m=HistGradientBoostingClassifier(**params,max_iter=120,learning_rate=.05,min_samples_leaf=30,early_stopping=False,random_state=SEED)
    else: m=RandomForestClassifier(**params,n_estimators=120,max_features=1.0,n_jobs=2,random_state=SEED)
    return Pipeline([('clip',ClipTrends()),('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler()),('model',m)])

def split(f,n,seed):
    splits=list(StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed).split(f,f.outcome,f.group))
    for a,b in splits:
        assert set(f.iloc[a].group).isdisjoint(f.iloc[b].group)
        assert f.iloc[a].outcome.nunique()==2 and f.iloc[b].outcome.nunique()==2
    return splits

def fit_model(kind,params,tr,features):
    p=pipeline(kind,params); w=weights(tr)
    p.fit(tr[features],tr.outcome,scale__sample_weight=w,model__sample_weight=w)
    return p

def tune(kind,tr,features,seed):
    ss=split(tr,3,seed); scores=[]
    for params in CONFIGS[kind]:
        pred=np.full(len(tr),np.nan)
        for a,b in ss:
            model=fit_model(kind,params,tr.iloc[a],features)
            pred[b]=model.predict_proba(tr.iloc[b][features])[:,1]
        assert np.isfinite(pred).all()
        scores.append(log_loss(tr.outcome,pred,sample_weight=weights(tr),labels=[0,1]))
    best=int(np.argmin(scores))
    return CONFIGS[kind][best],scores

def metrics(f,p,cal=True):
    y=f.outcome.to_numpy(); w=weights(f)
    r={'patients':f.group.nunique(),'landmarks':len(f),'event_patients':f.loc[f.outcome.eq(1),'group'].nunique(),'event_landmarks':int(y.sum()),'auc':roc_auc_score(y,p,sample_weight=w),'auprc':average_precision_score(y,p,sample_weight=w),'brier':brier_score_loss(y,p,sample_weight=w),'log_loss':log_loss(y,p,sample_weight=w,labels=[0,1]),'observed_rate':np.average(y,weights=w),'mean_prediction':np.average(p,weights=w)}
    if cal:
        lp=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
        r['calibration_intercept']=float(sm.GLM(y,np.ones((len(y),1)),family=sm.families.Binomial(),offset=lp,freq_weights=w).fit().params[0])
        r['calibration_slope']=float(sm.GLM(y,sm.add_constant(lp),family=sm.families.Binomial(),freq_weights=w).fit().params[1])
    return r

def uncertainty(f,predictions,dataset,reps=500):
    rng=np.random.default_rng(SEED); vals={k:{m:[] for m in ['auc','auprc','brier']} for k in predictions}; delta={k:[] for k in CONFIGS}
    center_groups={c:[g.index.to_numpy() for _,g in f.loc[f.center.eq(c)].groupby('group')] for c in f.center.unique()}
    for _ in range(reps):
        chunks=[]; wg=[]
        for groups in center_groups.values():
            for j in rng.integers(0,len(groups),len(groups)):
                chunks.append(groups[j]); wg.append(np.full(len(groups[j]),1/len(groups[j])))
        ix=np.concatenate(chunks); w=np.concatenate(wg); y=f.outcome.to_numpy()[ix]
        if len(np.unique(y))<2: continue
        aucs={}
        for name,p in predictions.items():
            pp=p[ix]; aucs[name]=roc_auc_score(y,pp,sample_weight=w)
            vals[name]['auc'].append(aucs[name]); vals[name]['auprc'].append(average_precision_score(y,pp,sample_weight=w)); vals[name]['brier'].append(brier_score_loss(y,pp,sample_weight=w))
        for k in CONFIGS: delta[k].append(aucs[k+'_B_dynamic']-aucs[k+'_A_current'])
    rows=[]
    for name,mm in vals.items():
        for m,v in mm.items(): rows.append({'dataset':dataset,'model':name,'metric':m,'low':np.quantile(v,.025),'high':np.quantile(v,.975),'valid_bootstraps':len(v)})
    for k,v in delta.items(): rows.append({'dataset':dataset,'model':k,'metric':'delta_auc_dynamic_minus_current','low':np.quantile(v,.025),'high':np.quantile(v,.975),'valid_bootstraps':len(v)})
    return rows

def run():
    f=prepare(); dev=f[f.center.eq(DEV)].reset_index(drop=True); ext=f[~f.center.eq(DEV)].reset_index(drop=True)
    outer=split(dev,5,SEED); fold=np.zeros(len(dev),int)
    for i,(a,b) in enumerate(outer): fold[b]=i+1
    save(dev[['patient_id','center']].assign(outer_fold=fold).drop_duplicates(),'04_patient_fold_assignment.csv')
    perf=[]; selections=[]; dp={}; ep={}
    for kind in CONFIGS:
        for feature_name,features in FEATURES.items():
            name=kind+'_'+feature_name; oof=np.full(len(dev),np.nan)
            for i,(a,b) in enumerate(outer):
                tr=dev.iloc[a].reset_index(drop=True); best,scores=tune(kind,tr,features,SEED+i)
                model=fit_model(kind,best,tr,features); oof[b]=model.predict_proba(dev.iloc[b][features])[:,1]
                selections.append({'model':name,'outer_fold':i+1,'parameters':json.dumps(best),'inner_losses':json.dumps(scores)})
                print(name,'outer fold',i+1,'complete',flush=True)
            best,scores=tune(kind,dev,features,SEED+100)
            model=fit_model(kind,best,dev,features); p=model.predict_proba(ext[features])[:,1]
            selections.append({'model':name,'outer_fold':'final_fit','parameters':json.dumps(best),'inner_losses':json.dumps(scores)})
            assert np.isfinite(oof).all(); dp[name]=oof; ep[name]=p
            perf.append({'model':name,'dataset':'development_nested_oof',**metrics(dev,oof)})
            perf.append({'model':name,'dataset':'external_combined',**metrics(ext,p)})
            for c in ext.center.unique():
                mask=ext.center.eq(c).to_numpy(); perf.append({'model':name,'dataset':c,**metrics(ext.loc[mask],p[mask])})
            save(pd.DataFrame(perf),'05_model_performance.csv'); save(pd.DataFrame(selections),'06_parameter_selection.csv')
    for label,frame,preds in [('development',dev,dp),('external',ext,ep)]:
        out=frame[['patient_id','center','sample_date','outcome']].copy()
        for k,p in preds.items(): out[k]=p
        out.to_csv(OUT/(label+'_predictions.csv.gz'),index=False,compression='gzip')
    ci=uncertainty(ext,ep,'external_combined')
    for c in ext.center.unique():
        ix=np.where(ext.center.eq(c))[0]
        ci+=uncertainty(ext.iloc[ix].reset_index(drop=True),{k:p[ix] for k,p in ep.items()},c)
    save(pd.DataFrame(ci),'07_external_patient_cluster_bootstrap.csv')
    # Calibration bins describe fixed external predictions; no recalibration.
    rows=[]
    for name,p in ep.items():
        temp=pd.DataFrame({'p':p,'y':ext.outcome,'w':weights(ext)})
        temp['bin']=pd.qcut(temp.p,5,duplicates='drop')
        for _,g in temp.groupby('bin',observed=True): rows.append({'model':name,'mean_prediction':np.average(g.p,weights=g.w),'observed_rate':np.average(g.y,weights=g.w),'landmarks':len(g)})
    cal=pd.DataFrame(rows); save(cal,'08_external_calibration_bins.csv')
    fig,ax=plt.subplots(figsize=(7,6)); ax.plot([0,1],[0,1],'k--',lw=1)
    for name,g in cal.groupby('model'): ax.plot(g.mean_prediction,g.observed_rate,'o-',label=name)
    ax.set(xlabel='Mean predicted probability',ylabel='Observed proportion',title='External calibration (patient-equal weights)',xlim=(0,.8),ylim=(0,.8)); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT/'external_calibration.png',dpi=200); plt.close(fig)
    print('COMPLETE',flush=True)

if __name__=='__main__': run()
