from pathlib import Path
import sys,json,hashlib,importlib,platform
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent
sys.path.insert(0,str(PREV))
import run_analysis as base
import numpy as np
import pandas as pd
import sklearn,joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss,roc_curve
import matplotlib.pyplot as plt

SEED=base.SEED
A=base.A; B=base.B
CONFIGS=[{'C':c} for c in [.03,.3,3]]

def save(d,name): d.to_csv(ROOT/name,index=False,encoding='utf-8-sig')
def hashfile(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def spline_pipeline(features,params):
    continuous=[i for i,c in enumerate(features) if c in ['age','log_WBC','HGB_margin','log_PLT_margin','log_NEUT_margin','log_prior_gap'] or c.startswith('trend_')]
    # Only continuous predictors get quadratic B-splines; binary flags remain binary.
    transform=ColumnTransformer([('splines',SplineTransformer(n_knots=3,degree=2,knots='uniform',include_bias=False,extrapolation='linear'),continuous)],remainder='passthrough',verbose_feature_names_out=False)
    return Pipeline([('clip',base.ClipTrends()),('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('nonlinear',transform),('scale',StandardScaler()),('model',LogisticRegression(**params,solver='lbfgs',max_iter=5000,random_state=SEED))])

def fit_spline(tr,features,params):
    p=spline_pipeline(features,params); w=base.weights(tr)
    p.fit(tr[features],tr.outcome,scale__sample_weight=w,model__sample_weight=w)
    return p

def tune_spline(tr,features,seed):
    scores=[]
    for params in CONFIGS:
        oof=np.full(len(tr),np.nan)
        for a,b in base.split(tr,3,seed):
            m=fit_spline(tr.iloc[a],features,params)
            oof[b]=m.predict_proba(tr.iloc[b][features])[:,1]
        assert np.isfinite(oof).all()
        scores.append(log_loss(tr.outcome,oof,sample_weight=base.weights(tr),labels=[0,1]))
    return CONFIGS[int(np.argmin(scores))],scores

def stat(f,p,w=None):
    y=f.outcome.to_numpy(); w=base.weights(f) if w is None else w
    return {'patients':f.group.nunique(),'landmarks':len(f),'event_patients':f.loc[f.outcome.eq(1),'group'].nunique(),'event_landmarks':int(y.sum()),'auc':roc_auc_score(y,p,sample_weight=w) if len(np.unique(y))>1 else np.nan,'auprc':average_precision_score(y,p,sample_weight=w) if y.sum()>0 else np.nan,'brier':brier_score_loss(y,p,sample_weight=w),'log_loss':log_loss(y,p,sample_weight=w,labels=[0,1]),'observed_rate':np.average(y,weights=w),'mean_prediction':np.average(p,weights=w)}

def boot(f,predictions,reps=1000,pairs=()):
    f=f.reset_index(drop=True); rng=np.random.default_rng(SEED+2)
    groups={c:[g.index.to_numpy() for _,g in f[f.center.eq(c)].groupby('group')] for c in f.center.unique()}
    vals={(k,m):[] for k in predictions for m in ['auc','auprc','brier']}
    dv={(a,b,m):[] for a,b in pairs for m in ['auc','brier']}
    for _ in range(reps):
        chunks=[]
        for gg in groups.values(): chunks.extend(gg[j] for j in rng.integers(0,len(gg),len(gg)))
        ix=np.concatenate(chunks); w=np.concatenate([np.full(len(g),1/len(g)) for g in chunks]); y=f.outcome.to_numpy()[ix]
        if len(np.unique(y))<2: continue
        mm={}
        for k,p in predictions.items():
            pp=np.asarray(p)[ix]; mm[k]={'auc':roc_auc_score(y,pp,sample_weight=w),'auprc':average_precision_score(y,pp,sample_weight=w),'brier':brier_score_loss(y,pp,sample_weight=w)}
            for m,v in mm[k].items(): vals[k,m].append(v)
        for a,b in pairs:
            for m in ['auc','brier']: dv[a,b,m].append(mm[a][m]-mm[b][m])
    rows=[]
    for (k,m),v in vals.items():
        rows.append({'model':k,'reference':'','metric':m,'estimate':stat(f,predictions[k])[m],'low':np.quantile(v,.025),'high':np.quantile(v,.975),'valid_replicates':len(v)})
    for (a,b,m),v in dv.items():
        rows.append({'model':a,'reference':b,'metric':'delta_'+m,'estimate':stat(f,predictions[a])[m]-stat(f,predictions[b])[m],'low':np.quantile(v,.025),'high':np.quantile(v,.975),'valid_replicates':len(v)})
    return pd.DataFrame(rows)

def get_features(name): return A if name.endswith('A_current') else B

def final_fit(name,params,dev):
    features=get_features(name)
    if name.startswith('spline_'): return fit_spline(dev,features,params)
    kind=name.rsplit('_',2)[0]
    return base.fit_model(kind,params,dev,features)

def main():
    # Preparation writes to round2 only; original study and round1 files stay unchanged.
    base.OUT=ROOT
    f=base.prepare()
    old=pd.read_csv(PREV/'deidentified_landmarks.csv.gz',parse_dates=['sample_date'])
    pd.testing.assert_frame_equal(f[old.columns].reset_index(drop=True),old,check_dtype=False)
    dev=f[f.center.eq(base.DEV)].reset_index(drop=True); ext=f[~f.center.eq(base.DEV)].reset_index(drop=True)
    old_dp=pd.read_csv(PREV/'development_predictions.csv.gz',parse_dates=['sample_date'])
    old_ep=pd.read_csv(PREV/'external_predictions.csv.gz',parse_dates=['sample_date'])
    keys=['center','patient_id','sample_date','outcome']
    pd.testing.assert_frame_equal(dev[keys],old_dp[keys],check_dtype=False)
    pd.testing.assert_frame_equal(ext[keys],old_ep[keys],check_dtype=False)
    dp={c:old_dp[c].to_numpy() for c in old_dp if c not in keys}
    ep={c:old_ep[c].to_numpy() for c in old_ep if c not in keys}
    selections=pd.read_csv(PREV/'06_parameter_selection.csv').to_dict('records')
    outer=base.split(dev,5,SEED); fold=np.zeros(len(dev),int)
    for i,(a,b) in enumerate(outer): fold[b]=i+1
    for suffix,features in [('A_current',A),('B_dynamic',B)]:
        name='spline_'+suffix; oof=np.full(len(dev),np.nan)
        for i,(a,b) in enumerate(outer):
            tr=dev.iloc[a].reset_index(drop=True); params,scores=tune_spline(tr,features,SEED+i)
            m=fit_spline(tr,features,params); oof[b]=m.predict_proba(dev.iloc[b][features])[:,1]
            selections.append({'model':name,'outer_fold':str(i+1),'parameters':json.dumps(params),'inner_losses':json.dumps(scores)})
            print(name,'outer',i+1,'complete',flush=True)
        params,scores=tune_spline(dev,features,SEED+100)
        m=fit_spline(dev,features,params); dp[name]=oof; ep[name]=m.predict_proba(ext[features])[:,1]
        selections.append({'model':name,'outer_fold':'final_fit','parameters':json.dumps(params),'inner_losses':json.dumps(scores)})
    selection=pd.DataFrame(selections); selection['outer_fold']=selection.outer_fold.astype(str)
    selection['best_inner_loss']=selection.inner_losses.map(lambda x:min(json.loads(x)))
    save(selection,'11_all_inner_selection_scores.csv')
    choices=[]; final_choices={}
    # Choice depends ONLY on inner developmental losses, never outer or external metrics.
    for procedure,suffix in [('select_current','A_current'),('select_dynamic','B_dynamic'),('select_all',None)]:
        oof=np.full(len(dev),np.nan)
        pool=selection if suffix is None else selection[selection.model.str.endswith(suffix)]
        for i,(a,b) in enumerate(outer):
            r=pool[pool.outer_fold.eq(str(i+1))].sort_values(['best_inner_loss','model']).iloc[0]
            oof[b]=dp[r.model][b]; choices.append({'procedure':procedure,'outer_fold':str(i+1),'selected_model':r.model,'parameters':r.parameters,'inner_log_loss':r.best_inner_loss})
        r=pool[pool.outer_fold.eq('final_fit')].sort_values(['best_inner_loss','model']).iloc[0]
        dp[procedure]=oof; ep[procedure]=ep[r.model].copy()
        choices.append({'procedure':procedure,'outer_fold':'final_fit','selected_model':r.model,'parameters':r.parameters,'inner_log_loss':r.best_inner_loss})
        final_choices[procedure]={'model':r.model,'parameters':json.loads(r.parameters)}
    save(pd.DataFrame(choices),'12_nested_algorithm_choices.csv')
    (ROOT/'selected_model_spec.json').write_text(json.dumps(final_choices,indent=2),encoding='utf-8')
    print('Development-only selection:',json.dumps(final_choices),flush=True)
    rows=[]
    for name in dp:
        rows.append({'model':name,'dataset':'development_nested_oof',**base.metrics(dev,dp[name])})
        rows.append({'model':name,'dataset':'external_combined',**base.metrics(ext,ep[name])})
        for c in ext.center.unique():
            ix=np.where(ext.center.eq(c))[0]; rows.append({'model':name,'dataset':c,**base.metrics(ext.iloc[ix],ep[name][ix])})
    save(pd.DataFrame(rows),'13_extended_performance.csv')
    for label,frame,preds in [('development',dev,dp),('external',ext,ep)]:
        p=frame[keys].copy()
        for k,v in preds.items(): p[k]=v
        p.to_csv(ROOT/(label+'_extended_predictions.csv.gz'),index=False,compression='gzip')
    # Save and verify the development-selected final fitted model.
    chosen=final_choices['select_all']; model=final_fit(chosen['model'],chosen['parameters'],dev); features=get_features(chosen['model'])
    np.testing.assert_allclose(model.predict_proba(ext[features])[:,1],ep['select_all'],rtol=1e-9,atol=1e-10)
    joblib.dump(model,ROOT/'development_selected_model.joblib')
    loaded=joblib.load(ROOT/'development_selected_model.joblib')
    np.testing.assert_allclose(loaded.predict_proba(ext[features])[:,1],ep['select_all'])
    comparisons=[('hist_boost_B_dynamic','spline_B_dynamic'),('random_forest_B_dynamic','spline_B_dynamic'),('spline_B_dynamic','spline_A_current'),('select_dynamic','select_current')]
    selected_names=['spline_A_current','spline_B_dynamic','hist_boost_B_dynamic','random_forest_B_dynamic','select_all','select_current','select_dynamic']
    intervals=boot(ext,{k:ep[k] for k in selected_names},pairs=comparisons)
    save(intervals,'14_external_intervals_and_comparisons.csv'); print('Paired bootstrap complete',flush=True)
    # Evaluation subsets reuse FIXED predictions; these are not retrained subgroup models.
    ordered=ext.sort_values('sample_date'); first=ordered.drop_duplicates('group').index.to_numpy(); first3=ordered.groupby('group',sort=False).head(3).index.to_numpy()
    subgroups={'all':np.arange(len(ext)),'first_per_patient':first,'first_three_per_patient':first3,'age_under18':np.where(ext.age.lt(18))[0],'age_18plus':np.where(ext.age.ge(18))[0],'with_prior':np.where(ext.has_prior.eq(1))[0],'without_prior':np.where(ext.has_prior.eq(0))[0],'prior_gap_up_to7':np.where(ext.prior_gap.between(1,7))[0],'interval_1day':np.where(ext.next_gap.eq(1))[0],'interval_2to3days':np.where(ext.next_gap.between(2,3))[0],'sex_code1':np.where(ext.sex2.eq(0))[0],'sex_code2':np.where(ext.sex2.eq(1))[0]}
    for c in ext.center.unique(): subgroups[c]=np.where(ext.center.eq(c))[0]
    sens=[]
    for label,ix in subgroups.items():
        if len(ix)==0: continue
        ss=ext.iloc[ix].reset_index(drop=True)
        for name in ['select_all','select_current','spline_B_dynamic','hist_boost_B_dynamic']:
            rr={'subgroup':label,'model':name,**stat(ss,ep[name][ix])}
            if ss.outcome.nunique()==2 and ss.loc[ss.outcome.eq(1),'group'].nunique()>=5:
                ints=boot(ss,{name:ep[name][ix]},reps=500)
                for metric in ['auc','brier']:
                    z=ints[ints.metric.eq(metric)].iloc[0]; rr[metric+'_low']=z.low; rr[metric+'_high']=z.high
            sens.append(rr)
        print('Sensitivity',label,'complete',flush=True)
    for name in ['select_all','select_current','spline_B_dynamic','hist_boost_B_dynamic']:
        sens.append({'subgroup':'all_landmarks_equal_weight','model':name,**stat(ext,ep[name],w=np.ones(len(ext)))})
    save(pd.DataFrame(sens),'15_fixed_model_sensitivity.csv')
    # Repeat a patient-size-restricted TRAINING analysis using development-selected type and hyperparameters.
    tr=dev.sort_values('sample_date').drop_duplicates('group').reset_index(drop=True)
    once=final_fit(chosen['model'],chosen['parameters'],tr)
    ix=first; pred=once.predict_proba(ext.iloc[ix][features])[:,1]
    rr={'analysis':'train_and_evaluate_first_only_fixed_hyperparameters',**stat(ext.iloc[ix].reset_index(drop=True),pred)}
    ints=boot(ext.iloc[ix].reset_index(drop=True),{'first_only':pred},reps=1000)
    save(pd.DataFrame([rr]),'16_first_only_refit.csv'); save(ints,'16_first_only_refit_intervals.csv')
    # Block permutation: shuffle related predictors together within centre; descriptive perturbation only.
    blocks={'Current WBC':['log_WBC'],'Current HGB':['HGB_margin'],'Current PLT':['log_PLT_margin'],'Current NEUT':['log_NEUT_margin'],'Age':['age'],'Sex':['sex2','sex_missing'],'History block':[c for c in B if c not in A]}
    rng=np.random.default_rng(SEED+3); original=stat(ext,ep['select_all']); importances=[]
    for label,cols in blocks.items():
        cols=[c for c in cols if c in features]
        if not cols: continue
        for rep in range(20):
            xx=ext[features].copy()
            for c in ext.center.unique():
                ix=np.where(ext.center.eq(c))[0]; order=rng.permutation(ix)
                xx.loc[ix,cols]=ext.loc[order,cols].to_numpy()
            pp=model.predict_proba(xx)[:,1]; st=stat(ext,pp)
            importances.append({'block':label,'repeat':rep+1,'auc_drop':original['auc']-st['auc'],'brier_increase':st['brier']-original['brier']})
        print('Permutation block',label,'complete',flush=True)
    imp=pd.DataFrame(importances); save(imp,'17_block_permutation_repeats.csv')
    summ=imp.groupby('block').agg(auc_drop_mean=('auc_drop','mean'),auc_drop_sd=('auc_drop','std'),brier_increase_mean=('brier_increase','mean')).reset_index().sort_values('auc_drop_mean')
    save(summ,'17_block_permutation_summary.csv')
    fig,ax=plt.subplots(figsize=(7,4)); ax.barh(summ.block,summ.auc_drop_mean,xerr=summ.auc_drop_sd); ax.axvline(0,color='black',lw=.7); ax.set_xlabel('AUC decrease after within-centre block permutation'); ax.set_title('Development-selected model: descriptive importance'); fig.tight_layout(); fig.savefig(ROOT/'Figure_3_block_importance.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    for name in ['select_all','spline_B_dynamic','hist_boost_B_dynamic']:
        x,y,_=roc_curve(ext.outcome,ep[name],sample_weight=base.weights(ext)); ax.plot(x,y,label=f'{name}: {stat(ext,ep[name])["auc"]:.3f}')
    ax.plot([0,1],[0,1],'k--',lw=1); ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Exploratory cross-centre ROC'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT/'Figure_1_external_ROC.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); ax.plot([0,1],[0,1],'k--',lw=1); cal=[]
    for name in ['select_all','spline_B_dynamic','hist_boost_B_dynamic']:
        tmp=pd.DataFrame({'p':ep[name],'y':ext.outcome,'w':base.weights(ext)}); tmp['bin']=pd.qcut(tmp.p,5,duplicates='drop')
        points=[]
        for _,g in tmp.groupby('bin',observed=True): points.append((np.average(g.p,weights=g.w),np.average(g.y,weights=g.w)))
        for x,y in points: cal.append({'model':name,'predicted':x,'observed':y})
        ax.plot(*np.array(points).T,'o-',label=name)
    ax.set(xlabel='Predicted probability',ylabel='Observed proportion',title='External calibration (patient equal)',xlim=(0,1),ylim=(0,1)); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT/'Figure_2_external_calibration.png',dpi=200); plt.close(fig); save(pd.DataFrame(cal),'18_calibration_bins.csv')
    # Store subgroup metadata only, no future lab values in model inputs.
    save(ext[['center','patient_id','sample_date','next_gap','has_prior','prior_gap']], 'external_subgroup_metadata.csv')
    manifest={'stage':'round2 exploratory; designed after round1 external results seen','seed':SEED,'python':platform.python_version(),'sklearn':sklearn.__version__,'spline':{'degree':2,'n_knots':3,'knots':'uniform','extrapolation':'linear','C':[.03,.3,3]},'selection':'lowest inner-development patient-equal log loss across algorithms and feature sets; outer predictions not used for selection','final_choices':final_choices,'bootstrap':'within-centre patient cluster; fixed fitted predictions; no training uncertainty','input_sha256':{str(p.relative_to(PREV)):hashfile(p) for p in [PREV/'run_analysis.py',PREV/'06_parameter_selection.csv',PREV/'deidentified_landmarks.csv.gz',PREV/'development_predictions.csv.gz',PREV/'external_predictions.csv.gz']},'script_sha256':hashfile(Path(__file__)),'checks':['raw reconstructed landmarks identical to round1','prediction row identity verified','patient isolated folds','frozen final model reproduces stored external predictions','serialized model reproduces predictions']}
    (ROOT/'round2_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ROUND2 COMPLETE',flush=True)

if __name__=='__main__': main()
