from pathlib import Path
import sys,joblib
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
R=Path(__file__).resolve().parent; ML=R.parent/'ml'/'round2';sys.path.insert(0,str(ML.parent));import run_analysis as base
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
navy='#205781';orange='#c55b28';green='#2f826d';grey='#6c7884'
def tag(ax,s):ax.text(-.09,1.06,s,transform=ax.transAxes,fontweight='bold',fontsize=12,va='top')
def save(fig,name):fig.savefig(R/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(R/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
# Group Figure 2: four complementary state-transition views.
m=pd.read_csv(R/'02_state_transition_matrix.csv',encoding='utf-8-sig');order=['C0','C1','C2+'];cent=['安医大','淮北煤炭','铜陵'];cl={'安医大':'FAH-AHMU','淮北煤炭':'WCEGH','铜陵':'TPH'}
fig,axs=plt.subplots(2,2,figsize=(10.3,8.4));ax=axs[0,0];x=m[(m.center=='Overall')&(m.interval=='1-3 days')];v=x.pivot(index='base',columns='next',values='pair_rate').loc[order,order].to_numpy();n=x.pivot(index='base',columns='next',values='events').loc[order,order].to_numpy();ax.imshow(v,cmap='Blues',vmin=0,vmax=.9)
for i in range(3):
 for j in range(3):ax.text(j,i,f'{v[i,j]*100:.1f}%\n(n={n[i,j]:,})',ha='center',va='center',color='white' if v[i,j]>.55 else '#17334b',fontsize=9)
ax.set_xticks(range(3),order);ax.set_yticks(range(3),order);ax.set_xlabel('Next observed state');ax.set_ylabel('Current state');ax.set_title('Overall transition matrix');tag(ax,'A')
ax=axs[0,1];z=m[(m.center.isin(cent))&(m.interval=='1-3 days')&(m.base=='C0')];xx=np.arange(3);bottom=np.zeros(3)
for dest,color in [('C0','#a8c6d8'),('C1','#3c7ea6'),('C2+',orange)]:
 yy=np.array([z[(z.center==c)&(z.next==dest)].pair_rate.iloc[0]*100 for c in cent]);ax.bar(xx,yy,bottom=bottom,color=color,label=dest,width=.60,edgecolor='white',linewidth=1.2);bottom+=yy
ax.set_xticks(xx,[cl[c] for c in cent]);ax.set_ylim(0,124);ax.set_ylabel('Share of C0-origin pairs (%)');ax.set_title('C0 next state by centre');ax.legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,.99),fontsize=8,handlelength=1.2,columnspacing=.8);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
for i,c in enumerate(cent):ax.text(i,102,f"n={int(z[z.center==c].pairs.iloc[0]):,}",ha='center',fontsize=8,color='#41586a')
tag(ax,'B')
ax=axs[1,0];z=m[(m.center=='Overall')&(m.base=='C0')&(m.next!='C0')&(m.interval.isin(['1 day','2-3 days']))];rates=[];labels=[]
for interval in ['1 day','2-3 days']:
 q=z[z.interval==interval]; rates.append(q.pair_rate.sum()*100);labels.append(interval)
ax.bar(labels,rates,color=[navy,orange],width=.52,edgecolor='white',linewidth=1.2);ax.set_ylim(0,24);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.set_ylabel('C0 to any critical state (%)');ax.set_title('Observed retest interval');
for i,val in enumerate(rates):ax.text(i,val+.5,f'{val:.1f}%',ha='center');tag(ax,'C')
ax=axs[1,1];z=m[(m.center.isin(cent))&(m.interval=='1-3 days')&(m.base=='C2+')&(m.next=='C2+')];vals=[z[z.center==c].pair_rate.iloc[0]*100 for c in cent];ax.bar([cl[c] for c in cent],vals,color=[green,'#50a18f','#76b7a5'],width=.56,edgecolor='white',linewidth=1.2);ax.set_ylim(0,85);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.set_ylabel('C2+ persistence at next test (%)');ax.set_title('Multi-analyte critical state persistence');
for i,val in enumerate(vals):ax.text(i,val+.8,f'{val:.1f}%',ha='center');tag(ax,'D')
fig.text(.08,.015,'All estimates describe complete observed adjacent test pairs; interval and centre comparisons are not causal.',fontsize=8);fig.tight_layout(rect=[0,.04,1,1],h_pad=2.1,w_pad=2);save(fig,'figure_2_state_group')
# Group Figure 3: estimands and descriptive gradients.
s=pd.read_csv(R/'01_estimand_and_threshold_sensitivity.csv',encoding='utf-8-sig');b=pd.read_csv(R/'05_patient_cluster_bootstrap.csv',encoding='utf-8-sig');near=pd.read_csv(R/'04_near_critical_gradient.csv',encoding='utf-8-sig');comp=pd.read_csv(R/'03_first_transition_components.csv',encoding='utf-8-sig')
fig,axs=plt.subplots(2,2,figsize=(10.3,8.4));ax=axs[0,0];xx=np.arange(3)
for est,off,color,label in [('first_patient',-.12,navy,'First per linked key'),('all_pairs',.12,orange,'All pairs')]:
 yy=[];lo=[];hi=[]
 for c in cent:
  r=s[(s.scenario=='primary')&(s.center==c)&(s.estimand==est)].iloc[0];q=b[(b.center==c)&(b.estimand==est)].iloc[0];yy.append(100*r.rate);lo.append(100*q.low);hi.append(100*q.high)
 yy=np.array(yy);ax.errorbar(xx+off,yy,yerr=[yy-np.array(lo),np.array(hi)-yy],fmt='o',capsize=3,color=color,label=label)
ax.set_xticks(xx,[cl[c] for c in cent]);ax.set_ylim(0,29);ax.set_ylabel('C0-to-critical conversion (%)');ax.set_title('Different units of analysis');ax.legend(frameon=False,fontsize=8);tag(ax,'A')
ax=axs[0,1];near['order']=near.near_critical_count.map({'0':0,'1':1,'2+':2});near=near.sort_values('order');y=100*near.rate.to_numpy();lo=100*near.wilson_low.to_numpy();hi=100*near.wilson_high.to_numpy();ax.bar(['0','1','2+'],y,color=['#b5cfe1','#6d9dbd',navy],width=.62,edgecolor='white',linewidth=1.2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.errorbar(np.arange(3),y,yerr=[y-lo,hi-y],fmt='none',color='#263d4e',capsize=4);ax.set_ylim(0,43);ax.set_xlabel('Near-critical analytes');ax.set_ylabel('First C0-pair conversion (%)');ax.set_title('Exploratory near-critical gradient');tag(ax,'B')
ax=axs[1,0];q=s[(s.center=='Overall')&(s.estimand=='first_patient')].copy();order2=['primary','WBC_50','HGB_70','HGB_80','PLT_10','PLT_30','NEUT_1','broad'];q=q.set_index('scenario').loc[order2];yy=np.arange(len(q));rr=100*q.rate.to_numpy();lo=100*q.wilson_low_first_only.to_numpy();hi=100*q.wilson_high_first_only.to_numpy();ax.errorbar(rr,yy,xerr=[rr-lo,hi-rr],fmt='o',color=navy,capsize=2);ax.set_yticks(yy,['Primary','WBC 50','HGB 70','HGB 80','PLT 10','PLT 30','NEUT 1','Broad']);ax.invert_yaxis();ax.set_xlim(9,29);ax.set_xlabel('First C0-pair conversion (%)');ax.set_title('Alternative thresholds');tag(ax,'C')
ax=axs[1,1];q=pd.read_csv(R/'13_stable_subtype_c0_transition.csv').set_index('subtype').loc[['AML','ALL','CML','CLL']];vals=100*q.c0_pair_rate.to_numpy();lo=100*q.c0_pair_cluster_low.to_numpy();hi=100*q.c0_pair_cluster_high.to_numpy();ax.bar(['AML','ALL','CML','CLL'],vals,color=[navy,'#5c9bb7',green,orange],width=.62,edgecolor='white',linewidth=1.2);ax.errorbar(np.arange(4),vals,yerr=[vals-lo,hi-vals],fmt='none',color='#263d4e',capsize=3);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.set_ylim(0,25);ax.set_ylabel('C0-to-critical pairs (%)');ax.set_title('Stable leukemia subtype');
for i,row in enumerate(q.itertuples()):ax.text(i,hi[i]+.7,f'{int(row.c0_pair_events)}/{int(row.c0_pairs)}',ha='center',fontsize=7.5);tag(ax,'D')
fig.text(.08,.015,'Patient identity was checked in the hospital history system; subtype panel excludes records with changing subtype labels.',fontsize=8);fig.tight_layout(rect=[0,.04,1,1],h_pad=2.2,w_pad=2);save(fig,'figure_3_estimands_group')
# Group Figure 4: exploratory model behaviour and calibration.
f=pd.read_csv(ML/'deidentified_landmarks.csv.gz',float_precision='round_trip');e=f[f.center!=base.DEV].reset_index(drop=True);pred=pd.read_csv(ML/'external_extended_predictions.csv.gz');assert len(e)==len(pred) and (e[['center','patient_id','sample_date']].astype(str).values==pred[['center','patient_id','sample_date']].astype(str).values).all()
y=e.outcome.to_numpy();w=1/e.groupby('group').group.transform('size').to_numpy(float);p=pred.select_all.to_numpy();p2=pred.spline_B_dynamic.to_numpy();perf=pd.read_csv(ML/'14_external_intervals_and_comparisons.csv');bins=pd.read_csv(ML/'18_calibration_bins.csv')
fig,axs=plt.subplots(2,2,figsize=(10.3,8.4));ax=axs[0,0]
for name,pp,col in [('Selected dynamic model',p,navy),('Dynamic spline regression',p2,orange)]:
 a,bb,_=roc_curve(y,pp,sample_weight=w);ax.plot(a,bb,label=name,color=col,lw=2)
ax.plot([0,1],[0,1],'--',color=grey,lw=.8);ax.set_xlabel('False-positive rate');ax.set_ylabel('Sensitivity');ax.set_title('External ROC');ax.legend(frameon=False,fontsize=8);tag(ax,'A')
ax=axs[0,1]
for name,pp,col in [('Selected dynamic model',p,navy),('Dynamic spline regression',p2,orange)]:
 precision,recall,_=precision_recall_curve(y,pp,sample_weight=w);ax.plot(recall,precision,label=name,color=col,lw=2)
ax.axhline(np.average(y,weights=w),ls='--',color=grey,lw=.8);ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.set_title('External precision–recall');ax.legend(frameon=False,fontsize=8);tag(ax,'B')
ax=axs[1,0];q=bins[bins.model=='select_all'];ax.plot(q.predicted,q.observed,'o-',color=navy);ax.plot([0,1],[0,1],'--',color=grey,lw=.8);ax.set_xlim(0,.65);ax.set_ylim(0,.65);ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed patient-weighted frequency');ax.set_title('Calibration (5 bins)');tag(ax,'C')
ax=axs[1,1];names=['Dynamic boost','Dynamic spline','Dynamic forest','Current boost'];keys=['hist_boost_B_dynamic','spline_B_dynamic','random_forest_B_dynamic','select_current'];rr=[];lo=[];hi=[]
for k in keys:
 q=perf[(perf.model==k)&(perf.reference.isna())&(perf.metric=='auc')].iloc[0];rr.append(q.estimate);lo.append(q.low);hi.append(q.high)
idx=np.arange(4);ax.errorbar(rr,idx,xerr=[np.array(rr)-np.array(lo),np.array(hi)-np.array(rr)],fmt='o',color=navy,capsize=3);ax.set_yticks(idx,names);ax.invert_yaxis();ax.set_xlim(.73,.93);ax.set_xlabel('Patient-equal external AUC');ax.set_title('Model comparison');tag(ax,'D')
fig.text(.08,.015,'External centres were previously inspected; model findings are exploratory. Discrimination does not establish calibration or clinical utility.',fontsize=8);fig.tight_layout(rect=[0,.04,1,1],h_pad=2.2,w_pad=2);save(fig,'figure_4_model_group')
# Group Figure 5: model dependence and accumulated local effects (ALE) on external data.
model=joblib.load(ML/'development_selected_model.joblib');features=base.B;X=e[features].copy();np.testing.assert_allclose(model.predict_proba(X)[:,1],p,atol=1e-10,rtol=1e-9)
def ale(feature,transform=lambda x:x,bins_count=12):
 x=X[feature].to_numpy(float);good=np.isfinite(x);edges=np.unique(np.quantile(x[good],np.linspace(.02,.98,bins_count+1)))
 delta=[];mass=[];cent=[]
 for low,high in zip(edges[:-1],edges[1:]):
  mask=good&(x>=low)&(x<high if high!=edges[-1] else x<=high)
  if not mask.any():continue
  a=X.loc[mask].copy();z=a.copy();a[feature]=low;z[feature]=high
  diff=model.predict_proba(z)[:,1]-model.predict_proba(a)[:,1];ww=w[mask];delta.append(np.average(diff,weights=ww));mass.append(ww.sum());cent.append((low+high)/2)
 curve=np.cumsum(delta);curve=curve-np.average(curve,weights=mass)
 return transform(np.array(cent)),100*curve,np.array(mass)
fig,axs=plt.subplots(2,2,figsize=(10.3,8.4));ax=axs[0,0];imp=pd.read_csv(ML/'17_block_permutation_summary.csv');imp=imp.sort_values('auc_drop_mean');yy=np.arange(len(imp));ax.barh(yy,imp.auc_drop_mean,color=navy);ax.errorbar(imp.auc_drop_mean,yy,xerr=imp.auc_drop_sd,fmt='none',color='#263d4e',capsize=2);ax.set_yticks(yy,imp.block);ax.axvline(0,color=grey,lw=.8);ax.set_xlabel('AUC decrease after block permutation');ax.set_title('External model dependence');tag(ax,'A')
for ax,feature,title,transform,xlabel,letter in [(axs[0,1],'log_PLT_margin','Platelet count',lambda z:np.expm1(z)+20,'PLT (×10⁹/L)','B'),(axs[1,0],'HGB_margin','Haemoglobin',lambda z:z+60,'HGB (g/L)','C'),(axs[1,1],'log_NEUT_margin','Neutrophil count',lambda z:np.expm1(z)+.5,'NEUT (×10⁹/L)','D')]:
 xx,yy,mass=ale(feature,transform);ax.plot(xx,yy,'o-',color=orange if letter=='B' else navy,lw=1.6,markersize=4);ax.axhline(0,color=grey,lw=.7,ls='--');ax.set_xlabel(xlabel);ax.set_ylabel('ALE of predicted risk (percentage points)');ax.set_title(title+' local model effect');tag(ax,letter)
fig.text(.08,.015,'Permutation bars: mean ± SD of 20 within-centre shuffles, not confidence intervals. ALE is centred; neither plot implies causation.',fontsize=8);fig.tight_layout(rect=[0,.04,1,1],h_pad=2.2,w_pad=2);save(fig,'figure_5_explainability_group')
print('four 4-panel group figures generated; model predictions verified')

