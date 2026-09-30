from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
s=pd.read_csv(R/'01_estimand_and_threshold_sensitivity.csv',encoding='utf-8-sig');s=s[(s.center=='Overall')&(s.estimand=='first_patient')].copy()
order=['primary','WBC_50','HGB_70','HGB_80','PLT_10','PLT_30','NEUT_1','broad'];s=s.set_index('scenario').loc[order].reset_index()
labels=['Primary','WBC ≥50','HGB <70','HGB <80','PLT <10','PLT <30','NEUT <1','Broad set']
fig,ax=plt.subplots(figsize=(7.3,5.1));y=np.arange(len(s));v=s.rate.to_numpy()*100;lo=s.wilson_low_first_only.to_numpy()*100;hi=s.wilson_high_first_only.to_numpy()*100
ax.errorbar(v,y,xerr=[v-lo,hi-v],fmt='o',color='#23608b',capsize=3,lw=1.3);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(10,28);ax.set_xlabel('First eligible C0-pair conversion (%)');ax.grid(axis='x',alpha=.2)
for yy,row in zip(y,s.itertuples()):ax.text(hi[yy]+.3,yy,f'{int(row.events)}/{int(row.denominator)}',va='center',fontsize=8)
fig.text(.08,.01,'Cohort eligibility changes with each threshold; estimates are descriptive, not paired effects.',fontsize=8);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(R/'supp_figure_1_threshold_sensitivity.png',dpi=300,bbox_inches='tight');fig.savefig(R/'supp_figure_1_threshold_sensitivity.pdf',bbox_inches='tight');plt.close(fig)

l=pd.read_csv(R/'09_linkage_sensitivity.csv',encoding='utf-8-sig');l=l[l.center=='铜陵'];fig,ax=plt.subplots(figsize=(7.3,4.2));labels=['Name + sex linked key','Visit/hospitalization number'];x=np.arange(2)
for i,(est,marker,color) in enumerate([('pair_rate','o','#23608b'),('first_rate','s','#c35c22')]):
 z=l.set_index('linkage_method').loc[['name_sex','source_id_sex']];rate=z[est].to_numpy()*100
 ax.plot(x,rate,marker=marker,color=color,lw=1.5,markersize=7,label='All eligible C0 pairs' if i==0 else 'First eligible C0 pair per key')
 for xx,r,row in zip(x,rate,z.itertuples()):
  numerator=row.pair_events if i==0 else row.first_events;denom=row.c0_pairs if i==0 else row.eligible_keys
  ax.text(xx+(.02 if i==0 else -.02),r+(.9 if i==0 else -.9),f'{numerator}/{denom}',ha='center',va='bottom' if i==0 else 'top',fontsize=9,color=color)
ax.set_xticks(x,labels);ax.set_xlim(-.3,1.3);ax.set_ylim(5,23);ax.set_ylabel('Observed C0-to-critical conversion (%)');ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,loc='upper center',fontsize=9)
fig.text(.08,.01,'Visit/hospitalization numbers are not stable patient IDs; the first-pair denominators are not comparable patient counts.',fontsize=8)
fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(R/'supp_figure_2_linkage_sensitivity.png',dpi=300,bbox_inches='tight');fig.savefig(R/'supp_figure_2_linkage_sensitivity.pdf',bbox_inches='tight');plt.close(fig)
print('2 supplementary figures generated')
