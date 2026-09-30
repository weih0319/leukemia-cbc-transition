from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parent
flow=pd.read_csv(R/'07_corrected_cohort_flow.csv',encoding='utf-8-sig')
m=pd.read_csv(R/'02_state_transition_matrix.csv',encoding='utf-8-sig')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.3,8.35));fig.patch.set_facecolor('#f5f8fb');ax.set_facecolor('#f5f8fb');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.text(.045,.965,'Cohort construction',ha='left',va='top',fontsize=18,fontweight='bold',color='#163b55')
ax.text(.045,.925,'Three centres  •  observed complete retests  •  1–3 calendar days',ha='left',va='top',fontsize=9.2,color='#637b8d')
# Compact all-centre context above the separate patient-flow cards.
summary=FancyBboxPatch((.045,.855),.91,.044,boxstyle='round,pad=.006,rounding_size=.012',facecolor='#e4edf4',edgecolor='none');ax.add_patch(summary)
ax.text(.5,.877,'28,463 source rows    →    24,837 patient-days    →    8,081 complete test pairs',ha='center',va='center',fontsize=9.0,color='#2d536c',fontweight='medium')
centres=[('安医大','FAH-AHMU','#23688d'),('淮北煤炭','WCEGH','#31806f'),('铜陵','TPH','#bb7946')]
x_positions=[.045,.355,.665];card_w=.29;card_y=.11;card_h=.715
stage_info=[('Source records','retained_source_rows'),('Patient-days','natural_day_rows'),('Adjacent pairs, 1–3 d','adjacent_1to3_days'),('Complete CBC pairs','complete_1to3_pairs'),('C0-origin pairs','c0_pairs')]
stage_y=[.665,.565,.465,.365,.265]
for (centre,display,accent),x in zip(centres,x_positions):
 row=flow[flow.center==centre].iloc[0]
 # Quiet shadow and white card establish three distinct lanes.
 ax.add_patch(FancyBboxPatch((x+.005,card_y-.006),card_w,card_h,boxstyle='round,pad=.008,rounding_size=.023',facecolor='#dce5eb',edgecolor='none',alpha=.6))
 ax.add_patch(FancyBboxPatch((x,card_y),card_w,card_h,boxstyle='round,pad=.008,rounding_size=.023',facecolor='white',edgecolor='#dfE7ec',linewidth=.8))
 ax.add_patch(FancyBboxPatch((x,card_y+card_h-.092),card_w,.092,boxstyle='round,pad=.008,rounding_size=.022',facecolor=accent,edgecolor='none'))
 # White overlay squares lower rounded corners of header, retaining a flat divider.
 ax.add_patch(plt.Rectangle((x,card_y+card_h-.092),card_w,.025,color=accent,ec='none'))
 ax.text(x+.022,card_y+card_h-.039,display,ha='left',va='center',fontsize=12,fontweight='bold',color='white')
 ax.text(x+.022,card_y+card_h-.070,'Leukemia records',ha='left',va='center',fontsize=7.8,color='#edf7fb')
 for j,((label,col),y) in enumerate(zip(stage_info,stage_y)):
  bx=x+.019;bw=card_w-.038
  ax.add_patch(FancyBboxPatch((bx,y-.031),bw,.062,boxstyle='round,pad=.004,rounding_size=.012',facecolor='#f3f7f9',edgecolor='#e2eaf0',linewidth=.7))
  ax.text(x+card_w/2,y+.013,label,ha='center',va='center',fontsize=8.0,color='#456071')
  ax.text(x+card_w/2,y-.014,f'{int(row[col]):,}',ha='center',va='center',fontsize=11.3,fontweight='bold',color='#183b53')
  if j<len(stage_info)-1:
   ax.annotate('',xy=(x+card_w/2,y-.062),xytext=(x+card_w/2,y-.037),arrowprops=dict(arrowstyle='-|>',color='#a3b5c0',lw=1.0,mutation_scale=9))
 # Highlight actual observed conversion, not an all-patient incidence.
 events=int(m[(m.center==centre)&(m.interval=='1-3 days')&(m.base=='C0')&(m.next!='C0')].events.sum())
 rate=100*events/int(row.c0_pairs)
 ax.annotate('',xy=(x+card_w/2,.205),xytext=(x+card_w/2,.228),arrowprops=dict(arrowstyle='-|>',color=accent,lw=1.2,mutation_scale=10))
 bx=x+.019;bw=card_w-.038
 ax.add_patch(FancyBboxPatch((bx,.136),bw,.069,boxstyle='round,pad=.005,rounding_size=.013',facecolor=accent,edgecolor='none'))
 ax.text(x+card_w/2,.182,'C0 → critical',ha='center',va='center',fontsize=8.1,fontweight='bold',color='white')
 ax.text(x+card_w/2,.154,f'{events:,}  ({rate:.1f}%)',ha='center',va='center',fontsize=9.8,fontweight='bold',color='white')
ax.text(.5,.070,'Complete CBC pair = WBC, HGB, PLT and NEUT available at both observed dates.',ha='center',fontsize=8.0,color='#5d7382')
ax.text(.5,.045,'Same-day repeats were aggregated by median; arrows describe inclusion, not disease progression.',ha='center',fontsize=7.8,color='#5d7382')
fig.savefig(R/'figure_1_cohort_flow.png',dpi=360,bbox_inches='tight',facecolor=fig.get_facecolor())
fig.savefig(R/'figure_1_cohort_flow.pdf',bbox_inches='tight',facecolor=fig.get_facecolor())
plt.close(fig)
print('redesigned three-centre flow figure generated')
