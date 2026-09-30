from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;d=pd.read_csv(R/'14_stable_subtype_state_matrix.csv',encoding='utf-8-sig');order=['AML','ALL','CML','CLL'];states=['C0','C1','C2+']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(9,7.8))
for ax,name in zip(axs.flat,order):
 q=d[d.subtype==name];rate=q.pivot(index='base_state',columns='next_state',values='rate').reindex(index=[0,1,2],columns=[0,1,2]).to_numpy();events=q.pivot(index='base_state',columns='next_state',values='events').reindex(index=[0,1,2],columns=[0,1,2]).to_numpy();pairs=q.drop_duplicates('base_state').set_index('base_state').loc[[0,1,2],'pairs'].to_numpy();ax.imshow(rate,cmap='Blues',vmin=0,vmax=.9)
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{100*rate[i,j]:.1f}%\n({events[i,j]}/{pairs[i]})',ha='center',va='center',fontsize=8,color='white' if rate[i,j]>.55 else '#163b54')
 ax.set_xticks(range(3),states);ax.set_yticks(range(3),states);ax.set_xlabel('Next observed state');ax.set_ylabel('Current state');ax.set_title(name,fontweight='bold')
fig.text(.07,.012,'Patients with changing recorded subtype labels were excluded; estimates describe observed complete adjacent test pairs.',fontsize=8)
fig.tight_layout(rect=[0,.03,1,1],h_pad=2.0,w_pad=1.7);fig.savefig(R/'supp_figure_3_subtype_matrices.png',dpi=300,bbox_inches='tight');fig.savefig(R/'supp_figure_3_subtype_matrices.pdf',bbox_inches='tight');plt.close(fig);print('subtype matrix figure saved')
