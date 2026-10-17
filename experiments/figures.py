"""Publication figures generated only from saved results or stated formulas."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'figures';R=ROOT/'results'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
blue='#205b8d';orange='#c0642b';green='#2c8064';gray='#687681'

def save(fig,name):
    fig.savefig(F/f'{name}.pdf');fig.savefig(F/f'{name}.png',dpi=180);plt.close(fig)

# Exact nonpartition witness, E=[[1,0],[0,1],[1/2,1/2]].
fig,ax=plt.subplots(figsize=(6.6,2.55));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
boxes=[(.1,2.5,2.5,.9,'Distinct beliefs\n$b=e_3$,  $c=(e_1+e_2)/2$'),
       (3.5,2.5,2.4,.9,'Same prediction\n$bE=cE=(1/2,1/2)$'),
       (6.9,2.5,2.8,.9,'A quotient edit must\nreturn one result'),
       (.1,.45,2.5,.9,'Apply the specified edit\n$e_3\mapsto e_1$; anchors fixed'),
       (3.5,.45,2.4,.9,'Different edited beliefs\n$bKE=(1,0)$\n$cKE=(1/2,1/2)$'),
       (6.9,.45,2.8,.9,'Conflict: one compressed\ninput, two required outputs')]
for x,y,w,h,t in boxes:
    color=orange if y<2 else blue
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.14',fc='#f7f9fb',ec=color,lw=1))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8.4,color='#182735')
for y in [2.95,.9]:
    for x1,x2 in [(2.8,3.3),(6.1,6.7)]:ax.annotate('',(x2,y),(x1,y),arrowprops=dict(arrowstyle='->',color=gray))
ax.annotate('',(1.35,1.6),(1.35,2.2),arrowprops=dict(arrowstyle='->',color=gray))
save(fig,'01_mixture_obstruction')

sharp=list(csv.DictReader((R/'sharp.csv').open()))
cycle=[x for x in sharp if x['family']=='cycle'];branch=[x for x in sharp if x['family']=='branch']
fig,axs=plt.subplots(1,2,figsize=(6.6,2.55),layout='constrained')
n=np.array([int(x['size']) for x in cycle]);axs[0].plot(n,n,'o-',ms=3,color=blue,label='With cyclic edit')
axs[0].plot(n,np.full(len(n),2),'--',color=gray,label='Passive prediction')
axs[0].set(xlabel='Number of latent states $n$',ylabel='Predictive linear dimension',title='(a) Dimension after editing')
axs[0].legend(frameon=False,fontsize=7,loc='upper left')
m=np.array([int(x['size']) for x in branch]);axs[1].plot(m,1-1/m,'-',color=blue,label=r'$1-1/m$')
axs[1].scatter(m,[float(x['radius']) for x in branch],color=orange,s=17,label='LP optimum',zorder=4)
axs[1].set(xlabel='Distinct target blocks $m$',ylabel='Minimum worst-row TV distortion',ylim=(.4,1),title='(b) Minimum repair cost')
axs[1].legend(frameon=False,fontsize=7)
save(fig,'02_sharp_families')

main=list(csv.DictReader((R/'main.csv').open()));support=list(csv.DictReader((R/'support.csv').open()))
part=[x for x in main if x['family']=='partition']
fig,axs=plt.subplots(1,2,figsize=(6.6,2.7),layout='constrained')
for r,color in [(2,blue),(3,orange)]:
    a=[x for x in part if int(x['r'])==r]
    axs[0].scatter([float(x['mean_macro']) for x in a],[float(x['radius']) for x in a],s=12,alpha=.65,color=color,label=f'{r} blocks')
axs[0].plot([0,.75],[0,.75],'--',color=gray,lw=.8)
axs[0].set(xlabel='Mean-aggregate baseline: worst-row TV',ylabel='Minimax repair: worst-row TV',title='(a) Optimal versus mean center',xlim=(.15,.7),ylim=(.15,.7))
axs[0].legend(frameon=False,fontsize=7)
for regime,color,label in [('free_support',orange,'Arbitrary support: 35/60 feasible'),('diagonal_allowed',blue,'Diagonal allowed: 60/60 feasible')]:
    a=[x for x in support if x['regime']==regime and x['solver_feasible']=='True']
    axs[1].scatter([float(x['unconstrained']) for x in a],[float(x['constrained']) for x in a],s=12,alpha=.65,color=color,label=label)
axs[1].plot([0,1],[0,1],'--',color=gray,lw=.8)
axs[1].set(xlabel='Unrestricted optimum',ylabel='Support-constrained optimum',title='(b) Support constraints',xlim=(0,1.02),ylim=(0,1.02))
axs[1].legend(frameon=True,facecolor='white',edgecolor='none',fontsize=6.3,loc='lower right')
save(fig,'03_repair_results')

composition=list(csv.DictReader((R/'composition.csv').open()))
fig,axs=plt.subplots(1,2,figsize=(6.6,2.55),layout='constrained')
axs[0].scatter([float(x['bound']) for x in composition],[float(x['tv']) for x in composition],s=8,alpha=.32,color=blue)
axs[0].plot([0,1],[0,1],'--',color=gray,lw=.8)
axs[0].set(xlabel=r'Coupling bound $1-(1-\delta)^q$',ylabel='Exact joint-output TV',title='(a) Joint transcript bound',xlim=(0,1.02),ylim=(0,1.02))
eps=np.logspace(-6,-1,6)
axs[1].loglog(eps,2*eps,'o-',color=blue,label='Passive one-token TV')
axs[1].loglog(eps,np.minimum(1,20*eps),'--',color=gray,label='Passive ten-token upper bound')
axs[1].loglog(eps,np.full(len(eps),.9),'-',color=orange,label='One-token TV after edit')
axs[1].set(xlabel=r'Near-aliasing parameter $\epsilon$',ylabel='TV distance',title='(b) Near-aliasing construction',ylim=(1e-6,1.5))
axs[1].legend(frameon=False,fontsize=6.5,loc='lower right')
save(fig,'04_composition_and_conditioning')

s=json.loads((R/'summary.json').read_text());ss=json.loads((R/'support_summary.json').read_text())
table=[]
for family in ['partition','mixture']:
    a=s[family]
    table.append(f"{family.title()} & 120 & {a['radius']['mean']:.3f} & "
                 +(f"{a['mean_macro']['mean']:.3f}" if family=='partition' else '---')
                 +f" & {a['identity']['mean']:.3f} & {1000*a['solve_seconds']['median']:.1f} \\\\")
(ROOT/'paper'/'results_table.tex').write_text('\n'.join(
    '\\newcommand{\\'+name+'Row}{'+row+'}'
    for name,row in zip(['Partition','Mixture'],table))+'\n')
print('Wrote four vector figures and generated results table.')
