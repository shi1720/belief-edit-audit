import sys,json,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from belief_edit import *
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
rows=[]
for n in [6,12,18]:
    for r in [2,3]:
        for seed in range(20):
            rng=np.random.default_rng(9000000+n*10000+r*100+seed)
            labels=rng.permutation(np.arange(n)%r);C=partition_matrix(labels)
            allowed=np.zeros((n,n),bool);K=np.zeros((n,n))
            for i in range(n):
                if seed<10: js=rng.choice(n,3,replace=False)
                else: js=np.r_[i,rng.choice(np.delete(np.arange(n),i),2,replace=False)]
                allowed[i,js]=True;K[i,js]=rng.dirichlet(np.full(3,.3))
            common=[]
            for g in range(r):
                common.append([h for h in range(r) if np.all(np.any(allowed[labels==g][:,labels==h],axis=1))])
            predicted=all(len(x)>0 for x in common)
            base=repair(K,C)
            try: sol=repair(K,C,allowed);actual=True
            except RuntimeError: sol=None;actual=False
            assert predicted==actual
            row=dict(n=n,r=r,seed=seed,regime='free_support' if seed<10 else 'diagonal_allowed',
                predicted_feasible=predicted,solver_feasible=actual,unconstrained=base.radius,
                constrained=np.nan,reduced_gap=np.nan)
            if sol:
                rad,_=partition_radius(K,labels,allowed)
                assert abs(rad-sol.radius)<1e-7
                assert sol.dual_violation<1e-7 and abs(sol.gap)<1e-7 and sol.feasibility<1e-7
                row['constrained']=sol.radius;row['reduced_gap']=abs(rad-sol.radius)
            np.savez_compressed(OUT/'instances'/f'support_n{n}_r{r}_s{seed}.npz',
                K=K,E=C,labels=labels,allowed=allowed,
                repaired=sol.kernel if sol else np.empty((0,0)),
                dual_eq=sol.dual_eq if sol else [],dual_ub=sol.dual_ub if sol else [])
            rows.append(row)
with (OUT/'support.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary={}
for regime in ['free_support','diagonal_allowed']:
    batch=[x for x in rows if x['regime']==regime]
    feasible=[x for x in batch if x['solver_feasible']]
    summary[regime]={'count':len(batch),'feasible':len(feasible),
        'mean_unconstrained':float(np.mean([x['unconstrained'] for x in batch])),
        'mean_constrained_feasible':float(np.mean([x['constrained'] for x in feasible])),
        'mean_extra_cost_feasible':float(np.mean([x['constrained']-x['unconstrained'] for x in feasible]))}
(OUT/'support_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
