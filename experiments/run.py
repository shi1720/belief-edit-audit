"""Run the frozen synthetic protocol and write raw instances, CSV, and summary."""
import sys,time,json,csv,platform,itertools,hashlib
from pathlib import Path
import numpy as np
import scipy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from belief_edit import *
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
(OUT/'instances').mkdir(exist_ok=True)


def interval(values,seed=8042):
    values=np.asarray(values,float)
    rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(values,size=(4000,len(values)),replace=True),axis=1)
    return {'mean':float(values.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'median':float(np.median(values)),'min':float(values.min()),'max':float(values.max())}


def run():
    started=time.time()
    rows=[];composition=[];sharp=[]
    for family in ['partition','mixture']:
        for n in [6,12,18]:
            for r in [2,3]:
                for seed in range(20):
                    rng=np.random.default_rng(100000*n+1000*r+seed)
                    labels=rng.permutation(np.arange(n)%r)
                    if family=='partition': E=partition_matrix(labels)
                    else: E=np.vstack([np.eye(r),rng.dirichlet(np.ones(r),size=n-r)])
                    K=rng.dirichlet(np.full(n,.3),size=n)
                    # Anchor lift, also valid for a partition with permuted labels.
                    anchors=[np.flatnonzero(labels==j)[0] for j in range(r)] if family=='partition' else list(range(r))
                    L=np.eye(n)[anchors]
                    p=np.linspace(.15,.85,r)
                    ops=[E@np.diag(p)@L,E@np.diag(1-p)@L]
                    U,_=observable_basis(ops)
                    # Distinct p identify r states using longer words for r=3.
                    assert U.shape[1]==r
                    V,_=observable_basis(ops+[K])
                    tick=time.perf_counter();sol=repair(K,E);seconds=time.perf_counter()-tick
                    actual=row_tv(K,sol.kernel)
                    assert abs(actual-sol.radius)<1e-7
                    assert sol.feasibility<1e-7 and sol.dual_violation<1e-7 and abs(sol.gap)<1e-7
                    assert compatibility_defect(sol.kernel,E)<1e-7
                    record=dict(family=family,n=n,r=r,seed=seed,passive_rank=U.shape[1],
                        edit_rank=V.shape[1],radius=actual,identity=row_tv(K,np.eye(n)),
                        defect_before=compatibility_defect(K,E),defect_after=compatibility_defect(sol.kernel,E),
                        solve_seconds=seconds,feasibility=sol.feasibility,
                        dual_violation=sol.dual_violation,gap=sol.gap,average=np.nan,reduced_gap=np.nan)
                    if family=='partition':
                        rad,centers=partition_radius(K,labels)
                        record['average']=row_tv(K,average_reconstruct(K,labels))
                        record['reduced_gap']=abs(rad-actual)
                        assert abs(rad-actual)<1e-7
                    name=f'{family}_n{n}_r{r}_s{seed}'
                    np.savez_compressed(OUT/'instances'/f'{name}.npz',K=K,E=E,
                        repaired=sol.kernel,labels=labels,dual_eq=sol.dual_eq,dual_ub=sol.dual_ub,
                        M0=ops[0],M1=ops[1])
                    rows.append(record)
                    if family=='mixture' and n==6 and r==2:
                        for k in range(10):
                            schedule=rng.random(8)<.5
                            b=rng.dirichlet(np.ones(n))
                            P=transcript(b,ops,K,schedule)
                            Q=transcript(b,ops,sol.kernel,schedule)
                            tv=float(abs(P-Q).sum()/2)
                            q=int(schedule.sum());bound=1-(1-actual)**q
                            assert tv<=bound+1e-8
                            composition.append(dict(seed=seed,schedule=''.join('1' if x else '0' for x in schedule),
                                q=q,tv=tv,bound=bound,radius=actual))
        print(f'{family}: {sum(x["family"]==family for x in rows)} instances',flush=True)
    # A second 40-seed composition-only batch yields 60 x 10 schedules total.
    for seed in range(20,60):
        rng=np.random.default_rng(600000+seed)
        n=6;r=2;E=np.vstack([np.eye(r),rng.dirichlet(np.ones(r),size=n-r)])
        K=rng.dirichlet(np.full(n,.3),size=n);sol=repair(K,E)
        L=np.eye(n)[:r];p=np.array([.15,.85]);ops=[E@np.diag(p)@L,E@np.diag(1-p)@L]
        for k in range(10):
            schedule=rng.random(8)<.5;b=rng.dirichlet(np.ones(n))
            P=transcript(b,ops,K,schedule);Q=transcript(b,ops,sol.kernel,schedule)
            tv=float(abs(P-Q).sum()/2);q=int(schedule.sum());bound=1-(1-sol.radius)**q
            assert tv<=bound+1e-8
            composition.append(dict(seed=seed,schedule=''.join('1' if x else '0' for x in schedule),q=q,tv=tv,bound=bound,radius=sol.radius))
    for n in range(3,21):
        p=np.r_[1.,np.zeros(n-1)];ops=[np.diag(p),np.diag(1-p)]
        K=np.eye(n)[(np.arange(n)+1)%n]
        U,_=observable_basis(ops);V,w=observable_basis(ops+[K]);sol=repair(K,U)
        assert U.shape[1]==2 and V.shape[1]==n and abs(sol.radius-.5)<1e-7
        labels=np.r_[0,np.ones(n-1,dtype=int)]
        sharp.append(dict(family='cycle',size=n,rank_before=2,rank_after=n,
                          radius=sol.radius,formula=.5,average=row_tv(K,average_reconstruct(K,labels))))
    for m in range(2,11):
        labels=np.r_[np.zeros(m,dtype=int),np.arange(1,m+1)]
        K=np.eye(2*m);K[:m]=np.eye(2*m)[np.arange(m,2*m)]
        C=partition_matrix(labels)
        ops=[np.diag(C[:,g]) for g in range(m+1)]
        U,_=observable_basis(ops);V,_=observable_basis(ops+[K])
        assert U.shape[1]==m+1 and V.shape[1]==2*m
        sol=repair(K,C)
        assert abs(sol.radius-(1-1/m))<1e-7
        sharp.append(dict(family='branch',size=m,rank_before=m+1,rank_after=2*m,
                          radius=sol.radius,formula=1-1/m,average=np.nan))
    for name,data in [('main',rows),('composition',composition),('sharp',sharp)]:
        with (OUT/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    summary={'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                'platform':platform.platform()},'elapsed_seconds':time.time()-started,'main_instances':len(rows),
                'composition_schedules':len(composition),'sharp_cases':len(sharp),
                'max_primal_violation':max(x['feasibility'] for x in rows),
                'max_dual_violation':max(x['dual_violation'] for x in rows),
                'max_abs_duality_gap':max(abs(x['gap']) for x in rows),
                'max_reduced_gap':max(x['reduced_gap'] for x in rows if x['family']=='partition'),
                'all_edit_rank_full':all(x['edit_rank']==x['n'] for x in rows),
                'composition_max_tv':max(x['tv'] for x in composition),
                'composition_violations':sum(x['tv']>x['bound']+1e-8 for x in composition)}
    for family in ['partition','mixture']:
        batch=[x for x in rows if x['family']==family]
        s={key:interval([x[key] for x in batch]) for key in ['radius','identity','solve_seconds']}
        s['mean_identity_reduction']=interval([x['identity']-x['radius'] for x in batch])
        if family=='partition':
            s['average']=interval([x['average'] for x in batch])
            s['mean_average_reduction']=interval([x['average']-x['radius'] for x in batch])
        summary[family]=s
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
