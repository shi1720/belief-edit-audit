"""Stronger baseline and independent replay of saved primal/dual LP witnesses.

Added after the initial main run. No LPs are solved by this script. The
mean-aggregate baseline preserves original within-block conditionals and is
stronger for this metric than reconstructing a uniform latent representative.
"""
import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from belief_edit import *
from run import interval
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'
rows=list(csv.DictReader((OUT/'main.csv').open()))
for row in rows:
    row['mean_macro']='nan'
    if row['family']=='partition':
        d=np.load(OUT/'instances'/f'partition_n{row["n"]}_r{row["r"]}_s{row["seed"]}.npz')
        K,labels=d['K'],d['labels'];C=partition_matrix(labels)
        P=K@C
        means=np.array([P[labels==g].mean(0) for g in range(C.shape[1])])
        L=lift_centers(K,labels,means)
        row['mean_macro']=row_tv(K,L)
        assert compatibility_defect(L,C)<1e-8
        assert float(row['radius'])<=row['mean_macro']+1e-7
with (OUT/'main.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
s=json.loads((OUT/'summary.json').read_text())
p=[row for row in rows if row['family']=='partition']
s['partition']['mean_macro']=interval([float(x['mean_macro']) for x in p])
s['partition']['mean_macro_reduction']=interval([float(x['mean_macro'])-float(x['radius']) for x in p])
s['numerical_rank_below_n']=sum(int(x['edit_rank'])<int(x['n']) for x in rows)
(OUT/'summary.json').write_text(json.dumps(s,indent=2))
# The verifier independently checks kernels, quotient equalities, primal costs,
# and the saved LP dual vectors, without trusting the original objective report.
verified=0;infeasible=0;max_gap=0.;max_residual=0.
for path in sorted((OUT/'instances').glob('*.npz')):
    d=np.load(path)
    if d['repaired'].size==0:
        # Independently check the combinatorial obstruction, no solver call.
        labels=d['labels'];allowed=d['allowed'];m=d['E'].shape[1]
        common=[all(any(np.all(np.any(allowed[labels==g][:,labels==h],axis=1))
                       for h in range(m)) for g in range(m))]
        assert not common[0]
        infeasible+=1;continue
    K,E,L=d['K'],d['E'],d['repaired']
    assert stochastic(L)
    assert compatibility_defect(L,E)<1e-7
    allowed=d['allowed'] if 'allowed' in d else None
    if allowed is not None: assert np.max(abs(L[~allowed]),initial=0)<1e-7
    # Preserve the actual numerical bases used by the optimizer: SVD/null-space
    # signs and rotations can vary across BLAS implementations. Check the bases
    # independently before using them to reconstruct the saved dual equations.
    U,N=d['observable'],d['annihilator']
    assert U.shape[1]+N.shape[1]==len(K)
    assert np.max(abs(U.T@U-np.eye(U.shape[1])),initial=0)<1e-7
    assert np.max(abs(N.T@N-np.eye(N.shape[1])),initial=0)<1e-7
    assert np.max(abs(U.T@N),initial=0)<1e-7
    assert np.linalg.norm((np.eye(len(K))-U@U.T)@E)<1e-7
    assert U.shape[1]==np.linalg.matrix_rank(E)
    c,A,b,B,e=repair_problem(K,E,allowed,basis=(U,N))
    y,z=d['dual_eq'],d['dual_ub']
    dual_res=max(0.,np.max(z),-np.min(c-B.T@y-A.T@z))
    assert dual_res<1e-7
    lower=e@y+b@z;upper=row_tv(K,L)
    assert abs(upper-lower)<1e-7
    max_gap=max(max_gap,abs(upper-lower));max_residual=max(max_residual,dual_res)
    verified+=1
report={'saved_repairs_verified':verified,'infeasibility_obstructions_verified':infeasible,
    'maximum_dual_residual':max_residual,'maximum_cost_dual_gap':max_gap,
    'verification':'float64 residual checks, not formal rational certification'}
(OUT/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
print('Stronger mean-aggregate baseline:',s['partition']['mean_macro'])
print('Paired improvement:',s['partition']['mean_macro_reduction'])
