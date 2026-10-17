"""Reconstruct every sharp-family model independently from saved table sizes."""
import csv,sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from belief_edit import *
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'results/sharp.csv').open()))
for row in rows:
    size=int(row['size'])
    if row['family']=='cycle':
        n=size;labels=np.r_[0,np.ones(n-1,dtype=int)]
        K=np.eye(n)[(np.arange(n)+1)%n]
    else:
        m=size;n=2*m;labels=np.r_[np.zeros(m,dtype=int),np.arange(1,m+1)]
        K=np.eye(n);K[:m]=np.eye(n)[np.arange(m,2*m)]
    C=partition_matrix(labels);ops=[np.diag(C[:,g]) for g in range(C.shape[1])]
    U,_=observable_basis(ops);V,_=observable_basis(ops+[K])
    assert U.shape[1]==int(row['rank_before'])
    assert V.shape[1]==int(row['rank_after'])
    sol=repair(K,C)
    assert abs(sol.radius-float(row['formula']))<1e-7
    assert abs(sol.radius-float(row['radius']))<1e-7
(ROOT/'results/sharp_verification.json').write_text(json.dumps({'verified_sharp_models':len(rows)},indent=2))
print('Verified ranks and costs of all',len(rows),'sharp models.')
