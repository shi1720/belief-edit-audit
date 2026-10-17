"""Compatibility and minimax repair for finite predictive belief representations.

All beliefs are ROW vectors. Columns of O encode observable future tests.
The numerical routines use float64; they are not exact-arithmetic certificates.
"""
from dataclasses import dataclass
from collections import deque
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog
from scipy import sparse


def stochastic(K, atol=1e-8):
    K = np.asarray(K, dtype=float)
    return (K.ndim == 2 and K.shape[0] == K.shape[1]
            and np.min(K) >= -atol and np.max(abs(K.sum(1)-1)) <= atol)


def observable_basis(operators, tol=1e-10):
    """Backward-space algorithm; returns orthonormal basis and actual test words.

    A word (a,b) means operators[a] @ operators[b] @ ones.
    Modified Gram-Schmidt is repeated to limit loss of orthogonality.
    """
    ops = [np.asarray(A, float) for A in operators]
    n = ops[0].shape[0]
    cols, words = [], []
    queue = deque([(np.ones(n), ())])
    while queue:
        v, word = queue.popleft()
        z = v.copy()
        for _ in range(2):
            for q in cols:
                z -= q * (q @ z)
        norm = np.linalg.norm(z)
        if norm <= tol * max(1., np.linalg.norm(v)):
            continue
        cols.append(z/norm)
        words.append(word)
        for a, A in enumerate(ops):
            queue.append((A @ v, (a,)+word))
        if len(cols) == n:
            break
    return np.column_stack(cols), words


def canonical_basis(O, tol=1e-10):
    u, s, _ = np.linalg.svd(np.asarray(O, float), full_matrices=False)
    rank = int(np.sum(s > tol*max(1., s[0])))
    return u[:, :rank]


def compatibility_defect(K, O):
    U = canonical_basis(O)
    return float(np.linalg.norm((np.eye(len(K))-U@U.T)@K@U, ord=2))


def witness(K, O, tol=1e-9):
    """Two valid priors agreeing on O but disagreeing on an edited O-column.

    O must contain ones in its column span. Returns None for compatible K.
    """
    U = canonical_basis(O)
    R = (np.eye(len(K))-U@U.T)@K@O
    j = int(np.argmax(np.linalg.norm(R, axis=0)))
    v = R[:, j]
    if np.linalg.norm(v) <= tol:
        return None
    pos, neg = np.maximum(v, 0), np.maximum(-v, 0)
    b, c = pos/pos.sum(), neg/neg.sum()
    return b, c, j


def row_tv(K, L):
    return float(np.max(np.sum(abs(np.asarray(K)-np.asarray(L)), axis=1)/2))


@dataclass
class Repair:
    kernel: np.ndarray
    radius: float
    dual_lower: float
    dual_violation: float
    feasibility: float
    gap: float
    dual_eq: np.ndarray
    dual_ub: np.ndarray
    observable: np.ndarray
    annihilator: np.ndarray


def repair_problem(K, O, allowed=None, basis=None):
    """LP with variables [L.flatten(), abs(L-K).flatten(), t], all >= 0."""
    K = np.asarray(K, float)
    n = len(K)
    if not stochastic(K):
        raise ValueError('K must be a row-stochastic square matrix')
    U = canonical_basis(O) if basis is None else np.asarray(basis[0],float)
    if np.linalg.norm((np.eye(n)-U@U.T)@np.ones(n)) > 1e-8:
        raise ValueError('Observable column span must contain ones')
    N = null_space(U.T) if basis is None else np.asarray(basis[1],float)
    nn = n*n
    # vec_row(N.T L U) = (N.T kron U.T) vec_row(L).
    compat = sparse.kron(sparse.csr_matrix(N.T), sparse.csr_matrix(U.T))
    rowsums = sparse.kron(sparse.eye(n), np.ones((1,n)))
    if allowed is None:
        support=sparse.csr_matrix((0,nn))
    else:
        allowed=np.asarray(allowed,bool)
        if allowed.shape!=K.shape: raise ValueError('allowed must have K.shape')
        forbidden=np.flatnonzero(~allowed.ravel())
        support=sparse.coo_matrix((np.ones(len(forbidden)),
                    (np.arange(len(forbidden)),forbidden)),shape=(len(forbidden),nn)).tocsr()
    Aeq0 = sparse.vstack([rowsums, compat, support], format='csr')
    Aeq = sparse.hstack([Aeq0, sparse.csr_matrix((Aeq0.shape[0],nn+1))], format='csr')
    beq = np.r_[np.ones(n), np.zeros(compat.shape[0]+support.shape[0])]
    I = sparse.eye(nn)
    z = sparse.csr_matrix((nn,1))
    Aub = sparse.vstack([
        sparse.hstack([I,-I,z]), sparse.hstack([-I,-I,z]),
        sparse.hstack([sparse.csr_matrix((n,nn)), rowsums, -2*np.ones((n,1))])
    ],format='csr')
    bub = np.r_[K.ravel(),-K.ravel(),np.zeros(n)]
    objective = np.zeros(2*nn+1)
    objective[-1] = 1.
    return objective, Aub, bub, Aeq, beq


def repair(K, O, allowed=None):
    U=canonical_basis(O);N=null_space(U.T)
    objective, Aub, bub, Aeq, beq = repair_problem(K,O,allowed,basis=(U,N))
    sol = linprog(objective, A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,
                  bounds=(0,None),method='highs',
                  options={'dual_feasibility_tolerance':1e-9,
                           'primal_feasibility_tolerance':1e-9})
    if not sol.success:
        raise RuntimeError(sol.message)
    y,z = sol.eqlin.marginals,sol.ineqlin.marginals
    reduced = objective - Aeq.T@y - Aub.T@z
    dual_violation = float(max(0., np.max(z), -np.min(reduced)))
    lower = float(beq@y + bub@z)
    primal_violation = float(max(0.,np.max(abs(Aeq@sol.x-beq)),
                                np.max(Aub@sol.x-bub),-np.min(sol.x)))
    n = len(K)
    return Repair(sol.x[:n*n].reshape(n,n),float(sol.fun),lower,
                  dual_violation,primal_violation,float(sol.fun-lower),y,z,U,N)


def partition_matrix(labels):
    labels = np.asarray(labels,int)
    return np.eye(int(max(labels))+1)[labels]


def partition_radius(K, labels, allowed=None):
    """Independent smaller LP: TV Chebyshev center of aggregate edit rows.

    This implementation does not call the full matrix repair solver.
    """
    labels=np.asarray(labels,int)
    C=partition_matrix(labels)
    P=K@C
    m=C.shape[1]
    centers=[]
    radii=[]
    for g in range(m):
        Pblock=P[labels==g]
        k=len(Pblock)
        # Variables q[m], u[k,m], t.
        count=m+k*m+1
        obj=np.zeros(count);obj[-1]=1
        A=[];b=[]
        for i in range(k):
            for j in range(m):
                a=np.zeros(count);a[j]=1;a[m+i*m+j]=-1
                A.append(a);b.append(Pblock[i,j])
                a=np.zeros(count);a[j]=-1;a[m+i*m+j]=-1
                A.append(a);b.append(-Pblock[i,j])
            a=np.zeros(count);a[m+i*m:m+(i+1)*m]=1;a[-1]=-2
            A.append(a);b.append(0)
        eq=np.zeros((1,count));eq[0,:m]=1
        bounds=[(0,None)]*count
        if allowed is not None:
            block_allowed=np.asarray(allowed,bool)[labels==g]
            common=np.array([np.all(np.any(block_allowed[:,labels==h],axis=1)) for h in range(m)])
            for h in range(m):
                if not common[h]: bounds[h]=(0,0)
        sol=linprog(obj,A_ub=np.array(A),b_ub=np.array(b),A_eq=eq,b_eq=[1],bounds=bounds,method='highs')
        if not sol.success: raise RuntimeError(sol.message)
        radii.append(float(sol.fun));centers.append(sol.x[:m])
    return max(radii),np.array(centers)


def average_reconstruct(K, labels):
    C=partition_matrix(labels)
    D=C.T/C.sum(0)[:,None]
    return C@D@K


def lift_centers(K, labels, centers):
    """Preserve K's within-target-block conditionals where they have mass."""
    labels=np.asarray(labels,int)
    L=np.zeros_like(K)
    for i in range(len(K)):
        for g in range(len(centers)):
            js=np.flatnonzero(labels==g)
            mass=K[i,js].sum()
            cond=K[i,js]/mass if mass>1e-15 else np.ones(len(js))/len(js)
            L[i,js]=centers[labels[i],g]*cond
    return L


def transcript(b, emission_ops, K, schedule):
    """Enumerate JOINT output probabilities for a fixed Boolean edit schedule."""
    states=[np.asarray(b,float)]
    for edit in schedule:
        states=[v@(K if edit else np.eye(len(K)))@A
                for v in states for A in emission_ops]
    return np.array([v.sum() for v in states])
