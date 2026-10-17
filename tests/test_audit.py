import sys,itertools,unittest
from pathlib import Path
from fractions import Fraction as F
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from belief_edit import (repair,compatibility_defect,partition_matrix,
                         partition_radius,lift_centers,row_tv,witness,
                         observable_basis,stochastic,repair_problem)


def exact_rank(A):
    A=[[F(x) for x in row] for row in A]
    rank=0
    for j in range(len(A[0])):
        piv=next((i for i in range(rank,len(A)) if A[i][j]),None)
        if piv is None: continue
        A[rank],A[piv]=A[piv],A[rank]
        q=A[rank][j];A[rank]=[x/q for x in A[rank]]
        for i in range(len(A)):
            if i!=rank:
                q=A[i][j];A[i]=[x-q*y for x,y in zip(A[i],A[rank])]
        rank+=1
        if rank==len(A): break
    return rank


def exhaustive_check():
    count=0
    for n in range(2,5):
        partition=[[F(1),F(0)]]+[[F(0),F(1)] for _ in range(n-1)]
        mixture=[[F(1),F(0)],[F(0),F(1)]]+[[F(i,n-1),1-F(i,n-1)] for i in range(1,n-1)]
        for E in [partition,mixture]:
            rank=exact_rank(E)
            for dest in itertools.product(range(n),repeat=n):
                KE=[E[j] for j in dest]
                expected=exact_rank([a+b for a,b in zip(E,KE)])==rank
                K=np.eye(n)[list(dest)]
                got=compatibility_defect(K,np.array(E,float))<1e-8
                if got!=expected: raise AssertionError((n,dest,E))
                count+=1
    return count


class AuditTests(unittest.TestCase):
    def test_exhaustive_rational(self):
        self.assertEqual(exhaustive_check(),574)

    def test_mixture_witness(self):
        E=np.array([[1,0],[0,1],[.5,.5]])
        K=np.array([[1,0,0],[0,1,0],[1,0,0.]])
        w=witness(K,E)
        self.assertIsNotNone(w)
        b,c,j=w
        np.testing.assert_allclose(b@E,c@E,atol=1e-10)
        self.assertGreater(abs((b-c)@K@E[:,j]),.1)

    def test_full_representation_needs_no_repair(self):
        K=np.array([[.2,.8],[.7,.3]])
        sol=repair(K,np.eye(2))
        self.assertLess(sol.radius,1e-8)
        np.testing.assert_allclose(sol.kernel,K,atol=1e-8)

    def test_partition_reduction(self):
        rng=np.random.default_rng(731)
        for _ in range(12):
            K=rng.dirichlet(np.ones(6),size=6)
            labels=np.array([0,0,0,1,1,2])
            C=partition_matrix(labels)
            r,centers=partition_radius(K,labels)
            sol=repair(K,C)
            self.assertAlmostEqual(r,sol.radius,places=7)
            L=lift_centers(K,labels,centers)
            self.assertAlmostEqual(row_tv(K,L),r,places=7)
            self.assertLess(compatibility_defect(L,C),1e-8)

    def test_branching_sharp(self):
        for m in range(2,8):
            labels=np.r_[np.zeros(m,dtype=int),np.arange(1,m+1)]
            K=np.eye(2*m)
            K[:m]=np.eye(2*m)[np.arange(m,2*m)]
            sol=repair(K,partition_matrix(labels))
            self.assertAlmostEqual(sol.radius,1-1/m,places=7)

    def test_cyclic_dimension(self):
        for n in range(3,11):
            p=np.r_[1.,np.zeros(n-1)]
            ops=[np.diag(p),np.diag(1-p)]
            K=np.eye(n)[(np.arange(n)+1)%n]
            U,_=observable_basis(ops)
            V,words=observable_basis(ops+[K])
            self.assertEqual(U.shape[1],2)
            self.assertEqual(V.shape[1],n)
            self.assertLessEqual(max(map(len,words)),n-1)
            self.assertAlmostEqual(repair(K,U).radius,.5,places=7)

    def test_noop_and_reset_compatible(self):
        E=np.array([[1,0],[0,1],[.3,.7],[.6,.4]])
        for K in [np.eye(4),np.tile([.1,.2,.3,.4],(4,1))]:
            self.assertLess(compatibility_defect(K,E),1e-10)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):repair(np.eye(2)*2,np.eye(2))
        with self.assertRaises(ValueError):repair(np.eye(2),np.array([[1],[0]]))

    def test_stochastic_kernel(self):
        self.assertTrue(stochastic(np.eye(3)))
        self.assertFalse(stochastic(np.array([[1.1,-.1],[0,1]])))

    def test_forbidden_transition_infeasibility(self):
        K=np.eye(3)[[1,2,0]]
        C=partition_matrix([0,1,1])
        with self.assertRaises(RuntimeError):repair(K,C,allowed=K>0)
        allowed=(K+np.eye(3))>0
        sol=repair(K,C,allowed=allowed)
        self.assertAlmostEqual(sol.radius,1.,places=7)
        self.assertLess(np.max(abs(sol.kernel[~allowed])),1e-9)

    def test_saved_dual_basis_portability(self):
        E=np.array([[1.,0],[0,1],[.3,.7],[.6,.4]])
        K=np.random.default_rng(56).dirichlet(np.ones(4),size=4)
        sol=repair(K,E)
        # The same span is provided in a deliberately different coordinate basis.
        rotation=np.array([[.8,-.6],[.6,.8]])
        c,A,b,B,e=repair_problem(K,E@rotation,basis=(sol.observable,sol.annihilator))
        lower=e@sol.dual_eq+b@sol.dual_ub
        self.assertAlmostEqual(lower,row_tv(K,sol.kernel),places=8)
        self.assertGreaterEqual(np.min(c-B.T@sol.dual_eq-A.T@sol.dual_ub),-1e-8)

if __name__=='__main__': unittest.main()
