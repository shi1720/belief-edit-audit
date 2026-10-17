# Frozen evaluation protocol

Written before running `experiments/run.py` (17 October 2026). This is a local
protocol, not an external preregistration. It fixes the following settings.

1. Exact rational checking: enumerate all deterministic kernels for n=2,3,4
   on partition and nonpartition representations; compare an independent
   Fraction-based rank criterion with the floating-point audit.
2. Sharp families: cyclic edits for n=3,...,20; branching edits for m=2,...,10.
   Check passive/augmented rank and the closed-form repair radii.
3. Partition suite: n=6,12,18, quotient size m=2,3, six strata, 20 seeds each.
   Each balanced partition has a randomly permuted assignment. K has independent
   Dirichlet(0.3) rows. Compare full repair, independent block-center LP, average
   reconstruction, and identity. Metrics: worst-row TV, validity residual,
   objective gap, and wall-clock solve time. Save every instance and repair.
4. Nonpartition suite: the same (n,r) and seeds. E contains r anchors and n-r
   independent Dirichlet(1) rows; K has Dirichlet(0.3) rows. Two-symbol passive
   models M_x=E diag(p_x) L, L selecting the anchors, with distinct p_1 in
   [0.15,0.85]. Measure compatibility, edit-closed rank, and repair radius.
5. Composition: 60 instances with n=6,r=2, ten fixed schedules of length eight,
   independent initial priors; enumerate all 256 joint output probabilities.
   Check the q-edit TV bound; no conditional-posterior bound is asserted.
6. Near-aliasing: static four-state emission model, epsilon=10^-1,...,10^-6;
   compare two nearly indistinguishable priors and a fixed edit. Report exact
   one-token TV and a ten-token passive TV upper bound.

No external dataset, pretrained language model, participant, or API output is
used. Random seeds identify generated mathematical instances, not training
replications. Descriptive bootstrap intervals quantify variation under these
generators only. No parameter will be selected to optimize a headline result.
Failures will be retained and reported; numerical checks use 1e-7 tolerance.
