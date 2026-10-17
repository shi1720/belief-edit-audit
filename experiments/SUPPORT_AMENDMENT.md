# Support-constrained extension

Added after the first main-suite run, before running the support experiment.
Motivation: unrestricted repair can create forbidden latent transitions.
This is an explicitly exploratory extension of the frozen protocol.

Use n=6,12,18; r=2,3; 20 seeds per stratum (120 cases). Each row has
three allowed destinations. For seeds 0--9, sample three without replacement;
for seeds 10--19, include the diagonal and sample two other destinations.
Sample desired probabilities from Dirichlet(0.3) on those destinations.
Labels are balanced and independently permuted. Save each instance.

Compare feasibility predicted by the intersection-of-target-blocks criterion
with the full constrained LP. For feasible cases compare its optimum to an
independent restricted-center LP. Also solve the unconstrained version.
Do not discard infeasible instances. Report both support regimes separately.
