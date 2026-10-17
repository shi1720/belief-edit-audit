# Editing What Prediction Forgets

**Support-Constrained Repair of Compressed Belief Dynamics**  
Shivam Gupta · Independent Researcher · 17 October 2026

[Read the manuscript](paper/main.pdf).

This repository implements a finite-model audit of stochastic belief edits.
It checks whether an edit is well-defined on a predictive representation, finds
contradictory priors when it is not, and solves a minimum worst-row
total-variation repair with optional forbidden-transition constraints.

The theory uses established predictive-state, weighted-automata, causal
abstraction, and lumpability foundations. The work supplies a focused
support-constrained formulation, block-center reduction, sharp examples,
and reproducible computational checks. It does **not** certify neural steering,
identify causal latent structure from passive logs, or claim exhaustive novelty.
This is a research manuscript, not an accepted publication.

## Reproduce

Python 3.9 or newer; the recorded run used Python 3.9.6 on ARM64 macOS.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python3 experiments/run.py
OPENBLAS_NUM_THREADS=1 python3 experiments/support.py
OPENBLAS_NUM_THREADS=1 python3 experiments/augment_and_verify.py
OPENBLAS_NUM_THREADS=1 python3 experiments/check_sharp.py
python3 experiments/figures.py
cd paper
tectonic main.tex
```

Tectonic is only needed to rebuild the PDF. No external service, paid model,
credentials, or dataset download is needed to reproduce the experiments.
Runtime values depend on the machine and are descriptive only.

## Recorded evidence

- 574 small deterministic cases checked using independent rational arithmetic.
- 240 generated low-rank model/edit instances.
- 120 support-constrained instances, including 25 infeasible specifications.
- 335 saved feasible repairs verified against saved dual vectors without
  rerunning an optimizer.
- 27 sharp-family models checked against formulas.
- 600 eight-step schedules, with all 256 joint output strings enumerated each.
- 10 unit tests; four vector figures generated from results or stated formulas.

The stronger mean-aggregate baseline has mean worst-row TV 0.397 versus 0.346
for minimax repair on the partition generator; the paired difference is 0.0517.
The larger improvement over uniform latent reconstruction should not be
mistaken for the improvement over this stronger baseline.

The default backward-space calculation is numerical. At its fixed tolerance,
38 generated cases returned rank below the latent dimension. Neither numerical
rank nor LP residual checks are formal exact-arithmetic certificates.

## Files

| Path | Purpose |
|---|---|
| `src/belief_edit.py` | Compatibility, witnesses, repair LP, reduced solver |
| `tests/test_audit.py` | Exact finite oracle and unit checks |
| `experiments/PROTOCOL.md` | Settings written before the main run |
| `experiments/SUPPORT_AMENDMENT.md` | Exploratory support extension |
| `experiments/augment_and_verify.py` | Added stronger baseline and saved-witness verification |
| `results/*.csv` | Complete numerical results, including infeasibility |
| `results/instances/*.npz` | Raw kernels, representations, repairs, dual vectors |
| `results/summary.json` | Environment and descriptive statistics |
| `results/verification.json` | Independent replay results |
| `figures/` | Vector PDFs and PNG previews |
| `paper/` | Manuscript, bibliography, and generated table |
| `literature.csv` | Primary-source comparison and scope |
| `APPLICATION_CONTEXT.txt` | Why this is a program work sample |

The main protocol was local, not externally preregistered. The support
extension and stronger baseline were added after the first main run and are
identified as such. Bootstrap intervals describe the specified generators,
not a population of deployed models.

## Limitations and reuse

Edits are row-stochastic linear maps on supplied beliefs. Arbitrary nonlinear
posterior conditioning and arbitrary neural activation patches are outside the
implemented contract. Operators and allowed transitions are assumed correct.
The metric depends on the chosen latent realization; a full-rank interface
always passes the exact compatibility check, even when inference is poorly
conditioned.

Copyright 2026 Shivam Gupta. A redistribution or commercial-use license has
not been selected. Contact the author regarding reuse. Public availability is
not a claim of patentability or a legal opinion about intellectual property.
