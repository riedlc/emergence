# Claims of emergence in AI should be grounded in information decomposition (NeurIPS'26) **SPOTLIGHT** 🔦

Companion code to 

> Riedl, C. & Rosas, F.E. (2026). **Claims of emergence in AI should be grounded in information decomposition.** *In Advances in Neural Information Processing Systems*, volume 39, 2026.

See [arXiv](https://arxiv.org/abs/2610.28479).

```bibtex
@inproceedings{riedl2026multi,
  title     = {Claims of emergence in AI should be grounded in information decomposition},
  author    = {Riedl, Christoph and Rosas, Fernando E.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  volume    = {39}
}
```

# Three emergence tests on a toy Ising system

A minimal, exactly solvable system on which the emergence capacity, unique-information and practical criteria can be understood before being applied to real data.

## Contents

| File | What it is |
|---|---|
| `pid_emergence.py` | The three tests, a bivariate PID (MMI, I_min and BROJA redundancy; BROJA for binary sources), the kinetic Ising ring, and the sampling / permutation-null layer. No dependencies beyond numpy. |
| `getting_started.ipynb` | Narrative notebook: one β in full, sweep over β, invariants, estimation layer, how to adapt to your data. |
| `test_invariants.py` | Provable properties only (β = 0 gives zeros, atoms non-negative and sum to joint MI under all three measures, full-order unique information of a deterministic macro is zero, flip-symmetry, sampled → exact, BROJA reproduces the closed-form AND / XOR / UNQ / RDN gates). |
| `fig/` | `sweep_n5.csv`, `estimation_n5.csv`, `ising_emergence_tests.png` (panels a to c), `exact_vs_sampled.png` written by the notebook. |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q test_invariants.py
jupyter nbconvert --to notebook --execute --inplace getting_started.ipynb
```

Sections 3 to 5 of the notebook are exact and need no seed. Section 6 uses seed 20261008.
Runtime is under a minute.

## Scope

n = 5 spins, synchronous Glauber dynamics, lag 1, macros = magnetisation and energy
(domain-wall count). Redundancy: MMI (primary), I_min and BROJA (second measures; MMI and I_min
coincide on this system, BROJA does not). The permutation null is a full shuffle, for illustration;
see the paper for block-shuffled and functional nulls. 
