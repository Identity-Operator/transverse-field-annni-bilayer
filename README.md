# Onset of a quantum Fisher–Selke sequence in the transverse-field ANNNI model and its bilayer

Code, result files and LaTeX source for the paper

> T.-V. Truong, V.-Q.-M. Nguyen, H.-L. Nguyen and T.-T. Nguyen,
> *Onset of a quantum Fisher–Selke sequence in the transverse-field ANNNI model and its bilayer* (2026).

The paper derives the low-field phase diagram of the transverse-field axial next-nearest-neighbor Ising (ANNNI)
model on the square lattice and on a bilayer near the multiphase point κ = 1/2, by perturbation theory in the
transverse field to order g⁸. It tests the result with stochastic series expansion (SSE) quantum Monte Carlo and
thermodynamic integration, including a pre-registered test of the ⟨23⟩ phase.

## Layout

The directory structure is the one the scripts were run in. Run every script from the repository root.

| Path | Contents |
|---|---|
| `phase_pipeline/qmc/sse.py` | SSE engine (numba): bilayer lattice, bond classes, cluster updates, template starts |
| `phase_pipeline/` (rest) | helpers used by the engine (`io_utils.py`, `qmc/engine.py`, `qmc/sampling.py`, `qmc/trotter.py`) |
| `experiments/pt_order*.py` | perturbation theory in exact rational arithmetic (orders g⁴, g⁶, g⁸; convex hulls) |
| `experiments/sse_*.py` | simulation drivers (template branches, anneals, benchmarks, checks) |
| `experiments/s5a_analyze_v3.py` | analysis of the pre-registered test (free energies, crossings, tests T1–T8) |
| `experiments/revision1_*.py` | analysis of the later runs (common lattice, replicates, sweeps, temperature) |
| `experiments/run_*.sh` | the run queues, in the order they were executed |
| `10_results/bilayer_annni_paper_results/reanalysis_v3/` | all results used in the paper |
| `manuscript/paper1_physics/` | manuscript and supplement source and PDFs, bibliography, figures, figure script, pre-registration plan |

## Tables and figures

| Item | Script | Result files (under `10_results/bilayer_annni_paper_results/reanalysis_v3/`) |
|---|---|---|
| Fig. 1 | `manuscript/paper1_physics/make_figures_p1.py model` | (schematic) |
| Fig. 2, Table IV, Table V | `experiments/sse_branches_v3.py`, `audit_12/secVA_numbers.py`, `experiments/wedge_fit_v3.py`; `make_figures_p1.py wedge` | `sse_validation/branches/`, `audit_12/secVA_numbers.json`, `wedge/`, `perturbation/pt_order8_tables.json` |
| Fig. 3 | `experiments/sse_branches_v3.py`, `experiments/sse_anneal_rect_v3.py`; `make_figures_p1.py anneal` | `sse_validation/branches/`, `sse_validation/anneal_rect/` |
| Fig. 4, Fig. 5, Table VI, Table S2 | `experiments/sse_seq_branches_v3.py`, `experiments/s5a_analyze_v3.py`; `make_figures_p1.py f23 window` | `sse_validation/seq_branches/`, `s5a_analysis/` |
| Fig. 6, Table S1 | `experiments/sse_ed_benchmark_v3.py`, `experiments/sse_tfim2d_v3.py`; `make_figures_p1.py validation` | `sse_validation/sse_ed_benchmark.csv`, `sse_validation/tfim2d*/`, `sse_validation/branches/`, `sse_validation/seq_branches/` |
| Table I | analytic, Eq. (5) | |
| Table II, Table III, Table VII | `experiments/pt_order4_v3.py`, `pt_order4_hull_v3.py`, `pt_order6_v3.py`, `pt_order8_v3.py`, `pt_order8_hull.py` | `perturbation/` (independent checks in `perturbation/audit_9b*/`) |
| Table S3 | `experiments/revision1_queue_v3.py`, `experiments/revision1_stats_v3.py` | `revision1/queue_V*.txt`, `revision1/queue_V*.json`, `revision1/replicates.json` |
| Error budget (Sec. S3) | `experiments/edge_rows_thermal_v3.py`, `experiments/s5a_analyze_v3.py` | `perturbation/edge_rows_thermal.txt`, `s5a_analysis/s5a_summary.txt` |
| End-to-end ED test (Sec. IV C) | `experiments/sse_ti_exact_check_v3.py` | `sse_validation/ti_exact_check*.txt` |
| Autocorrelation times (Sec. IV A) | `experiments/sse_tau_int_v3.py` | `sse_validation/tau_int.txt` |
| Low-field checks, frozen states (Sec. IV B) | `experiments/sse_lowg_check_v3.py`, `experiments/small_g_check_v3.py` | `sse_validation/lowg/`, `sse_validation/branches/` |

`manuscript/paper1_physics/s5a_prereg.md` is the written plan of the pre-registered test, with its change log; the
Supplemental Material (Sec. S2) summarizes it. Run logs are in `sse_validation/seq_branches/revision1.log` and
`sse_validation/overnight_queue.log`.

## Reproducing

Python 3.11 with the pinned packages in `requirements.txt`:

```
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PY=$PWD/.venv/bin/python
$PY experiments/pt_order8_v3.py --verify                 # perturbation theory and its checks
$PY experiments/s5a_analyze_v3.py                        # re-analysis of the pre-registered test from the stored branches
$PY manuscript/paper1_physics/make_figures_p1.py         # all figures
```

The analysis and figure scripts read the stored branch data and run in minutes. The simulations themselves (the
`run_*.sh` queues and the `sse_*.py` drivers) take hours to days on a desktop machine. The drivers write to the
same paths as the stored results, so rerun them in a copy of the repository.

## Licenses

Code: MIT (`LICENSE`). Result files and figures: CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`). The manuscript files are not
covered by these licenses.
