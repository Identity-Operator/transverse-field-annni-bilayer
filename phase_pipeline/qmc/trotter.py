"""Trotter (imaginary-time discretization) convergence benchmark."""
from __future__ import annotations

from pathlib import Path

from ..io_utils import write_rows
from .engine import run_pimc_batch
from .sampling import aggregate_seed_rows, make_param_row


def run_trotter_bench(outdir: Path, smoke=False, num_threads=4):
    stage = outdir / "qmc" / "trotter_benchmark"
    stage.mkdir(parents=True, exist_ok=True)
    rows = []
    Ms = [8,16] if smoke else [16,32,64,128]
    points = [
        make_param_row(0.0, 0.8, 0.5, "I", point_id="trotter_tfim"),
        make_param_row(0.6, 1.2, -0.5, "I", point_id="trotter_frustrated"),
    ]
    seeds = [101,203] if smoke else [101,203,307,409]
    for M in Ms:
        rr = run_pimc_batch(points, L=6 if not smoke else 4, T=0.5, Mtau=M, seeds=seeds,
                            therm=60 if smoke else 250, meas=120 if smoke else 500,
                            every=4 if smoke else 5, num_threads=num_threads)
        rows.extend(rr)
    agg = aggregate_seed_rows(rows)
    write_rows(stage / "trotter_seed_rows.csv", rows)
    write_rows(stage / "trotter_aggregated.csv", agg)
    return agg
