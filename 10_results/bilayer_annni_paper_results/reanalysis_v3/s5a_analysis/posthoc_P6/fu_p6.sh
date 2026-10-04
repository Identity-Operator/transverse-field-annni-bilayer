#!/usr/bin/env bash
# Declared post-hoc follow-up at P6 (kappa = 0.52, r = 1): (a) the 24 new chains alone, (b) pooled with the
# original 12 chains (36 per structure). The pre-registered P6 analysis is NOT touched (frozen script, frozen files).
# Usage: fu_p6.sh <seq_branches dir> <out dir>      (run from vu_work/)
set -eu
D=$1; OUT=$2
PY="$HOME/.venvs/annni-py311/bin/python -W ignore"
mkdir -p "$OUT/pooled36"
for f in seqbr_Lx20_Ly20_k0.52_r1_T0.15_23-2 seqbr_Lx20_Ly30_k0.52_r1_T0.15_23-3; do
  test -f "$D/${f}_fu1.csv" && test -f "$D/${f}_fu1.json" || { echo "missing $D/${f}_fu1.*"; exit 1; }
done
OMP_NUM_THREADS=1 $PY - "$D" "$OUT/pooled36" <<'PYEOF'
import sys, json, pandas as pd
from pathlib import Path
D, O = Path(sys.argv[1]), Path(sys.argv[2])
for f in ("seqbr_Lx20_Ly20_k0.52_r1_T0.15_23-2", "seqbr_Lx20_Ly30_k0.52_r1_T0.15_23-3"):
    a, b = pd.read_csv(D / f"{f}.csv"), pd.read_csv(D / f"{f}_fu1.csv")
    ja, jb = json.loads((D / f"{f}.json").read_text()), json.loads((D / f"{f}_fu1.json").read_text())
    assert ja["args"]["grid"] == jb["args"]["grid"] and ja["grid"] == jb["grid"], "grids differ"
    assert ja["seed0"] != jb["seed0"], "same seeds"
    b = b.copy(); b["chain"] = b["chain"] + 1000
    pool = pd.concat([a, b], ignore_index=True)
    n = pool.groupby("start").chain.nunique().to_dict()
    pool.to_csv(O / f"{f}_pooled36.csv", index=False)
    (O / f"{f}_pooled36.json").write_text(json.dumps({"args": ja["args"], "grid": ja["grid"], "pooled_from": [f"{f}.csv", f"{f}_fu1.csv"],
                                                      "seed0": [ja["seed0"], jb["seed0"]], "chains_per_start": n}, indent=1))
    print(f, "pooled chains per start:", n)
PYEOF
export OMP_NUM_THREADS=1
$PY experiments/s5a_analyze_v3.py --label P6_fu1_only_POSTHOC --kappa 0.52 --rperp 1 --out "$OUT" \
    --files "$D/seqbr_Lx20_Ly20_k0.52_r1_T0.15_23-2_fu1.csv" "$D/seqbr_Lx20_Ly30_k0.52_r1_T0.15_23-3_fu1.csv"
$PY experiments/s5a_analyze_v3.py --label P6_pooled36_POSTHOC --kappa 0.52 --rperp 1 --out "$OUT" \
    --files "$OUT/pooled36/seqbr_Lx20_Ly20_k0.52_r1_T0.15_23-2_pooled36.csv" "$OUT/pooled36/seqbr_Lx20_Ly30_k0.52_r1_T0.15_23-3_pooled36.csv"
