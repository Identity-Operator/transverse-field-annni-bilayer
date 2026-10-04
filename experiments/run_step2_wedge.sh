#!/usr/bin/env bash
# Step 2 of the Paper 1 plan: <3>-wedge crossings near the multiphase point, L = 12, T = 0.15.
#
# For each (kappa, r_perp) runs template branches (FM, <2>, <3>; ascending in g) plus one
# descending anneal; the free-energy crossings then come from
#   .venv/bin/python experiments/branch_free_energy_v3.py <csv>
# O(g^2) predictions of the crossing field, F = 6/[(2+r)(3+r)(5+r)]:
#   FM|<3> at g = sqrt(8 (1/2 - kappa) / F),  <3>|<2> at g = sqrt(4 (kappa - 1/2) / F)
#   r=0: kappa 0.47/0.48/0.49 -> 1.10/0.89/0.63;  0.51/0.52/0.53/0.625 -> 0.45/0.63/0.77/1.58
#   r=1: kappa 0.47/0.48/0.49 -> 1.70/1.39/0.98;  0.51/0.52/0.53/0.625 -> 0.69/0.98/1.20/2.45
#
# Each combo writes its own CSV (branches_L12_k<kappa>_r<rperp>.csv), and combos whose CSV
# already has a descending part are skipped, so the script can be re-run after an interruption.
#
# Usage (from vu_work/):  THREADS=10 bash experiments/run_step2_wedge.sh
set -u
cd "$(dirname "$0")/.."
THREADS=${THREADS:-6}
PY=${PY:-.venv/bin/python}
OUT=10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation/branches
KAPPAS="0.49 0.51 0.48 0.52 0.47 0.53 0.625 0.5625 0.75"   # closest to 1/2 first; the last two repeat the laptop run if its files are missing
LOG=$OUT/step2_wedge.log
mkdir -p "$OUT"
echo "== step 2 start $(date) threads=$THREADS" | tee -a "$LOG"
for r in 0 1; do
  if [ "$r" = 0 ]; then GUP=2.0; GTOP=3.5; else GUP=2.8; GTOP=4.5; fi
  for k in $KAPPAS; do
    f=$(ls "$OUT"/branches_L12_k${k}_r${r}*.csv 2>/dev/null | head -1)
    if [ -n "$f" ] && grep -q ",descending," "$f"; then
      echo "skip kappa=$k r=$r (done: $f)" | tee -a "$LOG"; continue
    fi
    echo "-- kappa=$k r=$r start $(date)" | tee -a "$LOG"
    $PY -W ignore experiments/sse_branches_v3.py --part branches --L 12 --T 0.15 \
        --kappa "$k" --rperp "$r" --g-max-up "$GUP" --g-top "$GTOP" --threads "$THREADS" \
        >> "$LOG" 2>&1 || echo "!! kappa=$k r=$r failed $(date)" | tee -a "$LOG"
    echo "-- kappa=$k r=$r end $(date)" | tee -a "$LOG"
  done
done
echo "== step 2 done $(date)" | tee -a "$LOG"
