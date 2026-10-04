#!/usr/bin/env bash
# Paper 1, revision after the internal referee round C2 (manuscript/review/referee_C2_{A_theory,B_numerics}.md).
# Runs in priority order; sse_seq_branches_v3 skips a case whose CSV is already complete.
#   V1-V2  <2>, <23>, <3> on ONE lattice with Ly = 60 (a multiple of 4, 5, 6) at kappa = 0.53, 0.52, r = 1:
#          tests the lattice-dependent offset (B2, A5)
#   A1-A4  descending anneals on 12 x 60 at kappa = 0.625, 0.75, r = 0, 1: which wavevector freezes (B3)
#   V3     40 x 40 at kappa = 0.53, r = 1 (B2 iii);  V4  Ly = 60 control at kappa = 0.53, r = 0
#   V5     T = 0.10 at kappa = 0.53, r = 1 (B2 iv); A5-A6 anneals ten times slower on 12 x 12 at kappa = 0.75 (B3)
#   V6     four times the sweeps per field at kappa = 0.53, r = 1 on 20 x 30 (B1 d)
# Usage (from vu_work/):  nohup bash experiments/run_revision1.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/annni-py311/bin/python}
TH=${THREADS:-9}
LOG=10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation/seq_branches/revision1.log
seq() {   # seq <label> <seqs> <Lx> <Ly> <kappa> <r> <T> <grid> [extra args]
  local lab=$1 s=$2 lx=$3 ly=$4 k=$5 r=$6 t=$7 g=$8; shift 8
  echo "-- $lab: seqs=$s Lx=$lx Ly=$ly kappa=$k r=$r T=$t $* start $(date '+%F %T')" >> "$LOG"
  $PY -W ignore experiments/sse_seq_branches_v3.py --seqs "$s" --Lx "$lx" --Ly "$ly" --kappa "$k" --rperp "$r" \
      --T "$t" --grid "$g" --chains 12 --threads "$TH" "$@" >> "$LOG" 2>&1 || echo "!! $lab failed $(date '+%F %T')" >> "$LOG"
  echo "-- $lab end $(date '+%F %T')" >> "$LOG"
}
ann() {   # ann <label> <Lx> <Ly> <kappa> <r> [extra args]
  local lab=$1 lx=$2 ly=$3 k=$4 r=$5; shift 5
  echo "-- $lab: anneal Lx=$lx Ly=$ly kappa=$k r=$r $* start $(date '+%F %T')" >> "$LOG"
  $PY -W ignore experiments/sse_anneal_rect_v3.py --Lx "$lx" --Ly "$ly" --kappa "$k" --rperp "$r" --threads "$TH" "$@" \
      >> "$LOG" 2>&1 || echo "!! $lab failed $(date '+%F %T')" >> "$LOG"
  echo "-- $lab end $(date '+%F %T')" >> "$LOG"
}
G_P1="0.1:0.6:0.1,0.62:0.80:0.02,0.7179:0.7179:1,0.9:1.0:0.1"
G_P3="0.1:1.0:0.1,1.02:1.24:0.02,1.1222:1.1222:1,1.3:1.4:0.1"
G_P6="0.1:0.8:0.1,0.86:1.02:0.01,0.9351:0.9351:1,1.1:1.2:0.1"

echo "== revision1 start $(date '+%F %T')" >> "$LOG"
seq V1 "2;2,3;3" 20 60 0.53 1 0.15 "$G_P3"
seq V2 "2;2,3;3" 20 60 0.52 1 0.15 "$G_P6"
ann A1 12 60 0.75 0
ann A2 12 60 0.75 1
ann A3 12 60 0.625 0
ann A4 12 60 0.625 1
seq V3 "2,3;2" 40 40 0.53 1 0.15 "$G_P3"
seq V4 "2;2,3;3" 20 60 0.53 0 0.15 "$G_P1"
seq V5a "2,3;2" 20 20 0.53 1 0.10 "$G_P3"
seq V5b "2,3;3" 20 30 0.53 1 0.10 "$G_P3"
ann A5 12 12 0.75 0 --therm 20000 --meas 30000 --tag-suffix _slow10
ann A6 12 12 0.75 1 --therm 20000 --meas 30000 --tag-suffix _slow10
seq V6 "2,3;3" 20 30 0.53 1 0.15 "$G_P3" --therm 8000 --meas 12000 --tag-suffix _x4
echo "== revision1 done $(date '+%F %T')" >> "$LOG"
