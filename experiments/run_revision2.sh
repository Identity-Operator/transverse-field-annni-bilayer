#!/usr/bin/env bash
# Paper 1, revision after the internal referee round C2, second driver (replaces the remaining part of
# run_revision1.sh after referee B's round-2 report). Waits for the V3 run of run_revision1.sh (PID given as $1),
# then runs, in priority order:
#   V6   four times the sweeps on 20 x 30 at kappa = 0.53, r = 1 (is the lattice offset kinetic?)
#   V4   <2>, <23>, <3> on 20 x 60 at kappa = 0.53, r = 0 (control)
#   V7   replicate of V1 with fresh seeds (20 x 60, kappa = 0.53, r = 1)          [referee B r2, strongly recommended]
#   V8   <2>, <23>, <3> on 20 x 60 at kappa = 0.52, r = 0                       [referee B r2, strongly recommended]
#   V5   T = 0.10 at kappa = 0.53, r = 1;  A5-A6 anneals ten times slower on 12 x 12 at kappa = 0.75
# Logs to the same revision1.log with the same line format. sse_seq_branches_v3 skips complete cases.
# Usage (from vu_work/):  nohup bash experiments/run_revision2.sh <V3 pid> > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/annni-py311/bin/python}
TH=${THREADS:-9}
LOG=10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation/seq_branches/revision1.log
V3PID=${1:-}
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
G_P5="0.1:0.5:0.1,0.52:0.68:0.01,0.5997:0.5997:1,0.8:0.8:0.1"

if [ -n "$V3PID" ]; then
  while kill -0 "$V3PID" 2>/dev/null; do sleep 30; done
  echo "-- V3 end $(date '+%F %T')" >> "$LOG"
fi
echo "== revision2 start $(date '+%F %T')" >> "$LOG"
seq V6 "2,3;3" 20 30 0.53 1 0.15 "$G_P3" --therm 8000 --meas 12000 --tag-suffix _x4
seq V4 "2;2,3;3" 20 60 0.53 0 0.15 "$G_P1"
seq V7 "2;2,3;3" 20 60 0.53 1 0.15 "$G_P3" --tag-suffix _rep
seq V8 "2;2,3;3" 20 60 0.52 0 0.15 "$G_P5"
seq V5a "2,3;2" 20 20 0.53 1 0.10 "$G_P3"
seq V5b "2,3;3" 20 30 0.53 1 0.10 "$G_P3"
ann A5 12 12 0.75 0 --therm 20000 --meas 30000 --tag-suffix _slow10
ann A6 12 12 0.75 1 --therm 20000 --meas 30000 --tag-suffix _slow10
echo "== revision2 done $(date '+%F %T')" >> "$LOG"
