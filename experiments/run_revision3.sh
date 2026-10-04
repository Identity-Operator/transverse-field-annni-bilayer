#!/usr/bin/env bash
# Paper 1, revision after the internal referee round C2, third driver (replaces the remaining part of
# run_revision2.sh after V7 showed that the kappa = 0.53, r = 1 lattice offset does not reproduce).
# Waits for the V5b run of run_revision2.sh (PID given as $1), then runs:
#   V2rep  replicate of V2 with fresh seeds (20 x 60, kappa = 0.52, r = 1): is the P6 offset real?
#   A5-A6  anneals ten times slower on 12 x 12 at kappa = 0.75
# Logs to the same revision1.log with the same line format. sse_seq_branches_v3 skips complete cases.
# Usage (from vu_work/):  nohup bash experiments/run_revision3.sh <V5b pid> > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/annni-py311/bin/python}
TH=${THREADS:-9}
LOG=10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation/seq_branches/revision1.log
V5PID=${1:-}
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
G_P6="0.1:0.8:0.1,0.86:1.02:0.01,0.9351:0.9351:1,1.1:1.2:0.1"

if [ -n "$V5PID" ]; then
  while kill -0 "$V5PID" 2>/dev/null; do sleep 30; done
  echo "-- V5b end $(date '+%F %T')" >> "$LOG"
fi
echo "== revision3 start $(date '+%F %T')" >> "$LOG"
seq V2rep "2;2,3;3" 20 60 0.52 1 0.15 "$G_P6" --tag-suffix _rep
ann A5 12 12 0.75 0 --therm 20000 --meas 30000 --tag-suffix _slow10
ann A6 12 12 0.75 1 --therm 20000 --meas 30000 --tag-suffix _slow10
echo "== revision3 done $(date '+%F %T')" >> "$LOG"
