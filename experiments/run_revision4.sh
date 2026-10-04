#!/usr/bin/env bash
# Paper 1, final referee round C3 (referee B, point 1): 2 x 2 design, lattice x sweeps, at kappa = 0.52, r = 1.
# Existing: 20 x 30 and 20 x 60 at the standard sweeps (orig + fu1; V2 + V2rep). New: both lattices with four times
# the sweeps and fresh seeds, <23> and <3> only (Delta F_3 is the quantity with the lattice shift):
#   V9a  20 x 30, 8000 + 12000 sweeps per field
#   V9b  20 x 60, 8000 + 12000 sweeps per field
# Logs to the same revision1.log with the same line format.
# Usage (from vu_work/):  nohup bash experiments/run_revision4.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/annni-py311/bin/python}
TH=${THREADS:-10}
LOG=10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation/seq_branches/revision1.log
seq() {   # seq <label> <seqs> <Lx> <Ly> <kappa> <r> <T> <grid> [extra args]
  local lab=$1 s=$2 lx=$3 ly=$4 k=$5 r=$6 t=$7 g=$8; shift 8
  echo "-- $lab: seqs=$s Lx=$lx Ly=$ly kappa=$k r=$r T=$t $* start $(date '+%F %T')" >> "$LOG"
  $PY -W ignore experiments/sse_seq_branches_v3.py --seqs "$s" --Lx "$lx" --Ly "$ly" --kappa "$k" --rperp "$r" \
      --T "$t" --grid "$g" --chains 12 --threads "$TH" "$@" >> "$LOG" 2>&1 || echo "!! $lab failed $(date '+%F %T')" >> "$LOG"
  echo "-- $lab end $(date '+%F %T')" >> "$LOG"
}
G_P6="0.1:0.8:0.1,0.86:1.02:0.01,0.9351:0.9351:1,1.1:1.2:0.1"
echo "== revision4 start $(date '+%F %T')" >> "$LOG"
seq V9a "2,3;3" 20 30 0.52 1 0.15 "$G_P6" --therm 8000 --meas 12000 --tag-suffix _x4
seq V9b "2,3;3" 20 60 0.52 1 0.15 "$G_P6" --therm 8000 --meas 12000 --tag-suffix _x4
echo "== revision4 done $(date '+%F %T')" >> "$LOG"
