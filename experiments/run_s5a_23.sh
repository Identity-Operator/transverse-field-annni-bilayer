#!/usr/bin/env bash
# Paper 1, stage S5a: pre-registered QMC test of the <23> phase (design: manuscript/paper1_physics/s5a_prereg.md,
# revised after the C1 review). Runs in priority order. Every call writes its own CSV and is skipped if complete,
# so the batch can be re-run after an interruption.
# Usage (from vu_work/):  PY=$HOME/.venvs/annni-py311/bin/python bash experiments/run_s5a_23.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/annni-py311/bin/python}
LOG=10_results/bilayer_annni_paper_results/reanalysis_v3/sse_validation/seq_branches/s5a.log
mkdir -p "$(dirname "$LOG")"
run() {   # run <label> <seqs> <Lx> <Ly> <kappa> <r> <T> <grid> <threads>
  echo "-- $1: seqs=$2 Lx=$3 Ly=$4 kappa=$5 r=$6 T=$7 start $(date '+%F %T')" | tee -a "$LOG"
  $PY -W ignore experiments/sse_seq_branches_v3.py --seqs "$2" --Lx "$3" --Ly "$4" --kappa "$5" --rperp "$6" \
      --T "$7" --grid "$8" --chains 12 --threads "$9" >> "$LOG" 2>&1 || echo "!! $1 failed $(date '+%F %T')" | tee -a "$LOG"
  echo "-- $1 end $(date '+%F %T')" | tee -a "$LOG"
}
# Grids: 0.1 steps, a fine band across the O(g^6) <23> window (+-0.06 or more), the theory g_c as an exact point,
# and 1-2 points beyond. g_c = O(g^6) exact-kappa <3>|<2> crossing (s5a_prereg.md, predictions table).
G_P1="0.1:0.6:0.1,0.62:0.80:0.02,0.7179:0.7179:1,0.9:1.0:0.1"      # r0 k .53    window 0.705-0.742
G_P2="0.1:0.8:0.1,0.82:1.30:0.02,0.9733:0.9733:1,1.4:1.5:0.1"      # r0 k .5625  window 0.940-1.056 (probe to 1.15)
G_P3="0.1:1.0:0.1,1.02:1.24:0.02,1.1222:1.1222:1,1.3:1.4:0.1"      # r1 k .53    window 1.105-1.153
G_P4="0.1:0.7:0.1,0.72:1.04:0.02,0.8525:0.8525:1,1.1:1.2:0.1"      # r0 k .545   window 0.830-0.899
G_P5="0.1:0.5:0.1,0.52:0.68:0.01,0.5997:0.5997:1,0.8:0.8:0.1"      # r0 k .52    window 0.592-0.612
G_P6="0.1:0.8:0.1,0.86:1.02:0.01,0.9351:0.9351:1,1.1:1.2:0.1"      # r1 k .52    window 0.925-0.951
G_P7="0.1:1.1:0.1,1.20:1.50:0.02,1.3369:1.3369:1,1.6:1.6:0.1"      # r1 k .545   window 1.307-1.395

echo "== S5a start $(date '+%F %T')" | tee -a "$LOG"
# Primary (T1-T4, T6): <23>+<2> on 20x20 and <23>+<3> on 20x30 (both T1 comparisons on one lattice each)
for spec in "P1 0.53 0 $G_P1" "P3 0.53 1 $G_P3" "P2 0.5625 0 $G_P2" "P4 0.545 0 $G_P4" \
            "P7 0.545 1 $G_P7" "P5 0.52 0 $G_P5" "P6 0.52 1 $G_P6"; do
  set -- $spec
  run "$1a" "2,3;2" 20 20 "$2" "$3" 0.15 "$4" 8
  run "$1b" "2,3;3" 20 30 "$2" "$3" 0.15 "$4" 8
done
# Competitors (T5) and the next-step search (T8): <233> on 20x32, <223> on 20x28, <2333> on 20x22 (P2 only)
run C2a "2,3,3" 20 32 0.5625 0 0.15 "$G_P2" 6
run C2b "2,2,3" 20 28 0.5625 0 0.15 "$G_P2" 6
run C2c "2,3,3,3" 20 22 0.5625 0 0.15 "$G_P2" 6
run C1a "2,3,3" 20 32 0.53 0 0.15 "$G_P1" 6
run C1b "2,2,3" 20 28 0.53 0 0.15 "$G_P1" 6
run C4a "2,3,3" 20 32 0.545 0 0.15 "$G_P4" 6
run C3a "2,3,3" 20 32 0.53 1 0.15 "$G_P3" 6
run C3b "2,2,3" 20 28 0.53 1 0.15 "$G_P3" 6
# Temperature check (T7) at P1
run K2a "2,3;2" 20 20 0.53 0 0.10 "$G_P1" 8
run K2b "2,3;3" 20 30 0.53 0 0.10 "$G_P1" 8
# Optional, only if time allows: square L = 40 size check at P1
run K1a "2,3;2" 40 40 0.53 0 0.15 "$G_P1" 8
echo "== S5a done $(date '+%F %T')" | tee -a "$LOG"
