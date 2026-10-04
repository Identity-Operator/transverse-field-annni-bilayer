import sys, time
from fractions import Fraction as Fr
_ROOT = __import__("pathlib").Path(__file__).resolve().parents[5]   # repository root (vu_work)
sys.path[:0] = [str(_ROOT / "experiments"), str(_ROOT)]
import pt_order6_v3 as P6
from indep_e6 import per_spin
cases = [(None, 0.5, 0), ((2,), 0.5, 0), ((3,), 0.5, 0), ((2, 3), 0.5, 0), ((2, 2, 3), 0.5, 0), ((2, 3, 3), 0.5, 0),
         ((2, 3), 0.53, 0), ((2, 4), 0.5, 0),
         (None, 0.5, 1), ((2,), 0.5, 1), ((3,), 0.5, 1), ((2, 3), 0.5, 1), ((2, 3), 0.55, 1)]
for seq, k, r in cases:
    t0 = time.time(); mine, N = per_spin(list(seq) if seq else None, k, r); t1 = time.time()
    theirs = P6.per_spin_series(seq, Fr(k).limit_denominator(1000), Fr(r)); t2 = time.time()
    print(f"{str(seq):10s} k={k} r={r} N={N:3d}  e2 {mine[2]:+.12f}/{float(theirs[1]):+.12f}  e4 {mine[4]:+.12f}/{float(theirs[2]):+.12f}"
          f"  e6 mine {mine[6]:+.12e} theirs {float(theirs[3]):+.12e}  diff {mine[6]-float(theirs[3]):+.1e}  ({t1-t0:.0f}s/{t2-t1:.0f}s)", flush=True)
