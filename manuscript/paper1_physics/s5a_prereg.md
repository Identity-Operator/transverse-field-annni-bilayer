# S5a pre-registration: QMC test of the ⟨23⟩ phase

Written 2026-09-29 and revised after the C1 design review (phase-classification-9b) and the results audit (phase-classification-12). No S5a production run had started at that point; the change log records every revision. **This version is final at launch.** Any later change goes into the change log with its reason, before the analysis is run.

- Code: `experiments/sse_seq_branches_v3.py`, `experiments/run_s5a_23.sh`.
- Analysis: `experiments/s5a_analyze_v3.py`, written independently by phase-classification-12.
- Theory: `experiments/pt_order4_v3.py`, `pt_order4_hull_v3.py`, `pt_order6_v3.py`.
- Proof note: `10_results/bilayer_annni_paper_results/reanalysis_v3/perturbation/audit_9b/proof_note_hull.tex`.

## Hypothesis and predictions

- **H.** Near κ = ½ the transverse field stabilises ⟨23⟩ (↑↑↓↓↓) between ⟨3⟩ and ⟨2⟩.
- **Theory (verified independently):**
  - At O(g⁴), only ⟨2⟩, ⟨3⟩ and ⟨23⟩ can be stable, for sequences of any length. The ⟨23⟩ κ-window is 0.00768 g⁴ wide at r = 0 and 0.00115 g⁴ at r = 1.
  - At O(g⁶), ⟨223⟩ opens between ⟨23⟩ and ⟨2⟩ (4.1e-4 g⁶, below QMC resolution). ⟨233⟩ does not open, but only by +2.0e-6 g⁶ at r = 0; O(g⁸) may flip that.
- **Predictions at exact κ, O(g⁶).** The centre g_c is the ⟨3⟩|⟨2⟩ crossing, so there F⟨23⟩ − F⟨2⟩ = F⟨23⟩ − F⟨3⟩.

| case | r | κ | g_c | ⟨23⟩ window | F⟨23⟩ − F⟨2,3⟩ at g_c | F⟨223⟩ − F⟨23⟩ | F⟨233⟩ − F⟨23⟩ | Δκ/g_c⁴ | λ | expected z (T1) |
|---|---|---|---|---|---|---|---|---|---|---|
| P5 | 0 | 0.52 | 0.5997 | 0.5923–0.6123 | −2.17e-4 | +6.0e-5 | +8.1e-5 | 0.0114 | 0.30 | ≈3.5 |
| P1 | 0 | 0.53 | 0.7179 | 0.7049–0.7418 | −4.98e-4 | +1.37e-4 | +1.87e-4 | 0.0134 | 0.36 | ≈6.5 |
| P4 | 0 | 0.545 | 0.8525 | 0.8302–0.8990 | −1.14e-3 | +3.12e-4 | +4.29e-4 | 0.0166 | 0.43 | ≥4 |
| P2 | 0 | 0.5625 | 0.9733 | 0.9396–1.0561 | −2.22e-3 | +6.00e-4 | +8.34e-4 | 0.0212 | 0.49 | ≥4 |
| P6 | 1 | 0.52 | 0.9351 | 0.9254–0.9512 | −1.75e-4 | +4.9e-5 | +6.6e-5 | 0.00158 | 0.31 | ≈3.5 |
| P3 | 1 | 0.53 | 1.1222 | 1.1051–1.1525 | −3.97e-4 | +1.09e-4 | +1.49e-4 | 0.00181 | 0.37 | ≥4 |
| P7 | 1 | 0.545 | 1.3369 | 1.3070–1.3951 | −8.98e-4 | +2.44e-4 | +3.38e-4 | 0.00220 | 0.45 | ≥4 |

- **Expected z** comes from per-chain spreads in the exploratory and step-2 data with 12 chains on the planned lattices (9b's estimate). It is uncertain by about ±30%.
- **Δκ/g_c⁴** is the O(g⁶) window in κ, converted from the g-window with the slope dκ*/dg of the ⟨3⟩|⟨2⟩ line. The O(g⁶)-corrected r0/r1 ratios at the actual g_c are:
  - P1/P3: 7.38
  - P4/P7: 7.55
  - P5/P6: 7.22
  - (6.7 at O(g⁴).)
- **Series convergence** is slow for the ⟨23⟩ stabilisation at r = 0 (the g⁶ term is 0.45× the g⁴ term at κ = 0.53). Only λ = g*/(2+r) ≤ 0.5 counts as a quantitative comparison. P2 sits at the edge.

## Design

- **Protocol (step 2):** ascending template branches, prefill 200, 2000 + 3000 sweeps per g, T = 0.15, **12 chains** per structure. Bilayer Lx × Ly, parallel stacking.
- **Lattices:** each T1 comparison is made on a single lattice.
  - ⟨23⟩ + ⟨2⟩ on 20 × 20.
  - ⟨23⟩ + ⟨3⟩ on 20 × 30 (N = 1200; q = 0.4π and π/3 are both on the grid).
  - This gives a built-in ⟨23⟩ size check: Ly = 20 vs 30.
- **Competitors:**
  - ⟨233⟩ on 20 × 32 at P1, P2, P3, P4.
  - ⟨223⟩ on 20 × 28 at P1, P2, P3.
  - ⟨2333⟩ on 20 × 22 at P2 only (exploratory).
  - These compare across lattices with ⟨23⟩. That is justified by rigid, gapped walls (exponentially small finite-size terms), by L12 vs L24 crossings agreeing within 0.33%, and by ⟨2⟩ free energies agreeing across L = 12, 16, 20 within 7e-5.
- **Grids** (`run_s5a_23.sh`):
  - 0.1 steps below the window;
  - a fine band (0.02; 0.01 at P5 and P6) across the O(g⁶) window with ≥0.06 margin (P2 up to 1.30, P4 up to 1.04);
  - **g_c as an exact grid point**;
  - 1–2 points beyond (P2: 1.4, 1.5; P4: 1.1, 1.2).
- **Order:**
  1. Primaries P1, P3, P2, P4, P7, P5, P6.
  2. Competitors C2 (P2: ⟨233⟩, ⟨223⟩, ⟨2333⟩), C1, C4, C3.
  3. T = 0.10 check K2 at P1.
  4. Optional L = 40 × 40 check K1 at P1.
- **Threads:** 8 for 24-chain calls, 6 for 12-chain calls. That leaves room for the other sessions.
- **Estimated wall time:** ≈ 8.5 h for the primaries, ≈ 2 h for competitors, ≈ 1 h for K2.

## Analysis (fixed)

1. **Free energies.**
   - Anchor: f_b(g) = e_PT(g0) − ∫ m_x dg per chain, with g0 = 0.05. e_PT is the O(g⁴) energy of the template (series error < 1e-8).
   - Circularity: the anchor carries e0 + e2 g0², and the tested O(g⁴) chord effect contributes only ≈ 1e-8 to it (g0⁴ × 1.2e-3). g_c and the κ values are fixed by theory, not by data.
   - Quadrature: **Richardson-corrected** on the 0.1-step coarse segment [0.1, g_f], where g_f is the last 0.1-grid point before the fine band. The plain trapezoid is reported too; its bias at g_c (1–5% of the signal, always in ⟨23⟩'s favour) is removed by the correction. Rule:
     - m (the number of 0.1 intervals in [0.1, g_f]) even: I_R = I_h + (I_h − I_2h)/3 over the whole segment.
     - m odd (P1, P2, P3, P6): Richardson over [0.1, g_f − 0.3] (m − 3 intervals, even), plus Simpson's 3/8 rule over [g_f − 0.3, g_f]. Both are O(h⁴).
     - Trapezoid on the first interval 0.05–0.1 (bias ~1e-8), on the fine band, on the step into the fine band, and above it.
   - Cuts: a chain is cut at its first q* change.
   - Also reported: the unanchored variant, and the measured-vs-PT z at g0 for every branch.
2. **Uncertainties:** jackknife SE over chains. Intervals use Welch-type t-quantiles (Welch–Satterthwaite dof for differences). Crossing CIs: jackknife with t_{n−1}.
3. **Crossings (window edges):** a cubic-spline root of ΔF(g), or linear interpolation in u = g² between bracketing points (phase-classification-12's check shows the two agree within ≤ 4e-4 for g* ≤ 1.6). The spline is primary.
4. **Diagnostics:** chains cut before g_c are counted and reported.

## Tests and thresholds

- **T1, existence (primary cases), intersection–union test at g_c:**
  - F⟨23⟩ − F⟨2⟩ < 0 on 20 × 20 AND F⟨23⟩ − F⟨3⟩ < 0 on 20 × 30, each one-sided at > 2σ (Welch t), with all chains uncut at g_c.
  - **Headline:** T1 passes in every primary case whose expected z ≥ 3, i.e. all seven.
  - A T1 failure in a case with λ ≤ 0.4 (P1, P3, P5, P6) is a contradiction with the theory. It is reported and investigated (lattice, T, protocol).
- **T2, edges:**
  - ⟨2⟩|⟨23⟩ is measured on 20 × 20 and ⟨23⟩|⟨3⟩ on 20 × 30, with 68% CIs.
  - Compared with exact-κ O(g⁴) and O(g⁶). Reported, not pass/fail.
  - Expectation: agreement with O(g⁶) within 2σ for λ ≤ 0.4, with growing deviations beyond.
- **T3, scaling:**
  - The measured κ-width is Δκ = Δg · dκ*/dg, with dκ*/dg taken from the **QMC** ⟨3⟩|⟨2⟩ crossings along the κ ladder (0.52, 0.53, 0.545, 0.5625 at each r). Theory serves as a cross-check.
  - Δκ/g_c⁴ is compared with the table above.
- **T4, bilayer narrowing:** the ratio of T3 widths for P1/P3, P4/P7 and P5/P6, against the O(g⁶) predictions 7.4, 7.6 and 7.2.
- **T5, competitors at g_c** (a consistency check):
  - F⟨223⟩ − F⟨23⟩ and F⟨233⟩ − F⟨23⟩, with the pulls against the predicted values in the table.
  - The SE on 20 × 28 and 20 × 32 is ≈ 1e-4, so T5 has real power mainly at P2 and P4.
  - **Stop rule:** if a competitor is below ⟨23⟩ at g_c by > 3σ in a primary case, stop, and make no ⟨23⟩ claim until it is understood.
- **T6, size:**
  - ⟨23⟩ on 20 × 20 vs 20 × 30 at every grid point (χ² reported).
  - ⟨3⟩ on 20 × 30 vs the step-2 L = 24 at P1 and P3.
  - Optionally, ⟨23⟩ and ⟨2⟩ on 40 × 40 at P1 (K1).
- **T7, temperature:** the T = 0.10 window edges at P1 agree with T = 0.15 within 2σ.
- **T8, next-step search** (exploratory; motivated by the audit finding F⟨233⟩ − F⟨23⟩ = −10(3)e-4 at g = 1.2, κ = 0.5625):
  - Above the ⟨23⟩ window at P2 and P4, report the crossings ⟨23⟩|⟨233⟩ and ⟨233⟩|⟨3⟩ (and ⟨2333⟩ at P2) with CIs.
  - A ⟨233⟩ interval that beats both ⟨23⟩ and ⟨3⟩ at > 2σ would be reported as "QMC indicates ⟨233⟩ beyond the range of the O(g⁶) series" (λ ≳ 0.55). It is not a test of the O(g⁶) prediction, whose margin is O(g⁸)-sensitive.

## What each outcome allows us to claim

- **T1 passes in all primaries, T5 is consistent, T6 passes:** "unbiased QMC confirms the ⟨23⟩ phase predicted at O(g⁴), at seven points for r = 0 and 1". Edges are reported against O(g⁴)/O(g⁶), and the bilayer narrowing per T4.
- **T1 fails at λ ≤ 0.4:** a contradiction; report it and investigate.
- **The stop rule is triggered:** no ⟨23⟩ claim beyond the specific pairwise comparisons; investigate.
- **In all cases:**
  - ⟨223⟩ stays a theory statement (below QMC resolution).
  - ⟨233⟩ is reported per T8.
  - No staircase claim is made.

## Known limits

- Only straight-wall template structures are compared. Floating or incommensurate states are not tested (the same caveat as for ⟨3⟩).
- There is one main temperature, plus the T = 0.10 check.
- The anchor, the λ ≤ 0.5 cutoff and the interpolation fix were chosen after step 2 (disclosed in the paper). S5a fixes them in advance.
- Finite-L equilibrium between degenerate wall arrangements is not sampled (walls are rigid). The per-structure free energies are the thermodynamic-limit sector quantities; the entropy of the arrangements per spin is ~1/Lx → 0.

## Change log (pre-launch revisions, 2026-09-29)

- **C1-1:** ⟨3⟩ moved from L = 12 to 20 × 30, shared with ⟨23⟩. Both T1 comparisons are now on a single lattice. Reason: L = 12 ⟨3⟩ noise made T1 underpowered at P1, P5 and P6.
- **C1-2:** grids extended: P2 fine band to 1.30, then 1.4 and 1.5; P4 to 1.04, then 1.1 and 1.2.
- **C1-3:** the theoretical g_c added as an exact grid point in every case.
- **C1-4:** T1 changed to an intersection–union test with Welch t, uncut chains at g_c, and pre-registered expected z. "≥ 2 of 3 at r = 1" deleted.
- **C1-5:** T5 predictions at g_c pre-stated. Competitors moved to 20 × 28 and 20 × 32. T5 labelled a consistency check.
- **C1-6:** Richardson-corrected quadrature made primary; the trapezoid is reported.
- **C1-7:** S1 and S2 (κ = 0.51) dropped as unpowerable. S3 dropped for time.
- **C1-8–11:** circularity note; T3 uses QMC dκ*/dg; T4 uses O(g⁶) ratios at the actual g_c; K1 made optional; threads set to 8 and 6.
- **Audit R4 (phase-classification-12):** exploratory ⟨233⟩ < ⟨23⟩ at g = 1.2, κ = 0.5625 (3.9σ). Added ⟨233⟩ at P4 and ⟨2333⟩ at P2, plus test T8.
- **Audit (12):** crossings by spline or linear-in-g² interpolation, not linear-in-g. CIs t-based.
- **Analysis clarifications (phase-classification-12's s5a_analyze_v3.py, frozen 11:55:40; confirmed by the author ~12:10).** At confirmation only P1a (20×20) existed, and it had been used only as a format check. No T1 verdict was possible without P1b. These interpret the text above; they do not change it.
  - (a) ">2σ one-sided" means a one-sided Welch-t probability < 0.02275.
  - (b) T5 and the stop rule compare the competitors with ⟨23⟩ on 20×30. The 20×20 difference is also reported.
  - (c) T8 considers ⟨233⟩/⟨2333⟩ against ⟨23⟩ and ⟨3⟩ on 20×30, at g ≥ g_c only.
  - (d) T3 ladder: a weighted fit of κ − ½ = a g*² + b g*⁴ to the QMC ⟨3⟩|⟨2⟩ crossings. There are 4 points at r = 0 and 3 at r = 1 (S3 was dropped).
  - (e) Window edges use the first + → − sign change. The number of sign changes is reported and multiple changes are flagged.
  - (f) m is taken from the specified 0.1-step grid piece (P7: m = 10).
  - (g) T6 passes if max|z| < 3 over the grid points. χ²/pt is reported. Per-point z of cumulative integrals are correlated.
  - (h) g_c must be an exact grid point (tolerance 6e-5).
- **Primary T1 outcome (recorded 2026-09-29 ~17:00, before any follow-up): 6 of 7 pass.**
  - P6 (κ = 0.52, r = 1) FAILS: F⟨23⟩ − F⟨2⟩ = −0.99(0.54)e-4 at g_c = 0.9351 (1.8σ; bar 2.07), predicted −1.75e-4. F⟨23⟩ − F⟨3⟩ = −2.47(0.58)e-4 (4.2σ).
  - Investigation, as required for λ ≤ 0.4: the value agrees with the prediction within 1.4σ. The edge analysis finds a window of 0.9297(30)–0.9577(52), with κ-width/g⁴ = 1.72(37)e-3 against 1.58e-3 predicted, shifted about 0.004 above theory, so g_c sits near its lower edge.
  - Classification: insufficient power, not a contradiction with theory. The pre-registered verdict stands as a FAIL and is reported as such.
- **Declared post-hoc follow-up at P6 (user-approved ~17:00).**
  - Design: the same grid and protocol, 24 additional chains per structure, new seeds (--tag-suffix _fu1).
  - Queued after the pre-registered batch.
  - Reported separately as a post-hoc extension, never merged into the pre-registered T1.
  - Code change: optional --tag-suffix in sse_seq_branches_v3.py. The default is unchanged, so the pre-registered names and seeds are unaffected.
  - Outcome (2026-09-29 21:58; frozen script, reanalysis_v3/s5a_analysis/posthoc_P6/). The pre-registered P6 verdict stays FAIL.
    - Follow-up alone (24 chains): A −2.13(0.46)e-4 (t −4.6), B −2.90(0.39)e-4 (t −7.5).
    - Pooled (36 chains): A −1.75(0.36)e-4 (t −4.9, equal to the prediction), B −2.76(0.32)e-4 (t −8.6).
    - Pooled window 0.9254(20)–0.9606(29): the lower edge equals O(g⁶), the upper edge is +3.2σ.
    - Original vs follow-up A: 1.6σ apart.
- **Bug fix in s5a_analyze_v3.py (phase-classification-12), after P1a existed, ~12:40.**
  - The Richardson/3-8 panels could read m_x from up to 0.3 in g beyond a chain's q*-cut. A cut chain is now integrated over its valid prefix only.
  - No effect on uncut chains: the P1a output is byte-identical. T1 requires uncut chains at g_c anyway.
  - The step-2 re-analysis changes by ≤ 4.9e-4 in F near cuts only.
- **C1 final (9b), 2026-09-29 11:48, after the launch at 11:46 but before any output file existed:** the Richardson rule for an odd number of coarse intervals (Richardson on an even sub-segment plus Simpson 3/8 on the last three intervals). The user approved the revised budget (≈ 11.5 h at 8 threads) before the launch.
