# paper1.tex: places that the new results make wrong or outdated

This review covers paper1.tex as committed in 9277a0d (unchanged since). Line numbers refer to that version.
The list is ordered from **wrong** to **outdated** to **placeholders that can now be filled**. Items marked
**DECISION** depend on choices that Hoang has not made yet.

Sources (abbreviations):
- **BR** = `sse_validation/branches/branches_L{12,24}_*.csv/json`
- **G5** = `sse_validation/branches_gtop5/`
- **LG** = `sse_validation/lowg/lowg_summary.json`
- **TF** = `sse_validation/tfim2d{,_T0.15}/tfim2d_summary.json`
- **WA** / **WU** = `wedge/wedge_fit_L12_anchored.json` / `wedge_fit_L12.json`
- **A4**, **H4**, **A6** = `perturbation/pt_order4_analysis.json`, `pt_order4_hull.json`, `pt_order6_analysis.json`

All paths are under `vu_work/10_results/bilayer_annni_paper_results/reanalysis_v3/`.

---

## A. Wrong

**A1. l. 308–313 (Sec. III.B): the κ = 0.75, r = 1 prediction and the "indicative at g ≳ 1" sentence.**
- **What is wrong:**
  - "⟨3⟩ never appears at κ = 0.75 in the bilayer" is contradicted by the data. The QMC ⟨3⟩|⟨2⟩ crossing is
    g\* = 2.694(4) (anchored; 2.690(11) unanchored). At L = 12, ⟨3⟩ order sets in near g ≈ 3.4–3.6, where
    S(π/3) on the descending branch rises from 0.02 at g = 3.9 to 0.10 at g = 3.3 (BR, G5). So ⟨3⟩ is stable
    for 2.69 ≲ g ≲ 3.5.
  - The quoted g₃|₂ values (1.12, 1.73, 2.24, 3.46) evaluate Eq. (wedge) at κ = ½. The exact-κ series is much
    closer to the data.
- **Numbers.** Crossing g\* of the ⟨3⟩ and ⟨2⟩ branches (A6 crossings, WA; QMC anchored, L = 12):

  | κ | r | Eq. (wedge) at κ = ½ | exact-κ O(g²) | O(g⁴) | O(g⁶) | QMC (anchored) |
  |---|---|---|---|---|---|---|
  | 0.5625 | 0 | 1.12 | 1.031 | 0.971 | 0.973 | 0.982(4) |
  | 0.5625 | 1 | 1.73 | 1.561 | 1.516 | 1.531 | 1.539(4) |
  | 0.75 | 0 | 2.24 | 1.755 | 1.567 | 1.599 | 1.862(3) |
  | 0.75 | 1 | 3.46 | 2.550 | 2.408 | 2.538 | 2.694(4) |

- **Suggested replacement** for l. 308–313:
  > For fixed $\kappa>1/2$, Eq.~\eqref{eq:wedge} predicts $\langle3\rangle$ above
  > $g_{3|2}\simeq[4(\kappa-1/2)/F(3)]^{1/2}$ to leading order. Away from $\kappa=1/2$ the crossing is better
  > obtained from the series~\eqref{eq:series} evaluated at the actual $\kappa$. Through order $g^6$ this gives
  > $g_{3|2}=0.973$ ($r=0$) and $1.531$ ($r=1$) at $\kappa=0.5625$, and $1.60$ and $2.54$ at $\kappa=0.75$. The
  > expansion parameter is $g/(2+r)$. For $g/(2+r)\lesssim0.5$ the series agrees with the simulations to 1--2\%
  > (Sec.~\ref{sec:results}). Beyond about 0.6 it fails: at $\kappa=0.75$ the simulated crossings lie 16\% and 6\%
  > above the $O(g^6)$ values.

  Also remove the claim that ⟨3⟩ never appears in the bilayer at κ = 0.75. If a sentence is wanted:
  > In the bilayer at $\kappa=0.75$, $\langle3\rangle$ is stable between $g\approx2.7$ and the paramagnetic
  > boundary near $g\approx3.5$.

  The ≈3.5 is an L = 12 onset, not a scaled g_c; state it as such or wait for Sec. V.C.
- **Section order.** Eq. (series) lives in the new Sec. III.D (`audit_9b/sec3_order4_6.tex`). Either move this
  paragraph after III.D or forward-reference it.

**A2. Anneal lock-in: abstract l. 60–61, intro l. 119–121, Methods l. 359–363, conclusion l. 445–447.**
- **What is wrong:** the text says the anneal "selects the modulation that forms at the paramagnetic transition"
  and locks into ⟨3⟩ "even where ⟨2⟩ is stable". The data show more. Every descending anneal ends in ⟨3⟩
  (q\* = π/3, energy equal to the ⟨3⟩ template), 6/6 chains, in all of these runs:
  - all nine κ from 0.47 to 0.75, both r, at L = 12;
  - the L = 24 cases;
  - starts at g = 3.5, 4.5 and 5 (BR, G5).

  This includes the ferromagnetic side: at κ = 0.47, r = 0 the anneal ends at e = −1.4911, close to the ⟨3⟩
  template's −1.4898 and far from FM's −1.5301.
- **Mechanism.** The wedge explains this. ⟨3⟩ is the stable phase at intermediate field on both sides of
  κ = ½, above the first-order crossings g\* with FM and with ⟨2⟩. The anneal therefore enters ⟨3⟩ in equilibrium
  and cannot follow the first-order transition out of it at g\*.
- **Suggested wording.**
  - Abstract l. 60–61:
    > annealing from the paramagnet ends in $\langle3\rangle$ at every $\kappa$ we studied, on both sides of
    > $\kappa=1/2$: it passes through the field range where $\langle3\rangle$ is stable and cannot follow the
    > first-order transition to the ferromagnet or to $\langle2\rangle$ at lower field.
  - Methods l. 362–363: replace "The anneal selects the modulation that forms at the paramagnetic transition"
    with:
    > The anneal passes through the field range in which $\langle3\rangle$ is stable (Sec.~\ref{sec:pt}); below
    > the first-order crossing to $\langle2\rangle$ the lock-out would require restructuring every wall at once.

    Add: "The same happens at every $\kappa$ from 0.47 to 0.75, for both $r$, including $\kappa<1/2$, where the
    ferromagnet is stable."
  - Intro l. 119–121: the same point in one clause.
  - Conclusion l. 445–447: "inherits the modulation of the ordering instability" → "passes through the
    field range where $\langle3\rangle$ is stable".
- **Annealer remark and plan item 7.** Plan item 7 predicted that at r = 1, κ = 0.75 an anneal would reach ⟨2⟩
  because g₃|₂ ≈ 3.46 would lie above g_c. That prediction failed, because the true crossing is 2.69, below
  the ordering field. If the remark is kept, this is its best evidence. **DECISION** (l. 448 \pending).

**A3. l. 367–368: "a separate set is annealed downward from g = 5".**
- Step 2 actually used g_top = 3.5 at r = 0 (5.0 at κ = 0.5625 and 0.75) and g_top = 4.5 at r = 1 (BR json
  args). Extra g_top = 5 reruns exist at κ = 0.5625 and 0.75, r = 1 (G5). All end in ⟨3⟩.
- Suggested wording: "annealed downward from $g=3.5$ ($r=0$) or $4.5$ ($r=1$); starting at $g=5$ changes
  nothing."
- l. 359: "annealed from $g=3$" matches the control run (control_L12_k0.75), so it is fine as is.

## B. Outdated or incomplete

**B1. l. 300–306 (the "Two features" paragraph and its \pending).**
- The "at κ₃|₂ … higher orders decide whether mixed structures such as ⟨2^k3⟩ appear" sentence and the
  \pending are now answered by Sec. III.D. Replace them with:
  > At $\kappa_{3|2}$, every sequence of two- and three-site domains has the same energy at order $g^2$;
  > Sec.~\ref{sec:pt46} shows that the fourth order stabilizes $\langle23\rangle$ there and the sixth order
  > $\langle223\rangle$.
- l. 301–302 (the FM|⟨3⟩ argument) is incomplete. Two-site domains also cost energy at κ_FM|3, and the new
  subsection states it correctly. Suggested wording: "At $\kappa_{\rm FM|3}$ every domain other than three
  sites costs energy at order $g^2$, so the transition … is a simple first-order level crossing."

**B2. l. 289–298: the ⟨3⟩ wedge width is quoted only at O(g²).**
- At O(g⁴) the upper boundary of the ⟨3⟩ wedge is ⟨3⟩|⟨23⟩, not ⟨3⟩|⟨2⟩. Its width becomes
  Δκ₃ = (3/8)F(3) g² + c g⁴, with c = 3599/224000 ≈ 0.0161 (r = 0) and 4579/1935360 ≈ 0.00237 (r = 1).
  Here c = k₁⁻ − k₁^F (H4 window_k1[0]; A4 k1_FM3).
- Suggested addition after l. 298:
  > These are the leading terms; the $g^4$ terms (Sec.~\ref{sec:pt46}) add $0.0161\,g^4$ ($r=0$) and
  > $0.0024\,g^4$ ($r=1$) to the width, and above $\kappa=1/2$ the $\langle3\rangle$ wedge ends at the
  > $\langle23\rangle$ phase rather than at $\langle2\rangle$.

**B3. Abstract l. 55–59, intro l. 110–112, conclusion l. 436–443: the higher-order results are missing.**
- **DECISION** (structure and claims): nothing should be added until the S5a QMC analysis is done.
- Suggested sentences once it is:
  - Abstract, after l. 59:
    > At fourth order, a further structure $\langle23\rangle$ ($\up\up\dn\dn\dn$) opens between $\langle3\rangle$
    > and $\langle2\rangle$, in a window $0.0077\,g^4$ wide at $r=0$ that the interlayer coupling narrows
    > 6.7-fold.
  - Intro l. 112: the same, plus "sixth order adds $\langle223\rangle$".
  - Conclusion l. 442–443: remove "the ⟨3⟩|⟨2⟩ boundary at higher order" from the open questions. Instead say
    that the sequence has been followed to O(g⁶) (⟨3⟩, ⟨23⟩, ⟨223⟩, ⟨2⟩), with further structures open.
- The bilayer narrowing of the ⟨3⟩ wedge (l. 59, 416) now has a measured value: 2.41 ± 0.02 on the ⟨3⟩|⟨2⟩
  side and 2.46 ± 0.02 on the FM|⟨3⟩ side (WA ratio_*_A_Bfix_gmaxlam), against 2.4 predicted. Any quote must
  disclose that the cutoff g\*/(2+r) ≤ 0.5 and the anchor were chosen post hoc. With a common cutoff g ≤ 1.4
  the ⟨3⟩|⟨2⟩ ratio is 2.23 ± 0.02.

**B4. l. 374–376 (the TI anchor).**
- **DECISION: whether to adopt the anchor.** The text takes f_b(g₀) = e_b(g₀) as measured. S5a (and the
  anchored step-2 reanalysis) instead take e_b(g₀) from the series, whose truncation error at g₀ is below 1e-8,
  and check it against the measurement.
- If adopted, add:
  > we take $e_b(g_0)$ from Eq.~\eqref{eq:series} (error below $10^{-8}$ at $g_0=0.05$) and check it against
  > the measured energy; over 72 branches at $L=12$ the deviations have $z$ scores with mean $-0.07$ and
  > standard deviation $1.10$.

  Source for the z-scores: the anchor summary in the 6e session's scratch; copy it into the repo before
  citing it.
- Also add the S5a quadrature and crossing rules (Richardson-corrected TI; crossings interpolated in g²; t-based
  intervals over chains). The step-2 numbers now in WA used trapezoid and linear-in-g interpolation. Re-run
  them with the S5a rules before quoting, or state which rules each number used.

**B5. l. 317–319 (Sec. III.C).**
- "requires flipping a whole row of $L_x$ spins, which enters only at order $g^{L_x}$" is correct for r = 0.
  For r > 0 add "($2L_x$ spins, order $g^{2L_x}$, in the bilayer)".
- Optional: note that the perturbative results refer to $L_x\to\infty$ (the F1 caveat; one clause suffices).

**B6. l. 207–209 (Sec. II.D, observables): optional.**
- If ⟨23⟩ enters the paper, add its saturation value: O(q\* = 2π/5) = 4(3+√5)/25 ≈ 0.838, next to 1 (FM, ⟨2⟩)
  and 8/9 (⟨3⟩). This is exact for ↑↑↓↓↓: |Σ s_j e^{iqj}| = 2·2cos 36°.

## C. Placeholders that can now be filled (from existing outputs)

**C1. l. 364, \pending{Update with the L = 24 low-field check}.** From LG (anneals from g = 3 at r = 0 and
g = 5 at r = 1, to g = 0.05; 6 chains; tolerance 0.005):
- At L = 12, every chain reaches e₀ except at κ = 0.75, where all chains end in ⟨3⟩ (excess 1/6, both r).
- At L = 24, walls are also trapped at κ = 0.25 (3 of 6 chains at r = 0, excess ≤ 0.041; 1 of 6 at r = 1), and
  at κ = 0.75, 1.0 and 1.5 (all or 5 of 6 chains).
- Suggested sentence:
  > On $24\times24$ bilayers, annealing also leaves walls in some chains at $\kappa=0.25$ and in nearly all
  > chains for $\kappa\ge0.75$, so annealing alone is not a valid equilibration protocol at any frustrated
  > $\kappa$ at this size.

**C2. l. 397–398, \pending{Low-field check …; TFIM crossings}.**
- The low-field check is as in C1. Template chains that start in the ground state reproduce the O(g²) slope
  to 0.02–3% (`branches/small_g_check_gmax0.8.json`; also make the r = 1 version, if it exists, citable).
- TFIM (TF):
  - With β = L, the L = 16/24 crossings are g = 3.0385(10) (U₄) and 3.0464(6) (ξ/L), against g_c = 3.04438.
  - At T = 0.15, the 32/48 crossings are 3.0387(8) and 3.0408(8).
  - **Check before writing:** the T = 0.15 value must be compared with the finite-T line of Hesselmann &
    Wessel at T = 0.15, not with the quantum critical point. I have not verified that reference value.

**C3. l. 407–409, 414–416 (Results placeholders).**
- These describe a design that changed. There are no κ cuts at fixed g: step 2 used g-scans at nine κ per r,
  and the fit is to y = Ax + Bx², with x = g\*² and y = |κ − ½| (wedge_fit_v3).
- Replace "comparison with $g_{3|2}$ from Eq. (wedge)" with a comparison against the exact-κ series through
  O(g⁶), as in the A1 table.
- The "$L_y$ not divisible by 4 or 6" check was not run. Either keep it as a to-do or drop it; the L = 24 check
  (five cases, crossings within 0.3%) covers size only for Ly = 24.
- Add a subsection for the S5a ⟨23⟩ test, once the analysis is in.
