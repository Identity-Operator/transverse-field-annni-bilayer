# Independent O(g⁸) audit (audit 9b, 2026-09-29/30)

**Scope.** This is an independent check of Referee A's O(g⁸) results (`manuscript/review/refA_scripts/pt8.py`,
report `referee_C2_A_theory.md`). I wrote all the code here from scratch (`lc8.py`). I read the referee's scripts
only after my numbers were final, to locate the one disagreement. CPU: at most 2 threads.

**Summary.** Almost everything agrees with Referee A. There is **one disagreement: the O(g⁸) margin of ⟨233⟩.** The
referee's number (−1.55e-6 g⁸, sign reversal at g ≈ 1.13 and 1.89) is the chord deviation of ε₈ at fixed κ = ½. It
omits the term k·ε₆′, which the boundary expansion κ = ½ + kg² + … generates at O(g⁸). With that term included the
margin is −9.75e-7 g⁸ at r = 0 and −1.84e-8 g⁸ at r = 1, and the truncated margin changes sign at g ≈ 1.42 and 2.60.
Both values lie beyond the edge-row scales g_c = 1 and 1.83, so no sign change inside the convergence range should be
claimed.

## Method (`lc8.py`)

The code is a linked-cluster Rayleigh–Schrödinger (RS) expansion.
- **Cluster weights.** w(C) = Σ_{S⊆C} (−1)^{|C|−|S|} E[S], where E[S] is the exact RS series with the field on S only.
  E[S] comes from the standard recursion, evaluated in exact Fractions.
- **Cluster size.** The property w_{2n}(C) = 0 for |C| > n is *asserted* for every cluster, not assumed, so O(g⁸)
  needs clusters of up to 4 sites.
- **Enumeration.** Clusters are enumerated on the **infinite** lattice: each connected cluster is attached to its
  smallest site in one unit cell. There is no periodic lattice and no wrap-around.
- **κ-derivatives.** κ enters as a truncated Taylor series (a jet) in δ = κ − κ₀, so all the κ-derivatives needed for
  W₈ come out of the same exact run.

## Checks

- **1D TFIM chain** (`validate_lc8.json`). The series −1, −1/4, −1/64, −1/256, **−25/16384** per spin is reproduced
  exactly.
- **Random 10-spin graphs** (`validate_lc8.json`). On 3 graphs, the sum over clusters of ≤ 4 sites equals the RS on
  the full 2¹⁰ space **exactly** through g⁸.
- **E₈ against ED** (`ed_e8_check.json`). On 4 gapped random graphs the full-space RS E₈ agrees with the ED fit to
  1e-7 – 1.4e-4 relative. The fit uses the absolute residual.
- **Cluster-free lattice check** (`fullrs8_check.json`). RS on periodic 5×Lʸ layers (r = 0) in the space of all states
  within 4 flips is exact for E₈. It reproduces ε₈ for FM, ⟨2⟩, ⟨3⟩ and ⟨23⟩ to 12 digits.
- **Lower orders** (`validate_lc8.json`). ε₂, ε₄ and ε₆ equal the audited O(g⁶) values for 8 structures, exactly for
  ε₂ and ε₄.

## 1. ε₈ per spin at κ = ½ (exact rationals in `g8_results.json`)

| structure | ε₈ (r=0) | ε₈ (r=1) | Referee A |
|---|---|---|---|
| FM | −6.83193e-5 | −2.01966e-6 | −6.83e-5 / −2.02e-6 |
| ⟨2⟩ | −5.04396e-5 | −1.77936e-6 | −5.04e-5 / −1.78e-6 |
| ⟨3⟩ | +6.02676e-4 | −1.43271e-6 | +6.03e-4 / −1.43e-6 |
| ⟨23⟩ | −3.03753e-4 | −5.92247e-6 | −3.04e-4 / −5.92e-6 |
| ⟨223⟩ | −2.73912e-4 | −5.38374e-6 | |
| ⟨233⟩ | +3.46084e-5 | −4.27349e-6 | |
| ⟨2223⟩ | −2.25149e-4 | −4.60158e-6 | |
| ⟨2333⟩ | +1.89536e-4 | −3.49873e-6 | |
| ⟨23223⟩ | −2.86346e-4 | −5.60821e-6 | |

The κ-derivatives ε₂′, ε₂″, ε₂‴, ε₄′, ε₄″ and ε₆′ for every structure are in `g8_results.json` (exact and float).
They enter the O(g⁸) boundary energy as

W₈ = k₂ε₂′ + k k₁ ε₂″ + (k³/6) ε₂‴ + k₁ε₄′ + (k²/2) ε₄″ + k ε₆′ + ε₈.

## 2. O(g⁸) boundaries

The O(g⁶) coefficients (k₂) reproduce the audited values: −3.064e-3, +5.264e-3 and +5.679e-3 at r = 0.

**The ⟨223⟩|⟨2⟩ line** (k₁ = k₁⁺, k₂ = k₂(223|2)):
- **⟨2223⟩ lies below the chord from ⟨2⟩ to ⟨223⟩:** by −8.969e-7 g⁸ at r = 0 and −1.882e-8 g⁸ at r = 1. The referee
  has ≈ −9.0e-7 and −1.9e-8, so we **agree**.
- **The ⟨2223⟩ window:** it opens for k₃ ∈ (1.71951e-3, 1.73767e-3), a width of **1.816e-5 g⁸**, at r = 0. At r = 1
  the width is 3.81e-7 g⁸. The referee has ≈ 1.8e-5, so we **agree**.
- **Four-domain range (range check):** ⟨22223⟩ lies exactly on the segment ⟨2223⟩–⟨2⟩, and ⟨2232223⟩ exactly on
  ⟨223⟩–⟨2223⟩ (height 0 in exact arithmetic). This confirms that W₈ is a sum over 4 consecutive domains.
- **Consequence:** by the elementary-cycle argument on the triple graph, ⟨2223⟩ is the only structure that can open
  on this line at O(g⁸).

**The ⟨23⟩|⟨223⟩ line** (k₂ = k₂(23|223)):
- ⟨23223⟩ lies **exactly** on the segment ⟨23⟩–⟨223⟩ at O(g⁸) (height 0). The referee finds the same, so we **agree**.

**The ⟨3⟩|⟨23⟩ line** (k₁ = k₁⁻, k₂ = k₂(3|23)): **DISAGREEMENT** for ⟨233⟩.

| | r = 0 | r = 1 |
|---|---|---|
| O(g⁶) height | +1.9666e-6 g⁶ | +1.2429e-7 g⁶ |
| O(g⁸) height (this audit) | **−9.747e-7 g⁸** | **−1.844e-8 g⁸** |
| truncated sign reversal | **g ≈ 1.42** | **g ≈ 2.60** |
| Referee A | −1.55e-6 g⁸, g ≈ 1.13 | g ≈ 1.89 |

- **Diagnosis.** The O(g⁸) height splits by term into ε₈ (−1.549e-6 at r = 0; −3.468e-8 at r = 1) and k ε₆′
  (+5.745e-7; +1.624e-8). All other W₈ terms give zero, because they span at most two domains.
  - Dropping k ε₆′ reproduces the referee exactly: −1.549e-6, with reversals at g = 1.127 and 1.893.
  - The referee's d8.py computes the deviation from `series8(nm, half, r)`, that is from ε₈ at fixed κ = ½.
- **Why the term belongs in.** On the ⟨3⟩|⟨23⟩ boundary κ = ½ + kg² + …, so ε₆(κ) contributes k ε₆′ g⁸.
- **Why ⟨2223⟩ and ⟨23223⟩ still agree.** Their chord deviations of three-domain-range terms vanish identically, so
  the omission does not affect those two results.
- **Other sequences on this line.** ⟨2333⟩, measured from the chord ⟨3⟩–⟨23⟩, has height +1.430e-6 g⁶ − 7.089e-7 g⁸ at
  r = 0 and +9.04e-8 g⁶ − 1.34e-8 g⁸ at r = 1.

**Exact-κ checks of other referee statements** (`exactk_g8.json`, `k5625_g1.2_O8.json`):
- At κ = 0.5625, r = 0, the O(g⁸) series has **no** ⟨23⟩|⟨3⟩ or ⟨23⟩|⟨233⟩ crossing for g ≤ 1.6. At O(g⁶) these
  crossings are at 1.056. This **agrees** with the referee's point 2(c).
- At g = 1.2 the O(g⁸) energies relative to ⟨23⟩ are ⟨233⟩ +1.214, ⟨2333⟩ +1.754 and ⟨3⟩ +3.195 (×10⁻⁴). The referee
  has 1.21, 1.75 and 3.20, so we **agree**.
- The exact-κ O(g⁶) → O(g⁸) window edges at the other S5a κ values are listed in `exactk_g8.json`. For example, the
  upper edge at κ = 0.53, r = 0 moves from 0.742 to 0.746, and at κ = 0.545 from 0.899 to 0.918. There are also
  spurious O(g⁸) roots near g ≈ 1.4 at r = 0, which are series artifacts.

## 3. Edge-row critical fields (`ladder_ed.json`)

These come from ED in Z₂ sectors, as crossings of L·gap between consecutive lengths L.

- **Transverse Ising chain (control):** the crossings are 1.00197 (L = 6/8), 1.00091, 1.00049, 1.00030 and 1.00019
  (L = 14/16), which converge to 1.
- **Two-leg ladder, J_leg = J_rung = 1:** the crossings are 1.83475 (4/6), 1.83212 (6/8) and 1.83209 (8/10), so
  **g_c = 1.832**. This confirms the referee's 1.83.
- **In units of λ = g/(2+r):** the edge rows become critical at λ = 0.50 (r = 0) and 0.61 (r = 1).

## Files

| file | contents |
|---|---|
| `lc8.py` | the method |
| `run_g8.py` → `g8_results.json` | all numbers above |
| `validate_lc8.py` / `ed_e8_check.py` / `fullrs8.py` | the checks |
| `ladder_ed.py` | edge-row critical fields |
| `exactk_g8.py` | exact-κ window edges |
