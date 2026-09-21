# Form factors and the Bc→J/ψ MC reweighting to Harrison-2024 lattice QCD

*Reference note for the Run-3 R(J/ψ) analysis. Part I is a self-contained
introduction to semileptonic form factors and their parametrisations, aimed at a
HEP physicist who does not work on form factors day-to-day. Part II documents the
final, validated procedure we use to reweight the signal MC from its generated
Kiselev form factors to the Harrison-2024 lattice-QCD form factors, with
systematic uncertainties.*

---

# Part I — What form factors are and how we parametrise them

## 1. Semileptonic decays and the origin of form factors

The signal decay is
$$
B_c^+ \to J/\psi\,\ell^+\nu_\ell,\qquad \ell=\mu,\tau,
$$
a $b\to c\,\ell\nu$ charged-current transition. Like every semileptonic decay it
**factorises** into two pieces that do not talk to each other:

- a **leptonic current**, $\bar u_\ell\gamma^\mu(1-\gamma_5)v_\nu$, which is pure
  electroweak physics — calculable exactly, and carrying the CKM factor $V_{cb}$ and
  the electroweak correction $\eta_{\rm EW}$;
- a **hadronic current**, $\langle J/\psi|\,\bar c\,\gamma^\mu(1-\gamma_5)\,b\,|B_c\rangle$,
  the amplitude for the $b\bar c$ bound state to turn into the $c\bar c$ bound state
  under the quark current. This is genuinely non-perturbative QCD: there is no Feynman
  diagram for it.

Because the hadronic current is a Lorentz vector built from only two independent
four-momenta ($p_{B_c}$, $p_{J/\psi}$) and the $J/\psi$ polarisation, Lorentz invariance
forces it into a fixed set of tensor structures, each multiplied by an unknown scalar
function of the only invariant available,
$$
q^2 = (p_{B_c}-p_{J/\psi})^2 = (p_\ell+p_\nu)^2 .
$$
Those scalar functions are the **form factors** (FFs). Physically, a form factor is the
transition amplitude as a function of how hard the daughter recoils: $q^2$ is the
invariant mass of the lepton pair, ranging from $q^2=0$ (leptons back-to-back, $J/\psi$
maximally boosted) to $q^2_{\max}=(M_{B_c}-M_{J/\psi})^2$ (leptons at rest relative to
each other, $J/\psi$ at rest — "zero recoil"). **Everything measurable** — the decay
rate, the angular distributions of the muon, and the ratio
$R(J/\psi)=\mathcal B(B_c\to J/\psi\tau\nu)/\mathcal B(B_c\to J/\psi\mu\nu)$ — is a
convolution of the (known) leptonic physics with these (unknown) form factors.

Why we care for $R(J/\psi)$: the CKM factor $V_{cb}$, $\eta_{\rm EW}$ and $G_F$ are
identical for the $\mu$ and $\tau$ channels and **cancel in the ratio**, so $R(J/\psi)$
is a clean test of lepton-flavour universality — *provided* the form factors are known,
because the heavier $\tau$ probes the form factors differently from the $\mu$ (it is
sensitive to the scalar/longitudinal pieces that the light $\mu$ barely sees).

## 2. The four form factors of a pseudoscalar → vector transition

$B_c$ is a pseudoscalar ($J^P=0^-$), $J/\psi$ a vector ($1^-$). The vector and axial
parts of the current give **four** independent form factors. Two common bases:

**Standard basis $(V,A_0,A_1,A_2)$.** $V$ multiplies the vector (parity-odd) structure;
$A_0,A_1,A_2$ the axial ones:
$$
\langle J/\psi(\varepsilon)|\bar c\gamma^\mu b|B_c\rangle
=\frac{2V(q^2)}{M_{B_c}+M_{J/\psi}}\,\epsilon^{\mu\nu\rho\sigma}\varepsilon^*_\nu p_\rho p'_\sigma,
$$
with the axial matrix element carried by $A_0,A_1,A_2$. A kinematic identity removes an
apparent pole at $q^2=0$, so only three of these four are independent at that point:
$$
A_0(0)=\frac{(M_{B_c}+M_{J/\psi})A_1(0)-(M_{B_c}-M_{J/\psi})A_2(0)}{2M_{J/\psi}} .
$$

**Helicity basis $(g,f,\mathcal F_1,\mathcal F_2)$.** These are fixed *kinematic*
recombinations that carry definite $J^P$ of the current, which is what unitarity bounds
(Section 4) organise by:
$$
g=\frac{2V}{M_{B_c}+M_{J/\psi}},\quad f=(M_{B_c}+M_{J/\psi})A_1,\quad
\mathcal F_2=2A_0,\quad \mathcal F_1=8M_{B_c}M_{J/\psi}\,A_{12},
$$
with $A_{12}$ a fixed combination of $A_1,A_2$. Intuitively: $g$ is transverse-vector,
$f$ and $\mathcal F_1$ are the transverse and longitudinal axial pieces, $\mathcal F_2$
is the scalar/longitudinal piece that dominates the $\tau$ channel. The two bases are
equivalent; different tools speak different ones (the lattice paper gives both; Hammer's
generator interface uses the helicity basis internally). An exact and useful relation is
$\mathcal F_1=8M_{B_c}M_{J/\psi}A_{12}$ — a *constant* proportionality — together with the
endpoint constraint $\mathcal F_1(q^2_{\max})=(M_{B_c}-M_{J/\psi})f(q^2_{\max})$, which
removes one free parameter.

## 3. Kinematics: analyticity and the variable $z$

A form factor $F(q^2)$ is an analytic function of $q^2$ with a specific structure: it is
real and smooth in the physical (decay) region $0\le q^2\le q^2_{\max}$; it has isolated
**poles** just above, at the masses of $b\bar c$ resonances (the $B_c^{(*)}$ states); and
it has a **branch cut** starting at the pair-production threshold $t_+=(M_B+M_{D^*})^2$,
where the current can create a real two-hadron state. This analytic structure is the
lever behind the modern parametrisations. One maps the cut $q^2$-plane onto the unit disc,
$$
z(q^2,t_0)=\frac{\sqrt{t_+-q^2}-\sqrt{t_+-t_0}}{\sqrt{t_+-q^2}+\sqrt{t_+-t_0}},
$$
where $t_0$ is a chosen expansion point. In the whole physical region $|z|$ is small
($\lesssim 0.05$ here), so any form factor is an fast-converging power series in $z$.

## 4. Three ways to get the numbers

The *shape* is constrained by analyticity; the *numbers* have to come from somewhere.

**(a) QCD sum rules — Kiselev (this MC's generator).** A model calculation. In the
EvtGen implementation used to generate our sample (`BC_VMN`, `whichfit=1`) all four FFs
share a single pole, $F(q^2)=F(0)/(1-q^2/M_{\rm pole}^2)$ with $M_{\rm pole}=4.5$ GeV and
$F_V=0.11,\ F_{A_+}=-0.074,\ F_{A_0}=5.9,\ F_{A_-}=0.12$. It was state of the art when the
generators were written. Its single-pole shape rises steeply toward $q^2_{\max}$, and it
carries no systematically-improvable uncertainty. It is now superseded.

**(b) Model-independent dispersive parametrisation — BGL / CLL.** Boyd–Grinstein–Lebed
(BGL) turn analyticity + unitarity into a rigorous form:
$$
\boxed{\,F(q^2)=\frac{1}{P(q^2)\,\phi(q^2)}\sum_{n} a_n\,z(q^2)^n\,}
$$
- $P(q^2)$ (**Blaschke factor**) is a known product that cancels the sub-threshold poles;
- $\phi(q^2)$ (**outer function**) is a known analytic function fixed by a perturbative QCD
  susceptibility $\chi$;
- the coefficients $a_n$ carry the actual physics, truncated at low order.

The payoff is a rigorous **unitarity bound** $\sum_n a_n^2\le 1$ (the total "strength" of
the current is bounded), which is a strong consistency check on any coefficient set.
Cohen–Lamm–Lebed (CLL) specialised BGL to $B_c\to J/\psi$; their coefficients are the
default in the Hammer reweighting tool.

**(c) Lattice QCD — Harrison (our target).** Harrison/HPQCD compute the hadronic matrix
elements from first principles on the lattice at several $q^2$ points and fit a
dispersive parametrisation with a full covariance. This is the modern, ab-initio input.
Two versions matter here: **Harrison-2020** (arXiv:2007.06957), which is the default that
ships inside Hammer, and **Harrison-2024** (arXiv:2503.15090), the newer, more precise
update that is our reweighting *target*.

## 5. Reweighting, and what Hammer does

The MC was generated with Kiselev form factors because that is what the generator offered.
To use Harrison-2024 without regenerating, we **reweight** each simulated event by
$$
w_{\rm event}=\frac{|\mathcal M_{\rm Harrison}|^2}{|\mathcal M_{\rm Kiselev}|^2}
= \frac{d\Gamma_{\rm Harrison}}{d\Gamma_{\rm Kiselev}}\Big|_{\rm this\ event},
$$
the ratio of the squared decay amplitudes at that event's exact kinematics. All the
electroweak/CKM factors cancel, so the weight is a pure form-factor ratio. The
[HAMMER](https://hammer.physics.lbl.gov) library computes these weights at generator
(truth) level, including the full spin-correlated decay chain, and also provides the
propagated **form-factor uncertainty** as a set of eigenvector variations (the systematic
band on the reweighted distributions).

---

# Part II — Final procedure: Harrison-2024 through Hammer

## 6. The one fact that dictates the whole procedure

Hammer's $B_c\to J/\psi$ BGL class (`BctoJpsiBGLVar`) already implements **Harrison-2020**
as its default, in the CLL/BGL formalism — *but in a specific internal convention* that is
not the textbook dispersive one. Concretely, reading Hammer's source, its evaluator uses
**two distinct $z$-variables** (one for $g$, one for $f/\mathcal F_1/\mathcal F_2$, built
from mass ratios $R_{b^*},R_{b0},R_{d0}$), its own outer functions, the endpoint constraint
built in, and it stores the coefficients as vectors `avec,bvec,cvec,dvec` (mapping to
$g,f,\mathcal F_1,\mathcal F_2$). There is **no $V_{cb}$ factor** in this class (it cancels
in the reweighting anyway).

The consequence is simple and strict: **coefficients only mean something to Hammer if they
are expressed in Hammer's own convention.** Numbers taken from a paper written in a
different convention will be silently mis-evaluated. So the task is not "find Harrison-2024
BGL coefficients" — it is "find the coefficients that make *Hammer's evaluator* reproduce
Harrison-2024's physical form-factor curves."

## 7. The procedure (five steps)

1. **Evaluate Harrison-2024 physical form factors.** Using the lattice ancillary
   (`harrison_ffs.py`, a thin wrapper over Harrison's released fit), compute the physical
   helicity form factors $g,f,\mathcal F_1,\mathcal F_2(q^2)$ on a dense $q^2$ grid,
   *with their full correlated covariance* (propagated with `gvar`). These curves are the
   parametrisation-independent truth we want Hammer to follow.

2. **Reimplement Hammer's exact forward model.** Port Hammer's `FFBctoJpsiBGL::evalAtPSPoint`
   to Python (`hammer_bgl_forward.py`): given `avec,bvec,cvec,dvec`, return
   $g,f,\mathcal F_1,\mathcal F_2$ in Hammer's convention. This is validated two ways —
   it reproduces Harrison-2020 to a few percent when fed Hammer's default coefficients,
   and the endpoint constraint $\mathcal F_1/f\to(M_{B_c}-M_{J/\psi})$ comes out exact.
   (**Corrected, Sep. 2026.** The first version stated that Hammer's internal P1
   equals $A_0$, i.e. half of Harrison's $\mathcal F_2=2A_0$, and fitted `dvec` to
   $\mathcal F_2/2$. That was read from `evalAtPSPoint` alone and is wrong. Hammer's
   amplitude never uses P1 as $A_0$: it maps $(f,g,\mathcal F_1,P_1)$ onto the
   Manohar–Wise basis, P1 enters only $a_-$ through
   $F_{mP1}=\sqrt r(1+r)/(M_b(1+r^2-2rw))$, and contracting the axial current with
   $q$ gives $A_0=\frac{1+r}{2\sqrt r}P_1$, constant in $q^2$, with $r=M_{J/\psi}/M_{B_c}$.
   Hence $P_1=\frac{\sqrt r}{1+r}\mathcal F_2$. The factor $1.063$ entered both the fit
   target and the rate formula, cancelled in every port-side test, and was caught
   only against Hammer's C++: see §10.)

3. **Fit the coefficients through that model** (`harrison_to_hammer_bgl.py`). Each form
   factor is *linear* in its coefficient vector, so the fit is a linear least-squares of
   `avec→g`, `bvec→f`, `cvec→F1` (with the endpoint-constrained leading term), `dvec→F2/2`,
   against the Harrison-2024 curves. Because the $z$-basis is nearly degenerate over the
   narrow physical $z$-range, the fit is **regularised** (Tikhonov, pulling the unconstrained
   direction toward Hammer's default): this leaves the reproduced curve unchanged
   (pull $\sim10^{-4}$) but returns physical, **unitarity-satisfying** coefficients rather
   than large oscillating ones. The output is `avec,bvec,cvec,dvec` in Hammer's convention.

4. **Load the central form factors and produce nominal weights.** Set the coefficients on
   `BctoJpsiBGLVar` via `set_options`, with the input scheme `Kiselev`; the per-event weight
   is `get_weight("Harrison")`. The decay is built as $B_c\to J/\psi\,\mu\,\nu$ (with the
   $J/\psi\to\mu\mu$ vertex included for full spin correlation), the neutrino taken from the
   stored truth. Sanity numbers: the average $\mu$-channel weight is $\langle w\rangle\approx
   0.5$ (Harrison predicts a somewhat lower rate than Kiselev), and $R(J/\psi)^{\rm SM}$ from
   the reweighted sample matches the lattice value $0.2597$.

5. **Propagate the uncertainty (systematic band).** Diagonalise the fitted-coefficient
   covariance; its eigenvectors (scaled by $\sqrt{\text{eigenvalue}}$) fill Hammer's
   `abcdmatrix`, and each of the 15 `delta_e` error coordinates is an independent
   $\pm1\sigma$ nuisance obtained with `set_ff_eigenvectors`. This yields one up/down
   template per eigen-direction — the blue envelope on the reweighted distributions —
   which propagate into the fit as orthogonal form-factor systematics.

The production script `hammer_ff_weights.py` implements steps 4–5, writing a friend tree
with the nominal weight and the eigenvector variations, aligned 1:1 with the source.

## 8. Validation (what makes us believe it)

- **Unitarity:** all four fitted coefficient vectors satisfy $\sum_n a_n^2<1$.
- **Forward-model closure:** the fit reproduces Harrison-2024's $g,f,\mathcal F_1$ curves
  with negligible pull; the model self-recovers Hammer's default coefficients to $10^{-13}$.
- **Machinery closure (independent of our coefficients):** reweighting between Hammer's
  built-in models round-trips to unity (Kiselev→Ebert→Kiselev $=1$ event-by-event), and
  Kiselev→(built-in Cohen/Harrison-2020) gives a sensible $\langle w\rangle\approx0.5$ —
  confirming the reweighting engine, the Kiselev denominator, and the decay construction.
- **Physical closure:** the reweighted gen-level $q^2$ spectrum (blue) is a modest,
  smooth softening of Kiselev (red) with a tight systematic envelope — exactly the size and
  shape expected from replacing an old sum-rules model with modern lattice form factors.
- **Absolute normalisation:** $\langle w\rangle_\mu\approx0.5$ is consistent with the Run-2
  analysis (AN-20-223), which measured $\bar w_\mu\approx0.6$ for the analogous
  Kiselev→lattice reweighting.

## 9. Residual items (before unblinding / production)

- ~~The $\mathcal F_2\!=\!A_0$ (factor-2) mapping~~ **Resolved** (§10): the mapping was
  wrong by $(1+r)/(2\sqrt r)=1.063$ and has been pinned from the amplitude mapping, as
  this item recommended.
- A few $\times10^{-3}$ of events have a slightly non-closing truth neutrino
  ($B_c\ne J/\psi+\mu+\nu$ at the $10^{-3}$ level from FSR/rounding), giving non-finite
  weights; these are excluded and should be handled by clamping $m^2\ge0$ or taking the
  neutrino from momentum conservation.
- The $\tau$ channel and the angular (helicity) distributions — where the $\tau$ sensitivity
  lives — reuse the same validated form factors and should get the same closure treatment as
  the $\mu$ $q^2$ spectrum shown here.

## 10. The P1 convention: how it was found and fixed (Sep. 2026)

**Symptom.** With the same coefficients, Hammer's C++ returned a larger $R(J/\psi)$ than the
Python port, and the excess scaled with the fraction of $\Gamma_\tau$ carried by $H_t$:

| configuration | Hammer (C++) | port, old convention | excess / $H_t$ share |
|---|---|---|---|
| `dvec`×0 | 0.2403(9) | 0.2402 | — ($H_t$ off: agree to $4\times10^{-4}$) |
| `dvec`×2 | 0.3291(11) | 0.3193 | 0.120 |
| Hammer's own defaults | 0.2592(9) | 0.2569 | 0.120 |

The last row uses Hammer's built-in coefficients, with no card and no fit, so the
disagreement was between the port and the C++ themselves.

**Cause.** `evalAtPSPoint` is identical in both. The difference is downstream: Hammer's
amplitude reads P1 through $a_-$, which implies $A_0=\frac{1+r}{2\sqrt r}P_1$; the port and
the fitter assumed $A_0=P_1$. Predicted excess $(1.063)^2-1=0.130$, measured $0.120\pm0.012$.

**Why nothing else caught it.** The wrong convention was used twice — as the fit target
and in the rate — and cancelled exactly, so Fig. 6, $R$ and $A_{\lambda_\tau}$ all closed.
Only an independent evaluator could break the symmetry.

**Confirmation.** With the corrected convention the port reproduces all five Hammer
measurements to within $0.2\sigma$ with no free parameter. The shipped card was fitted in
the old convention and is refused by `HammerSession` until refitted.
