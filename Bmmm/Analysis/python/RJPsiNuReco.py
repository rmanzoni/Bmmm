'''
RJpsiNuReco.py
==============

Longitudinal ("two-fold") reconstruction of the neutrino in

        Bc  ->  (3mu visible system)  +  nu

from three measured ingredients:

    1. the visible 3-muon four-momentum            p4_3mu
    2. the Bc flight direction  n_hat              (PV -> SV in data,
                                                    gen production -> decay
                                                    vertex in MC)
    3. the known Bc mass                            m_Bc

This is the standard technique used in R(J/psi): the only unknown is the
neutrino momentum *along* the Bc flight direction; everything transverse to it
is fixed by momentum conservation, leaving a quadratic with two solutions.

Physics
-------
Decompose every 3-momentum into a component parallel (||) and perpendicular (T)
to the unit flight direction n_hat. By construction the Bc carries no transverse
momentum w.r.t. its own flight direction, so transverse-momentum balance fixes
the neutrino transverse momentum completely:

        pT_nu (vector) = - pT_3mu (vector)     =>     |pT_nu| = |pT_3mu|

The single remaining unknown is the neutrino longitudinal momentum  x = p||_nu.
Energy conservation

        E_Bc = E_3mu + E_nu
        E_Bc^2 = m_Bc^2 + (p||_3mu + x)^2          (Bc has no pT w.r.t. n_hat)
        E_nu^2 = x^2 + pT_3mu^2                     (massless neutrino)

isolating the square root and squaring once leads to  A x^2 + B x + C = 0 :

        K = m_Bc^2 + p||_3mu^2 - pT_3mu^2 - E_3mu^2          (Image 1, "k")
        A = 4 ( p||_3mu^2 - E_3mu^2 )
        B = 4   p||_3mu   K
        C = K^2 - 4 E_3mu^2 pT_3mu^2

        x_{1,2} = ( -B +- sqrt(B^2 - 4 A C) ) / (2 A)        (Image 2)

Notes
-----
* A = 4(p||^2 - E^2) = -4(m_3mu^2 + pT_3mu^2) < 0 strictly (the visible system
  is three muons, so m_3mu > 0). The leading coefficient never vanishes, so the
  equation is genuinely quadratic and the "divide by 2A" is always safe.

* Exact for the single-massless-neutrino mode  Bc -> J/psi mu nu.
  For Bc -> J/psi tau nu the missing system is three neutrinos with a nonzero
  invariant mass, so the massless assumption is only approximate. An optional
  `m_miss` argument lets you put that mass back in if you ever want to (the
  default m_miss = 0 reproduces the equations on the paper exactly).

* Two real roots in general -> the classic two-fold ambiguity. In MC you can
  resolve it with `pick_closest(..., pz_true)`; in data common choices are the
  smaller |p_nu| / smaller |p_Bc| root, or carry both as alternative branches.
  When detector resolution pushes the discriminant slightly negative there is no
  real root; pass `clamp_negative_disc=True` to receive instead two identical
  roots at the parabola vertex -B/(2A) (the real part of the complex-conjugate
  pair) so downstream code that expects a 2-element list keeps working.

Conventions
-----------
INPUT four-vectors are taken in (E, px, py, pz) order as numpy arrays, matching
RJPsiGenHistory._p4; the public functions also transparently accept ROOT
LorentzVectors (anything exposing .energy()/.px()/.py()/.pz()) and ROOT
XYZVectors (anything exposing .x()/.y()/.z()), so the *same* solver works on
gen four-vectors and on the reco RJpsiCandidate (cand.p4(), cand.Bdirection).

OUTPUT four-vectors are *true* four-vector objects, not bare numpy arrays:
`nu_p4_from_pz` and `reconstruct` return a
ROOT.Math.LorentzVector<PxPyPzE4D<double>> -- the exact type returned by
reco::Candidate::p4() and RJpsiCandidate.p4() -- so they support full
four-vector algebra (p4_vis + p4_nu, .mass(), .Boost(), ...) and compose
directly with cand.p4(). If ROOT cannot be imported (running this module
standalone, outside CMSSW) a lightweight numpy-backed LorentzVector with the
same accessors is returned instead, so nothing here *requires* ROOT.
The scalar solver `solve_nu_pz` still returns plain Python floats for p||_nu.
'''

from __future__ import print_function
import numpy as np
from collections import namedtuple

# Bc mass [GeV]; try to reuse the constant from the gen-history module so there
# is a single source of truth, fall back to the PDG value if it is not importable
# (e.g. when running this file standalone without CMSSW / the Bmmm package).
try:
    from Bmmm.Analysis.RJPsiGenHistory import M_BC as M_BC          # noqa: F401
except Exception:
    try:
        from Bmmm.Analysis.RJPsiGenHistory import _p4 as _gen_p4    # noqa: F401
    except Exception:
        _gen_p4 = None
    M_BC = 6.27447

__all__ = [
    'M_BC', 'solve_nu_pz', 'nu_p4_from_pz', 'reconstruct', 'pick_closest',
    'NuSolution', 'gen_flight_dir', 'gen_visible_p4_3mu', 'gen_nu_reco',
    'make_p4',
]


# ---------------------------------------------------------------------------
# small input adapters: accept numpy, ROOT LorentzVector, ROOT XYZVector, list
# ---------------------------------------------------------------------------
def _to_np4(p4):
    '''Return a (4,) numpy array (E, px, py, pz) from numpy / ROOT LorentzVector
    / _NpLorentzVector.'''
    if isinstance(p4, np.ndarray):
        return p4.astype(float)
    if hasattr(p4, 'energy') and hasattr(p4, 'px'):
        return np.array([p4.energy(), p4.px(), p4.py(), p4.pz()], dtype=float)
    return np.asarray(p4, dtype=float)


def _to_np3(v):
    '''Return a (3,) numpy array from numpy / ROOT XYZVector / list.
    If a 4-vector is passed by mistake, its spatial part is used.'''
    if isinstance(v, np.ndarray):
        return v[1:].astype(float) if v.size == 4 else v.astype(float)
    if hasattr(v, 'x') and hasattr(v, 'y') and hasattr(v, 'z'):
        return np.array([v.x(), v.y(), v.z()], dtype=float)
    a = np.asarray(v, dtype=float)
    return a[1:] if a.size == 4 else a


def _unit(v3):
    n = float(np.sqrt(v3.dot(v3)))
    if n == 0.:
        raise ValueError('flight direction has zero length')
    return v3 / n


# ---------------------------------------------------------------------------
# four-vector output factory
# ---------------------------------------------------------------------------
# We keep all the algebra in numpy internally (clean, ROOT-free, unit-testable)
# but *hand back* true four-vector objects so the caller can do four-vector
# algebra directly. On a CMSSW node this is the very same
# ROOT.Math.LorentzVector<PxPyPzE4D<double>> that cand.p4() returns, so e.g.
#     bc = cand.p4() + reconstruct(cand.p4(), cand.Bdirection)[0].p4_nu
# just works. Off CMSSW we fall back to a tiny numpy-backed look-alike.

class _NpLorentzVector(object):
    '''Minimal stand-in for ROOT.Math.LorentzVector<PxPyPzE4D<double>>, used
    only when ROOT is not importable (e.g. the standalone self-test). It exposes
    the handful of accessors the rest of the codebase relies on, plus four-vector
    addition, so reconstructed neutrinos and parents behave like real
    four-vectors regardless of whether ROOT is present. Internally (E,px,py,pz).
    '''
    __slots__ = ('_e', '_px', '_py', '_pz')

    def __init__(self, E, px, py, pz):
        self._e  = float(E)
        self._px = float(px)
        self._py = float(py)
        self._pz = float(pz)

    # --- components (both lower- and upper-case, mirroring ROOT's interface)
    def energy(self): return self._e
    def E(self):      return self._e
    def t(self):      return self._e
    def T(self):      return self._e
    def px(self):     return self._px
    def Px(self):     return self._px
    def py(self):     return self._py
    def Py(self):     return self._py
    def pz(self):     return self._pz
    def Pz(self):     return self._pz

    # --- derived kinematics
    def p2(self):   return self._px * self._px + self._py * self._py + self._pz * self._pz
    def P(self):    return float(np.sqrt(self.p2()))
    def pt(self):   return float(np.sqrt(self._px * self._px + self._py * self._py))
    def Pt(self):   return self.pt()
    def perp(self): return self.pt()

    def mass2(self): return self._e * self._e - self.p2()
    def M2(self):    return self.mass2()
    def mass(self):
        m2 = self.mass2()
        return float(np.sqrt(m2)) if m2 >= 0. else -float(np.sqrt(-m2))
    def M(self):     return self.mass()

    def phi(self): return float(np.arctan2(self._py, self._px))
    def Phi(self): return self.phi()
    def eta(self):
        pt = self.pt()
        if pt == 0.:
            return float('inf') if self._pz > 0 else (float('-inf') if self._pz < 0 else 0.0)
        return float(np.arcsinh(self._pz / pt))
    def Eta(self): return self.eta()

    # --- algebra
    def __add__(self, other):
        o = _to_np4(other)
        return _NpLorentzVector(self._e + o[0], self._px + o[1],
                                self._py + o[2], self._pz + o[3])

    def __radd__(self, other):
        # make sum([...]) (which starts from int 0) work transparently
        if isinstance(other, int) and other == 0:
            return _NpLorentzVector(self._e, self._px, self._py, self._pz)
        return self.__add__(other)

    def __repr__(self):
        return ('LorentzVector(E=%.6g, px=%.6g, py=%.6g, pz=%.6g | m=%.6g)'
                % (self._e, self._px, self._py, self._pz, self.mass()))


_FV_TYPE = None        # cached ROOT LorentzVector<PxPyPzE4D<double>> *class*
_FV_RESOLVED = False   # have we already tried to import ROOT?


def _fourvector_type():
    '''Lazily resolve and cache ROOT.Math.LorentzVector<PxPyPzE4D<double>>.
    Returns None if ROOT is not importable so the module stays usable without
    it (and importing this module never drags in ROOT on its own).'''
    global _FV_TYPE, _FV_RESOLVED
    if not _FV_RESOLVED:
        _FV_RESOLVED = True
        try:
            import ROOT
            _FV_TYPE = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzE4D<double>')
        except Exception:
            _FV_TYPE = None
    return _FV_TYPE


def make_p4(E, px, py, pz):
    '''Build a true four-vector from (E, px, py, pz).

    Returns a ROOT.Math.LorentzVector<PxPyPzE4D<double>> -- identical to the
    type of reco::Candidate::p4() / RJpsiCandidate.p4() -- when ROOT is
    available, so it composes directly with cand.p4() and supports the full ROOT
    four-vector algebra (+, .mass(), .Boost(), .BoostToCM(), ...). Falls back to
    a numpy-backed LorentzVector with the same accessors when ROOT is absent.
    '''
    lv = _fourvector_type()
    if lv is not None:
        # NB: the PxPyPzE4D constructor signature is (px, py, pz, E)
        return lv(float(px), float(py), float(pz), float(E))
    return _NpLorentzVector(E, px, py, pz)


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------
def solve_nu_pz(p4_vis, flight_dir, m_parent=M_BC, m_miss=0.0,
                clamp_negative_disc=False):
    '''Solve the quadratic for the neutrino longitudinal momentum p||_nu.

    Parameters
    ----------
    p4_vis     : visible 4-momentum (E, px, py, pz) of the 3mu system.
                 numpy array or ROOT LorentzVector.
    flight_dir : parent flight direction (need not be normalised).
                 numpy 3-vector or ROOT XYZVector.
    m_parent   : parent (Bc) mass [GeV].
    m_miss     : invariant mass of the missing system [GeV]. 0 = single massless
                 neutrino (matches the paper exactly); set >0 for the tau mode.
    clamp_negative_disc : if True and the discriminant is < 0 (no real root,
                 typically from detector resolution), return two identical roots
                 equal to the parabola vertex -B/(2A) -- i.e. the real part of
                 the complex-conjugate pair -- instead of an empty list. The
                 returned 'clamped' flag is True in that case. Default False
                 reproduces the previous behaviour (empty list on disc < 0).

    Returns
    -------
    dict with keys:
        pz_solutions  : list of the roots for p||_nu (floats). Length 0, 1 or 2:
                        2 for disc >= 0; 2 identical (vertex) if disc < 0 and
                        clamp_negative_disc=True; 0 if disc < 0 otherwise;
                        1 only in the (unphysical) degenerate linear case A == 0.
        has_real      : bool, True if the discriminant is >= 0
        clamped       : bool, True if the two returned roots are the vertex
                        fallback for a negative discriminant (not true roots)
        discriminant  : B^2 - 4 A C
        pz_at_vertex  : -B/(2A); the parabola vertex, i.e. the midpoint of the
                        two roots. Useful as a single fallback value when the
                        discriminant goes slightly negative (resolution).
        A, B, C, K    : the quadratic coefficients and the helper K
        p_par_vis     : p||_3mu   (signed projection on n_hat)
        p_perp_vis    : |pT_3mu|  (>= 0)
        E_vis         : E_3mu
        n_hat         : the unit flight direction actually used
    '''
    p4v = _to_np4(p4_vis)
    E_vis = float(p4v[0])
    p_vis = p4v[1:]

    n = _unit(_to_np3(flight_dir))

    p_par_vis = float(p_vis.dot(n))                                   # p||_3mu
    p2 = float(p_vis.dot(p_vis))
    p_perp_vis = float(np.sqrt(max(0., p2 - p_par_vis * p_par_vis)))  # |pT_3mu|

    K = (m_parent * m_parent + p_par_vis * p_par_vis
         - p_perp_vis * p_perp_vis - E_vis * E_vis - m_miss * m_miss)

    A = 4.0 * (p_par_vis * p_par_vis - E_vis * E_vis)
    B = 4.0 * p_par_vis * K
    C = K * K - 4.0 * E_vis * E_vis * (p_perp_vis * p_perp_vis + m_miss * m_miss)

    disc = B * B - 4.0 * A * C

    # A is negative and non-zero for any real 3mu system, but guard anyway.
    pz_vertex = (-B / (2.0 * A)) if A != 0. else np.nan

    clamped = False
    if A != 0.:
        if disc >= 0.:                                  # two real roots (generic)
            root = np.sqrt(disc)
            sols = [(-B + root) / (2.0 * A), (-B - root) / (2.0 * A)]
        elif clamp_negative_disc:                       # disc < 0, optional fallback
            sols = [pz_vertex, pz_vertex]               # real part of the pair, twice
            clamped = True
        else:                                           # disc < 0, no real root
            sols = []
    else:                                               # A == 0: degenerate linear
        sols = [-C / B] if B != 0. else []

    return {
        'pz_solutions': sols,
        'has_real'    : disc >= 0.,
        'clamped'     : clamped,
        'discriminant': disc,
        'pz_at_vertex': pz_vertex,
        'A': A, 'B': B, 'C': C, 'K': K,
        'p_par_vis'   : p_par_vis,
        'p_perp_vis'  : p_perp_vis,
        'E_vis'       : E_vis,
        'n_hat'       : n,
    }


def nu_p4_from_pz(p4_vis, flight_dir, pz_nu, m_miss=0.0):
    '''Build the missing-system four-vector for a given longitudinal momentum.

    The 3-momentum follows purely from balance (independent of m_miss):
        p_nu = (p||_nu + p||_3mu) * n_hat - p_3mu
    so that p_nu is at +p||_nu along n_hat and exactly cancels the visible pT.
    The energy uses the (optional) missing mass: E = sqrt(m_miss^2 + |p_nu|^2).

    Returns a true four-vector (ROOT LorentzVector when ROOT is available, see
    `make_p4`), so it can be added straight to the visible p4.
    '''
    p4v = _to_np4(p4_vis)
    p_vis = p4v[1:]
    n = _unit(_to_np3(flight_dir))
    p_par_vis = float(p_vis.dot(n))

    p_nu = (pz_nu + p_par_vis) * n - p_vis
    E_nu = float(np.sqrt(m_miss * m_miss + p_nu.dot(p_nu)))
    return make_p4(E_nu, p_nu[0], p_nu[1], p_nu[2])


NuSolution = namedtuple('NuSolution', ['pz', 'p4_nu', 'p4_parent', 'parent_mass'])


def reconstruct(p4_vis, flight_dir, m_parent=M_BC, m_miss=0.0,
                clamp_negative_disc=False):
    '''Full reconstruction. Returns a list of NuSolution (one per returned root):

        NuSolution(pz, p4_nu, p4_parent, parent_mass)

    where `p4_nu` and `p4_parent` are *true four-vectors* (ROOT LorentzVectors
    when ROOT is available, see `make_p4`) with `p4_parent = p4_vis + p4_nu`, and
    `parent_mass` is the recovered invariant mass (a diagnostic; it comes back
    equal to m_parent up to FP rounding for every real root). The list is empty
    if there is no real solution, unless `clamp_negative_disc=True`, in which
    case a negative discriminant yields two identical vertex solutions (see
    `solve_nu_pz`); their parent_mass will *not* equal m_parent since no exact
    solution exists.
    '''
    res = solve_nu_pz(p4_vis, flight_dir, m_parent=m_parent, m_miss=m_miss,
                      clamp_negative_disc=clamp_negative_disc)
    p4v = _to_np4(p4_vis)
    out = []
    for pz in res['pz_solutions']:
        p4_nu = nu_p4_from_pz(p4v, flight_dir, pz, m_miss=m_miss)   # four-vector
        par = p4v + _to_np4(p4_nu)                                 # (E,px,py,pz)
        m2 = par[0] ** 2 - par[1] ** 2 - par[2] ** 2 - par[3] ** 2
        p4_parent = make_p4(par[0], par[1], par[2], par[3])        # four-vector
        out.append(NuSolution(pz=pz, p4_nu=p4_nu, p4_parent=p4_parent,
                              parent_mass=float(np.sqrt(max(0., m2)))))
    return out


def pick_closest(solutions, pz_true):
    '''Return the NuSolution whose pz is closest to pz_true (MC truth helper).
    Returns None if `solutions` is empty.'''
    if not solutions:
        return None
    return min(solutions, key=lambda s: abs(s.pz - pz_true))


# ---------------------------------------------------------------------------
# gen-level drivers (use RJPsiGenHistory.BcGenDecay). These touch gen-particle
# accessors and so only run at event time inside CMSSW, exactly like the rest of
# RJPsiGenHistory; the core solver above is fully standalone / unit-testable.
# ---------------------------------------------------------------------------
def _p4(p):
    '''(E, px, py, pz) from a gen particle, mirroring RJPsiGenHistory._p4.'''
    return np.array([p.energy(), p.px(), p.py(), p.pz()], dtype=float)


def gen_flight_dir(bc):
    '''Gen Bc flight direction from production -> decay vertex.

    The production vertex is the Bc's own (vx, vy, vz); the decay vertex is the
    production vertex of any Bc daughter. The Bc is neutral, so this is parallel
    to its momentum -- using p4[1:] instead gives the same direction at gen
    level, and is the natural cross-check.
    '''
    prod = np.array([bc.vx(), bc.vy(), bc.vz()], dtype=float)
    dau = bc.daughter(0)
    dec = np.array([dau.vx(), dau.vy(), dau.vz()], dtype=float)
    return dec - prod


def gen_visible_p4_3mu(bc_gen):
    '''Visible 3mu four-momentum at gen level: J/psi dimuon + bachelor muon.
    Returns None if the decay does not have both (e.g. hadronic Bc modes).'''
    if bc_gen is None or bc_gen.p4_jpsi is None or bc_gen.bachelor_mu is None:
        return None
    return bc_gen.p4_jpsi + _p4(bc_gen.bachelor_mu)


def gen_nu_reco(bc_gen, use_vertex_dir=True, m_miss=0.0,
                clamp_negative_disc=False):
    '''Closure-test driver on a BcGenDecay object.

    Reconstructs the two p||_nu solutions from gen quantities and compares them
    to the gen truth. Returns a dict (values NaN / empty when not computable):

        solutions   : list of NuSolution from `reconstruct`
        pz_nu_true  : true neutrino-system longitudinal momentum (proj. on n_hat)
        best        : the NuSolution closest to pz_nu_true (or None)
        dpz_best    : best.pz - pz_nu_true        (closure residual)
        p_bc_true   : true Bc |p|;  p_bc_reco_best: reconstructed Bc |p| (best sol)
        n_real      : number of real roots
    '''
    out = {'solutions': [], 'pz_nu_true': np.nan, 'best': None,
           'dpz_best': np.nan, 'p_bc_true': np.nan,
           'p_bc_reco_best': np.nan, 'n_real': 0}
    if bc_gen is None:
        return out

    p4_vis = gen_visible_p4_3mu(bc_gen)
    if p4_vis is None:
        return out

    n = _unit(_to_np3(_p4(bc_gen)) if not use_vertex_dir
              else gen_flight_dir(bc_gen.bc))

    sols = reconstruct(p4_vis, n, m_parent=M_BC, m_miss=m_miss,
                       clamp_negative_disc=clamp_negative_disc)
    out['solutions'] = sols
    out['n_real'] = len(sols)

    # gen truth: longitudinal momentum of the (summed) neutrino system
    if bc_gen.neutrinos:
        p_nu_true = np.sum([_p4(nu) for nu in bc_gen.neutrinos], axis=0)[1:]
        pz_true = float(p_nu_true.dot(n))
        out['pz_nu_true'] = pz_true
        best = pick_closest(sols, pz_true)
        out['best'] = best
        if best is not None:
            out['dpz_best'] = best.pz - pz_true
            p3 = _to_np4(best.p4_parent)[1:]
            out['p_bc_reco_best'] = float(np.sqrt(p3.dot(p3)))

    p_bc = _p4(bc_gen)[1:]
    out['p_bc_true'] = float(np.sqrt(p_bc.dot(p_bc)))
    return out


# ---------------------------------------------------------------------------
# self-test: synthetic Bc -> (visible) + nu event, boosted into the lab.
# Runs without requiring ROOT (a numpy-backed four-vector is used as fallback):
#     python -m Bmmm.Analysis.RJpsiNuReco   (or: python RJpsiNuReco.py)
# ---------------------------------------------------------------------------
def _boost_rest_to_lab(four, beta):
    '''Boost a rest-frame 4-vector into a lab where the system moves with `beta`.'''
    b2 = float(beta.dot(beta))
    if b2 <= 0.:
        return four.copy()
    g = 1.0 / np.sqrt(1.0 - b2)
    E, p = four[0], four[1:]
    bp = float(beta.dot(p))
    E_lab = g * (E + bp)
    p_lab = p + ((g - 1.0) * bp / b2 + g * E) * beta
    return np.array([E_lab, p_lab[0], p_lab[1], p_lab[2]])


def _selftest():
    rng = np.random.default_rng(1)
    print('=' * 70)
    print('RJpsiNuReco self-test  (m_Bc = %.5f GeV)' % M_BC)
    print('four-vector backend: %s'
          % ('ROOT LorentzVector<PxPyPzE4D>' if _fourvector_type() is not None
             else 'numpy fallback (_NpLorentzVector)'))
    print('=' * 70)

    n_ok = 0
    n_try = 200
    worst_dpz = 0.0
    worst_dmass = 0.0
    for _ in range(n_try):
        # --- build a two-body Bc -> (visible particle) + nu in the Bc rest frame
        m_vis = rng.uniform(3.2, 5.5)                  # J/psi + mu invariant mass
        E_vis_r = (M_BC**2 + m_vis**2) / (2 * M_BC)
        p_star = (M_BC**2 - m_vis**2) / (2 * M_BC)     # |p*| ; E_nu_rest = p_star
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        p4_vis_r = np.array([E_vis_r,  p_star * d[0],  p_star * d[1],  p_star * d[2]])
        p4_nu_r = np.array([p_star,  -p_star * d[0], -p_star * d[1], -p_star * d[2]])

        # --- boost into the lab (give the Bc a random lab momentum)
        p_bc = rng.uniform(-30, 30, size=3)
        E_bc = np.sqrt(M_BC**2 + p_bc.dot(p_bc))
        beta = p_bc / E_bc
        p4_vis = _boost_rest_to_lab(p4_vis_r, beta)
        p4_nu = _boost_rest_to_lab(p4_nu_r,  beta)

        flight = p_bc                                  # == Bc momentum direction
        n_hat = flight / np.linalg.norm(flight)
        pz_true = float(p4_nu[1:].dot(n_hat))

        sols = reconstruct(p4_vis, flight, m_parent=M_BC)
        assert len(sols) == 2, 'expected two real roots in the exact toy'

        best = pick_closest(sols, pz_true)
        dpz = abs(best.pz - pz_true)
        dmass = abs(best.parent_mass - M_BC)
        # the matched root must recover the true nu pz and the Bc mass
        ok = (dpz < 1e-6) and (dmass < 1e-6)
        n_ok += ok
        worst_dpz = max(worst_dpz, dpz)
        worst_dmass = max(worst_dmass, dmass)

    print('exact massless-nu closure : %d / %d events recover truth' % (n_ok, n_try))
    print('  worst |pz_reco - pz_true| = %.2e GeV' % worst_dpz)
    print('  worst |m_reco  - m_Bc   | = %.2e GeV' % worst_dmass)

    # --- one detailed example, printed in full
    print('-' * 70)
    m_vis = 3.6
    E_vis_r = (M_BC**2 + m_vis**2) / (2 * M_BC)
    p_star = (M_BC**2 - m_vis**2) / (2 * M_BC)
    d = np.array([0.3, -0.7, 0.5]); d /= np.linalg.norm(d)
    p4_vis_r = np.array([E_vis_r,  p_star*d[0],  p_star*d[1],  p_star*d[2]])
    p4_nu_r = np.array([p_star, -p_star*d[0], -p_star*d[1], -p_star*d[2]])
    p_bc = np.array([8.0, -4.0, 25.0])
    E_bc = np.sqrt(M_BC**2 + p_bc.dot(p_bc))
    beta = p_bc / E_bc
    p4_vis = _boost_rest_to_lab(p4_vis_r, beta)
    p4_nu = _boost_rest_to_lab(p4_nu_r, beta)
    flight = p_bc
    n_hat = flight / np.linalg.norm(flight)
    pz_true = float(p4_nu[1:].dot(n_hat))

    res = solve_nu_pz(p4_vis, flight, m_parent=M_BC)
    print('coefficients:  A=%.4f  B=%.4f  C=%.4f  K=%.4f' %
          (res['A'], res['B'], res['C'], res['K']))
    print('A < 0 (always quadratic)? ', res['A'] < 0)
    print('discriminant = %.4f  (>=0 -> two real roots)' % res['discriminant'])
    print('p||_3mu = %.4f   |pT_3mu| = %.4f   E_3mu = %.4f' %
          (res['p_par_vis'], res['p_perp_vis'], res['E_vis']))
    print('two p||_nu solutions: %.5f , %.5f GeV' % tuple(res['pz_solutions']))
    print('true   p||_nu       : %.5f GeV' % pz_true)
    sols = reconstruct(p4_vis, flight, m_parent=M_BC)
    for i, s in enumerate(sols):
        p3 = _to_np4(s.p4_parent)[1:]
        print('  sol %d: pz=%9.4f  recovered Bc mass=%.5f  Bc |p|=%8.4f  '
              '(p4_nu is %s)' %
              (i, s.pz, s.parent_mass, float(np.sqrt(p3.dot(p3))),
               type(s.p4_nu).__name__))
    print('true Bc |p| = %.4f GeV' % float(np.sqrt(p_bc.dot(p_bc))))

    # --- four-vector algebra demo: parent four-vector built directly
    s0 = sols[0]
    print('four-vector algebra on the returned objects:')
    print('  s.p4_nu.mass()        = %.4e GeV (>= 0, massless nu)' % abs(s0.p4_nu.mass()))
    print('  s.p4_parent.mass()    = %.5f GeV' % s0.p4_parent.mass())
    print('  s.p4_parent.pt()      = %.4f GeV' % s0.p4_parent.pt())

    # --- clamp_negative_disc demo: force disc < 0 with an infeasible m_miss
    # (a missing mass too large for the available phase space leaves no real
    #  neutrino, so the discriminant goes negative).
    print('-' * 70)
    m_miss_bad = 2.8
    r_default = solve_nu_pz(p4_vis, flight, m_parent=M_BC, m_miss=m_miss_bad)
    r_clamped = solve_nu_pz(p4_vis, flight, m_parent=M_BC, m_miss=m_miss_bad,
                            clamp_negative_disc=True)
    assert r_default['discriminant'] < 0., 'expected a negative discriminant here'
    identical = (len(r_clamped['pz_solutions']) == 2
                 and r_clamped['pz_solutions'][0] == r_clamped['pz_solutions'][1])
    at_vertex = (len(r_clamped['pz_solutions']) == 2
                 and r_clamped['pz_solutions'][0] == r_clamped['pz_at_vertex'])
    print('infeasible m_miss=%.2f: disc=%.4e  (no real root)' %
          (m_miss_bad, r_default['discriminant']))
    print('  default              -> %d solution(s), clamped=%s'
          % (len(r_default['pz_solutions']), r_default['clamped']))
    print('  clamp_negative_disc  -> %d solution(s), clamped=%s, identical=%s, == vertex(%.5f)=%s'
          % (len(r_clamped['pz_solutions']), r_clamped['clamped'],
             identical, r_clamped['pz_at_vertex'], at_vertex))
    clamp_ok = (len(r_default['pz_solutions']) == 0
                and r_clamped['clamped'] and identical and at_vertex)

    # --- bonus: massive missing system (tau-mode style) closes when m_miss is known
    print('-' * 70)
    m_vis = 3.6
    m_miss = 1.2                                       # pretend 3-nu system mass
    E_vis_r = (M_BC**2 + m_vis**2 - m_miss**2) / (2 * M_BC)
    E_miss_r = (M_BC**2 + m_miss**2 - m_vis**2) / (2 * M_BC)
    p_star = np.sqrt(max(0., E_miss_r**2 - m_miss**2))
    d = np.array([-0.2, 0.9, 0.3]); d /= np.linalg.norm(d)
    p4_vis_r = np.array([E_vis_r,  p_star*d[0],  p_star*d[1],  p_star*d[2]])
    p4_miss_r = np.array([E_miss_r, -p_star*d[0], -p_star*d[1], -p_star*d[2]])
    p_bc = np.array([5.0, 12.0, -7.0])
    E_bc = np.sqrt(M_BC**2 + p_bc.dot(p_bc))
    beta = p_bc / E_bc
    p4_vis = _boost_rest_to_lab(p4_vis_r, beta)
    p4_miss = _boost_rest_to_lab(p4_miss_r, beta)
    n_hat = p_bc / np.linalg.norm(p_bc)
    pz_true = float(p4_miss[1:].dot(n_hat))
    sols = reconstruct(p4_vis, p_bc, m_parent=M_BC, m_miss=m_miss)
    best = pick_closest(sols, pz_true)
    print('massive-missing closure (m_miss=%.2f): dpz=%.2e  dmass=%.2e' %
          (m_miss, abs(best.pz - pz_true), abs(best.parent_mass - M_BC)))

    print('=' * 70)
    all_ok = (n_ok == n_try) and clamp_ok
    print('ALL OK' if all_ok else 'FAILURES PRESENT')
    print('=' * 70)
    return all_ok


if __name__ == '__main__':
    _selftest()
