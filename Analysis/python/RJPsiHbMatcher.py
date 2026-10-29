"""
RJPsiHbMatcher.py

Gen matching for the HbToJpsiX background, where a b hadron produces a
J/psi(-> mu+ mu-) and the reconstructed bachelor muon may originate either from
the SAME b hadron as the J/psi (genuine Hb -> J/psi (+) mu (+X)) or from a
DIFFERENT source -- the other b in the bbbar event, a light-flavour decay, or a
fake. The latter two are the (combinatorial) background this measurement has to
control.

Unlike RJPsiMuonMatcher (hard-wired to the signal Bc decay, which it finds by
walking DOWN a known decay tree), here the J/psi can come from ANY b hadron,
directly or via charmonium feed-down (psi(2S)/chi_c -> J/psi), so the b-hadron
ancestor is found by walking UP the mother chain.

Per RJpsiCandidate convention:
    cand.jpsi_muons : the two reco muons forming the J/psi (pt-sorted)
    cand.mu         : the reco bachelor muon

After match_hb_candidate(cand, genparticles):
  per reco muon (role-agnostic reco<->gen match, so the mu*_gen_* branches fill):
    mu.gen_match  : matched reco::GenParticle (or None)
    mu.gen_dr     : matching deltaR (or NaN)
    mu.gen_role   : ROLE.NONE (signal roles are not asserted for Hb)
  per candidate (the new Hb-truth branches consume these):
    cand.gen_hb_same_mother      : 1.0 same b ancestor / 0.0 different / NaN undefined
    cand.gen_jpsi_b              : gen b-hadron ancestor of the J/psi (or None)
    cand.gen_bachelor_b          : gen b-hadron ancestor of the bachelor mu (or None)
    cand.gen_bachelor_mother     : immediate gen mother of the bachelor mu (or None)

The gen side (the status-1 gen-muon list) is identical for every candidate in an
event, so the caller can build it once with hb_status1_muons(genparticles) and
pass it in via gen_muons=...; if None it is rebuilt here (back-compatible).
"""
from __future__ import print_function

import math

# is_b_hadron: prefer the framework helper (uses the `particle` package); fall
# back to a PDG-digit rule so this module (and its __main__ self-test) runs
# outside CMSSW.
try:
    from Bmmm.Analysis.utils import is_b_hadron
except Exception:
    def is_b_hadron(pdgid):
        '''True if |pdgId| is a hadron carrying a b quark (digit 5 among the
        quark digits, i.e. excluding the rightmost 2J+1 digit).'''
        apid = abs(int(pdgid))
        if apid < 100:                 # leptons, gauge bosons, quarks: not hadrons
            return False
        q = apid // 10                 # drop the spin (2J+1) digit
        while q >= 10:
            if q % 10 == 5:
                return True
            q //= 10
        return q == 5

# deltaR: prefer the framework one, fall back to a local implementation.
try:
    from PhysicsTools.HeppyCore.utils.deltar import deltaR
except ImportError:
    def deltaR(a, b):
        deta = a.eta() - b.eta()
        dphi = math.acos(max(-1., min(1., math.cos(a.phi() - b.phi()))))
        return math.sqrt(deta * deta + dphi * dphi)

# reuse the role enum so mu.gen_role stays consistent across the two matchers.
try:
    from Bmmm.Analysis.RJPsiMuonMatcher import ROLE
except Exception:
    class ROLE(object):
        NONE = 0
        JPSI = 1
        BACHELOR = 2

MU = 13


# ----------------------------------------------------------------------------
# gen-particle identity & ancestry
# ----------------------------------------------------------------------------
def _gp_key(p):
    '''Stable per-event identity for a gen particle. Two PyROOT proxies of the
    SAME stored particle return identical momentum/vertex floats, while two
    distinct b hadrons in a bbbar event do not, so this is a safe substitute for
    object identity (which `is` does not provide across proxy re-reads).'''
    return (p.pdgId(), p.px(), p.py(), p.pz(), p.vx(), p.vy(), p.vz())


def same_object(a, b):
    return a is not None and b is not None and _gp_key(a) == _gp_key(b)


def _ancestor(p, pred, _guard=200):
    '''Walk strictly upward via mother(0) and return the first ancestor (not p
    itself) satisfying pred, or None.'''
    node = p
    for _ in range(_guard):
        if node.numberOfMothers() == 0:
            return None
        node = node.mother(0)
        if pred(node):
            return node
    return None


def b_hadron_ancestor(p):
    '''The closest b-hadron ancestor of p (None if p has none).'''
    if p is None:
        return None
    return _ancestor(p, lambda x: is_b_hadron(x.pdgId()))


# ----------------------------------------------------------------------------
# gen side: status-1 gen muons (built once per event)
# ----------------------------------------------------------------------------
def hb_status1_muons(genparticles):
    return [p for p in genparticles
            if abs(p.pdgId()) == MU and p.status() == 1]


# ----------------------------------------------------------------------------
# reco side: greedy one-to-one reco<->gen muon match (same scheme as the signal
# matcher), then resolve the b-hadron ancestry of the J/psi and the bachelor.
# ----------------------------------------------------------------------------
def match_hb_candidate(cand, genparticles, dr_max=0.04, require_charge=True,
                       gen_muons=None):
    '''Tag cand.mu / cand.jpsi_muons with gen_match/gen_dr/gen_role, then set the
    candidate-level Hb-truth attributes (see module docstring). Returns a summary
    dict for optional logging / cutflow.'''
    reco_muons = cand.muons
    for rmu in reco_muons:
        rmu.gen_role  = ROLE.NONE
        rmu.gen_match = None
        rmu.gen_dr    = float('nan')

    # candidate-level defaults (so the getters are always safe)
    cand.gen_hb_same_mother  = float('nan')
    cand.gen_jpsi_b          = None
    cand.gen_bachelor_b      = None
    cand.gen_bachelor_mother = None

    summary = {
        'n_matched'    : 0,
        'jpsi_has_b'   : False,
        'bachelor_has_b': False,
        'same_mother'  : float('nan'),
    }

    if gen_muons is None:
        gen_muons = hb_status1_muons(genparticles)
    if not gen_muons:
        return summary

    # all admissible (dr, reco_idx, gen_idx) pairs, then greedy one-to-one
    pairs = []
    for ri, rmu in enumerate(reco_muons):
        for gi, gmu in enumerate(gen_muons):
            if require_charge and rmu.charge() != gmu.charge():
                continue
            dr = deltaR(rmu, gmu)
            if dr < dr_max:
                pairs.append((dr, ri, gi))
    pairs.sort(key=lambda x: x[0])

    used_r, used_g = set(), set()
    for dr, ri, gi in pairs:
        if ri in used_r or gi in used_g:
            continue
        used_r.add(ri)
        used_g.add(gi)
        reco_muons[ri].gen_match = gen_muons[gi]
        reco_muons[ri].gen_dr    = dr
    summary['n_matched'] = len(used_r)

    # ---- resolve ancestry using the candidate's own role assignment ----------
    # the J/psi b mother is well defined only if BOTH reco J/psi muons matched a
    # gen muon and those two gen muons share the same b-hadron ancestor (i.e. the
    # reco dimuon really is one b-hadron's J/psi, feed-down included).
    g_jpsi0 = getattr(cand.jpsi_muons[0], 'gen_match', None)
    g_jpsi1 = getattr(cand.jpsi_muons[1], 'gen_match', None)
    g_bach  = getattr(cand.mu,            'gen_match', None)

    b_jpsi0 = b_hadron_ancestor(g_jpsi0)
    b_jpsi1 = b_hadron_ancestor(g_jpsi1)
    jpsi_b  = b_jpsi0 if same_object(b_jpsi0, b_jpsi1) else None

    bach_b  = b_hadron_ancestor(g_bach)

    cand.gen_jpsi_b      = jpsi_b
    cand.gen_bachelor_b  = bach_b
    cand.gen_bachelor_mother = (g_bach.mother(0)
                                if (g_bach is not None and g_bach.numberOfMothers() > 0)
                                else None)

    if jpsi_b is not None:
        # 1.0 iff the bachelor traces to the SAME b hadron; 0.0 otherwise
        # (different b, or bachelor not from a b at all -> combinatorial).
        cand.gen_hb_same_mother = 1.0 if same_object(jpsi_b, bach_b) else 0.0
    # else: J/psi b mother undefined -> leave NaN

    summary['jpsi_has_b']     = jpsi_b is not None
    summary['bachelor_has_b'] = bach_b is not None
    summary['same_mother']    = cand.gen_hb_same_mother
    return summary


# ----------------------------------------------------------------------------
# minimal self-test (pure python, no CMSSW): python -m Bmmm.Analysis.RJPsiHbMatcher
# ----------------------------------------------------------------------------
if __name__ == '__main__':

    class FP(object):
        def __init__(self, pdg, charge=0, eta=0., phi=0., pt=10., status=1,
                     p=(0., 0., 0.), v=(0., 0., 0.), mothers=None):
            self._pdg, self._q, self._eta, self._phi = pdg, charge, eta, phi
            self._pt, self._st = pt, status
            self._px, self._py, self._pz = p
            self._vx, self._vy, self._vz = v
            self._m = mothers or []
        def pdgId(self):             return self._pdg
        def charge(self):            return self._q
        def eta(self):               return self._eta
        def phi(self):               return self._phi
        def pt(self):                return self._pt
        def status(self):            return self._st
        def px(self):                return self._px
        def py(self):                return self._py
        def pz(self):                return self._pz
        def vx(self):                return self._vx
        def vy(self):                return self._vy
        def vz(self):                return self._vz
        def numberOfMothers(self):   return len(self._m)
        def mother(self, i):         return self._m[i]

    class Cand(object):
        def __init__(self, j0, j1, bach):
            self.jpsi_muons = [j0, j1]
            self.mu = bach
            self.muons = [j0, j1, bach]

    # sanity on the fallback b-hadron rule
    assert is_b_hadron(521) and is_b_hadron(531) and is_b_hadron(541)
    assert is_b_hadron(5122) and not is_b_hadron(443) and not is_b_hadron(211)

    # --- scenario A: SAME mother -------------------------------------------
    # B+ -> J/psi(->mu+ mu-) + D0bar(-> mu- ...) : bachelor mu and J/psi share B+
    Bp   = FP(521,  +1, p=(10., 0., 30.), v=(0., 0., 0.))
    jpsiA= FP(443,   0, p=( 5., 0., 15.), v=(0.1, 0., 0.5), mothers=[Bp])
    muAp = FP(-13,  +1, 0.50, 0.50, mothers=[jpsiA])
    muAm = FP( 13,  -1, 0.55, 0.52, mothers=[jpsiA])
    DA   = FP(-421,  0, p=( 3., 0., 10.), v=(0.1, 0., 0.5), mothers=[Bp])
    muAb = FP( 13,  -1, 1.20, 2.00, mothers=[DA])
    genA = [Bp, jpsiA, muAp, muAm, DA, muAb]

    candA = Cand(FP(-13, +1, 0.50, 0.50), FP(13, -1, 0.55, 0.52),
                 FP(13, -1, 1.205, 2.001))
    sA = match_hb_candidate(candA, genA, dr_max=0.05)
    assert candA.gen_hb_same_mother == 1.0, candA.gen_hb_same_mother
    assert candA.gen_jpsi_b is not None and candA.gen_bachelor_b is not None
    assert _gp_key(candA.gen_jpsi_b) == _gp_key(Bp)
    print('scenario A  same B    : same_mother =', candA.gen_hb_same_mother,
          ' jpsi_b pdg', candA.gen_jpsi_b.pdgId(),
          ' bach_b pdg', candA.gen_bachelor_b.pdgId(), 'OK')

    # --- scenario B: DIFFERENT mother (the other b) ------------------------
    B1   = FP(521,  +1, p=(10., 0., 30.), v=(0., 0., 0.))
    jpsiB= FP(443,   0, p=( 5., 0., 15.), v=(0.1, 0., 0.5), mothers=[B1])
    muBp = FP(-13,  +1, -0.30, 1.00, mothers=[jpsiB])
    muBm = FP( 13,  -1, -0.28, 1.02, mothers=[jpsiB])
    B2   = FP(531,  +1, p=(-8., 2., -20.), v=(0., 0., 0.))   # distinct momentum
    muBb = FP( 13,  -1,  0.90, -1.50, mothers=[B2])
    genB = [B1, jpsiB, muBp, muBm, B2, muBb]

    candB = Cand(FP(-13, +1, -0.30, 1.00), FP(13, -1, -0.28, 1.02),
                 FP(13, -1, 0.905, -1.501))
    sB = match_hb_candidate(candB, genB, dr_max=0.05)
    assert candB.gen_hb_same_mother == 0.0, candB.gen_hb_same_mother
    assert _gp_key(candB.gen_jpsi_b) == _gp_key(B1)
    assert _gp_key(candB.gen_bachelor_b) == _gp_key(B2)
    print('scenario B  other b   : same_mother =', candB.gen_hb_same_mother,
          ' jpsi_b pdg', candB.gen_jpsi_b.pdgId(),
          ' bach_b pdg', candB.gen_bachelor_b.pdgId(), 'OK')

    # --- scenario C: bachelor from a light hadron (no b ancestor) ----------
    B3   = FP(521,  +1, p=(10., 0., 30.), v=(0., 0., 0.))
    jpsiC= FP(443,   0, p=( 5., 0., 15.), v=(0.1, 0., 0.5), mothers=[B3])
    muCp = FP(-13,  +1, 0.10, 0.10, mothers=[jpsiC])
    muCm = FP( 13,  -1, 0.12, 0.12, mothers=[jpsiC])
    piC  = FP(211,  +1, p=(1., 0., 2.), v=(0.3, 0., 1.))     # no b ancestor
    muCb = FP( 13,  -1, 1.50, -2.00, mothers=[piC])
    genC = [B3, jpsiC, muCp, muCm, piC, muCb]

    candC = Cand(FP(-13, +1, 0.10, 0.10), FP(13, -1, 0.12, 0.12),
                 FP(13, -1, 1.505, -2.001))
    sC = match_hb_candidate(candC, genC, dr_max=0.05)
    assert candC.gen_hb_same_mother == 0.0, candC.gen_hb_same_mother
    assert candC.gen_jpsi_b is not None and candC.gen_bachelor_b is None
    print('scenario C  light mu  : same_mother =', candC.gen_hb_same_mother,
          ' jpsi_b pdg', candC.gen_jpsi_b.pdgId(), ' bach_b None OK')

    # --- scenario D: only one J/psi muon matched -> J/psi b undefined (NaN) -
    candD = Cand(FP(-13, +1, 0.10, 0.10), FP(13, -1, 9.0, 9.0),   # 2nd: no match
                 FP(13, -1, 1.505, -2.001))
    sD = match_hb_candidate(candD, genC, dr_max=0.05)
    assert candD.gen_hb_same_mother != candD.gen_hb_same_mother, 'expected NaN'  # NaN
    assert candD.gen_jpsi_b is None
    print('scenario D  fake jpsi : same_mother = NaN (jpsi b undefined) OK')

    print('all smoke tests passed')
