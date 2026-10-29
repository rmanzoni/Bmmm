"""
gen_match_3mu.py

Match the three reco muons of an RJpsiCandidate to the generator-level muons of
the signal Bc decay:
  * the two muons from the charmonium  X -> mu+ mu-   (X = J/psi, and via
    feed-down psi(2S)/chi_c -> J/psi -> mu mu), tagged ROLE.JPSI
  * the bachelor muon, either a direct Bc daughter (Bc -> J/psi mu nu) or from
    tau -> mu in the Bc -> J/psi tau nu channel, tagged ROLE.BACHELOR

Complements RJPsiGenHistory.py and re-uses its Bc finder / copy-navigation.

Matching is one-to-one (a gen muon is used at most once), same charge by
default, within dr_max. After the call, each reco muon carries:
    mu.gen_role   in {ROLE.NONE, ROLE.JPSI, ROLE.BACHELOR}
    mu.gen_match  the matched reco::GenParticle (or None)
    mu.gen_dr     the matching deltaR (or NaN)

Looper usage (MC only, per selected candidate) is at the bottom of this file.
"""
from __future__ import print_function

from Bmmm.Analysis.RJPsiGenHistory import (
    last_copy, find_decayed_bc, GAMMA,
    JPSI, PSI2S, CHI_C0, CHI_C1, CHI_C2, H_C,
)

# deltaR: prefer the framework one, fall back to a local implementation so the
# module (and its __main__ self-test) can run outside CMSSW.
try:
    from PhysicsTools.HeppyCore.utils.deltar import deltaR
except ImportError:
    import math

    def deltaR(a, b):
        deta = a.eta() - b.eta()
        dphi = math.acos(max(-1., min(1., math.cos(a.phi() - b.phi()))))
        return math.sqrt(deta * deta + dphi * dphi)

CHARMONIA = {JPSI, PSI2S, CHI_C0, CHI_C1, CHI_C2, H_C}
MU  = 13
TAU = 15


class ROLE(object):
    NONE     = 0
    JPSI     = 1
    BACHELOR = 2


ROLE_NAME = {ROLE.NONE: 'none', ROLE.JPSI: 'jpsi', ROLE.BACHELOR: 'bachelor'}


# ----------------------------------------------------------------------------
# gen-side: extract the signal muons from the Bc decay
# ----------------------------------------------------------------------------
def _muon_daughters(p):
    return [last_copy(p.daughter(i)) for i in range(p.numberOfDaughters())
            if abs(p.daughter(i).pdgId()) == MU]


def dimuon_from_resonance(res, _depth=0, _maxdepth=8):
    '''Descend a charmonium subtree and return the two muons of the X->mu mu
    vertex. Handles J/psi->mu mu directly and feed-down (psi(2S)->J/psi pi pi,
    chi_c->J/psi gamma) by recursing through non-photon daughters.'''
    res = last_copy(res)
    mus = _muon_daughters(res)
    if len(mus) >= 2:
        return mus[:2]
    if _depth >= _maxdepth:
        return mus
    for i in range(res.numberOfDaughters()):
        d = res.daughter(i)
        if d.pdgId() == res.pdgId() or abs(d.pdgId()) == GAMMA:
            continue
        found = dimuon_from_resonance(d, _depth + 1, _maxdepth)
        if len(found) >= 2:
            return found
    return mus


def muon_from_tau(tau, _depth=0, _maxdepth=4):
    '''Return the muon from tau -> mu nu nu (None if the tau decayed otherwise).'''
    tau = last_copy(tau)
    for i in range(tau.numberOfDaughters()):
        d = tau.daughter(i)
        if abs(d.pdgId()) == MU:
            return last_copy(d)
    if _depth < _maxdepth:
        for i in range(tau.numberOfDaughters()):
            d = tau.daughter(i)
            if d.pdgId() == tau.pdgId():
                continue
            m = muon_from_tau(d, _depth + 1, _maxdepth)
            if m is not None:
                return m
    return None


def signal_gen_muons(genparticles):
    '''Find the signal-Bc muons and tag them by role.
    Returns None if no decayed Bc is found, else a dict containing:
        bc, jpsi_mus (<=2), bachelor_mu (or None), bachelor_from_tau (bool),
        charmonium (or None), targets = [(genmu, role), ...] ready to match.'''
    bcs = find_decayed_bc(genparticles)
    if not bcs:
        return None
    bc = last_copy(bcs[0])

    jpsi_mus          = []
    bachelor_mu       = None
    bachelor_from_tau = False
    charmonium        = None

    for i in range(bc.numberOfDaughters()):
        d = bc.daughter(i)
        if d.pdgId() == bc.pdgId():
            continue
        ad = abs(d.pdgId())
        if ad in CHARMONIA and not jpsi_mus:
            charmonium = last_copy(d)
            jpsi_mus   = dimuon_from_resonance(charmonium)
        elif ad == MU and bachelor_mu is None:
            bachelor_mu       = last_copy(d)
            bachelor_from_tau = False
        elif ad == TAU and bachelor_mu is None:
            mu = muon_from_tau(last_copy(d))
            if mu is not None:
                bachelor_mu       = mu
                bachelor_from_tau = True

    targets = [(gm, ROLE.JPSI) for gm in jpsi_mus]
    if bachelor_mu is not None:
        targets.append((bachelor_mu, ROLE.BACHELOR))

    return {
        'bc'                : bc,
        'jpsi_mus'          : jpsi_mus,
        'bachelor_mu'       : bachelor_mu,
        'bachelor_from_tau' : bachelor_from_tau,
        'charmonium'        : charmonium,
        'targets'           : targets,
    }


# ----------------------------------------------------------------------------
# reco-side: one-to-one match of the candidate's muons to the gen targets
# ----------------------------------------------------------------------------
def match_candidate_muons(cand, genparticles, dr_max=0.03, require_charge=True, info=None):
    '''Assign mu.gen_role / mu.gen_match / mu.gen_dr to cand.mu1,mu2,mu3.
    Returns a summary dict (found_bc, n_jpsi_matched, bachelor_matched,
    bachelor_from_tau, jpsi_reco_idx, bachelor_reco_idx).

    `info` is the dict returned by signal_gen_muons(genparticles). The gen side
    is identical for every candidate in an event, so the caller can compute it
    once per event and pass it in; if None it is computed here (back-compatible).'''
    reco_muons = cand.muons

    for rmu in reco_muons:
        rmu.gen_role  = ROLE.NONE
        rmu.gen_match = None
        rmu.gen_dr    = float('nan')

    if info is None:
        info = signal_gen_muons(genparticles)
    summary = {
        'found_bc'          : info is not None,
        'n_jpsi_matched'    : 0,
        'bachelor_matched'  : False,
        'bachelor_from_tau' : bool(info['bachelor_from_tau']) if info else False,
        'jpsi_reco_idx'     : [],
        'bachelor_reco_idx' : -1,
    }
    if info is None or not info['targets']:
        return summary

    targets = info['targets']

    # all admissible (dr, reco_idx, gen_idx) pairs, then greedy one-to-one
    pairs = []
    for ri, rmu in enumerate(reco_muons):
        for gi, (gmu, role) in enumerate(targets):
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
        gmu, role = targets[gi]
        reco_muons[ri].gen_role  = role
        reco_muons[ri].gen_match = gmu
        reco_muons[ri].gen_dr    = dr

    summary['n_jpsi_matched']    = sum(1 for r in reco_muons if r.gen_role == ROLE.JPSI)
    summary['bachelor_matched']  = any(r.gen_role == ROLE.BACHELOR for r in reco_muons)
    summary['jpsi_reco_idx']     = [ri for ri, r in enumerate(reco_muons) if r.gen_role == ROLE.JPSI]
    summary['bachelor_reco_idx'] = next((ri for ri, r in enumerate(reco_muons)
                                         if r.gen_role == ROLE.BACHELOR), -1)
    return summary


# ----------------------------------------------------------------------------
# minimal self-test (runs with the deltaR fallback; no CMSSW needed):
#     python -m Bmmm.Analysis.gen_match_3mu
# ----------------------------------------------------------------------------
if __name__ == '__main__':

    class FP(object):
        def __init__(self, pdg, charge=0, eta=0., phi=0., pt=10., status=2, dau=None):
            self._pdg, self._q, self._eta, self._phi = pdg, charge, eta, phi
            self._pt, self._st, self._d = pt, status, dau or []
        def pdgId(self):            return self._pdg
        def charge(self):           return self._q
        def eta(self):              return self._eta
        def phi(self):              return self._phi
        def pt(self):               return self._pt
        def status(self):           return self._st
        def numberOfDaughters(self):return len(self._d)
        def daughter(self, i):      return self._d[i]

    class Cand(object):
        def __init__(self, m1, m2, m3):
            self.mu1, self.mu2, self.mu3 = m1, m2, m3

    # Bc- -> J/psi mu- nu ; J/psi -> mu+ mu-
    mup  = FP(-13, +1, 0.50, 0.50)
    mum  = FP( 13, -1, 0.55, 0.52)
    jpsi = FP(443,  0, 0.52, 0.51, dau=[mup, mum])
    bach = FP( 13, -1, 1.20, 2.00)
    bc   = FP(-541, -1, 0.60, 0.60, dau=[jpsi, bach, FP(-14, 0)])
    gen  = [bc, jpsi, mup, mum, bach]
    info = signal_gen_muons(gen)
    assert info and len(info['jpsi_mus']) == 2 and info['bachelor_mu'] is bach
    assert not info['bachelor_from_tau']

    cand = Cand(FP(-13, +1, 0.50, 0.50), FP(13, -1, 0.55, 0.52), FP(13, -1, 1.205, 2.001))
    s = match_candidate_muons(cand, gen, dr_max=0.05)
    assert cand.mu1.gen_role == ROLE.JPSI and cand.mu2.gen_role == ROLE.JPSI
    assert cand.mu3.gen_role == ROLE.BACHELOR
    assert s['n_jpsi_matched'] == 2 and s['bachelor_matched']
    print('test 1  Bc->J/psi mu nu : jpsi idx', s['jpsi_reco_idx'],
          'bachelor idx', s['bachelor_reco_idx'], 'OK')

    # Bc- -> J/psi tau- nu ; tau- -> mu- nu nu
    mup2  = FP(-13, +1, -0.30, 1.00)
    mum2  = FP( 13, -1, -0.28, 1.02)
    jpsi2 = FP(443,  0, -0.29, 1.01, dau=[mup2, mum2])
    taumu = FP( 13, -1,  0.90, -1.50)
    tau   = FP( 15, -1,  0.80, -1.40, dau=[taumu, FP(16, 0), FP(-14, 0)])
    bc2   = FP(-541, -1, 0.20, -0.50, dau=[jpsi2, tau, FP(-16, 0)])
    gen2  = [bc2, jpsi2, mup2, mum2, tau, taumu]
    info2 = signal_gen_muons(gen2)
    assert info2['bachelor_from_tau'] and info2['bachelor_mu'] is taumu
    print('test 2  Bc->J/psi tau nu (tau->mu) : bachelor_from_tau =',
          info2['bachelor_from_tau'], 'OK')

    print('all smoke tests passed')
