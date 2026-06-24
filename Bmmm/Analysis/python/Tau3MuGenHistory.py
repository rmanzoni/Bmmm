import numpy as np
from PhysicsTools.HeppyCore.utils.deltar import deltaR

# ----------------------------------------------------------------------------
# Gen truth for the displaced tau -> 3mu signal
#     Ds -> tau nu , tau -> mu a , a -> mu mu        (a long-lived, pdgId 9900015)
#
# Tau3MuGenDecay.from_genparticles(prunedGenParticles) walks the chain and stores
# the gen Ds, tau, a and the three gen muons, tagged by ROLE:
#     'mu_a'   : a muon from the a -> mu mu decay  (the displaced OS pair)
#     'mu_tau' : the bachelor muon from tau -> mu a
# match_candidate_muons() then attaches the nearest gen muon to each reco muon of
# a candidate (mu.gen_match / mu.gen_role / mu.gen_dr), exactly like the rjpsi
# matcher, so the per-muon gen_* branches are filled for signal MC.
# ----------------------------------------------------------------------------

A_PDGID   = 9900015   # the long-lived scalar a
TAU_PDGID = 15
DS_PDGID  = 431
MU_PDGID  = 13

ROLE = {'mu_a': 1, 'mu_tau': 2}   # numeric encoding stored in the ntuple


def _last_copy(p):
    '''Follow the same-pdgId daughter chain to the last copy of a particle.'''
    out = p
    changed = True
    while changed:
        changed = False
        for idau in range(out.numberOfDaughters()):
            dau = out.daughter(idau)
            if dau.pdgId() == out.pdgId():
                out = dau
                changed = True
                break
    return out


def _daughters(p, pdgid_abs):
    '''Direct daughters of p with |pdgId| == pdgid_abs (FSR photons ignored).'''
    out = []
    for idau in range(p.numberOfDaughters()):
        dau = p.daughter(idau)
        if abs(dau.pdgId()) == pdgid_abs:
            out.append(dau)
    return out


class Tau3MuGenDecay(object):

    def __init__(self, ds, tau, a, mu_a, mu_tau):
        self.ds     = ds        # the Ds (None if not found)
        self.tau    = tau       # the tau
        self.a      = a         # the long-lived scalar a
        self.mu_a   = mu_a      # list of the two muons from a -> mu mu
        self.mu_tau = mu_tau    # the bachelor muon from tau -> mu a

        # flat, role-tagged list used by the matcher
        self.gen_muons = []
        for imu in mu_a:
            imu.role = 'mu_a'
            self.gen_muons.append(imu)
        if mu_tau is not None:
            mu_tau.role = 'mu_tau'
            self.gen_muons.append(mu_tau)

    @classmethod
    def from_genparticles(cls, genparticles):
        '''
        Return a Tau3MuGenDecay for the first  tau -> mu a (a -> mu mu)  found,
        or None if the chain is not present (e.g. background MC / data).
        '''
        for gp in genparticles:
            if abs(gp.pdgId()) != TAU_PDGID:
                continue
            tau = _last_copy(gp)

            a_list = _daughters(tau, A_PDGID)
            mu_tau_list = _daughters(tau, MU_PDGID)
            if len(a_list) != 1 or len(mu_tau_list) != 1:
                continue

            a = _last_copy(a_list[0])
            mu_a = _daughters(a, MU_PDGID)
            if len(mu_a) != 2:
                continue

            # the Ds mother (tau <- Ds), if present
            ds = None
            for imom in range(gp.numberOfMothers()):
                mom = gp.mother(imom)
                if abs(mom.pdgId()) == DS_PDGID:
                    ds = mom
                    break

            return cls(ds, tau, a, list(mu_a), mu_tau_list[0])

        return None


##########################################################################################
def match_candidate_muons(cand, genparticles, dr_max=0.03, info=None):
    '''
    Attach the closest gen signal muon to each of the candidate's three reco muons:
        imu.gen_match : the matched gen muon (absent if no match within dr_max)
        imu.gen_role  : ROLE code of the match, np.nan if unmatched
        imu.gen_dr    : dR of the match,        np.nan if unmatched
    `info` is a pre-built Tau3MuGenDecay (computed once per event); if None it is
    built here. Each gen muon is used at most once (closest reco muon wins).
    '''
    if info is None:
        info = Tau3MuGenDecay.from_genparticles(genparticles)

    for imu in cand.muons:
        imu.gen_role = np.nan
        imu.gen_dr   = np.nan
        if hasattr(imu, 'gen_match'):
            del imu.gen_match

    if info is None or len(info.gen_muons) == 0:
        return

    # greedy closest-pair assignment over (reco muon, gen muon) within dr_max
    pairs = []
    for imu in cand.muons:
        for gmu in info.gen_muons:
            dr = deltaR(imu.eta(), imu.phi(), gmu.eta(), gmu.phi())
            if dr < dr_max:
                pairs.append((dr, imu, gmu))
    pairs.sort(key=lambda x: x[0])

    used_reco, used_gen = set(), set()
    for dr, imu, gmu in pairs:
        if id(imu) in used_reco or id(gmu) in used_gen:
            continue
        imu.gen_match = gmu
        imu.gen_role  = ROLE[gmu.role]
        imu.gen_dr    = dr
        used_reco.add(id(imu))
        used_gen.add(id(gmu))
