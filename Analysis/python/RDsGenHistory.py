'''
Generator truth for R(Ds): Bs -> Ds(*) (mu | tau) nu with Ds -> phi(K K) pi.

Pure python on the reco::GenParticle interface (pdgId, status, charge, pt,
eta, phi, energy, px, py, pz, vx, vy, vz, numberOfMothers, mother,
numberOfDaughters, daughter), so it runs on stand-ins in the unit tests
(test/rds/test_rds_kinematics.py).

Two levels.

EVENT level -- BsSignalDecay.from_genparticles(prunedGenParticles)
  Looks for a decayed Bs whose direct daughters (photons dropped) are
  Ds(*) + (mu | tau) + nu, preferring one with Ds -> phi pi and phi -> K K.
  signal code (gen_sig):
      0  Bs -> Ds   mu  nu        1  Bs -> Ds*  mu  nu
      2  Bs -> Ds   tau nu        3  Bs -> Ds*  tau nu
     -1  no such Bs in the event
  plus the Bs kinematics (lifetime and FONLL reweighting), the gen-level
  analysis quantities (q2, m_miss2, E*_mu, helicity angles) and the pre-FSR
  four-momenta Hammer needs.

CANDIDATE level -- match_candidate(cand_tracks, gen_charged)
  Matches the four reco tracks (k1, k2, pi, mu) to status-1 gen particles and
  classifies the candidate as in AN Sec. 6.1:
      0..3   the signal channels above (the Ds and the mu of the candidate are
             the ones of the signal decay)
      10     Bs -> Ds + mu, feed-down  (mu, or its tau, a direct Bs daughter)
      11     Bs -> Ds + mu, double charm (any other mu from the same Bs)
      12     B0 -> Ds + mu              13  B+ -> Ds + mu
      14     Lambda_b -> Ds + mu        15  other b hadron -> Ds + mu
      20     true Ds, mu from a different b hadron
      21     true Ds, mu not from a b hadron
      30     the three tracks are not the daughters of one Ds
     -1      at least one of the four tracks is not matched
'''

import numpy as np

from Bmmm.Analysis.RDsKinematics import (
    np4, mass2, semileptonic_observables, helicity_angles,
)

BS, B0, BP, LB = 531, 511, 521, 5122
DS, DSST, PHI  = 431, 433, 333
KAON, PION, MU, TAU, GAMMA, PI0 = 321, 211, 13, 15, 22, 111
NU_MU, NU_TAU = 14, 16
NEUTRINOS = (12, 14, 16)
CHARGED_STABLE = (11, 13, 211, 321, 2212)

SIGNAL_SIGNATURES = {
    (13, 14, DS)   : 0,
    (13, 14, DSST) : 1,
    (15, 16, DS)   : 2,
    (15, 16, DSST) : 3,
}
SIGNAL_NAMES = {0: 'Ds_mu_nu', 1: 'Dsst_mu_nu', 2: 'Ds_tau_nu', 3: 'Dsst_tau_nu', -1: 'none'}

NAN = float('nan')


def is_b_hadron(pdgid):
    '''b mesons (hundreds digit 5, not bottomonium) and b baryons (thousands
    digit 5). Digit rule, no lookup table.'''
    a = abs(int(pdgid)) % 10000
    if a >= 1000:
        return (a // 1000) % 10 == 5
    return (a // 100) % 10 == 5 and (a // 10) % 10 != 5


def is_charm_hadron(pdgid):
    a = abs(int(pdgid)) % 10000
    if a >= 1000:
        return (a // 1000) % 10 == 4
    return (a // 100) % 10 == 4 and (a // 10) % 10 != 4


##########################################################################################
#####      NAVIGATION
##########################################################################################

def key(p):
    '''Per-event identity of a gen particle (two proxies of the same stored
    particle return identical floats). Same idea as RJPsiHbMatcher._gp_key.'''
    return (p.pdgId(), p.px(), p.py(), p.pz(), p.vx(), p.vy(), p.vz())


def same(a, b):
    return a is not None and b is not None and key(a) == key(b)


def first_copy(p, _guard=200):
    '''Walk up through mothers with the same |pdgId| (copies, and B mixing,
    which flips the sign) to the first copy.'''
    for _ in range(_guard):
        if p.numberOfMothers() == 0:
            return p
        m = p.mother(0)
        if abs(m.pdgId()) != abs(p.pdgId()):
            return p
        p = m
    return p


def last_copy(p, _guard=200):
    '''Walk down through daughters with the same |pdgId| to the copy that
    decays (B mixing included).'''
    for _ in range(_guard):
        nxt = None
        for i in range(p.numberOfDaughters()):
            d = p.daughter(i)
            if abs(d.pdgId()) == abs(p.pdgId()):
                nxt = d
                break
        if nxt is None:
            return p
        p = nxt
    return p


def mother_of(p):
    '''The real mother: the mother of the first copy (None if none).'''
    fc = first_copy(p)
    return fc.mother(0) if fc.numberOfMothers() > 0 else None


def ancestor(p, pred, _guard=200):
    '''First strict ancestor (via mother(0)) satisfying pred, or None.'''
    node = p
    for _ in range(_guard):
        if node.numberOfMothers() == 0:
            return None
        node = node.mother(0)
        if pred(node):
            return node
    return None


def b_ancestor(p):
    '''Closest b-hadron ancestor, taken at its LAST copy so that two particles
    from the same decaying b give the same object even through B mixing.'''
    b = ancestor(p, lambda x: is_b_hadron(x.pdgId()))
    return last_copy(b) if b is not None else None


def direct_daughters(p, keep_photons=False):
    lc = last_copy(p)
    out = []
    for i in range(lc.numberOfDaughters()):
        d = lc.daughter(i)
        if not keep_photons and abs(d.pdgId()) == GAMMA:
            continue
        out.append(d)
    return out


def signature(p):
    return tuple(sorted(abs(d.pdgId()) for d in direct_daughters(p)))


def ds_decay(ds):
    '''(phi, pi, k_plus, k_minus) of a Ds -> phi(K K) pi decay at their first
    copies (pre-FSR), or None if the Ds decays otherwise.'''
    daus = direct_daughters(ds)
    phis = [d for d in daus if abs(d.pdgId()) == PHI]
    pis  = [d for d in daus if abs(d.pdgId()) == PION]
    if len(daus) != 2 or len(phis) != 1 or len(pis) != 1:
        return None
    ks = [d for d in direct_daughters(phis[0]) if abs(d.pdgId()) == KAON]
    if len(ks) != 2 or len(direct_daughters(phis[0])) != 2:
        return None
    kp = ks[0] if ks[0].charge() > 0 else ks[1]
    km = ks[1] if ks[0].charge() > 0 else ks[0]
    return phis[0], pis[0], kp, km


def tau_muon(tau):
    '''(mu, nu_tau, nu_mu) of tau -> mu nu nu, or None.'''
    daus = direct_daughters(tau)
    ids = sorted(abs(d.pdgId()) for d in daus)
    if ids != [MU, NU_MU, NU_TAU]:
        return None
    pick = dict((abs(d.pdgId()), d) for d in daus)
    return pick[MU], pick[NU_TAU], pick[NU_MU]


##########################################################################################
#####      EVENT LEVEL: the signal Bs decay
##########################################################################################

class BsSignalDecay(object):
    '''The signal decay of the event (see module docstring). Attributes:
        code        0..3, -1
        bs          the decaying Bs (last copy); bs_first its first copy
        dsx         the direct Ds or Ds* daughter (first copy)
        ds          the Ds (dsx itself, or the Ds from Ds*)
        dsg         the photon / pi0 of Ds* -> Ds (None for Ds)
        lep, nu     the direct lepton (mu or tau) and neutrino
        mu          the visible muon (lep, or the mu from the tau)
        tau_nus     (nu_tau, nu_mu) of the tau decay, or None
        phi, pi, kp, km   of Ds -> phi pi (None if the Ds decays otherwise)
    '''

    def __init__(self):
        self.code = -1
        self.bs = self.bs_first = self.dsx = self.ds = self.dsg = None
        self.lep = self.nu = self.mu = self.tau_nus = None
        self.phi = self.pi = self.kp = self.km = None
        self.n_bs = 0

    @classmethod
    def from_genparticles(cls, genparticles):
        out = cls()
        seen, cands = set(), []
        for p in genparticles:
            if abs(p.pdgId()) != BS or p.numberOfDaughters() == 0:
                continue
            lc = last_copy(p)
            if key(lc) in seen:
                continue
            seen.add(key(lc))
            cands.append(lc)
        out.n_bs = len(cands)

        best, best_rank = None, -1
        for bs in cands:
            sig = signature(bs)
            sig_nonu = tuple(x for x in sig)
            code = SIGNAL_SIGNATURES.get(sig_nonu, -1)
            if code < 0:
                continue
            daus = direct_daughters(bs)
            dsx  = [d for d in daus if abs(d.pdgId()) in (DS, DSST)][0]
            lep  = [d for d in daus if abs(d.pdgId()) in (MU, TAU)][0]
            nu   = [d for d in daus if abs(d.pdgId()) in NEUTRINOS][0]
            if abs(dsx.pdgId()) == DSST:
                dd  = direct_daughters(dsx, keep_photons=True)
                dss = [d for d in dd if abs(d.pdgId()) == DS]
                if not dss:
                    continue
                ds  = dss[0]
                gs  = [d for d in dd if abs(d.pdgId()) in (GAMMA, PI0)]
                dsg = gs[0] if gs else None
            else:
                ds, dsg = dsx, None
            hadr = ds_decay(ds)
            if abs(lep.pdgId()) == TAU:
                tm = tau_muon(lep)
                mu, tau_nus = (tm[0], tm[1:]) if tm is not None else (None, None)
            else:
                mu, tau_nus = lep, None
            # rank: visible final state complete > Ds -> phi pi only > neither
            rank = 2 * (hadr is not None) + (mu is not None)
            if rank > best_rank:
                best_rank = rank
                best = (bs, code, dsx, ds, dsg, lep, nu, mu, tau_nus, hadr)

        if best is None:
            return out
        (out.bs, out.code, out.dsx, out.ds, out.dsg, out.lep, out.nu,
         out.mu, out.tau_nus, hadr) = best
        out.bs_first = first_copy(out.bs)
        if hadr is not None:
            out.phi, out.pi, out.kp, out.km = hadr
        return out

    @property
    def visible_complete(self):
        return self.code >= 0 and self.phi is not None and self.mu is not None


def bs_gen_info(decay, gen_pv=None):
    '''Bs kinematics and flight, for the lifetime and FONLL corrections.
    Production point: gen_pv when given (genParticles:xyz0), else the vertex of
    the first copy. Decay point: the vertex of the daughters of the last copy.'''
    out = dict.fromkeys(('gen_bs_pt', 'gen_bs_eta', 'gen_bs_phi', 'gen_bs_y', 'gen_bs_mass',
                         'gen_bs_pdgid', 'gen_bs_beta', 'gen_bs_gamma',
                         'gen_bs_lxyz', 'gen_bs_lxy', 'gen_bs_ct',
                         'gen_bs_sv_x', 'gen_bs_sv_y', 'gen_bs_sv_z'), NAN)
    if decay is None or decay.bs is None:
        return out
    bs = decay.bs
    p  = np4(bs)
    m  = np.sqrt(max(mass2(p), 0.))
    pmag = np.sqrt(p[1] ** 2 + p[2] ** 2 + p[3] ** 2)
    out['gen_bs_pt']    = bs.pt()
    out['gen_bs_eta']   = bs.eta()
    out['gen_bs_phi']   = bs.phi()
    out['gen_bs_y']     = 0.5 * np.log((p[0] + p[3]) / (p[0] - p[3])) if p[0] > abs(p[3]) else NAN
    out['gen_bs_mass']  = m
    out['gen_bs_pdgid'] = decay.bs_first.pdgId()
    out['gen_bs_beta']  = pmag / p[0] if p[0] > 0 else NAN
    out['gen_bs_gamma'] = p[0] / m if m > 0 else NAN
    if bs.numberOfDaughters() > 0:
        d = bs.daughter(0)
        sv = np.array([d.vx(), d.vy(), d.vz()])
        if gen_pv is not None and np.all(np.isfinite(gen_pv)):
            pv = np.asarray(gen_pv, dtype=float)
        else:
            fc = decay.bs_first
            pv = np.array([fc.vx(), fc.vy(), fc.vz()])
        dist = sv - pv
        out['gen_bs_sv_x'], out['gen_bs_sv_y'], out['gen_bs_sv_z'] = sv
        out['gen_bs_lxyz'] = float(np.sqrt(np.dot(dist, dist)))
        out['gen_bs_lxy']  = float(np.hypot(dist[0], dist[1]))
        # proper decay length c*t = L m / |p|
        out['gen_bs_ct']   = out['gen_bs_lxyz'] * m / pmag if pmag > 0 else NAN
    return out


GEN_OBS_NAMES = ('q2', 'm_miss2', 'pt_miss', 'e_star_mu', 'e_hash_mu',
                 'cos_theta_k_pi', 'cos_theta_phi_ds', 'cos_theta_mu_w', 'chi')


def gen_observables(decay):
    '''The analysis quantities at generator level, with the true Bs momentum:
    q2 = (Bs - Ds(*))^2 (the W* mass), m_miss2 = (Bs - Ds - mu_visible)^2 and
    the rest as in the reconstruction (RDsKinematics), with the Ds (not the
    Ds*) and the visible muon. NaN when the visible final state is incomplete.'''
    out = dict(('gen_%s' % k, NAN) for k in GEN_OBS_NAMES)
    if decay is None or decay.code < 0:
        return out
    p_bs = np4(decay.bs)
    out['gen_q2'] = mass2(p_bs - np4(decay.dsx))
    if not decay.visible_complete:
        return out
    p_ds, p_mu = np4(decay.ds), np4(decay.mu)
    obs = semileptonic_observables(p_bs, p_ds, p_mu)
    for k in ('m_miss2', 'pt_miss', 'e_star_mu', 'e_hash_mu'):
        out['gen_%s' % k] = obs[k]
    # the kaon with the charge of the pion, as in the reconstruction
    k_same = decay.kp if decay.pi.charge() > 0 else decay.km
    hel = helicity_angles(p_bs, p_ds, np4(decay.phi), np4(decay.pi), np4(k_same), p_mu)
    for k, v in hel.items():
        out['gen_%s' % k] = v
    return out


# ----- Hammer leaves: pre-FSR four-momenta of the semileptonic decay ----------
HAM_PARTICLES  = ('dsx', 'ds', 'dsg', 'lep', 'nu', 'taumu', 'taunut', 'taunum')
HAM_COMPONENTS = ('pt', 'eta', 'phi', 'mass', 'pdgid')
HAM_BRANCHES   = ['gen_ham_%s_%s' % (p, c) for p in HAM_PARTICLES for c in HAM_COMPONENTS]


def _ham_fill(out, name, p):
    if p is None:
        return
    p4 = np4(p)
    out['gen_ham_%s_pt'    % name] = p.pt()
    out['gen_ham_%s_eta'   % name] = p.eta()
    out['gen_ham_%s_phi'   % name] = p.phi()
    out['gen_ham_%s_mass'  % name] = np.sqrt(max(mass2(p4), 0.))
    out['gen_ham_%s_pdgid' % name] = p.pdgId()


def gen_hammer_leaves(decay):
    '''{gen_ham_<particle>_<component>} for the four signal channels, NaN
    otherwise. The Bs itself is in the gen_bs_* branches. Daughters are taken
    at their first copy, i.e. the momenta of the decay vertex before FSR.'''
    out = dict.fromkeys(HAM_BRANCHES, NAN)
    if decay is None or decay.code < 0:
        return out
    _ham_fill(out, 'dsx', decay.dsx)
    if decay.dsg is not None:
        _ham_fill(out, 'ds', decay.ds)
        _ham_fill(out, 'dsg', decay.dsg)
    _ham_fill(out, 'lep', decay.lep)
    _ham_fill(out, 'nu', decay.nu)
    if decay.tau_nus is not None:
        _ham_fill(out, 'taumu', decay.mu)
        _ham_fill(out, 'taunut', decay.tau_nus[0])
        _ham_fill(out, 'taunum', decay.tau_nus[1])
    return out


##########################################################################################
#####      CANDIDATE LEVEL: reco <-> gen matching and classification
##########################################################################################

def charged_final_state(genparticles):
    '''Status-1 charged gen particles a track can be matched to. Built once per
    event (packedGenParticles or prunedGenParticles; pruned keeps the
    mothers, which the classification needs, so use prunedGenParticles).'''
    return [p for p in genparticles
            if p.status() == 1 and abs(p.pdgId()) in CHARGED_STABLE]


def _dr2(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return (eta1 - eta2) ** 2 + dphi ** 2


def match_track(obj, gen, dr_max):
    '''Closest same-charge gen particle within dr_max: (gen particle, dR) or
    (None, NaN).'''
    best, best_dr2 = None, dr_max * dr_max
    q, e, f = obj.charge(), obj.eta(), obj.phi()
    for g in gen:
        if g.charge() != q:
            continue
        d2 = _dr2(e, f, g.eta(), g.phi())
        if d2 < best_dr2:
            best, best_dr2 = g, d2
    return (best, float(np.sqrt(best_dr2))) if best is not None else (None, NAN)


def common_charm_ancestor(gens):
    '''The closest charm hadron that is an ancestor of every particle in
    `gens` (at its last copy), or None. Used to say whether the three hadrons
    come from one Ds (or from a D+, a Lambda_c, ...).'''
    chains = []
    for g in gens:
        chain, node = [], g
        for _ in range(200):
            if node.numberOfMothers() == 0:
                break
            node = node.mother(0)
            if is_charm_hadron(node.pdgId()):
                chain.append(last_copy(node))
        chains.append([key(c) for c in chain])
        if not chain:
            return None
    # the first element of g0's chain that is in every other chain
    first = gens[0]
    node = first
    for _ in range(200):
        if node.numberOfMothers() == 0:
            return None
        node = node.mother(0)
        if not is_charm_hadron(node.pdgId()):
            continue
        k = key(last_copy(node))
        if all(k in c for c in chains[1:]):
            return last_copy(node)
    return None


CAND_GEN_NAMES = ('gen_cand', 'gen_ds_true', 'gen_charm_pdgid', 'gen_ds_mother_pdgid',
                  'gen_ds_b_pdgid', 'gen_mu_b_pdgid', 'gen_mu_mother_pdgid',
                  'gen_mu_from_tau', 'gen_same_b', 'gen_n_matched')


def classify_candidate(gk1, gk2, gpi, gmu):
    '''Candidate-level truth from the four matched gen particles (None when
    unmatched). Returns {name in CAND_GEN_NAMES: value}.'''
    out = dict.fromkeys(CAND_GEN_NAMES, NAN)
    matched = [g for g in (gk1, gk2, gpi, gmu) if g is not None]
    out['gen_n_matched'] = float(len(matched))

    # hadron side: common charm ancestor of the three tracks
    charm = None
    if gk1 is not None and gk2 is not None and gpi is not None:
        charm = common_charm_ancestor([gk1, gk2, gpi])
    ds = charm if (charm is not None and abs(charm.pdgId()) == DS) else None
    if charm is not None:
        out['gen_charm_pdgid'] = charm.pdgId()
    out['gen_ds_true'] = float(ds is not None)
    if ds is not None:
        m = mother_of(ds)
        out['gen_ds_mother_pdgid'] = m.pdgId() if m is not None else 0.
    b_ds = b_ancestor(ds) if ds is not None else None
    if b_ds is not None:
        out['gen_ds_b_pdgid'] = first_copy(b_ds).pdgId()

    # muon side
    b_mu, mu_mother = None, None
    if gmu is not None:
        mu_mother = mother_of(gmu)
        out['gen_mu_mother_pdgid'] = mu_mother.pdgId() if mu_mother is not None else 0.
        out['gen_mu_from_tau'] = float(mu_mother is not None and abs(mu_mother.pdgId()) == TAU)
        b_mu = b_ancestor(gmu)
        if b_mu is not None:
            out['gen_mu_b_pdgid'] = first_copy(b_mu).pdgId()
    same_b = b_ds is not None and b_mu is not None and same(b_ds, b_mu)
    if b_ds is not None and gmu is not None:
        out['gen_same_b'] = float(same_b)

    # category
    if len(matched) < 4 or gmu is None or abs(gmu.pdgId()) != MU:
        code = -1
    elif ds is None:
        code = 30
    elif not same_b:
        code = 20 if b_mu is not None else 21
    else:
        b = abs(b_ds.pdgId())
        if b == BS:
            code = _bs_code(b_ds, ds, gmu, mu_mother)
        elif b == B0:
            code = 12
        elif b == BP:
            code = 13
        elif b == LB:
            code = 14
        else:
            code = 15
    out['gen_cand'] = float(code)
    return out


def _bs_code(bs, ds, gmu, mu_mother):
    '''0..3 for the signal channels, 10 feed-down, 11 double charm.'''
    bs = last_copy(bs)
    # the lepton of the candidate is a direct Bs daughter (or its tau is)
    lep = gmu if (mu_mother is not None and same(last_copy(mu_mother), bs)) else None
    if lep is None and mu_mother is not None and abs(mu_mother.pdgId()) == TAU:
        tau_mother = mother_of(mu_mother)
        if tau_mother is not None and same(last_copy(tau_mother), bs):
            lep = mu_mother
    if lep is None:
        return 11
    code = SIGNAL_SIGNATURES.get(signature(bs), -1)
    if code < 0:
        return 10
    # signal signature: the Ds of the candidate must be the one of the decay
    ds_mother = mother_of(ds)
    if ds_mother is not None and same(last_copy(ds_mother), bs):
        return code if abs(first_copy(ds).pdgId()) == DS and code in (0, 2) else 10
    if ds_mother is not None and abs(ds_mother.pdgId()) == DSST:
        dsst_mother = mother_of(ds_mother)
        if dsst_mother is not None and same(last_copy(dsst_mother), bs) and code in (1, 3):
            return code
    return 10
