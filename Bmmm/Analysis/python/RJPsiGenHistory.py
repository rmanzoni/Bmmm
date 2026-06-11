"""
RJPsiGenHistory.py

Reconstruct the MC generator history of a Bc -> (cocktail) sample and
classify, event by event, which Bc decay channel occurred.

Designed to run inside the FWLite-based inspector (inspector_rjpsi.py),
operating on the prunedGenParticles collection (a vector of reco::GenParticle).

The whole generator-level decay is reconstructed ONCE per event by
``BcGenDecay.from_genparticles`` and the resulting object is meant to be cached
on the event and reused by every downstream function (classification, q2 /
missing-mass kinematics, helicity angles), so the gen tree is walked only once.

Quick usage inside the looper (see the bottom of this file for the full snippet):

    from Bmmm.Analysis.RJPsiGenHistory import (
        BcGenDecay, gen_kinematics, gen_helicity_angles, print_gen_history)

    if options.mc:
        event.bc_gen = BcGenDecay.from_genparticles(event.genpr)   # once per event

    ...

    if options.mc and event.bc_gen is not None:
        tofill['bc_gen_decay'] = event.bc_gen.code      # int label, 0 = no Bc, -1 = unknown
        tofill.update({'gen_%s' % k: v for k, v in gen_kinematics(event.bc_gen).items()})
        tofill.update({'gen_%s' % k: v for k, v in gen_helicity_angles(event.bc_gen).items()})

The classification key is the multiset of |pdgId| of the *direct* Bc daughters,
after dropping PHOTOS photons and (optionally) neutrinos. For this decay table
that key is unique across all 22 channels, so the result is robust even if the
neutrinos do not survive gen pruning.
"""
from __future__ import print_function
from collections import OrderedDict

import numpy as np

# ----------------------------------------------------------------------------
# PDG ids of the particles that appear directly in the Bc decay table
# ----------------------------------------------------------------------------
BC       = 541
M_BC     = 6.27447   # PDG Bc+ mass [GeV]; RJPsiNuReco and the candidate/inspector import this
JPSI     = 443
PSI2S    = 100443
CHI_C0   = 10441
CHI_C1   = 20443
CHI_C2   = 445
H_C      = 10443
MU       = 13
NU_MU    = 14
TAU      = 15
NU_TAU   = 16
PI       = 211
KAON     = 321
KSTAR0   = 313
DS       = 431
DS_ST    = 433
D0       = 421
D0_ST    = 423
DPLUS    = 411
DPLUS_ST = 413
PROTON   = 2212
GAMMA    = 22

NEUTRINOS = {12, 14, 16}

# charmonia that can show up as the direct (recoil) Bc daughter
CHARMONIA = {JPSI, PSI2S, CHI_C0, CHI_C1, CHI_C2, H_C}

# ----------------------------------------------------------------------------
# Bc decay channels, in the same order as the EvtGen .dec table.
# Each entry: code -> (name, [direct Bc daughters, signs as written for MyBc-]).
# Matching is done on |pdgId| (charge-conjugate safe, covers CDecay MyBc+),
# so the signs below are only documentation.
# ----------------------------------------------------------------------------
BC_CHANNELS = OrderedDict()
BC_CHANNELS[ 1] = ('Jpsi_mu_nu'       , [JPSI,   MU,  NU_MU ])
BC_CHANNELS[ 2] = ('psi2S_mu_nu'      , [PSI2S,  MU,  NU_MU ])
BC_CHANNELS[ 3] = ('chic0_mu_nu'      , [CHI_C0, MU,  NU_MU ])
BC_CHANNELS[ 4] = ('chic1_mu_nu'      , [CHI_C1, MU,  NU_MU ])
BC_CHANNELS[ 5] = ('chic2_mu_nu'      , [CHI_C2, MU,  NU_MU ])
BC_CHANNELS[ 6] = ('hc_mu_nu'         , [H_C,    MU,  NU_MU ])
BC_CHANNELS[ 7] = ('Jpsi_tau_nu'      , [JPSI,   TAU, NU_TAU])
BC_CHANNELS[ 8] = ('psi2S_tau_nu'     , [PSI2S,  TAU, NU_TAU])
BC_CHANNELS[ 9] = ('Jpsi_pi'          , [JPSI,   PI          ])
BC_CHANNELS[10] = ('Jpsi_3pi'         , [JPSI,   PI, PI, PI  ])
BC_CHANNELS[11] = ('Jpsi_5pi'         , [JPSI,   PI, PI, PI, PI, PI])
BC_CHANNELS[12] = ('Jpsi_K'           , [JPSI,   KAON        ])
BC_CHANNELS[13] = ('Jpsi_Ds'          , [JPSI,   DS          ])
BC_CHANNELS[14] = ('Jpsi_Dsstar'      , [JPSI,   DS_ST       ])
BC_CHANNELS[15] = ('Jpsi_D0bar_K'     , [JPSI,   D0,   KAON  ])
BC_CHANNELS[16] = ('Jpsi_D0starbar_K' , [JPSI,   D0_ST, KAON ])
BC_CHANNELS[17] = ('Jpsi_Dstar_Kstar' , [JPSI,   DPLUS_ST, KSTAR0])
BC_CHANNELS[18] = ('Jpsi_D_Kstar'     , [JPSI,   DPLUS, KSTAR0])
BC_CHANNELS[19] = ('Jpsi_D'           , [JPSI,   DPLUS       ])
BC_CHANNELS[20] = ('Jpsi_Dstar'       , [JPSI,   DPLUS_ST    ])
BC_CHANNELS[21] = ('Jpsi_p_pbar_pi'   , [JPSI,   PROTON, PROTON, PI])
BC_CHANNELS[22] = ('Jpsi_K_K_pi'      , [JPSI,   KAON, KAON, PI])


def _signature(pdgids, strip_neutrinos=True):
    '''Canonical, charge-conjugate-safe key: sorted tuple of |pdgId|,
    photons always removed, neutrinos optionally removed.'''
    ids = [abs(i) for i in pdgids if abs(i) != GAMMA]
    if strip_neutrinos:
        ids = [i for i in ids if i not in NEUTRINOS]
    return tuple(sorted(ids))


# reverse lookup: neutrino-stripped signature -> channel code
# (verified collision-free for this decay table)
_SIG2CODE = {}
for _code, (_name, _daus) in BC_CHANNELS.items():
    _sig = _signature(_daus, strip_neutrinos=True)
    assert _sig not in _SIG2CODE, 'signature collision on %s' % (_sig,)
    _SIG2CODE[_sig] = _code

CODE2NAME = {code: name for code, (name, _) in BC_CHANNELS.items()}
CODE2NAME[-1] = 'unknown'
CODE2NAME[ 0] = 'no_Bc_found'


# ----------------------------------------------------------------------------
# gen-particle navigation helpers (reco::GenParticle via FWLite)
# ----------------------------------------------------------------------------
def _is_last_copy(p):
    '''True if no daughter carries the same pdgId (i.e. p is the copy that
    actually decays, after any Pythia self-copies / PHOTOS radiation).'''
    pid = p.pdgId()
    for i in range(p.numberOfDaughters()):
        if p.daughter(i).pdgId() == pid:
            return False
    return True


def last_copy(p, _guard=200):
    '''Descend through same-pdgId daughters to the last copy of p.'''
    pid = p.pdgId()
    for _ in range(_guard):
        nxt = None
        for i in range(p.numberOfDaughters()):
            d = p.daughter(i)
            if d.pdgId() == pid:
                nxt = d
                break
        if nxt is None:
            return p
        p = nxt
    return p


def find_decayed_bc(genparticles):
    '''Return the list of Bc (|pdgId|==541) last copies that actually decayed.
    For this signal sample there should be exactly one per event.'''
    out = []
    for p in genparticles:
        if abs(p.pdgId()) != BC:
            continue
        if p.numberOfDaughters() == 0:
            continue
        if not _is_last_copy(p):
            continue
        out.append(p)
    return out


def direct_daughter_ids(p, keep_photons=False, keep_neutrinos=True):
    ids = []
    for i in range(p.numberOfDaughters()):
        pid = p.daughter(i).pdgId()
        if (not keep_photons) and abs(pid) == GAMMA:
            continue
        if (not keep_neutrinos) and abs(pid) in NEUTRINOS:
            continue
        ids.append(pid)
    return ids


# ----------------------------------------------------------------------------
# numpy 4-vector helpers (boosts done in pure numpy so the kinematics stay
# ROOT-free / unit-testable)
# ----------------------------------------------------------------------------
def _p4(p):
    '''(E, px, py, pz) numpy array from a gen particle.'''
    return np.array([p.energy(), p.px(), p.py(), p.pz()], dtype=float)


def _mass2(four):
    '''Minkowski norm E^2 - |p|^2 of a (E, px, py, pz) array.'''
    return float(four[0] * four[0] - four[1] * four[1]
                 - four[2] * four[2] - four[3] * four[3])


def _boost_to_rest(four, system):
    '''Express `four` (E, px, py, pz) in the rest frame of `system` (same format).'''
    e_sys = system[0]
    if e_sys <= 0.:
        return four.copy()
    beta = system[1:] / e_sys
    b2 = float(beta.dot(beta))
    if b2 <= 0. or b2 >= 1.:           # already at rest, or unphysical -> no-op
        return four.copy()
    gamma = 1.0 / np.sqrt(1.0 - b2)
    pvec  = four[1:]
    bp    = float(beta.dot(pvec))
    e_new = gamma * (four[0] - bp)
    p_new = pvec + ((gamma - 1.0) * bp / b2 - gamma * four[0]) * beta
    return np.array([e_new, p_new[0], p_new[1], p_new[2]])


def _unit(v):
    n = np.sqrt(float(v.dot(v)))
    return v / n if n > 0. else v


# ----------------------------------------------------------------------------
# subtree finders used to resolve the cascade Bc -> charmonium(->mu mu) l nu
# ----------------------------------------------------------------------------
def _dimuon_under(res, _depth=0, _maxdepth=8):
    '''The two muons of the X -> mu mu vertex inside a charmonium subtree
    (handles J/psi directly and psi(2S)/chi_c -> J/psi -> mu mu feed-down).'''
    res = last_copy(res)
    mus = [last_copy(res.daughter(i)) for i in range(res.numberOfDaughters())
           if abs(res.daughter(i).pdgId()) == MU]
    if len(mus) >= 2:
        return mus[:2]
    if _depth >= _maxdepth:
        return mus
    for i in range(res.numberOfDaughters()):
        d = res.daughter(i)
        if d.pdgId() == res.pdgId() or abs(d.pdgId()) == GAMMA:
            continue
        found = _dimuon_under(d, _depth + 1, _maxdepth)
        if len(found) >= 2:
            return found
    return mus


def _muon_from_tau(tau, _depth=0, _maxdepth=4):
    '''The muon from tau -> mu nu nu (None otherwise).'''
    tau = last_copy(tau)
    for i in range(tau.numberOfDaughters()):
        if abs(tau.daughter(i).pdgId()) == MU:
            return last_copy(tau.daughter(i))
    if _depth < _maxdepth:
        for i in range(tau.numberOfDaughters()):
            d = tau.daughter(i)
            if d.pdgId() == tau.pdgId():
                continue
            m = _muon_from_tau(d, _depth + 1, _maxdepth)
            if m is not None:
                return m
    return None


def _collect_neutrinos(p, out, _depth=0, _maxdepth=12):
    '''Append (last copies of) all neutrinos in the subtree of p.'''
    p = last_copy(p)
    if _depth > _maxdepth:
        return
    for i in range(p.numberOfDaughters()):
        d = p.daughter(i)
        if d.pdgId() == p.pdgId():
            continue
        if abs(d.pdgId()) in NEUTRINOS:
            out.append(last_copy(d))
        else:
            _collect_neutrinos(d, out, _depth + 1, _maxdepth)


# ----------------------------------------------------------------------------
# single-Bc classification
# ----------------------------------------------------------------------------
def classify_bc(bc):
    '''Classify a single (last-copy) Bc. Returns (code, raw_daughter_ids, signature).
    code == -1 means the daughter signature is not in the table.'''
    raw = direct_daughter_ids(bc, keep_photons=True, keep_neutrinos=True)
    sig = _signature(raw, strip_neutrinos=True)
    code = _SIG2CODE.get(sig, -1)
    return code, raw, sig


# ----------------------------------------------------------------------------
# ONE-SHOT generator-level Bc decay reconstruction
#
# Walk the gen tree a single time and cache everything the downstream functions
# need. Build it once per event and stash it on the event:
#
#     event.bc_gen = BcGenDecay.from_genparticles(event.genpr)
#
# then reuse `event.bc_gen` in classify / kinematics / helicity, instead of
# re-finding the Bc and re-parsing its daughters in each of them.
# ----------------------------------------------------------------------------
class BcGenDecay(object):
    '''Reconstructed generator-level Bc -> charmonium(-> mu mu) + (l nu) decay.

    Attributes
    ----------
    bc          : the decaying (last-copy) Bc gen particle
    p4          : (E, px, py, pz) numpy array of the Bc
    code        : channel id from BC_CHANNELS (>0), -1 unknown, never 0 here
    name        : human-readable channel name
    daughters   : raw direct Bc daughter pdgIds (incl. gamma / nu)
    signature   : neutrino-stripped |pdgId| key used for matching
    n_bc        : number of decayed Bc found in the event (normally 1)

    charm       : direct charmonium daughter of the Bc (last copy), or None
    lepton      : direct charged lepton daughter (mu OR tau, last copy), or None
                  -> use this one for helicity (it is the true W* daughter)
    bachelor_mu : the bachelor muon, i.e. `lepton` itself when it is a mu, or the
                  mu from tau -> mu nu nu in the tau channels; None otherwise
                  -> use this one for the muon-energy / visible-mass kinematics
    jpsi_mus    : the two muons of the X -> mu mu vertex inside the charmonium
    mu_minus    : the negative muon of that pair (None if no clean pair)
    neutrinos   : list of (last-copy) neutrinos in the Bc subtree (may be empty)

    p4_charm    : cached _p4(charm)                       (None if no charm)
    p4_jpsi     : cached dimuon four-momentum _p4(mu+)+_p4(mu-) (None if no pair)

    Build with BcGenDecay.from_genparticles(...) (returns None if no Bc).
    '''

    def __init__(self, bc, n_bc=1):
        self.bc   = bc
        self.p4   = _p4(bc)
        self.n_bc = n_bc

        # classification (same key as classify_bc / classify_bc_event)
        self.code, self.daughters, self.signature = classify_bc(bc)
        self.name = CODE2NAME.get(self.code, 'unknown')

        # single pass over the direct Bc daughters: pick the charmonium and the
        # charged lepton (this is the block that used to be copy-pasted into
        # gen_kinematics and gen_helicity_angles).
        self.charm  = None    # direct charmonium daughter (last copy)
        self.lepton = None    # direct charged lepton: mu OR tau (last copy)
        for i in range(bc.numberOfDaughters()):
            d = bc.daughter(i)
            if d.pdgId() == bc.pdgId():
                continue
            ad = abs(d.pdgId())
            if ad in CHARMONIA and self.charm is None:
                self.charm = last_copy(d)
            elif ad in (MU, TAU) and self.lepton is None:
                self.lepton = last_copy(d)

        # bachelor muon: the lepton itself if it is a muon, else the muon from
        # tau -> mu nu nu (matches the original gen_kinematics behaviour)
        self.bachelor_mu = None
        if self.lepton is not None:
            if abs(self.lepton.pdgId()) == MU:
                self.bachelor_mu = self.lepton
            elif abs(self.lepton.pdgId()) == TAU:
                self.bachelor_mu = _muon_from_tau(self.lepton)

        # the two muons of the X -> mu mu vertex inside the charmonium
        self.jpsi_mus = _dimuon_under(self.charm) if self.charm is not None else []

        # cached charmonium and dimuon four-momenta + the negative muon
        self.p4_charm = _p4(self.charm) if self.charm is not None else None
        self.p4_jpsi  = None
        self.mu_minus = None
        if len(self.jpsi_mus) == 2:
            self.p4_jpsi  = _p4(self.jpsi_mus[0]) + _p4(self.jpsi_mus[1])
            mu_a, mu_b    = self.jpsi_mus
            self.mu_minus = mu_a if mu_a.charge() < 0 else mu_b

        # all neutrinos in the Bc subtree (may be empty if pruned)
        self.neutrinos = []
        _collect_neutrinos(bc, self.neutrinos)

    @classmethod
    def from_genparticles(cls, genparticles):
        '''Build the decay from a gen-particle collection.
        Returns None if there is no decayed Bc in the event.'''
        bcs = find_decayed_bc(genparticles)
        if not bcs:
            return None
        return cls(last_copy(bcs[0]), n_bc=len(bcs))

    def __repr__(self):
        return '<BcGenDecay %s (code %d) charm=%s lepton=%s n_jpsi_mu=%d n_nu=%d>' % (
            self.name, self.code,
            pdg_name(self.charm.pdgId())  if self.charm  is not None else 'None',
            pdg_name(self.lepton.pdgId()) if self.lepton is not None else 'None',
            len(self.jpsi_mus), len(self.neutrinos))


def classify_bc_event(genparticles):
    '''Back-compatible entry point. Returns (code, info_dict).

       Prefer ``BcGenDecay.from_genparticles(genparticles)`` directly and read
       its attributes; this wrapper is kept so existing callers keep working.

       code:  >0  channel id (see BC_CHANNELS)
               0  no Bc found in the collection
              -1  Bc found but signature unrecognised
    '''
    decay = BcGenDecay.from_genparticles(genparticles)
    if decay is None:
        return 0, {'name': 'no_Bc_found', 'n_bc': 0, 'daughters': [],
                   'signature': (), 'bc': None}
    info = {
        'name'      : decay.name,
        'n_bc'      : decay.n_bc,         # >1 would be unexpected, worth watching
        'bc'        : decay.bc,           # the decaying (last-copy) Bc gen particle
        'bc_pdgId'  : decay.bc.pdgId(),
        'daughters' : decay.daughters,    # full direct daughters incl. gamma/nu
        'signature' : decay.signature,    # the key actually used for matching
    }
    return decay.code, info


# ----------------------------------------------------------------------------
# adapter so the gen_* functions accept either a pre-built BcGenDecay (fast
# path, intended usage) or a raw gen-particle collection (rebuilds on the fly).
# ----------------------------------------------------------------------------
def _as_decay(obj):
    if obj is None or isinstance(obj, BcGenDecay):
        return obj
    return BcGenDecay.from_genparticles(obj)


# ----------------------------------------------------------------------------
# Gen-truth kinematics of the signal Bc semileptonic decay:
#   q2, missing mass^2, and the bachelor-muon energy in the Bc and J/psi frames.
# ----------------------------------------------------------------------------
def gen_kinematics(decay):
    '''Gen-truth kinematics of the signal Bc semileptonic decay.

    `decay` is the BcGenDecay built once per event (or, for convenience, a raw
    gen-particle collection, which is then reconstructed on the fly).

    Returns a dict (values are NaN when not computable, e.g. hadronic Bc modes
    that have no bachelor lepton, or when no Bc is found):

        q2          (p_Bc - p_charmonium)^2  ==  (lepton + neutrino) inv. mass^2  [GeV^2]
        m_miss2     (sum of the gen neutrinos)^2  ->  ~0 for Bc -> J/psi mu nu    [GeV^2]
        m_miss2_vis (p_Bc - p_visible)^2 with visible = dimuon + bachelor mu;
                    equals m_miss2 at truth but is robust if the nu's are pruned  [GeV^2]
        e_mu_bc     bachelor-muon energy in the Bc   rest frame                   [GeV]
        e_mu_jpsi   bachelor-muon energy in the J/psi rest frame                  [GeV]

    The charmonium used for q2 is the *direct* Bc daughter (J/psi, or psi(2S)/
    chi_c/h_c in feed-down); the J/psi frame is defined by the two muons of the
    X -> mu mu vertex, so it is the true dimuon resonance in every channel.
    '''
    out = {'q2': np.nan, 'm_miss2': np.nan, 'm_miss2_vis': np.nan,
           'e_mu_bc': np.nan, 'e_mu_jpsi': np.nan}

    decay = _as_decay(decay)
    if decay is None:
        return out

    p4_bc    = decay.p4
    p4_charm = decay.p4_charm
    p4_jpsi  = decay.p4_jpsi
    bachelor = decay.bachelor_mu

    # q^2 : recoil against the directly produced charmonium
    if p4_charm is not None:
        out['q2'] = _mass2(p4_bc - p4_charm)

    # missing mass^2 from the gen neutrino system (exact: 0 for one massless nu)
    if decay.neutrinos:
        out['m_miss2'] = _mass2(np.sum([_p4(n) for n in decay.neutrinos], axis=0))

    # missing mass^2 from the visible system (robust if neutrinos are pruned)
    if p4_jpsi is not None and bachelor is not None:
        out['m_miss2_vis'] = _mass2(p4_bc - (p4_jpsi + _p4(bachelor)))

    # bachelor-muon energy in the Bc and J/psi rest frames
    if bachelor is not None:
        p4_mu = _p4(bachelor)
        out['e_mu_bc'] = _boost_to_rest(p4_mu, p4_bc)[0]
        if p4_jpsi is not None:
            out['e_mu_jpsi'] = _boost_to_rest(p4_mu, p4_jpsi)[0]

    return out


# ----------------------------------------------------------------------------
# Helicity angles of the cascade  Bc -> J/psi(->mu+ mu-) W*(->l nu)
#   cos_theta_v : angle of the mu- in the J/psi rest frame w.r.t. the J/psi
#                 flight direction in the Bc rest frame
#   cos_theta_l : angle of the charged lepton (direct W* daughter: mu, or tau in
#                 the tau channel) in the W* rest frame w.r.t. the W* flight
#                 direction in the Bc rest frame
#   chi         : signed angle (rad) between the J/psi->mu mu and W*->l nu decay
#                 planes about the J/psi-W* axis, in the Bc rest frame
# Conventions match the standard B -> V(->P1 P2) l nu description (V = J/psi).
# ----------------------------------------------------------------------------
def gen_helicity_angles(decay):
    '''Return {'cos_theta_l', 'cos_theta_v', 'chi'} for the signal Bc cascade
    (NaN where not computable). Robust to neutrino pruning: the W* and neutrino
    momenta are reconstructed as (p_Bc - p_charmonium) and (p_W - p_lepton).

    `decay` is the BcGenDecay built once per event (or a raw gen-particle
    collection, reconstructed on the fly for convenience).
    '''
    out = {'cos_theta_l': np.nan, 'cos_theta_v': np.nan, 'chi': np.nan}

    decay = _as_decay(decay)
    if decay is None or decay.charm is None or decay.lepton is None:
        return out
    if len(decay.jpsi_mus) != 2:
        return out

    mu_minus = decay.mu_minus

    p4_bc   = decay.p4
    p4_lep  = _p4(decay.lepton)
    p4_jpsi = decay.p4_jpsi                  # true dimuon resonance
    p4_w    = p4_bc - decay.p4_charm         # virtual W* = (lepton + neutrino)

    # cos(theta_V): mu- in the J/psi frame vs the J/psi flight (= -Bc) direction
    mu_in_jpsi = _boost_to_rest(_p4(mu_minus), p4_jpsi)[1:]
    bc_in_jpsi = _boost_to_rest(p4_bc,         p4_jpsi)[1:]
    out['cos_theta_v'] = -float(np.dot(_unit(mu_in_jpsi), _unit(bc_in_jpsi)))

    # cos(theta_L): charged lepton in the W* frame vs the W* flight (= -Bc) direction
    lep_in_w = _boost_to_rest(p4_lep, p4_w)[1:]
    bc_in_w  = _boost_to_rest(p4_bc,  p4_w)[1:]
    out['cos_theta_l'] = -float(np.dot(_unit(lep_in_w), _unit(bc_in_w)))

    # chi: dihedral angle between the two decay planes, in the Bc rest frame
    mu_bc  = _boost_to_rest(_p4(mu_minus), p4_bc)[1:]
    lep_bc = _boost_to_rest(p4_lep,        p4_bc)[1:]
    w_bc   = _boost_to_rest(p4_w,          p4_bc)[1:]
    zhat = _unit(w_bc)                       # J/psi-W* axis (W* flight direction)
    n_v = _unit(np.cross(zhat, mu_bc))       # normal of the J/psi->mu mu plane
    n_l = _unit(np.cross(zhat, lep_bc))      # normal of the W*->l nu plane
    cos_chi = float(np.dot(n_v, n_l))
    sin_chi = float(np.dot(np.cross(n_v, n_l), zhat))
    out['chi'] = float(np.arctan2(sin_chi, cos_chi))
    return out


# ----------------------------------------------------------------------------
# OPTIONAL: sub-decay of the tau (useful for the 3mu R(J/psi) signal, ch. 7/8).
# Returns 'mu', 'e', 'had', or 'none' for how the tau from Bc decayed.
# ----------------------------------------------------------------------------
def tau_decay_mode(bc):
    taus = [bc.daughter(i) for i in range(bc.numberOfDaughters())
            if abs(bc.daughter(i).pdgId()) == TAU]
    if not taus:
        return 'none'
    tau = last_copy(taus[0])
    daus = [abs(tau.daughter(i).pdgId()) for i in range(tau.numberOfDaughters())]
    if 13 in daus:
        return 'mu'
    if 11 in daus:
        return 'e'
    if daus:
        return 'had'
    return 'none'


# ----------------------------------------------------------------------------
# pretty names (scikit-hep 'particle' if available, else a small fallback map)
# ----------------------------------------------------------------------------
try:
    from particle import Particle

    def pdg_name(pdgid):
        try:
            return Particle.from_pdgid(pdgid).name
        except Exception:
            return str(pdgid)
except ImportError:
    _FALLBACK = {
         541: 'B_c+',  -541: 'B_c-',   443: 'J/psi', 100443: 'psi(2S)',
       10441: 'chi_c0', 20443: 'chi_c1', 445: 'chi_c2', 10443: 'h_c',
          13: 'mu-',    -13: 'mu+',     14: 'nu_mu',    -14: 'nu_mu~',
          15: 'tau-',   -15: 'tau+',    16: 'nu_tau',   -16: 'nu_tau~',
         211: 'pi+',   -211: 'pi-',    111: 'pi0',
         321: 'K+',    -321: 'K-',     311: 'K0',  313: 'K*0', -313: 'K*0~',
         431: 'D_s+',  -431: 'D_s-',   433: 'D_s*+', -433: 'D_s*-',
         421: 'D0',    -421: 'D0~',    423: 'D*0',   -423: 'D*0~',
         411: 'D+',    -411: 'D-',     413: 'D*+',   -413: 'D*-',
        2212: 'p',    -2212: 'p~',      22: 'gamma',
    }

    def pdg_name(pdgid):
        return _FALLBACK.get(pdgid, str(pdgid))


def print_gen_history(genparticles, roots_pdgid=(BC,), max_depth=8, show_photons=True):
    '''Print the full decay tree(s) starting from the given root |pdgId|(s).
    Self-copies are collapsed so each physical particle appears once.'''
    roots = [p for p in genparticles
             if abs(p.pdgId()) in roots_pdgid and _is_last_copy(p)]
    for r in roots:
        _print_node(r, 0, max_depth, show_photons)


def _print_node(p, depth, max_depth, show_photons):
    p = last_copy(p)
    print('%s%-10s  pdgId=%-7d status=%-3d pt=%6.2f eta=%6.2f' % (
        '    ' * depth, pdg_name(p.pdgId()), p.pdgId(), p.status(), p.pt(), p.eta()))
    if depth >= max_depth:
        return
    for i in range(p.numberOfDaughters()):
        d = p.daughter(i)
        if d.pdgId() == p.pdgId():
            continue
        if (not show_photons) and abs(d.pdgId()) == GAMMA:
            continue
        _print_node(d, depth + 1, max_depth, show_photons)


# ----------------------------------------------------------------------------
# Integration snippet for inspector_rjpsi.py (reference; not executed on import)
# ----------------------------------------------------------------------------
"""
# 1) add the branches to your RJpsi branches list (e.g. in RJpsiBranches.py):
#        'bc_gen_decay'
#        'gen_q2', 'gen_m_miss2', 'gen_m_miss2_vis', 'gen_e_mu_bc', 'gen_e_mu_jpsi'
#        'gen_cos_theta_l', 'gen_cos_theta_v', 'gen_chi'
#    so they end up in `branches`.

from Bmmm.Analysis.RJPsiGenHistory import (
    BcGenDecay, gen_kinematics, gen_helicity_angles, pdg_name, print_gen_history)

# 2) inside the looper, in the `if options.mc:` block, reconstruct ONCE per event
#    and cache it on the event:

        event.bc_gen = BcGenDecay.from_genparticles(event.genpr)

        if options.verbose and i < 20 and event.bc_gen is not None:
            print('--- event %d : %s (code %d) ---' % (
                i, event.bc_gen.name, event.bc_gen.code))
            print('    direct Bc daughters:',
                  [pdg_name(x) for x in event.bc_gen.daughters])
            print_gen_history(event.genpr)

# 3) inside the per-candidate fill loop, reuse the cached object (no re-walking):

        if options.mc and event.bc_gen is not None:
            tofill['bc_gen_decay'] = event.bc_gen.code
            tofill.update({'gen_%s' % k: v
                           for k, v in gen_kinematics(event.bc_gen).items()})
            tofill.update({'gen_%s' % k: v
                           for k, v in gen_helicity_angles(event.bc_gen).items()})
        elif options.mc:
            tofill['bc_gen_decay'] = 0      # no Bc found in this event
"""
