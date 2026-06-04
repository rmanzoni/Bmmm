"""
gen_history.py

Reconstruct the MC generator history of a Bc -> (cocktail) sample and
classify, event by event, which Bc decay channel occurred.

Designed to run inside the FWLite-based inspector (inspector_rjpsi.py),
operating on the prunedGenParticles collection (a vector of reco::GenParticle).

Quick usage inside the looper (see bottom of this file for the full snippet):

    from Bmmm.Analysis.gen_history import classify_bc_event, print_gen_history

    if options.mc:
        code, info = classify_bc_event(event.genpr)
        tofill['bc_gen_decay'] = code           # int label, 0 = no Bc, -1 = unknown
        # for debugging a handful of events:
        # print_gen_history(event.genpr)

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


def classify_bc(bc):
    '''Classify a single (last-copy) Bc. Returns (code, raw_daughter_ids, signature).
    code == -1 means the daughter signature is not in the table.'''
    raw = direct_daughter_ids(bc, keep_photons=True, keep_neutrinos=True)
    sig = _signature(raw, strip_neutrinos=True)
    code = _SIG2CODE.get(sig, -1)
    return code, raw, sig


def classify_bc_event(genparticles):
    '''Main entry point. Returns (code, info_dict).
       code:  >0  channel id (see BC_CHANNELS)
               0  no Bc found in the collection
              -1  Bc found but signature unrecognised
    '''
    bcs = find_decayed_bc(genparticles)
    if len(bcs) == 0:
        return 0, {'name': 'no_Bc_found', 'n_bc': 0, 'daughters': [],
                   'signature': (), 'bc': None}
    bc = last_copy(bcs[0])
    code, raw, sig = classify_bc(bc)
    info = {
        'name'      : CODE2NAME.get(code, 'unknown'),
        'n_bc'      : len(bcs),          # >1 would be unexpected, worth watching
        'bc'        : bc,                # the decaying (last-copy) Bc gen particle
        'bc_pdgId'  : bc.pdgId(),
        'daughters' : raw,               # full direct daughters incl. gamma/nu
        'signature' : sig,               # the key actually used for matching
    }
    return code, info


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
# Integration snippet for inspector_rjpsi.py (reference; not executed on import)
# ----------------------------------------------------------------------------
"""
# 1) add the branch to your RJpsi branches list (e.g. in RJpsiBranches.py):
#        event_branches['bc_gen_decay'] = ... or just append 'bc_gen_decay'
#    so it ends up in `branches`.

# 2) inside the looper, in the `if options.mc:` block, BEFORE the per-cand loop:

        bc_code, bc_info = classify_bc_event(event.genpr)

# 3) inside the per-candidate fill loop (so every row gets the event-level label):

        tofill['bc_gen_decay'] = bc_code

# 4) (debug) dump the tree for the first few events:

        if options.verbose and i < 20:
            print('--- event %d : %s (code %d) ---' % (i, bc_info['name'], bc_code))
            print('    direct Bc daughters:', [pdg_name(x) for x in bc_info['daughters']])
            print_gen_history(event.genpr)
"""


# ----------------------------------------------------------------------------
# Gen-truth kinematics of the signal Bc semileptonic decay:
#   q2, missing mass^2, and the bachelor-muon energy in the Bc and J/psi frames.
# Boosts are done in pure numpy so this module stays ROOT-free / unit-testable.
# ----------------------------------------------------------------------------
CHARMONIA = {JPSI, PSI2S, CHI_C0, CHI_C1, CHI_C2, H_C}


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


def gen_kinematics(genparticles):
    '''Gen-truth kinematics of the signal Bc semileptonic decay.

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

    bcs = find_decayed_bc(genparticles)
    if not bcs:
        return out
    bc = last_copy(bcs[0])

    # direct Bc daughters: the charmonium and the bachelor lepton (mu, or tau->mu)
    charm, bachelor = None, None
    for i in range(bc.numberOfDaughters()):
        d = bc.daughter(i)
        if d.pdgId() == bc.pdgId():
            continue
        ad = abs(d.pdgId())
        if ad in CHARMONIA and charm is None:
            charm = last_copy(d)
        elif ad == MU and bachelor is None:
            bachelor = last_copy(d)
        elif ad == TAU and bachelor is None:
            bachelor = _muon_from_tau(last_copy(d))

    p4_bc = _p4(bc)

    # q^2 : recoil against the directly produced charmonium
    if charm is not None:
        out['q2'] = _mass2(p4_bc - _p4(charm))

    # missing mass^2 from the gen neutrino system (exact: 0 for one massless nu)
    nus = []
    _collect_neutrinos(bc, nus)
    if nus:
        out['m_miss2'] = _mass2(np.sum([_p4(n) for n in nus], axis=0))

    # J/psi four-momentum from the two muons of the X -> mu mu vertex
    p4_jpsi = None
    if charm is not None:
        jpsi_mus = _dimuon_under(charm)
        if len(jpsi_mus) == 2:
            p4_jpsi = _p4(jpsi_mus[0]) + _p4(jpsi_mus[1])

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
