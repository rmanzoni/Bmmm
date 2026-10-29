'''
Bs -> Ds(*) l nu form-factor reweighting with Hammer, for the R(Ds) ntuples.

Same structure as HammerFF (the Bc -> J/psi version): ONE implementation, TWO
entry points,
  * at ntuple production time   -- RDsInspector with --hammer <card>
  * on ntuples already written  -- test/rds/hammer/add_hammer_weights_rds.py
both building the same `leaves` dict and calling the same HammerDsSession,
so the two paths are identical by construction.

STATUS OF THIS MODULE: written without a Hammer installation and without the
R(Ds) coefficient cards. Everything Hammer-specific that this code cannot know
-- the FF process keys, the class names, the option strings with the BCL / BGL
coefficients and the eigenvector matrices, the decay-chain names -- is read
from the card, not guessed here. Validate it against Patricia's setup (closure
on the same events) before using the weights.

AN-24-173 Sec. 8.2: ISGW2 (generated) -> BCL for Bs -> Ds l nu, -> BGL for
Bs -> Ds* l nu, with 6 (BCL) and 10 (BGL) eigen-variations.

CARD (JSON):
{
  "name": "...",
  "covariance_status": "validated",
  "decay_chains": [["BsDsMuNu"], ["BsDsTauNu", "TauMuNuNu"],
                   ["BsDs*MuNu", "Ds*DsGamma"], ...],     # one include_decay each
  "ff_input":  {"<process key>": "ISGW2", ...},
  "groups": {
     "ds":   {"process": "<key>", "xtoy": "<class prefix>", "ff_target": "BCLVar",
              "n_coords": 6,  "options": ["<Hammer option string>", ...]},
     "dsst": {"process": "<key>", "xtoy": "<class prefix>", "ff_target": "BGLVar",
              "n_coords": 10, "options": [...]}
  },
  "dsst_decay_vertex": true      # add Ds* -> Ds gamma/pi0 to the process
}
"options" are passed verbatim to Hammer.set_options after init_run: the central
coefficients and the 1-sigma eigen-direction matrix, in Hammer's own syntax.

BRANCH SCHEMA (fixed):
  hammer_weight, hammer_ff_ev<ii>_up / _dn for ii = 00..09, hammer_status
  (0 ok / 1 not a signal channel / 2 gen leaves missing / 3 Hammer declined /
   4 weight not finite). For Bs -> Ds (6 directions) ev06..ev09 are NaN.
'''

import json
import math

import numpy as np

from Bmmm.Analysis.RDsGenHistory import HAM_BRANCHES

N_MAX_COORDS = 10
STATUS_OK, STATUS_NOT_SIGNAL, STATUS_NO_LEAVES, STATUS_DECLINED, STATUS_NONFINITE = range(5)

VAR_LABELS     = ['ev%02d_%s' % (j, tag) for j in range(N_MAX_COORDS) for tag in ('up', 'dn')]
BRANCH_NOMINAL = 'hammer_weight'
BRANCH_STATUS  = 'hammer_status'
BRANCH_NAMES   = [BRANCH_NOMINAL] + ['hammer_ff_%s' % lab for lab in VAR_LABELS] + [BRANCH_STATUS]

BS_LEAVES      = ['gen_bs_%s' % c for c in ('pt', 'eta', 'phi', 'mass', 'pdgid')]
INPUT_BRANCHES = ['gen_sig'] + BS_LEAVES + list(HAM_BRANCHES)

# gen_sig code -> (group, lepton)
CHANNELS = {0: ('ds', 'mu'), 1: ('dsst', 'mu'), 2: ('ds', 'tau'), 3: ('dsst', 'tau')}


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def leaves_from_row(row, i):
    '''Post-hoc: one row of an uproot array dict.'''
    return dict((b, row[b][i]) for b in INPUT_BRANCHES)


def needed_leaves(code):
    group, lep = CHANNELS[code]
    need = ['gen_bs', 'gen_ham_dsx', 'gen_ham_lep', 'gen_ham_nu']
    if group == 'dsst':
        need += ['gen_ham_ds', 'gen_ham_dsg']
    if lep == 'tau':
        need += ['gen_ham_taumu', 'gen_ham_taunut', 'gen_ham_taunum']
    return need


def channel_of(leaves):
    code = leaves.get('gen_sig', float('nan'))
    if not _finite(code) or int(round(float(code))) not in CHANNELS:
        return None, STATUS_NOT_SIGNAL
    code = int(round(float(code)))
    if not all(_finite(leaves.get('%s_pt' % p)) for p in needed_leaves(code)):
        return None, STATUS_NO_LEAVES
    return code, STATUS_OK


def load_card(path):
    with open(path) as fin:
        card = json.load(fin)
    for key in ('decay_chains', 'ff_input', 'groups'):
        if key not in card:
            raise KeyError('Hammer card %s has no %r' % (path, key))
    for g in ('ds', 'dsst'):
        grp = card['groups'].get(g)
        if grp is None:
            raise KeyError('Hammer card %s has no group %r' % (path, g))
        for key in ('process', 'xtoy', 'ff_target', 'n_coords'):
            if key not in grp:
                raise KeyError('Hammer card %s, group %s: no %r' % (path, g, key))
        if int(grp['n_coords']) > N_MAX_COORDS:
            raise ValueError('Hammer card %s, group %s: %d coordinates, the schema '
                             'holds %d' % (path, g, grp['n_coords'], N_MAX_COORDS))
    return card


class HammerDsSession(object):
    '''One Hammer run, built once per job (Hammer's init_run is global).'''

    def __init__(self, card_path, variations=True, allow_stale=False, verbose=True):
        import hammer as H                                     # noqa: N813
        self.H = H
        self.card = load_card(card_path)
        self.card_path = card_path
        self.variations = bool(variations)
        status = str(self.card.get('covariance_status', 'unknown')).lower()
        if self.variations and status != 'validated' and not allow_stale:
            raise RuntimeError("Hammer card %s: covariance_status=%r. Pass "
                               "':allow-stale' to write the variations anyway, or "
                               "':nominal' for the central weight only."
                               % (card_path, status))

        self.ham = H.Hammer()
        for chain in self.card['decay_chains']:
            self.ham.include_decay(list(chain))          # one call per chain
        targets = dict((g['process'], g['ff_target']) for g in self.card['groups'].values())
        self.ham.add_ff_scheme('Target', targets)
        self.ham.set_ff_input_scheme(dict(self.card['ff_input']))
        self.ham.set_units('GeV')
        self.ham.init_run()
        for g in self.card['groups'].values():
            for opt in g.get('options', []):
                self.ham.set_options(opt)

        self.dsst_vertex = bool(self.card.get('dsst_decay_vertex', True))
        self.n_ok = self.n_declined = self.n_nonfinite = 0
        if verbose:
            print('#### Hammer (R(Ds)): card %s (%s), variations %s'
                  % (card_path, self.card.get('name', '?'), 'ON' if self.variations else 'OFF'))

    # -- process construction ------------------------------------------------
    @staticmethod
    def _p4(lv, pfx):
        pt, eta, phi, m = (float(lv['%s_%s' % (pfx, c)]) for c in ('pt', 'eta', 'phi', 'mass'))
        px, py, pz = pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta)
        return (math.sqrt(px * px + py * py + pz * pz + m * m), px, py, pz)

    @staticmethod
    def _pid(lv, pfx):
        return int(round(float(lv['%s_pdgid' % pfx])))

    def _add(self, proc, lv, pfx):
        return proc.add_particle(self.H.Particle(self.H.FourMomentum(*self._p4(lv, pfx)),
                                                 self._pid(lv, pfx)))

    def _process(self, lv, code):
        group, lep = CHANNELS[code]
        proc = self.H.Process()
        i_b   = self._add(proc, lv, 'gen_bs')
        i_dsx = self._add(proc, lv, 'gen_ham_dsx')
        i_l   = self._add(proc, lv, 'gen_ham_lep')
        i_n   = self._add(proc, lv, 'gen_ham_nu')
        proc.add_vertex(i_b, [i_dsx, i_l, i_n])
        if group == 'dsst' and self.dsst_vertex:
            i_ds = self._add(proc, lv, 'gen_ham_ds')
            i_g  = self._add(proc, lv, 'gen_ham_dsg')
            proc.add_vertex(i_dsx, [i_ds, i_g])
        if lep == 'tau':
            i_m  = self._add(proc, lv, 'gen_ham_taumu')
            i_nt = self._add(proc, lv, 'gen_ham_taunut')
            i_nm = self._add(proc, lv, 'gen_ham_taunum')
            proc.add_vertex(i_l, [i_m, i_nt, i_nm])
        return proc

    # -- the one public method ------------------------------------------------
    def weights(self, leaves):
        out = dict((b, float('nan')) for b in BRANCH_NAMES)
        code, status = channel_of(leaves)
        out[BRANCH_STATUS] = float(status)
        if code is None:
            return out
        group = self.card['groups'][CHANNELS[code][0]]

        self.ham.init_event()
        if self.ham.add_process(self._process(leaves, code)) == 0:
            self.n_declined += 1
            out[BRANCH_STATUS] = float(STATUS_DECLINED)
            return out
        self.ham.process_event()

        n = int(group['n_coords'])
        coords = [np.zeros(n)]
        if self.variations:
            for j in range(n):
                for sign in (+1., -1.):
                    v = np.zeros(n)
                    v[j] = sign
                    coords.append(v)
        vals = []
        for c in coords:
            if self.variations:
                self.ham.set_ff_eigenvectors(group['xtoy'], group['ff_target'], [float(x) for x in c])
            vals.append(self.ham.get_weight('Target'))
        if self.variations:
            self.ham.set_ff_eigenvectors(group['xtoy'], group['ff_target'], [0.] * n)

        if not _finite(vals[0]):
            self.n_nonfinite += 1
            out[BRANCH_STATUS] = float(STATUS_NONFINITE)
            return out
        self.n_ok += 1
        out[BRANCH_NOMINAL] = float(vals[0])
        for name, val in zip(BRANCH_NAMES[1:1 + len(vals) - 1], vals[1:]):
            out[name] = float(val)
        return out

    def summary(self):
        return ('#### Hammer (R(Ds)): %d events reweighted, %d declined, %d non-finite'
                % (self.n_ok, self.n_declined, self.n_nonfinite))


def make_hammer_session(spec, verbose=True):
    '''--hammer spec: '' -> None; '<card.json>[:nominal][:allow-stale]'.'''
    if not spec:
        return None
    parts = spec.split(':')
    flags = set(parts[1:])
    unknown = flags - {'nominal', 'allow-stale'}
    if unknown:
        raise ValueError('--hammer: unknown flag(s) %s' % sorted(unknown))
    return HammerDsSession(parts[0], variations='nominal' not in flags,
                           allow_stale='allow-stale' in flags, verbose=verbose)
