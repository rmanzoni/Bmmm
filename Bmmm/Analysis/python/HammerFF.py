'''
Bc -> J/psi l nu form-factor reweighting with Hammer v2.

Kiselev (the EvtGen BC_VMN model the MC was generated with) -> Harrison-2024
lattice QCD, expressed in Hammer's own BctoJpsiBGLVar convention. See
Bmmm/Analysis/test/rjpsi/hammer/ for the fit that produces the coefficient card
and for the note describing the procedure.

ONE implementation, TWO entry points:
  * at ntuple production time  -- JpsiMuInspector with --hammer
  * on ntuples already written -- test/rjpsi/hammer/add_hammer_weights.py
Both build the same `leaves` dict and call the same HammerSession.weights(), so
the two paths are identical by construction, not by discipline. The closure test
is add_hammer_weights.py --closure.

BRANCH SCHEMA (fixed, independent of any fit):
  hammer_weight              nominal Kiselev -> Harrison weight
  hammer_ff_ev<ii>_up / _dn  ii = 00..14, the +/-1 sigma FF eigenvariations
  hammer_status              0 ok / 1 not a signal channel / 2 gen leaves
                             missing / 3 Hammer declined the process /
                             4 weight not finite
The 15 directions are structural: Hammer's BctoJpsiBGLVar carries 15 delta_e
error coordinates (avec0..3, bvec0..3, cvec0..2, dvec0..3), so the covariance is
15x15 and the ntuple always holds 2x15 variations. How many of them are actually
non-degenerate is a property of the FIT, not of the schema -- the card reports it
and the plotter drops the null ones. Keeping the schema fixed means a refit never
changes the ntuple layout.

Weights are written for gen_bc_decay == 1 (J/psi mu nu) and == 7 (J/psi tau nu)
only; every other row is NaN with a non-zero hammer_status, so "no weight" is
always distinguishable from "weight = 1".
'''

import json
import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# schema / configuration  (all confirmed against the live Hammer v2 install)
# ---------------------------------------------------------------------------
N_COORDS   = 15                  # delta_e0..14 of BctoJpsiBGLVar
PROCESS    = 'BcJpsi'            # FF-scheme key
XTOY       = 'BctoJpsi'          # set_ff_eigenvectors / set_options class prefix
SCHEME     = 'Harrison'          # our label for the target scheme
FF_TARGET  = 'BGLVar'
FF_INPUT   = 'Kiselev'
JPSI_PDGID = 443

MU_CODE, TAU_CODE = 1, 7

# gen-leaf prefixes written by RJPsiGenHistory.gen_hammer_p4 (see bc_branches)
BC_PREFIX = 'gen_b'
MUP, MUM  = 'gen_ham_mup', 'gen_ham_mum'
LEP, NU   = 'gen_ham_lep', 'gen_ham_nu'
TMU, TNT, TNM = 'gen_ham_taumu', 'gen_ham_taunut', 'gen_ham_taunum'

MU_LEAVES  = (BC_PREFIX, MUP, MUM, LEP, NU)
TAU_LEAVES = (BC_PREFIX, MUP, MUM, LEP, NU, TMU, TNT, TNM)
ALL_LEAVES = TAU_LEAVES
COMPONENTS = ('pt', 'eta', 'phi', 'mass', 'pdgid')

# every branch the reweighting needs to be read back from an existing ntuple
INPUT_BRANCHES = ['gen_bc_decay'] + ['%s_%s' % (p, c)
                                     for p in ALL_LEAVES for c in COMPONENTS]

STATUS_OK, STATUS_NOT_SIGNAL, STATUS_NO_LEAVES, STATUS_DECLINED, STATUS_NONFINITE = range(5)

VAR_LABELS = ['ev%02d_%s' % (j, tag)
              for j in range(N_COORDS) for tag in ('up', 'dn')]
BRANCH_NOMINAL = 'hammer_weight'
BRANCH_STATUS  = 'hammer_status'
BRANCH_NAMES   = ([BRANCH_NOMINAL]
                  + ['hammer_ff_%s' % lab for lab in VAR_LABELS]
                  + [BRANCH_STATUS])

# The only P1 convention a card may be fitted in. Hammer's amplitude reads P1
# through the a- form factor, which implies A0 = (1+r)/(2 sqrt r) * P1, i.e.
# P1 = sqrt(r)/(1+r) * F2 with Harrison's F2 = 2 A0. Cards fitted with the
# earlier assumption P1 = F2/2 carry an H_t 6.3% too large in amplitude.
P1_CONVENTION = 'P1 = sqrt(r)/(1+r) * F2'

DEFAULT_CARD = os.path.join(os.environ.get('CMSSW_BASE', ''), 'src', 'Bmmm',
                            'Analysis', 'data', 'harrison_bglvar.json')


# ---------------------------------------------------------------------------
# coefficient card
# ---------------------------------------------------------------------------
_VEC_LEN = {'avec': 4, 'bvec': 4, 'cvec': 3, 'dvec': 4}


def load_card(path):
    '''Read and validate the FF card. Raises rather than degrading: a wrong
    card is a wrong measurement, and a silent fallback to Hammer's default
    (Harrison-2020) would look exactly like a successful run.'''
    with open(path) as fin:
        card = json.load(fin)

    for key, n in _VEC_LEN.items():
        if key not in card:
            raise KeyError('FF card %s has no %r' % (path, key))
        if len(card[key]) != n:
            raise ValueError('FF card %s: %s has %d entries, expected %d'
                             % (path, key, len(card[key]), n))
        norm = float(np.sum(np.asarray(card[key], dtype=float) ** 2))
        if norm >= 1.:
            raise ValueError('FF card %s: %s violates the BGL unitarity bound '
                             '(sum a_n^2 = %.3f >= 1). The fit is not usable -- '
                             're-run harrison_to_hammer_bgl.py.' % (path, key, norm))

    evecs = np.asarray(card.get('evecs', []), dtype=float)
    sqev  = np.asarray(card.get('sqrt_evals', []), dtype=float)
    if evecs.size:
        if evecs.shape != (N_COORDS, N_COORDS):
            raise ValueError('FF card %s: evecs is %s, expected (%d, %d)'
                             % (path, evecs.shape, N_COORDS, N_COORDS))
        if sqev.shape != (N_COORDS,):
            raise ValueError('FF card %s: sqrt_evals has %d entries, expected %d'
                             % (path, sqev.size, N_COORDS))
    card['evecs'], card['sqrt_evals'] = evecs, sqev
    return card


def card_summary(card):
    sqev = card['sqrt_evals']
    ndof = int(np.sum(sqev > 1e-4 * sqev.max())) if sqev.size else 0
    return ('FF card %r (fit %s, covariance %s): unitarity %s ; %d/%d '
            'non-degenerate eigendirections'
            % (card.get('name', '?'), card.get('fit_date', '?'),
               card.get('covariance_status', '?'),
               ' '.join('%s=%.3f' % (k, np.sum(np.asarray(card[k]) ** 2))
                        for k in ('avec', 'bvec', 'cvec', 'dvec')),
               ndof, N_COORDS))


# ---------------------------------------------------------------------------
# leaves: the only input the reweighting takes
# ---------------------------------------------------------------------------
def leaves_from_row(row, i):
    '''Post-hoc builder: one row of an uproot array dict, keyed by the ntuple
    branch names.

    There is deliberately no second builder for the production-time path: the
    inspector hands over its event_tofill dict, which already holds exactly
    these branches under exactly these names. Same keys, same values, same
    code -- that is what makes the inline and post-hoc weights identical.
    '''
    return dict((b, row[b][i]) for b in INPUT_BRANCHES)


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def channel_of(leaves):
    '''"mu" / "tau" / None -- None also when the gen leaves are incomplete.'''
    code = leaves.get('gen_bc_decay', float('nan'))
    if not _finite(code):
        return None, STATUS_NOT_SIGNAL
    code = int(round(float(code)))
    if code not in (MU_CODE, TAU_CODE):
        return None, STATUS_NOT_SIGNAL
    needed = MU_LEAVES if code == MU_CODE else TAU_LEAVES
    if not all(_finite(leaves.get('%s_pt' % p)) for p in needed):
        return None, STATUS_NO_LEAVES
    return ('mu' if code == MU_CODE else 'tau'), STATUS_OK


# ---------------------------------------------------------------------------
# Hammer session
# ---------------------------------------------------------------------------
class HammerSession(object):
    '''One Hammer run: set up once, then weights() per event.

    Not a singleton by construction, but Hammer's init_run is a per-process
    global, so build exactly one of these per job.
    '''

    DECAY_CHAINS = (['BcJpsiMuNu', 'JpsiMuMu'],
                    ['BcJpsiTauNu', 'TauMuNuNu', 'JpsiMuMu'])

    def __init__(self, card_path=None, variations=True, allow_stale=False,
                 verbose=True, decay_chains=None, pure_ps_denominator=None,
                 pure_ps_numerator=None, ff_target=None, general_rates=True,
                 scale_coefficients=None, apply_card=True,
                 allow_legacy_p1=False):
        import hammer as H                                  # noqa: N813
        self.H = H

        # The FF card only means anything to a BGL class. For Kiselev (the
        # denominator test) it is neither loaded nor required.
        wants_card = ff_target is None or str(ff_target).startswith('BGL')
        if wants_card:
            self.card_path = card_path or DEFAULT_CARD
            self.card = load_card(self.card_path)
            self.variations = bool(variations) and self.card['evecs'].size > 0
        else:
            self.card_path, self.card, self.variations = None, None, False

        # BGLVar carries the 15 linearised FF variational indices; BGL does not.
        # When no eigenvariations are wanted, the plain class is far cheaper --
        # every rate integral and weight tensor drops 15 dimensions.
        self.ff_target = ff_target or (FF_TARGET if self.variations else 'BGL')
        self.has_variations = self.ff_target.endswith('Var')
        self.is_bgl = self.ff_target.startswith('BGL')
        if self.variations and not self.has_variations:
            raise ValueError('variations requested but ff_target=%r has no '
                             'variational indices' % self.ff_target)
        self.general_rates = bool(general_rates)

        # deliberate perturbation, for the negative control: scale one
        # coefficient block so that a silently-refused set_options becomes
        # visible instead of hiding behind Hammer's own default
        if scale_coefficients and self.card:
            key, factor = scale_coefficients
            self.card[key] = [x * float(factor) for x in self.card[key]]

        # a card fitted in the wrong P1 convention biases every tau weight;
        # refuse it unless explicitly asked (only diagnostics should ever ask)
        if self.card and self.card.get('p1_convention') != P1_CONVENTION \
                and not allow_legacy_p1:
            raise RuntimeError(
                'FF card %s was not fitted in the P1 convention Hammer uses '
                '(%r; card says %r). Its dvec gives H_t 6.3%% too large in '
                'amplitude, i.e. about +1%% on R(J/psi). Refit it with '
                'harrison_to_hammer_bgl.py --json. To run a diagnostic on it '
                "anyway, pass allow_legacy_p1 (':allow-legacy-p1' on --hammer)."
                % (self.card_path, P1_CONVENTION,
                   self.card.get('p1_convention', '<missing>')))

        status = str(self.card.get('covariance_status', 'unknown')).lower() \
            if self.card else 'n/a'
        if self.variations and status != 'validated' and not allow_stale:
            raise RuntimeError(
                "FF card %s declares covariance_status=%r. The eigenvariations "
                "are not trustworthy -- re-run harrison_to_hammer_bgl.py and set "
                "'validated', or pass allow_stale to write them anyway."
                % (self.card_path, self.card.get('covariance_status')))

        if verbose:
            print('#### Hammer: %s' % (card_summary(self.card) if self.card else
                                      'no FF card (numerator is %s, Hammer\'s own '
                                      'parametrisation)' % ff_target))
            print('####   %s -> %s (%s), variations %s, general rates %s'
                  % (FF_INPUT, SCHEME, self.ff_target,
                     'ON' if self.variations else 'OFF',
                     'ON' if self.general_rates else 'OFF (no rate integration)'))
            if pure_ps_denominator or pure_ps_numerator:
                print('####   pure phase space: denominator %s, numerator %s'
                      % (sorted(pure_ps_denominator or ()) or '-',
                         sorted(pure_ps_numerator or ()) or '-'))

        self.ham = H.Hammer()
        # one include_decay call PER CHAIN: a single call listing every vertex
        # is AND-ed and silently yields w == 1 for every event.
        self.decay_chains = [list(c) for c in (decay_chains or self.DECAY_CHAINS)]
        for chain in self.decay_chains:
            self.ham.include_decay(chain)
        self.ham.add_ff_scheme(SCHEME, {PROCESS: self.ff_target})
        self.ham.set_ff_input_scheme({PROCESS: FF_INPUT})
        self.ham.set_units('GeV')

        # Pure phase space vertices (manual sec. N): |M|^2 = 1 x m^(6-2n) for the
        # declared vertices, in the numerator and/or the denominator. Declaring
        # the whole chain pure-PS in the DENOMINATOR turns the weight from a
        # ratio into |M_numerator|^2 up to a constant -- which is what lets the
        # q2 spectrum be produced from Hammer alone, with no MC sample.
        # Must be called BEFORE init_run.
        if pure_ps_denominator:
            self.ham.add_pure_ps_vertices(set(pure_ps_denominator),
                                          H.WTerm.DENOMINATOR)
        if pure_ps_numerator:
            self.ham.add_pure_ps_vertices(set(pure_ps_numerator),
                                          H.WTerm.NUMERATOR)
        self.pure_ps = {'denominator': set(pure_ps_denominator or ()),
                        'numerator': set(pure_ps_numerator or ())}

        self.ham.init_run()

        # general_rates=False is only legal when WC specializations are applied
        # to the weights (manual sec. III K): without one, Hammer refuses with
        # "GeneralRates set to false with no WC specializations declared in
        # weights" and integrates the rate anyway. So it is NOT a way to skip the
        # rate integration in a plain run -- what actually avoids the expensive
        # tensor integral is ff_target='BGL' rather than 'BGLVar'.
        if not self.general_rates:
            self.ham.set_options('Hammer: {GeneralRates: false}')

        # the coefficient card only means anything to a BGL class; for
        # Kiselev, EFG etc. the parametrisation is Hammer's own and must be
        # left alone (that is the point of testing it)
        # apply_card=False leaves Hammer on its OWN built-in coefficients: used
        # to compare the C++ evaluator with the Python port with our fit
        # entirely out of the loop
        self.apply_card = bool(apply_card)
        if self.ff_target.startswith('BGL') and self.apply_card:
            self.ham.set_options(self._central_options())
        if self.variations:
            self.ham.set_options(self._abcdmatrix_options())

        # coordinate vectors, in BRANCH_NAMES order: nominal first, then
        # (up, dn) per direction. The 15-vector is sticky in Hammer, so every
        # call passes all 15 coordinates, not just the one that changes.
        self._coords = [np.zeros(N_COORDS)]
        for j in range(N_COORDS):
            for sign in (+1., -1.):
                vec = np.zeros(N_COORDS)
                vec[j] = sign
                self._coords.append(vec)
        if not self.variations:
            self._coords = self._coords[:1]

        if self.variations:
            self._verify_eigenvector_plumbing(verbose=verbose)

        self.n_ok = self.n_declined = self.n_nonfinite = 0

    def _verify_eigenvector_plumbing(self, verbose=True):
        '''Set one coordinate and read it back.

        set_ff_eigenvectors is sticky and takes the whole 15-vector; if the
        process/group strings are wrong it is a silent no-op and every
        "variation" comes back equal to the nominal weight -- a systematic of
        exactly zero, which looks like a small uncertainty rather than a bug.
        retrieve_ff_eigenvectors(process, group) makes that testable, so test it
        once at setup instead of trusting it for a whole production.
        '''
        probe = [0.] * N_COORDS
        probe[0] = 1.
        self.ham.set_ff_eigenvectors(XTOY, self.ff_target, probe)
        got = self.ham.retrieve_ff_eigenvectors(XTOY, self.ff_target)

        vals = sorted(float(v) for v in got.values())
        ok = (len(vals) == N_COORDS
              and abs(vals[-1] - 1.) < 1e-9
              and all(abs(v) < 1e-9 for v in vals[:-1]))
        if not ok:
            raise RuntimeError(
                'set_ff_eigenvectors(%r, %r, ...) did not take: read back %r. '
                'The eigenvariations would all equal the nominal weight. Check '
                'the process/group strings against show_available_ff_params().'
                % (XTOY, self.ff_target, got))

        self.ham.set_ff_eigenvectors(XTOY, self.ff_target, [0.] * N_COORDS)
        if verbose:
            print('####   eigenvector plumbing verified: %d coordinates, '
                  'set/retrieve round-trips' % len(vals))

    # -- option strings ----------------------------------------------------
    @staticmethod
    def _fmt(vec):
        return '[' + ', '.join('%.10g' % x for x in vec) + ']'

    def _central_options(self):
        return '%s%s: { %s }' % (XTOY, self.ff_target, ', '.join(
            '%s: %s' % (k, self._fmt(self.card[k]))
            for k in ('avec', 'bvec', 'cvec', 'dvec')))

    def _abcdmatrix_options(self):
        # column j = the 1-sigma principal direction j, so coordinate e_j = 1
        # shifts the coefficients by exactly 1 sigma along it.
        mat = self.card['evecs'] * self.card['sqrt_evals'][None, :]
        rows = '[' + ','.join('[' + ','.join('%.10g' % x for x in row) + ']'
                              for row in mat) + ']'
        return '%s%s: { abcdmatrix: %s }' % (XTOY, self.ff_target, rows)

    # -- process construction ---------------------------------------------
    @staticmethod
    def _p4(leaves, pfx):
        pt, eta, phi, mass = (float(leaves['%s_%s' % (pfx, c)])
                              for c in ('pt', 'eta', 'phi', 'mass'))
        px = pt * math.cos(phi)
        py = pt * math.sin(phi)
        pz = pt * math.sinh(eta)
        return (math.sqrt(px * px + py * py + pz * pz + mass * mass), px, py, pz)

    @staticmethod
    def _pid(leaves, pfx):
        return int(round(float(leaves['%s_pdgid' % pfx])))

    @staticmethod
    def _sum(*vecs):
        return tuple(sum(c) for c in zip(*vecs))

    def _add(self, proc, p4, pid):
        return proc.add_particle(self.H.Particle(self.H.FourMomentum(*p4), pid))

    def _process_mu(self, lv):
        bc, mup, mum = (self._p4(lv, BC_PREFIX), self._p4(lv, MUP), self._p4(lv, MUM))
        lep, nu = self._p4(lv, LEP), self._p4(lv, NU)
        proc = self.H.Process()
        i_bc = self._add(proc, bc, self._pid(lv, BC_PREFIX))
        i_j  = self._add(proc, self._sum(mup, mum), JPSI_PDGID)
        i_l  = self._add(proc, lep, self._pid(lv, LEP))
        i_n  = self._add(proc, nu,  self._pid(lv, NU))
        i_mp = self._add(proc, mup, self._pid(lv, MUP))
        i_mm = self._add(proc, mum, self._pid(lv, MUM))
        proc.add_vertex(i_bc, [i_j, i_l, i_n])
        proc.add_vertex(i_j, [i_mp, i_mm])
        return proc

    def _process_tau(self, lv):
        bc, mup, mum = (self._p4(lv, BC_PREFIX), self._p4(lv, MUP), self._p4(lv, MUM))
        nu  = self._p4(lv, NU)
        tmu, tnt, tnm = self._p4(lv, TMU), self._p4(lv, TNT), self._p4(lv, TNM)
        proc = self.H.Process()
        i_bc = self._add(proc, bc, self._pid(lv, BC_PREFIX))
        i_j  = self._add(proc, self._sum(mup, mum), JPSI_PDGID)
        i_t  = self._add(proc, self._sum(tmu, tnt, tnm), self._pid(lv, LEP))
        i_n  = self._add(proc, nu,  self._pid(lv, NU))
        i_mp = self._add(proc, mup, self._pid(lv, MUP))
        i_mm = self._add(proc, mum, self._pid(lv, MUM))
        i_tm = self._add(proc, tmu, self._pid(lv, TMU))
        i_tt = self._add(proc, tnt, self._pid(lv, TNT))
        i_tn = self._add(proc, tnm, self._pid(lv, TNM))
        proc.add_vertex(i_bc, [i_j, i_t, i_n])
        proc.add_vertex(i_j, [i_mp, i_mm])
        proc.add_vertex(i_t, [i_tm, i_tt, i_tn])
        return proc

    # -- the one public method --------------------------------------------
    def weights(self, leaves):
        '''{branch: value} for one event. NaN everywhere but hammer_status when
        the event is not a reweightable signal decay.'''
        out = dict((b, float('nan')) for b in BRANCH_NAMES)

        channel, status = channel_of(leaves)
        out[BRANCH_STATUS] = float(status)
        if channel is None:
            return out

        self.ham.init_event()
        proc = (self._process_mu if channel == 'mu' else self._process_tau)(leaves)
        if self.ham.add_process(proc) == 0:
            self.n_declined += 1
            out[BRANCH_STATUS] = float(STATUS_DECLINED)
            return out
        self.ham.process_event()

        vals = []
        for coord in self._coords:
            if self.has_variations:
                self.ham.set_ff_eigenvectors(XTOY, self.ff_target,
                                             [float(x) for x in coord])
            vals.append(self.ham.get_weight(SCHEME))

        if not _finite(vals[0]):
            # a handful of events per mille: the truth Bc four-momentum does not
            # exactly equal J/psi + lepton + nu (FSR / rounding), so a node mass^2
            # comes out slightly negative. Event-level, not a coefficient problem.
            self.n_nonfinite += 1
            out[BRANCH_STATUS] = float(STATUS_NONFINITE)
            return out

        self.n_ok += 1
        out[BRANCH_NOMINAL] = float(vals[0])
        for name, val in zip(BRANCH_NAMES[1:1 + len(vals) - 1], vals[1:]):
            out[name] = float(val)
        return out

    def summary(self):
        return ('#### Hammer: %d events reweighted, %d declined by Hammer, '
                '%d with non-finite weight'
                % (self.n_ok, self.n_declined, self.n_nonfinite))


def make_hammer_session(spec, verbose=True, **kwargs):
    '''Build a session from a command-line spec, or None when the spec is empty.

    spec forms:
      ''                     -> None (no reweighting)
      'default'              -> $CMSSW_BASE/src/Bmmm/Analysis/data/harrison_bglvar.json
      '/path/to/card.json'   -> that card
      '<spec>:nominal'       -> central weight only, no eigenvariations
      '<spec>:allow-stale'   -> write variations from a not-yet-validated card
      '<spec>:allow-legacy-p1' -> accept a card fitted with P1 = F2/2 (diagnostics only)
    '''
    if not spec:
        return None
    parts = spec.split(':')
    path = parts[0]
    flags = set(parts[1:])
    unknown = flags - {'nominal', 'allow-stale', 'allow-legacy-p1'}
    if unknown:
        raise ValueError('--hammer: unknown flag(s) %s' % sorted(unknown))
    return HammerSession(card_path=None if path in ('', 'default') else path,
                         variations='nominal' not in flags,
                         allow_stale='allow-stale' in flags,
                         allow_legacy_p1='allow-legacy-p1' in flags,
                         verbose=verbose, **kwargs)
