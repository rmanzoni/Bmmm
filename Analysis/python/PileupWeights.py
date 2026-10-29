'''
Pileup reweighting of Run 3 MC to the centrally produced data pileup profiles.

WHAT IS REWEIGHTED
  The MC "true" pileup, PileupSummaryInfo::getTrueNumInteractions() at bunch
  crossing 0 (the `nti` branch), is matched to the data distribution produced by
  pileupCalc.py in `--calcMode true` (TWiki PileupJSONFileforData, "True
  calculation mode": the two are the same quantity by construction).

      w(nti) = P_data(bin of nti) / P_MC(bin of nti)

  P_data  = data pileup histogram for one data-taking period, normalised to 1
            (overflow beyond the MC range included in the normalisation, so any
            data the MC cannot cover shows up as <w> < 1 instead of vanishing)
  P_MC    = the pileup profile the MC campaign was GENERATED with, i.e. the
            probValue list of the SimGeneral/MixingModule cfi. NOT the nti
            distribution in our ntuples: those are after the skim and the
            candidate selection, both of which depend on pileup.
  bin     = floor(nti). PileUp.cc draws nti uniformly inside [k, k+1) for a
            probFunction profile, so bin k of the profile is exactly nti in [k, k+1).

ONE implementation, TWO entry points (as for HammerFF):
  * at ntuple production time  -- JpsiChargedInspector with --pu <campaign>
  * on ntuples already written -- test/rjpsi/pileup/add_pu_weights.py
Both call PileupSession.weights_array(); add_pu_weights.py --closure proves
the two agree bit for bit.

BRANCH SCHEMA (fixed, independent of the campaign):
  pu_weight_<year>        nominal, minimum-bias cross section 69200 ub
  pu_weight_<year>_up     72400 ub  (+4.6%)
  pu_weight_<year>_down   66000 ub  (-4.6%)
  for year in 2022 .. 2026. The branch name says which data year the weight
  maps the MC onto. A campaign fills only the years it is used for:
      Summer22 / Summer22EE      -> 2022   (vs 2022 BCD / 2022 EFG)
      Summer23 / Summer23BPix    -> 2023   (vs 2023 BC  / 2023 D)
      Summer24                   -> 2024, 2025, 2026 (no 2025/2026 MC exists)
  Everything else is NaN, so "no weight for this year" is never confused with
  "weight = 1". Which data period sits behind each year is in the card.

THE CARD (Bmmm/Analysis/data/pu_weights_run3.json) holds the raw inputs: the
data histograms per period and variation, and the MC generation profiles. The
weights themselves are computed here, at load time, so there is exactly one
place where the ratio is formed. The card is built by
test/rjpsi/pileup/build_pu_card.py.

Do NOT renormalise the weights after the selection: <w> over selected events
differs from 1 whenever the selection efficiency depends on pileup, and that
difference is physics, not a normalisation bug.
'''

import json
import os

import numpy as np

SCHEMA_VERSION = 1

YEARS      = ('2022', '2023', '2024', '2025', '2026')
VARIATIONS = ('nominal', 'up', 'down')
_SUFFIX    = {'nominal': '', 'up': '_up', 'down': '_down'}

INPUT_BRANCH = 'nti'


def branch_name(year, variation):
    return 'pu_weight_%s%s' % (year, _SUFFIX[variation])


BRANCH_NAMES = [branch_name(y, v) for y in YEARS for v in VARIATIONS]

# the all-NaN row written when --pu is not given, or for data. Callers must copy
# it (dict.update does), never mutate it.
NAN_ROW = dict.fromkeys(BRANCH_NAMES, np.nan)

DEFAULT_CARD = os.path.join(os.environ.get('CMSSW_BASE', ''), 'src', 'Bmmm',
                            'Analysis', 'data', 'pu_weights_run3.json')


# ---------------------------------------------------------------------------
# card
# ---------------------------------------------------------------------------
def load_card(path):
    '''Read and validate the card. Raises rather than degrading: a card with a
    missing variation or a mismatched binning must never produce weights.'''
    if not os.path.isfile(path):
        raise IOError('pileup card %r not found -- build it with '
                      'test/rjpsi/pileup/build_pu_card.py' % path)
    with open(path) as f:
        card = json.load(f)

    def bad(msg):
        raise ValueError('pileup card %s: %s' % (path, msg))

    if card.get('schema') != SCHEMA_VERSION:
        bad('schema %r, this code reads %r' % (card.get('schema'), SCHEMA_VERSION))
    nbins = card.get('nbins')
    if not isinstance(nbins, int) or nbins <= 0:
        bad('nbins must be a positive int, got %r' % (nbins,))

    for name, prof in card.get('mc_profiles', {}).items():
        prob = prof.get('prob')
        if prob is None or len(prob) != nbins:
            bad('MC profile %r has %s bins, card has %d'
                % (name, None if prob is None else len(prob), nbins))
        p = np.asarray(prob, dtype=np.float64)
        if np.any(p < 0) or not np.isfinite(p).all() or p.sum() <= 0:
            bad('MC profile %r is not a valid probability list' % name)

    for name, per in card.get('data', {}).items():
        if per.get('year') not in YEARS:
            bad('data period %r has year %r, not one of %s' % (name, per.get('year'), YEARS))
        for var in VARIATIONS:
            h = per.get('hist', {}).get(var)
            if h is None or len(h) != nbins:
                bad('data period %r variation %r missing or not %d bins' % (name, var, nbins))
            if 'outside' not in per or var not in per['outside']:
                bad('data period %r has no "outside" entry for %r' % (name, var))
            if np.sum(h) + per['outside'][var] <= 0:
                bad('data period %r variation %r is empty' % (name, var))

    for name, camp in card.get('campaigns', {}).items():
        if camp.get('mc_profile') not in card.get('mc_profiles', {}):
            bad('campaign %r uses unknown MC profile %r' % (name, camp.get('mc_profile')))
        years = camp.get('periods', {})
        if not years:
            bad('campaign %r maps onto no data year' % name)
        for year, period in years.items():
            if year not in YEARS:
                bad('campaign %r maps onto year %r, not one of %s' % (name, year, YEARS))
            if period not in card.get('data', {}):
                bad('campaign %r year %s uses unknown data period %r' % (name, year, period))
            if card['data'][period]['year'] != year:
                bad('campaign %r puts period %r (year %s) under year %s'
                    % (name, period, card['data'][period]['year'], year))
    return card


def weight_table(data_hist, data_outside, mc_prob):
    '''The per-bin weight and the numbers that say whether it can be trusted.

    Returns (w, info). w[k] = P_data[k] / P_MC[k] where P_MC[k] > 0, NaN where
    the MC has no events (unreachable by construction: the MC cannot produce an
    nti the profile forbids). info:
      data_covered  fraction of the data profile sitting in bins the MC
                    populates. This is also <w> over the full generated MC
                    sample, so anything below 1 is data the MC cannot describe.
      neff_frac     effective sample size of the reweighted MC over its raw
                    size, (sum_k m_k w_k)^2 / sum_k m_k w_k^2 : the statistical
                    price of the reweighting (1 = free).
      w_max         the largest weight any generated event can receive.
    '''
    d = np.asarray(data_hist, dtype=np.float64)
    m = np.asarray(mc_prob,   dtype=np.float64)
    d = d / (d.sum() + float(data_outside))
    m = m / m.sum()
    populated = m > 0
    w = np.full_like(d, np.nan)
    w[populated] = d[populated] / m[populated]
    wp, mp = w[populated], m[populated]
    s1 = float(np.sum(mp * wp))
    s2 = float(np.sum(mp * wp * wp))
    info = {
        'data_covered': float(d[populated].sum()),
        'neff_frac'   : (s1 * s1 / s2) if s2 > 0 else float('nan'),
        'w_max'       : float(wp.max()) if wp.size else float('nan'),
    }
    return w, info


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------
class PileupSession(object):
    '''Weights for one MC campaign, for every data year it is used for.'''

    def __init__(self, campaign, card_path=None, allow_unconfirmed=False, verbose=True):
        self.card_path = card_path or DEFAULT_CARD
        self.card      = load_card(self.card_path)
        camps          = self.card['campaigns']
        if campaign not in camps:
            raise ValueError('--pu: campaign %r not in the card; known: %s'
                             % (campaign, sorted(camps)))
        camp = camps[campaign]
        if not camp.get('confirmed', False) and not allow_unconfirmed:
            raise RuntimeError(
                '--pu: the MC generation profile of campaign %r (%s) is not '
                'confirmed. Check it with test/rjpsi/pileup/'
                'mc_pu_profile_from_miniaod.py and set confirmed=True in '
                'pu_config_run3.py, or pass ":allow-unconfirmed".'
                % (campaign, camp['mc_profile']))

        self.campaign   = campaign
        self.mc_profile = camp['mc_profile']
        self.periods    = dict(camp['periods'])            # year -> data period
        self.nbins      = self.card['nbins']
        mc_prob         = self.card['mc_profiles'][self.mc_profile]['prob']

        self.tables = {}
        self.info   = {}
        for year, period in sorted(self.periods.items()):
            per = self.card['data'][period]
            for var in VARIATIONS:
                w, info = weight_table(per['hist'][var], per['outside'][var], mc_prob)
                self.tables[branch_name(year, var)] = w
                self.info[branch_name(year, var)]   = info
        # always-NaN branches: the years this campaign is not used for
        self.nan_branches = [b for b in BRANCH_NAMES if b not in self.tables]

        self.n_calls     = 0
        self.n_nonfinite = 0     # nti NaN, negative or beyond the profile
        self.n_unreached = 0     # nti inside the range but in a bin with P_MC = 0
        if verbose:
            print(self.summary(header_only=True))

    # ---- evaluation -----------------------------------------------------------
    def weights_array(self, nti):
        '''nti (any array-like) -> {branch: float32 array}. The single place the
        weight is evaluated; the inline and post-hoc paths both end up here.

        nti is taken as float32 first: that is what PileupSummaryInfo stores and
        what the ntuple stores, so floor() sees the same number on both paths.
        '''
        x   = np.atleast_1d(np.asarray(nti, dtype=np.float32))
        k   = np.floor(x)
        ok  = np.isfinite(x) & (k >= 0) & (k < self.nbins)
        idx = np.where(ok, k, 0).astype(np.int64)
        out = {}
        for b, table in self.tables.items():
            v = table[idx]
            v[~ok] = np.nan
            out[b] = v.astype(np.float32)
        for b in self.nan_branches:
            out[b] = np.full(x.shape, np.nan, dtype=np.float32)

        self.n_calls     += x.size
        self.n_nonfinite += int(np.count_nonzero(~ok))
        first = next(iter(self.tables.values()))
        self.n_unreached += int(np.count_nonzero(ok & ~np.isfinite(first[idx])))
        return out

    def weights(self, nti):
        '''Scalar version for the event loop: nti -> {branch: float}.'''
        return {b: float(v[0]) for b, v in self.weights_array(nti).items()}

    # ---- reporting ------------------------------------------------------------
    def summary(self, header_only=False):
        lines = ['#### pileup weights: campaign %s, MC profile %s'
                 % (self.campaign, self.mc_profile),
                 '####   card %s' % self.card_path]
        for year, period in sorted(self.periods.items()):
            per = self.card['data'][period]
            lines.append('####   pu_weight_%s*  <- data period %s (%s)'
                         % (year, period, per.get('source', {}).get('kind', '?')))
            for var in VARIATIONS:
                i = self.info[branch_name(year, var)]
                lines.append('####       %-8s data covered by MC %.5f   '
                             'effective MC fraction %.3f   max weight %.2f'
                             % (var, i['data_covered'], i['neff_frac'], i['w_max']))
        if not header_only:
            lines.append('####   evaluated %d times, %d with unusable nti, '
                         '%d in bins the MC profile forbids'
                         % (self.n_calls, self.n_nonfinite, self.n_unreached))
        return '\n'.join(lines)


def make_pu_session(spec, verbose=True):
    '''Build a session from a command-line spec, or None when the spec is empty.

    spec forms:
      ''                                  -> None (no pileup weights)
      'Summer24'                          -> campaign Summer24, default card
      'Summer24:/path/to/card.json'       -> that card
      '<spec>:allow-unconfirmed'          -> accept a campaign whose MC profile
                                             has not been checked yet
    '''
    if not spec:
        return None
    parts    = spec.split(':')
    campaign = parts[0]
    flags    = set(p for p in parts[1:] if p in ('allow-unconfirmed',))
    paths    = [p for p in parts[1:] if p not in flags]
    if len(paths) > 1:
        raise ValueError('--pu: more than one card path in %r' % spec)
    return PileupSession(campaign,
                         card_path=paths[0] if paths else None,
                         allow_unconfirmed='allow-unconfirmed' in flags,
                         verbose=verbose)
