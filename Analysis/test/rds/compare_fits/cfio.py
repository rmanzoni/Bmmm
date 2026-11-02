'''
Reading the R(Ds) ntuple for the fit-scheme comparison: what is in the file,
which branches to read, the selection of the truth-matched signal, and the
quantities derived offline (gen targets, choices between the two neutrino
solutions).

Everything is discovered from the branch names, so the suite runs on ntuples
with three schemes (cons, phic, free; before 2 Nov 2026) or four (+ dsc), and
picks up new Bs hypotheses or observables without changes.
'''

import os
import re
import json
import time
import hashlib

import numpy as np

NAN = float('nan')

# preferred order of the schemes in tables and plots; others follow, sorted
SCHEME_ORDER = ('cons', 'phic', 'dsc', 'free')
SCHEME_LABEL = {
    'cons': 'cons (phi + Ds)',
    'phic': 'phic (phi)',
    'dsc' : 'dsc (Ds)',
    'free': 'free (none)',
}
SCHEME_COLOR = {'cons': '#1b6ca8', 'phic': '#d1495b', 'dsc': '#2e933c', 'free': '#edae49'}
FALLBACK_COLORS = ('#6a4c93', '#8c564b', '#17becf', '#7f7f7f')

HYP_ORDER = ('coll', 'xyz_bs', 'math1_bs', 'math2_bs', 'math_avg', 'math_oracle')
# choices between the two neutrino solutions, computed offline. (Choosing by
# |p(Bs)| adds nothing: math1 is the lower-|p| solution by construction of the
# solver, except where the discriminant was clamped; the fraction is in the
# report.)
DERIVED_HYPS = ('math_avg', 'math_oracle')
# hypotheses that need the truth: shown, never ranked
ORACLE_HYPS = ('math_oracle',)

HYP_OBS = ('q2', 'm_miss2', 'pt_miss', 'pt_miss_sca', 'e_star_mu', 'e_hash_mu',
           'cos_theta_phi_ds', 'cos_theta_mu_w', 'chi', 'pt', 'beta')
ANGLE_OBS = ('chi',)

# fit-dependent observables without a Bs hypothesis
SCHEME_OBS = ('mu_doca_ds', 'mu_doca_ds_sig', 'mu_poca_along_ds',
              'mu_ip3d_bs_pv', 'mu_ip3d_bs_pv_sig', 'mu_ip3d_bs_sv_sig',
              'mu_dist_to_bs_dir', 'mu_dist_along_bs_dir_pv',
              'ds_lxyz_bs', 'ds_lxyz_bs_sig', 'ds_lxy_bs_sig', 'ds_cos3d_bs',
              'bs_prob', 'bs_chi2', 'bs_lxyz_sig', 'bs_lxy_sig', 'bs_cos3d', 'bs_mass',
              'ds_prob', 'ds_lxyz_sig', 'ds_cos3d', 'ds_mass', 'ds_pt', 'phi_mass',
              'mcorr', 'p_perp', 'ds_p_perp', 'cos_theta_k_pi',
              'gamma_n', 'gamma_pt', 'gamma_dm', 'gamma_e_coll_over_pt',
              'math_real_bs')

# per-vertex quantities read for the fit-quality and vertex studies
VTX_Q = ('good', 'status', 'x', 'y', 'z', 'chi2', 'ndof', 'prob', 'mass', 'pt',
         'lxyz', 'lxyz_sig', 'cos3d', 'cov_xx', 'cov_xy', 'cov_xz', 'cov_yy', 'cov_yz', 'cov_zz')
VERTICES = ('phi', 'ds', 'bs')

# scheme-independent observables (baseline in the tables; BDT inputs)
COMMON_OBS = ('mass', 'kkpi_mass', 'kk_mass', 'dr_ds_mu', 'dr_kk', 'dr_phi_pi',
              'mu_pt', 'mu_eta', 'pi_pt', 'k1_pt', 'k2_pt',
              'mu_bs_dxy_sig', 'mu_dxy_sig', 'mu_reliso_clean_03', 'b_reliso_clean_03')

GEN_BRANCHES = (['gen_cand', 'gen_sig'] +
                ['gen_bs_%s' % q for q in ('pt', 'eta', 'phi', 'mass', 'beta', 'sv_x', 'sv_y', 'sv_z',
                                           'lxyz', 'ct')] +
                ['gen_%s' % q for q in ('q2', 'm_miss2', 'pt_miss', 'e_star_mu', 'e_hash_mu',
                                        'cos_theta_k_pi', 'cos_theta_phi_ds', 'cos_theta_mu_w', 'chi')] +
                ['gen_ham_%s_%s' % (p, q) for p in ('dsx', 'ds') for q in ('pt', 'eta', 'phi', 'mass')] +
                ['gen_vis_mu_%s' % q for q in ('pt', 'eta', 'phi')] +
                ['gen_pv_x', 'gen_pv_y', 'gen_pv_z'] +
                ['%s_gen_pdgid' % o for o in ('mu', 'k1', 'k2', 'pi')] +
                ['mu_gen_pt'])

ID_BRANCHES = ('run', 'lumi', 'event', 'cand_idx', 'charge_class', 'mu_trig_match',
               'pv_x', 'pv_y', 'pv_z')

_HLT_RE = re.compile(r'^HLT_Mu\d+(p\d+)?_IP\d+(p\d+)?$')

CLASSES = (0, 1, 2, 3)
CLASS_LABEL = {0: r'$D_s\mu\nu$', 1: r'$D_s^*\mu\nu$', 2: r'$D_s\tau\nu$', 3: r'$D_s^*\tau\nu$'}
CLASS_NAME = {0: 'Ds mu', 1: 'Ds* mu', 2: 'Ds tau', 3: 'Ds* tau'}
CLASS_COLOR = {0: '#1b6ca8', 1: '#7fb3d5', 2: '#c0392b', 3: '#f1948a'}
GROUPS = {'Ds mu': (0,), 'Ds* mu': (1,), 'Ds tau': (2,), 'Ds* tau': (3,),
          'mu': (0, 1), 'tau': (2, 3)}

M_MU = 0.1056583755


def scheme_color(s, schemes):
    if s in SCHEME_COLOR:
        return SCHEME_COLOR[s]
    others = [x for x in schemes if x not in SCHEME_COLOR]
    return FALLBACK_COLORS[others.index(s) % len(FALLBACK_COLORS)]


##########################################################################################
#####      DISCOVERY
##########################################################################################

class Layout(object):
    '''What the file contains: schemes, their Bs hypotheses (stored and
    derived), the observables of each, the HLT paths.'''

    def __init__(self, keys):
        keys = set(keys)
        self.keys = keys
        found = set(m.group(1) for m in (re.match(r'^(\w+?)_bs_good$', k) for k in keys) if m)
        found = [s for s in found if '%s_ds_good' % s in keys]
        self.schemes = [s for s in SCHEME_ORDER if s in found] + sorted(set(found) - set(SCHEME_ORDER))
        if not self.schemes:
            raise RuntimeError('no fit scheme found (no <s>_bs_good / <s>_ds_good branches): '
                               'is this an R(Ds) ntuple?')

        self.hyps, self.gamma_hyps = {}, {}
        for s in self.schemes:
            stored = set()
            gam = set()
            for k in keys:
                m = re.match(r'^%s_(\w+)_q2$' % s, k)
                if not m:
                    continue
                h = m.group(1)
                if h.startswith('gamma_'):
                    gam.add(h)
                else:
                    stored.add(h)
            hyps = [h for h in HYP_ORDER if h in stored] + sorted(stored - set(HYP_ORDER))
            if 'math1_bs' in stored and 'math2_bs' in stored:
                hyps += list(DERIVED_HYPS)
            self.hyps[s] = hyps
            self.gamma_hyps[s] = sorted(gam)

        self.paths = sorted(k for k in keys if _HLT_RE.match(k))
        self.has = lambda b: b in keys

    def hyp_obs(self, s, h):
        '''Observables available for scheme s, hypothesis h.'''
        if h in DERIVED_HYPS:
            return [q for q in HYP_OBS if '%s_math1_bs_%s' % (s, q) in self.keys]
        if h.startswith('gamma_'):
            return [q for q in ('q2', 'm_miss2') if '%s_%s_%s' % (s, h, q) in self.keys]
        return [q for q in HYP_OBS if '%s_%s_%s' % (s, h, q) in self.keys]

    def scheme_obs(self, s):
        return [q for q in SCHEME_OBS if '%s_%s' % (s, q) in self.keys]

    def common_obs(self):
        return [q for q in COMMON_OBS if q in self.keys]

    def branches(self):
        '''Every branch the suite reads.'''
        out = set(b for b in tuple(ID_BRANCHES) + tuple(GEN_BRANCHES) if b in self.keys)
        out |= set(self.paths)
        out |= set(self.common_obs())
        for s in self.schemes:
            for v in VERTICES:
                out |= set(b for b in ('%s_%s_%s' % (s, v, q) for q in VTX_Q) if b in self.keys)
            out |= set('%s_%s' % (s, q) for q in self.scheme_obs(s))
            for h in self.hyps[s]:
                if h in DERIVED_HYPS:
                    continue
                out |= set('%s_%s_%s' % (s, h, q) for q in self.hyp_obs(s, h))
            for h in self.gamma_hyps[s]:
                out |= set('%s_%s_%s' % (s, h, q) for q in self.hyp_obs(s, h))
            # |p(Bs)| of the two neutrino solutions, for the derived choices
            for k in ('math1_bs_eta', 'math2_bs_eta'):
                if '%s_%s' % (s, k) in self.keys:
                    out.add('%s_%s' % (s, k))
            for o in ('mu', 'k1', 'k2', 'pi'):
                b = '%s_%s_rf_pt' % (s, o)
                if b in self.keys:
                    out.add(b)
        return sorted(out)

    def summary(self):
        lines = ['schemes: %s' % ', '.join(self.schemes)]
        for s in self.schemes:
            lines.append('  %s: hypotheses %s; with photon %s; %d topology observables'
                         % (s, ', '.join(self.hyps[s]), ', '.join(self.gamma_hyps[s]) or '-',
                            len(self.scheme_obs(s))))
        lines.append('HLT paths: %s' % (', '.join(self.paths) or 'none'))
        return '\n'.join(lines)


##########################################################################################
#####      READING
##########################################################################################

def _fingerprint(path, tree, branches, extra):
    st = None
    try:
        s = os.stat(path)
        st = (s.st_size, int(s.st_mtime))
    except OSError:
        pass
    h = hashlib.sha1(json.dumps([path, tree, st, sorted(branches), extra]).encode()).hexdigest()
    return h[:16]


def read(path, tree='tree', max_entries=None, step='200 MB', cache_dir=None, log=print):
    '''(Layout, dict of numpy arrays) for the truth-matched signal candidates
    (gen_cand 0..3) of `path`. Chunked, with a progress line per chunk; the
    selected arrays are cached in cache_dir (keyed on file, size, mtime,
    branches) so a re-run with other plotting options does not read the file
    again.'''
    import uproot
    t0 = time.time()
    with uproot.open(path) as f:
        t = f[tree]
        layout = Layout(t.keys())
        branches = layout.branches()
        n_tot = t.num_entries if max_entries is None else min(t.num_entries, max_entries)
    log('file %s: %s entries; reading %d branches' % (path, n_tot, len(branches)))
    log(layout.summary())

    key = _fingerprint(path, tree, branches, max_entries)
    cache = os.path.join(cache_dir, 'cache_%s.npz' % key) if cache_dir else None
    if cache and os.path.isfile(cache):
        log('reading the cached selection %s' % cache)
        with np.load(cache) as z:
            return layout, dict((k, z[k]) for k in z.files)

    parts = dict((b, []) for b in branches)
    n_done = n_sel = 0
    for arr in uproot.iterate('%s:%s' % (path, tree), expressions=branches, step_size=step,
                              library='np', entry_stop=max_entries):
        n = len(arr['gen_cand'])
        m = (arr['gen_cand'] >= 0) & (arr['gen_cand'] <= 3)
        for b in branches:
            parts[b].append(arr[b][m])
        n_done += n
        n_sel += int(m.sum())
        el = time.time() - t0
        rate = n_done / el if el > 0 else 0.
        eta = (n_tot - n_done) / rate if rate > 0 else 0.
        log('  read %d / %d entries (%.0f%%), %d signal candidates kept, %.0f entries/s, ETA %.0f s'
            % (n_done, n_tot, 100. * n_done / max(n_tot, 1), n_sel, rate, eta))
    data = dict((b, np.concatenate(v) if v else np.zeros(0)) for b, v in parts.items())
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, **data)
        log('cached the selection in %s' % cache)
    return layout, data


##########################################################################################
#####      SELECTION
##########################################################################################

def select(layout, d, trigger=True, species=True, one_per_event=True, log=print):
    '''Boolean mask of the candidates entering the comparison, and the cutflow:
      signal       gen_cand 0..3 (already applied in read)
      species      the kaons matched to kaons, the pion to a pion, the muon to
                   a muon (gen_cand matches by charge and dR only, so K and pi
                   swapped on the same three tracks also count as signal)
      trigger      OR of the parking paths fired and the muon matched to a
                   parking trigger object (mu_trig_match)
      one/event    one candidate per event: the lowest cand_idx
    '''
    n = len(d['gen_cand'])
    m = np.ones(n, dtype=bool)
    flow = [('truth-matched signal', m.copy())]
    if species:
        for o, pdg in (('k1', 321), ('k2', 321), ('pi', 211), ('mu', 13)):
            b = '%s_gen_pdgid' % o
            if b in d:
                m &= np.abs(d[b]) == pdg
        flow.append(('correct species', m.copy()))
    if trigger:
        if layout.paths:
            fired = np.zeros(n, dtype=bool)
            for p in layout.paths:
                fired |= d[p] > 0.5
            m &= fired
        if 'mu_trig_match' in d:
            m &= d['mu_trig_match'] > 0.5
        flow.append(('trigger (path OR + muon match)', m.copy()))
    if one_per_event and all(b in d for b in ('run', 'lumi', 'event', 'cand_idx')):
        idx = np.nonzero(m)[0]
        key = np.stack([d['run'][idx], d['lumi'][idx], d['event'][idx]], axis=1)
        order = np.lexsort((d['cand_idx'][idx], key[:, 2], key[:, 1], key[:, 0]))
        ks = key[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = np.any(ks[1:] != ks[:-1], axis=1)
        keep = np.zeros(n, dtype=bool)
        keep[idx[order[first]]] = True
        m &= keep
        flow.append(('one candidate per event', m.copy()))
    cls = d['gen_cand'].astype(int)
    rows = []
    for name, mm in flow:
        rows.append((name, [int(np.sum(mm & (cls == c))) for c in CLASSES]))
    for name, counts in rows:
        log('  %-34s %s' % (name, '  '.join('%s %d' % (CLASS_NAME[c], k) for c, k in zip(CLASSES, counts))))
    return m, rows


def common_mask(layout, d):
    '''Candidates for which every scheme has a usable Bs vertex: the per-scheme
    comparisons are made on the same events.'''
    m = np.ones(len(d['gen_cand']), dtype=bool)
    for s in layout.schemes:
        m &= d['%s_bs_good' % s] > 0.5
    return m


def class_weights(cls, r_target=(0.299, 0.244), r_mc=(1.0, 1.0)):
    '''Per-event weights moving the tau/mu ratio of the MC to r_target
    (R(Ds), R(Ds*)); the mu channels keep weight 1. With the sample made with
    R(Ds) = R(Ds*) = 1 (AN Sec. 4.2.1), r_mc = (1, 1).'''
    w = np.ones(len(cls))
    w[cls == 2] = r_target[0] / r_mc[0]
    w[cls == 3] = r_target[1] / r_mc[1]
    return w


##########################################################################################
#####      DERIVED QUANTITIES
##########################################################################################

def _p4(pt, eta, phi, m):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return np.stack([np.sqrt(px * px + py * py + pz * pz + m * m), px, py, pz])


def _m2(p):
    return p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2


def add_derived(layout, d):
    '''Adds, in place:
      gen targets      gen_q2_ds = (Bs - Ds)^2, what the reconstruction measures
                       (Ds, not Ds*); gen_q2_w = (Bs - Ds(*))^2 = gen_q2;
                       gen_m_miss2_dsx = (Bs - Ds(*) - mu)^2, the targets of the
                       Ds + gamma hypotheses; gen_pt_miss_sca; gen_bs_p;
                       gen_ds_pt
      neutrino-solution choices, per scheme, as hypotheses with all the
      observables (<s>_math_<x>_<q>):
          math_avg     the average of the two (circular for chi)
          math_oracle  the one closer to the true |p(Bs)|: needs the truth,
                       a reference only
    '''
    n = len(d['gen_cand'])
    get = lambda b: d[b] if b in d else np.full(n, NAN)

    bs = _p4(get('gen_bs_pt'), get('gen_bs_eta'), get('gen_bs_phi'), get('gen_bs_mass'))
    dsx = _p4(get('gen_ham_dsx_pt'), get('gen_ham_dsx_eta'), get('gen_ham_dsx_phi'), get('gen_ham_dsx_mass'))
    ds_st = _p4(get('gen_ham_ds_pt'), get('gen_ham_ds_eta'), get('gen_ham_ds_phi'), get('gen_ham_ds_mass'))
    # the Ds: the Ds of Ds* -> Ds gamma when there is one, else the direct Ds
    has_st = np.isfinite(ds_st[0])
    ds = np.where(has_st[None, :], ds_st, dsx)
    mu = _p4(get('gen_vis_mu_pt'), get('gen_vis_mu_eta'), get('gen_vis_mu_phi'), np.full(n, M_MU))

    d['gen_q2_ds'] = _m2(bs - ds)
    d['gen_q2_w'] = _m2(bs - dsx)
    d['gen_m_miss2_dsx'] = _m2(bs - dsx - mu)
    d['gen_pt_miss_sca'] = np.hypot(bs[1], bs[2]) - np.hypot(ds[1], ds[2]) - np.hypot(mu[1], mu[2])
    d['gen_bs_p'] = np.sqrt(bs[1] ** 2 + bs[2] ** 2 + bs[3] ** 2)
    d['gen_ds_pt'] = np.hypot(ds[1], ds[2])

    for s in layout.schemes:
        if 'math_avg' not in layout.hyps[s]:
            continue
        p1 = get('%s_math1_bs_pt' % s) * np.cosh(get('%s_math1_bs_eta' % s))
        p2 = get('%s_math2_bs_pt' % s) * np.cosh(get('%s_math2_bs_eta' % s))
        low = p1 <= p2
        orc = np.abs(p1 - d['gen_bs_p']) <= np.abs(p2 - d['gen_bs_p'])
        d['%s_math_choice_low_is_1' % s] = low.astype(float)
        d['%s_math_oracle_is_1' % s] = orc.astype(float)
        for q in layout.hyp_obs(s, 'math_avg'):
            a, b = get('%s_math1_bs_%s' % (s, q)), get('%s_math2_bs_%s' % (s, q))
            d['%s_math_oracle_%s' % (s, q)] = np.where(orc, a, b)
            if q in ANGLE_OBS:
                d['%s_math_avg_%s' % (s, q)] = np.arctan2(np.sin(a) + np.sin(b), np.cos(a) + np.cos(b))
            else:
                d['%s_math_avg_%s' % (s, q)] = 0.5 * (a + b)
    return d


# reco observable q of a Bs hypothesis -> (gen branch, kind of residual)
#   'diff' reco - gen, 'rel' reco / gen - 1, 'angle' wrapped difference
TARGETS = {
    'q2'              : ('gen_q2_ds', 'diff'),
    'm_miss2'         : ('gen_m_miss2', 'diff'),
    'pt_miss'         : ('gen_pt_miss', 'diff'),
    'pt_miss_sca'     : ('gen_pt_miss_sca', 'diff'),
    'e_star_mu'       : ('gen_e_star_mu', 'diff'),
    'e_hash_mu'       : ('gen_e_hash_mu', 'diff'),
    'cos_theta_phi_ds': ('gen_cos_theta_phi_ds', 'diff'),
    'cos_theta_mu_w'  : ('gen_cos_theta_mu_w', 'diff'),
    'chi'             : ('gen_chi', 'angle'),
    'pt'              : ('gen_bs_pt', 'rel'),
    'beta'            : ('gen_bs_beta', 'diff'),
}
# with the photon (Ds + gamma in place of the Ds): what they estimate
GAMMA_TARGETS = {
    'q2'     : ('gen_q2_w', 'diff'),
    'm_miss2': ('gen_m_miss2_dsx', 'diff'),
}
UNITS = {
    'q2': 'GeV$^2$', 'm_miss2': 'GeV$^2$', 'pt_miss': 'GeV', 'pt_miss_sca': 'GeV',
    'e_star_mu': 'GeV', 'e_hash_mu': 'GeV', 'cos_theta_phi_ds': '', 'cos_theta_mu_w': '',
    'chi': 'rad', 'pt': '(relative)', 'beta': '',
}
OBS_LABEL = {
    'q2': r'$q^2$', 'm_miss2': r'$m^2_{miss}$', 'pt_miss': r'$p_T^{miss}$',
    'pt_miss_sca': r'$p_T^{miss}$ (scalar)', 'e_star_mu': r'$E^*_\mu$', 'e_hash_mu': r'$E^\#_\mu$',
    'cos_theta_phi_ds': r'$\cos\theta_{\phi}$', 'cos_theta_mu_w': r'$\cos\theta_{\mu}$',
    'chi': r'$\chi$', 'pt': r'$p_T(B_s)$', 'beta': r'$\beta(B_s)$',
}


def residual(reco, gen, kind):
    if kind == 'rel':
        with np.errstate(divide='ignore', invalid='ignore'):
            return reco / gen - 1.
    if kind == 'angle':
        return (reco - gen + np.pi) % (2 * np.pi) - np.pi
    return reco - gen
