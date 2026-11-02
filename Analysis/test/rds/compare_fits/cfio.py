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

def _stat(path):
    try:
        s = os.stat(path)
        return (s.st_size, int(s.st_mtime))
    except OSError:
        return None


def _fingerprint(paths, tree, branches, extra):
    if len(paths) == 1:     # the key of the single-file versions, so old caches stay valid
        key = [paths[0], tree, _stat(paths[0]), sorted(branches), extra]
    else:
        key = [[(p, _stat(p)) for p in paths], tree, sorted(branches), extra]
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:16]


def input_files(spec, tree='tree', log=print):
    '''The ROOT files of `spec`: a file or URL, a directory (its *.root),
    a glob, or a comma-separated list of any of these. With more than one
    file, files that do not open or lack `tree` (e.g. still being copied)
    are skipped.'''
    import glob
    out = []
    for part in [x for x in spec.split(',') if x]:
        if os.path.isdir(part):
            out += sorted(glob.glob(os.path.join(part, '*.root')))
        elif any(c in part for c in '*?['):
            out += sorted(glob.glob(part))
        else:
            out.append(part)
    if len(out) <= 1:
        return out
    import uproot
    good = []
    for f in out:
        try:
            with uproot.open(f) as r:
                r[tree].num_entries
            good.append(f)
        except Exception as e:
            log('skipping %s: %s' % (f, str(e).splitlines()[0] if str(e) else type(e).__name__))
    return good


def _rss_gb():
    '''Resident memory of this process [GB] (Linux; NaN elsewhere).'''
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024. ** 2
    except OSError:
        pass
    return float('nan')


def _keep_dtype(name, a):
    '''float64 -> float32 (halves the memory), except covariances, which are
    inverted and need the precision.'''
    if a.dtype == np.float64 and '_cov_' not in name:
        return a.astype(np.float32)
    return a


_FILES = {}


def _read_block(job):
    '''One block of entries: the truth-matched signal candidates (gen_cand
    0..3) of entries [start, stop). Top-level so that worker processes can
    run it; each process opens the file once.'''
    path, tree, branches, start, stop = job
    import uproot
    if (path, tree) not in _FILES:
        _FILES[(path, tree)] = uproot.open(path)[tree]
    t = _FILES[(path, tree)]
    arr = t.arrays(branches, entry_start=start, entry_stop=stop, library='np')
    gc = arr['gen_cand']
    m = (gc >= 0) & (gc <= 3)
    return stop - start, dict((b, _keep_dtype(b, arr[b][m])) for b in branches)


def read(path, tree='tree', max_entries=None, max_candidates=1000000, block=100000, jobs=4, seed=1,
         cache_dir=None, log=print):
    '''(Layout, dict of numpy arrays, info) for the truth-matched signal
    candidates (gen_cand 0..3) of `path` (a file, directory, glob or
    comma-separated list, see input_files; blocks are drawn from all files).

    The file is read in blocks of `block` entries taken in random order, so
    that any part of it is an unbiased sample of the whole, and the reading
    stops once `max_candidates` candidates are kept (0: read everything).
    `jobs` processes read blocks in parallel. Floats are kept as float32.
    The result is cached in cache_dir (keyed on file, size, mtime, branches
    and the options above), so a re-run does not read the file again.'''
    import uproot
    t0 = time.time()
    paths = input_files(path, tree, log=log)
    if not paths:
        raise IOError('no readable ROOT file in %s' % path)
    n_file = []
    for i, p in enumerate(paths):
        with uproot.open(p) as f:
            t = f[tree]
            if i == 0:
                layout = Layout(t.keys())
                branches = layout.branches()
            n_file.append(t.num_entries)
    n_tot = sum(n_file) if max_entries is None else min(sum(n_file), max_entries)
    log('%s: %s entries; reading %d branches'
        % ('file %s' % paths[0] if len(paths) == 1 else '%d files in %s' % (len(paths), path), n_tot, len(branches)))
    log(layout.summary())

    key = _fingerprint(paths, tree, branches, [max_entries, max_candidates, block, seed])
    cache = os.path.join(cache_dir, 'cache_%s.npz' % key) if cache_dir else None
    if cache and os.path.isfile(cache):
        log('reading the cached selection %s' % cache)
        with np.load(cache) as z:
            data = dict((k, z[k]) for k in z.files if k != '__info__')
            info = json.loads(str(z['__info__'])) if '__info__' in z.files else {}
        return layout, data, info

    allb, left = [], n_tot              # blocks never cross a file boundary
    for p, n in zip(paths, n_file):
        n = min(n, left)
        left -= n
        allb += [(p, tree, branches, int(a), int(min(a + block, n))) for a in range(0, n, block)]
    order = np.random.default_rng(seed).permutation(len(allb))
    blocks = [allb[i] for i in order]
    target = max_candidates if max_candidates and max_candidates > 0 else None
    log('%d blocks of %d entries, random order; %s; %d parallel reader(s)'
        % (len(blocks), block, 'stop at %d signal candidates' % target if target else 'whole file', jobs))

    parts = dict((b, []) for b in branches)
    st = dict(n_done=0, n_sel=0, n_blocks=0)

    def add(res):
        n, arr = res
        for b in branches:
            parts[b].append(arr[b])
        st['n_done'] += n
        st['n_sel'] += len(arr['gen_cand'])
        st['n_blocks'] += 1
        el = time.time() - t0
        rate = st['n_done'] / el if el > 0 else 0.
        # remaining time: to the target if there is one, else to the end
        if target and st['n_sel'] > 0:
            todo = min(n_tot, st['n_done'] * target / float(st['n_sel'])) - st['n_done']
        else:
            todo = n_tot - st['n_done']
        log('  block %d/%d: %d entries read (%.1f%% of the file), %d signal candidates kept, '
            '%.0f entries/s, ETA %.0f s, memory %.1f GB'
            % (st['n_blocks'], len(blocks), st['n_done'], 100. * st['n_done'] / max(n_tot, 1), st['n_sel'],
               rate, max(todo, 0) / rate if rate > 0 else 0., _rss_gb()))

    done = lambda: target is not None and st['n_sel'] >= target
    if jobs <= 1:
        for job in blocks:
            add(_read_block(job))
            if done():
                break
    else:
        import concurrent.futures as cf
        it = iter(blocks)
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            pending = set()
            for job in it:                         # keep 2 blocks per process in flight
                pending.add(ex.submit(_read_block, job))
                if len(pending) >= 2 * jobs:
                    break
            while pending:
                fin, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for fu in fin:
                    add(fu.result())
                if done():
                    for fu in pending:
                        fu.cancel()
                    break
                for job in it:
                    pending.add(ex.submit(_read_block, job))
                    if len(pending) >= 2 * jobs:
                        break
    data = dict((b, np.concatenate(v) if v else np.zeros(0)) for b, v in parts.items())
    del parts
    info = dict(files=len(paths), entries_total=int(n_tot), entries_read=st['n_done'],
                fraction_read=st['n_done'] / float(max(n_tot, 1)), candidates_kept=st['n_sel'],
                blocks_read=st['n_blocks'], block=block, seed=seed)
    log('read %.1f%% of the file in %.0f s: %d signal candidates, %.1f GB in memory'
        % (100 * info['fraction_read'], time.time() - t0, st['n_sel'], _rss_gb()))
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, __info__=np.array(json.dumps(info)), **data)
        log('cached the selection in %s' % cache)
    return layout, data, info


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
    pt, eta, phi, m = (np.asarray(x, dtype=np.float64) for x in (pt, eta, phi, m))
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
