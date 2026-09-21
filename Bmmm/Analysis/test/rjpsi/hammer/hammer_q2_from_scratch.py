#!/usr/bin/env python3
r'''
dGamma/dq^2 for Bc -> J/psi l nu from HAMMER ALONE. No MC sample, no ntuple.

The trick is Hammer's pure phase space declaration (v2 manual sec. N): a vertex
declared pure-PS is evaluated as |M|^2 = 1 x m^(6-2n). Declare the Bc vertex
pure-PS in the DENOMINATOR and the weight stops being a ratio:

    w  =  |M_Harrison|^2 / (1 x m^(6-2n))   proportional to   |M_Harrison|^2

So: generate flat phase-space kinematics locally, ask Hammer for w, and bin

    dGamma/dq^2  proportional to  sum over events in the q2 bin of  J_PS * w

where J_PS is the phase-space Jacobian |p_Jpsi| * |p_lepton*| -- pure kinematics,
not a matrix element. Every piece of dynamics comes from Hammer's C++.

This is the validation the MC route cannot give you when the sample is
gen-filtered: there is no acceptance here because there is no sample.

    python3 hammer_q2_from_scratch.py -o plots
    python3 hammer_q2_from_scratch.py -o plots --check-generator
    python3 hammer_q2_from_scratch.py -o plots --boost 0.9

Three built-in controls:
  --check-generator   also declares the numerator pure-PS, so w is constant and
                      the binned spectrum MUST follow the analytic phase-space
                      curve |p_Jpsi| * |p_lepton*|. Tests the generator and the
                      Jacobian without involving the form factors at all.
  --boost             boosts every event along x before handing it to Hammer.
                      The result must not change: a frame dependence would mean
                      the four-momenta are being misread.
  R(J/psi)            the ratio of the integrated spectra, against 0.2597(27).

The overlay is ff_rate.dgamma_dq2 with the same FF card -- the analytic curve
already validated against Fig. 6 of arXiv:2503.15090. Agreement closes the last
link in the chain: Hammer's own evaluator against our port of it.
'''
import argparse
import math
import os
import time

import numpy as np

from Bmmm.Analysis.HammerFF import (
    DEFAULT_CARD, FF_TARGET, JPSI_PDGID, MU_CODE, PROCESS, SCHEME, TAU_CODE,
    XTOY, HammerSession, load_card,
)

import ff_rate as FR
import hammer_bgl_forward as HB

MBC, MJPSI = HB.MBC, HB.MJPSI
BC_PDGID = 541

CHANNELS = (
    # name, vertex, lepton pdgid, neutrino pdgid, lepton mass, label
    ('mu', 'BcJpsiMuNu', -13, 14, FR.M_MU, r'$B_c\to J/\psi\,\mu\,\nu$'),
    ('tau', 'BcJpsiTauNu', -15, 16, FR.M_TAU, r'$B_c\to J/\psi\,\tau\,\nu$'),
)


# ---------------------------------------------------------------------------
# flat phase space for Bc -> J/psi l nu, in the Bc rest frame
# ---------------------------------------------------------------------------
def two_body_momentum(M, m1, m2):
    lam = (M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2)
    return math.sqrt(max(lam, 0.)) / (2. * M)


def ps_jacobian(q2, mlep):
    """d(Phi_3) / (dq2 dOmega_V dOmega_l*) up to constants.

    Bc -> J/psi W*(q2) followed by W* -> l nu, so the measure factorises into
    two 2-body pieces,

        dPhi_3  ~  [|p_V| / M_Bc] dOmega_V  x  [|p_l*| / sqrt(q2)] dOmega_l  x  dq2

    Kinematics only; no dynamics enters here.

    The 1/sqrt(q2) is NOT optional and was missing in the first version of this
    script: dropping it overweights high q2 by exactly sqrt(q2), which showed up
    as a Hammer/analytic ratio climbing from 0.17 to 1.4 across the muon range.
    Cross-checked against dalitz_band_width() below, which derives the same
    measure a completely different way; the two agree to 1e-9.
    """
    if q2 <= mlep * mlep:
        return 0.
    p_v = two_body_momentum(MBC, MJPSI, math.sqrt(q2))
    p_l = (q2 - mlep * mlep) / (2. * math.sqrt(q2))
    return p_v * p_l / math.sqrt(q2)


def dalitz_band_width(q2, mlep):
    """dPhi_3/dq2, derived independently of ps_jacobian.

    Three-body phase space is flat in the Dalitz plane, so dPhi_3/dq2 is simply
    the width of the m^2(J/psi, l) band at fixed q2 = m^2(l, nu):
    (m^2_12)max - (m^2_12)min = 4 p1* p2*, with both momenta evaluated in the
    (l nu) rest frame (PDG kinematics review).

    This exists so the generator control is not circular. The first version
    validated the generator against ps_jacobian itself -- the function under
    test -- so it passed with chi2/ndf = 0.95 while the Jacobian was wrong by a
    factor sqrt(q2). A control that reuses what it is testing proves nothing.
    """
    if q2 <= mlep * mlep:
        return 0.
    m23 = math.sqrt(q2)
    e1 = (MBC * MBC - q2 - MJPSI * MJPSI) / (2. * m23)
    e2 = (q2 + mlep * mlep) / (2. * m23)
    p1 = math.sqrt(max(e1 * e1 - MJPSI * MJPSI, 0.))
    p2 = math.sqrt(max(e2 * e2 - mlep * mlep, 0.))
    return 4. * p1 * p2


def check_jacobian(mlep, npoints=200, tol=1e-6):
    """Assert the two derivations of dPhi_3/dq2 agree up to one constant."""
    lo, hi = mlep * mlep * 1.0001, (MBC - MJPSI) ** 2 - 1e-6
    qs = np.linspace(lo, hi, npoints)
    a = np.array([ps_jacobian(q2, mlep) for q2 in qs])
    b = np.array([dalitz_band_width(q2, mlep) for q2 in qs])
    ratio = a / np.maximum(b, 1e-300)
    spread = (ratio.max() - ratio.min()) / max(abs(ratio.mean()), 1e-300)
    if spread > tol:
        raise RuntimeError(
            'phase-space Jacobian disagrees with the Dalitz-plane derivation: '
            'ratio spans %.6g .. %.6g (spread %.2e). One of the two is wrong.'
            % (ratio.min(), ratio.max(), spread))
    return spread


def boost(p, beta, axis=0):
    '''Boost a four-vector [E, px, py, pz] with velocity beta along `axis`.'''
    if beta == 0.:
        return p
    g = 1. / math.sqrt(1. - beta * beta)
    out = np.array(p, dtype=float)
    e, pi = out[0], out[1 + axis]
    out[0] = g * (e + beta * pi)
    out[1 + axis] = g * (pi + beta * e)
    return out


def generate_event(rng, q2, mlep):
    '''Four-momenta of J/psi, lepton, neutrino in the Bc rest frame.

    The J/psi is put along +z: the overall orientation integrates out, so fixing
    it costs nothing and removes a source of variance. The lepton angles in the
    W* rest frame are sampled uniformly.
    '''
    sq = math.sqrt(q2)
    p_v = two_body_momentum(MBC, MJPSI, sq)
    e_v = math.sqrt(p_v * p_v + MJPSI * MJPSI)
    jpsi = np.array([e_v, 0., 0., p_v])
    w = np.array([MBC - e_v, 0., 0., -p_v])

    p_l = (q2 - mlep * mlep) / (2. * sq)
    ct = rng.uniform(-1., 1.)
    st = math.sqrt(max(1. - ct * ct, 0.))
    ph = rng.uniform(0., 2. * math.pi)
    lep_w = np.array([math.sqrt(p_l * p_l + mlep * mlep),
                      p_l * st * math.cos(ph), p_l * st * math.sin(ph), p_l * ct])
    nu_w = np.array([p_l, -lep_w[1], -lep_w[2], -lep_w[3]])

    # boost from the W* rest frame back to the Bc rest frame: W* moves along -z
    beta_w = -p_v / w[0]
    lep = boost(lep_w, beta_w, axis=2)
    nu = boost(nu_w, beta_w, axis=2)
    return jpsi, lep, nu


# ---------------------------------------------------------------------------
def build_process(H, bc, jpsi, lep, nu, lep_pdg, nu_pdg):
    proc = H.Process()
    i_bc = proc.add_particle(H.Particle(H.FourMomentum(*bc), BC_PDGID))
    i_j = proc.add_particle(H.Particle(H.FourMomentum(*jpsi), JPSI_PDGID))
    i_l = proc.add_particle(H.Particle(H.FourMomentum(*lep), lep_pdg))
    i_n = proc.add_particle(H.Particle(H.FourMomentum(*nu), nu_pdg))
    proc.add_vertex(i_bc, [i_j, i_l, i_n])
    return proc


# ---------------------------------------------------------------------------
# parallel execution
# ---------------------------------------------------------------------------
# Hammer's init_run is per-process global state, so each worker builds its own
# session. The parent deliberately builds none when --jobs > 1: forking a live
# session would duplicate that state.
_SESSION = None
_SESSION_KEY = None


def get_session(card, check_generator, verbose=False, tag='', ff_target='BGL',
                scale=None, apply_card=True, allow_legacy_p1=False):
    global _SESSION, _SESSION_KEY
    key = (card, bool(check_generator), ff_target, scale, apply_card,
           allow_legacy_p1)
    if _SESSION is None or _SESSION_KEY != key:
        # init_run builds the amplitude and form-factor tensors and is the fixed
        # cost of a worker. Time it and say so: with many workers and few events
        # each, this is the whole run time, and silence here looks like a hang.
        t0 = time.time()
        vertices = set(ch[1] for ch in CHANNELS)
        _SESSION = HammerSession(
            card_path=None if card in ('', 'default') else card.split(':')[0],
            variations=False, allow_stale=True, verbose=verbose,
            decay_chains=[[ch[1]] for ch in CHANNELS],
            pure_ps_denominator=vertices,
            pure_ps_numerator=vertices if check_generator else None,
            # no eigenvariations here, so the plain class is enough: 15 fewer
            # tensor dimensions everywhere, including in the rate integral. (Do
            # NOT try general_rates=False to skip that integral: Hammer refuses
            # it without a WC specialization and integrates anyway, just
            # noisily.)
            ff_target=ff_target, scale_coefficients=scale,
            apply_card=apply_card, allow_legacy_p1=allow_legacy_p1)
        _SESSION_KEY = key
        print('    %s Hammer session built in %.1f s (pid %d)'
              % (tag or '[init]', time.time() - t0, os.getpid()), flush=True)
    return _SESSION


def run_chunk(job):
    """One worker's slice: (channel index, nevents, seed, bins, beta, card, check)."""
    idx, nevents, seed, bins, beta, card, check, tag, ff_target, scale, \
        apply_card, cbins, legacy = job
    name, vertex, lep_pdg, nu_pdg, mlep, _ = CHANNELS[idx]
    # one thread per worker: 64 processes each spinning up BLAS threads will
    # thrash a node long before Hammer becomes the bottleneck
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(var, '1')
    ses = get_session(card, check, tag=tag, ff_target=ff_target, scale=scale,
                      apply_card=apply_card, allow_legacy_p1=legacy)
    rng = np.random.default_rng(seed)
    return spectrum(ses, name, vertex, lep_pdg, nu_pdg, mlep, nevents, bins,
                    beta, rng, progress=0, raw=True, cbins=cbins)


def spectrum(ses, name, vertex, lep_pdg, nu_pdg, mlep, nevents, bins, beta, rng,
             progress=0, raw=False, cbins=20):
    """Binned dGamma/dq^2 AND dGamma/dcos(theta_W) from Hammer weights over flat
    phase space. Both histograms carry the same weight J_PS * w per event."""
    import hammer as H

    lo, hi = mlep * mlep, (MBC - MJPSI) ** 2
    acc = {'sumw': np.zeros(bins), 'sumw2': np.zeros(bins),
           'csum': np.zeros(cbins), 'csum2': np.zeros(cbins),
           'nd': 0, 'nb': 0}

    # The eigenvector coordinates are STICKY: setting them once outside the loop
    # is enough. Calling set_ff_eigenvectors per event forces Hammer to rebuild
    # the variation tensor every time. With ff_target='BGL' there are no
    # variational indices at all and the call is skipped entirely.
    if ses.has_variations:
        ses.ham.set_ff_eigenvectors(XTOY, ses.ff_target, [0.] * 15)

    t0 = time.time()
    for iev in range(nevents):
        if progress and iev and iev % progress == 0:
            rate = iev / max(time.time() - t0, 1e-9)
            print('    [%s] %d/%d  %.0f ev/s  eta %.0f s'
                  % (name, iev, nevents, rate, (nevents - iev) / max(rate, 1e-9)),
                  flush=True)
        q2 = rng.uniform(lo, hi)
        jac = ps_jacobian(q2, mlep)
        if jac <= 0.:
            continue
        jpsi, lep, nu = generate_event(rng, q2, mlep)
        bc = np.array([MBC, 0., 0., 0.])
        if beta:
            bc, jpsi, lep, nu = (boost(x, beta) for x in (bc, jpsi, lep, nu))

        ses.ham.init_event()
        proc = build_process(H, bc, jpsi, lep, nu, lep_pdg, nu_pdg)
        if ses.ham.add_process(proc) == 0:
            acc['nd'] += 1
            continue
        ses.ham.process_event()
        w = ses.ham.get_weight(SCHEME)
        if not np.isfinite(w):
            acc['nb'] += 1
            continue

        ww = jac * w
        k = min(int((q2 - lo) / (hi - lo) * bins), bins - 1)
        acc['sumw'][k] += ww
        acc['sumw2'][k] += ww * ww
        # the angle is measured from the (possibly boosted) four-vectors with
        # the same function used on the EvtGen side: no convention is assumed
        c = FR.cos_theta_w(bc, jpsi, lep)
        kc = min(int((c + 1.) / 2. * cbins), cbins - 1)
        acc['csum'][kc] += ww
        acc['csum2'][kc] += ww * ww

    acc['dt'] = time.time() - t0
    if raw:
        return acc
    return finalise(acc, name, mlep, bins, cbins, nevents, jobs=1)


def finalise(acc, name, mlep, bins, cbins, nevents, jobs):
    """Turn accumulated sums into normalised densities, errors and integrals."""
    lo, hi = mlep * mlep, (MBC - MJPSI) ** 2
    edges = np.linspace(lo, hi, bins + 1)
    width = edges[1] - edges[0]
    norm = max(acc['sumw'].sum() * width, 1e-300)
    cedges = np.linspace(-1., 1., cbins + 1)
    cwidth = cedges[1] - cedges[0]
    cnorm = max(acc['csum'].sum() * cwidth, 1e-300)
    dt = max(acc.get('dt', 0.), 1e-9)
    print('[%-3s] %d events in %.1f s (%.0f ev/s, %d job%s), %d declined by '
          'Hammer, %d non-finite' % (name, nevents, dt, nevents / dt, jobs,
                                     '' if jobs == 1 else 's', acc['nd'], acc['nb']))
    return {
        'edges': edges, 'dens': acc['sumw'] / norm,
        'err': np.sqrt(acc['sumw2']) / norm,
        'integral': acc['sumw'].sum() * width,
        'integral_err': np.sqrt(acc['sumw2'].sum()) * width,
        'cedges': cedges, 'cdens': acc['csum'] / cnorm,
        'cerr': np.sqrt(acc['csum2']) / cnorm,
    }


def spectrum_parallel(idx, nevents, bins, beta, card, check, jobs, seed,
                      ff_target='BGL', scale=None, apply_card=True, cbins=20,
                      legacy=False):
    """Same result as spectrum(), split over `jobs` worker processes."""
    import concurrent.futures as cf
    import multiprocessing as mp

    name, _, _, _, mlep, _ = CHANNELS[idx]
    per = [nevents // jobs + (1 if k < nevents % jobs else 0) for k in range(jobs)]
    # Independent streams per worker. The first version used seed + 1000*idx + k,
    # so --seed 1 and --seed 2 shared 7 of their 8 worker streams: nearby seeds
    # gave nearly identical events while looking like independent runs.
    # SeedSequence.spawn guarantees statistically independent streams no matter
    # how close the user's seeds are.
    streams = np.random.SeedSequence([seed, idx]).spawn(len(per))
    work = [(idx, n, streams[k], bins, beta, card, check,
             '[%s w%02d]' % (name, k), ff_target, scale, apply_card, cbins,
             legacy)
            for k, n in enumerate(per) if n]

    if per and per[0] < 2000:
        print('    [%s] NOTE: %d events per worker. Each worker pays a full '
              'Hammer init_run, so beyond a few thousand events per worker more '
              'jobs just buy more start-up. Raise -n or lower -j.'
              % (name, per[0]), flush=True)

    tot = {'sumw': np.zeros(bins), 'sumw2': np.zeros(bins),
           'csum': np.zeros(cbins), 'csum2': np.zeros(cbins), 'nd': 0, 'nb': 0}
    t0 = time.time()
    done = 0
    print('    [%s] launching %d workers x %d events ...'
          % (name, len(work), per[0] if per else 0), flush=True)
    with cf.ProcessPoolExecutor(max_workers=jobs,
                                mp_context=mp.get_context('fork')) as pool:
        futures = [pool.submit(run_chunk, job) for job in work]
        for fut in cf.as_completed(futures):      # report out of order, as they land
            part = fut.result()
            for key in ('sumw', 'sumw2', 'csum', 'csum2', 'nd', 'nb'):
                tot[key] = tot[key] + part[key]
            done += 1
            print('    [%s] chunk %d/%d done, %.0f s elapsed'
                  % (name, done, len(work), time.time() - t0), flush=True)
    tot['dt'] = time.time() - t0
    return finalise(tot, name, mlep, bins, cbins, nevents, jobs)


def reference_histogram(spec, channel, edges, branch=None):
    """Normalised gen-level q2 density from an external sample.

    spec is FILE:TREE:BRANCH[:CODEBRANCH]. The branch is the generated q2 (or
    anything from which it was computed); CODEBRANCH, if present, selects the
    channel with the gen_bc_decay convention (1 = mu, 7 = tau).
    """
    import uproot

    parts = spec.split(':')
    fname, tname, bname = parts[0], parts[1], parts[2]
    if branch:
        bname = branch
    codebranch = parts[3] if len(parts) > 3 else None

    want = {'mu': MU_CODE, 'tau': TAU_CODE}[channel]
    reads = [bname] + ([codebranch] if codebranch else [])
    with uproot.open('%s:%s' % (fname, tname)) as tree:
        arrs = tree.arrays(reads, library='np')
    vals = np.asarray(arrs[bname], dtype=float)
    if codebranch:
        sel = np.asarray(arrs[codebranch], dtype=float)
        vals = vals[np.round(sel) == want]
    if vals.size == 0:
        raise RuntimeError('reference %r has no %s events' % (spec, channel))
    counts, _ = np.histogram(vals, bins=edges)
    width = edges[1] - edges[0]
    norm = max(counts.sum() * width, 1e-300)
    print('    [%s] reference %s: %d events from %s'
          % (channel, bname, vals.size, fname))
    # the reference has its own MC error; without it the chi2 is meaningless
    return counts / norm, np.sqrt(counts) / norm


def _compare(ax, axr, centres, dens, err, ref_of, edges, ref_err, xlabel,
             ylabel, ref_label, title, data_label):
    """Overlay one Hammer histogram on a reference curve; return (chi2, ndf)."""
    grid = np.linspace(edges[0], edges[-1], 400)
    ref = ref_of(grid)
    ref_norm = max(np.trapezoid(ref, grid), 1e-300)
    # average the reference OVER each bin, not at its centre: curvature alone
    # otherwise biases the chi2 badly
    ref_binned = np.array([
        np.trapezoid(ref_of(np.linspace(a, b, 21)), np.linspace(a, b, 21))
        / (b - a) for a, b in zip(edges[:-1], edges[1:])]) / ref_norm

    ax.errorbar(centres, dens, yerr=err, fmt='o', ms=3.5, color='C0',
                label=data_label)
    ax.plot(grid, ref / ref_norm, color='C3', lw=1.6, ls='--', label=ref_label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if data_label or ref_label:          # the convention-scoring dummy has none
        ax.legend(fontsize=8)

    good = ref_binned > 0
    tot_err = err if ref_err is None else np.sqrt(err ** 2 + ref_err ** 2)
    axr.axhline(1., color='k', lw=0.8)
    axr.errorbar(centres[good], (dens / ref_binned)[good],
                 yerr=(tot_err / ref_binned)[good], fmt='o', ms=3.5, color='C0')
    axr.set_ylabel('Hammer / reference')
    axr.set_xlabel(xlabel)
    axr.grid(alpha=0.25)
    pull = ((dens - ref_binned) / np.maximum(tot_err, 1e-300))[good]
    return float(np.sum(pull ** 2)), int(pull.size)


def afb_from_hist(cedges, cdens, cerr=None):
    """(forward - backward)/(forward + backward) from a histogram, with its MC
    error when per-bin errors are given. Bin edges fall on 0, so F and B are
    exact sums. Normalisation cancels in the ratio."""
    ctr = 0.5 * (cedges[:-1] + cedges[1:])
    w = cedges[1] - cedges[0]
    fwd, bwd = ctr > 0, ctr < 0
    f = np.sum(cdens[fwd]) * w
    b = np.sum(cdens[bwd]) * w
    s = max(f + b, 1e-300)
    afb = (f - b) / s
    if cerr is None:
        return afb
    sf = np.sqrt(np.sum((cerr[fwd] * w) ** 2))
    sb = np.sqrt(np.sum((cerr[bwd] * w) ** 2))
    err = np.sqrt((2. * b / s ** 2 * sf) ** 2 + (2. * f / s ** 2 * sb) ** 2)
    return afb, err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--plots-dir', default='hammer_from_scratch')
    ap.add_argument('--card', default='default',
                    help="FF card path, 'default', or 'hammer-default': the "
                         "latter leaves Hammer on its OWN built-in BGL "
                         "coefficients and evaluates the analytic side with the "
                         "same defaults -- the C++ evaluator against the Python "
                         "port with OUR FIT ENTIRELY OUT OF THE LOOP")
    ap.add_argument('-n', '--events', type=int, default=50000,
                    help='per channel. 50k already gives ~1%% per bin at 40 bins; '
                         'the statistical error scales as 1/sqrt(N) and Hammer '
                         'is the bottleneck, so raise it only if you need it')
    ap.add_argument('--progress', type=int, default=10000,
                    help='print a rate and ETA every N events (0 to silence)')
    ap.add_argument('--bins', type=int, default=40)
    ap.add_argument('--cbins', type=int, default=20,
                    help='bins in cos(theta_W)')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--boost', type=float, default=0.,
                    help='boost every event along x by this beta; results must '
                         'be unchanged')
    ap.add_argument('-j', '--jobs', type=int, default=1,
                    help='worker processes; one Hammer session per worker')
    ap.add_argument('--ff-target', default='BGL',
                    help="the Hammer FF class in the NUMERATOR. 'BGL' (default, "
                         "fast) or 'BGLVar' (what production uses) -- TEST 4 is "
                         "that the two give identical central spectra. "
                         "'Kiselev' instead produces Hammer's own Kiselev "
                         "spectrum with the coefficient card ignored: TEST 3, "
                         "to be compared with --reference against an unfiltered "
                         "EvtGen sample of the same DEC file.")
    ap.add_argument('--reference', default=None,
                    metavar='FILE:TREE:BRANCH[:CODEBRANCH]',
                    help='TEST 3: overlay gen-level distributions from an external '
                         '(EvtGen) sample. BRANCH is the q2 branch; the angle is '
                         'read from --reference-angle in the same tree.')
    ap.add_argument('--reference-angle', default='cos_theta_w',
                    help='angle branch in the --reference tree')
    ap.add_argument('--scale-coeff', default=None, metavar='NAME:FACTOR',
                    help="TEST 1 / H_t diagnostic: scale a coefficient block in "
                         "BOTH Hammer and the analytic curve. 'dvec:0' removes "
                         "H_t entirely; 'dvec:2' raises its share of Gamma_tau "
                         "to ~26%%. The printed Hammer-minus-analytic R, over the "
                         "H_t share, localises any discrepancy in H_t.")
    ap.add_argument('--allow-legacy-p1', action='store_true',
                    help='accept a card fitted with the old P1 = F2/2 '
                         'convention, e.g. to reproduce the H_t diagnostic. '
                         'Never for production.')
    ap.add_argument('--check-generator', action='store_true',
                    help='numerator pure-PS too, so w is constant: the spectrum '
                         'must then follow the analytic phase-space curve')
    args = ap.parse_args()

    # cheap, and it would have caught the sqrt(q2) bug before any Hammer call
    for _ch in CHANNELS:
        spread = check_jacobian(_ch[4])
    print('#### phase-space Jacobian agrees with the Dalitz-plane derivation '
          'to %.1e' % spread)

    hammer_default = (args.card == 'hammer-default')
    card_path = DEFAULT_CARD if args.card in ('', 'default', 'hammer-default') \
        else args.card.split(':')[0]

    perturbation = None
    if args.scale_coeff:
        _n, _f = args.scale_coeff.split(':')
        perturbation = (_n, float(_f))
        print('#### %s scaled by %g in BOTH Hammer and the analytic curve'
              % perturbation)

    if hammer_default:
        # the port's copy of Hammer's own default (Harrison-2020) coefficients
        vec = np.concatenate([HB.DEFAULT[k] for k in ('avec', 'bvec', 'cvec', 'dvec')])
        if perturbation:
            _card = dict((k, list(HB.DEFAULT[k])) for k in HB.DEFAULT)
            _card[perturbation[0]] = [x * perturbation[1] for x in _card[perturbation[0]]]
            vec = FR.coeffs_from_card(_card)
            print('#### [WARN] --scale-coeff with hammer-default perturbs the '
                  'ANALYTIC side only: Hammer keeps its built-in values')
        print('#### hammer-default: Hammer on its built-in coefficients, analytic '
              'on the port\'s copy of them -- fit out of the loop')
    elif args.ff_target.startswith('BGL'):
        card = load_card(card_path)
        if perturbation:
            card[perturbation[0]] = [x * perturbation[1] for x in card[perturbation[0]]]
        vec = FR.coeffs_from_card(card)
    else:
        vec = None       # Kiselev etc.: Hammer's own parametrisation, no card

    ses_is_bgl = args.ff_target.startswith('BGL')
    apply_card = not hammer_default
    sess_card = 'default' if hammer_default else args.card

    results = {}
    if args.jobs > 1:
        for idx, _ch in enumerate(CHANNELS):
            results[_ch[0]] = spectrum_parallel(
                idx, args.events, args.bins, args.boost, sess_card,
                args.check_generator, args.jobs, args.seed,
                ff_target=args.ff_target,
                scale=None if hammer_default else perturbation,
                apply_card=apply_card, cbins=args.cbins,
                legacy=args.allow_legacy_p1)
    else:
        ses = get_session(sess_card, args.check_generator, verbose=True,
                          tag='[serial]', ff_target=args.ff_target,
                          scale=None if hammer_default else perturbation,
                          apply_card=apply_card,
                          allow_legacy_p1=args.allow_legacy_p1)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, 999]))
        for _ch in CHANNELS:
            name, vertex, lep_pdg, nu_pdg, mlep, label = _ch
            results[name] = spectrum(ses, name, vertex, lep_pdg, nu_pdg, mlep,
                                     args.events, args.bins, args.boost, rng,
                                     progress=args.progress, cbins=args.cbins)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(args.plots_dir, exist_ok=True)

    have_analytic = ses_is_bgl and not args.reference
    data_label = 'Hammer %s, flat PS + pure-PS denominator' % args.ff_target

    # ------------------------------------------------------------- q2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [2, 1]})
    for col, (name, vertex, lep_pdg, nu_pdg, mlep, label) in enumerate(CHANNELS):
        res = results[name]
        edges = res['edges']
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax, axr = axes[0, col], axes[1, col]
        ref_err = None
        if args.reference:
            ref_label = 'EvtGen gen-level'
            ref_hist, ref_err = reference_histogram(args.reference, name, edges)
            ref_of = (lambda h, c: (lambda x: np.interp(np.atleast_1d(x), c, h)))(
                ref_hist, centres)
        elif args.check_generator:
            ref_label = 'analytic pure PS (independent Dalitz derivation)'
            ref_of = (lambda m: (lambda x: np.array(
                [dalitz_band_width(q, m) for q in np.atleast_1d(x)])))(mlep)
        elif have_analytic:
            ref_label = 'analytic, same coefficients'
            ref_of = (lambda m: (lambda x: FR.dgamma_dq2(x, vec, m)))(mlep)
        else:
            ax.errorbar(centres, res['dens'], yerr=res['err'], fmt='o', ms=3.5,
                        color='C0', label=data_label)
            ax.set_title(label)
            ax.legend(fontsize=8)
            axr.axis('off')
            print('[%-3s] q2: no reference to compare against (use --reference)'
                  % name)
            continue
        chi2, ndf = _compare(ax, axr, centres, res['dens'], res['err'], ref_of,
                             edges, ref_err, r'$q^2$ [GeV$^2$]',
                             r'$(1/\Gamma)\,d\Gamma/dq^2$  [GeV$^{-2}$]',
                             ref_label, label, data_label)
        print('[%-3s] q2      chi2/ndf vs %s = %.1f/%d = %.2f'
              % (name, ref_label, chi2, ndf, chi2 / max(ndf, 1)))
    fig.suptitle('q2 spectra from Hammer alone -- no MC sample'
                 + (' (generator control)' if args.check_generator else ''))
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots_dir, 'hammer_q2_from_scratch.png'), dpi=140)
    plt.close(fig)

    # ------------------------------------------------------- angular figure
    # What dGamma/dq2 cannot see: A_FB carries H-^2 - H+^2 (the relative sign of
    # g and f) and the H0 H_t interference. For the numerator this is the Hammer
    # counterpart of check_angular_observables.py, which only tested the port.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [2, 1]})
    for col, (name, vertex, lep_pdg, nu_pdg, mlep, label) in enumerate(CHANNELS):
        res = results[name]
        cedges = res['cedges']
        cctr = 0.5 * (cedges[:-1] + cedges[1:])
        ax, axr = axes[0, col], axes[1, col]
        afb_h, afb_e = afb_from_hist(cedges, res['cdens'], res['cerr'])
        if args.reference:
            ref_label = 'EvtGen gen-level'
            rh, re_ = reference_histogram(args.reference, name, cedges,
                                          branch=args.reference_angle)
            chi2, ndf = _compare(ax, axr, cctr, res['cdens'], res['cerr'],
                                 (lambda h: (lambda x: np.interp(
                                     np.atleast_1d(x), cctr, h)))(rh),
                                 cedges, re_, r'$\cos\theta_W$',
                                 r'$(1/\Gamma)\,d\Gamma/d\cos\theta_W$',
                                 ref_label, label, data_label)
            afb_r, afb_re = afb_from_hist(cedges, rh, re_)
            print('[%-3s] cosW    chi2/ndf vs %s = %.1f/%d = %.2f   '
                  'A_FB Hammer %+.4f +/- %.4f  EvtGen %+.4f +/- %.4f  (%+.1f sigma)'
                  % (name, ref_label, chi2, ndf, chi2 / max(ndf, 1), afb_h, afb_e,
                     afb_r, afb_re, (afb_h - afb_r) / max(np.hypot(afb_e, afb_re), 1e-300)))
        elif have_analytic:
            # the brace's sign convention relative to cos_theta_w() is not
            # assumed: compare with both and report which one describes Hammer
            fits = []
            for flip in (False, True):
                ref_of = (lambda m, f: (lambda x: FR.dgamma_dcos(
                    np.atleast_1d(x), vec, m, npoints=300, flip=f)))(mlep, flip)
                _dummy = plt.figure()
                _a1 = _dummy.add_subplot(211)
                _a2 = _dummy.add_subplot(212)
                c2, nd = _compare(_a1, _a2, cctr, res['cdens'], res['cerr'],
                                  ref_of, cedges, None, '', '', '', '', '')
                plt.close(_dummy)
                fits.append((c2, nd, flip, ref_of))
            best = min(fits, key=lambda t: t[0])
            c2, nd, flip, ref_of = best
            _compare(ax, axr, cctr, res['cdens'], res['cerr'], ref_of, cedges,
                     None, r'$\cos\theta_W$', r'$(1/\Gamma)\,d\Gamma/d\cos\theta_W$',
                     'analytic, same coefficients%s' % (' (c -> -c)' if flip else ''),
                     label, data_label)
            other = [t for t in fits if t is not best][0]
            ana = FR.dgamma_dcos(cctr, vec, mlep, npoints=300, flip=flip)
            afb_a = afb_from_hist(cedges, ana / max(np.sum(ana) * (cedges[1] - cedges[0]), 1e-300))
            # Only call it a convention if the flipped hypothesis actually FITS.
            # If neither does, picking the lesser evil and labelling it a
            # convention would hide a genuine angular mismatch.
            good_fit = c2 / max(nd, 1) < 3.
            if not good_fit:
                verdict = ('  [NEITHER convention describes Hammer: an ANGULAR '
                           'MISMATCH, not a convention]')
            elif flip:
                verdict = '  [matched with c -> -c: a CONVENTION, not physics]'
            else:
                verdict = ''
            print('[%-3s] cosW    chi2/ndf vs analytic = %.1f/%d = %.2f  '
                  '(other convention: %.1f/%d)%s'
                  % (name, c2, nd, c2 / max(nd, 1), other[0], other[1], verdict))
            print('[%-3s]         A_FB Hammer %+.4f +/- %.4f   analytic %+.4f   '
                  '(%+.1f sigma)' % (name, afb_h, afb_e, afb_a,
                                     (afb_h - afb_a) / max(afb_e, 1e-300)))
        else:
            ax.errorbar(cctr, res['cdens'], yerr=res['cerr'], fmt='o', ms=3.5,
                        color='C0', label=data_label)
            ax.set_title(label)
            ax.legend(fontsize=8)
            axr.axis('off')
            print('[%-3s] cosW: A_FB Hammer %+.4f +/- %.4f (no reference)'
                  % (name, afb_h, afb_e))
    fig.suptitle(r'$\cos\theta_W$ from Hammer alone -- the sign of $g$ and the '
                 r'$H_0 H_t$ interference')
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots_dir, 'hammer_costhetaW_from_scratch.png'),
                dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------ R(J/psi)
    num, num_e = results['tau']['integral'], results['tau']['integral_err']
    den, den_e = results['mu']['integral'], results['mu']['integral_err']
    r = num / max(den, 1e-300)
    r_err = r * math.sqrt((num_e / max(num, 1e-300)) ** 2
                          + (den_e / max(den, 1e-300)) ** 2)
    print('\nR(J/psi) from the Hammer spectra = %.4f +/- %.4f (MC stat)' % (r, r_err))
    if args.check_generator:
        print('  (meaningless under --check-generator: that is pure phase space)')
    elif ses_is_bgl:
        r_ana = FR.r_jpsi(vec, npoints=4000)
        share = FR.ht_share(vec, FR.M_TAU)
        diff = r - r_ana
        print('  analytic, same coefficients    = %.4f' % r_ana)
        print('  Hammer - analytic              = %+.4f +/- %.4f   (%+.1f sigma, '
              '%+.2f%%)' % (diff, r_err, diff / max(r_err, 1e-300),
                           100. * diff / r_ana))
        print('  H_t share of Gamma_tau          = %.1f%%' % (100. * share))
        if share > 1e-6:
            print('  excess / H_t share              = %.3f   <- constant across a '
                  'dvec scan means the discrepancy is in H_t'
                  % ((diff / r_ana) / share))
        print('  (lattice, for reference: 0.2597 +/- 0.0027)')
    else:
        print('  (Kiselev: the generator model, not expected to match the lattice)')
    print('plots -> %s/' % args.plots_dir)


if __name__ == '__main__':
    main()
