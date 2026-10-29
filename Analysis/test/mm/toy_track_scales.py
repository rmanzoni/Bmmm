'''Closure test of vertex_pull_study.py + fit_track_scales.py: track-level toy ntuples.
Every vertex is built from two straight muon tracks with stated errors
sigma(dxy), sigma(dsz); the real scatter is s_xy sigma(dxy), s_sz sigma(dsz).

  data : s_xy = SXY, s_sz = SSZ (command line), a fraction FNP of non-prompt
         J/psi (true forward displacement), flat background with scatter x1.5,
         optionally an extra charge-linked shift of both muons (CHG um, mu+ to
         its left, mu- to its right): widens ALONG only.
  MC   : non-prompt, GEN truth; raw: honest; corrected: stated x1.2 and
         scatter x1.2 (honest) -> data / MC corr = (s_xy, s_sz).
usage:
  python3 toy_track_scales.py OUTDIR SXY SSZ FNP CHG_um
  python3 vertex_pull_study.py fill --files OUTDIR/data_toy.root --label data2026X --out OUTDIR/data.npz
  python3 vertex_pull_study.py fill --files OUTDIR/mc_toy.root --label mc --out OUTDIR/mc.npz
  python3 vertex_pull_study.py plot OUTDIR/data.npz OUTDIR/mc.npz --out OUTDIR/vps
  python3 fit_track_scales.py OUTDIR/vps.json --out OUTDIR/ts
Results (15 Oct 2026, 600k data / 400k MC candidates):
  1.04 1.08 0.08 0 -> s_dxy 1.031 +- 0.007, s_dsz 1.081 +- 0.004, along test passed,
                      charge test -0.001 +- 0.026
  1.04 1.08 0.08 5 -> charge test -0.467 +- 0.027 (17 sigma); the along test has
                      too little power at this statistics (left-half errors 2-6% per bin)'''
import sys
import numpy as np
import uproot

J = 3.0969
out, SXY, SSZ, FNP, CHG = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), \
    float(sys.argv[4]), float(sys.argv[5]) * 1e-4
rng = np.random.default_rng(17)
VC = ('xx', 'xy', 'xz', 'yy', 'yz', 'zz')
IJ = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def kin(n):
    pt = 7 + rng.exponential(6, n)
    eta = rng.uniform(-2.0, 2.0, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    # decay: opening ~ 2 m / pT, split randomly between phi and eta
    op = 2 * J / pt * rng.uniform(0.7, 1.3, n)
    ang = rng.uniform(0, 2 * np.pi, n)
    dphi, deta = op * np.cos(ang) / 2, op * np.sin(ang) / 2
    f1 = rng.uniform(0.35, 0.65, n)
    return pt, eta, phi, (phi + dphi, eta + deta, f1 * pt * 1.05), (phi - dphi, eta - deta,
                                                                    (1 - f1) * pt * 1.05)


def tracks_vertex(Vt, mus, sd, sz, kd, kz, chg_shift, q1):
    '''measured vertex, stated covariance (from sd, sz), chi2, per-muon offsets'''
    n = len(Vt)
    A = np.zeros((n, 3, 3)); b = np.zeros((n, 3)); lines = []
    for i, (phi, eta, pt) in enumerate(mus):
        lam = np.arctan(np.sinh(eta)); cl = np.cos(lam)
        u = np.stack([cl * np.cos(phi), cl * np.sin(phi), np.sin(lam)], -1)
        e1 = np.stack([-np.sin(phi), np.cos(phi), np.zeros(n)], -1)
        e2 = np.cross(u, e1)
        d = rng.normal(0, 1, n) * sd[i] * kd
        z = rng.normal(0, 1, n) * sz[i] * kz
        q = q1 if i == 0 else -q1
        d = d + q * chg_shift                          # mu+ to its left (+e1)
        P = Vt + d[:, None] * e1 + z[:, None] * e2
        M = (e1[:, :, None] * e1[:, None, :] / sd[i][:, None, None] ** 2 +
             e2[:, :, None] * e2[:, None, :] / sz[i][:, None, None] ** 2)
        A += M; b += np.einsum('nij,nj->ni', M, P)
        lines.append((P, e1, e2, sd[i], sz[i], d, z))
    Cv = np.linalg.inv(A)
    v = np.einsum('nij,nj->ni', Cv, b)
    chi2 = 0
    for P, e1, e2, s1, s2, d, z in lines:
        r = P - v
        chi2 = chi2 + (np.einsum('ni,ni->n', r, e1) / s1) ** 2 + (np.einsum('ni,ni->n', r, e2) / s2) ** 2
    return v, Cv, chi2, lines


def truth(n, fnp):
    pt, eta, phi, m1, m2 = kin(n)
    q1 = rng.choice([-1., 1.], n)
    pvt = np.stack([rng.normal(0.1, 0.001, n), rng.normal(-0.2, 0.001, n), rng.normal(0, 3, n)], -1)
    npm = rng.random(n) < fnp
    L = np.where(npm, rng.exponential(0.02, n), 0.)    # transverse decay length [cm]
    Vt = pvt + L[:, None] * np.stack([np.cos(phi), np.sin(phi), np.sinh(eta)], -1)
    pvbs = pvt + rng.normal(0, 1, (n, 3)) * np.array([8e-4, 8e-4, 30e-4])
    pvoff = pvt + rng.normal(0, 1, (n, 3)) * np.array([8e-4, 8e-4, 30e-4])
    return dict(n=n, pt=pt, eta=eta, phi=phi, mus=[m1, m2], q1=q1, pvt=pvt, Vt=Vt, pvbs=pvbs,
                pvoff=pvoff)


def sample(T, kd, kz, chg, mc_scale=1.0, mass=None):
    n, pt, eta, phi, mus, q1 = T['n'], T['pt'], T['eta'], T['phi'], T['mus'], T['q1']
    pvt, Vt, pvbs, pvoff = T['pvt'], T['Vt'], T['pvbs'], T['pvoff']
    sd = [(15e-4 * (1 + 4 / m[2])) * mc_scale for m in mus]
    sz = [(35e-4 * (1 + 4 / m[2]) * np.cosh(m[1]) ** 0.5) * mc_scale for m in mus]
    v, Cv, chi2, lines = tracks_vertex(Vt, mus, sd, sz, kd, kz, chg, q1)
    Cpv = np.zeros((n, 3, 3)); Cpv[:, 0, 0] = Cpv[:, 1, 1] = (8e-4) ** 2; Cpv[:, 2, 2] = (30e-4) ** 2
    o = dict(pv_bs_x=pvbs[:, 0], pv_bs_y=pvbs[:, 1], pv_bs_z=pvbs[:, 2],
             pv_x=pvoff[:, 0], pv_y=pvoff[:, 1], pv_z=pvoff[:, 2],
             vx=v[:, 0], vy=v[:, 1], vz=v[:, 2], vtx_chi2=chi2, pt=pt, charge=np.zeros(n),
             pv_refit_valid=np.ones(n), pv_refit_ntrk=rng.integers(8, 60, n).astype(float),
             mass=rng.normal(J, 0.03, n) if mass is None else mass)
    for e, (i, j) in zip(VC, IJ):
        o['pv_bs_cov_' + e] = Cpv[:, i, j]
        o['vtx_cov_' + e] = Cv[:, i, j]
    for k, (mu, (phi_m, eta_m, pt_m)) in enumerate(zip(('mu1', 'mu2'), mus)):
        o[mu + '_pt'], o[mu + '_eta'], o[mu + '_phi'] = pt_m, eta_m, phi_m
        o[mu + '_charge'] = q1 if k == 0 else -q1
        o[mu + '_id_medium'] = np.ones(n)
        o[mu + '_cov_dxy_dxy'] = sd[k] ** 2
        o[mu + '_dz_e'] = sz[k] * np.cosh(eta_m)          # sigma(dz) = sigma(dsz) / cos(lambda)
        dx, dy, dzz = (pvoff - Vt).T
        s_, c_, sh = np.sin(phi_m), np.cos(phi_m), np.sinh(eta_m)
        o[mu + '_dxy'] = lines[k][5] + dx * s_ - dy * c_
        o[mu + '_dz'] = lines[k][6] * np.cosh(eta_m) - dzz + (dx * c_ + dy * s_) * sh
        o[mu + '_pix_first_layer'] = np.ones(n)
        o[mu + '_n_pix_hit'] = np.full(n, 4.)
        o[mu + '_gen_vx'], o[mu + '_gen_vy'], o[mu + '_gen_vz'] = Vt.T
        o[mu + '_gen_pdgid'] = np.full(n, 13.)
    o.update(gen_pv_x=pvt[:, 0], gen_pv_y=pvt[:, 1], gen_pv_z=pvt[:, 2],
             gen_jpsi_vx=Vt[:, 0], gen_jpsi_vy=Vt[:, 1], gen_jpsi_vz=Vt[:, 2],
             gen_jpsi_nmu=np.full(n, 2.))
    return o


def write(path, d):
    with uproot.recreate(path) as f:
        f.mktree('tree', {k: np.float32 if k not in ('vx', 'vy', 'vz', 'pv_bs_x', 'pv_bs_y',
                                                     'pv_bs_z', 'pv_x', 'pv_y', 'pv_z',
                                                     'gen_pv_x', 'gen_pv_y', 'gen_pv_z',
                                                     'gen_jpsi_vx', 'gen_jpsi_vy', 'gen_jpsi_vz')
                          else np.float64 for k in d})
        f['tree'].extend(d)


NS, NB = 600000, 150000
s = sample(truth(NS, FNP), SXY, SSZ, CHG)
bkg = sample(truth(NB, 0.0), 1.5, 1.5, 0.0, mass=rng.uniform(J - 0.25, J + 0.25, NB))
data = {k: np.concatenate([s[k], bkg[k]]) for k in s if not k.startswith('gen') and '_gen_' not in k}
write(out + '/data_toy.root', data)

NM = 400000
T = truth(NM, 1.0)                                      # all non-prompt
raw = sample(T, 1.0, 1.0, 0.0)                          # honest
corr = sample(T, 1.0, 1.0, 0.0, mc_scale=1.2)           # same events: stated x1.2, scatter x1.2
m = dict(corr)
for k in ('vx', 'vy', 'vz', 'vtx_chi2', 'mass', 'pt'):
    m[k + '_raw'] = raw[k]
for e in VC:
    m['vtx_cov_%s_raw' % e] = raw['vtx_cov_' + e]
for mu in ('mu1', 'mu2'):
    for k in ('pt', 'eta', 'phi', 'dxy', 'dz', 'dz_e', 'pix_first_layer', 'n_pix_hit'):
        m['%s_%s_raw' % (mu, k)] = raw['%s_%s' % (mu, k)]
    m[mu + '_cov_corr_dxy_dxy'] = corr[mu + '_cov_dxy_dxy']
    m[mu + '_cov_dxy_dxy'] = raw[mu + '_cov_dxy_dxy']
m['emu_weight'] = np.ones(NM)
m['pu_weight_2026'] = np.ones(NM)
write(out + '/mc_toy.root', m)
print('ok: data s_xy %.3f s_sz %.3f, non-prompt %.2f, charge shift %.1f um' % (SXY, SSZ, FNP, CHG * 1e4))
