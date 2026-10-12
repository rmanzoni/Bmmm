'''
CovflowEmulation -- MC muon tracks made data-like BEFORE the vertex fit.

The production-level version of the chain agreed in the covflow decision log
(12 Oct 2026; HITEMU.md sections 3, 4 and 7). Per MC muon, in this order:

  1. KILL (optional, with kill maps). One run range per event, drawn with the
     luminosity shares of the maps; on every surface the maps have, a muon
     crossing a cell where data is less efficient than MC loses its hit(s)
     with probability P(kill) = 1 - eps(data)/eps(MC); in cells where data is
     more efficient it gets a weight instead (eps(data)/eps(MC) with the hit,
     (1 - eps(data))/(1 - eps(MC)) without). The code is covflow's own
     hitemu.emulate, the one the offline closure uses. With the per-layer
     masks (always in Bmmm) the context after the kill is exact.
  2. ROUTE A, killed muons only: C' = f_MC^-1( f_MC(C; c); c' ), the MC flow
     both ways, and the parameters get delta_1 ~ N(0, clip(C' - C)).
  3. COVFLOW at the muon's own context (c, or c' if killed), on its base
     covariance B = C or C':  C* = f_data^-1( f_MC(B; c_f); c_f ).
  4. DELTA_2 on every muon: delta_2 ~ N(0, clip(C* - B)), and the fit gets
         C_fit = B + clip(C* - B):
     C* in the directions where data is less precise than MC, the MC
     uncertainty B where data is more precise (nothing to smear there).
  5. A new reco::Track with the smeared parameters and C_fit, and a clone of
     the pat::Muon carrying the smeared momentum with the PDG muon mass (the
     mass is assigned, never measured).

CLIPPING. "Data more precise in a direction" is a property of a direction in
the 5 track parameters, not of a muon, and it must not depend on the units of
q/p, lambda, phi, dxy, dsz. So the increment D is decomposed RELATIVE TO THE
BASE: with B = L L^T, M = L^-1 D L^-T = U diag(lam) U^T, the clipped
increment is L U diag(max(lam, 0)) U^T L^T. lam < 0 = data more precise than
MC by a factor sqrt(1 + lam) in that direction. The same rule clips delta_1.

ROLES. The Run 3 flows are trained per muon role (mu1 = leading, mu2 =
trailing muon of the dimuon ntuple). A muon gets ONE emulated track per
event (its measurement cannot depend on which pair it is put in), so the role
is its rank in the event: the leading muon uses the mu1 flows, every other
muon the mu2 flows. In a two-muon event this is exactly the pair role.

HELIX START POINT for the kill: the track's own reference point (a point on
the trajectory), with the track's momentum there; the kill maps' MC start
point must be one that agrees with it at the cell level ('dimuon_vertex' or
'track', HITEMU 3.6).

REPRODUCIBILITY. Every random number of an event comes from a generator
seeded with (seed, run, lumi, event).
'''

from __future__ import print_function

import os
import sys

import numpy as np

import ROOT

from Bmmm.Analysis.TrackHitContent import hit_content, pix_first_layer_code
from Bmmm.Analysis.utils import (convert_cov, is_pos_def, cov_upper_triangle,
                                 track_params, track_with_params)

MUON_MASS = 0.1056583755          # GeV, PDG
TRACK_PARAMS = ('qoverp', 'lambda', 'phi', 'dxy', 'dsz')
CLIP_TOL = 1e-6                   # a relative eigenvalue below -CLIP_TOL counts as clipped
ACCEPTED_KM_ORIGINS = ('dimuon_vertex', 'track')


def import_hitemu():
    '''covflow's hitemu (kill maps, emulate), from the covflow package on the
    path. hitemu imports its siblings as top-level modules (pixel_eff_maps,
    features), so the covflow directory itself goes on sys.path, at the END so
    that its generic module names (data, features) shadow nothing.'''
    import covflow
    base = list(covflow.__path__)[0]
    if base not in sys.path:
        sys.path.append(base)
    import hitemu
    return hitemu


def clip_increment(D, B):
    '''D, B (N,5,5): increment and positive-definite base. Returns
    (D_plus (N,5,5), A (N,5,5), lam (N,5)) with D_plus the increment clipped
    relative to B and A such that A @ xi, xi ~ N(0, 1), has covariance D_plus.'''
    D = 0.5 * (D + np.swapaxes(D, -1, -2))
    L = np.linalg.cholesky(B)
    Linv = np.linalg.inv(L)
    M = Linv @ D @ np.swapaxes(Linv, -1, -2)
    M = 0.5 * (M + np.swapaxes(M, -1, -2))
    lam, U = np.linalg.eigh(M)
    LU = L @ U
    lp = np.clip(lam, 0., None)
    D_plus = (LU * lp[:, None, :]) @ np.swapaxes(LU, -1, -2)
    A = LU * np.sqrt(lp)[:, None, :]
    return D_plus, A, lam


def lorentz_vector(px, py, pz, m):
    E = np.sqrt(px * px + py * py + pz * pz + m * m)
    try:
        return ROOT.reco.Candidate.LorentzVector(px, py, pz, E)
    except Exception:
        return ROOT.Math.LorentzVector('ROOT::Math::PxPyPzE4D<double>')(px, py, pz, E)


class MCTrackEmulator(object):
    '''One per job. process(muons, event) returns the event's muons as clones
    carrying the emulated track; see the module doc for what is done.

    correctors : {'mu1': CovFlowCorrector, 'mu2': CovFlowCorrector}
    killmaps   : kill-map stem (killmaps_<epoch>, .npz/.json) or None: no
                 kill, no route A (covflow + delta_2 only)
    delta1, delta2 : switch the smearings off for checks (the covariances
                 are still corrected)'''

    def __init__(self, correctors, killmaps=None, seed=12345, delta1=True, delta2=True,
                 verbose=True):
        if set(correctors) != {'mu1', 'mu2'}:
            raise RuntimeError('the emulation needs the mu1 AND the mu2 flows, got %s'
                               % sorted(correctors))
        self.corr = correctors
        self.seed = int(seed)
        self.delta1, self.delta2 = bool(delta1), bool(delta2)
        self.km = None
        self.H = None
        if killmaps:
            self.H = import_hitemu()
            self.km = self.H.KillMaps.load(killmaps)
            origin = self.H.meta_origin(self.km.meta, 'mc')
            if origin not in ACCEPTED_KM_ORIGINS:
                raise RuntimeError(
                    'kill maps %s: MC helix start point %r. The emulation starts the helix '
                    'at the track reference point, which agrees at the cell level only with '
                    '%s (HITEMU 3.6): rebuild the maps with --helix-origin-mc dimuon_vertex'
                    % (killmaps, origin, ' / '.join(ACCEPTED_KM_ORIGINS)))
            if verbose:
                print('#### emulation: kill maps %s, epoch %s, surfaces %s, %d run range(s), '
                      'MC start point %s'
                      % (killmaps, self.km.meta.get('epoch'), ' '.join(self.km.surfaces),
                         len(self.km.ranges), origin))
        self.n = dict(muons=0, killed=0, killed_L1=0, killed_D1=0, killed_other=0,
                      weight_ne1=0, route_a=0, d1_clipped=0, d2_clipped=0, failed=0,
                      not_pos_def=0, charge_flip=0)
        self.sum_w = 0.0

    # ------------------------------------------------------------------
    def _rng(self, event):
        aux = event.eventAuxiliary()
        return np.random.default_rng([self.seed, int(aux.run()), int(aux.luminosityBlock()),
                                      int(aux.event()) & 0xFFFFFFFFFFFF])

    @staticmethod
    def _kill_inputs(trks, hcs):
        g = lambda f: np.array([f(t) for t in trks], np.float64)
        h = lambda k: np.array([hc[k] for hc in hcs], np.float64)
        v = dict(pt=g(lambda t: t.pt()), eta=g(lambda t: t.eta()), phi=g(lambda t: t.phi()),
                 q=g(lambda t: t.charge()), x0=g(lambda t: t.vx()), y0=g(lambda t: t.vy()),
                 z0=g(lambda t: t.vz()),
                 n_pix=h('n_pix_hit'), n_pix_b=h('n_pix_b_hit'), n_pix_e=h('n_pix_e_hit'),
                 first_b=h('pix_first_b_layer'), first_e=h('pix_first_e_disk'))
        for k in ('pt', 'eta', 'phi', 'q', 'x0', 'y0', 'z0'):
            v['h' + k] = v[k]
        hi = lambda k: np.array([int(hc[k]) for hc in hcs], np.int64)
        v['mask'], v['count'] = hi('pix_valid_mask'), hi('pix_hit_count')
        v['miss_mask'], v['inact_mask'] = hi('pix_miss_mask'), hi('pix_inact_mask')
        return v

    @staticmethod
    def _emulated_hit_content(hc, e, i):
        '''The hit content of muon i after the kill (exact with the masks).'''
        new = dict(hc)
        new['n_pix_hit'] = int(e['n_pix'][i])
        new['n_pix_b_hit'] = int(e['n_pix_b'][i])
        new['n_pix_e_hit'] = int(e['n_pix_e'][i])
        new['pix_first_b_layer'] = int(e['first_b'][i])
        new['pix_first_e_disk'] = int(e['first_e'][i])
        if 'mask' in e:
            new['pix_valid_mask'] = int(e['mask'][i])
            new['pix_hit_count'] = int(e['count'][i])
            new['n_pix_layer'] = int(e['n_pix_layer'][i])
            new['pix_first_layer'] = int(e['first_layer'][i])
            if 'miss_mask' in e:
                new['pix_miss_mask'] = int(e['miss_mask'][i])
            # a pixel layer lost is a tracker layer lost
            new['n_trk_layer'] = int(hc['n_trk_layer'] - (hc['n_pix_layer'] - new['n_pix_layer']))
        else:
            new['pix_first_layer'] = pix_first_layer_code(new['pix_first_b_layer'],
                                                          new['pix_first_e_disk'])
        return new

    # ------------------------------------------------------------------
    def process(self, muons, event):
        '''muons sorted by decreasing pT. Returns the clones, sorted by the
        (smeared) pT, each with: _emu_track, cov (raw C), cov_corr (C_fit),
        covflow_ok, covflow_zmax, _hit_content (after the kill),
        _hit_content_raw, emu (dict of per-muon diagnostics).'''
        N = len(muons)
        if N == 0:
            return []
        rng = self._rng(event)
        trks = [m.bestTrack() for m in muons]
        hcs = [dict(hit_content(m)) for m in muons]
        C = np.array([convert_cov(t.covariance()) for t in trks])
        par0 = np.array([track_params(t) for t in trks])
        pd = np.array([is_pos_def(c) and np.all(np.diagonal(c) > 0) for c in C])
        roles = ['mu1'] + ['mu2'] * (N - 1)
        self.n['muons'] += N
        self.n['not_pos_def'] += int((~pd).sum())

        # ---- 1. kill --------------------------------------------------
        killed = np.zeros(N, bool)
        kL1, kD1, kOther = np.zeros(N, bool), np.zeros(N, bool), np.zeros(N, bool)
        w = np.ones(N)
        rix = -1
        hcs_new = hcs
        if self.km is not None:
            rix = int(rng.choice(len(self.km.ranges), p=self.km.lumi))
            e = self.H.emulate(self._kill_inputs(trks, hcs), np.full(N, rix), self.km, rng)
            killed = np.asarray(e['kill_any'], bool)
            kL1, kD1 = np.asarray(e['kill_L1'], bool), np.asarray(e['kill_D1'], bool)
            kOther = np.asarray(e['kill_other'], bool)
            w = np.asarray(e['w'], np.float64)
            hcs_new = [self._emulated_hit_content(hcs[i], e, i) if killed[i] else hcs[i]
                       for i in range(N)]
        self.n['killed'] += int(killed.sum())
        self.n['killed_L1'] += int(kL1.sum())
        self.n['killed_D1'] += int(kD1.sum())
        self.n['killed_other'] += int(kOther.sum())
        self.n['weight_ne1'] += int((w != 1).sum())
        self.sum_w += float(w.sum())

        # ---- the clones: raw context on the original, final on the clone
        clones = [ROOT.pat.Muon(m) for m in muons]
        for i, cl in enumerate(clones):
            cl._hit_content_raw = hcs[i]
            cl._hit_content = hcs_new[i]
        ctx_raw = {}
        ctx_fin = {}
        for i in range(N):
            corr = self.corr[roles[i]]
            ctx_raw[i] = corr.context(muons[i], trks[i], event)
            ctx_fin[i] = corr.context(clones[i], trks[i], event)

        # ---- 2-4. route A, covflow, delta_2, per role ------------------
        B = C.copy()
        d1 = np.zeros((N, 5))
        d2 = np.zeros((N, 5))
        C_fit = C.copy()
        ok = pd.copy()
        zmax = np.full(N, np.nan)
        n1clip = np.zeros(N, np.int64)
        n2clip = np.zeros(N, np.int64)
        lam2min = np.full(N, np.nan)
        for role in ('mu1', 'mu2'):
            idx = [i for i in range(N) if roles[i] == role and pd[i]]
            if not idx:
                continue
            corr = self.corr[role]
            try:
                ka = [i for i in idx if killed[i]]
                if ka:
                    Cp = corr.CFEAT.packed_to_matrix(corr.morph_mc_batch(
                        np.array([cov_upper_triangle(C[i]) for i in ka]),
                        np.array([ctx_raw[i] for i in ka]), np.array([ctx_fin[i] for i in ka])))
                    Dp, A, lam = clip_increment(Cp - C[ka], C[ka])
                    n1clip[ka] = (lam < -CLIP_TOL).sum(1)
                    if self.delta1:
                        d1[ka] = np.einsum('nij,nj->ni', A, rng.standard_normal((len(ka), 5)))
                    B[ka] = C[ka] + Dp          # = C' where C' - C >= 0
                    self.n['route_a'] += len(ka)
                    self.n['d1_clipped'] += int((n1clip[ka] > 0).sum())
                packed_star, zm = corr._morph_batch(
                    np.array([cov_upper_triangle(B[i]) for i in idx]),
                    np.array([ctx_fin[i] for i in idx]))
                Cs = corr.CFEAT.packed_to_matrix(packed_star)
                Dp, A, lam = clip_increment(Cs - B[idx], B[idx])
                n2clip[idx] = (lam < -CLIP_TOL).sum(1)
                lam2min[idx] = lam.min(1)
                if self.delta2:
                    d2[idx] = np.einsum('nij,nj->ni', A, rng.standard_normal((len(idx), 5)))
                C_fit[idx] = B[idx] + Dp
                zmax[idx] = zm
                self.n['d2_clipped'] += int((n2clip[idx] > 0).sum())
            except Exception as exc:
                print('[emulation] WARNING: %s batch of %d muon(s) failed: %s: %s -- left raw'
                      % (role, len(idx), type(exc).__name__, exc))
                ok[idx] = False
                B[idx], C_fit[idx], d1[idx], d2[idx] = C[idx], C[idx], 0., 0.
                self.n['failed'] += len(idx)

        # ---- 5. tracks and muon clones ---------------------------------
        par = par0 + d1 + d2
        for i, cl in enumerate(clones):
            trk = track_with_params(trks[i], par[i], C_fit[i]) if ok[i] else None
            if ok[i] and trk is None:
                self.n['charge_flip'] += 1
                ok[i] = False
            if trk is None:
                trk, par[i], C_fit[i], d1[i], d2[i] = trks[i], par0[i], C[i], 0., 0.
            mom = trk.momentum()
            cl.setP4(lorentz_vector(mom.x(), mom.y(), mom.z(), MUON_MASS))
            cl._emu_track = trk
            cl._p4_raw = muons[i].p4()
            cl.cov = C[i]
            cl.is_cov_pos_def = bool(pd[i])
            cl.cov_corr = C_fit[i]
            cl.covflow_ok = bool(ok[i])
            cl.covflow_zmax = float(zmax[i])
            cl.emu = dict(ok=bool(ok[i]), role=1 if roles[i] == 'mu1' else 2, range=rix,
                          killed=bool(killed[i]), killed_L1=bool(kL1[i]), killed_D1=bool(kD1[i]),
                          killed_other=bool(kOther[i]), weight=float(w[i]),
                          d1=d1[i].copy(), d2=d2[i].copy(), d1_nclip=int(n1clip[i]),
                          d2_nclip=int(n2clip[i]), d2_lam_min=float(lam2min[i]))
        clones.sort(key=lambda m: m.pt(), reverse=True)
        return clones

    def summary(self):
        n = max(self.n['muons'], 1)
        f = lambda k: '%d (%.2f%%)' % (self.n[k], 100. * self.n[k] / n)
        return ('[emulation] %d muons: killed %s (L1 %s, D1 %s, other %s), weight != 1 %s, '
                'mean weight %.4f, route A %s with delta_1 clipped %s, delta_2 clipped %s, '
                'not pos-def %s, failed %s, charge flip %s'
                % (self.n['muons'], f('killed'), f('killed_L1'), f('killed_D1'),
                   f('killed_other'), f('weight_ne1'), self.sum_w / n, f('route_a'),
                   f('d1_clipped'), f('d2_clipped'), f('not_pos_def'), f('failed'),
                   f('charge_flip')))
