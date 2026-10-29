'''
Bs -> Ds(-> phi(-> K K) pi) mu (nu) candidate for the R(Ds) ntuples.

Replaces the 2023 prototype. What it does, per candidate:

  1. Two sequential kinematic-fit schemes on the same four tracks
     (RDsKinVtxFitter), stored side by side so they can be compared on MC:
        cons   phi mass constraint on KK, then Ds mass constraint on phi + pi,
               then Ds + mu. Also phi + pi WITHOUT the Ds constraint ('dsphi'),
               which gives m(KKpi) with the phi constraint (the AN v4 / previous
               ntuples definition).
        free   no mass constraint anywhere: KK, KK + pi, KKpi + mu.
     Every quantity that depends on the fits is written once per scheme, with
     the prefix cons_ or free_.
  2. The primary vertex: the one with the smallest 3D distance from the line
     through the Bs vertex along the visible (KKpi mu) momentum (AN Eq. 3;
     Bs vertex of the cons scheme, of the free scheme if that failed, the PV
     closest in z to the muon if both failed), refitted with the beamspot
     constraint and the four signal tracks removed (PrimaryVertex, shared
     with the RJpsi and dimuon ntuples). The same PV serves both schemes.
  3. Displacements and pointing of the phi, Ds and Bs vertices, the Ds vertex
     relative to the Bs vertex, the muon impact parameters with respect to the
     Ds flight line.
  4. Bs momentum hypotheses (AN Sec. 5.4), per scheme: collinear, equal
     velocity along PV -> V_Bs and PV -> V_Ds (xyz_bs, xyz_ds) and the two
     neutrino solutions along each of the two directions (math1/2_bs,
     math1/2_ds); for each one q2, m_miss2, pT_miss, E*_mu, E#_mu, the Bs pT
     and the helicity angles (RDsKinematics).
  5. The soft photon of Ds* -> Ds gamma (AN Sec. 5.5), per scheme.
  6. Scheme-independent: raw masses, charges and charge class, reflection
     masses, PF isolation of mu, Ds and KKpi mu against the refitted PV.

Tracks are RDsTrack wrappers: the same pat object can be a kaon in one
candidate and the pion in another, each with its own mass and four-momentum,
and the momentum scale of the calibration (MC) applied. The track handed to
every fit and IP computation comes from TrackRecord.fit_track, the single
insertion point of the corrections, in this order:
    1. non-positive-definite (MiniAOD packing) covariance repaired (fix_track)
    2. momentum calibration (--trk-calib): q/p and its covariance row and
       column scaled by 1/f
    3. covariance scaling (--cov-scale): sigma_i -> s_i sigma_i, correlations kept
'''

import numpy as np
from scipy import stats
from PhysicsTools.HeppyCore.utils.deltar import deltaR

import ROOT
ROOT.gSystem.Load('libBmmmAnalysis')
from ROOT import RDsKinVtxFitter
# RJpsiKinVtxFitter holds the IPTools wrappers (signedIP3D, signedDecayLength3D,
# jetTrackDistance) and, through its includes, VertexDistance3D / XY
from ROOT import RJpsiKinVtxFitter

from Bmmm.Analysis.utils import (
    convert_cov, is_pos_def, fix_track, track_params, track_with_params,
    track_with_cov, scale_cov, compute_IP3D, COV_NO_SCALE,
)
from Bmmm.Analysis.PrimaryVertex import (
    PrimaryVertexRefitMixin, build_beamspot_vertex, build_hybrid_pv,
)
from Bmmm.Analysis.JpsiChargedCandidate import (
    _ISO_CONES, _ISO_CH_ALL_IDS, _ISO_CH_HAD_IDS, _ISO_NH_IDS, _ISO_PH_IDS,
    _ISO_DR_VETO_CH, _ISO_DR_VETO_NH, _ISO_PU_DR_VETO, _ISO_NEUTRAL_PT, _ISO_PU_PT,
)
from Bmmm.Analysis.RJPsiNuReco import reconstruct
from Bmmm.Analysis.RDsKinematics import (
    M_BS, M_DS, M_DSST, M_PHI, M_K, M_PI, M_MU,
    np4, mass, mass2, pt, eta, phi, p3mag, unit3,
    collinear_p4, equal_velocity_p4, semileptonic_observables, helicity_angles,
    e_gamma_collinear, reflection_masses, charge_class,
)

kinfit  = RDsKinVtxFitter()
iptools = RJpsiKinVtxFitter()

LorentzVector = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzE4D<double>')

SCHEMES = ('cons', 'free')
# vertices of each scheme, in the order RDsKinVtxFitter returns the trees
SCHEME_VERTICES = {'cons': ('phi', 'dsphi', 'ds', 'bs'),
                   'free': ('phi', 'ds', 'bs')}
DIRECTIONS = ('bs', 'ds')
HYPOTHESES = ('coll', 'xyz_bs', 'xyz_ds', 'math1_bs', 'math2_bs', 'math1_ds', 'math2_ds')
HYP_OBS    = ('pt', 'eta', 'phi', 'beta', 'q2', 'm_miss2', 'pt_miss', 'pt_miss_sca',
              'e_star_mu', 'e_hash_mu', 'cos_theta_phi_ds', 'cos_theta_mu_w', 'chi')
NAN = float('nan')


##########################################################################################
#####      TRACKS
##########################################################################################

class TrackRecord(object):
    '''One physical track (pat::PackedCandidate or pat::Muon) of the event,
    shared by every candidate and every mass hypothesis it enters.

        obj          the pat object
        collection   'pf', 'lost' or 'muon'; index its position there
        f, z         momentum calibration factor p'/p and its Gaussian number
                     (1, NaN without --trk-calib)
        cov          raw 5x5 covariance (numpy), is_cov_pos_def its check
    '''
    cov_scaler = None   # utils.CovScaler, installed once per job (--cov-scale)

    def __init__(self, obj, collection, index, f=1., z=NAN):
        self.obj        = obj
        self.collection = collection
        self.index      = index
        self.f          = f
        self.z          = z
        self.cov        = convert_cov(obj.bestTrack().covariance())
        self.is_cov_pos_def = bool(is_pos_def(self.cov))
        self.cov_fixed  = False
        self.cov_scale  = COV_NO_SCALE
        self._fit_track = None

    def fit_track(self):
        '''The reco::Track every fit and IP computation uses (module doc).'''
        if self._fit_track is not None:
            return self._fit_track
        raw = self.obj.bestTrack()
        trk = raw
        cov = self.cov
        if not self.is_cov_pos_def:
            trk = fix_track(raw)
            cov = convert_cov(trk.covariance())
            self.cov_fixed = True
        if self.f != 1.:
            params = track_params(trk)
            params[0] /= self.f
            d = np.diag([1. / self.f, 1., 1., 1., 1.])
            cov = d.dot(cov).dot(d)
            scaled = track_with_params(trk, params, cov)
            if scaled is not None:
                trk = scaled
        if self.cov_scaler is not None:
            self.cov_scale = tuple(self.cov_scaler(raw))
            cov = scale_cov(cov, self.cov_scale)
            trk = track_with_cov(trk, cov)
        self._fit_track = trk
        return trk


class RDsTrack(object):
    '''A TrackRecord under a mass hypothesis. Kinematics (p4, pt, px, ...,
    energy, mass) carry the calibrated momentum and the assigned mass; eta and
    phi are unchanged by the calibration. Everything else is delegated to the
    pat object. Per-candidate attributes (pv, bs, ...) live on the wrapper.'''

    def __init__(self, record, mass_hyp):
        o = record.obj
        f = record.f
        px, py, pz = o.px() * f, o.py() * f, o.pz() * f
        self.record = record
        self._mass  = mass_hyp
        self._np4   = np.array([np.sqrt(px * px + py * py + pz * pz + mass_hyp ** 2), px, py, pz])
        self._lv    = LorentzVector(px, py, pz, self._np4[0])
        self.cov            = record.cov
        self.is_cov_pos_def = record.is_cov_pos_def

    # calibrated kinematics
    def p4(self):     return self._lv
    def np4(self):    return self._np4
    def px(self):     return self._np4[1]
    def py(self):     return self._np4[2]
    def pz(self):     return self._np4[3]
    def pt(self):     return float(np.hypot(self._np4[1], self._np4[2]))
    def p(self):      return float(p3mag(self._np4))
    def energy(self): return self._np4[0]
    def mass(self):   return self._mass
    def eta(self):    return self.record.obj.eta()
    def phi(self):    return self.record.obj.phi()
    def raw_pt(self): return self.record.obj.pt()

    @property
    def _cov_fit_track(self):
        # read by CommonBranches.cov_track: errors from the track the fits used
        return self.record.fit_track()

    def fit_track(self):
        return self.record.fit_track()

    def __getattr__(self, name):
        # only reached for attributes not found on the wrapper itself
        if name in ('record', '_mass', '_np4', '_lv'):
            raise AttributeError(name)
        return getattr(self.record.obj, name)


##########################################################################################
#####      FIT TREES
##########################################################################################

def _tree_valid(tree):
    try:
        return bool(tree.get()) and (not tree.isEmpty()) and tree.isValid()
    except Exception:
        return False


def _kin_np4(kin_particle):
    v = kin_particle.currentState().kinematicParameters().vector()
    px, py, pz, m = v.At(3), v.At(4), v.At(5), v.At(6)
    return np.array([np.sqrt(px * px + py * py + pz * pz + m * m), px, py, pz])


class FitInfo(object):
    '''What is read off one kinematic tree: the top particle four-momentum,
    the decay vertex and its quality, the refitted children.'''

    def __init__(self, tree):
        self.valid = _tree_valid(tree)
        self.p4 = None
        self.vtx = None
        self.chi2 = self.ndof = self.prob = NAN
        self.children = []
        if not self.valid:
            return
        tree.movePointerToTheTop()
        self.p4  = _kin_np4(tree.currentParticle())
        self.vtx = tree.currentDecayVertex().get()
        self.chi2 = self.vtx.chiSquared()
        self.ndof = self.vtx.degreesOfFreedom()
        self.prob = stats.chi2.sf(self.chi2, self.ndof) if self.ndof > 0 else NAN
        if tree.movePointerToTheFirstChild():
            self.children.append(_kin_np4(tree.currentParticle()))
            while tree.movePointerToTheNextChild():
                self.children.append(_kin_np4(tree.currentParticle()))
        tree.movePointerToTheTop()
        self.tree = tree   # keep the tree alive: vtx points into it

    def position(self):
        p = self.vtx.position()
        return np.array([p.x(), p.y(), p.z()])

    def reco_vertex(self):
        '''reco::Vertex (position + covariance), for IPTools.'''
        pos = self.vtx.position()
        point = ROOT.reco.Vertex.Point(pos.x(), pos.y(), pos.z())
        return ROOT.reco.Vertex(point, self.vtx.error().matrix(),
                                self.vtx.chiSquared(), self.vtx.degreesOfFreedom(), 2)


##########################################################################################
#####      CANDIDATE
##########################################################################################

class RDsCandidate(PrimaryVertexRefitMixin):

    # ---- correction plumbing called by BaseInspector.main / looper -----------
    @classmethod
    def set_cov_scaler(cls, scaler):
        '''--cov-scale: applied to every track and to the muon (TrackRecord).'''
        TrackRecord.cov_scaler = scaler

    @classmethod
    def set_cov_corrector(cls, corrector):
        if corrector is not None:
            raise RuntimeError('--covflow is not available for the R(Ds) ntuples: '
                               'the covflow models are trained on Run 3 muons. Use '
                               '--cov-scale.')

    @classmethod
    def prime_cov_corrector(cls, objs, event=None):
        pass

    def __init__(self, mu, k1, k2, pi):
        '''mu, k1, k2, pi: RDsTrack wrappers, k1 the leading kaon in pT.'''
        self.mu, self.k1, self.k2, self.pi = mu, k1, k2, pi
        # removed from the PV refit and from the 'clean' isolation sums
        self.signal_objects = [mu, k1, k2, pi]
        self.muons = [mu]

        # the kaon with the charge of the pion (in a right-sign Ds: the K of
        # the K+ pi+ pair); for KK-wrong candidates fall back to k1
        if k1.charge() == pi.charge():
            self.k_same = k1
        elif k2.charge() == pi.charge():
            self.k_same = k2
        else:
            self.k_same = k1

        self.p4_kk     = k1.np4() + k2.np4()
        self.p4_kkpi   = self.p4_kk + pi.np4()
        self.p4_kkpimu = self.p4_kkpi + mu.np4()

        self.kk_charge    = int(k1.charge() + k2.charge())
        self.mupi_charge  = int(mu.charge() + pi.charge())
        self.charge_total = int(self.kk_charge + pi.charge() + mu.charge())
        self.charge_class = charge_class(k1.charge(), k2.charge(), pi.charge(), mu.charge())

    # ---- raw (calibrated) kinematics of the visible system ------------------
    def mass(self):   return mass(self.p4_kkpimu)
    def pt(self):     return pt(self.p4_kkpimu)
    def eta(self):    return eta(self.p4_kkpimu)
    def phi(self):    return phi(self.p4_kkpimu)
    def charge(self): return self.charge_total

    ##########################################################################
    #####      FITS
    ##########################################################################
    def fit(self):
        '''Both fit schemes. Fills self.fits[scheme][vertex] with FitInfo.'''
        t = [o.fit_track() for o in (self.k1, self.k2, self.pi, self.mu)]
        cons = kinfit.FitConstrained(t[0], t[1], t[2], t[3], M_K, M_PI, M_MU, M_PHI, M_DS)
        free = kinfit.FitFree(t[0], t[1], t[2], t[3], M_K, M_PI, M_MU)
        self.fits = {
            'cons': dict(zip(SCHEME_VERTICES['cons'], [FitInfo(cons[i]) for i in range(4)])),
            'free': dict(zip(SCHEME_VERTICES['free'], [FitInfo(free[i]) for i in range(3)])),
        }

    def ds_fit_for_cut(self, scheme):
        '''The Ds vertex the AN cut applies to: phi + pi without the Ds
        constraint in the cons scheme, KK + pi in the free scheme.'''
        return self.fits[scheme]['dsphi' if scheme == 'cons' else 'ds']

    def passes_vertex_cuts(self, cuts):
        for s in SCHEMES:
            f_phi, f_ds = self.fits[s]['phi'], self.ds_fit_for_cut(s)
            if f_phi.valid and f_ds.valid and \
               f_phi.prob > cuts['phi_vtx_prob'] and f_ds.prob > cuts['ds_vtx_prob']:
                return True
        return False

    ##########################################################################
    #####      PRIMARY VERTEX
    ##########################################################################
    def choose_pv(self, vertices, beamspot, pf, lost):
        bs_fit = None
        for s in SCHEMES:
            if self.fits[s]['bs'].valid:
                bs_fit = self.fits[s]['bs']
                break
        self.pv_from_scheme = SCHEMES.index(s) if bs_fit is not None else -1
        if bs_fit is not None:
            sv = bs_fit.vtx.position()
            d = ROOT.Math.XYZVector(self.p4_kkpimu[1], self.p4_kkpimu[2], self.p4_kkpimu[3])
            ip = [compute_IP3D(v, sv, d) for v in vertices]
            self.pv_idx = int(np.argmin(ip))
        else:
            self.pv_idx = int(np.argmin([abs(self.mu.bestTrack().dz(v.position())) for v in vertices]))
        self.pv = vertices[self.pv_idx]
        self.bs = build_beamspot_vertex(beamspot, self.pv.z())
        self.refit_primary_vertex(beamspot, pf, lost, vertices)
        self.pv_bs = self.pv_refit if self.pv_refit_valid else build_hybrid_pv(self.bs, self.pv)
        p = self.pv_bs.position()
        self.pv_pos = np.array([p.x(), p.y(), p.z()])

    ##########################################################################
    #####      DISPLACEMENT
    ##########################################################################
    def displacement(self, info, p4):
        '''lxy, lxyz (Measurement1D) and cos2d, cos3d of a vertex wrt the PV.'''
        out = {}
        if not info.valid:
            return out
        out['lxy']  = ROOT.VertexDistanceXY().distance(self.pv_bs, info.vtx.vertexState())
        out['lxyz'] = ROOT.VertexDistance3D().distance(self.pv_bs, info.vtx.vertexState())
        d = info.position() - self.pv_pos
        out['cos2d'] = _cos(d[:2], p4[1:3])
        out['cos3d'] = _cos(d, p4[1:])
        return out

    ##########################################################################
    #####      EVERYTHING ELSE
    ##########################################################################
    def compute(self, vertices, beamspot, pf, lost, photons, cuts):
        '''After fit() and passes_vertex_cuts(): PV, displacements, Bs
        hypotheses, IPs, photon, isolation, reflections.'''
        self.choose_pv(vertices, beamspot, pf, lost)
        self.out = {}
        for s in SCHEMES:
            self._compute_scheme(s, photons, cuts)
        self._compute_common(pf, vertices)

    def _compute_scheme(self, s, photons, cuts):
        o = self.out
        fits = self.fits[s]

        # ---- vertices -------------------------------------------------------
        for name, info in fits.items():
            pre = '%s_%s_' % (s, name)
            o[pre + 'good'] = float(info.valid)
            if not info.valid:
                continue
            pos = info.position()
            o[pre + 'x'], o[pre + 'y'], o[pre + 'z'] = pos
            o[pre + 'chi2'], o[pre + 'ndof'], o[pre + 'prob'] = info.chi2, info.ndof, info.prob
            o[pre + 'mass'] = mass(info.p4)
            o[pre + 'pt']   = pt(info.p4)
            disp = self.displacement(info, info.p4)
            for q in ('lxy', 'lxyz'):
                o[pre + q]          = disp[q].value()
                o[pre + q + '_err'] = disp[q].error()
                o[pre + q + '_sig'] = disp[q].significance()
            o[pre + 'cos2d'] = disp['cos2d']
            o[pre + 'cos3d'] = disp['cos3d']
            err = info.vtx.error().matrix()
            for lab, (i, j) in zip(('xx', 'xy', 'xz', 'yy', 'yz', 'zz'),
                                   ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
                o[pre + 'cov_' + lab] = err(i, j)

        ds, bs = fits['ds'], fits['bs']

        # ---- refitted momenta of the four tracks ------------------------------
        # phi tree children: [k1, k2]; Ds tree: [pi, phi]; Bs tree: [mu, Ds]
        rf = {}
        if fits['phi'].valid and len(fits['phi'].children) == 2:
            rf['k1'], rf['k2'] = fits['phi'].children
        if ds.valid and len(ds.children) == 2:
            rf['pi'] = ds.children[0]
        if bs.valid and len(bs.children) == 2:
            rf['mu'] = bs.children[0]
        for obj, p4 in rf.items():
            o['%s_%s_rf_pt' % (s, obj)]  = pt(p4)
            o['%s_%s_rf_eta' % (s, obj)] = eta(p4)
            o['%s_%s_rf_phi' % (s, obj)] = phi(p4)

        # ---- Ds vertex relative to the Bs vertex ------------------------------
        if ds.valid and bs.valid:
            m_xy = ROOT.VertexDistanceXY().distance(bs.vtx.vertexState(), ds.vtx.vertexState())
            m_3d = ROOT.VertexDistance3D().distance(bs.vtx.vertexState(), ds.vtx.vertexState())
            o[s + '_ds_lxy_bs']      = m_xy.value()
            o[s + '_ds_lxy_bs_err']  = m_xy.error()
            o[s + '_ds_lxy_bs_sig']  = m_xy.significance()
            o[s + '_ds_lxyz_bs']     = m_3d.value()
            o[s + '_ds_lxyz_bs_sig'] = m_3d.significance()
            d = ds.position() - bs.position()
            o[s + '_ds_cos3d_bs']    = _cos(d, ds.p4[1:])
            o[s + '_ds_cos2d_bs']    = _cos(d[:2], ds.p4[1:3])

        # ---- hadron side of the scheme ----------------------------------------
        # Ds tree children are [pi, phi(KK)], as pushed in RDsKinVtxFitter
        if not ds.valid:
            return
        p_ds = ds.p4
        p_pi, p_phi = (ds.children[0], ds.children[1]) if len(ds.children) == 2 else (self.pi.np4(), self.p4_kk)
        phi_fit = fits['phi']
        if phi_fit.valid and len(phi_fit.children) == 2:
            p_kk = phi_fit.children                         # [k1, k2], fit input order
            p_ksame = p_kk[0] if self.k_same is self.k1 else p_kk[1]
        else:
            p_ksame = self.k_same.np4()
        p_mu = self.mu.np4()
        visible = p_ds + p_mu

        directions = {}
        for lab, info in (('bs', bs), ('ds', ds)):
            if info.valid:
                directions[lab] = info.position() - self.pv_pos

        # ---- Bs momentum hypotheses -------------------------------------------
        hyps = {'coll': collinear_p4(visible)}
        for lab, d in directions.items():
            hyps['xyz_' + lab] = equal_velocity_p4(d, visible)
            sols = _nu_solutions(visible, d)
            if sols is not None:
                hyps['math1_' + lab], hyps['math2_' + lab] = sols[0], sols[1]
                o['%s_math_real_%s' % (s, lab)] = sols[2]

        for h, p_b in hyps.items():
            obs = semileptonic_observables(p_b, p_ds, p_mu)
            hel = helicity_angles(p_b, p_ds, p_phi, p_pi, p_ksame, p_mu)
            obs.update(hel)
            for q in HYP_OBS:
                o['%s_%s_%s' % (s, h, q)] = obs[q]
        # cos_theta_k_pi does not depend on the Bs hypothesis
        o[s + '_cos_theta_k_pi'] = helicity_angles(hyps['coll'], p_ds, p_phi, p_pi, p_ksame, p_mu)['cos_theta_k_pi']

        # ---- corrected mass along PV -> V_Bs ----------------------------------
        if 'bs' in directions:
            u = unit3(directions['bs'])
            for lab, p4 in (('', self.p4_kkpimu), ('ds_', p_ds)):
                par  = float(np.dot(p4[1:], u))
                perp = np.sqrt(max(0., float(np.dot(p4[1:], p4[1:])) - par * par))
                o['%s_%sp_perp' % (s, lab)] = perp
                if lab == '':
                    o[s + '_mcorr'] = np.sqrt(max(mass2(p4), 0.) + perp * perp) + perp

        # ---- muon impact parameters wrt the Ds flight line ---------------------
        mu_trk = self.mu.fit_track()
        if 'ds' in directions:
            d = directions['ds']
            refs = {'pv': self.pv_bs, 'sv': ds.reco_vertex()}
            for ref, vtx in refs.items():
                ip = iptools.signedIP3D(mu_trk, d[0], d[1], d[2], vtx)
                o['%s_mu_ip3d_ds_%s' % (s, ref)]         = ip.value()
                o['%s_mu_ip3d_ds_%s_err' % (s, ref)]     = ip.error()
                o['%s_mu_ip3d_ds_%s_sig' % (s, ref)]     = ip.significance()
            # AN "IP3D": signed decay length of the mu wrt V_Ds along the Ds flight
            dl = iptools.signedDecayLength3D(mu_trk, d[0], d[1], d[2], refs['sv'])
            o[s + '_mu_dl3d_ds']     = dl.value()
            o[s + '_mu_dl3d_ds_err'] = dl.error()
            o[s + '_mu_dl3d_ds_sig'] = dl.significance()
            jtd = iptools.jetTrackDistance(mu_trk, d[0], d[1], d[2], refs['pv'])
            o[s + '_mu_dist_to_ds_dir'] = abs(jtd.second.value())

        # ---- soft photon -------------------------------------------------------
        self._photon(s, photons, p_ds, p_mu, directions, cuts)

    def _photon(self, s, photons, p_ds, p_mu, directions, cuts):
        o = self.out
        best, best_dr, n_in = None, 1e9, 0
        ds_eta, ds_phi = eta(p_ds), phi(p_ds)
        for g in photons:
            pg = np.array([g.energy(), g.px(), g.py(), g.pz()])
            m_dsg = mass(p_ds + pg)
            if abs(m_dsg - M_DSST) >= cuts['gamma_window']:
                continue
            n_in += 1
            dr = deltaR(ds_eta, ds_phi, g.eta(), g.phi())
            if dr < best_dr:
                best, best_dr = (g, pg, m_dsg), dr
        o[s + '_gamma_n'] = n_in
        e_coll = e_gamma_collinear(p_ds)
        o[s + '_gamma_e_coll'] = e_coll
        if best is None:
            return
        g, pg, m_dsg = best
        o[s + '_gamma_pt']   = g.pt()
        o[s + '_gamma_eta']  = g.eta()
        o[s + '_gamma_phi']  = g.phi()
        o[s + '_gamma_dr']   = best_dr
        o[s + '_gamma_m_ds_gamma']  = m_dsg
        o[s + '_gamma_dm']   = m_dsg - mass(p_ds)
        o[s + '_gamma_e_coll_over_pt'] = e_coll / g.pt() if g.pt() > 0 else NAN
        p_dsg = p_ds + pg
        vis = p_dsg + p_mu
        hyps = {'coll': collinear_p4(vis)}
        if 'bs' in directions:
            hyps['xyz_bs'] = equal_velocity_p4(directions['bs'], vis)
        for h, p_b in hyps.items():
            o['%s_gamma_%s_q2' % (s, h)]      = mass2(p_b - p_dsg)
            o['%s_gamma_%s_m_miss2' % (s, h)] = mass2(p_b - p_dsg - p_mu)

    def _compute_common(self, pf, vertices):
        o = self.out
        o['mass'], o['pt'], o['eta'], o['phi'] = self.mass(), self.pt(), self.eta(), self.phi()
        o['charge']       = self.charge_total
        o['kk_charge']    = self.kk_charge
        o['mupi_charge']  = self.mupi_charge
        o['charge_class'] = self.charge_class

        # primary vertex: the refit when valid, else the hybrid (bs x,y + PV z)
        o['pv_idx']         = self.pv_idx
        o['pv_from_scheme'] = self.pv_from_scheme
        o['pv_refit_valid'] = float(self.pv_refit_valid)
        o['pv_refit_ntrk']  = self.pv_refit_ntrk
        o['pv_x'], o['pv_y'], o['pv_z'] = self.pv_pos
        o['pv_ntrk'] = self.pv_bs.tracksSize()
        o['pv_chi2'] = self.pv_bs.chi2()
        o['pv_ndof'] = self.pv_bs.ndof()
        err = self.pv_bs.error()
        for lab, (i, j) in zip(('xx', 'xy', 'xz', 'yy', 'yz', 'zz'),
                               ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
            o['pv_cov_' + lab] = err(i, j)
        o['bs_x'] = self.bs.position().x()
        o['bs_y'] = self.bs.position().y()

        o['kk_mass']   = mass(self.p4_kk)
        o['kkpi_mass'] = mass(self.p4_kkpi)
        for k, v in reflection_masses(self.k1.np4()[1:], self.k2.np4()[1:], self.pi.np4()[1:]).items():
            o['refl_' + k] = v
        o['dr_kk']     = deltaR(self.k1, self.k2)
        kk_eta, kk_phi = eta(self.p4_kk), phi(self.p4_kk)
        ds_eta, ds_phi = eta(self.p4_kkpi), phi(self.p4_kkpi)
        o['dr_phi_pi'] = deltaR(kk_eta, kk_phi, self.pi.eta(), self.pi.phi())
        o['dr_ds_mu']  = deltaR(ds_eta, ds_phi, self.mu.eta(), self.mu.phi())
        drs = [deltaR(self.mu, t) for t in (self.k1, self.k2, self.pi)]
        o['dr_mu_k1'], o['dr_mu_k2'], o['dr_mu_pi'] = drs
        o['max_dr_mu_tk'] = max(drs)
        self._isolation(pf, vertices)

    ##########################################################################
    #####      ISOLATION (the muon-POG algorithm against the refitted PV,
    #####      as JpsiChargedCandidate.compute_isolation, other objects)
    ##########################################################################
    def _isolation(self, pf, vertices):
        if pf is None:
            return
        custom_pv = self.pv_bs
        assoc = list(vertices)
        if 0 <= self.pv_idx < len(assoc):
            assoc[self.pv_idx] = custom_pv
        assoc_z = [v.position().z() for v in assoc]

        charged, neutral = [], []
        for c in pf:
            pdg = abs(c.pdgId())
            if pdg in _ISO_CH_ALL_IDS:
                vz = c.vz()
                ibest = min(range(len(assoc_z)), key=lambda i: abs(assoc_z[i] - vz))
                charged.append((c.eta(), c.phi(), c.pt(), pdg,
                                assoc[ibest] is custom_pv, self.is_signal_muon_cand(c)))
            elif pdg in _ISO_NH_IDS or pdg in _ISO_PH_IDS:
                if c.pt() > _ISO_NEUTRAL_PT:
                    neutral.append((c.eta(), c.phi(), c.pt(), pdg))

        objects = (('mu', self.mu.eta(), self.mu.phi(), self.mu.pt()),
                   ('ds', eta(self.p4_kkpi), phi(self.p4_kkpi), pt(self.p4_kkpi)),
                   ('b',  eta(self.p4_kkpimu), phi(self.p4_kkpimu), pt(self.p4_kkpimu)))
        o = self.out
        for cone in _ISO_CONES:
            rr = '%02d' % int(round(cone * 10))
            for name, oe, op, opt in objects:
                ch = ch_clean = pu = pu_clean = nh = ph = 0.
                for c_eta, c_phi, c_pt, pdg, from_pv, is_sig in charged:
                    dr = deltaR(oe, op, c_eta, c_phi)
                    if dr >= cone or dr <= _ISO_DR_VETO_CH:
                        continue
                    if from_pv:
                        if pdg in _ISO_CH_HAD_IDS:
                            ch += c_pt
                            if not is_sig:
                                ch_clean += c_pt
                    elif dr > _ISO_PU_DR_VETO and c_pt > _ISO_PU_PT:
                        pu += c_pt
                        if not is_sig:
                            pu_clean += c_pt
                for n_eta, n_phi, n_pt, pdg in neutral:
                    dr = deltaR(oe, op, n_eta, n_phi)
                    if dr >= cone or dr <= _ISO_DR_VETO_NH:
                        continue
                    if pdg in _ISO_NH_IDS:
                        nh += n_pt
                    else:
                        ph += n_pt
                iso       = ch       + max(0., nh + ph - 0.5 * pu)
                iso_clean = ch_clean + max(0., nh + ph - 0.5 * pu_clean)
                o['%s_iso_ch_%s' % (name, rr)]       = ch
                o['%s_iso_ch_clean_%s' % (name, rr)] = ch_clean
                o['%s_iso_pu_%s' % (name, rr)]       = pu
                o['%s_iso_nh_%s' % (name, rr)]       = nh
                o['%s_iso_ph_%s' % (name, rr)]       = ph
                o['%s_iso_%s' % (name, rr)]          = iso
                o['%s_iso_clean_%s' % (name, rr)]    = iso_clean
                o['%s_reliso_%s' % (name, rr)]       = iso / opt if opt > 0 else NAN
                o['%s_reliso_clean_%s' % (name, rr)] = iso_clean / opt if opt > 0 else NAN


def _cos(a, b):
    na = float(np.sqrt(np.dot(a, a)))
    nb = float(np.sqrt(np.dot(b, b)))
    return float(np.dot(a, b)) / (na * nb) if na > 0 and nb > 0 else NAN


def _nu_solutions(visible, direction):
    '''(math1, math2, real) Bs four-momenta from the two neutrino solutions
    along `direction`, real = 1 when the discriminant was not negative.'''
    try:
        sols = reconstruct(visible, direction, m_parent=M_BS, clamp_negative_disc=True)
    except Exception:
        return None
    if len(sols) < 2:
        return None
    p1, p2 = np4(sols[0].p4_parent), np4(sols[1].p4_parent)
    real = float(abs(sols[0].parent_mass - M_BS) < 1e-3)
    return p1, p2, real
