'''
Bs -> Ds(-> phi(-> K K) pi) mu (nu) candidate for the R(Ds) ntuples.

Replaces the 2023 prototype. What it does, per candidate:

  1. Three sequential kinematic-fit schemes on the same four tracks
     (RDsKinVtxFitter), stored side by side so they can be compared on MC:
        cons   phi mass constraint on KK, then Ds mass constraint on phi + pi,
               then Ds + mu.
        phic   phi mass constraint on KK only: phi + pi, then KKpi + mu.
               phic_ds_mass is m(KKpi) with the phi constraint (the AN v4 /
               previous ntuples definition).
        free   no mass constraint anywhere: KK, KK + pi, KKpi + mu.
     cons and phic have the same phi fit (written once, see fit()). Every
     quantity that depends on the fits is written once per scheme, with the
     prefix cons_, phic_ or free_.
  2. The primary vertex: the one with the smallest 3D distance from the line
     through the Bs vertex along the visible (KKpi mu) momentum (AN Eq. 3;
     Bs vertex of the first scheme in SCHEMES whose Bs fit succeeded, the PV
     closest in z to the muon if all failed), refitted with the beamspot
     constraint and the four signal tracks removed (PrimaryVertex, shared
     with the RJpsi and dimuon ntuples). The same PV serves all schemes.
  3. Displacements and pointing of the phi, Ds and Bs vertices, the Ds vertex
     relative to the Bs vertex. The muon relative to the Bs flight axis (IP3D
     wrt PV and V_Bs, distances along and to the axis) and to the muon-free Ds
     trajectory (DOCA, its position along the Ds, DOCA significance), the
     tau-sensitive observables: V_Bs is the Ds + mu vertex for every
     candidate, which in the tau channels is not the Bs decay vertex.
  4. Bs momentum hypotheses (AN Sec. 5.4), per scheme: collinear, equal
     velocity along the Bs flight direction PV -> V_Bs (xyz_bs) and the two
     neutrino solutions along it (math1/2_bs); for each one q2, m_miss2,
     pT_miss, E*_mu, E#_mu, the Bs pT and the helicity angles (RDsKinematics).
     PV -> V_Ds is not the Bs direction, so no hypothesis is built along it.
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
from scipy.special import chdtrc   # chi2 survival function (what stats.chi2.sf calls)
from PhysicsTools.HeppyCore.utils.deltar import deltaR

import ROOT
ROOT.gSystem.Load('libBmmmAnalysis')
from ROOT import RDsKinVtxFitter
# RJpsiKinVtxFitter holds the IPTools wrappers (signedIP3D, signedDecayLength3D,
# jetTrackDistance) and, through its includes, VertexDistance3D / XY
from ROOT import RJpsiKinVtxFitter

from Bmmm.Analysis.utils import (
    convert_cov, is_pos_def, fix_track, track_params, track_with_params,
    track_with_cov, scale_cov, COV_NO_SCALE,
)
from Bmmm.Analysis.PrimaryVertex import (
    PrimaryVertexRefitMixin, build_beamspot_vertex, build_hybrid_pv,
    MU_TRK_DR_MATCH, MU_TRK_RELPT_MATCH,
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

# the order sets the scheme the PV is chosen from (choose_pv, pv_from_scheme)
SCHEMES = ('cons', 'phic', 'free')
SCHEME_VERTICES = ('phi', 'ds', 'bs')
# the vertex cuts: phic carries the phi + pi fit without the Ds constraint
# (the AN definition) and its phi fit is the one cons uses, so OR-ing phic and
# free covers all three schemes
CUT_SCHEMES = ('phic', 'free')
HYPOTHESES = ('coll', 'xyz_bs', 'math1_bs', 'math2_bs')
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
        cov          raw 5x5 covariance (numpy), is_cov_pos_def its check; both
                     computed on first use (most tracks of the event never
                     enter a fit)
    '''
    cov_scaler = None   # utils.CovScaler, installed once per job (--cov-scale)

    def __init__(self, obj, collection, index, f=1., z=NAN):
        self.obj        = obj
        self.collection = collection
        self.index      = index
        self.f          = f
        self.z          = z
        self._cov       = None
        self._is_cov_pos_def = None
        self.cov_fixed  = False
        self.cov_scale  = COV_NO_SCALE
        self._fit_track = None

    @property
    def cov(self):
        if self._cov is None:
            self._cov = convert_cov(self.obj.bestTrack().covariance())
        return self._cov

    @property
    def is_cov_pos_def(self):
        if self._is_cov_pos_def is None:
            self._is_cov_pos_def = bool(is_pos_def(self.cov))
        return self._is_cov_pos_def

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
        self._lv    = None   # ROOT LorentzVector, built on first p4()

    @property
    def cov(self):            return self.record.cov
    @property
    def is_cov_pos_def(self): return self.record.is_cov_pos_def

    # calibrated kinematics
    def p4(self):
        if self._lv is None:
            self._lv = LorentzVector(self._np4[1], self._np4[2], self._np4[3], self._np4[0])
        return self._lv
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


def _pos_def_3x3(m):
    '''Sylvester's criterion on a 3x3 symmetric matrix (callable m(i, j)).'''
    a, b, c = m(0, 0), m(0, 1), m(0, 2)
    d, e, f = m(1, 1), m(1, 2), m(2, 2)
    return a > 0 and a * d - b * b > 0 and \
        a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d) > 0


class FitInfo(object):
    '''What is read off one kinematic tree: the top particle four-momentum,
    the decay vertex and its quality, the refitted children. Read right after
    the fit that made the tree, before the tree enters another fit (see
    RDsKinVtxFitter).

    A fit returned as valid is still a failed fit here if its chi2 is negative
    or its vertex covariance is not positive definite: the former
    KinematicConstrainedVertexFitter produced both for ~20% of the
    unconstrained fits (RDsKinVtxFitter). The Kalman fit that replaced it
    produces none; the check stays as a safety net. status: 0 good, 1 no fit
    (failed or not run), 2 negative chi2, 3 covariance not positive definite.'''

    def __init__(self, tree):
        self.valid = _tree_valid(tree)
        self.status = 0 if self.valid else 1
        self.p4 = None
        self.vtx = None
        self.chi2 = self.ndof = self.prob = NAN
        self.children = []
        if not self.valid:
            return
        tree.movePointerToTheTop()
        self._vtx_ref = tree.currentDecayVertex()   # keeps the vertex alive
        self.vtx = self._vtx_ref.get()
        chi2 = self.vtx.chiSquared()
        if chi2 < 0:
            self.valid, self.status, self.vtx = False, 2, None
            return
        if not _pos_def_3x3(self.vtx.error().matrix()):
            self.valid, self.status, self.vtx = False, 3, None
            return
        self.p4  = _kin_np4(tree.currentParticle())
        self.chi2 = chi2
        self.ndof = self.vtx.degreesOfFreedom()
        self.prob = float(chdtrc(self.ndof, self.chi2)) if self.ndof > 0 else NAN
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
#####      PER-EVENT ARRAYS
##########################################################################################

def _delta_r(eta1, phi1, eta2, phi2):
    '''heppy deltaR, vectorised with the same operations (bit-identical).'''
    de = eta1 - eta2
    dp = phi1 - phi2
    dp = np.where(dp > np.pi, dp - 2 * np.pi, dp)
    dp = np.where(dp < -np.pi, dp + 2 * np.pi, dp)
    return np.sqrt(de * de + dp * dp)


def _seq_sum(x):
    '''Sum in input order, as the scalar loop it replaces (bit-identical).'''
    return float(np.cumsum(x)[-1]) if len(x) else 0.


class EventArrays(object):
    '''What every candidate of an event reads from the event collections, read
    once into numpy: vertex positions, the PF candidates of the isolation, the
    photons, and the PV track sets of the PV refit (PrimaryVertex cache).'''

    def __init__(self, vertices, pf, photons):
        pos = [v.position() for v in vertices]
        self.vtx_pos = np.array([[p.x(), p.y(), p.z()] for p in pos], dtype=float).reshape(-1, 3)
        self.pv_track_cache = {}

        self.has_pf = pf is not None
        ch, nt = [], []
        for c in (pf if pf is not None else ()):
            pdg = abs(c.pdgId())
            if pdg in _ISO_CH_ALL_IDS:
                ch.append((c.eta(), c.phi(), c.pt(), pdg, c.vz(), c.charge()))
            elif pdg in _ISO_NH_IDS or pdg in _ISO_PH_IDS:
                pt_ = c.pt()
                if pt_ > _ISO_NEUTRAL_PT:
                    nt.append((c.eta(), c.phi(), pt_, pdg))
        ch = np.array(ch, dtype=float).reshape(-1, 6)
        self.ch_eta, self.ch_phi, self.ch_pt, ch_pdg, self.ch_vz, self.ch_q = ch.T
        self.ch_had = np.isin(ch_pdg, _ISO_CH_HAD_IDS)
        nt = np.array(nt, dtype=float).reshape(-1, 4)
        self.nt_eta, self.nt_phi, self.nt_pt, nt_pdg = nt.T
        self.nt_nh = np.isin(nt_pdg, _ISO_NH_IDS)

        g = np.array([(c.energy(), c.px(), c.py(), c.pz(), c.eta(), c.phi(), c.pt())
                      for c in photons], dtype=float).reshape(-1, 7)
        self.g_p4 = g[:, :4]
        self.g_eta, self.g_phi, self.g_pt = g[:, 4], g[:, 5], g[:, 6]


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
        '''All fit schemes, both stages (fit_vertex_cut + fit_rest).'''
        self.fit_vertex_cut()
        self.fit_rest()

    def _add(self, info, trk, m, mconstr=-1.):
        '''info's tree + trk (RDsKinVtxFitter.AddTrack), read at once. A chain
        stops at a failed fit.'''
        if not info.valid:
            return FitInfo(None)
        return FitInfo(kinfit.AddTrack(info.tree, trk, m, mconstr))

    def fit_vertex_cut(self):
        '''The fits passes_vertex_cuts needs: phi and Ds of phic and free.
        Fills self.fits[scheme][vertex] with FitInfo; the Bs vertices and the
        cons Ds stay unfitted (status 1) until fit_rest.'''
        k1, k2, pi = (o.fit_track() for o in (self.k1, self.k2, self.pi))
        phi  = FitInfo(kinfit.FitTwoTracks(k1, k2, M_K, M_PHI))
        kk   = FitInfo(kinfit.FitTwoTracks(k1, k2, M_K, -1.))
        none = FitInfo(None)
        # cons refits phi identically (fit_rest), so the phic one stands for both
        self.fits = {
            'cons': dict(zip(SCHEME_VERTICES, (phi, none, none))),
            'phic': dict(zip(SCHEME_VERTICES, (phi, self._add(phi, pi, M_PI), none))),
            'free': dict(zip(SCHEME_VERTICES, (kk, self._add(kk, pi, M_PI), none))),
        }

    def fit_rest(self):
        '''The remaining fits, for the candidates that pass the vertex cuts:
        the three Bs vertices and the cons Ds.'''
        k1, k2, pi, mu = (o.fit_track() for o in (self.k1, self.k2, self.pi, self.mu))
        f = self.fits
        for s in ('phic', 'free'):
            f[s]['bs'] = self._add(f[s]['ds'], mu, M_MU)
        # cons: its own phi fit (a composite enters one further fit only)
        if f['cons']['phi'].valid:
            phi_cons = FitInfo(kinfit.FitTwoTracks(k1, k2, M_K, M_PHI))
            f['cons']['ds'] = self._add(phi_cons, pi, M_PI, M_DS)
            f['cons']['bs'] = self._add(f['cons']['ds'], mu, M_MU)

    def passes_vertex_cuts(self, cuts):
        for s in CUT_SCHEMES:
            f_phi, f_ds = self.fits[s]['phi'], self.fits[s]['ds']
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
            # 3D distance of each PV from the line through V_Bs along the
            # visible momentum (utils.compute_IP3D, vectorised)
            sv = bs_fit.vtx.position()
            a = np.array([sv.x(), sv.y(), sv.z()]) - self.evt.vtx_pos
            d = self.p4_kkpimu[1:]
            cross = np.stack([a[:, 1] * d[2] - d[1] * a[:, 2],
                              a[:, 2] * d[0] - d[2] * a[:, 0],
                              a[:, 0] * d[1] - d[0] * a[:, 1]], axis=1)
            ip = np.sqrt((cross * cross).sum(axis=1)) / np.sqrt(d.dot(d))
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
        '''lxy, lxyz (Measurement1D) and cos2d, cos3d of a vertex wrt the PV.
        Cached on the FitInfo: cons and phic use the same phi FitInfo.'''
        out = {}
        if not info.valid:
            return out
        if getattr(info, 'disp', None) is not None:
            return info.disp
        out['lxy']  = ROOT.VertexDistanceXY().distance(self.pv_bs, info.vtx.vertexState())
        out['lxyz'] = ROOT.VertexDistance3D().distance(self.pv_bs, info.vtx.vertexState())
        d = info.position() - self.pv_pos
        out['cos2d'] = _cos(d[:2], p4[1:3])
        out['cos3d'] = _cos(d, p4[1:])
        info.disp = out
        return out

    ##########################################################################
    #####      EVERYTHING ELSE
    ##########################################################################
    def compute(self, vertices, beamspot, pf, lost, photons, cuts, evt=None):
        '''After fit() and passes_vertex_cuts(): PV, displacements, Bs
        hypotheses, IPs, photon, isolation, reflections. evt: the EventArrays
        shared by the candidates of the event (built here if not given).'''
        self.evt = evt if evt is not None else EventArrays(vertices, pf, photons)
        self.pv_track_cache = self.evt.pv_track_cache   # PrimaryVertex refit
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
            o[pre + 'status'] = float(info.status)
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

        # the Bs flight direction, PV -> V_Bs (V_Bs = the Ds + mu vertex: in the
        # tau channels the mu comes from the tau decay, so it is not the Bs
        # decay vertex; the same definition is used for every candidate)
        directions = {}
        if bs.valid:
            directions['bs'] = bs.position() - self.pv_pos

        # ---- Bs momentum hypotheses -------------------------------------------
        hyps = {'coll': collinear_p4(visible)}
        if 'bs' in directions:
            d = directions['bs']
            hyps['xyz_bs'] = equal_velocity_p4(d, visible)
            sols = _nu_solutions(visible, d)
            if sols is not None:
                hyps['math1_bs'], hyps['math2_bs'] = sols[0], sols[1]
                o[s + '_math_real_bs'] = sols[2]

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

        # ---- muon vs the Bs flight axis (as JpsiChargedCandidate.compute_ip and
        #      compute_jet_track_distance, with the single Bs vertex) ------------
        mu_trk = self.mu.fit_track()
        if 'bs' in directions:
            d = directions['bs']
            refs = {'pv': self.pv_bs, 'sv': bs.reco_vertex()}
            jtd = None
            for ref, vtx in refs.items():
                # lifetime-signed IP3D; wrt V_Bs it is small by construction,
                # the mu being in that fit
                ip = iptools.signedIP3D(mu_trk, d[0], d[1], d[2], vtx)
                o['%s_mu_ip3d_bs_%s' % (s, ref)]     = ip.value()
                o['%s_mu_ip3d_bs_%s_err' % (s, ref)] = ip.error()
                o['%s_mu_ip3d_bs_%s_sig' % (s, ref)] = ip.significance()
                jtd = iptools.jetTrackDistance(mu_trk, d[0], d[1], d[2], vtx)
                o['%s_mu_dist_along_bs_dir_%s' % (s, ref)] = jtd.first
            # line-to-line, independent of the anchor (IPTools returns it negative)
            o[s + '_mu_dist_to_bs_dir'] = abs(jtd.second.value())

        # ---- muon vs the Ds trajectory (V_Ds along the muon-free Ds momentum),
        #      which passes through the true Bs decay vertex: the mu meets it
        #      there in the mu channels and misses it by the tau IP in the tau
        #      ones ----------------------------------------------------------------
        if ds.valid:
            p3 = ds.p4[1:]
            jtd = iptools.jetTrackDistance(mu_trk, p3[0], p3[1], p3[2], ds.reco_vertex())
            o[s + '_mu_doca_ds'] = abs(jtd.second.value())
            # position of the closest approach along the Ds momentum, from V_Ds:
            # negative upstream, -(Ds flight length) in the mu channels.
            # IPTools' .first is minus that.
            o[s + '_mu_poca_along_ds'] = -jtd.first
            # IPTools has no error on the line-to-line distance; the Ds + mu
            # vertex fit (1 ndof) has chi2 = (DOCA / sigma)^2 to first order,
            # with the full covariances
            if bs.valid and bs.chi2 >= 0:
                o[s + '_mu_doca_ds_sig'] = float(np.sqrt(bs.chi2))

        # ---- soft photon -------------------------------------------------------
        self._photon(s, photons, p_ds, p_mu, directions, cuts)

    def _photon(self, s, photons, p_ds, p_mu, directions, cuts):
        o = self.out
        ev = self.evt
        # m(Ds gamma) of every photon, as RDsKinematics.mass
        p = p_ds + ev.g_p4
        m2 = p[:, 0] * p[:, 0] - p[:, 1] * p[:, 1] - p[:, 2] * p[:, 2] - p[:, 3] * p[:, 3]
        m_dsg_all = np.where(m2 >= 0, np.sqrt(np.abs(m2)), -np.sqrt(np.abs(m2)))
        in_win = np.nonzero(np.abs(m_dsg_all - M_DSST) < cuts['gamma_window'])[0]
        o[s + '_gamma_n'] = len(in_win)
        e_coll = e_gamma_collinear(p_ds)
        o[s + '_gamma_e_coll'] = e_coll
        if not len(in_win):
            return
        dr = _delta_r(eta(p_ds), phi(p_ds), ev.g_eta[in_win], ev.g_phi[in_win])
        ib = int(np.argmin(dr))              # the first of the closest, as the loop
        ig = in_win[ib]
        best_dr, m_dsg, pg = float(dr[ib]), float(m_dsg_all[ig]), ev.g_p4[ig]
        g_pt = float(ev.g_pt[ig])
        o[s + '_gamma_pt']   = g_pt
        o[s + '_gamma_eta']  = float(ev.g_eta[ig])
        o[s + '_gamma_phi']  = float(ev.g_phi[ig])
        o[s + '_gamma_dr']   = best_dr
        o[s + '_gamma_m_ds_gamma']  = m_dsg
        o[s + '_gamma_dm']   = m_dsg - mass(p_ds)
        o[s + '_gamma_e_coll_over_pt'] = e_coll / g_pt if g_pt > 0 else NAN
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
        ev = self.evt
        if not ev.has_pf:
            return
        # charged PF candidates: associated to the vertex closest in z, with the
        # refitted PV in place of the chosen one; from_pv if that is the PV
        assoc_z = ev.vtx_pos[:, 2].copy()
        has_pv = 0 <= self.pv_idx < len(assoc_z)
        if has_pv:
            assoc_z[self.pv_idx] = self.pv_bs.position().z()
        if has_pv and len(ev.ch_vz):
            ibest = np.argmin(np.abs(assoc_z[None, :] - ev.ch_vz[:, None]), axis=1)
            from_pv = ibest == self.pv_idx
        else:
            from_pv = np.zeros(len(ev.ch_vz), dtype=bool)

        # PrimaryVertexRefitMixin.is_signal_muon_cand, vectorised
        is_sig = np.zeros(len(ev.ch_vz), dtype=bool)
        for so in self.signal_objects:
            so_pt = so.pt()
            is_sig |= ((ev.ch_q == so.charge()) &
                       (_delta_r(so.eta(), so.phi(), ev.ch_eta, ev.ch_phi) < MU_TRK_DR_MATCH) &
                       (np.abs(ev.ch_pt - so_pt) < MU_TRK_RELPT_MATCH * so_pt))

        pv_had = from_pv & ev.ch_had
        pu_cand = ~from_pv & (ev.ch_pt > _ISO_PU_PT)

        objects = (('mu', self.mu.eta(), self.mu.phi(), self.mu.pt()),
                   ('ds', eta(self.p4_kkpi), phi(self.p4_kkpi), pt(self.p4_kkpi)),
                   ('b',  eta(self.p4_kkpimu), phi(self.p4_kkpimu), pt(self.p4_kkpimu)))
        o = self.out
        for name, oe, op, opt in objects:
            dr_ch = _delta_r(oe, op, ev.ch_eta, ev.ch_phi)
            dr_nt = _delta_r(oe, op, ev.nt_eta, ev.nt_phi)
            for cone in _ISO_CONES:
                rr = '%02d' % int(round(cone * 10))
                in_ch = (dr_ch < cone) & (dr_ch > _ISO_DR_VETO_CH)
                m_ch = in_ch & pv_had
                m_pu = in_ch & pu_cand & (dr_ch > _ISO_PU_DR_VETO)
                in_nt = (dr_nt < cone) & (dr_nt > _ISO_DR_VETO_NH)
                ch       = _seq_sum(ev.ch_pt[m_ch])
                ch_clean = _seq_sum(ev.ch_pt[m_ch & ~is_sig])
                pu       = _seq_sum(ev.ch_pt[m_pu])
                pu_clean = _seq_sum(ev.ch_pt[m_pu & ~is_sig])
                nh       = _seq_sum(ev.nt_pt[in_nt & ev.nt_nh])
                ph       = _seq_sum(ev.nt_pt[in_nt & ~ev.nt_nh])
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
