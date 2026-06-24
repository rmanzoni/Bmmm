import numpy as np
from scipy import stats
from itertools import combinations
from PhysicsTools.HeppyCore.utils.deltar import deltaR
from Bmmm.Analysis.utils import masses, is_pos_def, convert_cov, compute_IP3D

import ROOT
ROOT.gSystem.Load('libBmmmAnalysis')
# Tau3MuKinVtxFitter.h transitively #includes VertexDistance3D / VertexDistanceXY
# / IPTools (via RJpsiKinVtxFitter.h), so importing it also makes
# ROOT.VertexDistance3D / ROOT.VertexDistanceXY available (same dirty trick the
# rjpsi candidate relies on).
from ROOT import Tau3MuKinVtxFitter

# single fitter instance shared by all candidates
kinfit = Tau3MuKinVtxFitter()

# ROOT template instantiations, hoisted out of the per-candidate hot loop
Vector3D      = ROOT.Math.DisplacementVector3D(
    'ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')
LorentzVector = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzE4D<double>')

A_PDGID   = 9900015          # the long-lived scalar a (a -> mu mu)
TAU_PDGID = 15

# ----- signal-muon <-> packed-candidate matching (proximity, cross-collection) --
_MU_TRK_DR_MATCH    = 0.01
_MU_TRK_RELPT_MATCH = 0.05

# ----- PV-finder track filter (offlinePrimaryVertices TkFilterParameters) -------
_PV_TRK_MAX_NORM_CHI2  = 10.0
_PV_TRK_MIN_PIX_LAYERS = 2
_PV_TRK_MIN_TRK_LAYERS = 5
_PV_TRK_MAX_D0_SIG     = 4.0
_PV_TRK_MAX_D0_ERR     = 1.0
_PV_TRK_MAX_DZ_ERR     = 1.0
_PV_TRK_MIN_PT         = 0.0
_PV_TRK_MAX_ETA        = 2.4

# ----- PV refit fitter config + vertex acceptance (WithBS vertexCollections) ----
_PV_AVF_CHI2CUTOFF   = 2.5
_PV_MIN_NDOF         = 2.0
_PV_MAX_DIST_TO_BEAM = 1.0

# ----- PV refit track-set selection ---------------------------------------------
_PV_USED_IN_FIT        = 3
_PV_REFIT_MIN_TRK      = 2
_PV_REFIT_MAX_DZ_TO_PV = 0.1


class Tau3MuCandidate(ROOT.reco.CompositeCandidate):
    '''
    Displaced  tau -> 3mu  candidate for  Ds -> tau nu , tau -> mu a , a -> mu mu
    (a long-lived, pdgId 9900015). The whole tau decay is visible, so the object
    itself (a reco::CompositeCandidate) is the full 3-muon = tau system.

    Following the rjpsi philosophy the three muons are named, in the ntuple,
        mu1, mu2 : the displaced opposite-sign pair (the a -> mu mu muons, the
                   pair that fires HLT_DoubleMu4_3_LowMass), pt-sorted
        mu3      : the bachelor muon from tau -> mu a
    Internally these are self.a_muons (== [mu1, mu2]) and self.mu (== mu3); the
    a is stored as a reco::CompositeCandidate (self.a).

    Vertex / IP / displacement quantities are NOT computed in __init__: call
    compute_vtx_quantities(vertices, beamspot[, pf, lost]) for that, and
    compute_pf_cone(pf) for the R=0.4 PF-candidate cone.
    '''
    def __init__(self, a_muons, mu3):

        super().__init__()

        # covariance check, memoized (muons are shared across candidates)
        for imu in a_muons + [mu3]:
            if not hasattr(imu, 'cov'):
                imu.cov            = convert_cov(imu.bestTrack().covariance())
                imu.is_cov_pos_def = is_pos_def(imu.cov)

        self.muons   = sorted(a_muons + [mu3], key=lambda x: x.pt(), reverse=True)
        self.a_muons = sorted(a_muons        , key=lambda x: x.pt(), reverse=True)
        self.mu      = mu3

        # the displaced OS pair = the a candidate
        self.a = ROOT.reco.CompositeCandidate()
        self.a.addDaughter(self.a_muons[0])
        self.a.addDaughter(self.a_muons[1])
        self.a.setP4(self.a.daughter(0).p4() + self.a.daughter(1).p4())
        self.a.setCharge(int(self.a_muons[0].charge() + self.a_muons[1].charge()))
        self.a.setPdgId(A_PDGID)

        # the full 3-muon = tau system is the object itself
        self.setP4(self.a.p4() + self.mu.p4())
        self.setCharge(int(sum(imu.charge() for imu in self.muons)))
        self.setPdgId(int(TAU_PDGID * np.sign(self.mu.charge())))

    ##########################################################################
    #####      VERTEXING  (sequential:  a -> mu mu , then tau -> a + mu)
    ##########################################################################
    def compute_vtx_quantities(self, vertices, beamspot, pf=None, lost=None):
        '''
        Sequential vertex fit of the displaced 3mu system:

          1. fit the displaced OS pair (mu1, mu2) to the  a  vertex. NO mass
             constraint -- m(mu mu) = m(a) is the search variable.
          2. fit the MOTHER of that pair (the refitted a, with its covariance)
             together with the bachelor muon to a DIFFERENT, upstream vertex: the
             tau vertex. This is the proper hierarchical fit (Tau3MuKinVtxFitter.
             FitMotherPlusTrack), not a flat three-track fit.

        Then choose the PV, refit it (beamspot-constrained, the three signal muons
        removed), and compute, against the refit PV (self.pv_bs):
          - tau vertex  : sv_*    displacement / pointing (full 3mu momentum)
          - a   vertex  : a_*     displacement / pointing (a momentum), wrt PV
          - a   vertex  : a_wrt_tau_*  displacement of a from the tau vertex
                          (the a flight, the smoking gun of the long lifetime)
          - per-muon signed 3D IP wrt the PV, plus the bachelor IP wrt the a vtx.
        '''
        # ----- stage 1: the displaced OS pair (no mass constraint) ----------
        self.a_vertex_tree = self.fit_vertex(self.a_muons)
        self.a_good_vtx    = self.is_good_vtx(self.a_vertex_tree)

        # ----- stage 2: sequential  a + bachelor -> tau vertex --------------
        if self.a_good_vtx:
            self.vertex_tree = kinfit.FitMotherPlusTrack(
                self.a_vertex_tree, self.mu.bestTrack(), masses['mu'])
        else:
            self.vertex_tree = ROOT.RefCountedKinematicTree()
        self.good_vtx = self.is_good_vtx(self.vertex_tree)

        # ----- choose the primary vertex ------------------------------------
        # PV = smallest 3D IP wrt the tau flight line (tau SV along the 3mu p);
        # fall back to closest-in-dz to the leading muon if the tau fit failed.
        if self.good_vtx:
            self.vertex_tree.movePointerToTheTop()
            sv = self.vertex_tree.currentDecayVertex().get()
            pv_idx, ip3d_min = -1, np.inf
            for idx, ivtx in enumerate(vertices):
                ip3d = compute_IP3D(ivtx, sv.position(), self.p4().Vect())
                if ip3d < ip3d_min:
                    pv_idx, ip3d_min = idx, ip3d
            self.pv_idx = pv_idx
            self.pv     = vertices[pv_idx]
        else:
            self.pv_idx = min(
                range(len(vertices)),
                key=lambda i: abs(self.muons[0].bestTrack().dz(vertices[i].position())),
            )
            self.pv = vertices[self.pv_idx]

        # ----- beamspot as a reco::Vertex at the PV z position --------------
        self.bs = self.build_beamspot_vertex(beamspot, self.pv.z())

        # ----- per-candidate PV reference (refit, signal muons out) ---------
        self.refit_primary_vertex(beamspot, pf, lost, vertices)
        self.pv_bs = self.pv_refit if self.pv_refit_valid \
                     else self.build_hybrid_pv(self.bs, self.pv)

        # ----- refitted momenta ---------------------------------------------
        self.compute_refit_p4()

        # ----- displacement / pointing-angle quantities --------------------
        self.compute_displacement(self.vertex_tree  , self.good_vtx  , self  , self.pv_bs, prefix='sv_')
        self.compute_displacement(self.a_vertex_tree, self.a_good_vtx, self.a, self.pv_bs, prefix='a_' )

        # a flight FROM the tau vertex (a displaced from its own production point)
        if self.good_vtx:
            tau_reco_vtx = self.kin_to_reco_vertex(self.sv_vtx)
            self.compute_displacement(self.a_vertex_tree, self.a_good_vtx, self.a,
                                      tau_reco_vtx, prefix='a_wrt_tau_')
        else:
            self.nan_displacement('a_wrt_tau_')

        # ----- impact parameters --------------------------------------------
        self.compute_ip()

    ##########################################################################
    @staticmethod
    def fit_vertex(muons):
        '''Common-vertex fit of N muons (muon mass hypothesis each). No mass
        constraint -- used here for the displaced OS pair, whose invariant mass
        is the search variable. Returns the kinematic tree (empty on failure).'''
        tracks   = ROOT.std.vector('reco::Track')()
        mass_hyp = ROOT.std.vector('double')()
        for imu in muons:
            tracks.push_back(imu.bestTrack())
            mass_hyp.push_back(masses['mu'])
        return kinfit.Fit(tracks, mass_hyp)

    @staticmethod
    def is_good_vtx(vertex_tree):
        if not vertex_tree:
            return False
        return (not vertex_tree.isEmpty()) and vertex_tree.isValid()

    def compute_refit_p4(self):
        '''Refitted momenta:
            self.a_rfp4   : the a from the stage-1 OS-pair fit
            self.rfp4     : the tau from the stage-2 sequential fit
            self.a_muons[i].rfp4 : the two refitted a-muon momenta
        All None if the relevant fit failed.'''
        self.a_rfp4 = None
        self.rfp4   = None
        for imu in self.muons:
            imu.rfp4 = None

        if self.a_good_vtx:
            self.a_vertex_tree.movePointerToTheTop()
            self.a_rfp4 = self.refitted_p4(self.a_vertex_tree.currentParticle())
            self.a_vertex_tree.movePointerToTheFirstChild()
            self.a_muons[0].rfp4 = self.refitted_p4(self.a_vertex_tree.currentParticle())
            self.a_vertex_tree.movePointerToTheNextChild()
            self.a_muons[1].rfp4 = self.refitted_p4(self.a_vertex_tree.currentParticle())

        if self.good_vtx:
            self.vertex_tree.movePointerToTheTop()
            self.rfp4 = self.refitted_p4(self.vertex_tree.currentParticle())

    ##########################################################################
    def nan_displacement(self, prefix):
        setattr(self, prefix + 'vtx', None)
        for q in ['vtx_chi2', 'vtx_ndof', 'vtx_prob', 'lxy', 'lxyz', 'cos2d', 'cos3d']:
            setattr(self, prefix + q, np.nan)

    def compute_displacement(self, vertex_tree, good_vtx, cand, ref_vtx, prefix=''):
        '''
        For one fitted vertex, store (with the given prefix), all wrt ref_vtx:
            {prefix}vtx, vtx_chi2, vtx_ndof, vtx_prob
            {prefix}lxy, lxyz          (Measurement1D: .value/.error/.significance)
            {prefix}cos2d, cos3d       (pointing angle of `cand` momentum)
        `cand` supplies px/py/pz for the pointing angle (the tau for the tau
        vertex, the a for the a vertex). `ref_vtx` is a reco::Vertex.
        '''
        if not good_vtx:
            self.nan_displacement(prefix)
            return

        vertex_tree.movePointerToTheTop()
        vtx = vertex_tree.currentDecayVertex().get()
        setattr(self, prefix + 'vtx', vtx)

        chi2 = vtx.chiSquared()
        ndof = vtx.degreesOfFreedom()
        setattr(self, prefix + 'vtx_chi2', chi2)
        setattr(self, prefix + 'vtx_ndof', ndof)
        setattr(self, prefix + 'vtx_prob', stats.chi2.sf(chi2, ndof))

        lxy  = ROOT.VertexDistanceXY().distance(ref_vtx, vtx.vertexState())
        lxyz = ROOT.VertexDistance3D().distance(ref_vtx, vtx.vertexState())
        setattr(self, prefix + 'lxy' , lxy )
        setattr(self, prefix + 'lxyz', lxyz)

        vect_lxy = Vector3D(vtx.position().x() - ref_vtx.position().x(),
                            vtx.position().y() - ref_vtx.position().y(), 0.)
        vect_pt  = Vector3D(cand.px(), cand.py(), 0.)
        setattr(self, prefix + 'cos2d',
                vect_pt.Dot(vect_lxy) / (vect_pt.R() * vect_lxy.R()) if vect_lxy.R() > 0. else np.nan)

        vect_lxyz = Vector3D(vtx.position().x() - ref_vtx.position().x(),
                             vtx.position().y() - ref_vtx.position().y(),
                             vtx.position().z() - ref_vtx.position().z())
        vect_p    = Vector3D(cand.px(), cand.py(), cand.pz())
        setattr(self, prefix + 'cos3d',
                vect_p.Dot(vect_lxyz) / (vect_p.R() * vect_lxyz.R()) if vect_lxyz.R() > 0. else np.nan)

    def compute_ip(self):
        '''
        Signed 3D impact parameters (IPTools), lifetime-signed along the tau
        flight direction (PV -> tau vertex):
          - every muon wrt the PV:        mu{1,2,3}_ip3d[_err|_sig]  (on the muon)
          - the bachelor mu3 wrt the a vertex (mu3 is NOT in the a fit):
                                          mu3_ip3d_a[_err|_sig]      (on the cand)
        Filled only when the relevant vertex fit succeeded; NaN otherwise.
        '''
        for imu in self.muons:
            imu.ip3d = imu.ip3d_err = imu.ip3d_sig = np.nan
        self.mu3_ip3d_a = self.mu3_ip3d_a_err = self.mu3_ip3d_a_sig = np.nan

        # tau flight direction, used to lifetime-sign every IP
        if self.good_vtx:
            self.Bdirection = self.flight_direction(self.sv_vtx, self.pv_bs)
            dx, dy, dz = self.Bdirection.x(), self.Bdirection.y(), self.Bdirection.z()
            for imu in self.muons:
                ip = kinfit.signedIP3D(imu.bestTrack(), dx, dy, dz, self.pv_bs)
                imu.ip3d     = ip.value()
                imu.ip3d_err = ip.error()
                imu.ip3d_sig = ip.significance()
        else:
            self.Bdirection = None

        # bachelor mu3 wrt the a vertex (signed along the a flight if available,
        # else along the bachelor direction)
        if self.a_good_vtx:
            a_reco_vtx = self.kin_to_reco_vertex(self.a_vtx)
            if self.Bdirection is not None:
                dx, dy, dz = self.Bdirection.x(), self.Bdirection.y(), self.Bdirection.z()
            else:
                p3 = self.mu.p4().Vect()
                dx, dy, dz = p3.x(), p3.y(), p3.z()
            ip = kinfit.signedIP3D(self.mu.bestTrack(), dx, dy, dz, a_reco_vtx)
            self.mu3_ip3d_a     = ip.value()
            self.mu3_ip3d_a_err = ip.error()
            self.mu3_ip3d_a_sig = ip.significance()

    ##########################################################################
    #####      PF-CANDIDATE CONE  (R = 0.4 around the 3mu / tau direction)
    ##########################################################################
    def compute_pf_cone(self, pf, cone_dr=0.4, min_pt=0.):
        '''
        Collect ALL PF candidates within dR < cone_dr of the 3mu (tau) direction
        and store, as parallel python lists (one entry per PF candidate), the p4,
        impact parameters and pdgId -- consumed by the inspector as jagged ntuple
        branches (pf_*). Impact parameters are taken wrt the refit PV (self.pv_bs):
          pf_pt, pf_eta, pf_phi, pf_mass, pf_energy   : the p4
          pf_puppiweight                              : PUPPI weight
          pf_pdgid, pf_charge                         : identity
          pf_dr                                       : dR to the tau axis
          pf_dxy, pf_dxy_err, pf_dz, pf_dz_err        : packed-candidate IPs wrt PV
          pf_ip3d, pf_ip3d_sig                        : signed 3D IP wrt PV
                                                        (NaN without track details)
          pf_is_signal                                : 1 if it is one of the 3
                                                        signal muons (so it can be
                                                        removed downstream)
        No-op-safe: empty lists if pf is None.
        '''
        keys = ['pt', 'eta', 'phi', 'mass', 'energy', 'puppiweight', 'pdgid', 'charge', 'dr',
                'dxy', 'dxy_err', 'dz', 'dz_err', 'ip3d', 'ip3d_sig', 'is_signal']
        for k in keys:
            setattr(self, 'pf_' + k, [])

        if pf is None:
            self.pf_n = 0
            return

        eta0, phi0 = self.eta(), self.phi()
        pv_point   = self.pv_bs.position()

        # lifetime sign direction: the tau flight if available, else the 3mu p
        if getattr(self, 'Bdirection', None) is not None:
            dx, dy, dz = self.Bdirection.x(), self.Bdirection.y(), self.Bdirection.z()
        else:
            p3 = self.p4().Vect()
            dx, dy, dz = p3.x(), p3.y(), p3.z()

        for cand in pf:
            if cand.pt() < min_pt:
                continue
            dr = deltaR(eta0, phi0, cand.eta(), cand.phi())
            if dr >= cone_dr:
                continue

            self.pf_pt    .append(cand.pt())
            self.pf_eta   .append(cand.eta())
            self.pf_phi   .append(cand.phi())
            self.pf_mass  .append(cand.mass())
            self.pf_energy.append(cand.energy())
            self.pf_puppiweight.append(cand.puppiWeight())
            self.pf_pdgid .append(cand.pdgId())
            self.pf_charge.append(cand.charge())
            self.pf_dr    .append(dr)
            self.pf_is_signal.append(int(self.is_signal_muon_cand(cand)))

            # impact parameters: packed-candidate dxy/dz wrt the PV, and the
            # signed 3D IP from a full helical extrapolation when the candidate
            # carries track details (charged candidates).
            if cand.charge() != 0 and cand.hasTrackDetails():
                self.pf_dxy    .append(cand.dxy(pv_point))
                self.pf_dxy_err.append(cand.dxyError())
                self.pf_dz     .append(cand.dz(pv_point))
                self.pf_dz_err .append(cand.dzError())
                try:
                    ip = kinfit.signedIP3D(cand.pseudoTrack(), dx, dy, dz, self.pv_bs)
                    self.pf_ip3d    .append(ip.value())
                    self.pf_ip3d_sig.append(ip.significance())
                except Exception:
                    self.pf_ip3d    .append(np.nan)
                    self.pf_ip3d_sig.append(np.nan)
            else:
                self.pf_dxy    .append(np.nan)
                self.pf_dxy_err.append(np.nan)
                self.pf_dz     .append(np.nan)
                self.pf_dz_err .append(np.nan)
                self.pf_ip3d    .append(np.nan)
                self.pf_ip3d_sig.append(np.nan)

        self.pf_n = len(self.pf_pt)

    def is_signal_muon_cand(self, cand,
                            dr_max=_MU_TRK_DR_MATCH, rel_pt_max=_MU_TRK_RELPT_MATCH):
        '''True if the PF candidate is one of the three signal muons (proximity
        match by charge / dR / relative pt, since the muon and the PF candidate
        are in different collections).'''
        for imu in self.muons:
            if cand.charge() != imu.charge():
                continue
            if (deltaR(imu.eta(), imu.phi(), cand.eta(), cand.phi()) < dr_max and
                    abs(cand.pt() - imu.pt()) < rel_pt_max * imu.pt()):
                return True
        return False

    ##########################################################################
    #####      PRIMARY-VERTEX REFIT  (AVF, beamspot-constrained, muons out)
    ##########################################################################
    @staticmethod
    def passes_pv_track_filter(cand, trk, beamspot):
        '''Approximate offlinePrimaryVertices TkFilterParameters on a packed/lost
        candidate, so the PV refit is fed the offline-quality track set.'''
        if abs(trk.eta()) > _PV_TRK_MAX_ETA:
            return False
        if trk.normalizedChi2() > _PV_TRK_MAX_NORM_CHI2:
            return False
        hp = trk.hitPattern()
        if hp.pixelLayersWithMeasurement()   < _PV_TRK_MIN_PIX_LAYERS:
            return False
        if hp.trackerLayersWithMeasurement() < _PV_TRK_MIN_TRK_LAYERS:
            return False
        if cand.pt() <= _PV_TRK_MIN_PT:
            return False
        d0_err = cand.dxyError()
        dz_err = cand.dzError()
        if d0_err > _PV_TRK_MAX_D0_ERR or dz_err > _PV_TRK_MAX_DZ_ERR:
            return False
        d0 = cand.dxy(beamspot.position(cand.vz()))
        if d0_err > 0. and abs(d0) / d0_err > _PV_TRK_MAX_D0_SIG:
            return False
        return True

    def refit_primary_vertex(self, beamspot, pf, lost, vertices):
        '''Per-candidate beamspot-constrained AVF refit of the chosen PV with the
        three signal muons removed (Tau3MuKinVtxFitter.refitPVRemovingTracks,
        inherited from RJpsiKinVtxFitter). Identical strategy to the rjpsi skim:
        the PV track set is rebuilt in the loop from packed (pf) + lost (lost)
        candidates -- primary via the offline DA fit-track assignment (fromPV ==
        PVUsedInFit), with a closest-z + offline-quality-filter fallback.
        Sets self.pv_refit / self.pv_refit_valid; on any failure the caller falls
        back to the hybrid PV.'''
        self.pv_refit       = None
        self.pv_refit_valid = False

        if pf is None:
            return

        try:
            vtxs = list(vertices)
            if not vtxs:
                return

            mu_tracks = ROOT.std.vector('reco::Track')()
            for imu in self.muons:
                mu_tracks.push_back(imu.bestTrack())

            def _refit(selector, apply_filter):
                pv_tracks = ROOT.std.vector('reco::Track')()
                for coll in (pf, lost):
                    if coll is None:
                        continue
                    for cand in coll:
                        if not cand.hasTrackDetails():
                            continue
                        trk = cand.pseudoTrack()
                        if apply_filter and not self.passes_pv_track_filter(cand, trk, beamspot):
                            continue
                        if selector(cand):
                            pv_tracks.push_back(trk)
                if pv_tracks.size() < _PV_REFIT_MIN_TRK:
                    return None
                v = kinfit.refitPVRemovingTracks(
                    pv_tracks, mu_tracks, beamspot,
                    _MU_TRK_DR_MATCH, _MU_TRK_RELPT_MATCH,
                    _PV_AVF_CHI2CUTOFF, _PV_MIN_NDOF, _PV_MAX_DIST_TO_BEAM,
                )
                return v if v.isValid() else None

            refit = _refit(lambda cand: cand.fromPV(self.pv_idx) == _PV_USED_IN_FIT,
                           apply_filter=False)

            if refit is None or abs(refit.z() - self.pv.z()) > _PV_REFIT_MAX_DZ_TO_PV:
                vtx_z = [vv.position().z() for vv in vtxs]
                def _closest_is_pv(cand):
                    vz = cand.vz()
                    return min(range(len(vtx_z)),
                               key=lambda i: abs(vtx_z[i] - vz)) == self.pv_idx
                refit = _refit(_closest_is_pv, apply_filter=True)

            if refit is not None:
                self.pv_refit       = refit
                self.pv_refit_valid = True
        except Exception:
            self.pv_refit_valid = False

    ##########################################################################
    #####      static geometry helpers (shared idioms with the rjpsi candidate)
    ##########################################################################
    @staticmethod
    def build_beamspot_vertex(beamspot, z):
        bs_point = ROOT.reco.Vertex.Point(beamspot.x(z), beamspot.y(z), beamspot.z0())
        return ROOT.reco.Vertex(bs_point, beamspot.covariance3D(), 0., 0., 3)

    @staticmethod
    def build_hybrid_pv(bs, pv):
        return ROOT.reco.Vertex(
            ROOT.reco.Vertex.Point(bs.position().x(), bs.position().y(), pv.position().z()),
            pv.error(), pv.chi2(), pv.ndof(), pv.tracksSize())

    @staticmethod
    def kin_to_reco_vertex(kin_vtx, ndim=2):
        pos   = kin_vtx.position()
        point = ROOT.reco.Vertex.Point(pos.x(), pos.y(), pos.z())
        return ROOT.reco.Vertex(point, kin_vtx.error().matrix(),
                                kin_vtx.chiSquared(), kin_vtx.degreesOfFreedom(), ndim)

    @staticmethod
    def flight_direction(vtx, pv_ref):
        sv = vtx.position()
        return ROOT.Math.XYZVector(sv.x() - pv_ref.position().x(),
                                   sv.y() - pv_ref.position().y(),
                                   sv.z() - pv_ref.position().z())

    @staticmethod
    def refitted_p4(kin_particle):
        v  = kin_particle.currentState().kinematicParameters().vector()
        px, py, pz, m = v.At(3), v.At(4), v.At(5), v.At(6)
        e  = np.sqrt(px*px + py*py + pz*pz + m*m)
        return LorentzVector(px, py, pz, e)

    ##########################################################################
    #####      KINEMATICS (pt-sorted muons: muons[0], muons[1], muons[2])
    ##########################################################################
    def r(self):
        '''Cone radius: max distance between the 3mu direction and a muon.'''
        return max(deltaR(self.eta(), self.phi(), imu.eta(), imu.phi()) for imu in self.muons)
    def max_dr(self):
        return max(deltaR(imu, jmu) for imu, jmu in combinations(self.muons, 2))
    def dr12(self):
        return deltaR(self.a_muons[0], self.a_muons[1])
    def dr_a_mu(self):
        '''dR between the a direction and the bachelor muon.'''
        return deltaR(self.a.eta(), self.a.phi(), self.mu.eta(), self.mu.phi())
