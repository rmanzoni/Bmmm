import numpy as np
from scipy import stats
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from itertools import product, combinations
from Bmmm.Analysis.utils import convert_cov, is_pos_def, track_with_cov, make_cov_corrector
from Bmmm.Analysis.PrimaryVertex import (
    PrimaryVertexRefitMixin, build_beamspot_vertex, build_hybrid_pv,
)

import ROOT
ROOT.gSystem.Load('libBmmmAnalysis')
from ROOT import KVFitter # VertexDistance3D is contained here, dirt trick!!

# make these available everywhere in here
global vtxfit
vtxfit = KVFitter()
global tofit
tofit = ROOT.std.vector('reco::Track')()

# ROOT template instantiation, hoisted out of the per-candidate hot loop
# (same one JpsiChargedCandidate uses)
Vector3D = ROOT.Math.DisplacementVector3D(
    'ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')

ROLES = ('mu1', 'mu2')


class Candidate(PrimaryVertexRefitMixin):
    '''
    Dimuon candidate. mu1 / mu2 = the higher / lower pT muon.
    MISSING: use the post fit muon momenta
    '''

    # covflow, one utils.CovFlowCorrector per ROLE: the Run 3 flows are trained
    # separately on mu1 and on mu2 of this very ntuple (covflow.json cov_prefix
    # mu1_cov_ / mu2_cov_), so the leading muon of a pair is corrected with the
    # mu1 flow and the trailing one with the mu2 flow. A muon can be leading in
    # one pair and trailing in another: the two corrections are memoized apart
    # (cov_corr_mu1 / cov_corr_mu2, see attr_suffix). Empty = no correction, and
    # the candidate is bit-for-bit the one without this feature.
    # Installed once per job with set_cov_correctors (--covflow-mu1/-mu2).
    cov_correctors = {}

    @classmethod
    def set_cov_correctors(cls, specs):
        '''specs: {'mu1': run directory, 'mu2': run directory}; empty strings or
        a missing role leave that role uncorrected (refused: half a correction
        is not a validation of anything).'''
        given = {r: s for r, s in specs.items() if s}
        if not given:
            cls.cov_correctors = {}
            return cls.cov_correctors
        if set(given) != set(ROLES):
            raise RuntimeError('covflow needs a flow for both roles, got %s' % sorted(given))
        cls.cov_correctors = {r: make_cov_corrector(given[r], attr_suffix='_' + r, role=r)
                              for r in ROLES}
        return cls.cov_correctors

    @classmethod
    def prime_cov_correctors(cls, muons, event=None):
        '''Batched covflow for the event, one torch call per role, BEFORE any
        candidate is built. `muons` sorted by decreasing pT, as the inspector
        pairs them with combinations(): muons[i] leads every pair with j > i,
        so muons[:-1] can be a mu1 and muons[1:] a mu2.'''
        if not cls.cov_correctors or len(muons) < 2:
            return
        for imu in muons:
            if not hasattr(imu, 'cov'):
                imu.cov = convert_cov(imu.bestTrack().covariance())
                imu.is_cov_pos_def = is_pos_def(imu.cov)
        cls.cov_correctors['mu1'].prime(muons[:-1], event=event)
        cls.cov_correctors['mu2'].prime(muons[1:],  event=event)

    def fit_track(self, obj, role):
        '''The reco::Track handed to the vertex fit for muon `obj` in `role`:
        bestTrack() without covflow, or the same track rebuilt around the
        covflow-corrected covariance (utils.track_with_cov: momentum, charge and
        reference point untouched). A track covflow cannot correct (non positive
        definite, morph failure) is used raw and flagged <role>_covflow_ok = 0.
        Memoized per object and role.'''
        corrector = self.cov_correctors.get(role)
        if corrector is None:
            return obj.bestTrack()
        key = '_cov_fit_track_' + role
        cached = getattr(obj, key, None)
        if cached is not None:
            return cached
        raw = obj.bestTrack()
        corrected = corrector(obj, raw, obj.cov)
        fitted = raw if corrected is None else track_with_cov(raw, corrected)
        setattr(obj, key, fitted)
        return fitted

    def expose_cov(self, imu, role):
        '''Before filling the per-muon branches: put the role's correction where
        the shared getters read it (CommonBranches.cov_track -> _cov_fit_track,
        track_cov_corr_branches -> cov_corr / covflow_ok / covflow_zmax), so
        mu<i>_dxy_e and friends come from the covariance the fit used.'''
        if not self.cov_correctors:
            return
        imu._cov_fit_track = self.fit_track(imu, role)
        imu.cov_corr     = getattr(imu, 'cov_corr_'     + role)
        imu.covflow_ok   = getattr(imu, 'covflow_ok_'   + role)
        imu.covflow_zmax = getattr(imu, 'covflow_zmax_' + role)

    def __init__(self, muons, vertices, beamspot, pf=None, lost=None):
        # sort by pt
        self.muons = sorted([mu for mu in muons], key = lambda x : x.pt(), reverse = True)
        self.mu1 = self.muons[0]
        self.mu2 = self.muons[1]
        # check that the muon track covariance matrix is pos-def
        self.mu1.cov = convert_cov(self.mu1.bestTrack().covariance())
        self.mu2.cov = convert_cov(self.mu2.bestTrack().covariance())
        self.mu1.is_cov_pos_def = is_pos_def(self.mu1.cov)
        self.mu2.is_cov_pos_def = is_pos_def(self.mu2.cov)
        # choose as PV the one that's closest to the leading muon in the dz
        # parameter. Keep the INDEX as well as the vertex: refit_primary_vertex
        # needs it to read the packed candidates' PV association.
        vtxs = list(vertices)
        self.pv_idx = min(
            range(len(vtxs)),
            key = lambda i : abs(self.mu1.bestTrack().dz(vtxs[i].position()))
        )
        self.pv = vtxs[self.pv_idx]

        # create a Vertex type of object from the bs coordinates at the z of the chosen PV
        self.bs = build_beamspot_vertex(beamspot, self.pv.z())
        # keep the raw beamspot too: self.bs is it evaluated at the PV z and
        # wrapped as a vertex, which is not the same thing as its own x0/y0/z0
        self.beamspot = beamspot

        # ----- the wrt-PV reference, identical to the RJpsi channel -----------
        # per-candidate AdaptiveVertexFitter refit of the chosen PV, beamspot
        # constrained and with the signal muons removed. Falls back to the Run2
        # hybrid PV (beamspot x,y + PV z) when the packed candidates are not
        # available or the refit fails; pv_refit_valid says which one you got.
        self.refit_primary_vertex(beamspot, pf, lost, vtxs)
        self.pv_bs = self.pv_refit if self.pv_refit_valid \
                     else build_hybrid_pv(self.bs, self.pv)

        # we'll fit a vertex out of the three muons, shall we? 
        # ideally this can be triggered on demand, and just build a skinny candidate to 
        # check simple things, such as mass etc
        tofit.clear()
        for imu, role in zip(self.muons, ROLES):
            tofit.push_back(self.fit_track(imu, role))
        self.vtx = vtxfit.Fit(tofit)
        self.vtx.chi2 = self.vtx.normalisedChiSquared()
        self.vtx.prob = (1. - stats.chi2.cdf(self.vtx.chi2, 1)) if self.vtx.isValid() else np.nan 

        # ----- displacement, 2D and 3D, wrt the refit PV ----------------------
        # Reference is self.pv_bs: the beamspot-constrained refit with the signal
        # muons removed, or the hybrid PV when pv_refit_valid is 0. Same
        # reference and same conventions as the RJpsi channel, so lxy / lxyz /
        # cos2d / cos3d mean the same thing in both ntuples.
        #
        # lxy used to be measured from the bare beamspot (self.bs) and there was
        # no 3D at all. That quantity is kept as lxy_bs / cos2d_bs, unchanged, so
        # the switch can be validated rather than taken on trust.
        self.compute_displacement()

        # wrt the refit, beamspot-constrained, signal-removed PV (self.pv_bs),
        # the same reference the RJpsi channel uses for its flight direction
        self.pv_to_sv = ROOT.Math.XYZVector(
                            (self.vtx.position().x() - self.pv_bs.position().x()),
                            (self.vtx.position().y() - self.pv_bs.position().y()),
                            (self.vtx.position().z() - self.pv_bs.position().z())
                        ) if self.vtx.isValid() else np.nan
        self.Bdirection  = self.pv_to_sv/np.sqrt(self.pv_to_sv.Mag2()) if self.vtx.isValid() else np.nan                  
        self.Bdir_eta    = self.Bdirection.eta() if self.vtx.isValid() else np.nan                                
        self.Bdir_phi    = self.Bdirection.phi() if self.vtx.isValid() else np.nan                                
        self.mmm_p4_par  = self.p4().Vect().Dot(self.Bdirection) if self.vtx.isValid() else np.nan                   
        self.mmm_p4_perp = np.sqrt(self.p4().Vect().Mag2() - self.mmm_p4_par*self.mmm_p4_par) if self.vtx.isValid() else np.nan
        self.mcorr       = np.sqrt(self.p4().mass()*self.p4().mass() + self.mmm_p4_perp*self.mmm_p4_perp) + self.mmm_p4_perp if self.vtx.isValid() else np.nan
            
    def compute_raw_vertex(self):
        '''With covflow installed: the same dimuon fit with the RAW covariances,
        for the candidate that is written. vtx_chi2_raw / vtx_prob_raw /
        lxy_sig_raw / lxyz_sig_raw then sit next to the corrected ones in one
        file, the vertex-level analogue of cov_* / cov_corr_*. Without covflow
        they equal the nominal branches (no second fit is run).'''
        if not self.cov_correctors:
            self.vtx_raw, self.lxy_raw, self.lxyz_raw = self.vtx, self.lxy, self.lxyz
            return
        tofit.clear()
        for imu in self.muons:
            tofit.push_back(imu.bestTrack())
        vtx = vtxfit.Fit(tofit)
        vtx.chi2 = vtx.normalisedChiSquared()
        vtx.prob = (1. - stats.chi2.cdf(vtx.chi2, 1)) if vtx.isValid() else np.nan
        self.vtx_raw = vtx
        self.lxy_raw = self.lxyz_raw = None
        if vtx.isValid():
            state = vtx.vertexState()
            self.lxy_raw  = ROOT.VertexDistanceXY().distance(self.pv_bs, state)
            self.lxyz_raw = ROOT.VertexDistance3D().distance(self.pv_bs, state)

    def compute_displacement(self):
        '''
        Distance and pointing angle between the refit PV (self.pv_bs) and the
        dimuon vertex (self.vtx), in 2D and in 3D. Same reference, same
        conventions and same names as JpsiChargedCandidate.compute_displacement,
        so the two channels are directly comparable.

        Sets:
          lxy       : 2D distance from the refit PV        (Measurement1D)
          lxyz      : 3D distance from the refit PV        (Measurement1D)
          cos2d     : cosine of the 2D pointing angle (refit PV -> SV vs pT)
          cos3d     : cosine of the 3D pointing angle (refit PV -> SV vs p)
          lxy_bs    : 2D distance from the BARE beamspot   (Measurement1D)
          cos2d_bs  : 2D pointing angle wrt the bare beamspot

        The _bs pair is the pre-refit convention this channel used to have, kept
        unchanged as a fixed comparator: lxy - lxy_bs is the size of the switch.
        Everything is NaN (Measurement1D-less) for an invalid vertex.
        '''
        self.lxy = self.lxyz = self.lxy_bs = None
        self.cos2d = self.cos3d = self.cos2d_bs = np.nan

        if not self.vtx.isValid():
            return

        state = self.vtx.vertexState()
        sv    = self.vtx.position()

        # ----- distances, with their errors, via Measurement1D ---------------
        self.lxy    = ROOT.VertexDistanceXY().distance(self.pv_bs, state)
        self.lxyz   = ROOT.VertexDistance3D().distance(self.pv_bs, state)
        self.lxy_bs = ROOT.VertexDistanceXY().distance(self.bs,    state)

        def _cos(dx, dy, dz, px, py, pz):
            d = Vector3D(dx, dy, dz)
            p = Vector3D(px, py, pz)
            return p.Dot(d) / (p.R() * d.R()) if (d.R() > 0. and p.R() > 0.) else np.nan

        # ----- pointing angles ------------------------------------------------
        # 2D: transverse displacement from the refit PV against the pair pT
        self.cos2d = _cos(sv.x() - self.pv_bs.position().x(),
                          sv.y() - self.pv_bs.position().y(), 0.,
                          self.px(), self.py(), 0.)

        # 3D: full displacement from the refit PV against the pair momentum
        # (consistent with pv_to_sv / Bdirection below, so cos3d is the angle
        # between the flight direction and p)
        self.cos3d = _cos(sv.x() - self.pv_bs.position().x(),
                          sv.y() - self.pv_bs.position().y(),
                          sv.z() - self.pv_bs.position().z(),
                          self.px(), self.py(), self.pz())

        # and the pre-refit comparator, wrt the bare beamspot
        self.cos2d_bs = _cos(sv.x() - self.bs.position().x(),
                             sv.y() - self.bs.position().y(), 0.,
                             self.px(), self.py(), 0.)

        # keep the legacy attribute alive: some code (and the __str__ below)
        # reads cand.vtx.cos
        self.vtx.cos = self.cos2d

    def p4(self):
        return self.mu1.p4() + self.mu2.p4()
    def pt(self):
        return self.p4().pt()
    def eta(self):
        return self.p4().eta()
    def phi(self):
        return self.p4().phi()
    def mass(self):
        return self.p4().mass()
    def mass_corrected(self):    
        return self.mcorr
    def energy(self):
        return self.p4().energy()
    def px(self):
        return self.p4().px()
    def py(self):
        return self.p4().py()
    def pz(self):
        return self.p4().pz()
    def charge(self):
        return self.mu1.charge() + self.mu2.charge()
    def r(self):
        '''
        Cone radius parameter: max distance between the 3-mu candidate direction and one of the muons
        '''
        return max([deltaR(self.p4(), imu) for imu in self.muons])
    def max_dr(self):
        '''
        Max distance between pairwise muons
        '''
        return max([deltaR(imu, jmu) for imu, jmu in combinations(self.muons, 2)])
    def dr12(self):
        return deltaR(self.mu1, self.mu2)
    def __str__(self):
        to_return = [
            'cand mass %.2f pt %.2f eta %.2f phi %.2f' %(self.mass(), self.pt(), self.eta(), self.phi()),
            'cand vtx prob %2f vtx chi2 %.2f lxy %.4f lxy sig %.2f cos %.2f' %(self.vtx.prob, self.vtx.chi2, self.lxy.value(), self.lxy.significance(), self.vtx.cos),
            '\t mu1 pt %.2f eta %.2f phi %.2f' %(self.mu1.pt(), self.mu1.eta(), self.mu1.phi()),
            '\t mu2 pt %.2f eta %.2f phi %.2f' %(self.mu2.pt(), self.mu2.eta(), self.mu2.phi()),
        ]
        return '\n'.join(to_return)
