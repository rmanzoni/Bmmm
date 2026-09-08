import numpy as np
from scipy import stats
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from itertools import product, combinations
from Bmmm.Analysis.utils import convert_cov, is_pos_def
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

class Candidate(PrimaryVertexRefitMixin):
    '''
    3-muon candidate.
    MISSING: use the post fit muon momenta
    '''
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
        for imu in self.muons:
            tofit.push_back(imu.bestTrack())
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
