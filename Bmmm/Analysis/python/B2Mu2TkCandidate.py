import numpy as np
from scipy import stats
from itertools import product, combinations
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from Bmmm.Analysis.utils import masses, p4_with_mass, is_pos_def, convert_cov, fix_track, compute_IP3D

import ROOT
ROOT.gSystem.Load('libBmmmAnalysis')
from ROOT import KVFitter # VertexDistance3D is contained here, dirt trick!!
from ROOT import B4MuKinVtxFitter

# make these available everywhere in here
global vtxfit
vtxfit = KVFitter()
global tofit
tofit = ROOT.std.vector('reco::Track')()
global kinfit
kinfit = B4MuKinVtxFitter()

class B2Mu2TkCandidate():
    '''
    FIXME! brutal copy-paste from B4MuCandidate... should do better than this.
    '''
    def __init__(self, muons, tracks, vertices, beamspot):
        
        '''
        tracks must be adictionary {track:mass}
        '''
        # sort by pt
        self.muons     = sorted([mu for mu in muons        ], key = lambda x : x.pt(), reverse = True)
        self.tracks    = sorted([tk for tk in tracks.keys()], key = lambda x : x.pt(), reverse = True)

        self.mu1 = self.muons[0]
        self.mu2 = self.muons[1]
        self.tk1 = self.tracks[0]
        self.tk2 = self.tracks[1]

        self.tk1.old_mass = self.tk1.mass()
        self.tk2.old_mass = self.tk2.mass()

        self.tk1.old_p4 = self.tk1.p4()
        self.tk2.old_p4 = self.tk2.p4()

        self.tk1.new_mass = tracks[self.tk1]
        self.tk2.new_mass = tracks[self.tk2]

        self.tk1.new_p4 = p4_with_mass(self.tk1, self.tk1.new_mass)
        self.tk2.new_p4 = p4_with_mass(self.tk2, self.tk2.new_mass)

        self.tk1.p4 = lambda : self.tk1.new_p4
        self.tk2.p4 = lambda : self.tk2.new_p4

        self.tk1.mass = lambda : self.tk1.p4().mass()
        self.tk2.mass = lambda : self.tk2.p4().mass()

        self.tk1.energy = lambda : self.tk1.p4().energy()
        self.tk2.energy = lambda : self.tk2.p4().energy()

        # check that the muon track covariance matrix is pos-def
        self.mu1.cov = convert_cov(self.mu1.bestTrack().covariance())
        self.mu2.cov = convert_cov(self.mu2.bestTrack().covariance())
        self.tk1.cov = convert_cov(self.tk1.bestTrack().covariance())
        self.tk2.cov = convert_cov(self.tk2.bestTrack().covariance())
        self.mu1.is_cov_pos_def = is_pos_def(self.mu1.cov)
        self.mu2.is_cov_pos_def = is_pos_def(self.mu2.cov)
        self.tk1.is_cov_pos_def = is_pos_def(self.tk1.cov)
        self.tk2.is_cov_pos_def = is_pos_def(self.tk2.cov)
                
        mu1_tk = self.mu1.bestTrack() if self.mu1.is_cov_pos_def else fix_track(self.mu1.bestTrack())   
        mu2_tk = self.mu2.bestTrack() if self.mu2.is_cov_pos_def else fix_track(self.mu2.bestTrack())   
        tk1_tk = self.tk1.bestTrack() if self.tk1.is_cov_pos_def else fix_track(self.tk1.bestTrack())   
        tk2_tk = self.tk2.bestTrack() if self.tk2.is_cov_pos_def else fix_track(self.tk2.bestTrack())   
        
        self.vertex_tree = kinfit.Fit(mu1_tk, mu2_tk, tk1_tk, tk2_tk, \
                                      masses['mu'], masses['mu'], masses['k'], masses['k'])
        self.good_vtx = False
        
        if self.vertex_tree:
            self.good_vtx = ( (not self.vertex_tree.isEmpty()) and self.vertex_tree.isValid() )
        if self.good_vtx:
            self.compute_vtx_quantities(vertices, beamspot)
        else:
            # if secondary vertex is not good, default to finding PV and BS based on the leading muon
            # choose as PV the one that's closest to the leading muon in the dz parameter
            self.pv = sorted( [vtx for vtx in vertices], key = lambda vtx : abs( self.mu1.bestTrack().dz(vtx.position() ) ) )[0]
            
            # create a Vertex type of object from the bs coordinates at the z of the chosen PV
            bs_point = ROOT.reco.Vertex.Point(
                beamspot.x(self.pv.z()),
                beamspot.y(self.pv.z()),
                beamspot.z0(),
            )
    
            bs_error = beamspot.covariance3D()
            chi2 = 0.
            ndof = 0.
            self.bs = ROOT.reco.Vertex(bs_point, bs_error, chi2, ndof, 3) # size? say 3? does it matter?
        
        
    def compute_vtx_quantities(self, vertices, beamspot):

        self.vertex_tree.movePointerToTheTop()
        self.vtx = self.vertex_tree.currentDecayVertex().get()
        
        # find PV as the closest to the Bs flight direction
        # https://www.nagwa.com/en/explainers/939127418581/#:~:text=The%20perpendicular%20distance%20between%20a%20point%20and%20a%20line%20is,any%20point%20on%20the%20line.
        pv_idx = -1
        ip3d_min = np.inf
        for idx, ivtx in enumerate(vertices):
            ip3d = compute_IP3D(ivtx, self.vtx.position(), self.p4().Vect())
            if ip3d<ip3d_min:
                pv_idx = idx
                ip3d_min = ip3d
            
        self.pv = vertices[pv_idx]
                
        # create a Vertex type of object from the bs coordinates at the z of the chosen PV
        bs_point = ROOT.reco.Vertex.Point(
            beamspot.x(self.pv.z()),
            beamspot.y(self.pv.z()),
            beamspot.z0(),
        )

        bs_error = beamspot.covariance3D()
        chi2 = 0.
        ndof = 0.
        self.bs = ROOT.reco.Vertex(bs_point, bs_error, chi2, ndof, 3) # size? say 3? does it matter?

        self.vtx.ndof = self.vtx.degreesOfFreedom()
        self.vtx.chi2 = self.vtx.chiSquared()
        self.vtx.norm_chi2 = self.vtx.chi2/self.vtx.ndof
        self.vtx.prob = (1. - stats.chi2.cdf(self.vtx.chi2, self.vtx.ndof)) 
    
        # now compute some displacement related quantities, here in the transverse plane.
        # later can add 3D quantities
        
        # 2D
        self.lxy = ROOT.VertexDistanceXY().distance(self.bs, self.vtx.vertexState())
    
        vect_lxy = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
            self.vtx.position().x() - self.bs.position().x(),
            self.vtx.position().y() - self.bs.position().y(),
            0. 
        )
    
        vect_pt = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
            self.px(),
            self.py(),
            0. 
        )
    
        self.vtx.cos2d = vect_pt.Dot(vect_lxy) / (vect_pt.R() * vect_lxy.R()) if (vect_lxy.R() > 0.) else np.nan
        
        # 3D
        self.lxyz = ROOT.VertexDistance3D().distance(self.pv, self.vtx.vertexState())

        vect_lxyz = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
            self.vtx.position().x() - self.bs.position().x(), # transverse quantities always from BS
            self.vtx.position().y() - self.bs.position().y(), # transverse quantities always from BS
            self.vtx.position().z() - self.pv.position().z(),
        )
        
        vect_p = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
            self.px(),
            self.py(),
            self.pz(),
        )
     
        self.vtx.cos3d = vect_p.Dot(vect_lxyz) / (vect_p.R() * vect_lxyz.R())

        self.pv_to_sv = ROOT.Math.XYZVector(
                            (self.vtx.position().x() - self.pv.position().x()), 
                            (self.vtx.position().y() - self.pv.position().y()),
                            (self.vtx.position().z() - self.pv.position().z())
                        )
        self.Bdirection  = self.pv_to_sv/np.sqrt(self.pv_to_sv.Mag2())                  
        self.Bdir_eta    = self.Bdirection.eta()                                
        self.Bdir_phi    = self.Bdirection.phi()                                
        self.mmm_p4_par  = self.p4().Vect().Dot(self.Bdirection)                   
        self.mmm_p4_perp = np.sqrt(self.p4().Vect().Mag2() - self.mmm_p4_par*self.mmm_p4_par)
        self.mcorr       = np.sqrt(self.p4().mass()*self.p4().mass() + self.mmm_p4_perp*self.mmm_p4_perp) + self.mmm_p4_perp
        
        # can also do this https://github.com/CMSKStarMuMu/miniB0KstarMuMu/blob/master/miniKstarMuMu/plugins/miniKstarMuMu.cc#L809C48-L809C58
        self.vertex_tree.movePointerToTheFirstChild()
        mu1ref = self.vertex_tree.currentParticle()
        self.mu1.rfp4, _ = self.buildP4(mu1ref)

        self.vertex_tree.movePointerToTheNextChild()
        mu2ref = self.vertex_tree.currentParticle()
        self.mu2.rfp4, _ = self.buildP4(mu2ref)

        self.vertex_tree.movePointerToTheNextChild()
        tk1ref = self.vertex_tree.currentParticle()
        self.tk1.rfp4, _ = self.buildP4(tk1ref)

        self.vertex_tree.movePointerToTheNextChild()
        tk2ref = self.vertex_tree.currentParticle()
        self.tk2.rfp4, _ = self.buildP4(tk2ref)
        
        # bmass and mass uncertainty
        # FIXME! ugly naming
        self.vertex_tree.movePointerToTheTop()
        b4ref = self.vertex_tree.currentParticle()
        self.b4refUnc = self.vertex_tree.currentParticle().currentState().kinematicParametersError().matrix()
        self.bbp4, _ = self.buildP4(b4ref)

    @staticmethod
    def buildP4(ref):

        ref_x  = ref.currentState().kinematicParameters().vector().At(0)
        ref_y  = ref.currentState().kinematicParameters().vector().At(1)
        ref_z  = ref.currentState().kinematicParameters().vector().At(2)
        ref_px = ref.currentState().kinematicParameters().vector().At(3)
        ref_py = ref.currentState().kinematicParameters().vector().At(4)
        ref_pz = ref.currentState().kinematicParameters().vector().At(5)
        ref_m  = ref.currentState().kinematicParameters().vector().At(6)

        energy = np.sqrt(ref_px**2 + ref_py**2 + ref_pz**2 + ref_m**2)

        p4 = ROOT.Math.LorentzVector("ROOT::Math::PxPyPzE4D<double>")(ref_px, ref_py, ref_pz, energy)
        
        return p4, ref
        
    def create_refitted_p4(self, idx):
        mu = self.vtx.refittedTracks().at(idx).track()
        rfp4 = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzE4D<double>')(
                mu.px(), mu.py(), mu.pz(), np.sqrt(mu.p()**2 + self.mu1.mass()**2) )
        return rfp4
                           
    def p4(self):
        return self.mu1.p4() + self.mu2.p4() + self.tk1.p4() + self.tk2.p4()
    def p4_12(self):
        return self.mu1.p4() + self.mu2.p4()
    def p4_13(self):
        return self.mu1.p4() + self.tk1.p4()
    def p4_14(self):
        return self.mu1.p4() + self.tk2.p4()
    def p4_23(self):
        return self.mu2.p4() + self.tk1.p4()
    def p4_24(self):
        return self.mu2.p4() + self.tk2.p4()
    def p4_34(self):
        return self.tk1.p4() + self.tk2.p4()
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
        return self.mu1.charge() + self.mu2.charge() + self.tk1.charge() + self.tk2.charge()
    def charge12(self):
        return self.mu1.charge() + self.mu2.charge()
    def charge13(self):
        return self.mu1.charge() + self.tk1.charge()
    def charge14(self):
        return self.mu1.charge() + self.tk2.charge()
    def charge23(self):
        return self.mu2.charge() + self.tk1.charge()
    def charge24(self):
        return self.mu2.charge() + self.tk2.charge()
    def charge34(self):
        return self.tk1.charge() + self.tk2.charge()
    def r(self):
        '''
        Cone radius parameter: max distance between the 4-mu candidate direction and one of the muons
        '''
        return max([deltaR(self.p4(), imu) for imu in self.muons])
    def max_dr(self):
        '''
        Max distance between pairwise muons
        '''
        return max([deltaR(imu, jmu) for imu, jmu in combinations(self.muons, 2)])
    def dr12(self):
        return deltaR(self.mu1, self.mu2)
    def dr13(self):
        return deltaR(self.mu1, self.tk1)
    def dr14(self):
        return deltaR(self.mu1, self.tk2)
    def dr23(self):
        return deltaR(self.mu2, self.tk1)
    def dr24(self):
        return deltaR(self.mu2, self.tk2)
    def dr34(self):
        return deltaR(self.tk1, self.tk2)
    
    # FIXME!
    def mass12(self):
        return self.p4_12().mass()
    def mass13(self):
        return self.p4_13().mass()
    def mass14(self):
        return self.p4_14().mass()
    def mass23(self):
        return self.p4_23().mass()
    def mass24(self):
        return self.p4_24().mass()
    def mass34(self):
        return self.p4_34().mass()
    def __str__(self):
        to_return = [
            'cand mass %.2f pt %.2f eta %.2f phi %.2f' %(self.mass(), self.pt(), self.eta(), self.phi()),
            'cand vtx prob %2f vtx chi2 %.2f lxy %.4f lxy sig %.2f cos %.2f' %(self.vtx.prob, self.vtx.chi2, self.lxy.value(), self.lxy.significance(), self.vtx.cos),
            '\t mu1 pt %.2f eta %.2f phi %.2f' %(self.mu1.pt(), self.mu1.eta(), self.mu1.phi()),
            '\t mu2 pt %.2f eta %.2f phi %.2f' %(self.mu2.pt(), self.mu2.eta(), self.mu2.phi()),
            '\t tk1 pt %.2f eta %.2f phi %.2f' %(self.tk1.pt(), self.tk1.eta(), self.tk1.phi()),
            '\t tk2 pt %.2f eta %.2f phi %.2f' %(self.tk2.pt(), self.tk2.eta(), self.tk2.phi()),
        ]
        return '\n'.join(to_return)

    ######################################################################################
    ######################################################################################
    ####            __ _ _   _           _ 
    ####           / _(_) | | |         | |
    ####  _ __ ___| |_ _| |_| |_ ___  __| |
    #### | '__/ _ \  _| | __| __/ _ \/ _` |
    #### | | |  __/ | | | |_| ||  __/ (_| |
    #### |_|  \___|_| |_|\__|\__\___|\__,_|
    ####                                   
    ######################################################################################
    ######################################################################################

    def rf_p4(self):
        return self.mu1.rfp4 + self.mu2.rfp4 + self.tk1.rfp4 + self.tk2.rfp4
    def rf_p4_12(self):
        return self.mu1.rfp4 + self.mu2.rfp4
    def rf_p4_13(self):
        return self.mu1.rfp4 + self.tk1.rfp4
    def rf_p4_14(self):
        return self.mu1.rfp4 + self.tk2.rfp4
    def rf_p4_23(self):
        return self.mu2.rfp4 + self.tk1.rfp4
    def rf_p4_24(self):
        return self.mu2.rfp4 + self.tk2.rfp4
    def rf_p4_34(self):
        return self.tk1.rfp4 + self.tk2.rfp4
    def rf_pt(self):
        return self.rf_p4().pt()
    def rf_eta(self):
        return self.rf_p4().eta()
    def rf_phi(self):
        return self.rf_p4().phi()
    def rf_mass(self):
        return self.rf_p4().mass()
    def rf_energy(self):
        return self.rf_p4().energy()
    def rf_px(self):
        return self.rf_p4().px()
    def rf_py(self):
        return self.rf_p4().py()
    def rf_pz(self):
        return self.rf_p4().pz()
    def rf_r(self):
        '''
        Cone radius parameter: max distance between the 4-mu candidate direction and one of the muons
        '''
        return max([deltaR(self.rf_p4(), imu.rfp4) for imu in self.muons])
    def rf_max_dr(self):
        '''
        Max distance between pairwise muons
        '''
        return max([deltaR(imu.rfp4, jmu.rfp4) for imu, jmu in combinations(self.muons, 2)])
    def rf_dr12(self):
        return deltaR(self.mu1.rfp4, self.mu2.rfp4)
    def rf_dr13(self):
        return deltaR(self.mu1.rfp4, self.tk1.rfp4)
    def rf_dr14(self):
        return deltaR(self.mu1.rfp4, self.tk2.rfp4)
    def rf_dr23(self):
        return deltaR(self.mu2.rfp4, self.tk1.rfp4)
    def rf_dr24(self):
        return deltaR(self.mu2.rfp4, self.tk2.rfp4)
    def rf_dr34(self):
        return deltaR(self.tk1.rfp4, self.tk2.rfp4)
    def rf_mass12(self):
        return self.rf_p4_12().mass()
    def rf_mass13(self):
        return self.rf_p4_13().mass()
    def rf_mass14(self):
        return self.rf_p4_14().mass()
    def rf_mass23(self):
        return self.rf_p4_23().mass()
    def rf_mass24(self):
        return self.rf_p4_24().mass()
    def rf_mass34(self):
        return self.rf_p4_34().mass()




