import numpy as np
from scipy import stats
from itertools import product, combinations
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from Bmmm.Analysis.utils import masses

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

class B4MuCandidate():
    '''
    3-muon candidate.
    MISSING: use the post fit muon momenta
    '''
    def __init__(self, triplet, vertices, beamspot):
        # sort by pt
        self.muons = sorted([mu for mu in triplet], key = lambda x : x.pt(), reverse = True)
        self.mu1 = self.muons[0]
        self.mu2 = self.muons[1]
        self.mu3 = self.muons[2]
        self.mu4 = self.muons[3]
        # check that the muon track covariance matrix is pos-def
        self.mu1.cov = self.convert_cov(self.mu1.bestTrack().covariance())
        self.mu2.cov = self.convert_cov(self.mu2.bestTrack().covariance())
        self.mu3.cov = self.convert_cov(self.mu3.bestTrack().covariance())
        self.mu4.cov = self.convert_cov(self.mu4.bestTrack().covariance())
        self.mu1.is_cov_pos_def = self.is_pos_def(self.mu1.cov)
        self.mu2.is_cov_pos_def = self.is_pos_def(self.mu2.cov)
        self.mu3.is_cov_pos_def = self.is_pos_def(self.mu3.cov)
        self.mu4.is_cov_pos_def = self.is_pos_def(self.mu4.cov)
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
        
        self.vertex_tree = kinfit.Fit(self.mu1.bestTrack(), self.mu2.bestTrack(), self.mu3.bestTrack(), self.mu4.bestTrack(), masses['mu'])
        self.good_vtx = False
        try:
            if self.vertex_tree:
                self.good_vtx = ( (not self.vertex_tree.isEmpty()) and self.vertex_tree.isValid() )
            if self.good_vtx:
                self.compute_vtx_quantities()        
        except:
            import pdb ; pdb.set_trace()
        

    def compute_vtx_quantities(self):

        self.vertex_tree.movePointerToTheTop()
        self.vtx = self.vertex_tree.currentDecayVertex().get()
        
        self.vtx.ndof = self.vtx.degreesOfFreedom()
        self.vtx.chi2 = self.vtx.chiSquared()
        self.vtx.norm_chi2 = self.vtx.chi2/self.vtx.ndof
        self.vtx.prob = (1. - stats.chi2.cdf(self.vtx.chi2, self.vtx.ndof)) 
    
        # now compute some displacement related quantities, here in the transverse plane.
        # later can add 3D quantities
        self.lxy = ROOT.VertexDistanceXY().distance(self.bs, self.vtx.vertexState())
    
        vect_lxy = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                    self.vtx.position().x() - self.bs.position().x(),
                    self.vtx.position().y() - self.bs.position().y(),
                    0. )
    
        vect_pt = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                    self.px(),
                    self.py(),
                    0. )
    
        self.vtx.cos = vect_pt.Dot(vect_lxy) / (vect_pt.R() * vect_lxy.R()) if (vect_lxy.R() > 0.) else np.nan
        
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
        mu3ref = self.vertex_tree.currentParticle()
        self.mu3.rfp4, _ = self.buildP4(mu3ref)

        self.vertex_tree.movePointerToTheNextChild()
        mu4ref = self.vertex_tree.currentParticle()
        self.mu4.rfp4, _ = self.buildP4(mu4ref)

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
                           
    def convert_cov(self, m):
        return np.array([[m(i,j) for j in range(m.kCols)] for i in range(m.kRows)])

    def is_pos_def(self, x):
        '''
        https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
        '''
        return np.all(np.linalg.eigvals(x) > 0)

    def p4(self):
        return self.mu1.p4() + self.mu2.p4() + self.mu3.p4() + self.mu4.p4()
    def p4_12(self):
        return self.mu1.p4() + self.mu2.p4()
    def p4_13(self):
        return self.mu1.p4() + self.mu3.p4()
    def p4_14(self):
        return self.mu1.p4() + self.mu4.p4()
    def p4_23(self):
        return self.mu2.p4() + self.mu3.p4()
    def p4_24(self):
        return self.mu2.p4() + self.mu4.p4()
    def p4_34(self):
        return self.mu3.p4() + self.mu4.p4()
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
        return self.mu1.charge() + self.mu2.charge() + self.mu3.charge() + self.mu4.charge()
    def charge12(self):
        return self.mu1.charge() + self.mu2.charge()
    def charge13(self):
        return self.mu1.charge() + self.mu3.charge()
    def charge14(self):
        return self.mu1.charge() + self.mu4.charge()
    def charge23(self):
        return self.mu2.charge() + self.mu3.charge()
    def charge24(self):
        return self.mu2.charge() + self.mu4.charge()
    def charge34(self):
        return self.mu3.charge() + self.mu4.charge()
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
        return deltaR(self.mu1, self.mu3)
    def dr14(self):
        return deltaR(self.mu1, self.mu4)
    def dr23(self):
        return deltaR(self.mu2, self.mu3)
    def dr24(self):
        return deltaR(self.mu2, self.mu4)
    def dr34(self):
        return deltaR(self.mu3, self.mu4)
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
            '\t mu3 pt %.2f eta %.2f phi %.2f' %(self.mu3.pt(), self.mu3.eta(), self.mu3.phi()),
            '\t mu4 pt %.2f eta %.2f phi %.2f' %(self.mu4.pt(), self.mu4.eta(), self.mu4.phi()),
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
        return self.mu1.rfp4 + self.mu2.rfp4 + self.mu3.rfp4 + self.mu4.rfp4
    def rf_p4_12(self):
        return self.mu1.rfp4 + self.mu2.rfp4
    def rf_p4_13(self):
        return self.mu1.rfp4 + self.mu3.rfp4
    def rf_p4_14(self):
        return self.mu1.rfp4 + self.mu4.rfp4
    def rf_p4_23(self):
        return self.mu2.rfp4 + self.mu3.rfp4
    def rf_p4_24(self):
        return self.mu2.rfp4 + self.mu4.rfp4
    def rf_p4_34(self):
        return self.mu3.rfp4 + self.mu4.rfp4
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
        return deltaR(self.mu1.rfp4, self.mu3.rfp4)
    def rf_dr14(self):
        return deltaR(self.mu1.rfp4, self.mu4.rfp4)
    def rf_dr23(self):
        return deltaR(self.mu2.rfp4, self.mu3.rfp4)
    def rf_dr24(self):
        return deltaR(self.mu2.rfp4, self.mu4.rfp4)
    def rf_dr34(self):
        return deltaR(self.mu3.rfp4, self.mu4.rfp4)
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




