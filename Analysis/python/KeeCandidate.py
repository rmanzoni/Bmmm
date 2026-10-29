import numpy as np
from scipy import stats
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from itertools import product, combinations

import ROOT
ROOT.gSystem.Load('libBmmmAnalysis')
from ROOT import KVFitter # VertexDistance3D is contained here, dirt trick!!

# make these available everywhere in here
global e_mass
e_mass = 0.510998950e-3
global vtxfit
vtxfit = KVFitter()
global tofit
tofit = ROOT.std.vector('reco::Track')()

class DiEleCandidate():
    '''
    2-ele candidate.
    MISSING: use the post fit ele momenta
    '''
    def __init__(self, triplet, vertices, beamspot):
        # sort by pt
        self.eles = sorted([ele for ele in triplet], key = lambda x : x.pt(), reverse = True)
        self.ele1 = self.eles[0]
        self.ele2 = self.eles[1]
        # check that the ele track covariance matrix is pos-def
        self.ele1.cov = self.convert_cov(self.ele1.gsfTrack().get().covariance())
        self.ele2.cov = self.convert_cov(self.ele2.gsfTrack().get().covariance())
        self.ele1.is_cov_pos_def = self.is_pos_def(self.ele1.cov)
        self.ele2.is_cov_pos_def = self.is_pos_def(self.ele2.cov)
        # choose as PV the one that's closest to the leading ele in the dz parameter
        self.pv = sorted( [vtx for vtx in vertices], key = lambda vtx : abs( self.ele1.gsfTrack().get().dz(vtx.position() ) ) )[0]
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

        # we'll fit a vertex out of the three eles, shall we? 
        # ideally this can be triggered on demand, and just build a skinny candidate to 
        # check simple things, such as mass etc
        tofit.clear()
        for iele in self.eles:
            tofit.push_back(iele.gsfTrack().get())
        self.vtx = vtxfit.Fit(tofit)
        self.vtx.chi2 = self.vtx.normalisedChiSquared()
        self.vtx.prob = (1. - stats.chi2.cdf(self.vtx.chi2, 1)) if self.vtx.isValid() else np.nan 

        # now compute some displacement related quantities, here in the transverse plane.
        # later can add 3D quantities
        self.lxy = ROOT.VertexDistanceXY().distance(self.bs, self.vtx.vertexState()) if self.vtx.isValid() else np.nan

        vect_lxy = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                    self.vtx.position().x() - self.bs.position().x(),
                    self.vtx.position().y() - self.bs.position().y(),
                    0. ) if self.vtx.isValid() else np.nan

        vect_pt = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                    self.px(),
                    self.py(),
                    0. ) if self.vtx.isValid() else np.nan

        self.vtx.cos = vect_pt.Dot(vect_lxy) / (vect_pt.R() * vect_lxy.R()) if (self.vtx.isValid() and vect_lxy.R() > 0.) else np.nan
        
        self.pv_to_sv = ROOT.Math.XYZVector(
                            (self.vtx.position().x() - self.pv.position().x()), 
                            (self.vtx.position().y() - self.pv.position().y()),
                            (self.vtx.position().z() - self.pv.position().z())
                        ) if self.vtx.isValid() else np.nan
        self.Bdirection  = self.pv_to_sv/np.sqrt(self.pv_to_sv.Mag2()) if self.vtx.isValid() else np.nan                  
        self.Bdir_eta    = self.Bdirection.eta() if self.vtx.isValid() else np.nan                                
        self.Bdir_phi    = self.Bdirection.phi() if self.vtx.isValid() else np.nan                                
            
    @staticmethod
    def convert_cov(m):
        return np.nan_to_num(np.array([[(m(i,j)) for j in range(m.kCols)] for i in range(m.kRows)]), posinf=0., neginf=0.)

    @staticmethod
    def is_pos_def(x):
        '''
        https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
        '''
        # don't know why it fails sometimes
        try: 
            is_it = np.all(np.linalg.eigvals(x) > 0)
        except: 
            is_it = False
            # import pdb ; pdb.set_trace()
        return is_it

    @staticmethod
    def fix_track(trk, cov, delta=1e-9):
        '''
        https://github.com/CMSKStarMuMu/miniB0KstarMuMu/blob/master/miniKstarMuMu/plugins/miniKstarMuMu.cc#L1611-L1678
        '''
        if DiEleCandidate.is_pos_def(cov): 
            return trk
        
        new_cov = np.nan_to_num(cov, posinf=0., neginf=0.)
        min_eigenvalue = np.nan_to_num(min(np.linalg.eigvals(new_cov)))
        for i in range(new_cov.shape[0]):
            new_cov[i,i] = new_cov[i,i] - min_eigenvalue + delta
                           
        upper_triangle = []
        for i in range(new_cov.shape[0]):
            for j in range(new_cov.shape[1]):
                if i<=j:
                    upper_triangle.append(new_cov[i,j])
                    
        # https://root.cern/doc/v606/SMatrixDoc.html
        # che sudata sta merdata
        mycov = ROOT.Math.SMatrix('double', 5, 5, ROOT.Math.MatRepSym('double', 5) )(ROOT.Math.SVector('double', len(upper_triangle))(np.array(upper_triangle), len(upper_triangle)), False)
        #check_cov = DiEleCandidate.convert_cov(mycov)

        new_trk = ROOT.reco.Track(
            trk.chi2(), 
            trk.ndof(), 
            trk.referencePoint(), 
            trk.momentum(), 
            trk.charge(), 
            mycov, 
            trk.algo(), 
            ROOT.reco.TrackBase.TrackQuality(trk.qualityMask()),
        )

        if not DiEleCandidate.is_pos_def(new_cov): 
            DiEleCandidate.fix_track(new_trk, new_cov, delta)
        
        return new_trk

    def p4(self, kind=1):
        # combined gsf track + supercluster momentum https://cmssdt.cern.ch/lxr/source/DataFormats/EgammaCandidates/interface/GsfElectron.h
        if kind==3:
            # take momentum from gsf track
            p1 = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzM4D<double>')(
                    self.ele1.gsfTrack().px(),
                    self.ele1.gsfTrack().py(),
                    self.ele1.gsfTrack().pz(),
                    e_mass
            )
            p2 = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzM4D<double>')(
                    self.ele2.gsfTrack().px(),
                    self.ele2.gsfTrack().py(),
                    self.ele2.gsfTrack().pz(),
                    e_mass
            )
            p4 = p1 + p2
        else:
            p4 = self.ele1.p4(kind) + self.ele2.p4(kind)
        return p4
    def pt(self, kind=1):
        return self.p4(kind).pt()
    def eta(self, kind=1):
        return self.p4(kind).eta()
    def phi(self, kind=1):
        return self.p4(kind).phi()
    def mass(self, kind=1):
        return self.p4(kind).mass()
    def mass_corrected(self, kind=1):    
        mmm_p4_par  = self.p4(kind).Vect().Dot(self.Bdirection) if self.vtx.isValid() else np.nan                   
        mmm_p4_perp = np.sqrt(self.p4(kind).Vect().Mag2() - mmm_p4_par*mmm_p4_par) if self.vtx.isValid() else np.nan
        mcorr       = np.sqrt(self.mass(kind)*self.mass(kind) + mmm_p4_perp*mmm_p4_perp) + mmm_p4_perp if self.vtx.isValid() else np.nan
        return mcorr
    def energy(self, kind=1):
        return self.p4(kind).energy()
    def px(self, kind=1):
        return self.p4(kind).px()
    def py(self, kind=1):
        return self.p4(kind).py()
    def pz(self, kind=1):
        return self.p4(kind).pz()
    def charge(self):
        return self.ele1.charge() + self.ele2.charge()
    def r(self):
        '''
        Cone radius parameter: max distance between the 4-ele candidate direction and one of the eles
        '''
        return max([deltaR(self.p4(), iele) for iele in self.eles])
    def max_dr(self):
        '''
        Max distance between pairwise eles
        '''
        return max([deltaR(iele, jele) for iele, jele in combinations(self.eles, 2)])
    def dr(self):
        return deltaR(self.ele1, self.ele2)
    def __str__(self):
        to_return = [
            'cand mass %.2f pt %.2f eta %.2f phi %.2f' %(self.mass(), self.pt(), self.eta(), self.phi()),
            'cand vtx prob %2f vtx chi2 %.2f lxy %.4f lxy sig %.2f cos %.2f' %(self.vtx.prob, self.vtx.chi2, self.lxy.value(), self.lxy.significance(), self.vtx.cos),
            '\t ele1 pt %.2f eta %.2f phi %.2f' %(self.ele1.pt(), self.ele1.eta(), self.ele1.phi()),
            '\t ele2 pt %.2f eta %.2f phi %.2f' %(self.ele2.pt(), self.ele2.eta(), self.ele2.phi()),
        ]
        return '\n'.join(to_return)

class KeeCandidate():
    '''
    2-ele candidate.
    MISSING: use the post fit ele momenta
    '''
    def __init__(self, diele, trk, isotrks, vertices, beamspot, mass=0.493677): # kaon mass by default
        
        self.isotrks = isotrks
        self.trk = trk
        self.trk.setMass(mass)
        self.trk.cov = DiEleCandidate.convert_cov(self.trk.bestTrack().covariance())
        self.trk.is_cov_pos_def = DiEleCandidate.is_pos_def(self.trk.cov)

        self.diele = diele
        self.eles = self.diele.eles
        self.ele1 = self.diele.ele1
        self.ele2 = self.diele.ele2
        # import pdb ; pdb.set_trace()
        # FIXME! Energy not Mass!!!!
        self.pion1 = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzE4D<double>')(self.ele1.px(), self.ele1.py(), self.ele1.pz(), 0.13957)
        self.pion2 = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzE4D<double>')(self.ele2.px(), self.ele2.py(), self.ele2.pz(), 0.13957)
        
        # FIXME! improve the PV determination using the B direction
#         # choose as PV the one that's closest to the leading ele in the dz parameter
#         self.pv = sorted( [vtx for vtx in vertices], key = lambda vtx : abs( self.ele1.gsfTrack().get().dz(vtx.position() ) ) )[0]
#         # create a Vertex type of object from the bs coordinates at the z of the chosen PV
#         bs_point = ROOT.reco.Vertex.Point(
#             beamspot.x(self.pv.z()),
#             beamspot.y(self.pv.z()),
#             beamspot.z0(),
#         )
# 
#         bs_error = beamspot.covariance3D()
#         chi2 = 0.
#         ndof = 0.
#         self.bs = ROOT.reco.Vertex(bs_point, bs_error, chi2, ndof, 3) # size? say 3? does it matter?

        # we'll fit a vertex out of the three eles, shall we? 
        # ideally this can be triggered on demand, and just build a skinny candidate to 
        # check simple things, such as mass etc
        
        self.pv = self.diele.pv
        self.bs = self.diele.bs
        
        tofit.clear()
        for iele in self.eles:
            tofit.push_back(iele.gsfTrack().get())
        
        # add track
        # print('track pt %.2f and mass %.2f has track details %d' %(self.trk.pt(), self.trk.mass(), self.trk.hasTrackDetails()))
        tofit.push_back(DiEleCandidate.fix_track(self.trk.bestTrack(), self.trk.cov))
        
        self.vtx = vtxfit.Fit(tofit)
        self.vtx.chi2 = self.vtx.normalisedChiSquared()
        self.vtx.prob = (1. - stats.chi2.cdf(self.vtx.chi2, 1)) if self.vtx.isValid() else np.nan 

        # now compute some displacement related quantities, here in the transverse plane.
        # later can add 3D quantities
        self.lxy = ROOT.VertexDistanceXY().distance(self.bs, self.vtx.vertexState()) if self.vtx.isValid() else np.nan

        vect_lxy = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                    self.vtx.position().x() - self.bs.position().x(),
                    self.vtx.position().y() - self.bs.position().y(),
                    0. ) if self.vtx.isValid() else np.nan

        vect_pt = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
                    self.px(),
                    self.py(),
                    0. ) if self.vtx.isValid() else np.nan

        self.vtx.cos = vect_pt.Dot(vect_lxy) / (vect_pt.R() * vect_lxy.R()) if (self.vtx.isValid() and vect_lxy.R() > 0.) else np.nan
        
        self.pv_to_sv = ROOT.Math.XYZVector(
                            (self.vtx.position().x() - self.pv.position().x()), 
                            (self.vtx.position().y() - self.pv.position().y()),
                            (self.vtx.position().z() - self.pv.position().z())
                        ) if self.vtx.isValid() else np.nan
        self.Bdirection  = self.pv_to_sv/np.sqrt(self.pv_to_sv.Mag2()) if self.vtx.isValid() else np.nan                  
        self.Bdir_eta    = self.Bdirection.eta() if self.vtx.isValid() else np.nan                                
        self.Bdir_phi    = self.Bdirection.phi() if self.vtx.isValid() else np.nan                                

    def trk_iso(self, ptmin=0.5, drmax=0.5, dzmax=0.2):
        abs_iso = np.sum([tk.pt() for tk in self.isotrks if deltaR(tk, self)<drmax and tk.pt()>ptmin and abs(tk.dz(self.pv.position()))<dzmax and deltaR(tk, self.ele1)>0.01 and deltaR(tk, self.ele2)>0.01 and deltaR(tk, self.trk)>0.01])
        rel_iso = abs_iso / self.pt()
        return  abs_iso, rel_iso

    def p4(self, kind=1):
        return self.diele.p4(kind) + self.trk.p4() # combined gsf track + supercluster momentum https://cmssdt.cern.ch/lxr/source/DataFormats/EgammaCandidates/interface/GsfElectron.h
    def pt(self, kind=1):
        return self.p4(kind).pt()
    def eta(self, kind=1):
        return self.p4(kind).eta()
    def phi(self, kind=1):
        return self.p4(kind).phi()
    def mass(self, kind=1):
        return self.p4(kind).mass()
    def mass_corrected(self, kind=1):    
        mmm_p4_par  = self.p4(kind).Vect().Dot(self.Bdirection) if self.vtx.isValid() else np.nan                   
        mmm_p4_perp = np.sqrt(self.p4(kind).Vect().Mag2() - mmm_p4_par*mmm_p4_par) if self.vtx.isValid() else np.nan
        mcorr       = np.sqrt(self.mass(kind)*self.mass(kind) + mmm_p4_perp*mmm_p4_perp) + mmm_p4_perp if self.vtx.isValid() else np.nan
        return mcorr
    def energy(self, kind=1):
        return self.p4(kind).energy()
    def px(self, kind=1):
        return self.p4(kind).px()
    def py(self, kind=1):
        return self.p4(kind).py()
    def pz(self, kind=1):
        return self.p4(kind).pz()
    def charge(self):
        return self.diele.charge() + self.trk.charge()
    def charge_e1k(self):
        return self.ele1.charge() + self.trk.charge()
    def charge_e2k(self):
        return self.ele2.charge() + self.trk.charge()
    def mass_e1k(self):
        return (self.ele1.p4() + self.trk.p4()).mass()
    def mass_e2k(self):
        return (self.ele2.p4() + self.trk.p4()).mass()
    def mass_p1k(self):
        return (self.pion1 + self.trk.p4()).mass()
    def mass_p2k(self):
        return (self.pion2 + self.trk.p4()).mass()
    def dr_e1k(self):
        return deltaR(self.ele1, self.trk)
    def dr_e2k(self):
        return deltaR(self.ele2, self.trk)
    def max_dr(self):
        '''
        Cone radius parameter: max distance between the 4-ele candidate direction and one of the eles
        '''
        return max([deltaR(self.p4(), ip) for ip in self.eles + [self.trk]])
    def dr_ee_tk(self):
        '''
        Max distance between pairwise eles
        '''
        return deltaR(self.diele, self.trk)
    def r(self):
        return max([deltaR(iele, jele) for iele, jele in combinations(self.eles + [self.trk], 2)])

    def __str__(self):
        to_return = [
            'cand mass %.2f pt %.2f eta %.2f phi %.2f' %(self.mass(), self.pt(), self.eta(), self.phi()),
            'cand vtx prob %2f vtx chi2 %.2f lxy %.4f lxy sig %.2f cos %.2f' %(self.vtx.prob, self.vtx.chi2, self.lxy.value(), self.lxy.significance(), self.vtx.cos),
            '\t ele1 pt %.2f eta %.2f phi %.2f' %(self.ele1.pt(), self.ele1.eta(), self.ele1.phi()),
            '\t ele2 pt %.2f eta %.2f phi %.2f' %(self.ele2.pt(), self.ele2.eta(), self.ele2.phi()),
            '\t trk  pt %.2f eta %.2f phi %.2f' %(self.trk.pt() , self.trk.eta() , self.trk.phi() ),
        ]
        return '\n'.join(to_return)
