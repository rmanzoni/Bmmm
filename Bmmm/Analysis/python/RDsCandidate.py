import numpy as np
from scipy import stats
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from Bmmm.Analysis.utils import masses, compute_vertex_quantities, p4_with_mass, convert_cov, is_pos_def, fix_track
from itertools import product, combinations

import ROOT
ROOT.gSystem.Load('libBmmmAnalysis')
from ROOT import KVFitter # VertexDistance3D is contained here, dirt trick!!
from ROOT import RDsKinVtxFitter

# make these available everywhere in here
global vtxfit
vtxfit = KVFitter()
global tofit
tofit = ROOT.std.vector('reco::Track')()
global kinfit
kinfit = RDsKinVtxFitter()

##########################################################################################
class PhiCandidate():
    def __init__(self, k1, k2, bs, pv):
        self.k1 = k1
        self.k2 = k2
        self.bs = bs
        self.pv = pv 

        # compute correct four momenta and override default call to p4()
        self.k1.p4 = lambda : p4_with_mass(particle=self.k1, mass=masses['k'], root_type=1)
        self.k2.p4 = lambda : p4_with_mass(particle=self.k2, mass=masses['k'], root_type=1)
        self.k1.mass = lambda : self.k1.p4().mass()
        self.k2.mass = lambda : self.k2.p4().mass()
        self.k1.energy = lambda : self.k1.p4().energy()
        self.k2.energy = lambda : self.k2.p4().energy()

    def compute_vtx(self, full=False):
        # phi(1020) vertex
        tofit.clear()
        tofit.push_back(self.k1.bestTrack())
        tofit.push_back(self.k2.bestTrack())
        self.vtx = vtxfit.Fit(tofit)

        if self.vtx.isValid():
            self.vtx = compute_vertex_quantities(self.vtx, self.bs, self.p4(), self.pv, full)

    # Phi candidate kinematics
    def p4(self):
        return self.k1.p4() + self.k2.p4()
    def pt(self):
        return self.p4().pt()
    def eta(self):
        return self.p4().eta()
    def phi(self):
        return self.p4().phi()
    def energy(self):
        return self.p4().energy()
    def mass(self):
        return self.p4().mass()
    def charge(self):
        return self.k1.charge() + self.k2.charge()
    def dr_kk(self):
        return deltaR(self.k1, self.k2)        
    def dr_phi_k1(self):
        return deltaR(self, self.k1)
    def dr_phi_k2(self):
        return deltaR(self, self.k2)
    def max_dr_phi_k(self):
        return max(self.dr_phi_k1(), self.dr_phi_k2())

##########################################################################################
class DsCandidate():
    def __init__(self, phi, pi, bs, pv):
        self.phi1020 = phi
        self.pi      = pi
        self.bs      = bs
        self.pv      = pv 
 
        ############################################################
        self.pi_p4 = p4_with_mass(self.pi, masses['pi'], root_type=1)
        self.ds_p4 = self.phi1020.p4() + self.pi_p4

    def compute_vtx(self, full=False):
    
        # trigger the vtx computation of the Phi cand
        self.phi1020.compute_vtx(full=full)
        
        # Ds vertex
        tofit.clear()
        tofit.push_back(self.phi1020.k1.bestTrack())
        tofit.push_back(self.phi1020.k2.bestTrack())
        tofit.push_back(self.pi.bestTrack())
        self.vtx = vtxfit.Fit(tofit)

        if self.vtx.isValid():
            self.vtx = compute_vertex_quantities(self.vtx, self.bs, self.ds_p4, self.pv, full)

    # Ds candidate kinematics
    def p4(self):
        return self.ds_p4
    def pt(self):
        return self.p4().pt()
    def eta(self):
        return self.p4().eta()
    def phi(self):
        return self.p4().phi()
    def energy(self):
        return self.p4().energy()
    def mass(self):
        return self.p4().mass()
    def charge(self):
        return self.phi1020.charge() + self.pi.charge()
    def dr_phi_pi(self):
        return deltaR(self.phi1020, self.pi)        
    def dr_ds_k1(self):
        return deltaR(self, self.phi1020.k1)
    def dr_ds_k2(self):
        return deltaR(self, self.phi1020.k2)
    def dr_ds_pi(self):
        return deltaR(self, self.pi)
    def dr_ds_k1(self):
        return deltaR(self, self.phi1020.k1)
    def dr_ds_k2(self):
        return deltaR(self, self.phi1020.k2)

##########################################################################################
class RDsCandidate():
    '''
    '''
    def __init__(self, mu, kaons, pion, vertices, beamspot):
        # muon
        self.mu = mu
        self.mu.iso03 = mu.pfIsolationR03()
        self.mu.iso04 = mu.pfIsolationR04()
        
        # choose as PV the one that's closest to the leading muon in the dz parameter
        self.pv = sorted( [vtx for vtx in vertices], key = lambda vtx : abs( self.mu.bestTrack().dz(vtx.position() ) ) )[0]
        self.mu.pv = self.pv
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
        self.mu.bs = self.bs
        
        ############################################################
        self.kaons = sorted(kaons, key = lambda tk : tk.pt(), reverse = True)
        self.k1 = self.kaons[0]
        self.k2 = self.kaons[1]        
        self.k1.p4 = lambda : p4_with_mass(particle=self.k1, mass=masses['k'], root_type=1)
        self.k2.p4 = lambda : p4_with_mass(particle=self.k1, mass=masses['k'], root_type=1)
                
        self.pi = pion

        self.k1.pv = self.pv
        self.k2.pv = self.pv
        self.pi.pv = self.pv

        self.k1.bs = self.bs
        self.k2.bs = self.bs
        self.pi.bs = self.bs
        
        ############################################################
        self.phi1020 = PhiCandidate(self.k1     , self.k2, self.bs, self.pv)
        self.ds      = DsCandidate (self.phi1020, self.pi, self.bs, self.pv)
        
        ############################################################
        self.bs_p4 = self.ds.p4() + self.mu.p4()
        self.bs_p4_coll = self.bs_p4 * masses['bs'] / self.bs_p4.mass() 

    
    def compute_kinematics(self):
        
        if self.pi.charge() == self.k1.charge():
            self.k = self.k1
        else:
            self.k = self.k2

        # ROOT, geez... need TLorentzVectors for boost operations
        b_p4_tlv   = p4_with_mass(self.bs_p4_coll  , self.bs_p4_coll.mass(), root_type=2)
        mu_p4_tlv  = p4_with_mass(self.mu.p4()     , self.mu.mass()        , root_type=2)
        ds_p4_tlv  = p4_with_mass(self.ds.p4()     , self.ds.mass()        , root_type=2)
        phi_p4_tlv = p4_with_mass(self.phi1020.p4(), self.phi1020.mass()   , root_type=2)
        pi_p4_tlv  = p4_with_mass(self.pi.p4()     , self.pi.mass()        , root_type=2)
        k_p4_tlv   = p4_with_mass(self.k.p4()      , masses['k']           , root_type=2)

        # Bs boost
        self.boost = b_p4_tlv.BoostVector()
        self.beta  = b_p4_tlv.Beta()
        self.gamma = b_p4_tlv.Gamma()

        # Ds boost
        self.ds.boost = ds_p4_tlv.BoostVector()

        # phi(1020) boost
        self.phi1020.boost = phi_p4_tlv.BoostVector()
                
        # muon p4 in Bs rest frame
        mu_p4_in_b_rf = mu_p4_tlv.Clone() 
        mu_p4_in_b_rf.Boost(-self.boost)
        self.mu.p4_in_b_rf = mu_p4_in_b_rf

        # Ds p4 in Bs rest frame
        # the W* p4 in Bs rest frame opposite to that of the Ds
        ds_p4_in_b_rf = ds_p4_tlv.Clone()
        ds_p4_in_b_rf.Boost(-self.boost)
        self.ds.p4_in_b_rf = ds_p4_in_b_rf

        # W* opposite to Ds in the Bs restframe
        # mass equal to sqrt(q2) I presume
        w_p4_in_b_rf = ROOT.TLorentzVector() 
        w_p4_in_b_rf.SetVectM(-ds_p4_in_b_rf.Vect(), np.sqrt(self.q2()))
        w_p4_tlv = w_p4_in_b_rf.Clone()
        w_p4_tlv.Boost(self.boost)
        
        self.w = w_p4_tlv
        self.w.boost = self.w.BoostVector()
        self.w.p4_in_b_rf = w_p4_in_b_rf

        # phi(1020) p4 in Ds rest frame
        phi_p4_in_ds_rf = phi_p4_tlv.Clone() 
        phi_p4_in_ds_rf.Boost(-self.ds.boost)
        self.phi1020.p4_in_ds_rf = phi_p4_in_ds_rf

        # phi(1020) p4 in Bs rest frame
        phi_p4_in_b_rf = phi_p4_tlv.Clone() 
        phi_p4_in_b_rf.Boost(-self.boost)
        self.phi1020.p4_in_b_rf = phi_p4_in_b_rf

        # pion p4 in Ds rest frame (should be opposite to that of phi1020)
        pi_p4_in_ds_rf = pi_p4_tlv.Clone() 
        pi_p4_in_ds_rf.Boost(-self.ds.boost)
        self.pi.p4_in_ds_rf = pi_p4_in_ds_rf

        # pion p4 in phi(1020) rest frame
        pi_p4_in_phi_rf = pi_p4_tlv.Clone() 
        pi_p4_in_phi_rf.Boost(-self.phi1020.boost)
        self.pi.p4_in_phi_rf = pi_p4_in_phi_rf

        # kaon with same charge as pi p4 in phi(1020) rest frame
        k_p4_in_phi_rf = k_p4_tlv.Clone() 
        k_p4_in_phi_rf.Boost(-self.phi1020.boost)
        self.k.p4_in_phi_rf = k_p4_in_phi_rf

        # muon p4 in W* rest frame 
        mu_p4_in_w_rf = mu_p4_tlv.Clone() 
        mu_p4_in_w_rf.Boost(-self.w.boost)
        self.mu.p4_in_w_rf = mu_p4_in_w_rf
        
        # validation: phi and pi must be back to back
        # then this angle must by pi 3.1415
        # print(np.arccos(self.phi1020.p4_in_ds_rf.Vect().Unit().Dot(self.pi.p4_in_ds_rf.Vect().Unit())))
        
        # validation, these two must complement to 3.1415
        self.helicity_theta_pi_ds  = self.pi     .p4_in_ds_rf .Vect().Angle(self.ds.p4_in_b_rf  .Vect())
        self.helicity_theta_phi_ds = self.phi1020.p4_in_ds_rf .Vect().Angle(self.ds.p4_in_b_rf  .Vect())
        self.helicity_theta_mu_w   = self.mu     .p4_in_w_rf  .Vect().Angle(self.w .p4_in_b_rf  .Vect())
        self.helicity_theta_k_pi   = self.pi     .p4_in_phi_rf.Vect().Angle(self.k .p4_in_phi_rf.Vect())
        
        # now the angle between the two decay planes, Ds and W*
        # So, I'll create two vectors orthogonal to the planes, using vector product
        # and take the angle between them
        self.w_plane_normal  = self.mu.p4_in_w_rf.Vect() .Cross(self.w .p4_in_b_rf.Vect())       
        self.ds_plane_normal = self.pi.p4_in_ds_rf.Vect().Cross(self.ds.p4_in_b_rf.Vect())       
        self.helicity_phi_w_ds = self.w_plane_normal.Angle(self.ds_plane_normal)

        # another angle between "some planes"
        self.phi_plane_normal = self.pi.p4_in_phi_rf.Vect().Cross(self.k .p4_in_phi_rf.Vect())       
        self.helicity_phi_phi_ds = self.phi_plane_normal.Angle(self.ds_plane_normal)

        
    def compute_vtx(self, full=False):
    
        # trigger the vtx computation of the Ds cand
        self.ds.compute_vtx(full=True)

        ############################################################
        # Bs vertex
        tofit.clear()
        tofit.push_back(self.phi1020.k1.bestTrack())
        tofit.push_back(self.phi1020.k2.bestTrack())
        tofit.push_back(self.ds.pi.bestTrack())
        tofit.push_back(self.mu.bestTrack())
        self.vtx = vtxfit.Fit(tofit)
        
        if self.vtx.isValid():
            self.vtx, self.p4_par, self.p4_perp, self.mcorr = compute_vertex_quantities(self.vtx, self.bs, self.p4(), self.pv, full=True)

    def check_covariances(self):
        '''
        Sometimes tracks have negative covariance matrices, because of
        MINIAOD compression.
        If that happens, then it screws the KinematicFit
        So, we check and fix the covariance matrix to be pos def
        ''' 
        # FIXME! it was better written, but it messes up somehow, so I go full on pedant
        old_trk_k1 = self.k1.bestTrack()
        old_trk_k2 = self.k2.bestTrack()
        old_trk_pi = self.pi.bestTrack()
        old_trk_mu = self.mu.bestTrack()
        
        new_trk_k1 = fix_track(old_trk_k1)
        new_trk_k2 = fix_track(old_trk_k2)
        new_trk_pi = fix_track(old_trk_pi)
        new_trk_mu = fix_track(old_trk_mu)
        
        self.k1.bestTrack = lambda : new_trk_k1
        self.k2.bestTrack = lambda : new_trk_k2
        self.pi.bestTrack = lambda : new_trk_pi
        self.mu.bestTrack = lambda : new_trk_mu
       
    # Bs candidate kinematics
    def p4(self):
        return self.bs_p4
    def pt(self):
        return self.p4().pt()
    def eta(self):
        return self.p4().eta()
    def phi(self):
        return self.p4().phi()
    def energy(self):
        return self.p4().energy()
    def mass(self):
        return self.p4().mass()
    def px(self):
        return self.p4().px()
    def py(self):
        return self.p4().py()
    def pz(self):
        return self.p4().pz()

    # Bs candidate kinematics COLLINEAR APPROX
    def p4_collinear(self):
        return self.bs_p4_coll
    def pt_collinear(self):
        return self.p4_collinear().pt()
    def eta_collinear(self):
        return self.p4_collinear().eta()
    def phi_collinear(self):
        return self.p4_collinear().phi()
    def energy_collinear(self):
        return self.p4_collinear().energy()
    def mass_collinear(self):
        return self.p4_collinear().mass()
    def px_collinear(self):
        return self.p4(p4_collinear).px()
    def py_collinear(self):
        return self.p4_collinear().py()
    def pz_collinear(self):
        return self.p4_collinear().pz()

    # Bs corrected mass
    def mass_corrected(self):    
        return self.mcorr

    # Bs relevant kinematic quantities
    def m2_miss(self):
        return (self.p4_collinear() - self.mu.p4() - self.ds.p4()).mass2()
    def q2(self):
        return (self.p4_collinear() - self.ds.p4()).mass2()
    def e_star_mu(self):
        return self.mu.p4_in_b_rf.E()
    def e_hash_mu(self):
        return self.mu.p4_in_w_rf.E()
    def pt_miss_sca(self):
        return self.pt_collinear() - self.mu.pt() - self.ds.pt()
    def pt_miss_vec(self):
        return (self.p4_collinear() - self.mu.p4() - self.ds.p4()).pt()
    def ptvar(self):
        return self.ds.p4().pt() - self.mu.pt()

    # Helicity angles
    def theta_pi_ds(self):
        return self.helicity_theta_pi_ds 
    def theta_phi_ds(self):
        return self.helicity_theta_phi_ds 
    def theta_mu_w(self):
        return self.helicity_theta_mu_w 
    def phi_w_ds(self):
        return self.helicity_phi_w_ds
    def theta_k_pi(self):
        return self.helicity_theta_k_pi
    def phi_phi_ds(self):
        return self.helicity_phi_phi_ds 

    # charge
    def charge(self):
        return self.ds.charge() + self.mu.charge()

    # various delta R
    def dr_bs_pi(self):
        return deltaR(self, self.pi)        
    def dr_bs_k1(self):
        return deltaR(self, self.k1)
    def dr_bs_k2(self):
        return deltaR(self, self.k2)
    def dr_bs_mu(self):
        return deltaR(self, self.mu)
    def dr_bs_ds(self):
        return deltaR(self, self.ds)
    def dr_mu_ds(self):
        return deltaR(self.ds, self.mu)
    def dr_mu_k1(self):
        return deltaR(self.ds, self.k1)
    def dr_mu_k2(self):
        return deltaR(self.ds, self.k2)
    def dr_mu_pi(self):
        return deltaR(self.ds, self.pi)
    def max_dr_mu_kkpi(self):
        return max([self.dr_mu_k1(), self.dr_mu_k2(), self.dr_mu_pi()])


