import numpy as np
from scipy import stats
from itertools import product, combinations
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from Bmmm.Analysis.utils import masses, is_pos_def, convert_cov, fix_track, compute_IP3D
from Bmmm.Analysis.RJPsiNuReco import reconstruct

import ROOT
ROOT.gSystem.Load('libBmmmAnalysis')
from ROOT import KVFitter # VertexDistance3D is contained here, dirt trick!!
from ROOT import RJpsiKinVtxFitter

# make these available everywhere in here
global vtxfit
vtxfit = KVFitter()
global tofit
tofit = ROOT.std.vector('reco::Track')()
global kinfit
kinfit = RJpsiKinVtxFitter()

M_BC = 6.27447

class RJpsiCandidate(ROOT.reco.CompositeCandidate):
    '''
    Bc -> J/psi(-> mu mu) mu candidate.

    The three muons are kept both as a flat pt-sorted list (self.muons) and
    grouped (self.jpsi_muons + self.mu, the bachelor muon). The J/psi is stored
    as a reco::CompositeCandidate (self.jpsi) and the object itself is the full
    3-muon system.

    Vertex- and IP-related quantities are NOT computed in __init__: call
    compute_vtx_quantities(vertices, beamspot) for that.
    '''
    def __init__(self, jpsi_muons, mu3):

        super().__init__()

        for imu in jpsi_muons + [mu3]:
            imu.cov = self.convert_cov(imu.bestTrack().covariance())
            imu.is_cov_pos_def = self.is_pos_def(imu.cov)

        self.muons      = sorted(jpsi_muons + [mu3], key = lambda x : x.pt(), reverse = True)
        self.jpsi_muons = sorted(jpsi_muons        , key = lambda x : x.pt(), reverse = True)

        self.mu = mu3

        self.jpsi = ROOT.reco.CompositeCandidate()
        self.jpsi.addDaughter(self.jpsi_muons[0])
        self.jpsi.addDaughter(self.jpsi_muons[1])
        self.jpsi.setP4(self.jpsi.daughter(0).p4() + self.jpsi.daughter(1).p4())
        self.jpsi.setCharge(int(self.jpsi_muons[0].charge() + self.jpsi_muons[1].charge()))
        self.jpsi.setPdgId(443)           # optional: tag as J/psi

        self.setP4(self.jpsi.p4() + self.mu.p4())
        self.setCharge(int(sum([imu.charge() for imu in self.muons])))
        self.setPdgId(int(541 * np.sign(self.mu.charge())))



    ##########################################################################
    #####      HELICITY ANGLES
    ##########################################################################
    def compute_helicity_angles(self):
        '''Reco helicity angles for five Bc-momentum hypotheses:
            ev_jpsi : equal-velocity p4 along the PV->(J/psi vtx) flight direction
            ev_sv   : equal-velocity p4 along the PV->(3mu  vtx) flight direction
            coll    : pure collinear (equal-betagamma) p4 -- direction == visible p
            nu1/nu2 : the two solutions of the exact neutrino-pz quadratic
        PV is the hybrid reference (beamspot x,y + PV z) carried by Bdirection_*.
        '''
        nan = float('nan')
    
        visible_p4 = self.jpsi_rfp4 + self.mu.p4()
    
        # mu- of the J/psi pair = the two muons that are not the bachelor
        mu_minus = next((m for m in self.jpsi_muons if m.charge() < 0), None)
        p4_mu_v  = mu_minus.p4() if mu_minus is not None else None
        p4_lep   = self.mu.p4()
    
        bc_p4     = {}
        bdir_jpsi = getattr(self, 'Bdirection_jpsi', None)
        bdir_sv   = getattr(self, 'Bdirection_sv',   None)
    
        if bdir_jpsi is not None:
            _, bc_p4['jpsi'] = self.equal_velocity_p4(bdir_jpsi, visible_p4)
        if bdir_sv is not None:
            _, bc_p4['sv']   = self.equal_velocity_p4(bdir_sv,   visible_p4)
    
        # pure collinear: equal-velocity along the visible momentum itself
        # (self-consistent: same magnitude AND direction, no flight-direction input)
        coll_dir = ROOT.Math.XYZVector(visible_p4.px(), visible_p4.py(), visible_p4.pz())
        _, bc_p4['coll'] = self.equal_velocity_p4(coll_dir, visible_p4)
    
        # two neutrino solutions of the quadratic (mass-constrained, 3mu-vtx dir).
        # reuse the inspector's math*_b_p4_sv if already computed, else solve here.
        nu1 = getattr(self, 'math1_b_p4_sv', None)
        nu2 = getattr(self, 'math2_b_p4_sv', None)
        if nu1 is None or nu2 is None:
            nu_dir = bdir_sv if bdir_sv is not None else coll_dir
            sols = reconstruct(visible_p4, nu_dir, m_parent=M_BC, clamp_negative_disc=True)
            if nu1 is None and len(sols) >= 1: nu1 = visible_p4 + sols[0].p4_nu
            if nu2 is None and len(sols) >= 2: nu2 = visible_p4 + sols[1].p4_nu
        if nu1 is not None: bc_p4['nu1'] = nu1
        if nu2 is not None: bc_p4['nu2'] = nu2
    
        for key in ('jpsi', 'sv', 'coll', 'nu1', 'nu2'):
            p4 = bc_p4.get(key, None)
            if p4 is None or p4_mu_v is None:
                cv = cl = ch = nan
            else:
                cv, cl, ch = self.helicity_angles(p4, self.jpsi_rfp4, p4_mu_v, p4_lep)
            setattr(self, 'cos_theta_v_%s' % key, cv)
            setattr(self, 'cos_theta_l_%s' % key, cl)
            setattr(self, 'chi_%s'         % key, ch)

    ##########################################################################
    #####      VERTEXING
    ##########################################################################
    def compute_vtx_quantities(self, vertices, beamspot):
        '''
        Fit the J/psi (2-muon) and the full (3-muon) vertices, choose the
        primary vertex and, for each of the two secondary vertices, compute:
          - the 2D distance from the beamspot and its significance      (lxy)
          - the 3D distance from the primary vertex and its significance (lxyz)
          - the cosine of the 2D and 3D pointing angles            (cos2d, cos3d)
        Finally compute the signed 3D impact parameter of the bachelor muon
        (see compute_ip).

        Quantities for the 3-muon vertex are stored with no prefix, those for the
        J/psi vertex with a 'jpsi_' prefix (e.g. self.lxy vs self.jpsi_lxy).
        '''

        # ----- fit the two vertices (generalized N-body fitter) -------------
#         self.jpsi_vertex_tree = self.fit_vertex(self.jpsi_muons)  # 2-muon J/psi vertex
        self.jpsi_vertex_tree = self.fit_jpsi_vertex(self.jpsi_muons) # mass-constrained dimuon
        self.vertex_tree      = self.fit_vertex(self.muons)           # 3-muon vertex

        self.jpsi_good_vtx = self.is_good_vtx(self.jpsi_vertex_tree)
        self.good_vtx      = self.is_good_vtx(self.vertex_tree)

        # ----- choose the primary vertex ------------------------------------
        # PV = the one with the smallest 3D impact parameter w.r.t. the flight
        # direction of the 3-muon candidate, i.e. the line through the 3-muon
        # secondary vertex along the candidate momentum.
        # If the 3-muon vertex failed, fall back to the PV closest in dz to the
        # leading muon.
        if self.good_vtx:
            self.vertex_tree.movePointerToTheTop()
            sv = self.vertex_tree.currentDecayVertex().get()
            pv_idx, ip3d_min = -1, np.inf
            for idx, ivtx in enumerate(vertices):
                ip3d = compute_IP3D(ivtx, sv.position(), self.p4().Vect())
                if ip3d < ip3d_min:
                    pv_idx, ip3d_min = idx, ip3d
            self.pv = vertices[pv_idx]
        else:
            self.pv = sorted(
                [vtx for vtx in vertices],
                key = lambda vtx : abs(self.muons[0].bestTrack().dz(vtx.position())),
            )[0]

        # ----- beamspot as a reco::Vertex at the PV z position --------------
        self.bs = self.build_beamspot_vertex(beamspot, self.pv.z())

        # ----- displacement / pointing-angle quantities --------------------
        # the 3-muon vertex uses the full candidate momentum (self),
        # the J/psi vertex uses the dimuon momentum (self.jpsi)
        self.compute_jpsi_refit()
        self.compute_displacement(self.vertex_tree     , self.good_vtx     , self          , prefix=''     )
        self.compute_displacement(self.jpsi_vertex_tree, self.jpsi_good_vtx, self.jpsi_rfp4, prefix='jpsi_')

        # ----- bachelor-muon impact parameters ------------------------------
        self.compute_ip()

        if self.jpsi_good_vtx and self.good_vtx:
            visible_p4 = self.jpsi_rfp4 + self.mu.p4()
       
            self.bc_directions = {
                'jpsi': getattr(self, 'Bdirection_jpsi', None), 
                'sv'  : getattr(self, 'Bdirection_sv', None),
            }
            
            for label, direction in self.bc_directions.items():
                if direction is None: continue
                p3, p4 = self.equal_velocity_p4(direction, visible_p4)
                setattr(self, 'bc_full_p_%s'  % label, p3)
                setattr(self, 'bc_full_p4_%s' % label, p4)        
                
                p4_par  = self.p4().Vect().Dot(direction.unit())                   
                p4_perp = np.sqrt(self.p4().Vect().Mag2() - p4_par*p4_par)
                mcorr   = np.sqrt(self.p4().mass()*self.p4().mass() + p4_perp*p4_perp) + p4_perp

                setattr(self, 'p4_par_%s'  % label, p4_par )        
                setattr(self, 'p4_perp_%s' % label, p4_perp)        
                setattr(self, 'mcorr_%s'   % label, mcorr  )        
            
            self.p4_collinear = (self.mu.p4() + self.jpsi_rfp4) * M_BC / (self.mu.p4() + self.jpsi_rfp4).mass()
        
            self.compute_jet_track_distance()

    @staticmethod
    def fit_vertex(muons):
        '''
        Fit an arbitrary number of muons to a common vertex with the generalized
        RJpsiKinVtxFitter, assigning each muon the muon mass hypothesis.
        Returns the kinematic decay tree (empty if the fit fails).
        '''
        tracks   = ROOT.std.vector('reco::Track')()
        mass_hyp = ROOT.std.vector('double')()
        for imu in muons:
            tracks.push_back(imu.bestTrack())
            mass_hyp.push_back(masses['mu'])
        return kinfit.Fit(tracks, mass_hyp)

    @staticmethod
    def fit_jpsi_vertex(jpsi_muons):
        '''
        Fit the two J/psi muons to a common vertex WITH the J/psi mass constraint
        (TwoTrackMassKinematicConstraint). Drop-in replacement for fit_vertex on the
        dimuon: same return type, so all downstream handling is unchanged.
        '''
        mu1, mu2 = jpsi_muons[0], jpsi_muons[1]
        return kinfit.Fit2BodyMassConstraint(
            mu1.bestTrack(), mu2.bestTrack(),
            masses['mu'], masses['mu'], masses['jpsi'],   # PDG J/psi = 3.0969 GeV
        )

    def compute_jet_track_distance(self):
        track = self.mu.bestTrack()
        
        # PV reference: beamspot transverse position, PV longitudinal. Built once and
        # shared by both flight-direction hypotheses (only the IP sign differs between them).
        pv_ref = ROOT.reco.Vertex(
            ROOT.reco.Vertex.Point(self.bs.position().x(),    # beamspot x
                                   self.bs.position().y(),    # beamspot y
                                   self.pv.position().z()),   # PV z
            self.pv.error(), self.pv.chi2(), self.pv.ndof(), self.pv.tracksSize()
        )
        
        # 2 x 2:  {direction from jpsi-vtx | from 3mu-vtx}  x  {wrt PV | wrt that same SV}
        for label, vtx in self.bc_vertices.items():            # {'jpsi': self.jpsi_vtx, 'sv': self.vtx}
            if not self.jpsi_good_vtx: continue
            if not self.good_vtx: continue
            direction  = getattr(self, 'Bdirection_%s' % label)
            dx, dy, dz = direction.x(), direction.y(), direction.z()

            references = {'pv': pv_ref, 'sv': self.kin_to_reco_vertex(vtx)}
            for ref, ref_vtx in references.items():
                jtd = kinfit.jetTrackDistance(track, dx, dy, dz, ref_vtx)
                #print(f"label:{label}\t ref:{ref}\t jtd.first:{jtd.first}\t jtd.second:{jtd.second.value()}")
                setattr(self, 'mu_dist_along_b_dir_%s_%s' % (label, ref), jtd.first)
                
            # the distance bewteen the two lines is independent of the vertex
            # just use the last vertex from the previous loop
            jtd = kinfit.jetTrackDistance(track, dx, dy, dz, ref_vtx) 
            #print(f"label:{label}\t ref:{ref}\t jtd.first:{jtd.first}\t jtd.second:{jtd.second.value()}")
            setattr(self, f'mu_dist_to_b_dir_{label}'    , jtd.second.value())
#             setattr(self, f'mu_dist_to_b_dir_{label}_err', jtd.second.error())
#             setattr(self, f'mu_dist_to_b_dir_{label}_sig', jtd.second.significance())
        
#         import ipdb ; ipdb.set_trace()


    @staticmethod
    def is_good_vtx(vertex_tree):
        '''True if the kinematic decay tree exists, is non-empty and valid.'''
        if not vertex_tree:
            return False
        return (not vertex_tree.isEmpty()) and vertex_tree.isValid()

    @staticmethod
    def build_beamspot_vertex(beamspot, z):
        '''
        Build a reco::Vertex from the beamspot, evaluated at the given z, so it
        can be passed to the VertexDistance / IPTools tools alongside a vertex.
        '''
        bs_point = ROOT.reco.Vertex.Point(
            beamspot.x(z),
            beamspot.y(z),
            beamspot.z0(),
        )
        bs_error = beamspot.covariance3D()
        chi2 = 0.
        ndof = 0.
        return ROOT.reco.Vertex(bs_point, bs_error, chi2, ndof, 3)

    @staticmethod
    def kin_to_reco_vertex(kin_vtx, ndim=2):
        '''
        Wrap a KinematicVertex into a reco::Vertex (its position + 3x3 covariance),
        so it can be handed to IPTools, which only accepts reco::Vertex.
        '''
        pos   = kin_vtx.position()
        point = ROOT.reco.Vertex.Point(pos.x(), pos.y(), pos.z())
        error = kin_vtx.error().matrix()   # AlgebraicSymMatrix33
        return ROOT.reco.Vertex(point, error, kin_vtx.chiSquared(), kin_vtx.degreesOfFreedom(), ndim)

    def compute_displacement(self, vertex_tree, good_vtx, cand, prefix=''):
        '''
        For a single fitted vertex, compute and store (with the given prefix):
          {prefix}vtx       : the KinematicVertex
          {prefix}vtx_chi2  : vertex chi2
          {prefix}vtx_ndof  : vertex ndof
          {prefix}vtx_prob  : vertex fit probability
          {prefix}lxy       : 2D distance from the beamspot      (Measurement1D)
          {prefix}lxyz      : 3D distance from the primary vertex (Measurement1D)
          {prefix}cos2d     : cosine of the 2D pointing angle (beamspot -> SV vs pT)
          {prefix}cos3d     : cosine of the 3D pointing angle (PV -> SV vs p)

        Measurement1D objects expose .value(), .error() and .significance().
        'cand' is the (composite) candidate whose momentum defines the pointing
        angle: the full 3-muon system for the 3-muon vertex, the J/psi for the
        J/psi vertex.
        '''

        # if the fit failed, store NaNs and return
        if not good_vtx:
            setattr(self, prefix + 'vtx', None)
            for q in ['vtx', 'lxy', 'lxyz', 'cos2d', 'cos3d']:
                setattr(self, prefix + q, np.nan)
            return

        vertex_tree.movePointerToTheTop()
        vtx = vertex_tree.currentDecayVertex().get()
        setattr(self, prefix + 'vtx', vtx)

        # vertex quality
        chi2 = vtx.chiSquared()
        ndof = vtx.degreesOfFreedom()
        setattr(self, prefix + 'vtx_chi2', chi2)
        setattr(self, prefix + 'vtx_ndof', ndof)
        setattr(self, prefix + 'vtx_prob', 1. - stats.chi2.cdf(chi2, ndof))

        # 2D distance from the beamspot (significance via Measurement1D)
        lxy = ROOT.VertexDistanceXY().distance(self.bs, vtx.vertexState())
        setattr(self, prefix + 'lxy', lxy)

        # 3D distance from the primary vertex (significance via Measurement1D)
        lxyz = ROOT.VertexDistance3D().distance(self.pv, vtx.vertexState())
        setattr(self, prefix + 'lxyz', lxyz)

        # 3-vector helper
        Vector3D = ROOT.Math.DisplacementVector3D(
            'ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')

        # 2D pointing angle: transverse displacement always from the beamspot
        vect_lxy = Vector3D(
            vtx.position().x() - self.bs.position().x(),
            vtx.position().y() - self.bs.position().y(),
            0.,
        )
        vect_pt = Vector3D(cand.px(), cand.py(), 0.)
        setattr(self, prefix + 'cos2d',
                vect_pt.Dot(vect_lxy) / (vect_pt.R() * vect_lxy.R()) if vect_lxy.R() > 0. else np.nan)

        # 3D pointing angle: full displacement from the primary vertex
        vect_lxyz = Vector3D(
            vtx.position().x() - self.pv.position().x(),
            vtx.position().y() - self.pv.position().y(),
            vtx.position().z() - self.pv.position().z(),
        )
        vect_p = Vector3D(cand.px(), cand.py(), cand.pz())
        setattr(self, prefix + 'cos3d',
                vect_p.Dot(vect_lxyz) / (vect_p.R() * vect_lxyz.R()) if vect_lxyz.R() > 0. else np.nan)

    def compute_ip(self):
        '''
        Signed 3D impact parameter of the bachelor muon (self.mu), computed twice:
          - w.r.t. the primary vertex        -> mu_ip3d / mu_ip3d_err / mu_ip3d_sig
          - w.r.t. the J/psi (dimuon) vertex -> mu_jpsi_ip3d / _err / _sig

        Both are lifetime-signed along the Bc flight direction (PV -> 3-muon SV),
        using IPTools::signedImpactParameter3D (wrapped in RJpsiKinVtxFitter).
        The bachelor muon is NOT part of the J/psi vertex fit, so its IP w.r.t.
        that vertex is unbiased and small for genuine Bc -> J/psi mu decays.
        '''

        # Bc flight direction (PV -> SV), one per vertex hypothesis
        self.bc_vertices = {'jpsi': self.jpsi_vtx, 'sv': self.vtx}
        for label, vtx in self.bc_vertices.items():
            if not self.jpsi_good_vtx: continue
            if not self.good_vtx: continue
            setattr(self, 'Bdirection_%s' % label, self.flight_direction(vtx, self.bs, self.pv))

        track = self.mu.bestTrack()
        
        # PV reference: beamspot transverse position, PV longitudinal. Built once and
        # shared by both flight-direction hypotheses (only the IP sign differs between them).
        pv_ref = ROOT.reco.Vertex(
            ROOT.reco.Vertex.Point(self.bs.position().x(),    # beamspot x
                                   self.bs.position().y(),    # beamspot y
                                   self.pv.position().z()),   # PV z
            self.pv.error(), self.pv.chi2(), self.pv.ndof(), self.pv.tracksSize()
        )
        
        # 2 x 2:  {direction from jpsi-vtx | from 3mu-vtx}  x  {wrt PV | wrt that same SV}
        for label, vtx in self.bc_vertices.items():            # {'jpsi': self.jpsi_vtx, 'sv': self.vtx}
            if not self.jpsi_good_vtx: continue
            if not self.good_vtx: continue
            direction  = getattr(self, 'Bdirection_%s' % label)
            dx, dy, dz = direction.x(), direction.y(), direction.z()
        
            references = {'pv': pv_ref, 'sv': self.kin_to_reco_vertex(vtx)}
            for ref, ref_vtx in references.items():
                ip = kinfit.signedIP3D(track, dx, dy, dz, ref_vtx)
                setattr(self, 'mu_ip3d_%s_%s'     % (label, ref), ip.value())
                setattr(self, 'mu_ip3d_%s_%s_err' % (label, ref), ip.error())
                setattr(self, 'mu_ip3d_%s_%s_sig' % (label, ref), ip.significance())
        
    @staticmethod
    def refitted_p4(kin_particle):
        '''
        Lorentz vector (PxPyPzE) from a fitted KinematicParticle: the track momentum
        AFTER the (mass-constrained) common-vertex fit. Works for any tree (also the
        3-muon one). Uses kinematicParameters().vector() = (x,y,z, px,py,pz, m).
        '''
        v  = kin_particle.currentState().kinematicParameters().vector()
        px, py, pz, m = v.At(3), v.At(4), v.At(5), v.At(6)
        e  = np.sqrt(px*px + py*py + pz*pz + m*m)
        return ROOT.Math.LorentzVector('ROOT::Math::PxPyPzE4D<double>')(px, py, pz, e)

    @staticmethod
    def flight_direction(vtx, bs, pv):
        '''
        PV -> SV flight direction. Transverse part from the beamspot,
        longitudinal from the PV (matches the corrected-mass / IP convention).
        '''
        sv = vtx.position()
        return ROOT.Math.XYZVector(
            sv.x() - bs.position().x(),
            sv.y() - bs.position().y(),
            sv.z() - pv.position().z(),
        ) 

    @staticmethod
    def equal_velocity_p4(direction, visible_p4):
        '''
        Equal-velocity ("rest-frame") Bc momentum along `direction`:
            |p_Bc| = (M_BC / m_vis) * |p_vis| ,   E_Bc = sqrt(M_BC^2 + |p_Bc|^2)
        Returns (p3, p4): a ROOT XYZVector and a PxPyPzE4D LorentzVector.
        '''
        p3 = direction.unit() * (M_BC / visible_p4.mass() * visible_p4.P())
        p4 = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzE4D<double>')(
            p3.x(), p3.y(), p3.z(), np.sqrt(M_BC**2 + p3.Mag2())
        )
        return p3, p4

    def compute_jpsi_refit(self):
        '''
        Refitted muon momenta under the J/psi mass + common-vertex constraint, plus
        the refitted (constrained) J/psi momentum:
            self.jpsi_mu1_rfp4, self.jpsi_mu2_rfp4  -> refitted muon 4-momenta
            self.jpsi_rfp4                          -> refitted, mass-constrained J/psi
        Children follow the order given to the fit: jpsi_muons[0] -> mu1, [1] -> mu2.
        '''
        for q in ['jpsi_mu1_rfp4', 'jpsi_mu2_rfp4', 'jpsi_rfp4']:
            setattr(self, q, None)
    
        if not self.jpsi_good_vtx:
            return
    
        tree = self.jpsi_vertex_tree
    
        tree.movePointerToTheTop()                          # mass-constrained J/psi
        self.jpsi_rfp4 = self.refitted_p4(tree.currentParticle())
    
        tree.movePointerToTheFirstChild()                   # first refitted muon
        self.jpsi_muons[0].jpsi_rfp4 = self.refitted_p4(tree.currentParticle())
        tree.movePointerToTheNextChild()                    # second refitted muon
        self.jpsi_muons[1].jpsi_rfp4 = self.refitted_p4(tree.currentParticle())

        self.rfp4 = self.jpsi_rfp4 + self.mu.p4()        


    ##########################################################################
    #####      helicity angles
    ##########################################################################

    @staticmethod
    def _np4(p4):
        '''[E, px, py, pz] from a four-vector, or from a (p4, ...) tuple as
        returned by buildP4 / equal_velocity_p4.'''
        if not hasattr(p4, 'energy') and isinstance(p4, (tuple, list)):
            p4 = p4[0]
        return np.array([p4.energy(), p4.px(), p4.py(), p4.pz()], dtype=float)
        
    @staticmethod
    def _unit3(v):
        n = np.sqrt(float(v.dot(v)))
        return v / n if n > 0. else v
    
    @staticmethod
    def _boost(four, beta):
        '''Boost the 4-vector `four` into a frame moving with velocity `beta`
        (3-array) relative to the current frame. Pure numpy; sign convention
        matches RJPsiGenHistory.gen_helicity_angles.'''
        b2 = float(beta.dot(beta))
        if b2 <= 0.:
            return np.array(four, dtype=float)
        g  = 1. / np.sqrt(1. - b2)
        p  = np.asarray(four[1:], dtype=float)
        bp = float(beta.dot(p))
        e  = g * (four[0] - bp)
        pp = p + ((g - 1.) * bp / b2 - g * four[0]) * beta
        return np.array([e, pp[0], pp[1], pp[2]])
    
    @staticmethod
    def helicity_angles(p4_bc, p4_jpsi, p4_mu_v, p4_lep):
        '''Helicity angles of  Bc -> J/psi(->mu+ mu-) W*(->l nu)  for a given
        reconstructed Bc four-momentum hypothesis. Returns three plain floats
        (NaN where not computable):
    
            cos_theta_v : mu- in the J/psi rest frame  vs the J/psi flight
                          direction in the Bc rest frame
            cos_theta_l : charged lepton (bachelor mu) in the W* rest frame vs
                          the W* flight direction in the Bc rest frame
            chi         : signed angle between the J/psi and W* decay planes
    
        Conventions match RJPsiGenHistory.gen_helicity_angles, so reco and gen
        are directly comparable. NB: in the tau channel the bachelor mu is only a
        proxy for the true W* lepton, so cos_theta_l (and chi) are smeared there
        by design -- that smearing is part of what separates mu from tau.
        '''
        nan = float('nan')
        cos_v = cos_l = chi = nan
    
#         import ipdb ; ipdb.set_trace()
    
        p_bc   = RJpsiCandidate._np4(p4_bc)
        p_jpsi = RJpsiCandidate._np4(p4_jpsi)
        p_muv  = RJpsiCandidate._np4(p4_mu_v)
        p_lep  = RJpsiCandidate._np4(p4_lep)
        p_w    = p_bc - p_jpsi
    
        boost, unit = RJpsiCandidate._boost, RJpsiCandidate._unit3
    
        # cos_theta_v : J/psi -> mu+ mu-
        if p_jpsi[0] > 0.:
            bj = p_jpsi[1:] / p_jpsi[0]
            if float(bj.dot(bj)) < 1.:
                mu_j = boost(p_muv, bj)
                bc_j = boost(p_bc,  bj)
                ref  = -unit(bc_j[1:])               # J/psi flight dir in Bc frame
                cos_v = float(unit(mu_j[1:]).dot(ref))
    
        # cos_theta_l : W* -> l nu
        if p_w[0] > 0. and (p_w[0]**2 - float(p_w[1:].dot(p_w[1:]))) > 0.:
            bw = p_w[1:] / p_w[0]
            if float(bw.dot(bw)) < 1.:
                lep_w = boost(p_lep, bw)
                bc_w  = boost(p_bc,  bw)
                ref   = -unit(bc_w[1:])              # W* flight dir in Bc frame
                cos_l = float(unit(lep_w[1:]).dot(ref))
    
        # chi : dihedral between the two decay planes, in the Bc rest frame
        if p_bc[0] > 0.:
            bb = p_bc[1:] / p_bc[0]
            if float(bb.dot(bb)) < 1.:
                jb = boost(p_jpsi, bb)[1:]
                mb = boost(p_muv,  bb)[1:]
                lb = boost(p_lep,  bb)[1:]
                z  = unit(jb)                        # J/psi flight axis in Bc frame
                nv = unit(np.cross(mb, z))           # normal to J/psi plane
                nl = unit(np.cross(lb, z))           # normal to W* plane
                if float(nv.dot(nv)) > 0. and float(nl.dot(nl)) > 0.:
                    cc  = float(np.clip(nv.dot(nl), -1., 1.))
                    ss  = float(np.cross(nl, nv).dot(z))
                    chi = float(np.arctan2(ss, cc))
    
        return cos_v, cos_l, chi
        
    ##########################################################################
    #####      KINEMATICS (pt-sorted muons: muons[0], muons[1], muons[2])
    ##########################################################################
    def r(self):
        '''Cone radius: max distance between the 3-mu direction and a muon.'''
        return max([deltaR(self.eta(), self.phi(), imu.eta(), imu.phi()) for imu in self.muons])
    def max_dr(self):
        '''Max distance between any pair of muons.'''
        return max([deltaR(imu, jmu) for imu, jmu in combinations(self.muons, 2)])
    def dr12(self):
        return deltaR(self.muons[0], self.muons[1])
    def dr13(self):
        return deltaR(self.muons[0], self.muons[2])
    def dr23(self):
        return deltaR(self.muons[1], self.muons[2])
    def mass12(self):
        return (self.muons[0].p4() + self.muons[1].p4()).mass()
    def mass13(self):
        return (self.muons[0].p4() + self.muons[2].p4()).mass()
    def mass23(self):
        return (self.muons[1].p4() + self.muons[2].p4()).mass()
    def charge12(self):
        return self.muons[0].charge() + self.muons[1].charge()
    def charge13(self):
        return self.muons[0].charge() + self.muons[2].charge()
    def charge23(self):
        return self.muons[1].charge() + self.muons[2].charge()

    ##########################################################################
    #####      MISC HELPERS
    ##########################################################################
    def convert_cov(self, m):
        return np.array([[m(i,j) for j in range(m.kCols)] for i in range(m.kRows)])

    def is_pos_def(self, x):
        '''
        https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
        '''
        return np.all(np.linalg.eigvals(x) > 0)
