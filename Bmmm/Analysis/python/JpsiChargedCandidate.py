import numpy as np
from scipy import stats
from itertools import product, combinations
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch
from Bmmm.Analysis.utils import masses, is_pos_def, convert_cov, fix_track, compute_IP3D, p4_with_mass
from Bmmm.Analysis.RJPsiNuReco import reconstruct, M_BC   # M_BC: single source (RJPsiGenHistory)

import ROOT
ROOT.gSystem.Load('libBmmmAnalysis')
# RJpsiKinVtxFitter.h #includes VertexDistance3D/XY and IPTools, so this import
# also makes ROOT.VertexDistance3D / ROOT.VertexDistanceXY available (same
# "dirty trick" the old, otherwise-unused KVFitter import was kept around for).
from ROOT import RJpsiKinVtxFitter

# single fitter instance shared by all candidates
kinfit = RJpsiKinVtxFitter()

# ROOT template instantiations, hoisted out of the per-candidate hot loop
Vector3D      = ROOT.Math.DisplacementVector3D(
    'ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')
LorentzVector = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzE4D<double>')

# ----- signal-muon <-> track/candidate matching ----------------------------
# the muon best-track and the unpacked / packed candidate live in different
# collections, so they are matched by proximity, not by reference. Keep in sync
# with the C++ refitPVRemovingTracks defaults (drMatch / relPtMatch).
_MU_TRK_DR_MATCH    = 0.01    # dR(cand, muon) match window
_MU_TRK_RELPT_MATCH = 0.05    # |pt_cand - pt_mu| / pt_mu match window

# ----- PV-finder track filter (offlinePrimaryVertices TkFilterParameters) -----
# The closest-z PV association in refit_primary_vertex must be fed the same track
# set the offline PV reconstruction used; otherwise the refit input is broader
# (softer / lower-quality tracks, and all of lostTracks) than the real PV, which
# inflates pv_ntrk and degrades the refit resolution. These mirror the
# TrackFilterForPVFinding cuts of unsortedOfflinePrimaryVertices (the producer the
# old primaryVertexRefit cloned). VERIFY against your release, e.g.
#   print(process.primaryVertexRefit.TkFilterParameters.dumpPython())
# The values below are the standard offline defaults; trackQuality is "any", so
# no trackHighPurity() requirement is imposed. Packed-candidate quantities are
# stored at reduced precision, so the replica is close but not bit-exact, and
# soft tracks below the packing thresholds are simply absent from miniAOD.
_PV_TRK_MAX_NORM_CHI2  = 10.0   # maxNormalizedChi2
_PV_TRK_MIN_PIX_LAYERS = 2      # minPixelLayersWithHits
_PV_TRK_MIN_TRK_LAYERS = 5      # minSiliconLayersWithHits (pixel + strip)
_PV_TRK_MAX_D0_SIG     = 4.0    # maxD0Significance (transverse IP wrt the beamline)
_PV_TRK_MAX_D0_ERR     = 1.0    # maxD0Error [cm]
_PV_TRK_MAX_DZ_ERR     = 1.0    # maxDzError [cm]
_PV_TRK_MIN_PT         = 0.0    # minPt [GeV]
_PV_TRK_MAX_ETA        = 2.4

# ----- PV refit fitter config + vertex acceptance (WithBS vertexCollections) ---
# Match the offline PrimaryVertexProducer WithBS collection so the in-loop refit
# reproduces offlinePrimaryVerticesWithBS. VERIFY against your release dump.
_PV_AVF_CHI2CUTOFF     = 2.5    # AdaptiveVertexFitter annealing cutoff (default is 3.0!)
_PV_MIN_NDOF           = 2.0    # minNdof (WithBS); ndof = 2*sum(weights) - 3
_PV_MAX_DIST_TO_BEAM   = 1.0    # maxDistanceToBeam [cm]

# ----- PV refit track-set selection -------------------------------------------
# Primary: read the offline Deterministic-Annealing fit-track assignment straight
# from the packed candidates (pat::PackedCandidate::PVUsedInFit == 3) -- the tracks
# the offline PV fit used for the chosen PV. No re-clustering, no closest-z
# approximation, O(1) per candidate. Fall back to closest-z only if that yields too
# few tracks (association not populated, or the chosen-PV index is not aligned with
# the candidates' reference collection).
_PV_USED_IN_FIT          = 3    # pat::PackedCandidate::PVAssociationQuality::PVUsedInFit
_PV_REFIT_MIN_TRK        = 2    # minimum tracks to attempt a refit (AVF needs >= 2)
_PV_REFIT_MAX_DZ_TO_PV   = 0.1  # [cm] max |z(refit) - z(chosen PV)|; else fromPV picked
                                #      the wrong vertex -> closest-z fallback

# ----- PF isolation (replicates the muon-POG PFIsolation, custom PV) ---------
# pdgId sets and selections from the BPH vertexing+isolation slides (slides 18, 24).
_ISO_CONES        = (0.3, 0.4)                              # cone radii; suffixes 03, 04
_ISO_CH_ALL_IDS   = (211, 321, 11, 13, 2212, 999211)       # all charged iso candidates
_ISO_CH_HAD_IDS   = (211, 321, 2212, 999211)               # charged-hadron sum (NO leptons)
_ISO_NH_IDS       = (111, 130, 310, 2112)                  # neutral hadrons
_ISO_PH_IDS       = (22,)                                  # photons
_ISO_DR_VETO_CH   = 1e-4                                   # inner veto, charged
_ISO_DR_VETO_NH   = 1e-2                                   # inner veto, neutral
_ISO_PU_DR_VETO   = 0.01                                   # PU charged inner veto
_ISO_NEUTRAL_PT   = 0.5                                    # min pt, neutral candidates
_ISO_PU_PT        = 0.5                                    # min pt, PU charged candidates

class BachelorTrack(object):
    '''
    Lightweight view of a pat::PackedCandidate pseudo-track under a chosen mass
    hypothesis. Everything is delegated to the underlying track EXCEPT the
    4-momentum (p4 / mass / energy), which carry the assigned mass.

    A wrapper is used instead of monkey-patching the track itself so that the
    SAME physical track can be reconstructed under two mass hypotheses (kaon and
    pion) at the same time, each candidate/hypothesis holding its own p4, without
    one clobbering the other. Per-candidate attributes set downstream (cov, pv,
    bs, jet, gen_match, ...) live on the wrapper, not on the shared track.
    '''
    def __init__(self, track, mass):
        object.__setattr__(self, '_trk', track)
        object.__setattr__(self, '_p4',  p4_with_mass(track, mass, root_type=1))

    def p4(self):     return self._p4
    def mass(self):   return self._p4.mass()
    def energy(self): return self._p4.energy()

    def __getattr__(self, name):
        # only reached for attributes not found on the wrapper instance itself
        if name in ('_trk', '_p4'):
            raise AttributeError(name)
        return getattr(self._trk, name)


class JpsiChargedCandidate(ROOT.reco.CompositeCandidate):
    '''
    Generalized  J/psi(-> mu mu) + bachelor  candidate:  the base class of both
    the J/psi mu (JpsiMuCandidate, bachelor = muon) and the J/psi + track
    (JpsiTkCandidate, bachelor = kaon/pion pseudo-track) reconstructions.

    The two J/psi muons and the bachelor are kept both as a flat pt-sorted list
    (self.muons) and grouped (self.jpsi_muons + self.mu, the bachelor). The J/psi
    is stored as a reco::CompositeCandidate (self.jpsi) and the object itself is
    the full (2mu + bachelor) system. The bachelor is any object exposing
    p4()/charge()/bestTrack()/eta()/phi()/pt()/mass(): a pat::Muon in the J/psi mu
    case, a BachelorTrack in the J/psi + track case.

    All vertexing / displacement / IP / PV-refit / isolation / helicity machinery
    lives here, once. Subclasses set only what genuinely differs:
      * self._bachelor_mass  : the mass hypothesis assigned to the bachelor in the
                               full common-vertex fit (muon mass by default)
      * the composite mother pdgId, via the mother_pdgid argument of __init__

    Vertex- and IP-related quantities are NOT computed in __init__: call
    compute_vtx_quantities(vertices, beamspot) for that.
    '''

    # mass hypothesis assigned to the bachelor track in the full common-vertex
    # fit. Muon mass by default (JpsiMuCandidate); JpsiTkCandidate sets the kaon
    # mass and re-fits under the pion mass in compute_alt_bachelor.
    _bachelor_mass = masses['mu']

    def __init__(self, jpsi_muons, bachelor, mother_pdgid=541):

        super().__init__()

        # covariance check, memoized: muons are shared across candidates within
        # an event, so do this once per muon, not once per (muon, triplet)
        for imu in jpsi_muons + [bachelor]:
            if not hasattr(imu, 'cov'):
                imu.cov = convert_cov(imu.bestTrack().covariance())
                imu.is_cov_pos_def = is_pos_def(imu.cov)

        self.muons      = sorted(jpsi_muons + [bachelor], key = lambda x : x.pt(), reverse = True)
        self.jpsi_muons = sorted(jpsi_muons             , key = lambda x : x.pt(), reverse = True)

        self.mu = bachelor

        self.jpsi = ROOT.reco.CompositeCandidate()
        self.jpsi.addDaughter(self.jpsi_muons[0])
        self.jpsi.addDaughter(self.jpsi_muons[1])
        self.jpsi.setP4(self.jpsi.daughter(0).p4() + self.jpsi.daughter(1).p4())
        self.jpsi.setCharge(int(self.jpsi_muons[0].charge() + self.jpsi_muons[1].charge()))
        self.jpsi.setPdgId(443)           # optional: tag as J/psi

        self.setP4(self.jpsi.p4() + self.mu.p4())
        self.setCharge(int(sum([imu.charge() for imu in self.muons])))
        self.setPdgId(int(mother_pdgid * np.sign(self.mu.charge())))



    ##########################################################################
    #####      HELICITY ANGLES
    ##########################################################################
    def compute_helicity_angles(self, prefix='', bachelor_p4=None, visible_p4=None):
        '''Reco helicity angles for five Bc-momentum hypotheses:
            ev_jpsi : equal-velocity p4 along the PV->(J/psi vtx) flight direction
            ev_sv   : equal-velocity p4 along the PV->(3mu  vtx) flight direction
            coll    : pure collinear (equal-betagamma) p4 -- direction == visible p
            nu1/nu2 : the two solutions of the exact neutrino-pz quadratic
        PV is the hybrid reference (beamspot x,y + PV z) carried by Bdirection_*.

        Every hypothesis needs the refitted J/psi: if the J/psi fit failed
        (jpsi_rfp4 is None) nothing is set and the branches default to NaN.

        prefix / bachelor_p4 / visible_p4 select the mass hypothesis. The defaults
        (prefix='', bachelor_p4 = self.mu.p4(), visible_p4 = jpsi_rfp4 + mu.p4())
        reproduce the original single-hypothesis behaviour exactly, reading the
        unprefixed Bdirection_/bc_full_p4_/math*_b_p4_/p4_collinear attributes and
        writing cos_theta_v_/cos_theta_l_/chi_ . JpsiTkCandidate calls it a second
        time with prefix='pi_' and the pion bachelor 4-momentum to fill the pion
        block from the pi_-prefixed attributes.
        '''
        nan = float('nan')

        if getattr(self, 'jpsi_rfp4', None) is None:
            return

        if bachelor_p4 is None:
            bachelor_p4 = self.mu.p4()
        if visible_p4 is None:
            visible_p4 = self.jpsi_rfp4 + bachelor_p4
    
        # mu- of the J/psi pair = the two muons that are not the bachelor
        mu_minus = next((m for m in self.jpsi_muons if m.charge() < 0), None)
        p4_mu_v  = mu_minus.p4() if mu_minus is not None else None
        p4_lep   = bachelor_p4
    
        bc_p4     = {}
        bdir_jpsi = getattr(self, 'Bdirection_%sjpsi' % prefix, None)
        bdir_sv   = getattr(self, 'Bdirection_%ssv'   % prefix, None)
    
        # equal-velocity hypotheses: reuse the p4s already stored by
        # compute_vtx_quantities / compute_alt_bachelor, recompute only if missing
        for label, bdir in (('jpsi', bdir_jpsi), ('sv', bdir_sv)):
            if bdir is None:
                continue
            p4 = getattr(self, '%sbc_full_p4_%s' % (prefix, label), None)
            bc_p4[label] = p4 if p4 is not None else self.equal_velocity_p4(bdir, visible_p4)[1]
    
        # pure collinear: equal-velocity along the visible momentum itself
        # (self-consistent: same magnitude AND direction, no flight-direction input)
        coll = getattr(self, '%sp4_collinear' % prefix, None)
        if coll is None:
            coll = visible_p4 * (M_BC / visible_p4.mass())
        bc_p4['coll'] = coll
    
        # two neutrino solutions of the quadratic (mass-constrained, 3mu-vtx dir).
        # reuse the inspector's math*_b_p4_sv if already computed, else solve here.
        nu1 = getattr(self, '%smath1_b_p4_sv' % prefix, None)
        nu2 = getattr(self, '%smath2_b_p4_sv' % prefix, None)
        if nu1 is None or nu2 is None:
            nu_dir = bdir_sv if bdir_sv is not None else \
                     ROOT.Math.XYZVector(visible_p4.px(), visible_p4.py(), visible_p4.pz())
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
            setattr(self, '%scos_theta_v_%s' % (prefix, key), cv)
            setattr(self, '%scos_theta_l_%s' % (prefix, key), cl)
            setattr(self, '%schi_%s'         % (prefix, key), ch)

    ##########################################################################
    #####      VERTEXING
    ##########################################################################
    def compute_vtx_quantities(self, vertices, beamspot, pf=None, lost=None):
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

        The chosen PV is refit, beamspot-constrained and with the three signal
        muons removed (refit_primary_vertex), and that refit replaces the hybrid
        PV as the single wrt-PV reference (self.pv_bs).

        `pf` (optional) is the packedPFCandidates collection (event.pf). When
        given it enables the custom PF isolation of the bachelor muon and the
        J/psi (compute_isolation), recomputed against the refit PV.
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
            self.pv_idx = pv_idx
            self.pv     = vertices[pv_idx]
        else:
            self.pv_idx = min(
                range(len(vertices)),
                key = lambda i: abs(self.muons[0].bestTrack().dz(vertices[i].position())),
            )
            self.pv = vertices[self.pv_idx]

        # ----- beamspot as a reco::Vertex at the PV z position --------------
        self.bs = self.build_beamspot_vertex(beamspot, self.pv.z())

        # ----- primary vertex used as the reference for IP / displacement /
        #       isolation ------------------------------------------------------
        # Preferred: a per-candidate, beamspot-constrained AdaptiveVertexFitter
        # refit of the chosen PV with the three signal muons removed
        # (refit_primary_vertex) -- a properly formed reco::Vertex with its own
        # 3D covariance and the beamspot info, as prescribed by the BPH slides.
        # Fallback (no packed candidates, e.g. pf not loaded): the Run2 hybrid PV
        # (beamspot x,y + PV z, PV covariance).
        # Either way the result is self.pv_bs, built ONCE and shared by
        # compute_displacement, compute_ip, compute_jet_track_distance and
        # compute_isolation, so every wrt-PV quantity uses the same reference.
        self.refit_primary_vertex(beamspot, pf, lost, vertices)
        self.pv_bs = self.pv_refit if self.pv_refit_valid \
                     else self.build_hybrid_pv(self.bs, self.pv)

        # ----- displacement / pointing-angle quantities --------------------
        # the 3-muon vertex uses the full candidate momentum (self),
        # the J/psi vertex uses the dimuon momentum (self.jpsi)
        self.compute_jpsi_refit()
        self.compute_displacement(self.vertex_tree     , self.good_vtx     , self          , prefix=''     )
        self.compute_displacement(self.jpsi_vertex_tree, self.jpsi_good_vtx, self.jpsi_rfp4, prefix='jpsi_')

        # ----- bachelor-muon impact parameters ------------------------------
        self.compute_ip()

        # ----- Bc-momentum hypotheses, one per flight-direction label -------
        # mcorr / p4_par / p4_perp need only the direction (they use the raw
        # 3-muon p4), so they are filled for every label whose own vertex fit
        # succeeded. The equal-velocity p4 and p4_collinear additionally need
        # the refitted J/psi, hence the extra jpsi_rfp4 guard.
        self.bc_directions = {
            'jpsi': getattr(self, 'Bdirection_jpsi', None),
            'sv'  : getattr(self, 'Bdirection_sv'  , None),
        }

        visible_p4 = (self.jpsi_rfp4 + self.mu.p4()) if self.jpsi_rfp4 is not None else None

        for label, direction in self.bc_directions.items():
            if direction is None:
                continue

            p4_par  = self.p4().Vect().Dot(direction.unit())
            p4_perp = np.sqrt(max(0., self.p4().Vect().Mag2() - p4_par * p4_par))
            mcorr   = np.sqrt(self.p4().mass()**2 + p4_perp**2) + p4_perp
            setattr(self, 'p4_par_%s'  % label, p4_par )
            setattr(self, 'p4_perp_%s' % label, p4_perp)
            setattr(self, 'mcorr_%s'   % label, mcorr  )

            if visible_p4 is not None:
                p3, p4 = self.equal_velocity_p4(direction, visible_p4)
                setattr(self, 'bc_full_p_%s'  % label, p3)
                setattr(self, 'bc_full_p4_%s' % label, p4)

        if visible_p4 is not None:
            # scaling the whole visible 4-vector by M_BC/m_vis is exactly the
            # equal-velocity p4 along the visible direction
            self.p4_collinear = visible_p4 * (M_BC / visible_p4.mass())

        self.compute_jet_track_distance()

        # ----- PF isolation of the bachelor muon and the J/psi, recomputed
        #       against the refit PV (no-op if pf is None) ---------------------
        self.compute_isolation(pf, vertices)

    ##########################################################################
    #####      ALTERNATIVE BACHELOR-MASS HYPOTHESIS  (duplicated 3-body fit)
    ##########################################################################
    def compute_alt_bachelor(self, alt_bachelor_p4, alt_mass, prefix):
        '''
        Duplicate the full (2mu + bachelor) reconstruction under an ALTERNATIVE
        bachelor mass hypothesis, storing every hypothesis-dependent quantity
        under `prefix` (e.g. 'pi_'). Used by JpsiTkCandidate to add the pion
        (Bc+ -> J/psi pi+) block next to the kaon (B+ -> J/psi K+) block.

        MUST be called AFTER compute_vtx_quantities, whose shared, hypothesis-
        INDEPENDENT results are reused verbatim: the chosen PV (self.pv /
        self.pv_bs / self.bs), the mass-constrained J/psi dimuon fit
        (self.jpsi_vtx / self.jpsi_good_vtx / self.jpsi_rfp4) and the isolation.

        Only what depends on the bachelor mass is recomputed here:
          * the full 2mu+bachelor common vertex, re-fit with the alternative mass
            on the SAME tracks (fit_vertex(..., bachelor_mass=alt_mass))
          * the displacement / pointing of that vertex             ({prefix}...)
          * the two flight directions, the bachelor signed-IP grid and the
            jet-track distances tied to them                        ({prefix}mu_*)
          * the corrected-mass / equal-velocity Bc-momentum block   ({prefix}...)

        `alt_bachelor_p4` is the bachelor 4-momentum under the alternative mass (a
        PxPyPzE4D LorentzVector). The neutrino / q2 / m_miss2 / helicity block is
        added prefix-aware by the inspector, from Bdirection_{prefix}{jpsi,sv} and
        this alt_bachelor_p4 (see inspector_jpsi_tk.py). The jpsi-direction IP is
        mass-independent, so its {prefix} copy equals the primary hypothesis by
        construction; it is stored anyway to keep a complete, uniform block.
        '''
        track = self.mu.bestTrack()

        # ---- alternative-mass full (2mu + bachelor) vertex, SAME tracks --------
        alt_tree = self.fit_vertex(self.muons, bachelor_mass=alt_mass)
        alt_good = self.is_good_vtx(alt_tree)
        setattr(self, prefix + 'vertex_tree', alt_tree)
        setattr(self, prefix + 'good_vtx',    alt_good)

        # composite / visible p4 under the alternative bachelor mass
        comp_p4    = self.jpsi.p4() + alt_bachelor_p4
        visible_p4 = (self.jpsi_rfp4 + alt_bachelor_p4) if self.jpsi_rfp4 is not None else None
        setattr(self, prefix + 'p4',   comp_p4)
        setattr(self, prefix + 'rfp4', visible_p4)

        # ---- displacement / pointing of the alt 3-body vertex ------------------
        # (the J/psi-vertex block is shared and NOT duplicated). compute_displacement
        # reads self.pv_bs / self.bs / self.pv (shared) and stores {prefix}vtx,
        # {prefix}vtx_chi2/ndof/prob, {prefix}lxy, {prefix}lxyz, {prefix}cos2d/3d/3dbs.
        self.compute_displacement(alt_tree, alt_good, comp_p4, prefix=prefix)

        # ---- per-label vertices for the alt hypothesis -------------------------
        # jpsi direction reuses the SHARED J/psi vertex; sv direction uses the alt
        # 3-body vertex.
        alt_vertices  = {'jpsi': self.jpsi_vtx     , 'sv': getattr(self, prefix + 'vtx')}
        alt_vtx_valid = {'jpsi': self.jpsi_good_vtx, 'sv': alt_good}

        for label, vtx in alt_vertices.items():
            if not alt_vtx_valid[label]:
                continue
            setattr(self, 'Bdirection_%s%s' % (prefix, label), self.flight_direction(vtx, self.pv_bs))

        # ---- bachelor signed-IP 3D grid (2 x 2) under the alt directions -------
        for label, vtx in alt_vertices.items():
            if not alt_vtx_valid[label]:
                continue
            direction  = getattr(self, 'Bdirection_%s%s' % (prefix, label))
            dx, dy, dz = direction.x(), direction.y(), direction.z()
            references = {'pv': self.pv_bs, 'sv': self.kin_to_reco_vertex(vtx)}
            for ref, ref_vtx in references.items():
                ip = kinfit.signedIP3D(track, dx, dy, dz, ref_vtx)
                setattr(self, '%smu_ip3d_%s_%s'     % (prefix, label, ref), ip.value())
                setattr(self, '%smu_ip3d_%s_%s_err' % (prefix, label, ref), ip.error())
                setattr(self, '%smu_ip3d_%s_%s_sig' % (prefix, label, ref), ip.significance())

        # ---- bachelor jet-track distances under the alt directions -------------
        for label, vtx in alt_vertices.items():
            if not alt_vtx_valid[label]:
                continue
            direction  = getattr(self, 'Bdirection_%s%s' % (prefix, label))
            dx, dy, dz = direction.x(), direction.y(), direction.z()
            references = {'pv': self.pv_bs, 'sv': self.kin_to_reco_vertex(vtx)}
            jtd = None
            for ref, ref_vtx in references.items():
                jtd = kinfit.jetTrackDistance(track, dx, dy, dz, ref_vtx)
                setattr(self, '%smu_dist_along_b_dir_%s_%s' % (prefix, label, ref), jtd.first)
            setattr(self, '%smu_dist_to_b_dir_%s' % (prefix, label), jtd.second.value())

        # ---- corrected mass / equal-velocity Bc momentum, per direction --------
        alt_directions = {
            'jpsi': getattr(self, 'Bdirection_%sjpsi' % prefix, None),
            'sv'  : getattr(self, 'Bdirection_%ssv'   % prefix, None),
        }
        setattr(self, prefix + 'bc_directions', alt_directions)

        for label, direction in alt_directions.items():
            if direction is None:
                continue
            p4_par  = comp_p4.Vect().Dot(direction.unit())
            p4_perp = np.sqrt(max(0., comp_p4.Vect().Mag2() - p4_par * p4_par))
            mcorr   = np.sqrt(comp_p4.mass()**2 + p4_perp**2) + p4_perp
            setattr(self, '%sp4_par_%s'  % (prefix, label), p4_par )
            setattr(self, '%sp4_perp_%s' % (prefix, label), p4_perp)
            setattr(self, '%smcorr_%s'   % (prefix, label), mcorr  )
            if visible_p4 is not None:
                p3, p4 = self.equal_velocity_p4(direction, visible_p4)
                setattr(self, '%sbc_full_p_%s'  % (prefix, label), p3)
                setattr(self, '%sbc_full_p4_%s' % (prefix, label), p4)

        if visible_p4 is not None:
            setattr(self, prefix + 'p4_collinear', visible_p4 * (M_BC / visible_p4.mass()))

    def fit_vertex(self, particles, bachelor_mass=None):
        '''
        Fit the J/psi muons + the bachelor to a common vertex with the generalized
        RJpsiKinVtxFitter. Every J/psi muon gets the muon mass; the bachelor
        (self.mu, matched by object identity) gets its own mass hypothesis --
        self._bachelor_mass by default, or `bachelor_mass` when given (used to
        re-fit the SAME tracks under an alternative bachelor mass, e.g. the pion
        hypothesis in JpsiTkCandidate.compute_alt_bachelor).

        For JpsiMuCandidate the bachelor is a muon and _bachelor_mass is the muon
        mass, so every track gets the muon mass -- identical to the original
        RJpsi fit. Returns the kinematic decay tree (empty if the fit fails).
        '''
        m_bach   = self._bachelor_mass if bachelor_mass is None else bachelor_mass
        tracks   = ROOT.std.vector('reco::Track')()
        mass_hyp = ROOT.std.vector('double')()
        for ip in particles:
            tracks.push_back(ip.bestTrack())
            mass_hyp.push_back(m_bach if ip is self.mu else masses['mu'])
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
        '''
        Distances of the bachelor-muon track from the Bc flight axis, from
        IPTools::jetTrackDistance (linearized track):

          mu_dist_along_b_dir_<dir>_<ref> : signed distance ALONG the axis, from
                                            the reference vertex to the point of
                                            closest approach (up/downstream)
          mu_dist_to_b_dir_<dir>          : line-to-line distance between the
                                            track and the axis; independent of
                                            the anchor vertex, so one value per
                                            direction

        Same 2x2 grid, per-label gating and hybrid PV (self.pv_bs) as compute_ip.
        '''
        track = self.mu.bestTrack()

        # 2 x 2:  {direction from jpsi-vtx | from 3mu-vtx}  x  {wrt PV | wrt that same SV}
        for label, vtx in self.bc_vertices.items():
            if not self.bc_vtx_valid[label]:
                continue
            direction  = getattr(self, 'Bdirection_%s' % label)
            dx, dy, dz = direction.x(), direction.y(), direction.z()

            references = {'pv': self.pv_bs, 'sv': self.kin_to_reco_vertex(vtx)}
            jtd = None
            for ref, ref_vtx in references.items():
                jtd = kinfit.jetTrackDistance(track, dx, dy, dz, ref_vtx)
                setattr(self, 'mu_dist_along_b_dir_%s_%s' % (label, ref), jtd.first)

            # the distance between the two lines is independent of the vertex:
            # reuse the last result instead of making a third C++ call
            setattr(self, 'mu_dist_to_b_dir_%s' % label, jtd.second.value())
#             setattr(self, 'mu_dist_to_b_dir_%s_err' % label, jtd.second.error())
#             setattr(self, 'mu_dist_to_b_dir_%s_sig' % label, jtd.second.significance())


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
    def build_hybrid_pv(bs, pv):
        '''
        Hybrid primary vertex: beamspot transverse position + PV longitudinal
        position, with the PV covariance. Matches the flight_direction convention
        (transverse from the beamspot, longitudinal from the PV), so every
        wrt-PV quantity uses one and the same reference point.
        '''
        return ROOT.reco.Vertex(
            ROOT.reco.Vertex.Point(bs.position().x(),   # beamspot x
                                   bs.position().y(),   # beamspot y
                                   pv.position().z()),  # PV z
            pv.error(), pv.chi2(), pv.ndof(), pv.tracksSize()
        )

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
          {prefix}vtx       : the KinematicVertex (None if the fit failed)
          {prefix}vtx_chi2  : vertex chi2
          {prefix}vtx_ndof  : vertex ndof
          {prefix}vtx_prob  : vertex fit probability
          {prefix}lxy       : 2D distance from the refit PV     (Measurement1D)
          {prefix}lxyz      : 3D distance from the refit PV      (Measurement1D)
          {prefix}cos2d     : cosine of the 2D pointing angle (refit PV -> SV vs pT)
          {prefix}cos3d     : cosine of the 3D pointing angle (refit PV -> SV vs p)
          {prefix}cos3dbs   : cos3d wrt the bare hybrid reference (beamspot x,y +
                              PV z), i.e. the pre-refit convention -- kept as a
                              fixed before/after comparator

        All wrt-PV displacement quantities use self.pv_bs: the per-candidate
        beamspot-constrained, signal-removed refit when available, otherwise the
        Run2 hybrid PV. Per the BPH slides this is the best reference in both the
        transverse plane (beamspot constraint) and 3D (signal-track removal); the
        bare-beamspot quantity is retained only as cos3dbs for validation.

        Measurement1D objects expose .value(), .error() and .significance().
        'cand' is the (composite) candidate whose momentum defines the pointing
        angle: the full 3-muon system for the 3-muon vertex, the J/psi for the
        J/psi vertex.
        '''

        # if the fit failed, store None/NaNs and return
        if not good_vtx:
            setattr(self, prefix + 'vtx', None)
            for q in ['vtx_chi2', 'vtx_ndof', 'vtx_prob', 'lxy', 'lxyz', 'cos2d', 'cos3d', 'cos3dbs']:
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
        setattr(self, prefix + 'vtx_prob', stats.chi2.sf(chi2, ndof))

        # 2D distance from the refit PV (significance via Measurement1D)
        lxy = ROOT.VertexDistanceXY().distance(self.pv_bs, vtx.vertexState())
        setattr(self, prefix + 'lxy', lxy)

        # 3D distance from the refit PV (significance via Measurement1D)
        lxyz = ROOT.VertexDistance3D().distance(self.pv_bs, vtx.vertexState())
        setattr(self, prefix + 'lxyz', lxyz)

        # 2D pointing angle: transverse displacement from the refit PV
        vect_lxy = Vector3D(
            vtx.position().x() - self.pv_bs.position().x(),
            vtx.position().y() - self.pv_bs.position().y(),
            0.,
        )
        vect_pt = Vector3D(cand.px(), cand.py(), 0.)
        setattr(self, prefix + 'cos2d',
                vect_pt.Dot(vect_lxy) / (vect_pt.R() * vect_lxy.R()) if vect_lxy.R() > 0. else np.nan)

        # 3D pointing angle: full displacement from the refit PV (consistent with
        # flight_direction, so cos3d ~ angle(Bdirection, p))
        vect_lxyz = Vector3D(
            vtx.position().x() - self.pv_bs.position().x(),
            vtx.position().y() - self.pv_bs.position().y(),
            vtx.position().z() - self.pv_bs.position().z(),
        )
        vect_p = Vector3D(cand.px(), cand.py(), cand.pz())
        setattr(self, prefix + 'cos3d',
                vect_p.Dot(vect_lxyz) / (vect_p.R() * vect_lxyz.R()) if vect_lxyz.R() > 0. else np.nan)

        # 3D pointing angle wrt the BARE hybrid PV (beamspot x,y + PV z), always,
        # regardless of the refit: a fixed pre-refit comparator. Equals cos3d when
        # the refit is unavailable (then self.pv_bs is itself the hybrid).
        vect_lxyz_bs = Vector3D(
            vtx.position().x() - self.bs.position().x(),
            vtx.position().y() - self.bs.position().y(),
            vtx.position().z() - self.pv.position().z(),
        )
        setattr(self, prefix + 'cos3dbs',
                vect_p.Dot(vect_lxyz_bs) / (vect_p.R() * vect_lxyz_bs.R()) if vect_lxyz_bs.R() > 0. else np.nan)

    def compute_ip(self):
        '''
        Signed 3D impact parameters of the bachelor muon (self.mu), on a 2x2 grid:

            flight direction  in {jpsi, sv}  (PV -> J/psi vertex | PV -> 3mu vertex)
          x reference vertex  in {pv, sv}    (hybrid PV | that hypothesis's OWN vertex)

        stored as  mu_ip3d_<dir>_<ref>[_err|_sig], lifetime-signed along the given
        flight direction with IPTools::signedImpactParameter3D (wrapped in
        RJpsiKinVtxFitter). The bachelor muon is NOT part of either vertex fit, so
        its IP w.r.t. those vertices is unbiased and small for true Bc -> J/psi mu nu.

        Each hypothesis is computed independently: a label is filled whenever its
        OWN vertex fit succeeded (jpsi_good_vtx / good_vtx), so e.g. the jpsi-based
        quantities survive a failed 3-muon fit. Missing hypotheses are simply left
        unset and end up NaN in the ntuple via safe_get.
        '''
        # per-label vertex and validity: each flight-direction hypothesis is tied
        # to its own secondary vertex
        self.bc_vertices  = {'jpsi': self.jpsi_vtx     , 'sv': self.vtx     }
        self.bc_vtx_valid = {'jpsi': self.jpsi_good_vtx, 'sv': self.good_vtx}

        for label, vtx in self.bc_vertices.items():
            if not self.bc_vtx_valid[label]:
                continue
            setattr(self, 'Bdirection_%s' % label, self.flight_direction(vtx, self.pv_bs))

        track = self.mu.bestTrack()

        # 2 x 2:  {direction from jpsi-vtx | from 3mu-vtx}  x  {wrt PV | wrt that same SV}
        for label, vtx in self.bc_vertices.items():
            if not self.bc_vtx_valid[label]:
                continue
            direction  = getattr(self, 'Bdirection_%s' % label)
            dx, dy, dz = direction.x(), direction.y(), direction.z()

            references = {'pv': self.pv_bs, 'sv': self.kin_to_reco_vertex(vtx)}
            for ref, ref_vtx in references.items():
                ip = kinfit.signedIP3D(track, dx, dy, dz, ref_vtx)
                setattr(self, 'mu_ip3d_%s_%s'     % (label, ref), ip.value())
                setattr(self, 'mu_ip3d_%s_%s_err' % (label, ref), ip.error())
                setattr(self, 'mu_ip3d_%s_%s_sig' % (label, ref), ip.significance())

    ##########################################################################
    #####      PRIMARY-VERTEX REFIT  (AVF, beamspot-constrained, muons out)
    ##########################################################################
    @staticmethod
    def passes_pv_track_filter(cand, trk, beamspot):
        '''
        Approximate the offlinePrimaryVertices TkFilterParameters
        (TrackFilterForPVFinding) on a packed/lost candidate, so the PV refit is fed
        the track set the offline PV reconstruction would have used rather than every
        nearby candidate. Cuts (constants at file top): normalized chi2, pixel /
        tracker layers, transverse-IP significance wrt the beamline at the track z,
        IP errors, and pt.

        cand : the pat::PackedCandidate (kinematics, IP, IP errors)
        trk  : cand.pseudoTrack(), passed in to avoid rebuilding it (chi2, layers)
        The caller must have checked cand.hasTrackDetails().

        Packed quantities are reduced-precision, so this is a close but not bit-exact
        replica; the layer accessors go through the pseudo-track hit pattern -- verify
        they are populated in your PackedCandidate version (pv_refit_valid collapsing
        to ~0 is the canary for a bad accessor).
        '''
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

        # transverse IP significance wrt the beamline evaluated at the candidate z
        # (value recomputed against the beamspot; error is the stored packed dxyError)
        d0 = cand.dxy(beamspot.position(cand.vz()))
        if d0_err > 0. and abs(d0) / d0_err > _PV_TRK_MAX_D0_SIG:
            return False

        return True

    def refit_primary_vertex(self, beamspot, pf, lost, vertices):
        '''
        Per-candidate primary vertex: AdaptiveVertexFitter refit of the chosen PV
        with the transverse beamspot constraint and the three signal muons
        removed (RJpsiKinVtxFitter.refitPVRemovingTracks, which reproduces the
        BPH-slides PVRefitter). Replaces the Run2 hybrid PV (beamspot x,y + PV z)
        with a properly formed reco::Vertex carrying its own 3D covariance and the
        beamspot information.

        The PV track set is rebuilt IN THE LOOP from the packed (pf) + lost (lost)
        candidates. Primary selection: the offline Deterministic-Annealing fit-track
        assignment, read directly from the packed-candidate PV association
        (fromPV(pv_idx) == PVUsedInFit) -- the tracks the offline PV fit actually
        used for the chosen PV, with no re-clustering and no closest-z approximation,
        O(1) per candidate, and already offline-quality (no re-filter applied).
        Fallback (if that yields < _PV_REFIT_MIN_FROMPV_TRK tracks, e.g. the
        association is not populated or the chosen-PV index is not aligned with the
        candidates' reference collection): the closest-z PV/PU association used by
        compute_isolation, with the offline track-quality filter
        (passes_pv_track_filter) applied. Either way this removes the dependency on a
        persisted primaryVertexRefit:WithBS carrying its track references, so that
        collection and recoTracks_unpackedTracksAndVertices can be dropped from the
        skim.

        Sets:
          self.pv_refit       : the refitted reco::Vertex (invalid on failure)
          self.pv_refit_valid : bool, True iff a valid refit was obtained

        On any failure -- pf not loaded, too few surviving tracks, fit failure --
        pv_refit_valid is False and the caller falls back to the hybrid PV, so
        behaviour without packed candidates is unchanged.
        '''
        self.pv_refit       = None
        self.pv_refit_valid = False

        if pf is None:
            return  # no packed candidates -> cannot rebuild the PV track set

        try:
            vtxs = list(vertices)
            if not vtxs:
                return

            # signal muons to remove from the refit (proximity match in the C++: the
            # muon best-track and the candidate pseudo-track live in different
            # collections, matched by charge / dR / rel-pt). Same for either track
            # selection below, so build them once.
            mu_tracks = ROOT.std.vector('reco::Track')()
            for imu in self.muons:
                mu_tracks.push_back(imu.bestTrack())

            def _refit(selector, apply_filter):
                # collect the PV track set (selector, with optional quality filter),
                # run the beamspot-constrained AVF refit with the signal muons
                # removed, and return a VALID reco::Vertex or None.
                pv_tracks = ROOT.std.vector('reco::Track')()
                for coll in (pf, lost):
                    if coll is None:
                        continue
                    for cand in coll:
                        if not cand.hasTrackDetails():
                            continue
                        trk = cand.pseudoTrack()  # built once; filter + refit input
                        if apply_filter and not self.passes_pv_track_filter(cand, trk, beamspot):
                            continue
                        if selector(cand):
                            pv_tracks.push_back(trk)
                if pv_tracks.size() < _PV_REFIT_MIN_TRK:
                    return None
                v = kinfit.refitPVRemovingTracks(
                    pv_tracks, mu_tracks, beamspot,
                    _MU_TRK_DR_MATCH,      # drMatch
                    _MU_TRK_RELPT_MATCH,   # relPtMatch
                    _PV_AVF_CHI2CUTOFF,    # AVF annealing cutoff (offline WithBS)
                    _PV_MIN_NDOF,          # minNdof (WithBS)
                    _PV_MAX_DIST_TO_BEAM,  # maxDistanceToBeam [cm]
                )
                return v if v.isValid() else None

            # ---- primary: offline DA fit-track set, straight from fromPV --------
            # PVUsedInFit means the offline PV fit used this track for vertex pv_idx;
            # trust it as-is -- it already passed the offline TkFilterParameters, so
            # no closest-z and no re-filter (flip apply_filter=True for uniformity).
            refit = _refit(lambda cand: cand.fromPV(self.pv_idx) == _PV_USED_IN_FIT,
                           apply_filter=False)

            # ---- z-consistency guard ---------------------------------------------
            # the refit MUST sit on the chosen PV. A large |z(refit) - z(pv)| means
            # fromPV gathered a different vertex's tracks (PV-index misalignment
            # between the WithBS collection read here and the offlineSlimmedPrimary-
            # Vertices the associations reference). The track-count check inside
            # _refit cannot catch this -- it returns plenty of (wrong) tracks -- so
            # guard on z here and, if it fails (or fromPV gave nothing), fall back to
            # the closest-z association + offline track-quality filter.
            if refit is None or abs(refit.z() - self.pv.z()) > _PV_REFIT_MAX_DZ_TO_PV:
                vtx_z = [vv.position().z() for vv in vtxs]  # hoist PyROOT calls out
                def _closest_is_pv(cand):
                    vz = cand.vz()
                    return min(range(len(vtx_z)),
                               key=lambda i: abs(vtx_z[i] - vz)) == self.pv_idx
                # closest-z picks tracks nearest pv_idx by construction, so it is not
                # subject to the misalignment above; accept its valid result as-is.
                refit = _refit(_closest_is_pv, apply_filter=True)

            if refit is not None:
                self.pv_refit       = refit
                self.pv_refit_valid = True
        except Exception:
            # any PyROOT / collection issue -> silent fall back to hybrid PV
            self.pv_refit_valid = False

    def is_signal_muon_cand(self, cand,
                            dr_max=_MU_TRK_DR_MATCH,
                            rel_pt_max=_MU_TRK_RELPT_MATCH):
        '''
        True if the candidate `cand` (a PF candidate or a track) is one of the
        three signal muons, matched by charge, dR and relative pt. The muon and
        the PF candidate are in different collections, so this is a proximity
        match, identical in spirit to the C++ PV-refit removal.
        '''
        for imu in self.muons:
            if cand.charge() != imu.charge():
                continue
            if (deltaR(imu.eta(), imu.phi(), cand.eta(), cand.phi()) < dr_max and
                    abs(cand.pt() - imu.pt()) < rel_pt_max * imu.pt()):
                return True
        return False

    ##########################################################################
    #####      PF ISOLATION  (muon-POG algorithm, recomputed vs the refit PV)
    ##########################################################################
    def compute_isolation(self, pf, vertices):
        '''
        PF isolation of the bachelor muon (self.mu) and the J/psi (self.jpsi),
        recomputed against the custom PV (refit, signal-removed, beamspot-
        constrained, min-IP3D), faithfully replicating the muon-POG PFIsolation
        algorithm with the PV assumption swapped -- the prescription from the BPH
        vertexing+isolation slides (slides 13-20, 24).

        Charged candidates are split into PV vs PU by CLOSEST-Z VERTEX
        ASSOCIATION against the beamspot-constrained vertex collection with the
        chosen PV replaced by the custom refit: a candidate is "from PV" iff its
        nearest-in-z vertex IS the custom PV. (A flat dz cut is deliberately NOT
        used: the slides show it lets pileup through, since vertices sit <1 mm
        apart.) Neutrals are summed with a pt>0.5 threshold and an inner veto.

        For each object O in {mu, jpsi} and cone R in {03, 04}:
          O_iso_ch_RR / O_iso_ch_clean_RR : charged-hadron sum from the custom PV
                                            (pdgId 211/321/2212/999211, no leptons);
                                            'clean' also removes the 3 signal muons
          O_iso_pu_RR / O_iso_pu_clean_RR : charged sum NOT from the custom PV
                                            (leptons included; dR>0.01, pt>0.5)
          O_iso_nh_RR                     : neutral-hadron sum (111/130/310/2112)
          O_iso_ph_RR                     : photon sum (22)
          O_iso_RR    / O_iso_clean_RR    : ch + max(0, nh + ph - 0.5*pu)
          O_reliso_RR / O_reliso_clean_RR : the above divided by pt(O)

        No-op (branches left NaN via safe_get) when pf is None.
        '''
        if pf is None:
            return

        custom_pv = self.pv_bs

        # association collection: the BS-constrained PVs, with the chosen PV
        # swapped for the custom (signal-removed) refit, so that
        # "closest-z vertex IS custom_pv" is a meaningful PV/PU tag
        assoc = list(vertices)
        if 0 <= self.pv_idx < len(assoc):
            assoc[self.pv_idx] = custom_pv

        # ---- classify each PF candidate ONCE per candidate (not per cone/obj)
        # charged: (eta, phi, pt, pdg, from_pv, is_signal)
        # neutral: (eta, phi, pt, pdg)
        charged, neutral = [], []
        for cand in pf:
            pdg = abs(cand.pdgId())
            if pdg in _ISO_CH_ALL_IDS:
                vz     = cand.vz()
                best   = min(assoc, key=lambda v: abs(v.position().z() - vz))
                charged.append((cand.eta(), cand.phi(), cand.pt(), pdg,
                                best is custom_pv, self.is_signal_muon_cand(cand)))
            elif pdg in _ISO_NH_IDS or pdg in _ISO_PH_IDS:
                pt = cand.pt()
                if pt > _ISO_NEUTRAL_PT:
                    neutral.append((cand.eta(), cand.phi(), pt, pdg))

        objects = (('mu', self.mu), ('jpsi', self.jpsi))

        for cone in _ISO_CONES:
            rr = '%02d' % int(round(cone * 10))
            for name, obj in objects:
                oe, op, opt = obj.eta(), obj.phi(), obj.pt()

                ch = ch_clean = pu = pu_clean = nh = ph = 0.

                # charged: charged-hadron-from-PV sum + pileup sum
                for c_eta, c_phi, c_pt, pdg, from_pv, is_sig in charged:
                    dr = deltaR(oe, op, c_eta, c_phi)
                    if dr >= cone or dr <= _ISO_DR_VETO_CH:
                        continue
                    if from_pv:
                        if pdg in _ISO_CH_HAD_IDS:        # no leptons in CH sum
                            ch += c_pt
                            if not is_sig:
                                ch_clean += c_pt
                    else:
                        if dr > _ISO_PU_DR_VETO and c_pt > _ISO_PU_PT:
                            pu += c_pt                    # leptons included in PU
                            if not is_sig:
                                pu_clean += c_pt

                # neutral hadrons + photons (no PV association, no signal removal)
                for n_eta, n_phi, n_pt, pdg in neutral:
                    dr = deltaR(oe, op, n_eta, n_phi)
                    if dr >= cone or dr <= _ISO_DR_VETO_NH:
                        continue
                    if pdg in _ISO_NH_IDS:
                        nh += n_pt
                    elif pdg in _ISO_PH_IDS:
                        ph += n_pt

                iso       = ch       + max(0., nh + ph - 0.5 * pu)
                iso_clean = ch_clean + max(0., nh + ph - 0.5 * pu_clean)

                setattr(self, '%s_iso_ch_%s'       % (name, rr), ch)
                setattr(self, '%s_iso_ch_clean_%s' % (name, rr), ch_clean)
                setattr(self, '%s_iso_pu_%s'       % (name, rr), pu)
                setattr(self, '%s_iso_pu_clean_%s' % (name, rr), pu_clean)
                setattr(self, '%s_iso_nh_%s'       % (name, rr), nh)
                setattr(self, '%s_iso_ph_%s'       % (name, rr), ph)
                setattr(self, '%s_iso_%s'          % (name, rr), iso)
                setattr(self, '%s_iso_clean_%s'    % (name, rr), iso_clean)
                setattr(self, '%s_reliso_%s'       % (name, rr), iso       / opt if opt > 0. else np.nan)
                setattr(self, '%s_reliso_clean_%s' % (name, rr), iso_clean / opt if opt > 0. else np.nan)

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
        return LorentzVector(px, py, pz, e)

    @staticmethod
    def flight_direction(vtx, pv_ref):
        '''
        pv_ref -> SV flight direction in full 3D from the single reference vertex
        pv_ref (the refit PV when available, otherwise the Run2 hybrid PV
        beamspot x,y + PV z). With the hybrid this reproduces the old
        transverse-from-beamspot / longitudinal-from-PV convention exactly.
        '''
        sv = vtx.position()
        return ROOT.Math.XYZVector(
            sv.x() - pv_ref.position().x(),
            sv.y() - pv_ref.position().y(),
            sv.z() - pv_ref.position().z(),
        ) 

    @staticmethod
    def equal_velocity_p4(direction, visible_p4):
        '''
        Equal-velocity ("rest-frame") Bc momentum along `direction`:
            |p_Bc| = (M_BC / m_vis) * |p_vis| ,   E_Bc = sqrt(M_BC^2 + |p_Bc|^2)
        Returns (p3, p4): a ROOT XYZVector and a PxPyPzE4D LorentzVector.
        '''
        p3 = direction.unit() * (M_BC / visible_p4.mass() * visible_p4.P())
        p4 = LorentzVector(p3.x(), p3.y(), p3.z(), np.sqrt(M_BC**2 + p3.Mag2()))
        return p3, p4

    def compute_jpsi_refit(self):
        '''
        Refitted muon momenta under the J/psi mass + common-vertex constraint, plus
        the refitted (constrained) J/psi momentum:
            self.jpsi_mu1_rfp4, self.jpsi_mu2_rfp4  -> refitted muon 4-momenta
            self.jpsi_rfp4                          -> refitted, mass-constrained J/psi
            self.rfp4                               -> jpsi_rfp4 + bachelor-mu p4
        Children follow the order given to the fit: jpsi_muons[0] -> mu1, [1] -> mu2.

        The refit p4 is also stored on the muon objects (imu.jpsi_rfp4) for the
        per-muon branches. Muons are SHARED between candidates within an event, so
        these attributes are reset to None on all three muons first: a muon refit
        in a previous candidate must not leak its old momentum into this one (e.g.
        when it is the bachelor here, or when this candidate's fit failed).
        '''
        for q in ['jpsi_mu1_rfp4', 'jpsi_mu2_rfp4', 'jpsi_rfp4', 'rfp4']:
            setattr(self, q, None)
        for imu in self.muons:
            imu.jpsi_rfp4 = None
    
        if not self.jpsi_good_vtx:
            return
    
        tree = self.jpsi_vertex_tree
    
        tree.movePointerToTheTop()                          # mass-constrained J/psi
        self.jpsi_rfp4 = self.refitted_p4(tree.currentParticle())
    
        tree.movePointerToTheFirstChild()                   # first refitted muon
        self.jpsi_mu1_rfp4 = self.refitted_p4(tree.currentParticle())
        tree.movePointerToTheNextChild()                    # second refitted muon
        self.jpsi_mu2_rfp4 = self.refitted_p4(tree.currentParticle())

        self.jpsi_muons[0].jpsi_rfp4 = self.jpsi_mu1_rfp4
        self.jpsi_muons[1].jpsi_rfp4 = self.jpsi_mu2_rfp4

        self.rfp4 = self.jpsi_rfp4 + self.mu.p4()


    ##########################################################################
    #####      helicity angles
    ##########################################################################

    @staticmethod
    def _np4(p4):
        '''[E, px, py, pz] from a four-vector, or from a (p4, ...) tuple as
        returned by refitted_p4 / equal_velocity_p4.'''
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
    
        p_bc   = JpsiChargedCandidate._np4(p4_bc)
        p_jpsi = JpsiChargedCandidate._np4(p4_jpsi)
        p_muv  = JpsiChargedCandidate._np4(p4_mu_v)
        p_lep  = JpsiChargedCandidate._np4(p4_lep)
        p_w    = p_bc - p_jpsi
    
        boost, unit = JpsiChargedCandidate._boost, JpsiChargedCandidate._unit3
    
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
