'''
R(Ds) ntuplizer: Bs -> Ds(-> phi(-> K K) pi) mu (nu) from the 2018 B-parking
data (ParkingBPH1-6, UL2018 MiniAODv2) and the matching UL2018 MC.

A BaseInspector channel (JpsiChargedInspector): argument parsing, lumi mask,
trigger bookkeeping, muon preselection, batched uproot output and cutflow are
the shared ones. The 2018 parking trigger is BParking2018Trigger, the same
code as the Vcb (J/psi + track) parking ntuples.

Per event
  1. trigger: OR of the ten HLT_Mu*_IP* paths (MC: --savenontrig keeps all)
  2. muons: pT > 7, |eta| < 1.5, PF. NO trigger-matching requirement: the
     match is stored (mu_trig_dr, mu_trig_match, mu_<path>_tag) and applied
     offline
  3. tracks (packedPFCandidates + lostTracks): AN Sec. 5.2, around each muon
  4. combinatorics in numpy on the (calibrated) momenta: phi window, widened
     KKpi window, m(KKpi mu) < 8 GeV, any charge combination
  5. both kinematic-fit schemes per surviving triplet (RDsCandidate); kept if
     either passes the AN vertex-probability cuts
  6. every kept candidate is a row (no arbitration); cand_idx numbers them by
     decreasing pT of KKpi mu

MC only
  * --trk-calib: momentum calibration (rds_momcalib v4) on every track and the
    muon, before the combinatorics and the fits (TrackMomentumCalib)
  * a second tree, tree_gen, with one row per processed event: the signal
    decay at generator level, the Bs flight (lifetime / FONLL weights), the
    Hammer inputs and weights. Needed for acceptance and normalisation studies
    on the unfiltered samples.
  * gen matching and classification of every candidate (RDsGenHistory)
  * --hammer <card>: Bs -> Ds(*) form-factor weights (HammerDsFF); without it
    the hammer_* branches are NaN and can be added later with
    test/rds/hammer/add_hammer_weights_rds.py

--cov-scale applies to the kaons, the pion and the muon alike.
'''

import os
import re
import numpy as np
from collections import OrderedDict
from DataFormats.FWLite import Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR, bestMatch

from Bmmm.Analysis.JpsiChargedInspector import BaseInspector
from Bmmm.Analysis.BParking2018Trigger import BParking2018TriggerMixin, paths
from Bmmm.Analysis.RDsBranches import (
    branches, event_branches, muon_branches, track_branches, muon_trigger_names,
    cand_names, gen_event_names, gen_tree_branches, L1_SEEDS, safe_get,
    EVENT_EXTRA_KEYS, EVENT_GEN_KEYS, BS_GEN_NAMES,
)
from Bmmm.Analysis.RDsCandidate import RDsCandidate, RDsTrack, TrackRecord
from Bmmm.Analysis.RDsCuts import cuts
from Bmmm.Analysis.RDsKinematics import M_K, M_PI, M_MU, kkpi_triplets
from Bmmm.Analysis.RDsGenHistory import (
    BsSignalDecay, bs_gen_info, gen_observables, gen_hammer_leaves,
    charged_final_state, match_track, classify_candidate, mother_of,
)
from Bmmm.Analysis.HammerDsFF import make_hammer_session as make_hammer_ds_session
from Bmmm.Analysis.HammerDsFF import BRANCH_NAMES as HAMMER_BRANCH_NAMES
from Bmmm.Analysis.TrackMomentumCalib import make_trk_calib, TrackMomentumCalib
from Bmmm.Analysis.L1Seeds2018 import L1Seeds
from Bmmm.Analysis.GenVertices import GenPVReader
from Bmmm.Analysis.utils import cutflow, drop_hlt_version

NAN = float('nan')

# 2018 eras by run range
ERAS_2018 = ((1, 315252, 316995), (2, 317080, 319310), (3, 319337, 320065), (4, 320413, 325175))
_BPH_RE = re.compile(r'ParkingBPH([1-6])')


def era_of(run):
    for code, lo, hi in ERAS_2018:
        if lo <= run <= hi:
            return code
    return NAN


extra_handles = OrderedDict()
extra_handles['glb_alg'] = (('gtStage2Digis', ''),      Handle('BXVector<GlobalAlgBlk>'))
extra_handles['l1min'  ] = (('patTrigger',    'l1min'), Handle('pat::PackedTriggerPrescales'))
extra_handles['l1max'  ] = (('patTrigger',    'l1max'), Handle('pat::PackedTriggerPrescales'))


class RDsInspector(BParking2018TriggerMixin, BaseInspector):

    DESCRIPTION      = 'R(Ds) ntuplizer: Bs -> Ds(phi(KK) pi) mu nu, 2018 B-parking'
    CHANNEL          = 'rds'
    CUTS             = cuts
    CANDIDATE        = RDsCandidate
    BRANCHES         = branches
    PATHS            = paths
    EVENT_BRANCHES   = event_branches
    CAND_BRANCHES    = None
    MUON_BRANCHES    = muon_branches
    SAFE_GET         = safe_get
    EVENT_GEN_KEYS   = EVENT_GEN_KEYS
    EVENT_EXTRA_KEYS = EVENT_EXTRA_KEYS
    MIN_MUONS        = 1
    PU_BLOCK         = False    # the Run 3 card does not apply; nti is stored
    EXTRA_HANDLES    = extra_handles
    HAMMER_FACTORY   = staticmethod(make_hammer_ds_session)

    # ==================================================================
    #  SETUP
    # ==================================================================
    def add_arguments(self, parser):
        parser.add_argument('--trk-calib', dest='trk_calib', default='', type=str,
                            help='MC only: track momentum calibration, '
                                 "'<rds_trkcalib_2018BParking_v4.json>[:<variation>]'. "
                                 'The momcal applier rds_trkcalib.py must sit next '
                                 'to the JSON or in data/trkcalib/. See TrackMomentumCalib.')
        parser.add_argument('--no-l1', dest='no_l1', action='store_true',
                            help='do not read the L1 seed tables (L1 branches NaN)')

    def setup(self, options):
        self.options = options
        self.TRK_CALIB = make_trk_calib(getattr(options, 'trk_calib', ''), options.mc)

        self.L1 = None
        if not getattr(options, 'no_l1', False):
            self.L1 = L1Seeds(list(self.PATHS))
            if set(self.L1.seeds) != set(L1_SEEDS):
                raise RuntimeError('the L1 tables hold seeds %s, the schema expects %s'
                                   % (sorted(self.L1.seeds), sorted(L1_SEEDS)))

        self.GEN_PV = GenPVReader(Handle) if options.mc else None

        # ParkingBPH part from the input file names (one part per job is the
        # norm: CRAB and SLURM jobs never mix datasets)
        parts = set(_BPH_RE.findall(options.inputFiles))
        self.BPH_PART = float(parts.pop()) if len(parts) == 1 else NAN
        if not options.mc and len(parts) > 1:
            print('WARNING: input files from several ParkingBPH parts: bph_part = NaN')

        self.n_cands_built = 0

    def extra_trees(self):
        return {'tree_gen': gen_tree_branches} if self.options.mc else {}

    def finish(self, options):
        if self.TRK_CALIB is not None:
            print(self.TRK_CALIB.summary())
        scaler = TrackRecord.cov_scaler
        if scaler is not None and hasattr(scaler, 'n_clamped'):
            print('#### sigma_dxy morph %s: %d tracks where the morph asked for less than '
                  'the beamspot width (ratio to the beamspot error used)'
                  % (scaler.name, scaler.n_clamped))
        if self.GEN_PV is not None:
            print('#### gen PV: read in %d events, missing in %d'
                  % (self.GEN_PV.n_ok, self.GEN_PV.n_missing))

    # ==================================================================
    #  GEN (MC): once per processed event, whatever the trigger did
    # ==================================================================
    def process_event_gen(self, event, options):
        event.rds_gen = None
        if not options.mc:
            return
        decay  = BsSignalDecay.from_genparticles(event.genpr)
        gen_pv = self.GEN_PV(event) if self.GEN_PV is not None else None

        gen = {}
        gen['gen_pv_x'], gen['gen_pv_y'], gen['gen_pv_z'] = gen_pv if gen_pv is not None else (NAN, NAN, NAN)
        gen['gen_sig']  = float(decay.code)
        gen['gen_n_bs'] = float(decay.n_bs)
        gen.update(bs_gen_info(decay if decay.code >= 0 else None, gen_pv))
        gen.update(gen_observables(decay))
        for o, p in (('mu', decay.mu), ('kp', decay.kp), ('km', decay.km), ('pi', decay.pi)):
            gen['gen_vis_%s_pt' % o]  = p.pt()  if p is not None else NAN
            gen['gen_vis_%s_eta' % o] = p.eta() if p is not None else NAN
            gen['gen_vis_%s_phi' % o] = p.phi() if p is not None else NAN
        gen.update(gen_hammer_leaves(decay))
        if self.HAMMER is not None:
            gen.update(self.HAMMER.weights(gen))
        else:
            gen.update(dict.fromkeys(HAMMER_BRANCH_NAMES, NAN))
        event.rds_gen = gen

        aux = event.eventAuxiliary()
        row = dict(gen)
        row['run'], row['lumi'], row['event'] = aux.run(), aux.luminosityBlock(), aux.event()
        row['npu'] = event.pu_at_bx0.getPU_NumInteractions()
        row['nti'] = event.pu_at_bx0.getTrueNumInteractions()
        self.extra_rows['tree_gen'].append(row)

    def setup_event_gen(self, event, options, event_tofill):
        if not options.mc or event.rds_gen is None:
            for b in EVENT_GEN_KEYS:
                event_tofill[b] = NAN
            return None
        for b in EVENT_GEN_KEYS:
            event_tofill[b] = event.rds_gen.get(b, NAN)
        # status-1 charged gen particles for the track matching, once per event
        gen_charged = charged_final_state(event.genpr) + \
                      [p for p in event.genpk if abs(p.pdgId()) in (11, 13, 211, 321, 2212)]
        return gen_charged

    # ==================================================================
    #  EVENT-LEVEL EXTRAS
    # ==================================================================
    def fill_event_extra(self, event, options, event_tofill):
        aux = event.eventAuxiliary()
        event_tofill['era']      = era_of(aux.run()) if not options.mc else NAN
        event_tofill['bph_part'] = self.BPH_PART if not options.mc else NAN

        if self.L1 is not None:
            self.L1.fill(event, event.glb_alg, options.mc, event_tofill)
        else:
            for s in L1_SEEDS:
                event_tofill[s] = event_tofill[s + '_ps'] = NAN

        # L1 prescale range of each HLT path (min / max over its seeds)
        names = event.object().triggerNames(event.trg_res)
        for p in self.PATHS:
            event_tofill[p + '_l1min_ps'] = NAN
            event_tofill[p + '_l1max_ps'] = NAN
        for iname in names.triggerNames():
            iname = str(iname)
            path = drop_hlt_version(iname)
            if path not in self.PATHS:
                continue
            idx = names.triggerIndex(iname)
            for lab, h in (('l1min', event.l1min), ('l1max', event.l1max)):
                if h is None:
                    continue
                try:
                    val = float(h.getPrescaleForIndex(idx))
                except Exception:
                    continue
                key = '%s_%s_ps' % (path, lab)
                old = event_tofill[key]
                event_tofill[key] = val if old != old else max(old, val)

        if options.mc and event.rds_gen is not None:
            for b in ('gen_pv_x', 'gen_pv_y', 'gen_pv_z'):
                event_tofill[b] = event.rds_gen[b]
        else:
            for b in ('gen_pv_x', 'gen_pv_y', 'gen_pv_z'):
                event_tofill[b] = NAN

    # ==================================================================
    #  CANDIDATES
    # ==================================================================
    def _track_records(self, event, options, cuts):
        '''TrackRecord for every track passing the event-level selection, with
        the momentum calibration applied in MC.'''
        aux = event.eventAuxiliary()
        run, lumi, evt = aux.run(), aux.luminosityBlock(), aux.event()
        records = []
        for coll_name, coll in (('pf', event.pf), ('lost', event.ltrk)):
            for idx, tk in enumerate(coll):
                if tk.charge() == 0 or not tk.hasTrackDetails():
                    continue
                if tk.pt() <= cuts['tk_pt'] or abs(tk.eta()) >= cuts['tk_eta']:
                    continue
                if abs(tk.pdgId()) in cuts['tk_veto_pdgid']:
                    continue
                f, z = 1., NAN
                if self.TRK_CALIB is not None:
                    z, f = self.TRK_CALIB.z_and_f(run, lumi, evt,
                                                  TrackMomentumCalib.key(coll_name, idx),
                                                  tk.pt(), tk.eta(), tk.charge(), kind='hadron')
                records.append(TrackRecord(tk, coll_name, idx, f, z))
        return records

    def _muon_wrapper(self, event, imu, idx):
        f, z = 1., NAN
        if self.TRK_CALIB is not None:
            aux = event.eventAuxiliary()
            z, f = self.TRK_CALIB.z_and_f(aux.run(), aux.luminosityBlock(), aux.event(),
                                          TrackMomentumCalib.key('muon', idx),
                                          imu.pt(), imu.eta(), imu.charge(), kind='muon')
        return RDsTrack(TrackRecord(imu, 'muon', idx, f, z), M_MU)

    def build_candidates(self, muons, event, options, cuts, good_tobjs):
        # the sigma_dxy morph (--cov-scale <morph.json>) needs the beamspot
        scaler = TrackRecord.cov_scaler
        if scaler is not None and hasattr(scaler, 'set_beamspot'):
            scaler.set_beamspot(event.bs)

        records = self._track_records(event, options, cuts)
        if len(records) < 3:
            return []
        cutflow['at least 3 tracks'] += 1

        as_k  = [RDsTrack(r, M_K)  for r in records]
        as_pi = [RDsTrack(r, M_PI) for r in records]
        eta_ = np.array([r.obj.eta() for r in records])
        phi_ = np.array([r.obj.phi() for r in records])
        self._photons = [c for c in event.pf if c.pdgId() == 22]
        self._event_tracks = (as_k, as_pi)

        # position of each selected muon in slimmedMuons (calibration key)
        mu_index = dict(((m.pt(), m.eta(), m.phi()), i) for i, m in enumerate(event.muons))

        cands = []
        for imu in muons:
            mu = self._muon_wrapper(event, imu, mu_index.get((imu.pt(), imu.eta(), imu.phi()), -1))
            # PV closest in z to the muon, for the track preselection only
            pv = min(event.vtx, key=lambda v: abs(imu.bestTrack().dz(v.position())))
            mu_dz = imu.bestTrack().dz(pv.position())

            dphi = (phi_ - imu.phi() + np.pi) % (2 * np.pi) - np.pi
            dr = np.sqrt((eta_ - imu.eta()) ** 2 + dphi ** 2)
            near = np.nonzero((dr > cuts['min_dr_mu_tk']) & (dr < cuts['max_dr_mu_tk']))[0]
            sel = []
            for i in near:
                trk = records[i].obj.bestTrack()
                if abs(trk.dz(pv.position()) - mu_dz) >= cuts['max_dz_mu_tk']:
                    continue
                if abs(trk.dxy(pv.position())) >= cuts['max_dxy_tk']:
                    continue
                sel.append(i)
            if len(sel) < 3:
                continue
            cutflow['\tmuons with >= 3 tracks'] += 1

            # leading first, so that k1 is the leading kaon
            sel.sort(key=lambda i: as_k[i].pt(), reverse=True)
            sel = np.array(sel)
            px = np.array([as_k[i].px() for i in sel])
            py = np.array([as_k[i].py() for i in sel])
            pz = np.array([as_k[i].pz() for i in sel])
            triplets = kkpi_triplets(px, py, pz, mu.np4(), cuts['phi_window'],
                                     cuts['kkpi_min'], cuts['kkpi_max'], cuts['max_mass'],
                                     phi_mass=cuts['phi_mass'])
            cutflow['\ttriplets in the mass windows'] += len(triplets)

            for a, b, c in triplets:
                i, j, k = sel[a], sel[b], sel[c]
                cand = RDsCandidate(mu, as_k[i], as_k[j], as_pi[k])
                cand.fit()
                if not cand.passes_vertex_cuts(cuts):
                    continue
                cutflow['\tcandidates passing the vertex cuts'] += 1
                cands.append(cand)
        return cands

    def sort_candidates(self, cands):
        cands.sort(key=lambda c: c.pt(), reverse=True)
        for i, c in enumerate(cands):
            c.cand_idx = i
        return cands

    # ==================================================================
    #  ONE ROW PER CANDIDATE
    # ==================================================================
    def fill_candidate(self, icand, event, options, cuts, gen_state, good_tobjs):
        icand.compute(event.vtx, event.bs, event.pf, event.ltrk, self._photons, cuts)
        icand.out['cand_idx'] = icand.cand_idx

        mu = icand.mu
        for obj in (mu, icand.k1, icand.k2, icand.pi):
            obj.pv = icand.pv_bs
            obj.bs = icand.bs
            if not hasattr(obj, '_jet_done'):
                jet, dr2 = bestMatch(obj, event.jets)
                if dr2 < self._JET_MATCH_DR2:
                    obj.jet = jet
                obj._jet_done = True
        mu.iso03 = mu.pfIsolationR03()
        mu.iso04 = mu.pfIsolationR04()

        # ---- gen matching (MC): once per object per event ----------------------
        if options.mc and gen_state is not None:
            for obj in (mu, icand.k1, icand.k2, icand.pi):
                if not hasattr(obj, '_gen_done'):
                    g, dr = match_track(obj, gen_state, cuts['gen_dr'])
                    if g is not None:
                        obj.gen_match = g
                        obj.gen_dr = dr
                        m = mother_of(g)
                        obj.gen_mother_pdgid = m.pdgId() if m is not None else 0.
                    obj._gen_done = True
            icand.out.update(classify_candidate(
                *[getattr(o, 'gen_match', None) for o in (icand.k1, icand.k2, icand.pi, mu)]))

        cand_tofill = self._CAND_TEMPLATE.copy()

        for branch, getter in muon_branches.items():
            cand_tofill['mu_%s' % branch] = safe_get(getter, mu, verbose=options.verbose, name=branch)

        # trigger matching of the muon to the parking objects
        trig_dr = self.trigger_dr(mu, good_tobjs)
        cand_tofill['mu_trig_dr']    = trig_dr
        cand_tofill['mu_trig_match'] = float(trig_dr < cuts['hlt_dr']) if trig_dr == trig_dr else 0.
        for (idx, path), val in self.path_tags([mu], event, good_tobjs, cuts['hlt_dr']).items():
            cand_tofill['mu_%s_tag' % path] = val

        for name, obj in (('k1', icand.k1), ('k2', icand.k2), ('pi', icand.pi)):
            for branch, getter in track_branches.items():
                cand_tofill['%s_%s' % (name, branch)] = safe_get(getter, obj, verbose=options.verbose, name=branch)

        out = icand.out
        for name in cand_names:
            if name in out:
                cand_tofill[name] = out[name]
        return cand_tofill
