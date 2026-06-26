from DataFormats.FWLite import Handle
from collections import OrderedDict

# ----------------------------------------------------------------------------
# FWLite handles for the displaced tau -> 3mu ntuplizer, reading (private) MiniAOD.
#
# 'pf' is the packedPFCandidates collection: it is used BOTH for the R=0.4 cone
# stored around the 3mu axis AND, together with 'ltrk' (lostTracks), to rebuild
# the PV track set for the beamspot-constrained PV refit (the same trick as the
# rjpsi skim, see Tau3MuCandidate.refit_primary_vertex).
# ----------------------------------------------------------------------------

handles_mc = OrderedDict()
handles_mc['genpr'] = ('prunedGenParticles'  , Handle('std::vector<reco::GenParticle>')     )
handles_mc['genpk'] = ('packedGenParticles'  , Handle('std::vector<pat::PackedGenParticle>'))
handles_mc['pu'   ] = ('slimmedAddPileupInfo', Handle('std::vector<PileupSummaryInfo>')     )

handles = OrderedDict()
handles['muons'  ] = ('slimmedMuons'                 , Handle('std::vector<pat::Muon>')                   )
handles['pf'     ] = ('packedPFCandidates'           , Handle('std::vector<pat::PackedCandidate>')        )
handles['ltrk'   ] = ('lostTracks'                   , Handle('std::vector<pat::PackedCandidate>')        )
handles['vtx'    ] = ('offlineSlimmedPrimaryVertices', Handle('std::vector<reco::Vertex>')                )
handles['trg_res'] = (('TriggerResults', '', 'HLT' ) , Handle('edm::TriggerResults'        )              )
handles['trg_ps' ] = (('patTrigger'    , '')         , Handle('pat::PackedTriggerPrescales')              )
handles['bs'     ] = ('offlineBeamSpot'              , Handle('reco::BeamSpot')                           )
handles['tobjs'  ] = ('slimmedPatTrigger'            , Handle('std::vector<pat::TriggerObjectStandAlone>'))
handles['jets'   ] = ('slimmedJets'                  , Handle('std::vector<pat::Jet>')                    )
# PUPPI MET: a one-entry std::vector<pat::MET>; the event-level branches read
# event.met[0]. Used for the 3mu transverse mass (the longitudinal-nu reco does
# NOT use MET: it fixes the nu transverse momentum from the PV->SV direction).
handles['met'    ] = ('slimmedMETsPuppi'             , Handle('std::vector<pat::MET>')                    )
