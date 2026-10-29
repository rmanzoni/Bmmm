import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunesRun3ECM13p6TeV.PythiaCP5Settings_cfi import *
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *

_generator = cms.EDFilter(
    "Pythia8GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    comEnergy = cms.double(13600.0),
    ExternalDecays = cms.PSet(
        EvtGen130 = cms.untracked.PSet(
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2014_NOLONGLIFE.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt_2014.pdl'),
            list_forced_decays     = cms.vstring('MyBs0', 'MyAntiBs0'),        
            operates_on_particles = cms.vint32(531, -531),
            convertPythiaCodes = cms.untracked.bool(False),
            user_decay_embedded= cms.vstring([
                # https://github.com/cms-data/GeneratorInterface-EvtGenInterface/blob/master/DECAY_2020_NOLONGLIFE.DEC
                'Alias      MyBs0      B_s0',
                'Alias      MyAntiBs0  anti-B_s0',
                'ChargeConj MyBs0      MyAntiBs0',
                'Decay MyBs0',
                '  1.       mu+     mu-     mu+     mu-     PHSP;',
                'Enddecay',
                'CDecay MyAntiBs0',
                'End',
            ]),
        ),
        parameterSets = cms.vstring('EvtGen130')
    ),
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            'SoftQCD:nonDiffractive = on',
            'PTFilter:filter = on', # this turn on the filter
            'PTFilter:quarkToFilter = 5', # PDG id of q quark
            'PTFilter:scaleToFilter = 1.0',
		),
        parameterSets = cms.vstring(
            'pythia8CommonSettings',
            'pythia8CP5Settings',
            'processParameters',
        ),
    ),
)

_generator.PythiaParameters.processParameters.extend(EvtGenExtraParticles)
from GeneratorInterface.Core.ExternalGeneratorFilter import ExternalGeneratorFilter
generator = ExternalGeneratorFilter(_generator)

FourMuonFilter = cms.EDFilter(
    "MCMultiParticleFilter",
    NumRequired = cms.int32(4),
    AcceptMore  = cms.bool(True),
    ParticleID  = cms.vint32(13),
    PtMin       = cms.vdouble(1.),
    EtaMax      = cms.vdouble(2.5),
    Status      = cms.vint32(1),
)

ProductionFilterSequence = cms.Sequence(
    generator 
    * FourMuonFilter
)


'''

see a recent 2022 Run3 example here

https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_setup/EGM-Run3Summer22wmLHEGS-00004

cmsDriver.py Configuration/GenProduction/python/Bs4Mu-fragment.py \
--python_filename Bs4Mu_GEN_cfg.py \
--eventcontent RAWSIM \
--customise Configuration/DataProcessing/Utils.addMonitoring \
--datatier GEN \
--fileout file:Bs4Mu_GEN.root \
--conditions 124X_mcRun3_2022_realistic_v12 \
--beamspot Realistic25ns13p6TeVEarly2022Collision \
--step GEN \
--geometry DB:Extended \
--era Run3 \
--no_exec \
--mc -n 10000 \
--customise_commands="from GeneratorInterface.Core.ExternalGeneratorFilter import ExternalGeneratorFilter; process.generator=ExternalGeneratorFilter(process.generator)" \
--nThreads 8 


OR GENSIM HERE!
https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_setup/BPH-Run3Summer22GS-00058

cmsDriver.py Configuration/GenProduction/python/BPH-Run3Summer22GS-00058-fragment.py \
--python_filename BPH-Run3Summer22GS-00058_1_cfg.py \
--eventcontent RAWSIM \
--customise Configuration/DataProcessing/Utils.addMonitoring \
--datatier GEN-SIM \
--fileout file:BPH-Run3Summer22GS-00058.root \
--conditions 124X_mcRun3_2022_realistic_v12 \
--beamspot Realistic25ns13p6TeVEarly2022Collision \
--step GEN,SIM \
--geometry DB:Extended \
--era Run3 \
--no_exec \
--mc \
-n 100





#############################
####       GEN-SIM        ###
#############################
cmsDriver.py Configuration/GenProduction/python/Bs4Mu-fragment.py \
--python_filename Bs4Mu_GENSIM_cfg.py \
--eventcontent RAWSIM \
--customise Configuration/DataProcessing/Utils.addMonitoring \
--datatier GEN-SIM \
--fileout file:Bs4Mu_GENSIM.root \
--conditions 124X_mcRun3_2022_realistic_v12 \
--beamspot Realistic25ns13p6TeVEarly2022Collision \
--step GEN,SIM \
--geometry DB:Extended \
--era Run3 \
--no_exec \
--mc -n 10000 \
--nThreads 8 



'''