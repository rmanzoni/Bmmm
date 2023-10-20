import FWCore.ParameterSet.Config as cms

"""
this works for conditions which change on the run boundary
it cycles through runs 272007 - 400000 and writes the record out for each one
"""

process = cms.Process("WriteTest")

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ('analysis') 
#default is Run2016B
options.register('startRunNr',272007,options.multiplicity.singleton,options.varType.int,"start run number ")
options.register('endRunNr',400000,options.multiplicity.singleton,options.varType.int,"end run number" )
# options.register('globalTag',"113X_dataRun3_HLT_v3",options.multiplicity.singleton,options.varType.string,"global tag" )
options.register('globalTag',"102X_upgrade2018_realistic_v15",options.multiplicity.singleton,options.varType.string,"global tag" )

options.parseArguments()

process.source = cms.Source( "EmptySource",
                             numberEventsInRun=cms.untracked.uint32(1),
                             firstRun=cms.untracked.uint32(options.startRunNr)
                           )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.endRunNr-options.startRunNr+1))
 
process.writer = cms.EDAnalyzer("FWLiteESRecordWriterAnalyzer",
   fileName = cms.untracked.string(options.outputFile),
   L1TUtmTriggerMenuRcd = cms.untracked.VPSet(
       cms.untracked.PSet(
           type=cms.untracked.string("L1TUtmTriggerMenu"),
           label=cms.untracked.string("")
           )
       ),
   L1TGlobalPrescalesVetosRcd = cms.untracked.VPSet(
       cms.untracked.PSet(
         type=cms.untracked.string("L1TGlobalPrescalesVetos"),
           label=cms.untracked.string("")
           )
       )  
   )
process.GlobalTag = cms.ESSource( "PoolDBESSource",
    globaltag = cms.string( options.globalTag ),
    RefreshEachRun = cms.untracked.bool( False ),
    snapshotTime = cms.string( "" ),
    toGet = cms.VPSet( 
    ),
    pfnPostfix = cms.untracked.string( "None" ),
    DBParameters = cms.PSet( 
      connectionRetrialTimeOut = cms.untracked.int32( 60 ),
      idleConnectionCleanupPeriod = cms.untracked.int32( 10 ),
      enableReadOnlySessionOnUpdateConnection = cms.untracked.bool( False ),
      enablePoolAutomaticCleanUp = cms.untracked.bool( False ),
      messageLevel = cms.untracked.int32( 0 ),
      authenticationPath = cms.untracked.string( "." ),
      connectionRetrialPeriod = cms.untracked.int32( 10 ),
      connectionTimeOut = cms.untracked.int32( 0 ),
      enableConnectionSharing = cms.untracked.bool( True )
    ),
    RefreshAlways = cms.untracked.bool( False ),
    connect = cms.string( "frontier://FrontierProd/CMS_CONDITIONS" ),
    ReconnectEachRun = cms.untracked.bool( False ),
    RefreshOpenIOVs = cms.untracked.bool( False ),
    DumpStat = cms.untracked.bool( False )
)

process.out = cms.EndPath(process.writer)
