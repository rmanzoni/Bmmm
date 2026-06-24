'''
Branch definitions for the GEN-level tau -> 3mu inspector.

Same idiom as B4MuBranches: dictionaries of {branch_name : getter} that are
evaluated against the event / candidate / muon objects, plus a flat `branches`
list used to initialise the per-row `tofill` dictionary.
'''

import numpy as np


##########################################################################################
#####      EVENT
##########################################################################################
event_branches = {
    'run'    : lambda ev : ev.eventAuxiliary().run()             ,
    'lumi'   : lambda ev : ev.eventAuxiliary().luminosityBlock() ,
    'event'  : lambda ev : ev.eventAuxiliary().event()           ,
    'ncands' : lambda ev : ev.ncands                             ,
}


##########################################################################################
#####      SINGLE MUON  (applied with a per-muon prefix)
##########################################################################################
muon_branches = {
    'pt'     : lambda mu : mu.pt()     ,
    'eta'    : lambda mu : mu.eta()    ,
    'phi'    : lambda mu : mu.phi()    ,
    'mass'   : lambda mu : mu.mass()   ,
    'e'      : lambda mu : mu.energy() ,
    'px'     : lambda mu : mu.px()     ,
    'py'     : lambda mu : mu.py()     ,
    'pz'     : lambda mu : mu.pz()     ,
    'charge' : lambda mu : mu.charge() ,
    'pdgid'  : lambda mu : mu.pdgId()  ,
    'vx'     : lambda mu : mu.vx()     ,
    'vy'     : lambda mu : mu.vy()     ,
    'vz'     : lambda mu : mu.vz()     ,
}

# order matters: first the prompt muon from the tau, then the displaced pair
muon_labels = ['mu_tau', 'mu_disp1', 'mu_disp2']


##########################################################################################
#####      CANDIDATE
##########################################################################################
cand_branches = {
    # whole three-muon system
    'tau3mu_pt'     : lambda c : c.pt()             ,
    'tau3mu_eta'    : lambda c : c.eta()            ,
    'tau3mu_phi'    : lambda c : c.phi()            ,
    'tau3mu_mass'   : lambda c : c.mass()           ,
    'tau3mu_e'      : lambda c : c.energy()         ,
    'tau3mu_px'     : lambda c : c.p4().px()        ,
    'tau3mu_py'     : lambda c : c.p4().py()        ,
    'tau3mu_pz'     : lambda c : c.p4().pz()        ,
    'tau3mu_charge' : lambda c : c.charge()         ,

    # displaced opposite-sign muon pair
    'pair_pt'       : lambda c : c.pair_p4().pt()   ,
    'pair_eta'      : lambda c : c.pair_p4().eta()  ,
    'pair_phi'      : lambda c : c.pair_p4().phi()  ,
    'pair_mass'     : lambda c : c.pair_p4().mass() ,
    'pair_e'        : lambda c : c.pair_p4().energy(),
    'pair_px'       : lambda c : c.pair_p4().px()   ,
    'pair_py'       : lambda c : c.pair_p4().py()   ,
    'pair_pz'       : lambda c : c.pair_p4().pz()   ,
    'pair_charge'   : lambda c : c.pair_charge()    ,

    # displacement of the pair (cm)
    'decay_length'  : lambda c : c.decay_length()   ,
    'lxy'           : lambda c : c.lxy()            ,
    'lz'            : lambda c : c.lz()             ,
    'ctau'          : lambda c : c.ctau()          ,
    'pv_x'          : lambda c : c.prod_vertex().x(),
    'pv_y'          : lambda c : c.prod_vertex().y(),
    'pv_z'          : lambda c : c.prod_vertex().z(),
    'sv_x'          : lambda c : c.decay_vertex().x(),
    'sv_y'          : lambda c : c.decay_vertex().y(),
    'sv_z'          : lambda c : c.decay_vertex().z(),

    # gen scalar (for cross-checks)
    'scalar_pt'     : lambda c : c.scalar.pt()      ,
    'scalar_eta'    : lambda c : c.scalar.eta()     ,
    'scalar_phi'    : lambda c : c.scalar.phi()     ,
    'scalar_mass'   : lambda c : c.scalar.mass()    ,

    # angular separation between the scalar and the prompt muon
    'dr_scalar_mu'  : lambda c : c.dr_scalar_mu()   ,
}


##########################################################################################
#####      FLAT LIST OF ALL BRANCHES
##########################################################################################
branches = list(event_branches.keys())
for label in muon_labels:
    branches += ['%s_%s' % (label, b) for b in muon_branches.keys()]
branches += list(cand_branches.keys())
