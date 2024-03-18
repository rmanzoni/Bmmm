from __future__ import print_function
import re
import sys
import particle
import numpy as np
from array import array
from scipy import stats
from particle import Particle
from collections import defaultdict, Callable, OrderedDict

import ROOT
ROOT.gSystem.Load('libBmmmAnalysis')
from ROOT import KVFitter # VertexDistance3D is contained here, dirt trick!!

##########################################################################################
##########################################################################################

masses = {}
masses['bs' ] = particle.literals.B_s_0    .mass/1000.
masses['phi'] = particle.literals.phi_1020 .mass/1000.
masses['ds' ] = particle.literals.D_s_minus.mass/1000.
masses['k'  ] = particle.literals.K_plus   .mass/1000.
masses['pi' ] = particle.literals.pi_plus  .mass/1000.
masses['mu' ] = particle.literals.mu_plus  .mass/1000.

##########################################################################################
##########################################################################################

def drop_hlt_version(string, pattern=r"_v\d+"):
    regex = re.compile(pattern + "$")
    if regex.search(string):
        match = re.search(pattern, string)
        return string[:match.start()]
    else:
        return string

##########################################################################################
##########################################################################################

# two different implementations for python 2/3
# in python 3 dictionaries are sorted by default

class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))

def zero():
    return 0

ver = sys.version_info[0]

cutflow = defaultdict(zero) if ver==3 else DefaultOrderedDict(zero)

##########################################################################################
##########################################################################################

diquarks = [
    1103,
    2101,
    2103,
    2203,
    3101,
    3103,
    3201,
    3203,
    3303,
    4101,
    4103,
    4201,
    4203,
    4301,
    4303,
    4403,
    5101,
    5103,
    5201,
    5203,
    5301,
    5303,
    5401,
    5403,
    5503,
]

excitedBs = [
    513,
    523,
    533,
    543,
    # others?
]

##########################################################################################
##########################################################################################

def isAncestor(a, p):
    if a == p :
        return True
    for i in xrange(0,p.numberOfMothers()):
        if isAncestor(a,p.mother(i)):
            return True
    return False

##########################################################################################
##########################################################################################

def printAncestors(particle, ancestors=[], verbose=True):
    for i in xrange(0, particle.numberOfMothers()):
        mum = particle.mother(i)
#         if mum is None: import pdb ; pdb.set_trace()
        if abs(mum.pdgId())<8 or \
           abs(mum.pdgId())==21 or \
           abs(mum.pdgId()) in diquarks or\
           abs(mum.pdgId()) in excitedBs or\
           abs(mum.eta()) > 1000: # beam protons
            continue
        # don't count B oscillations
        if mum.pdgId() == -particle.pdgId() and abs(particle.pdgId()) in [511, 531]:
            continue 
        if not mum.isLastCopy(): continue
        try:
            if verbose: print(' <-- ', Particle.from_pdgid(mum.pdgId()).name, end = '')
            ancestors.append(mum)
            printAncestors(mum, ancestors=ancestors, verbose=verbose)
        except:
            if verbose: print(' <-- ', 'pdgid', mum.pdgId(), end = '')
            ancestors.append(mum)
            printAncestors(mum, ancestors=ancestors, verbose=verbose)
        else:
            pass
    particle.ancestors = ancestors

##########################################################################################
##########################################################################################

def compute_vertex_quantities(vtx, bs, p4, pv, full=False):

    vtx.chi2 = vtx.normalisedChiSquared()
    vtx.prob = (1. - stats.chi2.cdf(vtx.chi2, 1)) 
        
    vtx.lxy  = ROOT.VertexDistanceXY().distance(bs, vtx.vertexState())
    # FIXME! want BS for transverse quantities
    vtx.lxyz = ROOT.VertexDistance3D().distance(pv, vtx.vertexState())

    # 2D
    vect_lxy = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
        vtx.position().x() - bs.position().x(),
        vtx.position().y() - bs.position().y(),
        0. 
    )
    
    vect_pt = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
        p4.px(),
        p4.py(),
        0.
    )

    vtx.cos2d = vect_pt.Dot(vect_lxy) / (vect_pt.R() * vect_lxy.R())
    
    # 3D
    vect_lxyz = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
        vtx.position().x() - bs.position().x(), # transverse quantities always from BS
        vtx.position().y() - bs.position().y(), # transverse quantities always from BS
        vtx.position().z() - pv.position().z(),
    )
    
    vect_p = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
        p4.px(),
        p4.py(),
        p4.pz(),
    )
 
    vtx.cos3d = vect_p.Dot(vect_lxyz) / (vect_p.R() * vect_lxyz.R())
       
    if full:
        # transverse coordinate from beamspot
        pv_to_sv = ROOT.Math.XYZVector(
            (vtx.position().x() - bs.position().x()), 
            (vtx.position().y() - bs.position().y()),
            (vtx.position().z() - pv.position().z())
        )
    
        direction     = pv_to_sv/np.sqrt(pv_to_sv.Mag2())                  
        direction_eta = direction.eta()                                
        direction_phi = direction.phi() 
                                       
        p4_par  = p4.Vect().Dot(direction)                   
        p4_perp = np.sqrt(p4.Vect().Mag2() - p4_par*p4_par)
        mcorr   = np.sqrt(p4.mass()*p4.mass() + p4_perp*p4_perp) + p4_perp
        
        return vtx, p4_par, p4_perp, mcorr
    
    else:
        return vtx
        
##########################################################################################
##########################################################################################

def p4_with_mass(particle, mass, root_type=0):
    vec = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<double>')(particle.pt(), particle.eta(), particle.phi(), mass)
    if root_type==0:
        return vec
    elif root_type==1:
        return ROOT.Math.LorentzVector('ROOT::Math::PxPyPzE4D<double>')(vec.px(), vec.py(), vec.pz(), vec.energy())
    else:
        new_vec = ROOT.TLorentzVector() 
        new_vec.SetPtEtaPhiE(vec.pt(), vec.eta(), vec.phi(), vec.energy())
        return new_vec
        

def fillRecoTree(ntuple_reco, tofill_reco):
    ntuple_reco.Fill(array('f', tofill_reco.values()))

##########################################################################################
##########################################################################################

def isMyDs(ds, minpt=0.5, maxeta=2.5):
    daus = []
    for idau in range(ds.numberOfDaughters()):
        dau = ds.daughter(idau)
        if dau.pdgId()==22: 
            continue # exclude FSR
        if abs(dau.pdgId())==211:
            if dau.pt()<minpt or abs(dau.eta())>maxeta:
                continue # only pions in the acceptance
            ds.pion = dau
        if abs(dau.pdgId())==333:
            if dau.numberOfDaughters()!=2: 
                continue
            for jdau in range(dau.numberOfDaughters()):
                if abs(dau.daughter(jdau).pdgId())!=321 or \
                   dau.daughter(jdau).pt < minpt        or \
                   abs(dau.daughter(jdau).eta()) > maxeta:
                    continue # only kaons in the acceptance
            ds.phi_meson = dau
        daus.append(dau.pdgId())
    daus.sort(key = lambda x : abs(x))
    return daus==[211, 333] or daus==[-211, 333]


##########################################################################################
##########################################################################################

def convert_cov(m):
    return np.array([[m(i,j) for j in range(m.kCols)] for i in range(m.kRows)])

def is_pos_def(x):
    '''
    https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    '''
    return np.all(np.linalg.eigvals(x) > 0)

def fix_track(trk, delta=1e-9):
    '''
    https://github.com/CMSKStarMuMu/miniB0KstarMuMu/blob/master/miniKstarMuMu/plugins/miniKstarMuMu.cc#L1611-L1678
    '''
    
    cov = convert_cov(trk.covariance())
    
    if is_pos_def(cov): 
        return trk
    
    if int(np.__version__.split('.')[1])<17:
        new_cov = np.nan_to_num(cov) # missing keyword, check docs
    else:
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

    if not is_pos_def(new_cov): 
        fix_track(new_trk, delta)
    
    return new_trk

##########################################################################################
##########################################################################################

#@np.njit
def compute_mass(p1, p2, m1, m2, p1p2):
    mass_squared = m1**2 + m2**2 + 2*np.sqrt(m1**2 + p1**2)*np.sqrt(m2**2 + p2**2) - 2*p1p2
    return np.sqrt(mass_squared) 




