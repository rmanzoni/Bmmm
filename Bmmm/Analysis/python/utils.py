from __future__ import print_function
import re
import sys
import json
import gzip
import particle
import numpy as np
from array import array
from scipy import stats
from particle import Particle
from collections import defaultdict, OrderedDict
try:
    from collections.abc import Callable   # python >= 3.3 (mandatory from 3.10)
except ImportError:
    from collections import Callable       # legacy python 2

import ROOT
ROOT.gSystem.Load('libBmmmAnalysis')
from ROOT import KVFitter # VertexDistance3D is contained here, dirt trick!!

##########################################################################################
##########################################################################################

masses = {}
masses['bs'  ] = particle.literals.B_s_0    .mass/1000.
masses['phi' ] = particle.literals.phi_1020 .mass/1000.
masses['ds'  ] = particle.literals.D_s_minus.mass/1000.
masses['k'   ] = particle.literals.K_plus   .mass/1000.
masses['pi'  ] = particle.literals.pi_plus  .mass/1000.
masses['mu'  ] = particle.literals.mu_plus  .mass/1000.
masses['jpsi'] = particle.literals.Jpsi_1S  .mass/1000.

##########################################################################################
##########################################################################################

_HLT_VER_RE = re.compile(r'(_part\d+|_v\d+)+$')
def drop_hlt_version(s): 
    return _HLT_VER_RE.sub('', s)

# def drop_hlt_version(string, pattern=r"_v\d+"):
#     regex = re.compile(pattern + "$")
#     if regex.search(string):
#         match = re.search(pattern, string)
#         return string[:match.start()]
#     else:
#         return string

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
                   dau.daughter(jdau).pt() < minpt      or \
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

##########################################################################################
#####      TRACK COVARIANCE MATRIX: persistency + rescaling
##########################################################################################
# reco::Track stores a 5x5 SYMMETRIC covariance over the curvilinear parameters,
# in the reco::TrackBase order below. 15 independent elements (i <= j), taken
# row-major over the upper triangle -- the SAME order SMatrix wants when it is
# built from a flat SVector, which is what fix_track/track_with_cov rely on.
COV_PARAM_NAMES = ('qoverp', 'lambda', 'phi', 'dxy', 'dsz')
COV_INDEX_PAIRS = [(i, j) for i in range(5) for j in range(i, 5)]
COV_ELEMENT_NAMES = ['%s_%s' % (COV_PARAM_NAMES[i], COV_PARAM_NAMES[j])
                     for i, j in COV_INDEX_PAIRS]
COV_NO_SCALE = (1., 1., 1., 1., 1.)

def cov_upper_triangle(cov):
    '''The 15 independent elements of a 5x5 covariance, in COV_INDEX_PAIRS order.
    This is what goes into the ntuple, one branch per element.'''
    return [cov[i][j] for i, j in COV_INDEX_PAIRS]

def smatrix55_from_cov(cov):
    '''numpy 5x5 -> ROOT::Math::SMatrix<double,5,5,MatRepSym>, via the flat
    upper triangle. https://root.cern/doc/v606/SMatrixDoc.html'''
    upper = np.asarray(cov_upper_triangle(cov), dtype=np.float64)
    return ROOT.Math.SMatrix('double', 5, 5, ROOT.Math.MatRepSym('double', 5))(
        ROOT.Math.SVector('double', len(upper))(upper, len(upper)), False)

def track_with_cov(trk, cov):
    '''A new reco::Track identical to trk except for its covariance matrix.

    reco::Track has no covariance setter -- it is built once and then read -- so
    the only way to hand a modified covariance to a vertex fitter is to rebuild
    the track around it. Everything else (reference point, momentum, charge,
    chi2/ndof, algo, quality) is copied over, so the trajectory is untouched and
    only the uncertainty changes.'''
    return ROOT.reco.Track(
        trk.chi2(),
        trk.ndof(),
        trk.referencePoint(),
        trk.momentum(),
        trk.charge(),
        smatrix55_from_cov(cov),
        trk.algo(),
        ROOT.reco.TrackBase.TrackQuality(trk.qualityMask()),
    )

def scale_cov(cov, scales):
    '''Rescale the uncertainty of each track parameter WITHOUT touching any
    correlation: cov' = D cov D with D = diag(scales).

        sigma_i' = scales[i] * sigma_i          (diagonal picks up scales[i]^2)
        rho_ij'  = cov'_ij / (sigma_i' sigma_j')
                 = scales[i] scales[j] cov_ij / (scales[i] sigma_i scales[j] sigma_j)
                 = rho_ij                        exactly, by construction

    so the correction can be measured and applied one parameter at a time (say
    sigma_dxy) without silently deforming the rest of the matrix. Positive
    definiteness is preserved for any positive scales (congruence transform).

    Note the 5th parameter is dsz, not dz: dzError() = sigma_dsz / |sin(theta)|,
    so scaling dsz scales sigma_dz by the same factor.
    '''
    d = np.diag(np.asarray(scales, dtype=np.float64))
    return d.dot(np.asarray(cov, dtype=np.float64)).dot(d)

def scale_track_cov(trk, scales, cov=None):
    '''reco::Track -> reco::Track with sigma_i -> scales[i] * sigma_i.
    Returns the input track untouched when the scales are all unity, so the
    no-correction path costs nothing. `cov` lets the caller pass the already
    converted numpy matrix (candidates memoize it as obj.cov).'''
    if scales is None or np.allclose(scales, 1.):
        return trk
    if cov is None:
        cov = convert_cov(trk.covariance())
    return track_with_cov(trk, scale_cov(cov, scales))

##########################################################################################

class CovScaler(object):
    '''Per-track covariance scale factors, one per curvilinear parameter.

    Subclasses implement scales(trk) -> 5-tuple in COV_PARAM_NAMES order. The
    magnitudes come from a data/MC measurement (which is what the new cov_*
    branches are for); this class is only the plumbing that applies them.
    '''
    def scales(self, trk):
        raise NotImplementedError

    def __call__(self, trk):
        return self.scales(trk)

class ConstantCovScaler(CovScaler):
    '''Flat scale factor per parameter, e.g. ConstantCovScaler(dxy=1.05).
    Mostly for closure tests and systematics: shift one parameter by a known
    amount and check what moves downstream.'''
    def __init__(self, **kwargs):
        vals = list(COV_NO_SCALE)
        for key, val in kwargs.items():
            if key not in COV_PARAM_NAMES:
                raise ValueError('unknown track parameter %r, expected one of %s'
                                 % (key, list(COV_PARAM_NAMES)))
            vals[COV_PARAM_NAMES.index(key)] = float(val)
        self._scales = tuple(vals)

    def scales(self, trk):
        return self._scales

class BinnedCovScaler(CovScaler):
    '''Scale factors from a 2D (pt, |eta|) lookup table, the shape the
    measurement naturally produces. JSON schema:

        {"pt_edges"      : [1.0, 2.0, ...],
         "abs_eta_edges" : [0.0, 0.8, ...],
         "scales"        : {"dxy": [[...], ...]}}   # [ipt][ieta]

    Values outside the covered range are clamped to the edge bin: an
    extrapolated correction is worse than a frozen one, but you should know how
    often it happens -- n_out_of_range counts it.
    '''
    def __init__(self, pt_edges, abs_eta_edges, scales):
        self.pt_edges      = np.asarray(pt_edges, dtype=np.float64)
        self.abs_eta_edges = np.asarray(abs_eta_edges, dtype=np.float64)
        self.tables = {}
        for key, table in scales.items():
            if key not in COV_PARAM_NAMES:
                raise ValueError('unknown track parameter %r, expected one of %s'
                                 % (key, list(COV_PARAM_NAMES)))
            arr = np.asarray(table, dtype=np.float64)
            expected = (len(self.pt_edges) - 1, len(self.abs_eta_edges) - 1)
            if arr.shape != expected:
                raise ValueError('scale table for %r has shape %s, expected %s'
                                 % (key, arr.shape, expected))
            self.tables[key] = arr
        self.n_out_of_range = 0

    @classmethod
    def from_json(cls, path):
        with open(path) as fin:
            payload = json.load(fin)
        return cls(payload['pt_edges'], payload['abs_eta_edges'], payload['scales'])

    def scales(self, trk):
        pt, abs_eta = trk.pt(), abs(trk.eta())
        if not (self.pt_edges[0] <= pt < self.pt_edges[-1]) or \
           not (self.abs_eta_edges[0] <= abs_eta < self.abs_eta_edges[-1]):
            self.n_out_of_range += 1
        ipt  = int(np.clip(np.searchsorted(self.pt_edges,      pt,      'right') - 1,
                           0, len(self.pt_edges) - 2))
        ieta = int(np.clip(np.searchsorted(self.abs_eta_edges, abs_eta, 'right') - 1,
                           0, len(self.abs_eta_edges) - 2))
        vals = list(COV_NO_SCALE)
        for key, table in self.tables.items():
            vals[COV_PARAM_NAMES.index(key)] = float(table[ipt][ieta])
        return tuple(vals)

class CorrectionlibCovScaler(CovScaler):
    '''Scale factors from a correctionlib CorrectionSet, so the same JSON can be
    shared with other analyses. `corrections` maps a track parameter to the
    correction name; each correction is evaluated as (pt, |eta|).'''
    def __init__(self, path, corrections):
        import correctionlib  # lazy: not a dependency of the rest of the package
        self.cset = correctionlib.CorrectionSet.from_file(path)
        for name in corrections.values():
            if name not in self.cset:
                raise KeyError('correction %r not in %s' % (name, path))
        for key in corrections:
            if key not in COV_PARAM_NAMES:
                raise ValueError('unknown track parameter %r, expected one of %s'
                                 % (key, list(COV_PARAM_NAMES)))
        self.corrections = dict(corrections)

    def scales(self, trk):
        pt, abs_eta = trk.pt(), abs(trk.eta())
        vals = list(COV_NO_SCALE)
        for key, name in self.corrections.items():
            vals[COV_PARAM_NAMES.index(key)] = float(self.cset[name].evaluate(pt, abs_eta))
        return tuple(vals)

def make_cov_scaler(spec):
    '''Build a CovScaler from a command-line string (see --cov-scale):

        ''                      -> None, no scaling at all (the default)
        'dxy=1.05,dsz=1.02'     -> ConstantCovScaler
        'table.json'            -> BinnedCovScaler.from_json, or
                                   CorrectionlibCovScaler if the file is a
                                   correctionlib CorrectionSet (schema_version)
        'table.json:dxy=sigma_dxy_scale'
                                -> CorrectionlibCovScaler with an explicit
                                   parameter -> correction-name mapping
    '''
    if spec is None or not spec.strip():
        return None
    spec = spec.strip()

    # correctionlib file with an explicit mapping
    if ':' in spec and spec.split(':', 1)[0].endswith(('.json', '.json.gz')):
        path, mapping = spec.split(':', 1)
        corrections = dict(tok.split('=', 1) for tok in mapping.split(',') if tok)
        return CorrectionlibCovScaler(path, corrections)

    if spec.endswith(('.json', '.json.gz')):
        with (gzip.open(spec, 'rt') if spec.endswith('.gz') else open(spec)) as fin:
            payload = json.load(fin)
        if 'schema_version' in payload:
            corrections = {key: 'sigma_%s_scale' % key for key in COV_PARAM_NAMES
                           if 'sigma_%s_scale' % key in
                           [icorr['name'] for icorr in payload.get('corrections', [])]}
            if not corrections:
                raise ValueError('%s is a correctionlib file but has no '
                                 'sigma_<param>_scale correction; pass the mapping '
                                 'explicitly as file.json:dxy=<name>' % spec)
            return CorrectionlibCovScaler(spec, corrections)
        return BinnedCovScaler(payload['pt_edges'], payload['abs_eta_edges'],
                               payload['scales'])

    kwargs = {}
    for token in spec.split(','):
        if not token:
            continue
        key, _, val = token.partition('=')
        kwargs[key.strip()] = float(val)
    return ConstantCovScaler(**kwargs)

def is_pos_def(x):
    '''
    https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    eigvalsh: the covariance matrix is symmetric, so use the symmetric
    eigensolver (faster and guaranteed real eigenvalues).
    '''
    return np.all(np.linalg.eigvalsh(np.nan_to_num(x)) > 0)

def fix_track(trk, delta=1e-9):
    '''
    https://github.com/CMSKStarMuMu/miniB0KstarMuMu/blob/master/miniKstarMuMu/plugins/miniKstarMuMu.cc#L1611-L1678
    '''
    
    cov = convert_cov(trk.covariance())
    
    if is_pos_def(cov): 
        return trk
    
    #if int(np.__version__.split('.')[1])<17:
    #    new_cov = np.nan_to_num(cov) # missing keyword, check docs
    #else:
    #    new_cov = np.nan_to_num(cov, posinf=0., neginf=0.)    

    new_cov = np.nan_to_num(cov)    

    min_eigenvalue = np.nan_to_num(min(np.linalg.eigvals(new_cov)))
    for i in range(new_cov.shape[0]):
        new_cov[i,i] = new_cov[i,i] - min_eigenvalue + delta

    # same rebuild-the-track-around-a-new-covariance trick as scale_track_cov;
    # the SMatrix incantation lives in smatrix55_from_cov now
    new_trk = track_with_cov(trk, new_cov)

    if not is_pos_def(new_cov):
        fix_track(new_trk, delta)
    
    return new_trk

##########################################################################################
##########################################################################################

#@np.njit
def compute_mass(p1, p2, m1, m2, p1p2):
    mass_squared = m1**2 + m2**2 + 2*np.sqrt(m1**2 + p1**2)*np.sqrt(m2**2 + p2**2) - 2*p1p2
    return np.sqrt(mass_squared) 

##########################################################################################
##########################################################################################

def compute_IP3D(pv, sv, direction):
    '''3D distance of the point `pv` from the line through `sv` along `direction`:
           d = |(sv - pv) x n| / |n|
       NB: use .R() (3D magnitude). .rho()/.Rho() is the CYLINDRICAL radius
       sqrt(x^2+y^2) and would return a mixed transverse projection instead of
       the point-line distance.'''
    pv_to_sv_vector = ROOT.Math.DisplacementVector3D('ROOT::Math::Cartesian3D<double>,ROOT::Math::DefaultCoordinateSystemTag')( 
        sv.x()-pv.x(),
        sv.y()-pv.y(),
        sv.z()-pv.z() 
    )
    
    ip3d = abs(pv_to_sv_vector.Cross(direction).R() / direction.R())
    return ip3d


##########################################################################################
##########################################################################################

# https://stackoverflow.com/questions/36695256/python-asyncio-how-to-mock-aiter-method
class AsyncIter:    
    def __init__(self, items):    
        self.items = items    

    async def __aiter__(self):    
        for item in self.items:    
            yield item    

from particle import Particle

##########################################################################################
##########################################################################################

def is_b_hadron(pdgid: int) -> bool:
    """
    Returns True if the PDG ID corresponds to a hadron containing a b-quark.
    Works for both mesons and baryons.
    """
    try:
        p = Particle.from_pdgid(abs(pdgid))
        # quarks is e.g. ('b', 'u', 'd') or ('b', 'u') etc.
        return 'b' in p.quarks
    except Exception:
        return False
        
