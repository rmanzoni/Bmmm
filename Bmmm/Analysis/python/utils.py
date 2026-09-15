from __future__ import print_function
import os
import re
import sys
import json
import gzip
import particle
import numpy as np
from array import array
from glob import glob
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

##########################################################################################
#####      INPUT FILE RESOLUTION
##########################################################################################
# --inputFiles -> the list of paths handed to FWLite Events(), shared by every
# channel so the same argument means the same thing everywhere.
#
# Three shapes of entry live in this repository and each needs different
# treatment, which is why a blanket rule does not work:
#
#   root://host//store/...   already a full URL           -> use as is
#   /store/...               a CMS LFN, needs a door      -> prepend the redirector
#   /pnfs/psi.ch/...         a mounted POSIX path         -> use as is
#
# The J/psi + charged-object inspector used to prepend the redirector to every
# line of a .txt list, which is right only for the middle case: it double-
# prefixed entries that were already URLs and sent the T3 skim lists
# (files_bc_skim / files_hb_skim, both /pnfs/...) through xrootd. The dimuon
# inspector prepended it to none, which is wrong for the first. Production
# submissions are unaffected either way -- the submitters read the lists
# themselves and pass comma-separated, already-resolved URLs -- so this is the
# interactive path, but it should still do the right thing.
_URL_SCHEME_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.-]*://')

# characters that make a local path a pattern rather than a name
_GLOB_CHARS = ('*', '?', '[')

def resolve_input_file(path, redirector=''):
    '''One entry -> the path to open, or None for a blank line.'''
    path = path.strip()
    if not path:
        return None
    if _URL_SCHEME_RE.match(path):
        return path                      # already addressed, leave it alone
    if path.startswith('/store/') and redirector:
        return redirector + path         # bare LFN: concatenated verbatim, so the
                                         # redirector's own trailing slashes are
                                         # preserved (root://host///store/... is
                                         # the form already used in this repo)
    return path                          # local or mounted, open directly

def resolve_input_files(input_files, redirector=''):
    '''--inputFiles -> the list of paths to hand to Events().

    An argument ending in .txt is read as a file list, anything else is split on
    commas. Every entry is then resolved on its own merits: a URL and an LFN are
    passed through resolve_input_file, and only a local path is globbed -- and
    only when it actually looks like a pattern, so that a name that simply does
    not exist reaches ROOT and produces a real error instead of vanishing.'''
    if input_files.endswith('.txt'):
        with open(input_files) as fin:
            entries = fin.read().splitlines()
    else:
        entries = input_files.split(',')

    resolved = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if _URL_SCHEME_RE.match(entry) or entry.startswith('/store/'):
            resolved.append(resolve_input_file(entry, redirector))
        elif any(c in entry for c in _GLOB_CHARS):
            hits = sorted(glob(entry))
            if not hits:
                raise IOError('no file matches the pattern %r' % entry)
            resolved.extend(hits)
        else:
            resolved.append(resolve_input_file(entry, redirector))

    resolved = [e for e in resolved if e is not None]
    if not resolved:
        raise IOError('--inputFiles=%r resolved to no files at all' % input_files)

    # A bare LFN with no redirector cannot be opened by FWLite: say so here,
    # where the argument is still visible, rather than letting Events() report
    # a missing file with no hint as to why.
    bare = [e for e in resolved if e.startswith('/store/')]
    if bare:
        raise IOError(
            '%d input file(s) are bare LFNs and no --redirector was given, e.g.\n'
            '    %s\n'
            'pass --redirector=root://cms-xrd-global.cern.ch/ (or the full URL, '
            'or a site-local /pnfs path)' % (len(bare), bare[0]))

    return resolved

##########################################################################################
##########################################################################################

_HLT_VER_RE = re.compile(r'(_part\d+|_v\d+)+$')
def drop_hlt_version(s):
    # str() first: iterating the std::vector<std::string> behind
    # TriggerNames::triggerNames() hands back cppyy std::string proxies rather
    # than python strings on the ROOT shipped with recent CMSSW releases, and
    # re refuses those. Harmless when it is already a str.
    return _HLT_VER_RE.sub('', str(s))

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
#####      VERTEX COVARIANCE MATRIX
##########################################################################################
# The 3x3 position covariance of a fitted vertex, 6 independent elements taken
# row-major over the upper triangle (same convention as the track block above).
#
# Why persist it, given the tracks are already persisted: the vertex covariance
# is DERIVED from the covariances of the tracks that were fitted --
# C_V = (sum_i A_i^T G_i A_i)^-1 -- so once the input tracks carry the right
# covariance and the vertex is refitted, the vertex covariance follows by
# construction and must NOT be corrected a second time. It is persisted to
# VALIDATE that, not to correct it: if the data/MC agreement of the vertex
# covariance does not close after the track-level correction, the mismodelling
# is not (only) in the per-track covariance.
#
# The primary vertex is the exception -- see the pv_cov_* branches.
VTX_COV_INDEX_PAIRS = [(i, j) for i in range(3) for j in range(i, 3)]
VTX_COV_ELEMENT_NAMES = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']

def vertex_error_matrix(vtx):
    '''The 3x3 position error matrix of a vertex, for every flavour in use in
    this package -- they disagree on both the method name and the return type:

      reco::Vertex     .error()         -> math::Error<3> (an SMatrix), used as is
      KinematicVertex  .error()         -> GlobalError, whose .matrix() is the SMatrix
      VertexState      .error()         -> GlobalError, ditto
      TransientVertex  .positionError() -> GlobalError, ditto   (KVFitter output,
                                           i.e. the dimuon channel)

    Both the accessor and the unwrapping are chosen from the objects themselves
    rather than from the vertex type, which is what makes one helper serve all
    of them.'''
    err = vtx.error() if hasattr(vtx, 'error') else vtx.positionError()
    return err.matrix() if hasattr(err, 'matrix') else err

def vertex_cov_element(vtx, i, j):
    '''cov(i,j) of a vertex position.'''
    return vertex_error_matrix(vtx)(i, j)

def vertex_covariance(vtx):
    '''Full 3x3 position covariance as a numpy array.'''
    mat = vertex_error_matrix(vtx)
    return np.array([[mat(i, j) for j in range(3)] for i in range(3)])

def vertex_cov_upper_triangle(vtx):
    '''The 6 independent elements, in VTX_COV_INDEX_PAIRS order.'''
    cov = vertex_covariance(vtx)
    return [cov[i][j] for i, j in VTX_COV_INDEX_PAIRS]

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

##########################################################################################
#####      TRACK COVARIANCE CORRECTION: full matrix (covflow)
##########################################################################################
# A CovScaler can only stretch the diagonal: cov -> D cov D, every correlation
# frozen. covflow corrects the WHOLE matrix -- all 5 scales and all 10
# correlations at once -- by transporting the 15 unconstrained features
#
#     y[0:5]  = log sigma_i
#     y[5:15] = atanh(canonical partial correlations)
#
# through  y_corr = f_data^-1( f_mc(y | c) | c ),  f_* being the two conditional
# normalising flows trained by the covflow package. The feature map is a
# bijection R^15 -> {PD 5x5}, so no output can be an invalid covariance.
#
# The two objects below are DELIBERATELY separate from CovScaler rather than a
# subclass of it: a scaler answers "by how much", a corrector answers "what
# matrix", and collapsing the two contracts would let a covflow run silently
# write meaningless <obj>_cov_scale_* branches. --cov-scale and --covflow are
# mutually exclusive for the same reason.
#
# Everything downstream is unchanged: the corrected matrix goes through the same
# track_with_cov() rebuild as scale_track_cov(), i.e. it reaches the vertex fits,
# the IP3D grid and the jet-track distances via JpsiChargedCandidate.fit_track.

# Per-track context variables the flow may be conditioned on. The NAME is what
# goes in covflow.json; the value must be computed from the very track whose
# covariance is being corrected -- for a muon that is bestTrack(), which is also
# where the covariance itself comes from. Do not switch to innerTrack() here
# without switching the training branches too.
#
# The hit-content getters are spelled EXACTLY as the <obj>_n_pix_*_hit branches
# in CommonBranches, so the context the flow sees at ntuplization time is the
# same quantity it was trained on. If one of the two moves, the other must.
COVFLOW_CONTEXT_GETTERS = {
    'pt'            : lambda obj, trk, ev : trk.pt(),
    'log_pt'        : lambda obj, trk, ev : np.log(max(trk.pt(), 1e-6)),
    'eta'           : lambda obj, trk, ev : trk.eta(),
    'abs_eta'       : lambda obj, trk, ev : abs(trk.eta()),
    'phi'           : lambda obj, trk, ev : trk.phi(),
    'n_pix_hit'     : lambda obj, trk, ev : trk.hitPattern().numberOfValidPixelHits(),
    'n_pix_b_hit'   : lambda obj, trk, ev : trk.hitPattern().numberOfValidPixelBarrelHits(),
    'n_pix_e_hit'   : lambda obj, trk, ev : trk.hitPattern().numberOfValidPixelEndcapHits(),
    'n_pix_layer'   : lambda obj, trk, ev : trk.hitPattern().pixelLayersWithMeasurement(),
    'n_valid_hit'   : lambda obj, trk, ev : trk.numberOfValidHits(),
    'n_trk_layer'   : lambda obj, trk, ev : trk.hitPattern().trackerLayersWithMeasurement(),
    'chi2_norm'     : lambda obj, trk, ev : trk.normalizedChi2(),
    # EVENT-level. Spelled exactly as the matching branch in CommonBranches --
    # npv must mean len(event.vtx) here as it does everywhere else, or the flow
    # is conditioned on a different quantity at application time than it was
    # trained on. These need the event handed in: see prime().
    'npv'           : lambda obj, trk, ev : len(ev.vtx),
}

# Which of the above cannot be evaluated from the track alone. A corrector
# conditioned on any of these refuses to run without an event rather than
# silently substituting anything.
COVFLOW_EVENT_CONTEXT = set(['npv'])

# The covariance is only defined in feature space if it is positive definite
# with strictly positive variances; covflow.features raises otherwise. Tracks
# that fail are left UNCORRECTED and counted, never silently patched: a
# non-PD input covariance is a reconstruction problem, not a covflow one.
COV_NAN_5X5 = np.full((5, 5), np.nan)


class CovCorrector(object):
    '''Full 5x5 covariance replacement, applied to every track before it is
    handed to a fitter. Subclasses implement correct(obj, trk, cov) -> 5x5.

    `obj` is the pat::Muon / pat::PackedCandidate, `trk` its bestTrack() and
    `cov` the raw covariance as a numpy array (candidates memoize it as
    obj.cov). All three are passed because the conditioning variables live on
    different ones depending on the context set.

    Return None to leave the track untouched.
    '''
    def correct(self, obj, trk, cov):
        raise NotImplementedError

    def __call__(self, obj, trk, cov):
        return self.correct(obj, trk, cov)

    def prime(self, objs):
        '''Optional: pre-compute the correction for a whole collection in one
        go. Default is a no-op; CovFlowCorrector overrides it because a single
        batched torch call over an event's muons costs a fraction of one call
        per muon.'''
        return


class CovFlowCorrector(CovCorrector):
    '''The covflow morph, evaluated in-process with torch.

    Constructed from the directory a covflow training run wrote, which must
    contain flow_mc.pt, flow_data.pt, scalers.json and -- added by hand, once --
    covflow.json describing how those weights were trained:

        {"context"    : ["log_pt", "eta", "n_pix_hit"],
         "features"   : "all",
         "param"      : "logsigma_corr",
         "transforms" : 4,
         "hidden"     : [128, 128],
         "bins"       : 8,
         "seed"       : 0}

    covflow.json is MANDATORY and has no defaults for `context`: flows are saved
    as a bare state_dict, so nothing in the checkpoint records which variables it
    was conditioned on or in what order. Getting that order wrong produces a
    perfectly well-formed, completely wrong correction, and no downstream check
    would catch it. The hyperparameters are checked for free -- load_state_dict
    is strict, so a wrong `transforms`/`hidden`/`bins` raises rather than loads.

    Optional keys: "active_features" (indices of the full 15 to actually apply),
    "latent_bounds" {"lo": [...], "hi": [...]} from the training run, "device",
    "batch".
    '''

    REQUIRED_FILES = ('flow_mc.pt', 'flow_data.pt', 'scalers.json', 'covflow.json')

    def __init__(self, directory, overrides=None):
        # lazy: torch/zuko/covflow are not dependencies of the rest of Bmmm, and
        # a job running without --covflow must not pay for them
        import torch
        from covflow import correct as CF
        from covflow import data as CD
        from covflow import features as CFEAT
        from covflow import flows as CFL

        self.torch = torch
        self.CF, self.CFEAT, self.CFL = CF, CFEAT, CFL

        # the packed order must be the SAME 15 elements in the SAME order on
        # both sides, or every correction is a silent permutation
        if list(CFEAT.PACK_NAMES) != list(COV_ELEMENT_NAMES):
            raise RuntimeError('covflow PACK_NAMES and Bmmm COV_ELEMENT_NAMES '
                               'disagree:\n  covflow %s\n  Bmmm    %s'
                               % (list(CFEAT.PACK_NAMES), list(COV_ELEMENT_NAMES)))

        self.directory = directory
        missing = [f for f in self.REQUIRED_FILES
                   if not os.path.isfile(os.path.join(directory, f))]
        if missing:
            raise IOError('%s is not a covflow run directory: missing %s'
                          % (directory, ', '.join(missing)))

        with open(os.path.join(directory, 'covflow.json')) as fin:
            cfg = json.load(fin)
        cfg.update(overrides or {})

        self.context_names = list(cfg['context'])
        unknown = [n for n in self.context_names if n not in COVFLOW_CONTEXT_GETTERS]
        if unknown:
            raise KeyError('unknown covflow context variable(s) %s; known: %s'
                           % (unknown, sorted(COVFLOW_CONTEXT_GETTERS)))
        self.getters = [COVFLOW_CONTEXT_GETTERS[n] for n in self.context_names]
        self.needs_event = bool(set(self.context_names) & COVFLOW_EVENT_CONTEXT)
        self._event = None

        self.param    = cfg.get('param', 'logsigma_corr')
        self.device   = cfg.get('device', 'cpu')
        self.batch    = int(cfg.get('batch', 200000))
        self.idx      = CFEAT.subset_indices(cfg.get('features', 'all'))
        active        = cfg.get('active_features', None)
        self.active_features = None if active is None else list(active)

        self.scaler = CD.Standardiser.load(os.path.join(directory, 'scalers.json'))
        if len(self.scaler.ctx_mean) != len(self.context_names):
            raise ValueError(
                'covflow.json lists %d context variables %s but scalers.json was '
                'fitted on %d. The context set in covflow.json must be exactly '
                'the --context of the training run, in the same order.'
                % (len(self.context_names), self.context_names,
                   len(self.scaler.ctx_mean)))

        fcfg = CFL.FlowConfig(n_features = len(self.idx),
                              n_context  = len(self.context_names),
                              transforms = int(cfg.get('transforms', 4)),
                              hidden     = tuple(cfg.get('hidden', [128, 128])),
                              bins       = int(cfg.get('bins', 8)),
                              seed       = int(cfg.get('seed', 0)))
        self.flow_config = fcfg
        self.flow_mc   = CFL.load_flow(fcfg, os.path.join(directory, 'flow_mc.pt'))
        self.flow_data = CFL.load_flow(fcfg, os.path.join(directory, 'flow_data.pt'))
        torch.set_num_threads(1)   # one core per job; torch's pool fights ROOT

        bounds = cfg.get('latent_bounds', None)
        self.latent_lo = None if bounds is None else np.asarray(bounds['lo'], float)
        self.latent_hi = None if bounds is None else np.asarray(bounds['hi'], float)

        self.to_feat, self.to_mat = CFEAT.get_transforms(self.param)

        self.n_seen        = 0
        self.n_corrected   = 0
        self.n_not_pos_def = 0
        self.n_failed      = 0
        self.n_out_of_latent_range = 0

    # ---------------------------------------------------------------------
    def context(self, obj, trk, event):
        '''The conditioning vector for one track, in covflow.json order.'''
        return [float(getter(obj, trk, event)) for getter in self.getters]

    def _morph_batch(self, packed, ctx):
        '''packed (N,15), ctx (N,k) -> (packed_corr (N,15), zmax (N,)).

        This is covflow.correct.morph inlined so the latent z -- which morph
        discards -- can be kept for the per-track diagnostic. It MUST stay
        step-for-step identical to morph(); test_covflow_correction.py asserts
        exactly that against the covflow implementation.
        '''
        y_mc  = self.to_feat(self.CFEAT.packed_to_matrix(packed))
        ys_mc = self.scaler.x(y_mc)
        cs    = self.scaler.c(ctx)

        z      = self.CFL.data_to_latent(self.flow_mc, ys_mc[:, self.idx], cs,
                                         device=self.device, batch=self.batch)
        ys_sub = self.CFL.latent_to_data(self.flow_data, z, cs,
                                         device=self.device, batch=self.batch)

        ys_corr = ys_mc.copy()
        ys_corr[:, self.idx] = ys_sub
        y_corr = self.scaler.x_inv(ys_corr)

        if self.active_features is not None:
            mask = np.zeros(self.CFEAT.N_FEATURES, dtype=bool)
            mask[self.active_features] = True
            y_corr = np.where(mask[None, :], y_corr, y_mc)

        packed_corr = self.CFEAT.matrix_to_packed(self.to_mat(y_corr))

        # how far into the tail of the MC latent this track sits. The training
        # run's honest bound is the per-component min/max of the DATA latents;
        # pass it through covflow.json as latent_bounds to use it, otherwise
        # fall back to |z| and cut offline on the branch.
        if self.latent_lo is not None:
            out = (z < self.latent_lo[None, :]) | (z > self.latent_hi[None, :])
            self.n_out_of_latent_range += int(out.any(1).sum())
        zmax = np.max(np.abs(z), axis=1)
        return packed_corr, zmax

    # ---------------------------------------------------------------------
    def prime(self, objs, event=None):
        '''Correct a whole collection in ONE torch call and memoize the result
        on the objects. Call once per event, before candidate building: the same
        muon enters many candidates, and the correction must be identical in all
        of them.

        `event` is required when the context includes an event-level variable
        (npv); it is also cached, so the single-track fallback in fit_track --
        which has no event in hand -- uses the one from the current event.'''
        if event is not None:
            self._event = event
        if self.needs_event and self._event is None:
            raise RuntimeError(
                'the covflow context %s includes an event-level variable but no '
                'event was passed to prime(). Conditioning on a stale or absent '
                'npv is worse than not correcting at all.' % self.context_names)

        todo = []
        for obj in objs:
            if hasattr(obj, 'cov_corr'):
                continue
            trk = obj.bestTrack()
            if not hasattr(obj, 'cov'):
                obj.cov = convert_cov(trk.covariance())
                obj.is_cov_pos_def = is_pos_def(obj.cov)
            self.n_seen += 1
            if not (obj.is_cov_pos_def and np.all(np.diagonal(obj.cov) > 0)):
                self._mark_uncorrected(obj)
                self.n_not_pos_def += 1
                continue
            todo.append((obj, trk))

        if not todo:
            return

        packed = np.array([cov_upper_triangle(obj.cov) for obj, _ in todo])
        ctx    = np.array([self.context(obj, trk, self._event) for obj, trk in todo])

        try:
            packed_corr, zmax = self._morph_batch(packed, ctx)
        except Exception as exc:
            # one bad track must not kill a multi-hour job, but it must be
            # loud and it must be counted
            print('[covflow] WARNING: morph failed on a batch of %d track(s): '
                  '%s: %s -- left uncorrected' % (len(todo), type(exc).__name__, exc))
            for obj, _ in todo:
                self._mark_uncorrected(obj)
            self.n_failed += len(todo)
            return

        for k, (obj, _) in enumerate(todo):
            obj.cov_corr     = self.CFEAT.packed_to_matrix(packed_corr[k])
            obj.covflow_ok   = True
            obj.covflow_zmax = float(zmax[k])
            self.n_corrected += 1

    @staticmethod
    def _mark_uncorrected(obj):
        obj.cov_corr     = COV_NAN_5X5
        obj.covflow_ok   = False
        obj.covflow_zmax = np.nan

    # ---------------------------------------------------------------------
    def correct(self, obj, trk, cov):
        '''Single-track entry point, for anything prime() did not cover (the
        J/psi + track channel's bachelor, say). Returns None when the track
        cannot be corrected, which fit_track reads as "use the raw track".'''
        if not hasattr(obj, 'cov_corr'):
            self.prime([obj])
        return obj.cov_corr if obj.covflow_ok else None

    def summary(self):
        return ('[covflow] %d track(s) seen, %d corrected, %d skipped (not '
                'positive definite), %d skipped (morph failure), %d outside the '
                'training latent range'
                % (self.n_seen, self.n_corrected, self.n_not_pos_def,
                   self.n_failed, self.n_out_of_latent_range))


def make_cov_corrector(spec):
    '''Build a CovCorrector from a command-line string (see --covflow):

        ''                          -> None, no correction (the default)
        '/path/to/covflow_run1'     -> CovFlowCorrector on that run directory
        '/path/run1:device=cuda,batch=4096'
                                    -> same, with covflow.json keys overridden
                                       (JSON values, so lists and strings work:
                                       features=[0,1,2], param="log_cholesky")
    '''
    if spec is None or not spec.strip():
        return None
    spec = spec.strip()

    overrides = {}
    if ':' in spec and not os.path.isdir(spec):
        path, _, mapping = spec.partition(':')
        for token in mapping.split(','):
            if not token:
                continue
            key, _, val = token.partition('=')
            try:
                overrides[key.strip()] = json.loads(val)
            except ValueError:
                overrides[key.strip()] = val
    else:
        path = spec

    if not os.path.isdir(path):
        raise IOError('--covflow expects the directory a covflow training run '
                      'wrote (flow_mc.pt, flow_data.pt, scalers.json, '
                      'covflow.json); %r is not a directory' % path)
    return CovFlowCorrector(path, overrides)


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

# largest diagonal shift fix_track will escalate to before giving up:
# enough to clear float granularity for any physical track covariance
_FIX_TRACK_MAX_DELTA = 1e-3

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

    # Check what the track actually STORES, not the double we just built.
    # reco::TrackBase keeps the covariance as float, and delta is the same size
    # as float granularity once a diagonal element reaches ~1e-1 -- a badly
    # measured track, which is exactly the case this function exists for -- so
    # the shift can be rounded away on the way in. Retry with a bigger delta
    # until it survives.
    #
    # The previous check tested new_cov, whose smallest eigenvalue is delta by
    # construction, so it never fired; and it discarded the result of the
    # recursive call, so it would have done nothing if it had.
    if is_pos_def(convert_cov(new_trk.covariance())):
        return new_trk

    if delta >= _FIX_TRACK_MAX_DELTA:
        print('WARNING: fix_track could not make the stored covariance '
              'positive definite up to delta %g; returning the track as is '
              '(its cov_pos_def flag stays False).' % delta)
        return new_trk

    return fix_track(new_trk, delta * 10.)

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
        
