'''
Track momentum calibration (rds_momcalib / rmanzoni/momcal) applied at ntuple
production, MC only.

The correction itself is NOT reimplemented here: the momcal applier
(rds_trkcalib.py) and its correction file are loaded as they are, so the
numbers are exactly the ones the fitter and the cross-checks validated.

    --trk-calib <file.json>[:<variation>]

The applier module is looked for next to the JSON file first, then in
Bmmm/Analysis/data/trkcalib/ (or $BMMM_DATADIR/trkcalib/). Copy both from the
momcal repository:

    applier/python/rds_trkcalib.py
    applier/data/rds_trkcalib_2018BParking_v4.json

Track type. v4 has its own resolution terms for muons (per_bin_muon), selected
with factor(..., kind='muon'). A correction file with muon-specific terms and
an applier without the `kind` argument is a hard error: using it would give the
muon the hadron resolution without saying so.

Per track (README of momcal, section 6):
    key  = index in its collection + an offset per collection
           (packedPFCandidates: 0, lostTracks: 1 000 000, slimmedMuons: 2 000 000),
           the same physical track has the same key in every candidate
    z    = gauss(run, lumi, event, key)
    f    = factor(p, pt, eta, q, z, kind)        p'/p, direction unchanged
Both z and f are stored per track in the ntuple, so every variation can be
recomputed offline from the raw momenta.
'''

import os
import sys
import inspect
import importlib.util

import numpy as np

KEY_OFFSET = {'pf': 0, 'lost': 1000000, 'muon': 2000000}


def _datadir():
    return os.environ.get('BMMM_DATADIR', '') or os.path.join(
        os.environ.get('CMSSW_BASE', ''), 'src', 'Bmmm', 'Analysis', 'data')


def _load_applier(json_path):
    candidates = [os.path.join(os.path.dirname(os.path.abspath(json_path)), 'rds_trkcalib.py'),
                  os.path.join(_datadir(), 'trkcalib', 'rds_trkcalib.py')]
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location('rds_trkcalib', path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules['rds_trkcalib'] = mod
            spec.loader.exec_module(mod)
            return mod, path
    raise IOError('--trk-calib: rds_trkcalib.py not found in %s. Copy it from the '
                  'momcal repository (applier/python/).' % candidates)


class TrackMomentumCalib(object):

    def __init__(self, json_path, variation='nominal', verbose=True):
        if not os.path.isfile(json_path):
            raise IOError('--trk-calib: correction file %s not found' % json_path)
        self.mod, self.applier_path = _load_applier(json_path)
        self.cal = self.mod.TrackCalib(json_path)
        self.json_path = json_path
        self.variation = variation
        if variation not in self.cal.variation_names:
            raise ValueError('--trk-calib: unknown variation %r; known: %s'
                             % (variation, self.cal.variation_names))

        self.has_kind = 'kind' in inspect.signature(self.cal.factor).parameters
        has_muon_terms = 'per_bin_muon' in self.cal.cfg
        if has_muon_terms and not self.has_kind:
            raise RuntimeError('--trk-calib: %s has muon-specific terms (per_bin_muon) '
                               'but the applier %s has no `kind` argument. Use the '
                               'applier of the same momcal version as the file.'
                               % (json_path, self.applier_path))
        self.name = '%s %s' % (self.cal.cfg.get('name', '?'), self.cal.cfg.get('version', '?'))
        self.n_tracks = 0
        if verbose:
            print('#### track momentum calibration: %s (%s), variation %s, applier %s'
                  % (self.name, json_path, variation, self.applier_path))
            print('####   muon-specific terms: %s' % ('yes' if has_muon_terms else 'no'))

    @staticmethod
    def key(collection, index):
        return KEY_OFFSET[collection] + int(index)

    def z_and_f(self, run, lumi, event, key, pt, eta, charge, kind='hadron'):
        '''(z, f) for one track: f = p'/p.'''
        z = float(self.cal.gauss(run, lumi, event, key))
        p = pt * np.cosh(eta)
        if self.has_kind:
            f = self.cal.factor(p, pt, eta, charge, z, self.variation, kind=kind)
        else:
            f = self.cal.factor(p, pt, eta, charge, z, self.variation)
        self.n_tracks += 1
        return z, float(f)

    def summary(self):
        return '#### track momentum calibration %s: %d tracks corrected' % (self.name, self.n_tracks)


def make_trk_calib(spec, mc):
    '''--trk-calib spec -> TrackMomentumCalib or None.'''
    if not spec:
        return None
    if not mc:
        raise RuntimeError('--trk-calib without --mc: the momentum correction maps '
                           'MC onto data and must not be applied to data.')
    path, _, variation = spec.partition(':')
    return TrackMomentumCalib(path, variation or 'nominal')
