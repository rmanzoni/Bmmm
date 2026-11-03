'''
Muon trigger scale factors of the 2018 B-parking paths (MC only), from the
tag-and-probe maps of data/bparking_trigger_sf (efficiency_sf_splot.py:
J/psi probes, Charmonium data vs Hb MC, sPlot-subtracted data,
SF = eff_data / eff_MC, Clopper-Pearson errors).

    --trig-sf <scale_factors.json>[:<map>]      default map pt_vs_dxy_sig

Per candidate muon, in the binning of the map:
    x = muon pT  (the RAW pT, as in the probe skim, which predates the
                  momentum calibration)
    y = |dxy / sigma_dxy| wrt the beamspot (mu_bs_dxy_sig; in MC with the
                  sigma_dxy morph, as the probe skim's dxy_sig_morph)
Values outside the map are taken from the nearest edge bin and flagged
(mu_trig_sf_inrange = 0): pT > 50 GeV, |dxy/sigma| > 100, and below the
first bin (|dxy/sigma| < 3, where none of the HLT_Mu*_IP* paths fires).
Branches: mu_trig_sf, mu_trig_sf_up, mu_trig_sf_dn (= sf +- the asymmetric
errors), mu_trig_sf_inrange. Everything can be recomputed offline from
mu_pt_raw and mu_bs_dxy_sig.
'''

import json
import numpy as np

NAN = float('nan')
SF_BRANCHES = ('trig_sf', 'trig_sf_up', 'trig_sf_dn', 'trig_sf_inrange')
# map name -> (x getter key, y getter key) in the filled candidate row
_AXES = {
    'pt_vs_dxy_sig': ('mu_pt_raw', 'mu_bs_dxy_sig'),
    'pt_vs_eta'    : ('mu_pt_raw', 'mu_eta'),
    'eta_vs_dxy_sig': ('mu_eta', 'mu_bs_dxy_sig'),
}


class TriggerScaleFactors(object):

    def __init__(self, path, name='pt_vs_dxy_sig'):
        with open(path) as fin:
            payload = json.load(fin)
        if name not in payload.get('2d', {}):
            raise KeyError('--trig-sf: map %r not in %s (2d maps: %s)'
                           % (name, path, sorted(payload.get('2d', {}))))
        if name not in _AXES:
            raise KeyError('--trig-sf: no variable mapping for map %r' % name)
        m = payload['2d'][name]
        self.path, self.name = path, name
        self.ex = np.asarray(m['bin_edges_x'], float)
        self.ey = np.asarray(m['bin_edges_y'], float)
        nan = lambda a: np.array([[NAN if v is None else v for v in row] for row in a], float)
        self.sf, self.lo, self.hi = nan(m['sf']), nan(m['err_lo']), nan(m['err_hi'])
        if self.sf.shape != (len(self.ex) - 1, len(self.ey) - 1):
            raise ValueError('--trig-sf: map %s has shape %s, edges %d x %d'
                             % (name, self.sf.shape, len(self.ex), len(self.ey)))
        self.keys = _AXES[name]
        self.n, self.n_out = 0, 0
        print('#### trigger scale factors: %s, map %s (%s x |%s|)' % (path, name, self.keys[0], self.keys[1]))

    @staticmethod
    def _bin(edges, v):
        i = int(np.searchsorted(edges, v, side='right')) - 1
        inside = 0 <= i < len(edges) - 1
        return min(max(i, 0), len(edges) - 2), inside

    def __call__(self, row):
        '''{trig_sf, trig_sf_up, trig_sf_dn, trig_sf_inrange} for a filled row.'''
        x, y = row.get(self.keys[0], NAN), row.get(self.keys[1], NAN)
        y = abs(y) if 'eta' in self.keys[1] or 'dxy' in self.keys[1] else y
        x = abs(x) if 'eta' in self.keys[0] else x
        if not (x == x and y == y):
            return dict.fromkeys(SF_BRANCHES, NAN)
        i, okx = self._bin(self.ex, x)
        j, oky = self._bin(self.ey, y)
        sf = self.sf[i, j]
        self.n += 1
        self.n_out += not (okx and oky)
        return {'trig_sf': sf, 'trig_sf_up': sf + self.hi[i, j], 'trig_sf_dn': sf - self.lo[i, j],
                'trig_sf_inrange': float(okx and oky)}

    def summary(self):
        return ('#### trigger scale factors %s: %d muons, %d outside the map (edge bin used)'
                % (self.name, self.n, self.n_out))


def make_trig_sf(spec, mc):
    if not spec:
        return None
    if not mc:
        raise RuntimeError('--trig-sf without --mc: the scale factors correct MC to data.')
    path, _, name = spec.partition(':')
    return TriggerScaleFactors(path, name or 'pt_vs_dxy_sig')
