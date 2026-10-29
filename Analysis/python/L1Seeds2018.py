'''
L1 seeds of the 2018 parking paths: decision and prescale per event.

The same logic as the dimuon ntuplizer (test/mm/inspector_mm_analysis.py), put
in a module so the R(Ds) ntuplizer reads the L1 information exactly the way the
trigger tag-and-probe ntuples do. The dimuon inspector itself is not changed.

Inputs, all in Bmmm/Analysis/data (or $BMMM_DATADIR, as for the dimuon jobs):

  HLT_<path>.pickle           brilcalc L1 prescale tables, one per HLT path:
                              {L1 seed: {run: {lumi section: prescale}}}
  l1menus/goodRuns2013to2022ByYear.json
                              run -> L1 menu name
  l1menus/L1Menu_*.pickle     L1 menu: {L1 seed: algorithm bit}

Per event and per L1 seed of the union over the requested HLT paths:

  <L1 seed>       final decision, from gtStage2Digis (BXVector<GlobalAlgBlk>),
                  bit looked up in the menu of the run; 0 if the seed is not in
                  that menu
  <L1 seed>_ps    the prescale of that lumi section from the brilcalc table;
                  the last lumi section of the run beyond the table; 0 if the
                  run is not in the table. MC: 1.

The ten parking paths share the same ten seeds (L1_SingleMu6er1p5 ...
L1_SingleMu25), so their tables are merged seed by seed and run by run (union
of the lumi sections). The dimuon inspector instead lets the last table loaded
win for each seed, which drops the runs that table does not cover. For CRAB the
merged table is written once to MERGED_NAME (a few MB, gzipped) by
write_merged(), instead of shipping the ten 3-17 MB pickles.

In MC the menu is L1Menu_Collisions2018_v1_0_0 (as in the dimuon ntuples).
A run outside the period the menu map covers gets an empty menu (decisions 0)
and a warning, never 2018 bits looked up in the wrong menu.
'''

import os
import gzip
import json
import pickle

import numpy as np

MC_MENU = 'L1Menu_Collisions2018_v1_0_0'
MENUS_2018 = (
    'L1Menu_Collisions2018_v2_1_0',
    'L1Menu_Collisions2018_v2_0_0',
    'L1Menu_Collisions2018_v1_0_0',
    'L1Menu_Collisions2018_0_0_1',
)


MERGED_NAME = 'l1_seed_prescales_bparking2018.pickle.gz'


def load_tables(hlt_paths, datadir):
    '''{seed: {run: {ls: prescale}}}: the merged file when present, else the
    per-path brilcalc pickles merged here.'''
    merged = os.path.join(datadir, MERGED_NAME)
    if os.path.isfile(merged):
        with gzip.open(merged, 'rb') as fin:
            return pickle.load(fin), merged
    out, missing = {}, []
    for path in hlt_paths:
        fname = os.path.join(datadir, '%s.pickle' % path)
        if not os.path.isfile(fname):
            missing.append(path)
            continue
        with open(fname, 'rb') as fin:
            table = pickle.load(fin)
        for seed, runs in table.items():
            dst = out.setdefault(seed, {})
            for run, lss in runs.items():
                dst.setdefault(run, {}).update(lss)
    if missing:
        raise IOError('L1Seeds: no L1 prescale table for %s in %s (and no %s). '
                      'Pass --no-l1 to run without the L1 branches.'
                      % (missing, datadir, MERGED_NAME))
    return out, datadir


def write_merged(hlt_paths, datadir, out_path):
    '''Write the merged table (used by the CRAB submitter).'''
    tables, _ = load_tables(hlt_paths, datadir)
    with gzip.open(out_path, 'wb') as fout:
        pickle.dump(tables, fout, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def data_dir():
    '''$BMMM_DATADIR when set (CRAB jobs ship the inputs in the job
    directory), else $CMSSW_BASE/src/Bmmm/Analysis/data.'''
    return os.environ.get('BMMM_DATADIR', '') or os.path.join(
        os.environ.get('CMSSW_BASE', ''), 'src', 'Bmmm', 'Analysis', 'data')


class L1Seeds(object):

    def __init__(self, hlt_paths, datadir=None, verbose=True):
        datadir = datadir or data_dir()
        self.prescales, source = load_tables(hlt_paths, datadir)

        self.seeds = sorted(self.prescales)
        # max lumi section per (seed, run), computed once
        self.max_ls = dict((l1, dict((run, max(ls)) for run, ls in runs.items()))
                           for l1, runs in self.prescales.items())

        with open(os.path.join(datadir, 'l1menus', 'goodRuns2013to2022ByYear.json')) as fin:
            good = json.load(fin)
        self.run_menu = dict((r['run_number'], r['l1_menu']) for r in good['2018'])
        self.run_min, self.run_max = min(self.run_menu), max(self.run_menu)

        self.menus = {}
        for name in MENUS_2018:
            with open(os.path.join(datadir, 'l1menus', '%s.pickle' % name), 'rb') as fin:
                self.menus[name] = pickle.load(fin)

        self.warned = set()
        if verbose:
            print('#### L1 seeds: %d seeds from %d HLT paths (%s)'
                  % (len(self.seeds), len(hlt_paths), source))

    def branch_names(self):
        out = []
        for l1 in self.seeds:
            out += [l1, l1 + '_ps']
        return out

    def _warn(self, run, msg):
        if run not in self.warned:
            print('WARNING: %s' % msg)
            self.warned.add(run)

    def menu(self, run, mc):
        if mc:
            return self.menus[MC_MENU]
        if not (self.run_min <= run <= self.run_max):
            self._warn(run, 'RUN %d is outside the range the L1 menu map covers '
                            '(%d-%d): L1 decisions left at 0.' % (run, self.run_min, self.run_max))
            return {}
        name = self.run_menu.get(run)
        if name is None:
            closest = min(self.run_menu, key=lambda x: abs(x - run))
            self._warn(run, 'RUN %d not in the run -> menu map, using the menu of '
                            'run %d.' % (run, closest))
            name = self.run_menu[closest]
        return self.menus.get(name, {})

    def fill(self, event, glb_alg, mc, tofill):
        '''Write <seed> and <seed>_ps for every seed into tofill.'''
        aux  = event.eventAuxiliary()
        run  = aux.run()
        ls   = aux.luminosityBlock()
        menu = self.menu(run, mc)

        decisions = None
        if glb_alg is not None:
            try:
                decisions = glb_alg.at(0, 0)
            except Exception:
                decisions = None

        for l1 in self.seeds:
            if mc:
                tofill[l1 + '_ps'] = 1.
            else:
                runs = self.prescales[l1]
                if run in runs:
                    table = runs[run]
                    if ls in table:
                        tofill[l1 + '_ps'] = table[ls]
                    elif ls > self.max_ls[l1][run]:
                        tofill[l1 + '_ps'] = table[self.max_ls[l1][run]]
                    else:
                        tofill[l1 + '_ps'] = np.nan
                else:
                    tofill[l1 + '_ps'] = 0.

            if decisions is not None and l1 in menu:
                tofill[l1] = float(decisions.getAlgoDecisionFinal(menu[l1]))
            elif decisions is None:
                tofill[l1] = np.nan
            else:
                tofill[l1] = 0.
