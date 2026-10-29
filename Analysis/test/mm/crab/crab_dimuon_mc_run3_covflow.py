'''
CRAB3 submitter: Run 3 dimuon MC ntuples with covflow applied BEFORE the
dimuon vertex fit -- the closure of the covariance correction.

Everything is crab_dimuon_mc_run3.py (same datasets, sandbox, pileup weights,
output layout; its create_config is reused, not copied). What is added:

  1. One covflow production (the per-epoch, per-muon flows of covflow's
     configs/run3_epochs.py) is shipped in the sandbox, epoch by epoch, and the
     inspector runs with --covflow=covflow_<epoch>/@mu@: the leading muon of a
     pair is corrected with the mu1 flow, the trailing one with the mu2 flow,
     and the vertex fit, vtx_prob, lxy_sig, dxy_e, ... all use the corrected
     covariance. vtx_prob_raw / lxy_sig_raw / mu<i>_cov_* keep the uncorrected
     values of the same candidate.

  2. One task per (dataset, epoch). The MC campaign fixes the epochs:
         Summer22     -> 2022_preEE        Summer23     -> 2023_preBPix
         Summer22EE   -> 2022_postEE       Summer23BPix -> 2023_postBPix
         Summer24     -> 2024, 2025, 2026  (one MC sample, three trainings)
     so the Summer24 samples are processed three times, once per epoch's flows.
     The epoch is in the request name and in the output tag.

  3. Preflight at submission: every flow pair is loaded with the very class the
     jobs use (Bmmm CovFlowCorrector), under the worker-node conditions
     (PYTHONNOUSERSITE=1, pylibs/ first): torch from CMSSW, zuko and covflow
     from pylibs/, manifest, role, strict state_dict load. A failure here would
     have failed every job.

  4. --emulate: the full MC emulation instead of covflow alone (inspector
     --emulate, Bmmm CovflowEmulation.MCTrackEmulator). Per event: one run
     range drawn with the luminosity shares, pixel hits killed from the kill
     maps (P(kill) = 1 - eff_data/eff_MC; weight where data beats MC), route A
     for the killed muons (C -> C', delta_1 ~ N(0, clip(C' - C))), covflow at
     each muon's final context (C*), delta_2 ~ N(0, clip(C* - B)) on every muon
     with B = C or C'. The fit uses the smeared parameters and B + clip(C* - B);
     the *_raw branches keep the untouched MC. The kill maps of the epoch are
     shipped next to the flows (covflow_<epoch>/killmaps_<epoch>.npz/.json) and
     preflighted with the job's own loader: MC helix origin, epoch, surfaces.
     --no-killmaps emulates covflow + delta_2 only.

Before running:
    cmsenv
    cd $CMSSW_BASE/src/Bmmm/Analysis/test/mm/crab
    ./make_pylibs.sh --covflow /work/manzoni/correct_track_covariance/covflow
    source /cvmfs/cms.cern.ch/crab3/crab.sh
    voms-proxy-init -rfc -voms cms -valid 192:00
    python3 crab_dimuon_mc_run3_covflow.py --dry-run     # preflight + configs only
    python3 crab_dimuon_mc_run3_covflow.py               # submit
    python3 crab_dimuon_mc_run3_covflow.py --emulate     # full emulation, own tag

Submit one costing task first (--max-files 5 --only Summer24 --epochs 2026):
covflow adds two small torch calls per event and torch to the job memory.
'''

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from multiprocessing import Process

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crab_dimuon_mc_run3 as base     # noqa: E402  (datasets, create_config, submit)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--runs', default='/work/manzoni/correct_track_covariance/covflow-runs/'
                                  'run3_epochs_05oct26_trgmatch_e1200',
                help='covflow production: <runs>/<epoch>/<mu>/task_0 (default: %(default)s)')
ap.add_argument('--task', default='task_0', help='run subdirectory (default: %(default)s)')
ap.add_argument('--epochs', nargs='+', default=None,
                help='only these epochs (default: all the campaign maps to)')
ap.add_argument('--only', nargs='+', default=None, metavar='CAMPAIGN',
                help='only these MC campaigns, e.g. Summer24')
ap.add_argument('--samples', nargs='+', default=['HbToJPsiMuMu', 'BcToJPsiMuMu'],
                help='primary-dataset prefixes to run on (default: Hb and Bc, the '
                     'latter being covflow\'s independent validation sample)')
ap.add_argument('--max-files', type=int, default=None, help='costing run: input files per task')
ap.add_argument('--tag', default=None,
                help='appended to the output directory and work area (default: covflow_11oct2026, '
                     'or covflowemu_12oct2026 with --emulate)')
ap.add_argument('--emulate', action='store_true',
                help='full MC emulation: hit kill, route A + delta_1, covflow, delta_2 '
                     '(see 4. above)')
ap.add_argument('--killmaps', default='/work/manzoni/correct_track_covariance/covflow/'
                                      'hitemu_allprobes/{epoch}/killmaps/killmaps_{epoch}',
                help='with --emulate: kill-map stem, {epoch} substituted '
                     '(.npz + .json; default: %(default)s)')
ap.add_argument('--no-killmaps', action='store_true',
                help='with --emulate: no hit kill (covflow + delta_2 only)')
ap.add_argument('--emu-seed', type=int, default=12345,
                help='with --emulate: base seed, each event uses (seed, run, lumi, event) '
                     '(default: %(default)s)')
ap.add_argument('--no-delta1', action='store_true', help='with --emulate: no delta_1 (checks)')
ap.add_argument('--no-delta2', action='store_true', help='with --emulate: no delta_2 (checks)')
ap.add_argument('--track-scales', default=None,
                help='with --emulate: JSON from fit_track_scales.py (schema trackscales/1), '
                     'shipped with the job: delta_3, the data\'s own understatement of its '
                     'track errors')
ap.add_argument('--dry-run', action='store_true', help='preflight and print, submit nothing')
args = ap.parse_args()
if not args.emulate and (args.no_killmaps or args.no_delta1 or args.no_delta2 or args.track_scales):
    ap.error('--no-killmaps / --no-delta1 / --no-delta2 / --track-scales need --emulate')
if args.track_scales:
    _ts = json.load(open(args.track_scales))
    if _ts.get('schema') != 'trackscales/1':
        ap.error('--track-scales %s: schema %r, expected trackscales/1'
                 % (args.track_scales, _ts.get('schema')))
    if not re.match(r'^[A-Za-z0-9._-]+$', os.path.basename(args.track_scales)):
        ap.error('--track-scales: the file name may only contain letters, digits and ._- '
                 '(it travels as a CRAB scriptArgs value)')
if args.tag is None:
    args.tag = 'covflowemu_12oct2026' if args.emulate else 'covflow_11oct2026'
use_km = args.emulate and not args.no_killmaps

CAMPAIGN_EPOCHS = {
    'Summer22'    : ['2022_preEE'],
    'Summer22EE'  : ['2022_postEE'],
    'Summer23'    : ['2023_preBPix'],
    'Summer23BPix': ['2023_postBPix'],
    'Summer24'    : ['2024', '2025', '2026'],
}
FLOW_FILES = ('covflow.json', 'flow_mc.pt', 'flow_data.pt', 'scalers.json')
STAGE = os.path.join(HERE, 'covflow_stage')

base.work_area = 'crab_dimuon_mc_run3_pu_%s' % args.tag
base.out_dir   = 'dimuon_ntuples_mc_run3_pu_%s' % args.tag
base.max_files = args.max_files


def die(msg):
    raise SystemExit('[PREFLIGHT] ' + msg)


# ----------------------------------------------------------------------------
# what to run: (dataset, campaign, epoch)
# ----------------------------------------------------------------------------
tasks = []
for ds in base.productions:
    if not any(ds.split('/')[1].startswith(s) for s in args.samples):
        continue
    camp = base.mc_campaign(ds)
    if args.only and camp not in args.only:
        continue
    if camp not in CAMPAIGN_EPOCHS:
        die('no covflow epoch for MC campaign %s (%s)' % (camp, ds))
    for ep in CAMPAIGN_EPOCHS[camp]:
        if args.epochs and ep not in args.epochs:
            continue
        tasks.append((ds, camp, ep))
if not tasks:
    die('nothing to submit with these options')
epochs = sorted({ep for _, _, ep in tasks})

# ----------------------------------------------------------------------------
# stage the flows: covflow_stage/covflow_<epoch>/<mu>/{4 files}, shipped whole,
# plus covflow_<epoch>/killmaps_<epoch>.{npz,json} with --emulate
# ----------------------------------------------------------------------------
pylibs = os.path.join(HERE, 'pylibs')
for need in ('covflow', 'zuko'):
    if not os.path.isdir(os.path.join(pylibs, need)):
        die('pylibs/%s missing: build pylibs with  ./make_pylibs.sh --covflow <covflow repo>' % need)

staged = {}
for ep in epochs:
    dst = os.path.join(STAGE, 'covflow_%s' % ep)
    # restaged from scratch: no kill maps left over from an earlier --emulate
    # submission travel with a covflow-only one
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    for mu in ('mu1', 'mu2'):
        src = os.path.join(args.runs, ep, mu, args.task)
        miss = [f for f in FLOW_FILES if not os.path.isfile(os.path.join(src, f))]
        if miss:
            die('%s lacks %s' % (src, ', '.join(miss)))
        os.makedirs(os.path.join(dst, mu), exist_ok=True)
        for f in FLOW_FILES:
            shutil.copy2(os.path.join(src, f), os.path.join(dst, mu, f))
    if use_km:
        stem = args.killmaps.format(epoch=ep)
        stem = stem[:-4] if stem.endswith('.npz') else stem
        for ext in ('.npz', '.json'):
            if not os.path.isfile(stem + ext):
                die('kill maps %s%s not found (--killmaps, or --no-killmaps)' % (stem, ext))
            shutil.copy2(stem + ext, os.path.join(dst, 'killmaps_%s%s' % (ep, ext)))
    staged[ep] = dst

# ----------------------------------------------------------------------------
# preflight: load every flow pair as a worker node would
# ----------------------------------------------------------------------------
cmssw_python = os.path.join(os.environ.get('CMSSW_BASE', ''), 'python')
env = dict(os.environ, PYTHONNOUSERSITE='1',
           PYTHONPATH=os.pathsep.join([pylibs, cmssw_python, os.environ.get('PYTHONPATH', '')]))
check = r'''
import sys, json
import torch, zuko
from Bmmm.Analysis.utils import make_cov_corrector
use_km = sys.argv[1] == '1'
out = {'torch': torch.__version__, 'zuko': zuko.__version__, 'killmaps': {}}
for d in sys.argv[2:]:
    corr = {}
    for mu in ('mu1', 'mu2'):
        c = corr[mu] = make_cov_corrector('%s/%s' % (d, mu), attr_suffix='_' + mu, role=mu)
        out['%s/%s' % (d, mu)] = [c.schema, c.role, c.context_names, len(c.idx)]
    if use_km:
        # the job's own constructor: hitemu from pylibs/covflow, KillMaps.load,
        # the MC helix-origin check
        from Bmmm.Analysis.CovflowEmulation import MCTrackEmulator
        ep = d.rstrip('/').split('covflow_')[-1]
        e = MCTrackEmulator(corr, killmaps='%s/killmaps_%s' % (d, ep), verbose=False)
        m = e.km.meta
        out['killmaps'][d] = [m.get('epoch'), e.H.meta_origin(m, 'mc'), list(e.km.surfaces),
                              len(e.km.ranges), bool(m.get('hit_masks')), e.H.__file__]
print(json.dumps(out))
'''
proc = subprocess.run([sys.executable, '-c', check, '1' if use_km else '0']
                      + [staged[ep] for ep in epochs],
                      env=env, capture_output=True, text=True)
if proc.returncode != 0:
    die('the flows%s do not load under worker-node conditions:\n%s\n%s\n'
        'If torch is the missing module, this CMSSW python has no torch and the jobs '
        'cannot run covflow on the grid.' % (' / kill maps' if use_km else '',
                                             proc.stdout[-2000:], proc.stderr[-3000:]))
report = json.loads(proc.stdout.strip().splitlines()[-1])
zuko_shipped = report.pop('zuko')
print('[PREFLIGHT] torch %s, zuko %s (worker-node conditions)' % (report.pop('torch'), zuko_shipped))
for ep in epochs:
    for mu in ('mu1', 'mu2'):
        trained = json.load(open(os.path.join(staged[ep], mu, 'covflow.json'))).get('zuko_version')
        if trained and trained != 'unknown' and trained != zuko_shipped:
            print('[PREFLIGHT] note: %s/%s was trained with zuko %s, the jobs run %s (the '
                  'flows loaded strictly above; ZUKO_VERSION=%s ./make_pylibs.sh ... to match)'
                  % (ep, mu, trained, zuko_shipped, trained))
kmrep = report.pop('killmaps')
for k, v in sorted(report.items()):
    print('[PREFLIGHT] %-60s %s, %s, context %s, %d features' % (os.path.relpath(k, HERE), *v))
for ep in epochs:
    if not use_km:
        break
    km_ep, origin, surfaces, nranges, masks, hpath = kmrep[staged[ep]]
    if km_ep != ep:
        die('covflow_%s carries kill maps built for epoch %r (%s)'
            % (ep, km_ep, args.killmaps.format(epoch=ep)))
    print('[PREFLIGHT] killmaps_%-14s epoch %s, MC origin %s, %d range(s), surfaces %s, %s'
          % (ep, km_ep, origin, nranges, ' '.join(surfaces),
             'per-layer masks' if masks else 'L1/D1 only (no per-layer masks)'))
if use_km:
    print('[PREFLIGHT] hitemu loaded from %s' % os.path.relpath(hpath, HERE))
if args.emulate:
    print('[PREFLIGHT] emulation: %s, delta_1 %s, delta_2 %s, seed %d'
          % ('kill maps' if use_km else 'NO kill maps (covflow + delta_2)',
             'off' if args.no_delta1 else 'on', 'off' if args.no_delta2 else 'on', args.emu_seed))
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(STAGE) for f in fs)
print('[PREFLIGHT] staged flows%s: %.1f MB' % (' and kill maps' if use_km else '', size / 1e6))

# ----------------------------------------------------------------------------
# configs
# ----------------------------------------------------------------------------
cfgs = []
for ds, camp, ep in tasks:
    cfg = base.create_config(ds)
    sfx = ('_emu' if args.emulate else '_cf') + ep
    cfg.General.requestName = (cfg.General.requestName + sfx)[:100]
    cfg.Data.outputDatasetTag = '%s_%s%s' % (base.out_dir, camp, sfx)
    # the directory NAME only: the CRAB server refuses '/' and '@' in scriptArgs
    # values (400 "Incorrect 'scriptargs' parameter"); crab_script.sh adds /<mu>
    sa = ['COVFLOW=covflow_%s' % ep]
    if args.emulate:
        sa += ['EMULATE=1', 'EMUSEED=%d' % args.emu_seed]
        if use_km:
            sa += ['KILLMAPS=killmaps_%s' % ep]       # inside covflow_<ep>/
        if args.no_delta1:
            sa += ['NODELTA1=1']
        if args.no_delta2:
            sa += ['NODELTA2=1']
        if args.track_scales:
            sa += ['TRACKSCALES=%s' % os.path.basename(args.track_scales)]
    cfg.JobType.scriptArgs = list(cfg.JobType.scriptArgs) + sa
    cfg.JobType.inputFiles = list(cfg.JobType.inputFiles) + [staged[ep]] + \
        ([os.path.abspath(args.track_scales)] if args.track_scales else [])
    cfgs.append(cfg)

# what the CRAB server accepts in a scriptArgs value, checked here rather than
# discovered as a 400 at submission
import re as _re
bad = [a for c in cfgs for a in c.JobType.scriptArgs
       if not _re.match(r'^[A-Za-z0-9_]+=[A-Za-z0-9._:,-]*$', a)]
if bad:
    die('scriptArgs the CRAB server would refuse (only letters, digits and ._:,- in '
        'values): %s' % sorted(set(bad)))

for what, keys in (('requestName', [c.General.requestName for c in cfgs]),
                   ('output tag', ['%s/%s' % (ds.split('/')[1], c.Data.outputDatasetTag)
                                   for (ds, _, _), c in zip(tasks, cfgs)])):
    dup = [k for k, n in Counter(keys).items() if n > 1]
    if dup:
        die('duplicate %s: %s' % (what, dup))

for (ds, camp, ep), cfg in zip(tasks, cfgs):
    print('%-12s %-14s %s\n    -> %s  [%s]' % (camp, ep, ds, cfg.General.requestName,
                                              cfg.Data.outputDatasetTag))
    if args.dry_run:
        print('    scriptArgs: %s' % ' '.join(cfg.JobType.scriptArgs))
    if args.dry_run:
        continue
    p = Process(target=base.submit, args=(cfg,))
    p.start()
    p.join()
print('%s %d task(s)%s' % ('dry run:' if args.dry_run else 'sent', len(cfgs),
                           '' if args.dry_run else ' -- check above that none says "failed submitting"'))
