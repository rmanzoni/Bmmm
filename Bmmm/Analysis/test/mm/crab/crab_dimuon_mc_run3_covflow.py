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

Before running:
    cmsenv
    cd $CMSSW_BASE/src/Bmmm/Analysis/test/mm/crab
    ./make_pylibs.sh --covflow /work/manzoni/correct_track_covariance/covflow
    source /cvmfs/cms.cern.ch/crab3/crab.sh
    voms-proxy-init -rfc -voms cms -valid 192:00
    python3 crab_dimuon_mc_run3_covflow.py --dry-run     # preflight + configs only
    python3 crab_dimuon_mc_run3_covflow.py               # submit

Submit one costing task first (--max-files 5 --only Summer24 --epochs 2026):
covflow adds two small torch calls per event and torch to the job memory.
'''

import argparse
import json
import os
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
ap.add_argument('--tag', default='covflow_11oct2026',
                help='appended to the output directory and work area (default: %(default)s)')
ap.add_argument('--dry-run', action='store_true', help='preflight and print, submit nothing')
args = ap.parse_args()

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
# stage the flows: covflow_stage/covflow_<epoch>/<mu>/{4 files}, shipped whole
# ----------------------------------------------------------------------------
pylibs = os.path.join(HERE, 'pylibs')
for need in ('covflow', 'zuko'):
    if not os.path.isdir(os.path.join(pylibs, need)):
        die('pylibs/%s missing: build pylibs with  ./make_pylibs.sh --covflow <covflow repo>' % need)

staged = {}
for ep in epochs:
    dst = os.path.join(STAGE, 'covflow_%s' % ep)
    for mu in ('mu1', 'mu2'):
        src = os.path.join(args.runs, ep, mu, args.task)
        miss = [f for f in FLOW_FILES if not os.path.isfile(os.path.join(src, f))]
        if miss:
            die('%s lacks %s' % (src, ', '.join(miss)))
        os.makedirs(os.path.join(dst, mu), exist_ok=True)
        for f in FLOW_FILES:
            shutil.copy2(os.path.join(src, f), os.path.join(dst, mu, f))
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
out = {'torch': torch.__version__, 'zuko': zuko.__version__}
for d in sys.argv[1:]:
    for mu in ('mu1', 'mu2'):
        c = make_cov_corrector('%s/%s' % (d, mu), attr_suffix='_' + mu, role=mu)
        out['%s/%s' % (d, mu)] = [c.schema, c.role, c.context_names, len(c.idx)]
print(json.dumps(out))
'''
proc = subprocess.run([sys.executable, '-c', check] + [staged[ep] for ep in epochs],
                      env=env, capture_output=True, text=True)
if proc.returncode != 0:
    die('the flows do not load under worker-node conditions:\n%s\n%s\n'
        'If torch is the missing module, this CMSSW python has no torch and the jobs '
        'cannot run covflow on the grid.' % (proc.stdout[-2000:], proc.stderr[-3000:]))
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
for k, v in sorted(report.items()):
    print('[PREFLIGHT] %-60s %s, %s, context %s, %d features' % (os.path.relpath(k, HERE), *v))
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(STAGE) for f in fs)
print('[PREFLIGHT] staged flows: %.1f MB' % (size / 1e6))

# ----------------------------------------------------------------------------
# configs
# ----------------------------------------------------------------------------
cfgs = []
for ds, camp, ep in tasks:
    cfg = base.create_config(ds)
    cfg.General.requestName = (cfg.General.requestName + '_cf' + ep)[:100]
    cfg.Data.outputDatasetTag = '%s_%s_cf%s' % (base.out_dir, camp, ep)
    cfg.JobType.scriptArgs = list(cfg.JobType.scriptArgs) + ['COVFLOW=covflow_%s/@mu@' % ep]
    cfg.JobType.inputFiles = list(cfg.JobType.inputFiles) + [staged[ep]]
    cfgs.append(cfg)

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
        continue
    p = Process(target=base.submit, args=(cfg,))
    p.start()
    p.join()
print('%s %d task(s)' % ('dry run:' if args.dry_run else 'submitted', len(cfgs)))
