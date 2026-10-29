#!/usr/bin/env python3
'''
Submitter for the SLURM system (PSI Tier-3) -- Run 3 MC, skims of 24 Sep 2026.

Produces R(J/psi) ntuples from the privately produced MC skims
(tag rjpsi_run3_24sep26_v2), published on DAS (instance prod/phys03) and
stored at T3_CH_PSI. One output directory per dataset:

    RJpsi_25Sep2026_notrig_<kind>_<era>_v1      kind = Bc_inclusive | Bc_FilJPsi | Hb

What each job runs:

    inspector_rjpsi.py --mc --skim --savenontrig [--hammer <card>] --pu <campaign>:<card>

  * --skim         the skim writes the muons as selectedMuons (process SKIM)
  * --savenontrig  the MC skim has no HLT filter; trigger decision kept as branch
  * --hammer       Bc samples ONLY: Kiselev -> Harrison-2024 FF weights
                   (hammer_weight, hammer_ff_ev00..14_{up,dn}, hammer_status)
  * --pu           EVERY sample: pileup weights pu_weight_<year>[_up|_down] for
                   the sample's MC campaign (era Run3Summer22EE -> Summer22EE).
                   The card data/pu_weights_run3.json is validated at submission
                   with the same PileupSession the jobs build, and snapshotted
                   into the out_dir like the FF card. Unconfirmed MC profiles are
                   refused unless --pu-allow-unconfirmed.
  * no --covflow   the flow was trained on 2018 UL MC -> Run 3 data; applying
                   it to Run 3 MC would be an unvalidated correction.
  * Bc lifetime weights gen_bc_ctau_weight* are always on for MC.

Input:
  * default: file list + events per file from DAS (dasgoclient, prod/phys03);
    every LFN is then checked to exist on the local /pnfs mount (catches a
    dataset, or part of one, that is not at PSI).
  * --source pnfs: glob the CRAB output directory on /pnfs instead (no event
    counts -> fixed --files-per-job).
  Worker nodes do not mount /pnfs: jobs read through xrootd at
  root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/...

Chunking: files are grouped so that each job gets ~--events-per-job events
(at most --max-files-per-job files). All files of a chunk are processed by ONE
inspector call: Hammer integrates the BGLVar rate tensor once per python
process (minutes), so one call per file would pay that N times per job.
The output is therefore one file per chunk, no hadd.

Run from Bmmm/Analysis/test/rjpsi in a CMSSW_16_0_8 shell after `cmsenv`
and with a valid grid proxy on a shared filesystem:

    python3 submitter_mc_run3.py --list                     # samples and out_dirs
    python3 submitter_mc_run3.py --test 1                   # 1 chunk per sample, into <out_dir>_test  <-- first!
    python3 submitter_mc_run3.py --dry-run                  # scripts + provenance, submit nothing
    python3 submitter_mc_run3.py                            # everything
    python3 submitter_mc_run3.py --only Bc_inclusive_Run3Summer22EE --resubmit 3,17
                                                            # resubmit existing chunk scripts

Read the test jobs (wall time vs number of events in chunks.json, memory,
hammer_status, <hammer_weight>) before launching the full production and
adjust --events-per-job accordingly: the default is NOT calibrated on Run 3.
'''

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from glob import glob

##########################################################################################
# CONFIGURATION
##########################################################################################

# (kind, era, DAS dataset). era must appear in the dataset name as <era>MiniAOD.
SAMPLES = [
#     ('Hb'          , 'Run3Summer22EE'    , '/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/manzoni-rjpsi_run3_24sep26_v2_HbToJPsiMuMu_3MuFilter_Run3Summer22EEMiniAODv4_130X_mcRun3_2022_realistic_postEE_v6_v2-ed252f49127e0a18b17bf4333df62b77/USER'),
#     ('Hb'          , 'Run3Summer22'      , '/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/manzoni-rjpsi_run3_24sep26_v2_HbToJPsiMuMu_3MuFilter_Run3Summer22MiniAODv4_130X_mcRun3_2022_realistic_v5_v2-c424f8fdc47472099c67ab1ccdfa56dd/USER'),
#     ('Hb'          , 'Run3Summer23BPix'  , '/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/manzoni-rjpsi_run3_24sep26_v2_HbToJPsiMuMu_3MuFilter_Run3Summer23BPixMiniAODv4_130X_mcRun3_2023_realistic_postBPix_v6_v2-8de201320981fa3d391484bb39f183d1/USER'),
#     ('Hb'          , 'Run3Summer23'      , '/HbToJPsiMuMu_3MuFilter_TuneCP5_13p6TeV_pythia8-evtgen/manzoni-rjpsi_run3_24sep26_v2_HbToJPsiMuMu_3MuFilter_Run3Summer23MiniAODv4_130X_mcRun3_2023_realistic_v15_v2-471a844a92d5e9e62aed9ac427b1edba/USER'),
    ('Bc_inclusive', 'Run3Summer22EE'    , '/BcToJPsiMuMu_inclusive_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/manzoni-rjpsi_run3_24sep26_v2_BcToJPsiMuMu_inclusive_Run3Summer22EEMiniAODv4_130X_mcRun3_2022_realistic_postEE_v6_v2-ed252f49127e0a18b17bf4333df62b77/USER'),
    ('Bc_inclusive', 'Run3Summer22'      , '/BcToJPsiMuMu_inclusive_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/manzoni-rjpsi_run3_24sep26_v2_BcToJPsiMuMu_inclusive_Run3Summer22MiniAODv4_130X_mcRun3_2022_realistic_v5_v2-c424f8fdc47472099c67ab1ccdfa56dd/USER'),
    ('Bc_inclusive', 'Run3Summer23BPix'  , '/BcToJPsiMuMu_inclusive_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/manzoni-rjpsi_run3_24sep26_v2_BcToJPsiMuMu_inclusive_Run3Summer23BPixMiniAODv4_130X_mcRun3_2023_realistic_postBPix_v6_v4-8de201320981fa3d391484bb39f183d1/USER'),
    ('Bc_inclusive', 'Run3Summer23'      , '/BcToJPsiMuMu_inclusive_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/manzoni-rjpsi_run3_24sep26_v2_BcToJPsiMuMu_inclusive_Run3Summer23MiniAODv4_Fixed_130X_mcRun3_2023_realistic_v15_v3-471a844a92d5e9e62aed9ac427b1edba/USER'),
    ('Bc_FilJPsi'  , 'RunIII2024Summer24', '/BcToJPsiMuMu_Fil-JPsi_TuneCP5_13p6TeV_bcvegpy2-pythia8-evtgen/manzoni-rjpsi_run3_24sep26_v2_BcToJPsiMuMu_Fil_JPsi_RunIII2024Summer24MiniAODv6_150X_mcRun3_2024_realistic_v2_v6-02e6d2667309f3a5253e083c494e07f6/USER'),
]

# Hammer FF weights: Bc samples only
HAMMER_BY_KIND = {'Hb': False, 'Bc_inclusive': True, 'Bc_FilJPsi': True}

PRODUCTION_TAG = 'RJpsi_25Sep2026_notrig'
VERSION        = 'v2'

def out_dir_of(kind, era):
    return '%s_%s_%s_%s' % (PRODUCTION_TAG, kind, era, VERSION)

# release / tools
REQUIRED_RELEASE = 'CMSSW_16_0_8'           # Hammer is built against its Python 3.9
HAMMER_ENV       = '/work/manzoni/hammer/hammer_env.sh'
CARD_NAME        = 'harrison_bglvar.json'
PU_CARD_NAME     = 'pu_weights_run3.json'
CFG              = 'inspector_rjpsi.py'
OUT_FILE_NAME    = 'rjpsi'

# storage
DAS_INSTANCE = 'prod/phys03'
SE_HOST      = 't3dcachedb03.psi.ch:1094'
PNFS_PREFIX  = '/pnfs/psi.ch/cms/trivcat'   # PNFS_PREFIX + LFN = path on the UI mount
PNFS_USER    = PNFS_PREFIX + '/store/user/manzoni'
SCRATCH      = '/scratch/manzoni'

# batch
QUEUE    = 'standard'; TIME_MIN = 720
# QUEUE  = 'short'   ; TIME_MIN = 60
# QUEUE  = 'long'    ; TIME_MIN = 10080
MEM_MB   = 2500
NODELIST = 't3wn[80-91]'

##########################################################################################
# COMMAND LINE
##########################################################################################

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--only', action='append', default=[], metavar='SUBSTR',
                help='process only samples whose out_dir contains SUBSTR (repeatable, OR)')
ap.add_argument('--list', action='store_true',
                help='print the selected samples and their out_dirs, then exit')
ap.add_argument('--source', choices=['das', 'pnfs'], default='das',
                help='where the file list comes from (default: das)')
ap.add_argument('--events-per-job', type=int, default=200000,
                help='target events per job, DAS source only (default: 200000, NOT calibrated)')
ap.add_argument('--max-files-per-job', type=int, default=20,
                help='cap on files per job, DAS source only (default: 20)')
ap.add_argument('--files-per-job', type=int, default=1,
                help='files per job, pnfs source only (no event counts there)')
ap.add_argument('--test', type=int, default=0, metavar='N',
                help='submit only the first N chunks of each sample, into <out_dir>_test')
ap.add_argument('--dry-run', action='store_true',
                help='write job scripts, file lists and provenance; submit nothing')
ap.add_argument('--resubmit', default='', metavar='I,J,...',
                help='re-sbatch existing chunk scripts of ONE already-submitted sample')
ap.add_argument('--skip-pnfs-check', action='store_true',
                help='do not stat every input LFN on the /pnfs mount (DAS source)')
ap.add_argument('--pu-allow-unconfirmed', action='store_true',
                help='accept MC campaigns whose generation pileup profile is not yet '
                     'confirmed in pu_config_run3.py (jobs run with :allow-unconfirmed)')
args = ap.parse_args()


def pu_campaign(era):
    '''Sample era -> pileup-card campaign: Run3Summer22EE -> Summer22EE,
    RunIII2024Summer24 -> Summer24 (the keys of pu_config_run3.CAMPAIGNS).'''
    import re
    campaign = re.sub(r'^(RunIII\d{4}|Run3)', '', era)
    if not campaign or campaign == era:
        raise ValueError('cannot map era %r onto a pileup campaign' % era)
    return campaign

##########################################################################################
# HELPERS
##########################################################################################

def die(msg):
    raise SystemExit('[FATAL] ' + msg)


def progress(i, n, label):
    '''One-line progress bar on stderr.'''
    width = 30
    frac  = float(i) / n if n else 1.
    bar   = '#' * int(round(width * frac)) + '.' * (width - int(round(width * frac)))
    sys.stderr.write('\r    %-18s [%s] %d/%d' % (label, bar, i, n))
    if i >= n:
        sys.stderr.write('\n')
    sys.stderr.flush()


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT,
                                       universal_newlines=True).strip()
    except subprocess.CalledProcessError as exc:
        return '<failed: %s>' % exc.output.strip()


def lfn_to_url(lfn):
    return 'root://%s/%s%s' % (SE_HOST, PNFS_PREFIX, lfn)       # -> root://host//pnfs/...


def output_dataset_tag(dataset):
    '''/primary/manzoni-<outputDatasetTag>-<psethash>/USER -> outputDatasetTag'''
    processed = dataset.split('/')[2]
    if not processed.startswith('manzoni-'):
        die('unexpected processed dataset name %r' % processed)
    return processed[len('manzoni-'):processed.rfind('-')]


def das_files(dataset):
    '''[(lfn, nevents)] from DAS. Fails loudly on errors, empties and duplicates.'''
    query = 'file dataset=%s instance=%s | grep file.name, file.nevents' % (dataset, DAS_INSTANCE)
    res = subprocess.run(['dasgoclient', '-query=%s' % query],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    if res.returncode != 0:
        die('dasgoclient failed for %s:\n%s' % (dataset, res.stderr.strip()))
    out = []
    for line in res.stdout.splitlines():
        tok = line.split()
        if not tok:
            continue
        if len(tok) != 2 or not tok[0].startswith('/store/'):
            die('cannot parse DAS line %r for %s' % (line, dataset))
        out.append((tok[0], int(tok[1])))
    if not out:
        die('DAS returned no files for %s (instance %s)' % (dataset, DAS_INSTANCE))
    lfns = [f for f, _ in out]
    if len(set(lfns)) != len(lfns):
        die('DAS returned duplicate LFNs for %s' % dataset)
    return sorted(out)


def pnfs_files(dataset):
    '''[(lfn, None)] by globbing the CRAB output area on /pnfs.
    CRAB layout: <outLFNDirBase>/<primary>/<outputDatasetTag>/<timestamp>/000X/*.root.
    Refuses if more than one CRAB task (timestamp) directory is found.'''
    primary = dataset.split('/')[1]
    tag     = output_dataset_tag(dataset)
    hits = []
    for base in (PNFS_USER, PNFS_USER + '/*'):
        hits += glob('%s/%s/%s/*/*/*.root' % (base, primary, tag))
    hits = sorted(set(hits))
    if not hits:
        die('no files on /pnfs for %s (looked for %s/[*/]%s/%s/*/*/*.root)'
            % (dataset, PNFS_USER, primary, tag))
    tasks = sorted(set(os.path.dirname(os.path.dirname(h)) for h in hits))
    if len(tasks) > 1:
        die('more than one CRAB task directory for %s -- pick one, or use --source das:\n    %s'
            % (dataset, '\n    '.join(tasks)))
    return [(h[len(PNFS_PREFIX):], None) for h in hits]


def check_on_pnfs(files):
    missing = []
    for i, (lfn, _) in enumerate(files):
        if not os.path.isfile(PNFS_PREFIX + lfn):
            missing.append(lfn)
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            progress(i + 1, len(files), 'pnfs check')
    if missing:
        die('%d of %d files are not on the PSI /pnfs mount (not at T3_CH_PSI?), e.g.\n    %s'
            % (len(missing), len(files), '\n    '.join(missing[:5])))


def make_chunks(files):
    '''Group (lfn, nevents) into chunks.'''
    if args.source == 'pnfs':
        n = args.files_per_job
        return [files[i:i + n] for i in range(0, len(files), n)]
    chunks, cur, cur_ev = [], [], 0
    for lfn, nev in files:
        if cur and (cur_ev + nev > args.events_per_job or len(cur) >= args.max_files_per_job):
            chunks.append(cur)
            cur, cur_ev = [], 0
        cur.append((lfn, nev))
        cur_ev += nev
    if cur:
        chunks.append(cur)
    return chunks


def sbatch(out_dir, jobid):
    cmd = ' '.join([
        'sbatch',
        '-p %s' % QUEUE,
        '--account=t3',
        '-o %s/logs/chunk%d.log' % (out_dir, jobid),
        '-e %s/errs/chunk%d.err' % (out_dir, jobid),
        '--job-name=%d_%s' % (jobid, out_dir),
        '--time=%d' % TIME_MIN,
        '--mem=%d' % MEM_MB,
        '--nodes=1 --ntasks=1 --nodelist=%s' % NODELIST,
        '%s/submitter_chunk%d.sh' % (out_dir, jobid),
    ])
    print(cmd)
    if not args.dry_run:
        if os.system(cmd) != 0:
            die('sbatch failed for chunk %d of %s' % (jobid, out_dir))

##########################################################################################
# SELECTION
##########################################################################################

for kind, era, ds in SAMPLES:
    if kind not in HAMMER_BY_KIND:
        die('unknown kind %r' % kind)
    if (era + 'MiniAOD') not in ds:
        die('era %r does not match dataset %s' % (era, ds))

selected = [(k, e, d) for k, e, d in SAMPLES
            if not args.only or any(s in out_dir_of(k, e) for s in args.only)]
if not selected:
    die('--only %s matches no sample' % args.only)
if len(set(out_dir_of(k, e) for k, e, _ in selected)) != len(selected):
    die('two samples map to the same out_dir')

if args.list:
    for k, e, d in selected:
        print('%-60s hammer=%-5s pu=%-12s %s' % (out_dir_of(k, e), HAMMER_BY_KIND[k],
                                                 pu_campaign(e), d))
    sys.exit(0)

##########################################################################################
# RESUBMISSION: re-sbatch existing scripts, nothing is regenerated
##########################################################################################

if args.resubmit:
    if len(selected) != 1:
        die('--resubmit needs exactly one sample (use --only), got %d' % len(selected))
    out_dir = out_dir_of(*selected[0][:2]) + ('_test' if args.test else '')
    ids = sorted(set(int(x) for x in args.resubmit.split(',') if x.strip()))
    for jobid in ids:
        if not os.path.isfile('%s/submitter_chunk%d.sh' % (out_dir, jobid)):
            die('%s/submitter_chunk%d.sh does not exist' % (out_dir, jobid))
        target = '%s/%s/%s_chunk%d.root' % (PNFS_USER, out_dir, OUT_FILE_NAME, jobid)
        if os.path.exists(target):
            die('%s already exists; remove it first if you really want to redo chunk %d'
                % (target, jobid))
    for jobid in ids:
        sbatch(out_dir, jobid)
    print('#### %s %d chunk(s) of %s' % ('wrote (dry run)' if args.dry_run else 'resubmitted',
                                         len(ids), out_dir))
    sys.exit(0)

##########################################################################################
# PREFLIGHT: everything that would make every job fail is caught here, once
##########################################################################################

cwd        = os.getcwd()
cmssw_base = os.environ.get('CMSSW_BASE', '')
scram_arch = os.environ.get('SCRAM_ARCH', '')
if not cmssw_base:
    die('CMSSW_BASE is not set -- run `cmsenv` in %s/src first.' % REQUIRED_RELEASE)
if os.path.basename(os.path.normpath(cmssw_base)) != REQUIRED_RELEASE:
    die('CMSSW_BASE is %s, expected a %s area (Hammer is built against its Python).'
        % (cmssw_base, REQUIRED_RELEASE))
if not os.path.isfile(CFG):
    die('%s not found in %s: run the submitter from Bmmm/Analysis/test/rjpsi.' % (CFG, cwd))

if args.source == 'das' and shutil.which('dasgoclient') is None:
    die('dasgoclient not found (cmsenv / cvmfs?), or use --source pnfs.')
if not args.dry_run and shutil.which('sbatch') is None:
    die('sbatch not found: not on a SLURM submit host?')

proxy = os.environ.get('X509_USER_PROXY', '')
if not proxy or not os.path.isfile(proxy):
    die('X509_USER_PROXY is unset or points to a missing file.')
if proxy.startswith('/tmp/'):
    die('X509_USER_PROXY=%s is node-local; the worker nodes cannot read it. '
        'Create it on a shared filesystem, e.g. voms-proxy-init --voms cms --valid 192:00 '
        '--out $HOME/.x509up_u$(id -u) && export X509_USER_PROXY=$HOME/.x509up_u$(id -u)' % proxy)
if subprocess.call(['voms-proxy-info', '-exists', '-valid', '24:00'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
    die('grid proxy missing or valid for less than 24 h: renew it (voms-proxy-init --voms cms --valid 192:00).')

need_hammer = any(HAMMER_BY_KIND[k] for k, _, _ in selected)
card, card_src = None, None
if need_hammer:
    if not os.path.isfile(HAMMER_ENV):
        die('%s not found: Hammer is not installed where the jobs expect it.' % HAMMER_ENV)
    # validated with the SAME code the jobs use; importing HammerFF does not import hammer
    from Bmmm.Analysis.HammerFF import P1_CONVENTION, load_card
    card_src = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'data', CARD_NAME)
    if not os.path.isfile(card_src):
        die('FF card %s not found.' % card_src)
    card = load_card(card_src)
    if card.get('p1_convention') != P1_CONVENTION:
        die('FF card %s is not fitted in the P1 convention Hammer uses (%r, card says %r).'
            % (card_src, P1_CONVENTION, card.get('p1_convention')))
    if str(card.get('covariance_status', '')).lower() != 'validated':
        die('FF card %s has covariance_status=%r: the jobs would refuse to write the '
            'eigenvariations. Set it to "validated" once the covariance is validated.'
            % (card_src, card.get('covariance_status')))

# pileup card: every sample gets weights. Validated with the SAME session the
# jobs build (schema, campaign present, MC profile confirmed), for every
# selected sample, before anything is written or submitted.
from Bmmm.Analysis.PileupWeights import PileupSession
pu_card_src = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'data', PU_CARD_NAME)
if not os.path.isfile(pu_card_src):
    die('pileup card %s not found: build it with test/rjpsi/pileup/build_pu_card.py.'
        % pu_card_src)
for _kind, _era, _ in selected:
    try:
        PileupSession(pu_campaign(_era), card_path=pu_card_src,
                      allow_unconfirmed=args.pu_allow_unconfirmed, verbose=True)
    except Exception as exc:
        die('pileup weights for %s %s: %s' % (_kind, _era, exc))
with open(pu_card_src, 'rb') as fcard:
    pu_card_sha = hashlib.sha256(fcard.read()).hexdigest()

repo      = os.path.join(cmssw_base, 'src', 'Bmmm')
bmmm_head = sh('git -C %s rev-parse HEAD' % repo)
dirty     = sh('git -C %s status --porcelain --untracked-files=no' % repo)
if dirty:
    print('[WARN] the Bmmm checkout has uncommitted changes; they are recorded in PROVENANCE.json')

##########################################################################################
# FILE LISTS AND CHUNKS (all samples first: nothing is submitted if one sample fails)
##########################################################################################

plan = []
for kind, era, dataset in selected:
    out_dir = out_dir_of(kind, era) + ('_test' if args.test else '')
    print('\n#### %s' % out_dir)
    print('     %s' % dataset)

    if os.path.exists(out_dir):
        die('local directory %s already exists: bump VERSION, or use --resubmit.' % out_dir)
    existing = glob('%s/%s/%s_chunk*.root' % (PNFS_USER, out_dir, OUT_FILE_NAME))
    if existing:
        die('%d output file(s) already in %s/%s: refusing to overwrite.'
            % (len(existing), PNFS_USER, out_dir))

    files = das_files(dataset) if args.source == 'das' else pnfs_files(dataset)
    if args.source == 'das' and not args.skip_pnfs_check:
        check_on_pnfs(files)
    chunks = make_chunks(files)

    n_ev = sum(n for _, n in files) if args.source == 'das' else None
    print('     %d files, %s events, %d chunks, hammer=%s'
          % (len(files), n_ev if n_ev is not None else '?', len(chunks), HAMMER_BY_KIND[kind]))
    plan.append((kind, era, dataset, out_dir, files, chunks, n_ev))

##########################################################################################
# WRITE AND SUBMIT
##########################################################################################

JOB_HEAD = '''#!/bin/bash
# {out_dir} -- chunk {jobid}: {nfiles} file(s), {nevents} events

SCRATCH_DIR={scratch}/{out_dir}
mkdir -p $SCRATCH_DIR
echo ">>>> host $(hostname), scratch $SCRATCH_DIR"

# --- CMSSW runtime (native, no container) ---
export SCRAM_ARCH={scram_arch}
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd {cmssw_base}/src
eval `scramv1 runtime -sh`
echo ">>>> CMSSW_BASE=$CMSSW_BASE"
echo ">>>> using python: $(which python3)"
python3 -c "import sys; print('>>>> python startup OK', sys.version.split()[0])"
if [ $? -ne 0 ]; then
    echo ">>>> FATAL: cmsenv did not take effect. Aborting chunk {jobid}."
    exit 1
fi
'''

JOB_HAMMER = '''
# --- Hammer on the path, after cmsenv (it must not be overridden) ---
source {hammer_env}
python3 -c "import hammer; print('>>>> hammer', getattr(hammer, 'version', '?'), hammer.__file__)"
if [ $? -ne 0 ]; then
    echo ">>>> FATAL: hammer is not importable on this node. Aborting chunk {jobid}."
    exit 1
fi
'''

JOB_BODY = '''
# --- grid proxy: private copy that survives the whole job, removed at exit ---
PROXY_COPY=$SCRATCH_DIR/x509proxy_{jobid}
cp "$X509_USER_PROXY" $PROXY_COPY || {{ echo ">>>> FATAL: cannot copy the grid proxy"; exit 1; }}
chmod 600 $PROXY_COPY
export X509_USER_PROXY=$PROXY_COPY
trap 'rm -f $PROXY_COPY' EXIT

OUT=$SCRATCH_DIR/{outfile}_chunk{jobid}.root

# --- run from the output dir ---
cd {dir}
echo ">>>> now running in $PWD"

python3 {dir}/{cfg} \\
    --inputFiles={infiles} \\
    --logfreq=5000 \\
    --destination=$SCRATCH_DIR \\
    --logger={dir}/logs/cutflow_chunk{jobid} \\
    --savenontrig \\
    --mc \\
    --skim \\
{hammer_opt}    --pu={pu_spec} \\
    --filename={outfile}_chunk{jobid}
if [ $? -ne 0 ]; then
    echo ">>>> FAILED: inspector for chunk {jobid}, no transfer"
    rm -f $OUT
    exit 1
fi

ls -latrh $SCRATCH_DIR

xrdcp $OUT root://{se_host}//{pnfs_user}/{out_dir}/{outfile}_chunk{jobid}.root
if [ $? -eq 0 ]; then
    echo ">>>> xrdcp succeeded, cleaning scratch"
    rm -f $OUT
else
    echo ">>>> xrdcp FAILED for chunk {jobid}, $OUT kept for inspection"
    exit 1
fi
'''

n_total = 0
for kind, era, dataset, out_dir, files, chunks, n_ev in plan:
    use_hammer = HAMMER_BY_KIND[kind]
    job_dir    = os.path.join(cwd, out_dir)

    os.makedirs(os.path.join(out_dir, 'logs'))
    os.makedirs(os.path.join(out_dir, 'errs'))
    os.makedirs(os.path.join(PNFS_USER, out_dir), exist_ok=True)
    shutil.copy2(CFG, out_dir)

    # FF card snapshot: the jobs read THIS copy, so refitting the card in the
    # release later cannot change a running production.
    card_used, card_sha = None, None
    if use_hammer:
        card_used = os.path.join(job_dir, CARD_NAME)
        shutil.copy2(card_src, card_used)
        with open(card_used, 'rb') as fcard:
            card_sha = hashlib.sha256(fcard.read()).hexdigest()

    # pileup card snapshot, same reason as the FF card
    pu_card_used = os.path.join(job_dir, PU_CARD_NAME)
    shutil.copy2(pu_card_src, pu_card_used)
    pu_spec = '%s%s:%s' % (pu_campaign(era),
                           ':allow-unconfirmed' if args.pu_allow_unconfirmed else '',
                           pu_card_used)

    with open(os.path.join(out_dir, 'files.txt'), 'w') as flist:
        for lfn, nev in files:
            flist.write('%s %s\n' % (lfn, nev if nev is not None else ''))
    with open(os.path.join(out_dir, 'chunks.json'), 'w') as fch:
        json.dump([{'chunk': i, 'files': [f for f, _ in c],
                    'nevents': (sum(n for _, n in c) if args.source == 'das' else None)}
                   for i, c in enumerate(chunks)], fch, indent=1)

    provenance = {
        'submitted'        : datetime.datetime.now().isoformat(timespec='seconds'),
        'out_dir'          : out_dir,
        'dataset'          : dataset,
        'das_instance'     : DAS_INSTANCE,
        'kind'             : kind,
        'era'              : era,
        'source'           : args.source,
        'n_files'          : len(files),
        'n_events_das'     : n_ev,
        'n_chunks'         : len(chunks),
        'events_per_job'   : args.events_per_job if args.source == 'das' else None,
        'max_files_per_job': args.max_files_per_job if args.source == 'das' else None,
        'files_per_job'    : args.files_per_job if args.source == 'pnfs' else None,
        'test'             : args.test,
#         'inspector_flags'  : '--mc --skim --savenontrig' + (' --hammer <card>' if use_hammer else ''),
        'inspector_flags'  : '--mc --skim ' + (' --hammer <card>' if use_hammer else '')
                             + ' --pu <campaign>:<card>',
        'pileup'           : {'campaign': pu_campaign(era), 'card': pu_card_used,
                              'sha256': pu_card_sha,
                              'allow_unconfirmed': args.pu_allow_unconfirmed},
        'covflow'          : None,
        'cmssw_base'       : cmssw_base,
        'scram_arch'       : scram_arch,
        'bmmm_commit'      : bmmm_head,
        'bmmm_dirty_files' : dirty.splitlines() if dirty else [],
        'hammer_env'       : HAMMER_ENV if use_hammer else None,
        'ff_card'          : ({'file': card_used, 'sha256': card_sha,
                               'name': card.get('name'), 'fit_date': card.get('fit_date'),
                               'p1_convention': card.get('p1_convention'),
                               'covariance_status': card.get('covariance_status')}
                              if use_hammer else None),
        'batch'            : {'queue': QUEUE, 'time_min': TIME_MIN, 'mem_mb': MEM_MB,
                              'nodelist': NODELIST},
    }
    with open(os.path.join(out_dir, 'PROVENANCE.json'), 'w') as fprov:
        json.dump(provenance, fprov, indent=1)

    n_sub = 0
    for jobid, chunk in enumerate(chunks):
        if args.test and jobid >= args.test:
            break
        fmt = dict(
            out_dir    = out_dir,
            jobid      = jobid,
            nfiles     = len(chunk),
            nevents    = sum(n for _, n in chunk) if args.source == 'das' else '?',
            scratch    = SCRATCH,
            scram_arch = scram_arch,
            cmssw_base = cmssw_base,
            hammer_env = HAMMER_ENV,
            dir        = job_dir,
            cfg        = CFG,
            infiles    = ','.join(lfn_to_url(f) for f, _ in chunk),
            outfile    = OUT_FILE_NAME,
            hammer_opt = ('    --hammer %s \\\n' % card_used) if use_hammer else '',
            pu_spec    = pu_spec,
            se_host    = SE_HOST,
            pnfs_user  = PNFS_USER,
        )
        script = JOB_HEAD.format(**fmt)
        if use_hammer:
            script += JOB_HAMMER.format(**fmt)
        script += JOB_BODY.format(**fmt)

        with open('%s/submitter_chunk%d.sh' % (out_dir, jobid), 'wt') as flauncher:
            flauncher.write(script)
        sbatch(out_dir, jobid)
        n_sub += 1

    n_total += n_sub
    print('#### %s %d chunk(s) of %d into %s'
          % ('wrote (dry run)' if args.dry_run else 'submitted', n_sub, len(chunks), out_dir))

print('\n#### done: %d job(s) over %d sample(s), Bmmm %s%s'
      % (n_total, len(plan), bmmm_head[:10], ' (dirty)' if dirty else ''))