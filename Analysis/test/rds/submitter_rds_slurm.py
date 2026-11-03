'''
SLURM submitter (PSI Tier-3) for the R(Ds) ntuplizer.

Same job layout as test/rjpsi/submitter_data_pnfs.py: one job per chunk of
input files, each file processed separately on /scratch, the parts merged with
hadd (both 'tree' and, for MC, 'tree_gen'), the result copied to
/pnfs/psi.ch/cms/trivcat/store/user/manzoni/<out_dir>/ with xrdcp. A chunk with
a failed part is not merged nor copied.

Run from a shell where `cmsenv` was done in the CMSSW area the jobs must use
(CMSSW_16_0_8 at PSI): the release is read from $CMSSW_BASE.

    python3 submitter_rds_slurm.py --mode data --files files_bph1_2018D.txt --out-dir rds_bph1_2018D_v1
    python3 submitter_rds_slurm.py --mode mc   --files files_signal.txt     --out-dir rds_signal_v1 \\
        --trk-calib $CMSSW_BASE/src/Bmmm/Analysis/data/trkcalib/rds_trkcalib_2018BParking_v4.json \\
        --cov-scale <table.json>
    python3 submitter_rds_slurm.py --mode mc ... --trig-sf <scale_factors.json> \\
        --cov-scale <sigma_dxy morph>.json --cov-scale-muon-only
    python3 submitter_rds_slurm.py ... --copy-input       # xrdcp each input to /scratch first
    python3 submitter_rds_slurm.py ... --dry-run          # write the job scripts, do not submit
    python3 submitter_rds_slurm.py ... --resubmit 3,17    # only these chunks

Provenance: at submission <out_dir>/provenance.json (git commit, branch,
pushed or not, uncommitted files, sha1 of the diff, the command line, the
options, sha256 of every correction file) and <out_dir>/uncommitted.diff
(`git diff HEAD`) are written and copied to the pnfs output directory. Each
job also prints the commit and diff hash it actually ran on (the jobs import
the code from the live working tree: edit nothing until they are done).

Fit schemes: the jobs export RDS_FIT_SCHEMES (--fit-schemes, default free;
see RDsCandidate.SCHEMES).

File lists: one LFN (/store/..., read through --redirector), /pnfs path (read
through the PSI door) or root:// URL per line.

--copy-input copies each remote input to the job's /scratch area with xrdcp
(3 attempts) before processing it, and deletes it afterwards. Use it for
samples with no replica close to PSI: streaming the UL18 MiniAOD through the
global redirector ran at ~0.3 ev/s (~30 h per file), the copy takes ~25 min
for a 2.3 GB file and the processing ~30 min.
'''

import os
import sys
import json
import time
import socket
import hashlib
import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--mode', choices=('data', 'mc'), required=True)
parser.add_argument('--files', required=True, help='text file, one input per line')
parser.add_argument('--out-dir', required=True)
parser.add_argument('--files-per-job', type=int, default=1)
parser.add_argument('--queue', default='standard', choices=('short', 'standard', 'long'))
parser.add_argument('--time', type=int, default=720, help='minutes')
parser.add_argument('--mem', type=int, default=4000,
                    help='MB per job (the cluster default, 1920, is too little for a full MiniAOD file)')
parser.add_argument('--redirector', default='root://cms-xrd-global.cern.ch/',
                    help='for /store/... LFNs; /pnfs paths go through the PSI door')
parser.add_argument('--nodelist', default='', help="e.g. 't3wn[80-91]'")
parser.add_argument('--lumi-mask', default='golden', help="data only ('golden', a file or a directory)")
parser.add_argument('--trk-calib', default='', help='MC only, see TrackMomentumCalib')
parser.add_argument('--cov-scale', default='')
parser.add_argument('--cov-scale-muon-only', action='store_true', help='--cov-scale on the muon only')
parser.add_argument('--trig-sf', default='', help="MC only, '<scale_factors.json>[:<map>]'")
parser.add_argument('--fit-schemes', default='free', help='RDS_FIT_SCHEMES of the jobs')
parser.add_argument('--allow-unpushed', action='store_true',
                    help='submit although the code has uncommitted changes or the commit is not pushed')
parser.add_argument('--hammer', default='', help='MC only, Hammer card')
parser.add_argument('--maxevents', type=int, default=-1)
parser.add_argument('--copy-input', action='store_true', help='xrdcp each input to /scratch first')
parser.add_argument('--resubmit', default='', help='comma-separated chunk indices')
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()

user = os.environ.get('USER', 'manzoni')
cmssw_base = os.environ.get('CMSSW_BASE', '')
scram_arch = os.environ.get('SCRAM_ARCH', '')
if not cmssw_base:
    raise RuntimeError('CMSSW_BASE is not set -- cmsenv in your CMSSW_X_Y_Z/src first.')
if args.mode == 'data' and (args.trk_calib or args.hammer or args.trig_sf or args.cov_scale):
    raise RuntimeError('--trk-calib / --hammer / --trig-sf / --cov-scale are MC only')

cfg = os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'test', 'rds', 'inspector_rds.py')
se_base = '/pnfs/psi.ch/cms/trivcat/store/user/%s' % user
out_file_name = 'rds'


def resolve(f):
    if f.startswith('root://'):
        return f
    if f.startswith('/pnfs/'):
        return 'root://t3dcachedb03.psi.ch:1094/' + f
    if f.startswith('/store/'):
        return args.redirector.rstrip('/') + '/' + f
    return f


with open(args.files) as fin:
    files = [resolve(l.strip()) for l in fin if l.strip() and not l.startswith('#')]
chunks = [files[i:i + args.files_per_job] for i in range(0, len(files), args.files_per_job)]
resub = set(int(x) for x in args.resubmit.split(',') if x)

flags = []
if args.mode == 'mc':
    flags += ['--mc', '--savenontrig']
    if args.trk_calib:
        flags.append('--trk-calib=%s' % os.path.abspath(args.trk_calib))
    if args.hammer:
        flags.append('--hammer=%s' % os.path.abspath(args.hammer))
    if args.trig_sf:
        path, _, name = args.trig_sf.partition(':')
        flags.append('--trig-sf=%s%s' % (os.path.abspath(path), ':' + name if name else ''))
else:
    lm = args.lumi_mask
    flags.append('--lumi-mask=%s' % (lm if lm in ('golden', 'none') else os.path.abspath(lm)))
if args.cov_scale:
    flags.append('--cov-scale=%s' % os.path.abspath(args.cov_scale))
    if args.cov_scale_muon_only:
        flags.append('--cov-scale-muon-only')
flags.append('--maxevents=%d' % args.maxevents)

local_dir = os.path.abspath(args.out_dir)
for d in (local_dir, local_dir + '/logs', local_dir + '/errs'):
    os.makedirs(d, exist_ok=True)
if not args.dry_run:
    os.makedirs(os.path.join(se_base, args.out_dir), exist_ok=True)
with open(os.path.join(local_dir, 'submission.txt'), 'w') as fout:
    fout.write('cmssw_base %s\nmode %s\nfiles %s\nflags %s\nRDS_FIT_SCHEMES %s\n'
               % (cmssw_base, args.mode, os.path.abspath(args.files), ' '.join(flags), args.fit_schemes))


# ---- provenance -----------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fin:
        for blk in iter(lambda: fin.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


sys.path.insert(0, os.path.join(cmssw_base, 'src', 'Bmmm', 'Analysis', 'python'))
from Provenance import provenance   # noqa: E402  (plain module, no CMSSW import)
prov = provenance(with_diff=True)
if (prov['dirty'] or not prov['pushed']) and not args.allow_unpushed and not args.dry_run:
    raise RuntimeError('the code has uncommitted changes (%s) or commit %s is not pushed: commit and '
                       'push first, or pass --allow-unpushed (the diff is saved either way)'
                       % (prov['dirty'], prov['commit']))
diff = prov.pop('diff')
corr = {}
for f in flags:
    if f.startswith(('--trk-calib=', '--cov-scale=', '--trig-sf=', '--hammer=')):
        path = f.split('=', 1)[1].split(':')[0]
        if os.path.isfile(path):
            corr[f.split('=', 1)[0].lstrip('-')] = {'path': path, 'sha256': sha256(path)}
# the applier module the job will load (TrackMomentumCalib._load_applier order)
for cand in (os.path.join(os.path.dirname(os.path.abspath(args.trk_calib or '.')), x)
             for x in ('rds_trkcalib.py', 'applier/python/rds_trkcalib.py')):
    if args.trk_calib and os.path.isfile(cand):
        corr['trk-calib-applier'] = {'path': cand, 'sha256': sha256(cand)}
        break
record = {
    'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'host': socket.gethostname(), 'user': user,
    'command': ' '.join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
    'out_dir': args.out_dir, 'pnfs': os.path.join(se_base, args.out_dir),
    'mode': args.mode, 'file_list': os.path.abspath(args.files),
    'file_list_sha256': sha256(args.files), 'cmssw_base': cmssw_base, 'scram_arch': scram_arch,
    'flags': flags, 'RDS_FIT_SCHEMES': args.fit_schemes, 'code': prov, 'corrections': corr,
}
with open(os.path.join(local_dir, 'provenance.json'), 'w') as fout:
    json.dump(record, fout, indent=1)
with open(os.path.join(local_dir, 'uncommitted.diff'), 'w') as fout:
    fout.write(diff)
print('provenance: commit %s (%s)%s' % (prov['commit'][:10], prov['branch'],
      '' if not prov['dirty'] else ', with uncommitted changes (uncommitted.diff)'))
if not args.dry_run:
    for f in ('provenance.json', 'uncommitted.diff', 'submission.txt'):
        os.system('xrdcp -f -s %s root://t3dcachedb03.psi.ch:1094//%s/%s/%s'
                  % (os.path.join(local_dir, f), se_base, args.out_dir, f))

scratch = '/scratch/%s/%s' % (user, args.out_dir)

for ijob, ichunk in enumerate(chunks):
    if resub and ijob not in resub:
        continue
    lines = [
        '#!/bin/bash',
        'mkdir -p %s' % scratch,
        'FAILED_PARTS=0',
        'export SCRAM_ARCH=%s' % scram_arch,
        'source /cvmfs/cms.cern.ch/cmsset_default.sh',
        'cd %s/src && eval `scramv1 runtime -sh`' % cmssw_base,
        'python3 -c "import sys; print(\'>>>> python startup OK\', sys.version.split()[0])" || exit 1',
        'export RDS_FIT_SCHEMES=%s' % args.fit_schemes,
        'cp $X509_USER_PROXY %s/x509proxy_%d && chmod 600 %s/x509proxy_%d' % (scratch, ijob, scratch, ijob),
        'export X509_USER_PROXY=%s/x509proxy_%d' % (scratch, ijob),
        'cd %s' % local_dir,
        '',
    ]
    for idx, ifile in enumerate(ichunk):
        infile = ifile
        if args.copy_input and ifile.startswith('root://'):
            infile = '%s/input_chunk%d_part%d.root' % (scratch, ijob, idx)
            lines += [
                'COPIED=0',
                'for attempt in 1 2 3; do',
                '  xrdcp -f %s %s && COPIED=1 && break' % (ifile, infile),
                '  echo ">>>> xrdcp attempt $attempt failed for %s"; sleep 60' % ifile,
                'done',
            ]
        run = ('python3 -u %s --inputFiles=%s --destination=%s --filename=%s_chunk%d_part%d '
               '--logfreq=1000 --logger=%s/logs/logger_chunk%d_part%d %s'
               % (cfg, infile, scratch, out_file_name, ijob, idx, local_dir, ijob, idx, ' '.join(flags)))
        fail = ('echo ">>>> FAILED: part%d of chunk%d (%s)"; FAILED_PARTS=$((FAILED_PARTS+1))'
                % (idx, ijob, ifile))
        if infile != ifile:
            lines += [
                'if [ $COPIED -eq 1 ]; then',
                '  %s' % run,
                '  if [ $? -ne 0 ]; then %s; fi' % fail,
                'else',
                '  %s' % fail,
                'fi',
                'rm -f %s' % infile,
            ]
        else:
            lines += [run, 'if [ $? -ne 0 ]; then %s; fi' % fail]
    out = '%s/%s_chunk%d.root' % (scratch, out_file_name, ijob)
    lines += [
        'if [ $FAILED_PARTS -gt 0 ]; then echo ">>>> $FAILED_PARTS part(s) failed: no merge, no copy"; exit 1; fi',
        'hadd -f -k %s %s/%s_chunk%d_part*.root || exit 1' % (out, scratch, out_file_name, ijob),
        'xrdcp -f %s root://t3dcachedb03.psi.ch:1094//%s/%s/%s_chunk%d.root'
        % (out, se_base, args.out_dir, out_file_name, ijob),
        'if [ $? -eq 0 ]; then rm -f %s %s/%s_chunk%d_part*.root; else echo ">>>> xrdcp FAILED, scratch kept"; exit 1; fi'
        % (out, scratch, out_file_name, ijob),
    ]
    script = '%s/submitter_chunk%d.sh' % (local_dir, ijob)
    with open(script, 'w') as fout:
        fout.write('\n'.join(lines) + '\n')
    cmd = ' '.join(['sbatch', '-p %s' % args.queue, '--account=t3',
                    '-o %s/logs/chunk%d.log' % (local_dir, ijob),
                    '-e %s/errs/chunk%d.err' % (local_dir, ijob),
                    '--job-name=%d_%s' % (ijob, args.out_dir),
                    '--time=%d' % args.time, '--mem=%d' % args.mem, '--nodes=1 --ntasks=1']
                   + (['--nodelist=%s' % args.nodelist] if args.nodelist else []) + [script])
    print(cmd)
    if not args.dry_run:
        os.system(cmd)

print('%d chunk(s), %d file(s); output in %s/%s' % (len(chunks), len(files), se_base, args.out_dir))
