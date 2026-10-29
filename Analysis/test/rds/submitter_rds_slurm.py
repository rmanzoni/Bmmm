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
    python3 submitter_rds_slurm.py ... --dry-run          # write the job scripts, do not submit
    python3 submitter_rds_slurm.py ... --resubmit 3,17    # only these chunks

File lists: one LFN (/store/..., read through --redirector), /pnfs path (read
through the PSI door) or root:// URL per line.
'''

import os
import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--mode', choices=('data', 'mc'), required=True)
parser.add_argument('--files', required=True, help='text file, one input per line')
parser.add_argument('--out-dir', required=True)
parser.add_argument('--files-per-job', type=int, default=1)
parser.add_argument('--queue', default='standard', choices=('short', 'standard', 'long'))
parser.add_argument('--time', type=int, default=720, help='minutes')
parser.add_argument('--redirector', default='root://cms-xrd-global.cern.ch/',
                    help='for /store/... LFNs; /pnfs paths go through the PSI door')
parser.add_argument('--nodelist', default='', help="e.g. 't3wn[80-91]'")
parser.add_argument('--lumi-mask', default='golden', help="data only ('golden', a file or a directory)")
parser.add_argument('--trk-calib', default='', help='MC only, see TrackMomentumCalib')
parser.add_argument('--cov-scale', default='')
parser.add_argument('--hammer', default='', help='MC only, Hammer card')
parser.add_argument('--maxevents', type=int, default=-1)
parser.add_argument('--resubmit', default='', help='comma-separated chunk indices')
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()

user = os.environ.get('USER', 'manzoni')
cmssw_base = os.environ.get('CMSSW_BASE', '')
scram_arch = os.environ.get('SCRAM_ARCH', '')
if not cmssw_base:
    raise RuntimeError('CMSSW_BASE is not set -- cmsenv in your CMSSW_X_Y_Z/src first.')
if args.mode == 'data' and (args.trk_calib or args.hammer):
    raise RuntimeError('--trk-calib / --hammer are MC only')

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
else:
    lm = args.lumi_mask
    flags.append('--lumi-mask=%s' % (lm if lm in ('golden', 'none') else os.path.abspath(lm)))
if args.cov_scale:
    flags.append('--cov-scale=%s' % os.path.abspath(args.cov_scale))
flags.append('--maxevents=%d' % args.maxevents)

local_dir = os.path.abspath(args.out_dir)
for d in (local_dir, local_dir + '/logs', local_dir + '/errs'):
    os.makedirs(d, exist_ok=True)
if not args.dry_run:
    os.makedirs(os.path.join(se_base, args.out_dir), exist_ok=True)
with open(os.path.join(local_dir, 'submission.txt'), 'w') as fout:
    fout.write('cmssw_base %s\nmode %s\nfiles %s\nflags %s\n'
               % (cmssw_base, args.mode, os.path.abspath(args.files), ' '.join(flags)))

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
        'cp $X509_USER_PROXY %s/x509proxy_%d && chmod 600 %s/x509proxy_%d' % (scratch, ijob, scratch, ijob),
        'export X509_USER_PROXY=%s/x509proxy_%d' % (scratch, ijob),
        'cd %s' % local_dir,
        '',
    ]
    for idx, ifile in enumerate(ichunk):
        lines += [
            'python3 -u %s --inputFiles=%s --destination=%s --filename=%s_chunk%d_part%d '
            '--logfreq=1000 --logger=%s/logs/logger_chunk%d_part%d %s'
            % (cfg, ifile, scratch, out_file_name, ijob, idx, local_dir, ijob, idx, ' '.join(flags)),
            'if [ $? -ne 0 ]; then echo ">>>> FAILED: part%d of chunk%d (%s)"; FAILED_PARTS=$((FAILED_PARTS+1)); fi'
            % (idx, ijob, ifile),
        ]
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
                    '--time=%d' % args.time, '--nodes=1 --ntasks=1']
                   + (['--nodelist=%s' % args.nodelist] if args.nodelist else []) + [script])
    print(cmd)
    if not args.dry_run:
        os.system(cmd)

print('%d chunk(s), %d file(s); output in %s/%s' % (len(chunks), len(files), se_base, args.out_dir))
