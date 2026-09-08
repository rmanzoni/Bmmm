'''
LFNs -> PFNs that have been PROVED to open, one door at a time.

Measured on t3ui07 (check_file_access.sh, 2022C, a real file from DAS):

    bare LFN      FAIL   1814 ms      FWLite Events()  FAIL
    site TFC      FAIL   1764 ms      (edmFileUtil -d, off-site fiction)
    regional AAA  ok     7884 ms      FWLite Events()  ok, 16156 events
    global AAA    ok     7758 ms

So the rewriting is genuinely needed -- FWLite does not resolve a bare LFN the
way PoolSource does for a cmsRun job -- and the regional redirector is the
right default.

What this module adds on top of that is the reason CRAB was chosen for this
campaign in the first place: xrootd is sometimes unreliable. Picking one door
blindly means a job dies when that door is having a bad minute, even though
another door serves the same file. So each file is OPENED here, before the
event loop starts, and if the first door does not answer the next one is tried.
A job then fails only when no door anywhere can serve the file -- and it fails
in three lines of log rather than in a cppyy traceback 40 minutes in.

    python3 resolve_pfns.py --lfns lfns.txt --out pfns.txt
    python3 resolve_pfns.py --lfns lfns.txt --out pfns.txt \
                            --doors xrootd-cms.infn.it,cms-xrd-global.cern.ch
'''

from __future__ import print_function

import argparse
import sys
import time

# Order matters: nearest first. On a worker node CRAB has already put the job
# at a site holding the dataset, so the regional door normally answers with
# that site's own storage; the global one is the complete but slower fallback.
DEFAULT_DOORS = ['xrootd-cms.infn.it', 'cms-xrd-global.cern.ch']


def can_open(url, timeout_note=''):
    '''True if ROOT can open the URL and it looks like a CMSSW file.'''
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    try:
        handle = ROOT.TFile.Open(url)
    except Exception:
        return False
    if not handle or handle.IsZombie():
        return False
    ok = bool(handle.Get('Events'))
    handle.Close()
    return ok


def resolve(lfn, doors):
    '''The first URL for this LFN that actually opens, or None.'''
    if '://' in lfn:                 # already addressed: trust it, but check it
        return lfn if can_open(lfn) else None

    for door in doors:
        url   = 'root://%s/%s' % (door, lfn) if lfn.startswith('/') else lfn
        start = time.time()
        if can_open(url):
            print('  [ ok ] %-24s %5.1f s  %s' % (door, time.time() - start, url))
            return url
        print('  [fail] %-24s %5.1f s  %s' % (door, time.time() - start, url))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lfns' , required=True, help='one LFN per line')
    parser.add_argument('--out'  , required=True, help='where to write the working PFNs')
    parser.add_argument('--doors', default=','.join(DEFAULT_DOORS),
                        help='xrootd doors to try, in order, comma separated')
    parser.add_argument('--allow-partial', action='store_true',
                        help='proceed with the files that did open instead of failing')
    args = parser.parse_args()

    doors = [d.strip() for d in args.doors.split(',') if d.strip()]
    with open(args.lfns) as fin:
        lfns = [ln.strip() for ln in fin if ln.strip()]

    resolved, failed = [], []
    for lfn in lfns:
        print('%s' % lfn)
        url = resolve(lfn, doors)
        (resolved if url else failed).append(url or lfn)

    if failed:
        print('\n%d of %d input file(s) could not be opened through any of %s:'
              % (len(failed), len(lfns), doors), file=sys.stderr)
        for lfn in failed:
            print('  %s' % lfn, file=sys.stderr)
        # Default to failing: a job that silently processes 3 of its 5 files
        # produces an ntuple that looks fine and is quietly incomplete, and the
        # lumi accounting then claims those files were covered.
        if not args.allow_partial:
            print('refusing to run on a partial input set (--allow-partial to '
                  'override)', file=sys.stderr)
            return 1

    if not resolved:
        print('no input file could be opened at all', file=sys.stderr)
        return 1

    with open(args.out, 'w') as fout:
        fout.write('\n'.join(resolved) + '\n')
    print('\n%d/%d file(s) resolved -> %s' % (len(resolved), len(lfns), args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
