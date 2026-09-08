#!/bin/bash
#
# Run ONE grid job on this machine, exactly as the worker node would, BEFORE
# submitting 257 of them.
#
# The point is fidelity, not convenience: it reproduces the two things that
# make a grid job different from an interactive run, and that is where both
# failures of this campaign came from.
#
#   * CRAB FLATTENS the sandbox. Everything in JobType.inputFiles lands in one
#     working directory with no directory structure, and the job runs there.
#     A path that works from test/mm/crab/ can be absent on the worker node.
#     This copies the same files into a scratch directory and runs there.
#
#   * CRAB PARSES FrameworkJobReport.xml afterwards, whatever the job ran, and
#     fails the job if it cannot. A missing or unparsable report fails a job
#     that did perfect physics (that was exit code 50115).
#
# It checks what CRAB checks: the script's exit code, that every file named in
# JobType.outputFiles exists, and that the job report exists and parses.
#
#   ./test_job_locally.sh                       # 2000 events off one 2022C file
#   ./test_job_locally.sh --maxevents 200       # quicker
#   ./test_job_locally.sh --lfn /store/data/... # a specific file
#   ./test_job_locally.sh --keep                # leave the scratch dir to poke at
#
# Run it from inside a CMSSW release with the grid environment set up:
#   cmsenv && voms-proxy-init -voms cms
#
# Exit status is 0 only if a real grid job would have been marked finished.

set -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# No hard-coded LFN. An earlier version of this script carried one that LOOKED
# like a file from this dataset and was not: step 1b then failed on a file that
# had never existed, which is a worse failure than no default at all. Ask DAS
# for a real file instead, or pass --lfn.
DATASET='/ParkingDoubleMuonLowMass1/Run2022C-PromptReco-v1/MINIAOD'
LFN=''
MAXEVENTS=2000
KEEP=0
JOBID=1

while [ $# -gt 0 ]; do
    case "$1" in
        --lfn)       LFN=$2      ; shift 2 ;;
        --dataset)   DATASET=$2  ; shift 2 ;;
        --maxevents) MAXEVENTS=$2; shift 2 ;;
        --jobid)     JOBID=$2    ; shift 2 ;;
        --keep)      KEEP=1      ; shift   ;;
        -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

# ------------------------------------------------------------------------------
# 0. Environment. A grid job has CMSSW and a proxy; without them this test is
#    not testing the same thing, so say so rather than fail confusingly later.
# ------------------------------------------------------------------------------
FAIL=0
say_fail() { echo "  [FAIL] $*"; FAIL=1; }
say_ok()   { echo "  [ ok ] $*"; }

echo "=== 0. environment ==="
[ -n "$CMSSW_BASE" ] && say_ok "CMSSW_BASE=$CMSSW_BASE" \
                     || say_fail "no CMSSW_BASE -- run cmsenv first"
command -v edmFileUtil >/dev/null && say_ok "edmFileUtil on PATH" \
                                  || say_fail "edmFileUtil not on PATH"
if voms-proxy-info -exists -valid 0:10 >/dev/null 2>&1; then
    say_ok "valid grid proxy ($(voms-proxy-info -timeleft 2>/dev/null)s left)"
else
    say_fail "no valid grid proxy -- voms-proxy-init -voms cms"
fi
python3 -c 'import uproot' 2>/dev/null && say_ok "uproot importable" \
                                       || say_fail "uproot not importable by python3"
python3 -c 'import Bmmm.Analysis.MuMuBranches' 2>/dev/null \
    && say_ok "Bmmm.Analysis importable" \
    || say_fail "Bmmm.Analysis not importable -- is the release built (scram b)?"
[ $FAIL -ne 0 ] && { echo; echo "environment not ready; fix the above first"; exit 1; }

# ------------------------------------------------------------------------------
# 1. Build the flattened sandbox, the way CRAB does.
#    Keep this list in sync with JobType.inputFiles in crab_dimuon_data_run3.py.
# ------------------------------------------------------------------------------
echo
echo "=== 1. flattened sandbox ==="
SANDBOX=$(mktemp -d "${TMPDIR:-/tmp}/bmmm_jobtest.XXXXXX")
echo "  $SANDBOX"

for f in "$HERE/PSet.py" "$HERE/crab_script.sh" "$HERE/make_fjr.py" \
         "$HERE/resolve_pfns.py" "$HERE/../inspector_mm_analysis.py"; do
    if [ ! -f "$f" ]; then
        echo "  [FAIL] missing sandbox file: $f"; exit 1
    fi
    cp "$f" "$SANDBOX/"
    say_ok "$(basename "$f")"
done

if [ -z "$LFN" ]; then
    command -v dasgoclient >/dev/null || {
        echo "  [FAIL] dasgoclient not on PATH and no --lfn given"; exit 1; }
    echo "  asking DAS for a file in $DATASET ..."
    LFN=$(dasgoclient -query "file dataset=$DATASET" -limit 1 2>/dev/null | head -1)
    [ -n "$LFN" ] || { echo "  [FAIL] DAS returned no file for $DATASET"; exit 1; }
fi

# CRAB writes this per job; crab_script.sh prefers it over PSet.py
python3 - "$SANDBOX/job_input_file_list_${JOBID}.txt" "$LFN" <<'PYEOF'
import json, sys
with open(sys.argv[1], 'w') as f:
    json.dump([sys.argv[2]], f)
PYEOF
say_ok "job_input_file_list_${JOBID}.txt -> $LFN"

# ------------------------------------------------------------------------------
# 1b. Can the input file actually be opened?
#
# This is its own step because a PFN that merely LOOKS right is what killed the
# second campaign: `edmFileUtil -d` rewrote the LFN into a PSI path for a file
# PSI does not host, and nothing noticed until TFile::Open failed on the worker
# node. Open it here, cheaply, and say so.
# ------------------------------------------------------------------------------
echo
echo "=== 1b. can the input be opened? ==="
REDIRECTOR=${BMMM_REDIRECTOR:-xrootd-cms.infn.it}
TESTURL="root://${REDIRECTOR}/${LFN}"
if python3 - "$TESTURL" <<'PYEOF'
import sys, ROOT
ROOT.gErrorIgnoreLevel = ROOT.kFatal
f = ROOT.TFile.Open(sys.argv[1])
if not f or f.IsZombie():
    sys.exit(1)
print('  [ ok ] opened, %.1f MB' % (f.GetSize() / 1024.**2))
f.Close()
PYEOF
then
    say_ok "$TESTURL"
else
    say_fail "cannot open $TESTURL"
    say_fail "the grid job will fail the same way; try BMMM_REDIRECTOR=cms-xrd-global.cern.ch"
    [ $KEEP -eq 0 ] && rm -rf "$SANDBOX"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Run it, from inside the sandbox, with nothing else on the path.
# ------------------------------------------------------------------------------
echo
echo "=== 2. running crab_script.sh ${JOBID} ==="
echo "    (capping at $MAXEVENTS events; the grid job has no cap)"
BMMM_TEST_MAXEVENTS=$MAXEVENTS
export BMMM_TEST_MAXEVENTS

START=$(date +%s)
( cd "$SANDBOX" && bash crab_script.sh "$JOBID" ) 2>&1 | tee "$SANDBOX/job.log"
RC=${PIPESTATUS[0]}
ELAPSED=$(( $(date +%s) - START ))

# ------------------------------------------------------------------------------
# 3. Check what CRAB checks.
# ------------------------------------------------------------------------------
echo
echo "=== 3. what CRAB would check ==="
FAIL=0

[ "$RC" -eq 0 ] && say_ok "scriptExe exit code 0" \
                || say_fail "scriptExe exit code $RC  (this is the code CRAB reports)"

# every file named in JobType.outputFiles must exist and be non-empty
for out in dimuon_ntuple.root; do
    if [ -s "$SANDBOX/$out" ]; then
        say_ok "$out present ($(du -h "$SANDBOX/$out" | cut -f1))"
    else
        say_fail "$out missing or empty -- CRAB would fail with 60302"
    fi
done

FJR="$SANDBOX/FrameworkJobReport.xml"
if [ -s "$FJR" ]; then
    if python3 -c "import xml.etree.ElementTree as ET, sys; ET.parse(sys.argv[1])" "$FJR" 2>/dev/null; then
        say_ok "FrameworkJobReport.xml parses"
        python3 - "$FJR" <<'PYEOF'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
tot  = root.findtext('File/TotalEvents', '0')
runs = root.findall('File/Runs/Run')
nlum = sum(len(r.findall('LumiSection')) for r in runs)
print('  [ ok ] report claims %s entries, %d run(s), %d lumi(s)' % (tot, len(runs), nlum))
if nlum == 0:
    print('  [warn] no lumis in the report -- crab report will show nothing '
          'processed. Is --lumi-json reaching the ntuplizer?')
PYEOF
    else
        say_fail "FrameworkJobReport.xml does not parse -- CRAB would fail with 50115"
    fi
else
    say_fail "no FrameworkJobReport.xml -- CRAB would fail with 50115"
fi

# the ntuple must actually be readable and have the branches downstream expects
if [ -s "$SANDBOX/dimuon_ntuple.root" ]; then
    python3 - "$SANDBOX/dimuon_ntuple.root" <<'PYEOF' || echo "  [FAIL] ntuple unreadable"
import sys, uproot
with uproot.open(sys.argv[1]) as f:
    t = f['tree']
    keys = set(t.keys())
    print('  [ ok ] tree: %d entries, %d branches' % (t.num_entries, len(keys)))
    need = ['run', 'lumi', 'event', 'mass', 'lxy', 'lxyz', 'cos2d', 'cos3d',
            'pv_refit_valid', 'vtx_cov_xx', 'pv_bs_cov_xx',
            'HLT_DoubleMu4_3_LowMass']
    missing = [b for b in need if b not in keys]
    if missing:
        print('  [FAIL] branches missing from the ntuple: %s' % missing)
        sys.exit(1)
    print('  [ ok ] every expected branch present')
    if t.num_entries:
        import numpy as np
        v = t['pv_refit_valid'].array(library='np')
        frac = float(np.mean(v > 0))
        tag = 'ok' if frac > 0.5 else 'warn'
        print('  [%s] pv_refit_valid = %.1f%% -- below ~50%% means the PV refit is '
              'falling back to the hybrid PV' % (tag.rjust(4), 100 * frac))
PYEOF
fi

echo
echo "=== summary ==="
echo "  wall time      : ${ELAPSED}s for $MAXEVENTS events"
echo "  log            : $SANDBOX/job.log"
if [ $FAIL -eq 0 ] && [ "$RC" -eq 0 ]; then
    echo "  VERDICT        : a grid job would have finished. Safe to submit."
else
    echo "  VERDICT        : a grid job would have FAILED. Do not submit yet."
fi

if [ $KEEP -eq 1 ]; then
    echo "  sandbox kept   : $SANDBOX"
else
    rm -rf "$SANDBOX"
fi

[ $FAIL -eq 0 ] && [ "$RC" -eq 0 ]
