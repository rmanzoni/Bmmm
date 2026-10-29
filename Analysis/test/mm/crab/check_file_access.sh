#!/bin/bash
#
# Which way of naming an input file can FWLite actually open?
#
# This exists because the answer is not obvious and should not be guessed. A
# cmsRun job never faces the question: CRAB writes bare LFNs into
# process.source.fileNames and CMSSW's PoolSource resolves them itself, via the
# worker node's site-local catalogue with an xrootd fallback. A scriptExe job
# runs FWLite instead, and FWLite's Events() hands its strings much closer to
# TFile::Open -- so whether a bare LFN works depends on whether the storage
# factory picks it up, which depends on the environment. That is a question
# about this release on this machine, so measure it rather than reason about it.
#
# Takes a REAL file from the dataset (via dasgoclient -- no invented GUIDs) and
# tries each form in turn, reporting which open and how fast.
#
#   ./check_file_access.sh
#   ./check_file_access.sh --dataset /ParkingDoubleMuonLowMass1/Run2022D-PromptReco-v1/MINIAOD
#
# Run it with cmsenv done and a valid proxy. Whatever wins here is what
# crab_script.sh should use -- and if the bare LFN wins, crab_script.sh can
# stop rewriting paths altogether.

set -o pipefail

DATASET='/ParkingDoubleMuonLowMass1/Run2022C-PromptReco-v1/MINIAOD'
LFN=''

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset) DATASET=$2; shift 2 ;;
        --lfn)     LFN=$2    ; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

# ------------------------------------------------------------------------------
# A real file, asked for rather than assumed.
# ------------------------------------------------------------------------------
if [ -z "$LFN" ]; then
    command -v dasgoclient >/dev/null || {
        echo "dasgoclient not on PATH -- run cmsenv, or pass --lfn" >&2; exit 1; }
    echo "asking DAS for a file in $DATASET ..."
    LFN=$(dasgoclient -query "file dataset=$DATASET" -limit 1 2>/dev/null | head -1)
    [ -n "$LFN" ] || { echo "DAS returned no file for $DATASET" >&2; exit 1; }
fi
echo "file: $LFN"
echo

# where DAS says the replicas are: if no site near you has it, a slow open is
# the network, not the naming scheme, and that is worth knowing before judging
if command -v dasgoclient >/dev/null; then
    echo "replicas:"
    dasgoclient -query "site file=$LFN" 2>/dev/null | sed 's/^/  /' | head -10
    echo
fi

# ------------------------------------------------------------------------------
# The candidates, in the order crab_script.sh should prefer them.
# ------------------------------------------------------------------------------
try_open() {
    local label="$1" url="$2"
    local start=$(date +%s%N)
    if python3 - "$url" >/dev/null 2>&1 <<'PYEOF'
import sys, ROOT
ROOT.gErrorIgnoreLevel = ROOT.kFatal
f = ROOT.TFile.Open(sys.argv[1])
if not f or f.IsZombie():
    sys.exit(1)
if not f.Get('Events'):
    sys.exit(2)          # opened, but not a CMSSW file
f.Close()
PYEOF
    then
        printf '  [ ok ] %-22s %5d ms   %s\n' "$label" \
               $(( ($(date +%s%N) - start) / 1000000 )) "$url"
        return 0
    fi
    printf '  [FAIL] %-22s %5d ms   %s\n' "$label" \
           $(( ($(date +%s%N) - start) / 1000000 )) "$url"
    return 1
}

echo "TFile::Open, each naming scheme:"
try_open "bare LFN"        "$LFN"                              && BARE=1     || BARE=0
try_open "site TFC"        "$(edmFileUtil -d "$LFN" 2>/dev/null)" && TFC=1   || TFC=0
try_open "regional AAA"    "root://xrootd-cms.infn.it/$LFN"    && REGIONAL=1 || REGIONAL=0
try_open "global AAA"      "root://cms-xrd-global.cern.ch/$LFN" && GLOBAL=1  || GLOBAL=0

# and the one that matters: can FWLite itself count the events?
echo
echo "FWLite Events(), on the first scheme that opened:"
for URL in "$LFN" "root://xrootd-cms.infn.it/$LFN" "root://cms-xrd-global.cern.ch/$LFN"; do
    if python3 - "$URL" 2>/dev/null <<'PYEOF'
import sys
from DataFormats.FWLite import Events
ev = Events([sys.argv[1]])
print('  [ ok ] FWLite opened it: %d events' % ev.size())
PYEOF
    then
        echo "         via: $URL"
        break
    else
        echo "  [FAIL] FWLite could not use: $URL"
    fi
done

echo
echo "=== what this means for crab_script.sh ==="
if [ "$BARE" = 1 ]; then
    echo "  The bare LFN opens. FWLite is picking up the site catalogue through"
    echo "  the storage factory, so crab_script.sh can pass LFNs straight through"
    echo "  and let CMSSW resolve them -- exactly as a cmsRun job would, and with"
    echo "  the site-local read and the AAA fallback both for free. Prefer that."
elif [ "$REGIONAL" = 1 ]; then
    echo "  The bare LFN does NOT open, so the path rewriting is genuinely needed."
    echo "  The regional redirector works: keep the current default."
elif [ "$GLOBAL" = 1 ]; then
    echo "  Only the global redirector works from here."
    echo "  Set BMMM_REDIRECTOR=cms-xrd-global.cern.ch."
else
    echo "  Nothing opened. Check the proxy (voms-proxy-info) and whether this"
    echo "  file has any replica left (the 'replicas' list above) before"
    echo "  concluding anything about the naming scheme."
fi
[ "$TFC" = 1 ] || echo "  (the site TFC path does not open here, as expected off-site --"
[ "$TFC" = 1 ] || echo "   edmFileUtil -d rewrites the LFN with local rules, it does not"
[ "$TFC" = 1 ] || echo "   check that a replica exists)"
