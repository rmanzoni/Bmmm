#!/bin/bash
#
# What CRAB actually executes on the worker node (JobType.scriptExe).
#
# CRAB has already unpacked the sandbox and set up the CMSSW environment by the
# time this runs, and has rewritten PSet.py with this job's share of the input
# dataset. CRAB passes the job number as $1.
#
# The one job of this script is to turn those input files into something FWLite
# can open, and to run the ntuplizer on them.

set -e
set -o pipefail

JOBID=${1:-0}

echo "=================================================================="
echo " dimuon ntuple, CRAB job ${JOBID}"
echo " host    : $(hostname)"
echo " date    : $(date)"
echo " pwd     : $(pwd)"
echo " CMSSW   : ${CMSSW_BASE}"
echo "=================================================================="

# Only the Run 3 low-mass dimuon path. Read by MuMuBranches at import, before
# the branch list is built, so the ntuple carries this path's decision,
# prescale and tag-and-probe branches and not ~200 columns of 2018 paths that
# no Run 3 menu contains. Must be exported BEFORE python starts.
export BMMM_MM_HLT_PATHS=HLT_DoubleMu4_3_LowMass

# ------------------------------------------------------------------------------
# Resolve this job's input files to site-local PFNs.
#
# CRAB writes LFNs (/store/...) into PSet.py, and FWLite needs a real URL.
# edmFileUtil -d asks the site's file catalogue where the replica actually is,
# so a job scheduled at a site holding the data reads it off local storage
# rather than pulling it across the wide area. That is the point of using CRAB
# here at all -- xrootd across the WAN is the flaky part. If the catalogue
# cannot answer, fall back to the global redirector for that file rather than
# failing the whole job.
# ------------------------------------------------------------------------------
python3 - > lfns.txt <<'PYEOF'
from PSet import process
for f in process.source.fileNames:
    print(f)
PYEOF

echo "--- input files for this job ---"
: > pfns.txt
while read -r LFN; do
    [ -z "$LFN" ] && continue
    if PFN=$(edmFileUtil -d "$LFN" 2>/dev/null) && [ -n "$PFN" ]; then
        echo "  local : $PFN"
    else
        PFN="root://cms-xrd-global.cern.ch/${LFN}"
        echo "  AAA   : $PFN   (edmFileUtil could not resolve it)"
    fi
    echo "$PFN" >> pfns.txt
done < lfns.txt

INFILES=$(paste -sd, pfns.txt)
if [ -z "$INFILES" ]; then
    echo "ERROR: no input files resolved for this job" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# Run it. The output name is fixed and must match JobType.outputFiles; CRAB
# appends the job id on the storage side, so jobs cannot collide.
# ------------------------------------------------------------------------------
python3 inspector_mm_analysis.py \
    --inputFiles="${INFILES}" \
    --filename=dimuon_ntuple  \
    --destination=.           \
    --logfreq=5000

echo "--- produced ---"
ls -l dimuon_ntuple.root

echo "job ${JOBID} done: $(date)"
