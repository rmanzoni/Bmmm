#!/bin/bash
#
# What CRAB actually executes on the worker node (JobType.scriptExe).
#
# CRAB has already unpacked the sandbox and set up the CMSSW environment by the
# time this runs, and has told this job which files to process. CRAB passes the
# job number as $1.
#
# Three things happen here: work out this job's input files, turn them into
# something FWLite can open, and run the ntuplizer -- then write the
# FrameworkJobReport CRAB insists on reading afterwards.

set -o pipefail

JOBID=${1:-0}

echo "=================================================================="
echo " dimuon ntuple, CRAB job ${JOBID}"
echo " host    : $(hostname)"
echo " date    : $(date)"
echo " pwd     : $(pwd)"
echo " CMSSW   : ${CMSSW_BASE}"
echo "=================================================================="
ls -l

# Only the Run 3 low-mass dimuon path. Read by MuMuBranches at import, before
# the branch list is built, so the ntuple carries this path's decision,
# prescale and tag-and-probe branches and not ~200 columns of 2018 paths that
# no Run 3 menu contains. Must be exported BEFORE python starts.
export BMMM_MM_HLT_PATHS=HLT_DoubleMu4_3_LowMass

OUTFILE=dimuon_ntuple.root

# ------------------------------------------------------------------------------
# 1. This job's input files.
#
# Recent CRAB writes them as a JSON list in job_input_file_list_<jobid>.txt and
# that is the authoritative source; PSet.py is the older route and is kept as a
# fallback (in the sandbox it is a stub that unpickles PSet.pkl, which is why it
# has to be imported rather than read). Same order of preference as
# NanoAODTools' crabhelper.
# ------------------------------------------------------------------------------
: > lfns.txt
if [ -f "job_input_file_list_${JOBID}.txt" ]; then
    echo "--- inputs from job_input_file_list_${JOBID}.txt ---"
    python3 - "job_input_file_list_${JOBID}.txt" > lfns.txt <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    entries = json.load(f)
for e in entries:
    # entries are plain strings in every CRAB version seen so far, but a dict
    # with a 'lfn' key has appeared; handle both rather than crash on one
    print(e['lfn'] if isinstance(e, dict) else e)
PYEOF
else
    echo "--- inputs from PSet.py ---"
    python3 - > lfns.txt <<'PYEOF'
from PSet import process
for f in process.source.fileNames:
    print(f)
PYEOF
fi

if [ ! -s lfns.txt ]; then
    echo "ERROR: could not determine this job's input files" >&2
    ls -l
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Turn the LFNs into something FWLite can open, and PROVE each one opens.
#
# Measured, not assumed -- check_file_access.sh on t3ui07 against a real 2022C
# file from DAS:
#
#     bare LFN      FAIL      FWLite Events()  FAIL
#     site TFC      FAIL      (edmFileUtil -d, an off-site fiction)
#     regional AAA  ok        FWLite Events()  ok, 16156 events
#     global AAA    ok
#
# So the rewriting is needed: FWLite does not resolve a bare LFN the way
# PoolSource does for a cmsRun job, which is the one real cost of the scriptExe
# route. The regional door is the right default.
#
# resolve_pfns.py then opens every file before the event loop starts and falls
# through to the next door if one does not answer -- xrootd being sometimes
# unreliable is why this campaign runs on CRAB at all, and a job should not die
# because one door had a bad minute when another serves the same file.
# BMMM_DOORS overrides the list and its order.
# ------------------------------------------------------------------------------
DOORS=${BMMM_DOORS:-xrootd-cms.infn.it,cms-xrd-global.cern.ch}

echo "--- resolving this job's input files (doors: ${DOORS}) ---"
python3 resolve_pfns.py --lfns lfns.txt --out pfns.txt --doors "${DOORS}"
if [ $? -ne 0 ]; then
    echo "ERROR: could not open this job's input files through any door" >&2
    # still write a report, so CRAB shows THIS failure rather than BadFWJRXML
    python3 make_fjr.py --output "${OUTFILE}" --inputs lfns.txt || true
    exit 65
fi

INFILES=$(paste -sd, pfns.txt)
if [ -z "$INFILES" ]; then
    echo "ERROR: no input files resolved for this job" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# 3. Run it. The output name is fixed and must match JobType.outputFiles; CRAB
#    appends the job id on the storage side, so jobs cannot collide.
#
#    Deliberately NOT under `set -e`: if the ntuplizer dies we still want to
#    write a job report, so that CRAB shows the real exit code instead of
#    burying it under BadFWJRXML.
# ------------------------------------------------------------------------------
python3 inspector_mm_analysis.py       \
    --inputFiles="${INFILES}"          \
    --filename=dimuon_ntuple           \
    --destination=.                    \
    --lumi-json=processed_lumis.json   \
    --logfreq=5000
RC=$?
echo "ntuplizer exit code: ${RC}"

# ------------------------------------------------------------------------------
# 4. The FrameworkJobReport.
#
# CRAB's post-job parses FrameworkJobReport.xml whatever the job ran. cmsRun
# writes one; a scriptExe does not, and without it every job fails with
#     exit code 50115 : BadFWJRXML
# regardless of whether the analysis worked. Write it here, from the ntuple and
# the run/lumi map the ntuplizer just produced.
# ------------------------------------------------------------------------------
echo "--- produced ---"
ls -l "${OUTFILE}" processed_lumis.json 2>&1

python3 make_fjr.py               \
    --output    "${OUTFILE}"      \
    --inputs    pfns.txt          \
    --lumi-json processed_lumis.json

echo "--- FrameworkJobReport.xml ---"
cat FrameworkJobReport.xml

echo "job ${JOBID} done: $(date), exit ${RC}"
exit ${RC}
