#!/bin/bash
#
# CRAB scriptExe wrapper for the dimuon inspector.
#
# A transliteration of test/tau3mu/crab/crab_script.sh, which works. The only
# additions are the two marked DIFFERENT, both forced by the dimuon inspector.
#
# CRAB runs this *instead of* cmsRun, after:
#   - setting up the CMSSW environment (cmsenv already done -> python3 is the
#     CMSSW python, FWLite is importable),
#   - tweaking PSet.py so that process.source.fileNames holds *this* job's
#     slice of the input dataset (LFNs, e.g. /store/data/...),
#   - placing a valid grid proxy in $X509_USER_PROXY.
#
# CRAB passes the job id as $1.
#
# Because locality scheduling is left ON (we do NOT set Data.ignoreLocality),
# CRAB sends each job to a site that hosts the data, so the redirector below
# resolves to the LOCAL xrootd door -> LAN read, not a WAN read.

set -o pipefail

echo "=================== dimuon scriptExe ==================="
echo ">>> job id   : $1"
echo ">>> host     : $(hostname)"
echo ">>> pwd      : $(pwd)"
echo ">>> proxy    : ${X509_USER_PROXY}"
echo ">>> python3  : $(which python3)"
echo ">>> CMSSW    : ${CMSSW_BASE}"
echo ">>> files in working dir:"
ls -la
echo "========================================================"

# non-CMSSW python packages (particle, uproot) shipped in ./pylibs/ via
# JobType.inputFiles. Prepend them to PYTHONPATH. PYTHONNOUSERSITE guards
# against accidentally importing from any stray ~/.local on the WN.
export PYTHONNOUSERSITE=1
if [ -d pylibs ]; then
    export PYTHONPATH="${PWD}/pylibs:${PYTHONPATH}"
    echo ">>> added $(pwd)/pylibs to PYTHONPATH"
    # a numpy inside pylibs shadows the CMSSW one, and CMSSW's scipy is
    # compiled against the 1.x ABI. Say so here rather than let it surface as
    # an opaque ImportError.
    for shadowed in numpy scipy; do
        if [ -d "pylibs/${shadowed}" ]; then
            echo ">>> FATAL: pylibs/${shadowed} shadows the CMSSW one."
            echo ">>>        rebuild pylibs with make_pylibs.sh and resubmit."
            exit 1
        fi
    done
else
    echo ">>> WARNING: pylibs/ not present in working dir"
fi

# --- DIFFERENT (1): rebuild the Bmmm.Analysis package ------------------------
# inspector_mm_analysis.py imports the PACKAGE, not flattened siblings:
#     from Bmmm.Analysis.MuMuBranches import ...
# CRAB ships src/Bmmm/Analysis/python as ./python/. Turn it back into an
# importable Bmmm/Analysis tree here rather than relying on sendPythonFolder
# having landed where python expects it -- this way the import cannot depend on
# CRAB internals.
if [ -d python ]; then
    mkdir -p bmmmpkg/Bmmm/Analysis
    touch bmmmpkg/Bmmm/__init__.py bmmmpkg/Bmmm/Analysis/__init__.py
    cp python/*.py bmmmpkg/Bmmm/Analysis/ 2>/dev/null
    export PYTHONPATH="${PWD}/bmmmpkg:${PYTHONPATH}"
    echo ">>> rebuilt Bmmm/Analysis from ./python ($(ls python/*.py | wc -l) modules)"
else
    echo ">>> WARNING: ./python not present -- relying on sendPythonFolder alone"
fi

# --- DIFFERENT (2): where the inspector looks for the L1 menus ---------------
# it reads $CMSSW_BASE/src/Bmmm/Analysis/data/l1menus at import; the sandbox
# carries l1menus/ in the job directory instead.
if [ -d l1menus ]; then
    export BMMM_DATADIR="${PWD}"
    echo ">>> BMMM_DATADIR=${PWD} (l1menus/ present)"
else
    echo ">>> WARNING: l1menus/ not present in working dir"
fi

# say plainly whether the imports resolve, BEFORE the event loop, so a failure
# here is one line in the log rather than a traceback with no context
echo ">>> import check:"
python3 -c "
import Bmmm.Analysis.MuMuBranches, Bmmm.Analysis.MuMuCandidate
import particle, uproot, numpy, scipy
print('    all imports OK')
" || { echo ">>> FATAL: imports failed, aborting job $1"; ls -la; exit 1; }

OUTNAME="dimuon_ntuple"
# Extra inspector flags, passed by the submitter through JobType.scriptArgs.
# Empty by default, so a data job runs exactly as before this was added --
# the MC submitter sets MC=1 SAVENONTRIG=1.
EXTRA_FLAGS=""
for arg in "$@"; do
    case "${arg}" in
        OUTNAME=*)     OUTNAME="${arg#*=}" ;;
        MC=1)          EXTRA_FLAGS="${EXTRA_FLAGS} --mc" ;;
        SAVENONTRIG=1) EXTRA_FLAGS="${EXTRA_FLAGS} --savenontrig" ;;
    esac
done
echo ">>> output basename: ${OUTNAME}"
echo ">>> extra flags    : ${EXTRA_FLAGS:-<none>}"

# --- pull this job's input LFNs out of the CRAB-tweaked PSet ---
python3 - > inputfiles.txt <<'PYEOF'
import PSet
for f in PSet.process.source.fileNames:
    print(f)
PYEOF

NFILES=$(grep -c . inputfiles.txt)
echo ">>> this job has ${NFILES} input file(s):"
cat inputfiles.txt

if [ "${NFILES}" -eq 0 ]; then
    echo ">>> FATAL: no input files in tweaked PSet. Aborting job $1."
    exit 1
fi

# --- build comma-separated --inputFiles string, prefixing bare LFNs ---
REDIRECTOR="root://cms-xrd-global.cern.ch/"

INFILES=$(python3 - "${REDIRECTOR}" <<'PYEOF'
import sys
red = sys.argv[1]
out = []
for line in open('inputfiles.txt'):
    f = line.strip()
    if not f:
        continue
    if f.startswith('root://') or f.startswith('file:'):
        out.append(f)                 # already a usable PFN
    elif f.startswith('/store/'):
        out.append(red + f)           # LFN -> xrootd URL
    else:
        out.append(f)
print(','.join(out))
PYEOF
)

echo ">>> launching inspector ..."
# -u: unbuffered, so a crash traceback is fully flushed to the job log.
python3 -u inspector_mm_analysis.py \
    --inputFiles="${INFILES}" \
    --filename="${OUTNAME}" \
    --destination=. \
    --logfreq=5000 \
    --maxevents=-1 \
    ${EXTRA_FLAGS}
RC=$?

if [ ${RC} -ne 0 ]; then
    echo ">>> inspector FAILED (exit ${RC}) for job $1"
    echo ">>> working dir at failure:"
    ls -la
    exit ${RC}
fi

if [ ! -f "${OUTNAME}.root" ]; then
    echo ">>> FATAL: ${OUTNAME}.root was not produced. Aborting job $1."
    exit 1
fi

echo ">>> done. output:"
ls -latrh "${OUTNAME}.root"
echo "=================== scriptExe finished ==================="
# NB: FrameworkJobReport.xml is shipped via JobType.inputFiles and left
# untouched, which satisfies CRAB's bookkeeping for non-cmsRun jobs.
