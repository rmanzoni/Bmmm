#!/bin/bash
#
# CRAB scriptExe wrapper for the R(Ds) ntuplizer (test/rds/inspector_rds.py).
#
# Same mechanism as test/bparking2018/crab/crab_script.sh (Vcb parking
# ntuples): CRAB runs this INSTEAD of cmsRun, after setting up CMSSW, writing
# this job's input LFNs into PSet.py and placing the proxy. CRAB passes the job
# id as $1, then JobType.scriptArgs (KEY=value):
#
#   LUMIMASK=<dir>      data: certification mask (mandatory for data)
#   MC=1                --mc
#   SAVENONTRIG=1       --savenontrig
#   TRKCALIB=<json>     MC: --trk-calib (momcal applier next to the JSON)
#   COVSCALE=<table>    --cov-scale
#   HAMMER=<card>       MC: --hammer
#
# BMMM_DATADIR points at the job directory, where the merged L1 table,
# l1menus/ and trkcalib/ were shipped.

set -o pipefail

echo "==================== R(Ds) (2018 B-parking) scriptExe ===================="
echo ">>> job id   : $1"
echo ">>> host     : $(hostname)"
echo ">>> pwd      : $(pwd)"
echo ">>> proxy    : ${X509_USER_PROXY}"
echo ">>> python3  : $(which python3)"
echo ">>> CMSSW    : ${CMSSW_BASE}"
echo ">>> files in working dir:"
ls -la
echo "=================================================================="

# --- non-CMSSW python packages (particle) shipped in ./pylibs ----------------
export PYTHONNOUSERSITE=1
if [ -d pylibs ]; then
    export PYTHONPATH="${PWD}/pylibs:${PYTHONPATH}"
    echo ">>> added $(pwd)/pylibs to PYTHONPATH"
    # a numpy/scipy in pylibs shadows the CMSSW one (NumPy 1.x/2.x ABI clash)
    for shadowed in numpy scipy; do
        if [ -d "pylibs/${shadowed}" ]; then
            echo ">>> FATAL: pylibs/${shadowed} shadows the CMSSW one."
            echo ">>>        rebuild pylibs with make_pylibs.sh and resubmit."
            exit 1
        fi
    done
else
    echo ">>> FATAL: pylibs/ not present in working dir (particle is needed). Aborting job $1."
    exit 1
fi

# --- rebuild the Bmmm.Analysis package from the shipped ./python -------------
# the inspector imports the PACKAGE (from Bmmm.Analysis.RDs... import ...);
# CRAB ships src/Bmmm/Analysis/python as ./python/, turned back into an
# importable tree here so the import does not depend on CRAB internals.
if [ -d python ]; then
    mkdir -p bmmmpkg/Bmmm/Analysis
    touch bmmmpkg/Bmmm/__init__.py bmmmpkg/Bmmm/Analysis/__init__.py
    cp python/*.py bmmmpkg/Bmmm/Analysis/ 2>/dev/null
    export PYTHONPATH="${PWD}/bmmmpkg:${PYTHONPATH}"
    echo ">>> rebuilt Bmmm/Analysis from ./python ($(ls python/*.py | wc -l) modules)"
else
    echo ">>> FATAL: ./python not present in working dir. Aborting job $1."
    exit 1
fi

# --- say plainly whether the imports resolve, BEFORE the event loop ----------
echo ">>> import check:"
python3 -c "
import Bmmm.Analysis.RDsInspector, Bmmm.Analysis.RDsCandidate
import particle, uproot, numpy, scipy
print('    all imports OK')
" || { echo ">>> FATAL: imports failed, aborting job $1"; ls -la; exit 1; }

# --- KEY=value arguments from JobType.scriptArgs -----------------------------
export BMMM_DATADIR="${PWD}"
echo ">>> BMMM_DATADIR=${BMMM_DATADIR}"
ls -la l1menus l1_seed_prescales_bparking2018.pickle.gz || { echo ">>> FATAL: L1 inputs missing"; exit 1; }

OUTNAME="rds_ntuple"
LUMIMASK=""
IS_MC=0
EXTRA_FLAGS=""
for arg in "$@"; do
    case "${arg}" in
        OUTNAME=*)     OUTNAME="${arg#*=}" ;;
        LUMIMASK=*)    LUMIMASK="${arg#*=}" ;;
        MC=1)          IS_MC=1 ; EXTRA_FLAGS="${EXTRA_FLAGS} --mc" ;;
        SAVENONTRIG=1) EXTRA_FLAGS="${EXTRA_FLAGS} --savenontrig" ;;
        TRKCALIB=*)    EXTRA_FLAGS="${EXTRA_FLAGS} --trk-calib=${PWD}/${arg#*=}" ;;
        COVSCALE=*)    EXTRA_FLAGS="${EXTRA_FLAGS} --cov-scale=${PWD}/${arg#*=}" ;;
        HAMMER=*)      EXTRA_FLAGS="${EXTRA_FLAGS} --hammer=${PWD}/${arg#*=}" ;;
    esac
done
echo ">>> output basename: ${OUTNAME}"

if [ "${IS_MC}" -eq 1 ]; then
    # simulation: no certification. A mask here is a submitter bug.
    if [ -n "${LUMIMASK}" ]; then
        echo ">>> FATAL: LUMIMASK given together with MC=1. Aborting job $1."
        exit 1
    fi
    echo ">>> MC job: no lumi mask"
else
    # Certification mask, applied EVENT BY EVENT by the inspector. The CRAB-side
    # Data.lumiMask only decides which FILES this job gets: the inspector reads
    # the PSet fileNames, never lumisToProcess, so without --lumi-mask a file
    # with one certified lumi would be ntuplized whole.
    if [ -z "${LUMIMASK}" ] || [ ! -e "${LUMIMASK}" ]; then
        echo ">>> FATAL: LUMIMASK='${LUMIMASK}' missing or not in the job directory. Aborting job $1."
        ls -la
        exit 1
    fi
    EXTRA_FLAGS="${EXTRA_FLAGS} --lumi-mask=${PWD}/${LUMIMASK}"
    echo ">>> lumi mask      : ${PWD}/${LUMIMASK}"
    ls -la "${LUMIMASK}"
fi
echo ">>> extra flags    : ${EXTRA_FLAGS}"

# --- this job's input LFNs, from the CRAB-tweaked PSet ------------------------
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

# --- comma-separated --inputFiles, bare LFNs -> xrootd URLs ------------------
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
        out.append(f)
    elif f.startswith('/store/'):
        out.append(red + f)
    else:
        out.append(f)
print(','.join(out))
PYEOF
)

echo ">>> launching inspector ..."
# -u: unbuffered, so a crash traceback is fully flushed to the job log
python3 -u inspector_rds.py \
    --inputFiles="${INFILES}" \
    --filename="${OUTNAME}" \
    --destination=. \
    --logfreq=1000 \
    --maxevents=-1 \
    ${EXTRA_FLAGS}
RC=$?

if [ ${RC} -ne 0 ]; then
    echo ">>> inspector FAILED (exit ${RC}) for job $1"
    ls -la
    exit ${RC}
fi

if [ ! -f "${OUTNAME}.root" ]; then
    echo ">>> FATAL: ${OUTNAME}.root was not produced. Aborting job $1."
    exit 1
fi

echo ">>> done. output:"
ls -latrh "${OUTNAME}.root"
echo "======================== scriptExe finished ======================"
# FrameworkJobReport.xml is shipped via JobType.inputFiles and left untouched,
# which satisfies CRAB's bookkeeping for non-cmsRun jobs.
