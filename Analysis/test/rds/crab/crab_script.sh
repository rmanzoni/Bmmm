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
#   TRIGSF=<json>       MC: --trig-sf (trigger scale factors)
#   COVMUONLY=1         --cov-scale-muon-only (morph the muon only)
#   MAXEVENTS=<n>       test only: stop after n events (default all)
#   FITSCHEMES=<list>   RDS_FIT_SCHEMES, vertex-fit schemes computed/written
#                       (default free)
#
# provenance.json (written by the submitter: commit, options, sha256 of each
# correction file) travels in the sandbox and is printed at the start.
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
MAXEVENTS=-1
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
        TRIGSF=*)      EXTRA_FLAGS="${EXTRA_FLAGS} --trig-sf=${PWD}/${arg#*=}" ;;
        COVMUONLY=1)   EXTRA_FLAGS="${EXTRA_FLAGS} --cov-scale-muon-only" ;;
        MAXEVENTS=*)   MAXEVENTS="${arg#*=}" ;;
        FITSCHEMES=*)  export RDS_FIT_SCHEMES="${arg#*=}" ;;
    esac
done
echo ">>> output basename: ${OUTNAME}"
export RDS_FIT_SCHEMES="${RDS_FIT_SCHEMES:-free}"
echo ">>> RDS_FIT_SCHEMES : ${RDS_FIT_SCHEMES}"
if [ -f provenance.json ]; then
    echo ">>> provenance (from the submission):"
    cat provenance.json
else
    echo ">>> WARNING: provenance.json not in the sandbox"
fi

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

# --- where to read each input from (resolve_inputs.py) ------------------------
# CRAB runs the job at a site hosting the data (Data.ignoreLocality False and
# overflow off, see crab_rds_common.py), so the site's own storage comes
# first, from its site-local-config + storage.json (XRootD protocol), then the
# regional federation of the site-local-config, the global redirector last.
# Nov 2026: reading through cms-xrd-global only ("Redirect limit has been
# reached", then corrupted baskets) killed ~1100 jobs in one evening.
python3 resolve_inputs.py inputfiles.txt > sources.txt \
    || { echo ">>> FATAL: resolve_inputs.py failed for job $1"; exit 1; }
echo ">>> site config: ${SITECONFIG_PATH}"
echo ">>> sources per input (tried in order):"
cat sources.txt

# reachable: the file answers a stat within 2 min
reachable() {
    local url="$1"
    case "${url}" in
        root://*) ;;
        *) return 0 ;;
    esac
    local host="${url#root://}"; host="${host%%/*}"
    local path="${url#root://${host}/}"
    timeout 120 xrdfs "${host}" stat "/${path#/}" < /dev/null > /dev/null 2>&1
}

# --- one inspector process per input file, then merge ------------------------
# A fresh process per file keeps the memory of each one bounded and
# independent of files_per_job (RSS plateaus at ~2 GB, Nov 2026), and a crash
# names its file. A file that cannot be read, or whose inspector fails, is
# retried from the next source. The parts are merged with merge_ntuples.py
# (uproot, ZSTD 5, 25k-entry baskets; it checks entries and run/lumi/event
# sums against the parts).
echo ">>> launching inspector, one process per input file ..."
PARTS=""
K=0
while read -r SOURCES; do
    [ -z "${SOURCES}" ] && continue
    PART="${OUTNAME}_part${K}"
    OK=0
    RC=1
    for INF in ${SOURCES}; do
        if ! reachable "${INF}"; then
            echo ">>> [part ${K}] not reachable: ${INF}"
            continue
        fi
        echo ">>> [part ${K}] ${INF}   start $(date +%T)"
        rm -f "${PART}.root"
        # -u: unbuffered, so a crash traceback is fully flushed to the job log
        python3 -u inspector_rds.py \
            --inputFiles="${INF}" \
            --filename="${PART}" \
            --destination=. \
            --logfreq=1000 \
            --maxevents=${MAXEVENTS} \
            --logger="logger_${PART}" \
            ${EXTRA_FLAGS} < /dev/null
        RC=$?
        if [ ${RC} -eq 0 ] && [ -f "${PART}.root" ]; then
            OK=1
            break
        fi
        echo ">>> [part ${K}] inspector FAILED (exit ${RC}) on ${INF}, trying the next source"
    done
    if [ ${OK} -ne 1 ]; then
        echo ">>> inspector FAILED on every source of part ${K} for job $1: ${SOURCES}"
        ls -la
        exit $(( RC == 0 ? 1 : RC ))
    fi
    echo ">>> [part ${K}] done $(date +%T)"
    PARTS="${PARTS} ${PART}.root"
    K=$((K + 1))
done < sources.txt

# also for a single part: the rewrite turns the inspector's 5000-row baskets
# into 25k-row ones (same size, ~2.5x faster to read back)
python3 -u merge_ntuples.py --compression zstd:5 --step 25000 "${OUTNAME}.root" ${PARTS} \
    || { echo ">>> FATAL: merge_ntuples.py failed for job $1"; exit 1; }
rm -f ${PARTS}

if [ ! -f "${OUTNAME}.root" ]; then
    echo ">>> FATAL: ${OUTNAME}.root was not produced. Aborting job $1."
    exit 1
fi

echo ">>> done. output:"
ls -latrh "${OUTNAME}.root"
echo "======================== scriptExe finished ======================"
# FrameworkJobReport.xml is shipped via JobType.inputFiles and left untouched,
# which satisfies CRAB's bookkeeping for non-cmsRun jobs.
