#!/bin/bash
#
# Build pylibs/ with ONLY the packages CMSSW does not already provide.
#
# `pip install --target=pylibs particle uproot` is not safe on its own: pip
# resolves dependencies against PyPI, not against CMSSW, so it drags in numpy
# (2.0.2 at the time of writing) even though CMSSW ships one. Put that on
# PYTHONPATH and it shadows CMSSW's numpy, and CMSSW's scipy -- compiled
# against the 1.x ABI -- fails to import:
#
#     A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2
#
# which is precisely the error this whole exercise has been tripping over.
#
# So: install everything into a staging directory, then delete from it anything
# the CMSSW python can already import on its own. What survives is the genuine
# gap, and it runs against CMSSW's numpy rather than replacing it.
#
#   ./make_pylibs.sh            # build (or rebuild) pylibs/
#   ./make_pylibs.sh --check    # only report what CMSSW provides, change nothing
#   ./make_pylibs.sh --covflow /path/to/covflow-repo
#                               # also zuko (no dependencies: torch must come from
#                               # CMSSW) and the covflow package, for
#                               # crab_dimuon_mc_run3_covflow.py

set -o pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE" || exit 1

# Measured on t3ui07 / CMSSW_16_0_8 (./make_pylibs.sh --check):
#   CMSSW HAS      numpy 1.26.4, scipy 1.13.1, uproot 5.6.6, awkward 2.8.9,
#                  awkward_cpp, fsspec, packaging, attr, typing_extensions,
#                  importlib_metadata, zipp, xxhash, cramjam
#   CMSSW LACKS    particle, hepunits
# so only particle (and its hepunits dependency) actually ships: 921 KB. Ask for
# uproot anyway -- it costs nothing, the pruning drops it, and if a future
# release stops shipping it the gap is filled automatically.
WANT="particle uproot"

COVFLOW_REPO=""
if [ "$1" = "--covflow" ]; then
    COVFLOW_REPO="$2"
    if [ ! -f "${COVFLOW_REPO}/correct.py" ] || [ ! -f "${COVFLOW_REPO}/flows.py" ]; then
        echo "--covflow needs the covflow repository (correct.py, flows.py, ...); got '${COVFLOW_REPO}'" >&2
        exit 1
    fi
fi

if [ -z "$CMSSW_BASE" ]; then
    echo "CMSSW_BASE is not set -- run cmsenv first" >&2
    exit 1
fi

# What can the CMSSW python import with ~/.local and pylibs both shut out?
# That is exactly the worker node's situation.
echo "=== what CMSSW provides (with ~/.local shut out) ==="
PYTHONNOUSERSITE=1 PYTHONPATH= python3 - <<'PYEOF'
mods = ['numpy', 'scipy', 'uproot', 'awkward', 'awkward_cpp', 'particle',
        'hepunits', 'fsspec', 'packaging', 'attr', 'typing_extensions',
        'importlib_metadata', 'zipp', 'xxhash', 'cramjam', 'torch', 'zuko']
for m in mods:
    try:
        mod = __import__(m)
        print('  %-20s %s' % (m, getattr(mod, '__version__', 'present')))
    except Exception:
        print('  %-20s MISSING' % m)
PYEOF

if [ "$1" = "--check" ]; then
    exit 0
fi

echo
echo "=== installing $WANT into a staging directory ==="
rm -rf pylibs_staging pylibs
PYTHONNOUSERSITE=1 python3 -m pip install --no-cache-dir --quiet \
    --target=pylibs_staging $WANT || exit 1

echo
echo "=== pruning anything CMSSW already has ==="
# candidate top-level names: directories and single-file modules, minus the
# packaging metadata
KEPT=0
DROPPED=0
for entry in pylibs_staging/*; do
    base=$(basename "$entry")
    case "$base" in
        *.dist-info|*.egg-info|__pycache__|bin|*.libs) continue ;;
    esac
    mod="${base%.py}"

    # can CMSSW import it WITHOUT the staging directory?
    if PYTHONNOUSERSITE=1 PYTHONPATH= python3 -c "import $mod" 2>/dev/null; then
        echo "  drop  $mod  (CMSSW has it)"
        rm -rf "$entry"
        rm -rf pylibs_staging/${mod}-*.dist-info pylibs_staging/${mod}.libs
        DROPPED=$((DROPPED+1))
    else
        echo "  keep  $mod"
        KEPT=$((KEPT+1))
    fi
done
mv pylibs_staging pylibs
echo "  -> kept $KEPT, dropped $DROPPED"

if [ -n "${COVFLOW_REPO}" ]; then
    echo
    echo "=== covflow: zuko (no deps) and the covflow package from ${COVFLOW_REPO} ==="
    # --no-deps: zuko's dependency is torch, which must be CMSSW's (a pip torch
    # is hundreds of MB and would bring its own numpy)
    # pin with ZUKO_VERSION=x.y.z to the zuko_version of the covflow.json you
    # ship; crab_dimuon_mc_run3_covflow.py compares the two at submission
    ZUKO_VERSION="${ZUKO_VERSION:-}"
    PYTHONNOUSERSITE=1 python3 -m pip install --no-cache-dir --quiet --no-deps \
        --target=pylibs "zuko${ZUKO_VERSION:+==${ZUKO_VERSION}}" || exit 1
    mkdir -p pylibs/covflow
    cp "${COVFLOW_REPO}"/*.py pylibs/covflow/
    ( cd "${COVFLOW_REPO}" && git rev-parse HEAD 2>/dev/null ) > pylibs/covflow/COMMIT || true
    echo "  zuko $(ls -d pylibs/zuko-*.dist-info 2>/dev/null | sed 's/.*zuko-//;s/.dist-info//')," \
         "covflow $(cut -c1-10 pylibs/covflow/COMMIT 2>/dev/null)"
fi

echo
echo "=== does the job's import set work now? ==="
# same conditions as crab_script.sh: no ~/.local, pylibs first
if PYTHONNOUSERSITE=1 PYTHONPATH="$HERE/pylibs" python3 - <<'PYEOF'
import sys
bad = []
import os
mods = ['numpy', 'scipy', 'particle', 'uproot', 'ROOT']
if os.path.isdir('pylibs/covflow'):
    mods += ['torch', 'zuko', 'covflow.correct', 'covflow.flows']
for m in mods:
    try:
        mod = __import__(m)
        print('  [ ok ] %-12s %s' % (m, getattr(mod, '__version__', 'present')))
    except Exception as err:
        print('  [FAIL] %-12s %s: %s' % (m, type(err).__name__, err))
        bad.append(m)
if bad:
    sys.exit(1)
PYEOF
then
    echo
    echo "pylibs/ is ready: $(du -sh pylibs | cut -f1)"
else
    echo
    echo "STILL BROKEN -- do not submit. The failure above is what a worker node"
    echo "would hit. If numpy is the complaint, something in pylibs/ still"
    echo "shadows the CMSSW one; check with:  ls pylibs/ | head -40" >&2
    exit 1
fi
