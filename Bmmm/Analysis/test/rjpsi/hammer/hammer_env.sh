#!/bin/bash
# Put Hammer on the path of an already-cmsenv'd shell (or a SLURM job script).
#
#   cmsenv
#   source $CMSSW_BASE/src/Bmmm/Analysis/test/rjpsi/hammer/hammer_env.sh
#
# install_hammer.sh writes the authoritative copy of this next to the install,
# with the Python version resolved; this one just forwards to it so that jobs
# only ever need a path inside the repo.
HAMMER_PREFIX=${HAMMER_PREFIX:-/work/manzoni/hammer}

if [ -f "$HAMMER_PREFIX/hammer_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$HAMMER_PREFIX/hammer_env.sh"
else
    echo "[WARN] $HAMMER_PREFIX/hammer_env.sh not found -- run install_hammer.sh" >&2
    pyver=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    export PYTHONPATH=$HAMMER_PREFIX/lib64/python${pyver}/site-packages:$HAMMER_PREFIX/lib/python${pyver}/site-packages:${PYTHONPATH:-}
    export LD_LIBRARY_PATH=$HAMMER_PREFIX/lib64:$HAMMER_PREFIX/lib:${LD_LIBRARY_PATH:-}
fi

python3 "$(dirname "${BASH_SOURCE[0]:-$0}")/check_hammer_api.py"
