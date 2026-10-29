#!/bin/tcsh
# tcsh version of hammer_env.sh -- the login shell on t3ui is tcsh, where
# sourcing the bash one dies with "export: Command not found."
#
#   cmsenv
#   source $CMSSW_BASE/src/Bmmm/Analysis/test/rjpsi/hammer/hammer_env.csh
#
# SLURM job scripts are #!/bin/bash and should source hammer_env.sh instead.
# Both just forward to the copy install_hammer.sh generated next to the install,
# which has the real module path baked in -- this file never guesses it.
if (! $?HAMMER_PREFIX) setenv HAMMER_PREFIX /work/manzoni/hammer

if (-f ${HAMMER_PREFIX}/hammer_env.csh) then
    source ${HAMMER_PREFIX}/hammer_env.csh
else
    echo "[FATAL] ${HAMMER_PREFIX}/hammer_env.csh not found -- run install_hammer.sh"
    echo "        (it writes that file with the installed module path in it)"
endif

# $0 is not the script path in a sourced tcsh file, so locate the checker
# through the release rather than relative to this file.
if ($?CMSSW_BASE) then
    python3 ${CMSSW_BASE}/src/Bmmm/Analysis/test/rjpsi/hammer/check_hammer_api.py
else
    echo "[WARN] CMSSW_BASE unset: run cmsenv, then check with check_hammer_api.py"
endif
