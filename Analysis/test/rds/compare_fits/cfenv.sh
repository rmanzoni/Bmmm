# source cfenv.sh (bash or zsh): the CMSSW python with its own numpy 1.x, so
# that the CMSSW scikit-learn imports. A numpy 2 installed with pip --user
# shadows the CMSSW numpy and breaks scikit-learn; PYTHONNOUSERSITE=1 hides it
# (uproot, scipy, matplotlib come from CMSSW too).
_cf_here=$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)
[ -n "$CMS_PATH" ] || source /cvmfs/cms.cern.ch/cmsset_default.sh
pushd "$_cf_here" > /dev/null
# the scram launcher occasionally fails to find python ("in: command not
# found") and prints nothing to eval: retry once, then say so
for _cf_try in 1 2; do
    eval `scramv1 runtime -sh`
    case "$(command -v python3)" in /cvmfs/*) break ;; esac
    sleep 2
done
popd > /dev/null
case "$(command -v python3)" in
    /cvmfs/*) export PYTHONNOUSERSITE=1 ;;
    *) echo "cfenv.sh: cmsenv failed, python3 is $(command -v python3)" >&2 ;;
esac
unset _cf_here _cf_try
