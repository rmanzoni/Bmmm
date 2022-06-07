# Bmmm

## installation
```
cmsrel CMSSW_10_6_28
cd CMSSW_10_6_28/src
cmsenv
git init
git remote add origin git@github.com:rmanzoni/Bmmm.git
git fetch origin
git checkout main
scram b
```

## run

```
cd $CMSSW_BASE/src/Bmmm/Analysis/test
ipython -i -- inspector_bmmm_analysis.py --inputFiles="../../../../../rds/CMSSW_10_6_28/src/Bmmm/MINI/Bmmm_signal_MINI.root" --filename=signal --mc
```

