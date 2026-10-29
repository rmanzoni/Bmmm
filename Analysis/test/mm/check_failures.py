import re
import subprocess

# nchunks = len(chunks)   # same value as in your submitter


# nchunks = 261 # 2018C
nchunks = 1127 # 2018D


out = subprocess.check_output(
    "xrdfs root://t3dcachedb03.psi.ch ls "
    "/pnfs/psi.ch/cms/trivcat/store/user/manzoni/Charmonium_Run2018D-UL2018_MiniAODv2_GT36-v1_19May2026_v1",
    shell=True,
    text=True,
)

done = set(
    int(re.search(r'chunk(\d+)\.root', line).group(1))
    for line in out.splitlines()
    if 'chunk' in line
)

missing = sorted(set(range(nchunks)) - done)

print(missing)




# [manzoni@t3ui07 test]$ lll Charmonium_Run2018C-UL2018_MiniAODv2_GT36-v1_19May2026_v1 | grep submitter | wc
#     261    2349   21814
#     
# (hammer3p8) [manzoni@t3ui07 test]$ lll /pnfs/psi.ch/cms/trivcat/store/user/manzoni/Charmonium_Run2018C-UL2018_MiniAODv2_GT36-v1_19May2026_v1 | grep root | wc
#     259    2331   19319
# [159, 259]
# 
# [manzoni@t3ui07 test]$ lll Charmonium_Run2018D-UL2018_MiniAODv2_GT36-v1_19May2026_v1 | grep submitter | wc
#    1127   10143   94685    
# (hammer3p8) [manzoni@t3ui07 test]$ lll /pnfs/psi.ch/cms/trivcat/store/user/manzoni/Charmonium_Run2018D-UL2018_MiniAODv2_GT36-v1_19May2026_v1 | grep root | wc
#    1099    9891   82444
# [110, 195, 252, 261, 405, 411, 414, 419, 612, 623, 635, 683, 697, 717, 750, 756, 769, 781, 957, 968, 972, 975, 985, 992, 996, 998, 1004, 1008]


