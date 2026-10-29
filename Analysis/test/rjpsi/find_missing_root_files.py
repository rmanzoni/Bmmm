#!/usr/bin/env python3

import re
import os
from pathlib import Path

submitter_dir = Path("/work/manzoni/rjpsi_run3/CMSSW_15_1_1/src/Bmmm/Analysis/test/rjpsi/RJpsi_23Jun2026_notrig_data2024_partial_v1")
root_dir = Path("/pnfs/psi.ch/cms/trivcat/store/user/manzoni/RJpsi_23Jun2026_notrig_data2024_partial_v1")

submitter_pattern = re.compile(r"submitter_chunk(\d+)\.sh$")
root_pattern = re.compile(r"rjpsi_chunk(\d+)\.root$")

# Extract chunk indices from submitter scripts
submitter_indices = {
    int(m.group(1))
    for f in submitter_dir.iterdir()
    if (m := submitter_pattern.match(f.name))
}

# Extract chunk indices from ROOT files
root_indices = {
    int(m.group(1))
    for f in root_dir.iterdir()
    if (m := root_pattern.match(f.name))
}

# Chunks for which a submitter exists but the ROOT file is missing
missing_indices = sorted(submitter_indices - root_indices)

print(f"Found {len(missing_indices)} missing ROOT files")
print("Missing chunk indices:")
print(missing_indices)


##################################
##################################
##################################

queue = 'standard'; time = 720
# queue = 'short'   ; time = 60
# queue = 'long'    ; time = 10080


for jobid in missing_indices:

    command_sh_batch = ' '.join([
        'sbatch',
        '-p %s'%queue,
        '--account=t3',
        '-o %s/logs/chunk%d.log' %(submitter_dir, jobid),
        '-e %s/errs/chunk%d.err' %(submitter_dir, jobid),
        '--job-name=%d_%s' %(jobid, submitter_dir),
        '--time=%d'%time,
        '--nodes=1 --ntasks=1 --nodelist=t3wn[80-91]',
        # '-w t3wn70,t3wn71,t3wn72,t3wn73', # only the best nodes
        '%s/submitter_chunk%d.sh' %(submitter_dir, jobid),
    ])

    print(command_sh_batch)
    os.system(command_sh_batch)
