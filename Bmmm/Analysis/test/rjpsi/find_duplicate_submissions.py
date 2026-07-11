#!/usr/bin/env python3

import re
from pathlib import Path
from collections import defaultdict

submitter_dir = Path("RJpsi_23Jun2026_notrig_data2024_partial_v1")

# input file -> list of submitters containing it
occurrences = defaultdict(list)

submitter_pattern = "submitter_chunk*.sh"

for submitter in sorted(submitter_dir.glob(submitter_pattern)):

    with open(submitter) as f:
        content = f.read()

    # extract all --inputFiles=... occurrences
    matches = re.findall(r"--inputFiles=([^\s]+)", content)

    for infile in matches:
        occurrences[infile].append(submitter.name)

duplicates = {
    infile: submitters
    for infile, submitters in occurrences.items()
    if len(submitters) > 1
}

if not duplicates:
    print("No duplicated input files found.")
else:
    print(f"Found {len(duplicates)} duplicated input file(s):\n")

    for infile, submitters in sorted(duplicates.items()):
        print(infile)
        print(f"  appears in {len(submitters)} submitters:")
        for s in submitters:
            print(f"    {s}")
        print()
        
        
for infile, submitters in sorted(duplicates.items()):
    print(
        infile,
        " -> ",
        ", ".join(submitters),
    )