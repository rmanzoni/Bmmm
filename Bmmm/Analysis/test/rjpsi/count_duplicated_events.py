#!/usr/bin/env python3

import uproot
import pandas as pd
from multiprocessing import Process
group_by = ["run", "lumi", "event", "pt", "jpsi_mass"]


# Open ROOT file and read the relevant branches
with uproot.open("data.root") as f:
    tree = f["tree"]

    df = tree.arrays(
        group_by,
        library="pd",
    )

# Count duplicate entries based on all four branches
duplicates = df.duplicated(
    subset=group_by,
    keep=False,
)

n_duplicate_rows = duplicates.sum()

# Number of duplicated groups
duplicate_groups = (
    df[duplicates]
    .groupby(group_by)
    .size()
)

n_duplicate_groups = len(duplicate_groups)

print(f"Total entries: {len(df)}")
print(f"Duplicated rows: {n_duplicate_rows}")
print(f"Duplicated groups: {n_duplicate_groups}")

if n_duplicate_groups > 0:
    print("\nTop duplicated entries:")
    print(
        duplicate_groups
        .sort_values(ascending=False)
#         .head(20)
    )