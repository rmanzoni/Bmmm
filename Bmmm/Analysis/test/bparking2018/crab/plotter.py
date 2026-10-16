#!/usr/bin/env python3

import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import uproot
from tqdm import tqdm


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

pattern = (
    "/pnfs/psi.ch/cms/trivcat/store/user/manzoni/"
    "jpsi_tk_ntuples_bparking2018_08oct2026/"
#     "Run2018*/ParkingBPH*/"
    "Run2018D/ParkingBPH1/"
    "jpsi_tk_ntuples_bparking2018_08oct2026_UL2018_MiniAODv2_v1/"
    "261008*/*/*root"
)

tree_name = "tree"
mass_branch = "rf_mass"
selection_branch = "sv_prob"

n_bins = 400
mass_min = 4.5
mass_max = 6.2
selection_threshold = 0.01

# Tune these according to PNFS performance and available resources.
n_workers = 16
step_size = "600 MB"

output_file = "rf_mass_sv_prob_gt_0p01.png"


# ------------------------------------------------------------------
# Process one ROOT file
# ------------------------------------------------------------------

bin_edges = np.linspace(mass_min, mass_max, n_bins + 1)


def process_file(filename):
    """Return histogram and event counters for one ROOT file."""

    hist = np.zeros(n_bins, dtype=np.int64)
    n_processed = 0
    n_selected = 0
    n_in_range = 0

    with uproot.open(filename) as root_file:
        tree = root_file[tree_name]

        for arrays in tree.iterate(
            expressions=[mass_branch, selection_branch],
            step_size=step_size,
            library="np",
        ):
            mass = arrays[mass_branch]
            sv_prob = arrays[selection_branch]

            n_processed += len(mass)

            # Apply the selection before histogramming.
            selected_mass = mass[sv_prob > selection_threshold]
            n_selected += len(selected_mass)

            n_in_range += np.count_nonzero(
                (selected_mass >= mass_min)
                & (selected_mass < mass_max)
            )

            hist += np.histogram(
                selected_mass, bins=bin_edges
            )[0]

    return filename, hist, n_processed, n_selected, n_in_range


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    files = sorted(glob.glob(pattern))
    
    files = files[:1500]

    if not files:
        print(f"No ROOT files found matching:\n{pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} ROOT files")
    print(f"Using {n_workers} workers; chunk size = {step_size}")

    total_hist = np.zeros(n_bins, dtype=np.int64)
    total_processed = 0
    total_selected = 0
    total_in_range = 0
    failures = []

    # Each worker processes a separate file. The progress bar advances
    # whenever an entire file has finished.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_file, filename): filename
            for filename in files
        }

        with tqdm(
            total=len(files),
            desc="Processing ROOT files",
            unit="file",
            dynamic_ncols=True,
        ) as progress:
            for future in as_completed(futures):
                filename = futures[future]

                try:
                    (
                        _,
                        hist,
                        n_processed,
                        n_selected,
                        n_in_range,
                    ) = future.result()

                    total_hist += hist
                    total_processed += n_processed
                    total_selected += n_selected
                    total_in_range += n_in_range

                except Exception as exc:
                    failures.append((filename, str(exc)))

                finally:
                    progress.update(1)
                    progress.set_postfix(
                        selected=f"{total_selected:,}",
                        failed=len(failures),
                    )

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.stairs(total_hist, bin_edges, linewidth=1.2)

    ax.set_xlabel(r"$m_{\mathrm{rf}}$ [GeV]")
    ax.set_ylabel("Events")
    ax.set_xlim(mass_min, mass_max)
    ax.set_title(r"$s_{\mathrm{vtx}} > 0.01$")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.show()

    # Summary
    print("\nSummary")
    print("-" * 45)
    print(f"Files matched:             {len(files):,}")
    print(f"Files processed:           {len(files) - len(failures):,}")
    print(f"Files failed:              {len(failures):,}")
    print(f"Events processed:          {total_processed:,}")
    print(f"Selected events:           {total_selected:,}")
    print(f"Selected events in range:  {total_in_range:,}")
    print(f"Histogram entries:         {total_hist.sum():,}")
    print(f"Plot saved to:              {output_file}")

    if failures:
        print("\nFailed files:")
        for filename, error in failures:
            print(f"  {filename}\n    {error}")


if __name__ == "__main__":
    main()
