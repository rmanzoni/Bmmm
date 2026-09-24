#!/usr/bin/env python3

import argparse
import re
import subprocess
from pathlib import Path
from collections import defaultdict


def get_year(dataset):
    """Extract Run2022, Run2023, ... from a dataset name."""
    match = re.search(r"Run(20\d{2})", dataset)
    if not match:
        raise ValueError(f"Could not determine year from dataset:\n{dataset}")
    return match.group(1)


def make_index_dataset(dataset, index):
    """Replace all occurrences of ParkingDoubleMuonLowMass0 with the requested index."""
    return dataset.replace(
        "ParkingDoubleMuonLowMass0",
        f"ParkingDoubleMuonLowMass{index}"
    )


def query_files(dataset, site=None, instance=None):
    """Run dasgoclient and return the files reported by DAS."""
    query = f"file dataset={dataset}"

    if instance:
        query += f" instance={instance}"

    if site:
        query += f" site={site}"

    cmd = [
        "dasgoclient",
        f"-query={query}",
    ]
    
    print(f"  Querying: {dataset}")
    print(f"  {cmd}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print("    ERROR running dasgoclient:")
        print(result.stderr.strip())
        return []

    files = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    print(f"    -> {len(files)} files")

    return files


def main():

    parser = argparse.ArgumentParser(
        description="Build year-grouped file lists from skimmed DAS datasets."
    )

    parser.add_argument(
        "input",
        help="Text file containing the index-0 datasets",
    )

    parser.add_argument(
        "--site",
        default=None,
        help="Optional DAS site restriction, e.g. T2_CH_CSCS",
    )

    parser.add_argument(
        "--instance",
        default=None,
        help="Use prod/phys03 for privately produced datasets",
    )

    parser.add_argument(
        "--date",
        default="24sep26",
        help="Date suffix for output files (default: 24sep26)",
    )

    parser.add_argument(
        "--output-prefix",
        default="files_data",
        help="Output file prefix (default: files_data)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Read datasets
    # ------------------------------------------------------------------

    with open(args.input) as f:
        datasets = [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]

    print(f"Found {len(datasets)} index-0 datasets\n")

    # ------------------------------------------------------------------
    # Store files grouped by year
    # ------------------------------------------------------------------

    files_by_year = defaultdict(list)

    # Keep track of datasets so we can print a useful summary
    datasets_by_year = defaultdict(list)

    for dataset0 in datasets:

        year = get_year(dataset0)
        datasets_by_year[year].append(dataset0)

        print("=" * 80)
        print(f"YEAR {year}")
        print(f"Base dataset: {dataset0}")

        # --------------------------------------------------------------
        # Loop over indices 0 ... 7
        # --------------------------------------------------------------

        for index in range(8):

            dataset = make_index_dataset(dataset0, index)

            files = query_files(
                dataset,
                site=args.site,
                instance=args.instance,
            )

            files_by_year[year].extend(files)

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Writing output files")
    print("=" * 80)

    for year in sorted(files_by_year):

        # Remove duplicates while preserving order
        unique_files = list(dict.fromkeys(files_by_year[year]))

        output = Path(
            f"{args.output_prefix}{year}_cscs_{args.date}.txt"
        )

        with output.open("w") as f:
            for filename in unique_files:
                f.write(filename + "\n")

        print(
            f"{output}: "
            f"{len(unique_files)} unique files "
            f"(from {len(files_by_year[year])} DAS entries)"
        )


if __name__ == "__main__":
    main()