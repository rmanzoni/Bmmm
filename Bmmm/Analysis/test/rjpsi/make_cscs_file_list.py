#!/usr/bin/env python3
"""
Build year-grouped file lists from the skimmed (USER) datasets on DAS.

The input file lists the index-0 datasets (ParkingDoubleMuonLowMass0). For
each of them, the eight datasets ParkingDoubleMuonLowMass0..7 are looked up.

The trailing version in the dataset name (..._vN-<hash>/USER) can differ
between the eight indices, e.g.

  ParkingDoubleMuonLowMass0_Run2022G_22Sep2023_v1-<hash>/USER
  ParkingDoubleMuonLowMass4_Run2022G_22Sep2023_v2-<hash>/USER

so the version is replaced by a wildcard, DAS is asked for all VALID datasets
matching it, and the most recent one (highest N) that has files is used.

Complementary versions
----------------------
Some trailing versions are not re-issues of the same input but different
inputs covering different runs (e.g. PromptReco_v1 and PromptReco_v2). When
two such versions are both listed in the input file, each entry keeps its own
version: the other listed versions are excluded from its candidates.

Fail loud
---------
If any of the 8 indices has no usable dataset, the output files are NOT
written and the script exits with an error, unless --allow-missing is given.
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# ..._v<N>-<32-hex hash>/USER
VERSION_RE = re.compile(r"^(?P<prefix>.+)_v(?P<ver>\d+)(?P<suffix>-[0-9a-f]{32}/USER)$")


def get_year(dataset):
    """Extract Run2022, Run2023, ... from a dataset name."""
    match = re.search(r"Run(20\d{2})", dataset)
    if not match:
        raise ValueError(f"Could not determine year from dataset:\n{dataset}")
    return match.group(1)


def split_version(dataset):
    """Return (prefix, version, suffix) of a dataset name ending in _vN-<hash>/USER."""
    m = VERSION_RE.match(dataset)
    if not m:
        raise ValueError(
            "Dataset name does not end in _vN-<hash>/USER, cannot wildcard "
            f"the version:\n{dataset}"
        )
    return m.group("prefix"), int(m.group("ver")), m.group("suffix")


def make_index_dataset(dataset, index):
    """Replace all occurrences of ParkingDoubleMuonLowMass0 with the requested index."""
    return dataset.replace(
        "ParkingDoubleMuonLowMass0",
        f"ParkingDoubleMuonLowMass{index}"
    )


def run_das(query):
    """Run dasgoclient and return the non-empty output lines (None on error)."""
    cmd = ["dasgoclient", f"-query={query}"]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"    ERROR running {cmd}:")
        print("    " + result.stderr.strip())
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def query_datasets(pattern, instance=None):
    """VALID datasets matching a wildcard pattern (DAS default status is VALID)."""
    query = f"dataset dataset={pattern}"
    if instance:
        query += f" instance={instance}"
    return run_das(query)


def query_files(dataset, site=None, instance=None):
    """Files of one dataset."""
    query = f"file dataset={dataset}"
    if instance:
        query += f" instance={instance}"
    if site:
        query += f" site={site}"
    return run_das(query)


def resolve_index(prefix, suffix, base_version, excluded_versions, args):
    """
    Find the most recent valid version with files for one index.

    Returns (dataset, version, files, candidates_found) or
    (None, None, [], candidates_found) if nothing usable exists.
    """
    pattern = f"{prefix}_v*{suffix}"
    names = query_datasets(pattern, args.instance)
    if names is None:
        names = []

    # The DAS wildcard also matches e.g. "_v2_v1"; keep exact "_vN" only.
    strict = re.compile(re.escape(prefix) + r"_v(\d+)" + re.escape(suffix) + r"$")
    candidates = {}
    for name in names:
        m = strict.match(name)
        if m:
            candidates[int(m.group(1))] = name

    usable = sorted(
        (v for v in candidates if v not in excluded_versions),
        reverse=True,
    )

    for version in usable:
        dataset = candidates[version]
        files = query_files(dataset, site=args.site, instance=args.instance)
        n = len(files) if files is not None else 0
        print(f"      v{version}: {n} files")
        if n > 0:
            return dataset, version, files, candidates

    return None, None, [], candidates


def diagnose_missing(prefix, args):
    """Broader search (any version, any hash) to help understand a missing dataset."""
    pattern = f"{prefix}_v*-*/USER"
    names = query_datasets(pattern, args.instance) or []
    if names:
        print("      Broader search (any version, any hash) finds:")
        for name in names:
            print(f"        {name}")
    else:
        print("      Broader search (any version, any hash) finds nothing.")


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

    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write the file lists even if some datasets could not be found",
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

    # Versions listed in the input for the same prefix+hash are complementary
    # inputs (e.g. PromptReco_v1 / PromptReco_v2): keep them apart.
    listed_versions = defaultdict(set)
    parsed = []
    for dataset0 in datasets:
        prefix0, version0, suffix = split_version(dataset0)
        listed_versions[(prefix0, suffix)].add(version0)
        parsed.append((dataset0, prefix0, version0, suffix))

    # ------------------------------------------------------------------
    # Resolve every index of every base dataset
    # ------------------------------------------------------------------

    files_by_year = defaultdict(list)
    changed = []     # (base dataset, index, chosen dataset)
    missing = []     # (base dataset, index)
    ambiguous = []   # (base dataset, index, candidate versions, chosen version)

    n_total = len(parsed) * 8
    n_done = 0

    for dataset0, prefix0, version0, suffix in parsed:

        year = get_year(dataset0)
        excluded = listed_versions[(prefix0, suffix)] - {version0}

        print("=" * 80)
        print(f"YEAR {year}")
        print(f"Base dataset: {dataset0}")
        if excluded:
            print(f"  (versions {sorted(excluded)} are listed separately in the "
                  f"input and are excluded here)")

        for index in range(8):
            n_done += 1
            prefix = make_index_dataset(prefix0, index)
            print(f"  [{n_done}/{n_total}] index {index}: {prefix}_v*{suffix}")

            dataset, version, files, candidates = resolve_index(
                prefix, suffix, version0, excluded, args
            )

            if dataset is None:
                print(f"    !! NO VALID DATASET WITH FILES "
                      f"(valid versions found: {sorted(candidates) or 'none'})")
                diagnose_missing(prefix, args)
                missing.append((dataset0, index))
                continue

            print(f"    -> using v{version}: {len(files)} files")
            files_by_year[year].extend(files)

            if version != version0:
                changed.append((dataset0, index, dataset))
            usable = [v for v in candidates if v not in excluded]
            if len(usable) > 1:
                ambiguous.append((dataset0, index, sorted(usable), version))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nDatasets whose version differs from the index-0 one: {len(changed)}")
    for dataset0, index, dataset in changed:
        print(f"  index {index} of {dataset0}\n    -> {dataset}")

    print(f"\nIndices with more than one valid version (most recent used): {len(ambiguous)}")
    for dataset0, index, versions, chosen in ambiguous:
        print(f"  index {index} of {dataset0}: versions {versions}, used v{chosen}")

    print(f"\nIndices with no usable dataset: {len(missing)}")
    for dataset0, index in missing:
        print(f"  index {index} of {dataset0}")

    if missing and not args.allow_missing:
        print("\nERROR: some datasets are missing, file lists NOT written. "
              "Fix the input or rerun with --allow-missing.")
        sys.exit(1)

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

    if missing:
        print(f"\nWARNING: written with {len(missing)} missing datasets (--allow-missing).")


if __name__ == "__main__":
    main()