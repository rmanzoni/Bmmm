#!/usr/bin/env python3
"""
hadd_uproot.py

uproot-based replacement for:

    hadd -f -k data.root /pnfs/psi.ch/cms/trivcat/store/user/manzoni/skims/ParkingDoubleMuonLowMass*/rjpsi_run3_23jun26_v2_ParkingDoubleMuonLowMass*_Run2024*_v*/*/*/*root

It merges every TTree found across all matched input files into a single
output file, streaming in chunks so memory use stays flat no matter how
many input files/events there are (hadd itself is not memory-safe for very
large merges of many small files, which is exactly the CRAB-skim situation
here: hundreds of PDM0-7 jobs x many small ROOT files).

--------------------------------------------------------------------------
Size reduction strategy (chosen, not incidental)
--------------------------------------------------------------------------
1. Output is written with ZSTD compression instead of hadd's default
   zlib. ZSTD at level ~9-12 typically beats zlib both on ratio *and*
   read-back speed for HEP ntuples.

2. Every float64 branch is downcast to float32 while streaming.
   Physics rationale: these are reconstructed kinematic/vertex/isolation
   quantities (masses, angles, IP, DCA, iso sums, ...). They come out of
   the fitters as float64 but the underlying detector/measurement
   resolution is nowhere near double precision (~1e-16 relative) -- it's
   more like 1e-3 to 1e-6 relative for the best-measured quantities (e.g.
   vertex positions in cm, IP significances). float32 (~7 significant
   digits, i.e. ~1e-7 relative) keeps far more precision than the
   detector resolution and roughly halves the on-disk size of every
   floating-point branch. Integer and boolean branches are left
   untouched: there's no size to save there (already minimal width) and
   no precision to trade away.

Net effect on a typical flat ntuple that is float-branch-dominated:
size roughly 2-3x smaller than a plain hadd, on top of the compression
gain from ZSTD itself.

--------------------------------------------------------------------------
Behavioural notes (mapping to the hadd flags used in the original command)
--------------------------------------------------------------------------
-f  (force overwrite)   -> output is always recreated. Use --no-force to
                            instead abort if the output file already exists.
-k  (keep going)         -> a file that can't even be opened is skipped
                            with a warning rather than aborting the whole
                            merge (see find_common_trees). Errors *while*
                            streaming a given tree are also caught per-tree
                            so one bad tree doesn't take down the others.
glob expansion           -> done in Python (glob.glob), not the shell, so
                            you can (and should) quote the pattern. This
                            also avoids ARG_MAX issues when a directory
                            glob expands to more files than a shell can
                            hand to hadd directly.

Usage
-----
    python3 hadd_uproot.py -o data.root \\
        "/pnfs/psi.ch/cms/trivcat/store/user/manzoni/skims/ParkingDoubleMuonLowMass*/rjpsi_run3_23jun26_v2_ParkingDoubleMuonLowMass*_Run2024*_v*/*/*/*root"

    # keep float64 as-is, just recompress (still smaller than plain hadd):
    python3 hadd_uproot.py -o data.root --no-downcast "<pattern>"

    # tune the chunk size (rows per iterate step) if you need to trade
    # memory for speed:
    python3 hadd_uproot.py -o data.root --step-size 100000 "<pattern>"

    # use 8 worker processes: files for each tree are split into 8 buckets,
    # each merged concurrently into a temp file, then consolidated into the
    # real output with one final fast serial pass. Also caps concurrently
    # open input files at 8 (never hundreds):
    python3 hadd_uproot.py -o data.root --workers 8 "<pattern>"

    # everything (worker temp parts AND the full merged file while it's
    # being built) is staged under /scratch/manzoni by default, and only
    # the finished file is moved into place next to -o at the very end --
    # so a space-constrained work area never sees anything in-progress.
    # Override the scratch location, or skip staging entirely, with:
    python3 hadd_uproot.py -o data.root --scratch-dir /scratch/manzoni "<pattern>"
    python3 hadd_uproot.py -o data.root --no-stage "<pattern>"
"""

import argparse
import concurrent.futures
import fnmatch
import glob
import multiprocessing
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import awkward as ak
import uproot


def human_size(nbytes):
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def human_time(seconds):
    if seconds != seconds or seconds is None:  # NaN check, no float('nan') import needed
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


class Progress:
    """Single-line progress bar spanning the whole merge (all trees combined),
    driven by exact entry counts collected up front (from TTree metadata --
    no branch data read, so this costs nothing extra beyond the existing
    per-file scan in find_common_trees).

    Reports elapsed time, ETA (from entries/sec throughput so far), and an
    estimated final output size (extrapolated from the output file's current
    on-disk size divided by the fraction of entries written so far -- this
    assumes a roughly uniform compression ratio across the dataset, which
    holds well for files sharing the same schema, as CRAB-job outputs do).

    Falls back to plain periodic lines (no carriage-return overwriting) when
    stderr isn't a terminal, so redirecting to a log file doesn't produce a
    mess of partial lines.
    """

    def __init__(self, total_entries, out_path, min_interval=0.5):
        self.total_entries = max(total_entries, 1)
        self.out_path = out_path
        self.is_tty = sys.stderr.isatty()
        self.min_interval = min_interval
        self.t_start = time.time()
        self._last_print = 0.0
        self.done = 0

    def update(self, n_rows):
        self.done += n_rows
        now = time.time()
        if now - self._last_print < self.min_interval and self.done < self.total_entries:
            return
        self._last_print = now
        self._render(now)

    def _render(self, now):
        elapsed = now - self.t_start
        frac = min(self.done / self.total_entries, 1.0)
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total_entries - self.done) / rate if rate > 0 else float("nan")

        try:
            out_size = os.path.getsize(self.out_path)
        except OSError:
            out_size = 0
        est_total = out_size / frac if frac > 0 else 0

        bar_width = 30
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)

        msg = (
            f"[{bar}] {frac * 100:5.1f}%  "
            f"{self.done}/{self.total_entries} entries  "
            f"elapsed {human_time(elapsed)}  "
            f"eta {human_time(eta)}  "
            f"out~{human_size(est_total)}"
        )
        if self.is_tty:
            print("\r" + msg + " " * 8, end="", file=sys.stderr, flush=True)
        else:
            print(msg, file=sys.stderr, flush=True)

    def close(self):
        self._render(time.time())
        if self.is_tty:
            print(file=sys.stderr)  # leave the finished bar on its own line


def find_common_trees(files):
    """Scan every input file once and return
    ({tree_name: [files containing it]}, {tree_name: total entries}, bad_files).

    Mirrors hadd's own behaviour of merging every top-level TTree it finds
    (not just one named tree), and mirrors -k by skipping unreadable files
    with a warning instead of raising. Entry counts come from TTree metadata
    only (no branch data is read), so this is effectively free and doubles
    as the exact entry total the progress bar needs.
    """
    trees = {}
    tree_entries = {}
    bad_files = []
    for fpath in files:
        try:
            with uproot.open(fpath) as f:
                for key, classname in f.classnames().items():
                    if classname.startswith("TTree"):
                        name = key.split(";")[0]
                        trees.setdefault(name, []).append(fpath)
                        try:
                            n = f[key].num_entries
                        except Exception:
                            n = 0
                        tree_entries[name] = tree_entries.get(name, 0) + n
        except Exception as exc:
            print(f"[warn] -k: skipping unreadable file {fpath}: {exc}", file=sys.stderr)
            bad_files.append(fpath)
    return trees, tree_entries, bad_files


def is_float64(array):
    """True if an awkward array's numeric leaf type is float64 (works for
    both flat and jagged/var-length branches)."""
    return "float64" in str(ak.type(array))


def downcast_chunk(chunk):
    """Downcast every float64 field in a {branch: array} dict to float32.
    Leaves int/bool/other branches untouched. Never raises: if a cast
    fails for some exotic branch type, that branch is passed through
    unchanged rather than aborting the merge.
    """
    out = {}
    for name, arr in chunk.items():
        a = arr if isinstance(arr, ak.Array) else ak.Array(arr)
        if is_float64(a):
            try:
                out[name] = ak.values_astype(a, np.float32)
                continue
            except Exception as exc:
                print(f"[warn] could not downcast branch '{name}': {exc}", file=sys.stderr)
        out[name] = arr
    return out


def branch_names_for_tree(tree_name, file_list):
    """Branch names for a tree, read once from its first file (all files
    contributing to the same tree are assumed to share a schema, same as
    hadd itself assumes)."""
    with uproot.open(file_list[0]) as f:
        return set(f[tree_name].keys())


def branch_type_spec(array):
    """Derive an explicit per-entry type spec (numpy dtype, or the .content
    of an Awkward ArrayType for jagged branches) for a single branch's array.

    This exists specifically to AVOID uproot's own mktree(name, data)
    auto-detection of "is this a type spec or actual data?", which has at
    least one uproot-version-dependent bug: on some installs it fails to
    recognize an Awkward array as such and leaves the raw array where a
    type was expected, raising "not a NumPy dtype or an Awkward datashape:
    <Array ...>". Computing the type ourselves, explicitly, sidesteps that
    ambiguity entirely regardless of which uproot release is installed.
    """
    if isinstance(array, np.ndarray):
        return array.dtype
    a = array if isinstance(array, ak.Array) else ak.Array(array)
    t = ak.type(a)  # e.g. "34 * int64" or "34 * var * float32" (ArrayType)
    return t.content  # per-entry type, without the outer array-length dimension


def branch_types_for_chunk(chunk):
    """{branch: type spec} for every branch in a chunk, via branch_type_spec."""
    return {name: branch_type_spec(arr) for name, arr in chunk.items()}


def make_drop_filter(drop_patterns):
    """Build a filter_name callable for uproot.iterate that excludes any
    branch matching one of drop_patterns (exact names or shell-style globs,
    e.g. 'mu3_jetdist_*'). Filtering here means dropped branches are never
    even read off disk -- saves I/O and time, not just output size.
    Returns None if there's nothing to drop (so iterate takes its default,
    slightly faster, path).
    """
    if not drop_patterns:
        return None

    def _filter(name):
        # uproot probes both the plain branch name and a "/name" path form,
        # and includes the branch if EITHER passes -- so the leading slash
        # must be stripped before matching, or exclusions are silently
        # ignored (the "/name" form always passes a naive != / fnmatch check).
        simple = name.lstrip("/")
        return not any(fnmatch.fnmatchcase(simple, pat) for pat in drop_patterns)

    return _filter


def merge_tree(out_file, tree_name, file_list, step_size, downcast, drop_filter=None, on_chunk=None):
    """Stream one tree across all files that contain it into out_file.

    on_chunk, if given, is called with the row count of every chunk written
    (on_chunk(n_rows)). It's a plain callable rather than a Progress object
    directly so the same function works whether it's running in the main
    process (on_chunk=progress.update) or inside a worker process, where
    on_chunk=queue.put reports back to the main process instead (a Progress
    object itself can't cross a process boundary meaningfully).
    """
    expressions = [f"{fpath}:{tree_name}" for fpath in file_list]
    n_entries_written = 0
    tree_created = tree_name in out_file

    for chunk in uproot.iterate(
        expressions, step_size=step_size, library="ak", how=dict,
        filter_name=drop_filter,
    ):
        if downcast:
            chunk = downcast_chunk(chunk)

        n_rows = len(next(iter(chunk.values())))
        if not tree_created:
            # Create the tree with explicit, self-computed types (see
            # branch_type_spec) rather than handing raw data to mktree and
            # letting uproot guess -- see NB in branch_type_spec's docstring.
            out_file.mktree(tree_name, branch_types_for_chunk(chunk))
            tree_created = True
        out_file[tree_name].extend(chunk)

        n_entries_written += n_rows
        if on_chunk is not None:
            on_chunk(n_rows)

    return n_entries_written


COMPRESSION_MAP = {
    "zstd": uproot.compression.ZSTD,
    "lz4": uproot.compression.LZ4,
    "lzma": uproot.compression.LZMA,
    "zlib": uproot.compression.ZLIB,
}


def build_compression(name, level):
    """Build an uproot compression object from a plain (name, level) pair.
    Kept as plain strings/ints (rather than passing compression objects or
    closures around) so this is trivially picklable across process
    boundaries for the multicore path.
    """
    if name == "none":
        return None
    return COMPRESSION_MAP[name](level=level)


def partition_round_robin(items, n_buckets):
    """Split items into up to n_buckets roughly-even lists, round-robin.
    Empty buckets are dropped (e.g. if there are fewer files than workers)."""
    buckets = [[] for _ in range(n_buckets)]
    for i, item in enumerate(items):
        buckets[i % n_buckets].append(item)
    return [b for b in buckets if b]


_worker_progress_queue = None  # set once per worker process via _init_worker


def _init_worker(progress_queue):
    """ProcessPoolExecutor initializer: runs once when each worker process
    starts. A raw multiprocessing.Queue can only be handed to a worker at
    process-creation time (via initializer/initargs, which uses the proper
    fork/spawn bootstrap path) -- NOT as a per-task argument to submit()
    (that goes through an ordinary pickling path and raises "Queue objects
    should only be shared between processes through inheritance"). Stashing
    it in a module-level global here is what makes it usable from
    _merge_bucket_worker below without receiving it as a submit() argument.
    """
    global _worker_progress_queue
    _worker_progress_queue = progress_queue


def _merge_bucket_worker(tree_name, bucket_files, tmp_path, step_size, downcast,
                          drop_patterns, tmp_compression_name, tmp_compression_level):
    """Runs in a worker process: merge one bucket of input files for one tree
    into its own small temp output file. Reconstructs the drop filter and
    compression object locally from plain, picklable arguments (patterns/
    strings/ints) rather than receiving closures or uproot objects directly.
    Reports progress via the queue stashed by _init_worker (see above).
    """
    drop_filter = make_drop_filter(drop_patterns)
    compression = build_compression(tmp_compression_name, tmp_compression_level)
    with uproot.recreate(tmp_path, compression=compression) as out_file:
        n = merge_tree(
            out_file, tree_name, bucket_files, step_size, downcast,
            drop_filter=drop_filter, on_chunk=_worker_progress_queue.put,
        )
    return n, tmp_path


def _drain_progress(progress_queue, progress):
    if progress is None:
        return
    while True:
        try:
            n_rows = progress_queue.get_nowait()
        except Exception:
            break
        progress.update(n_rows)


def merge_tree_multicore(tree_name, file_list, n_workers, step_size, downcast,
                          drop_patterns, tmp_compression_name, tmp_compression_level,
                          tmp_dir, output_basename, progress):
    """Parallel front-end to merge_tree: splits file_list into n_workers
    buckets, has a process pool merge each bucket into its own temp file
    concurrently (this is the genuinely multicore part -- decompression,
    downcasting, and recompression of each bucket happen on separate cores),
    then returns the list of temp file paths for the caller to consolidate
    with a single final (fast, serial) merge_tree pass.

    n_workers is an explicit, hard cap on how many input files are open at
    once (one per worker), matching the earlier "never open hundreds of
    files at once" requirement -- it's just no longer capped at exactly 1.
    """
    n_workers = max(1, min(n_workers, len(file_list)))
    buckets = partition_round_robin(file_list, n_workers)

    ctx = multiprocessing.get_context("fork")  # matches this analysis's existing
                                                # ProcessPoolExecutor convention
    # A plain ctx.Queue() (not multiprocessing.Manager().Queue()) -- Manager
    # queues route every put/get through an extra server process and are
    # noticeably slower; a raw Queue is picklable enough to hand to worker
    # processes directly under the same context, no manager needed.
    progress_queue = ctx.Queue()

    tmp_paths = []
    futures = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx,
        initializer=_init_worker, initargs=(progress_queue,),
    ) as pool:
        for i, bucket in enumerate(buckets):
            tmp_path = os.path.join(tmp_dir, f"{output_basename}.{tree_name}.part{i}.root")
            tmp_paths.append(tmp_path)
            fut = pool.submit(
                _merge_bucket_worker, tree_name, bucket, tmp_path, step_size, downcast,
                drop_patterns, tmp_compression_name, tmp_compression_level,
            )
            futures[fut] = tmp_path

        pending = set(futures)
        failed = []
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=0.2)
            _drain_progress(progress_queue, progress)
            for fut in done:
                tmp_path = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    # -k semantics at the bucket level: warn, drop that
                    # bucket's temp file from the final consolidation, keep
                    # the rest of the merge going.
                    print(f"[warn] -k: worker for '{tmp_path}' failed: {exc}", file=sys.stderr)
                    failed.append(tmp_path)
        _drain_progress(progress_queue, progress)

    tmp_paths = [p for p in tmp_paths if p not in failed and os.path.exists(p)]
    return tmp_paths





def main():
    parser = argparse.ArgumentParser(
        description="uproot-based hadd replacement with ZSTD compression and "
        "float64->float32 downcasting for smaller output files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pattern", help="glob pattern for input ROOT files (quote it!)")
    parser.add_argument("-o", "--output", required=True, help="output ROOT file")
    parser.add_argument(
        "--no-force", action="store_true",
        help="abort instead of overwriting an existing output file (hadd's -f is default here)",
    )
    parser.add_argument(
        "--no-downcast", action="store_true",
        help="keep float64 branches as-is (only recompress, no precision change)",
    )
    parser.add_argument(
        "--compression", default="zstd", choices=["zstd", "lz4", "lzma", "zlib", "none"],
        help="output compression algorithm (default: zstd)",
    )
    parser.add_argument(
        "--compression-level", type=int, default=9,
        help="compression level (default: 9; zstd max is 22, higher = smaller/slower)",
    )
    parser.add_argument(
        "--step-size", default="50 MB",
        help="uproot.iterate step size: an entry count (e.g. 50000) or a memory "
        "target string (e.g. '50 MB', default). Matches the 50k-row flush "
        "cadence already used elsewhere in this analysis's ntuplizer.",
    )
    parser.add_argument(
        "--drop", action="append", default=[], metavar="PATTERN",
        help="branch name or shell-style glob pattern to exclude from the output "
        "(e.g. --drop 'mu3_jetdist_*' --drop mu1_dxy). Can be given multiple "
        "times, and/or as a comma-separated list in one --drop. Matching "
        "branches are never read off disk in the first place, so this saves "
        "I/O and time as well as output size. Applies independently to each "
        "merged tree; a pattern that doesn't match any branch anywhere is "
        "reported as a warning (likely a typo).",
    )
    parser.add_argument(
        "--tree", action="append", default=[], metavar="NAME",
        help="only merge tree(s) with this exact name, skip every other tree "
        "found in the input files (can be given multiple times, and/or as a "
        "comma-separated list). Default: merge every tree found (like hadd). "
        "Use this for flat ntuple output (typically a single tree, e.g. "
        "--tree tree) to avoid wasting time on trees you don't want, and to "
        "get an immediate clear error if the glob pattern accidentally "
        "points at the wrong files (e.g. raw EDM skims instead of the "
        "ntuplizer output) rather than a wall of unrelated per-tree failures.",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="disable the live progress bar (just print tree-level start/end lines)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="number of worker processes to merge each tree's files with (default: 1, "
        "fully sequential, same behaviour as before). With N>1, files for a given "
        "tree are split into N buckets, each merged concurrently by its own process "
        "into a small temp file, then consolidated into the real output with one "
        "final fast serial pass. N is also a hard cap on concurrently-open input "
        "files (one per worker) -- e.g. --workers 8 opens at most 8 files at once, "
        "never hundreds.",
    )
    parser.add_argument(
        "--tmp-compression", default="none", choices=["zstd", "lz4", "lzma", "zlib", "none"],
        help="compression for the intermediate per-worker temp files (default: "
        "none, for fast writes -- they're immediately re-read and deleted by the "
        "final consolidation pass, which applies --compression once. Only worth "
        "changing if the scratch area itself is short on space).",
    )
    parser.add_argument(
        "--scratch-dir", default="/scratch/manzoni", metavar="DIR",
        help="stage everything here while working: per-worker temp part files, "
        "AND the full merged file while it's being built. Only the finished, "
        "complete file is moved into --output's directory at the very end -- "
        "the space-constrained work area never sees anything in-progress. "
        "Default: /scratch/manzoni. A unique subdirectory is created per run "
        "and removed when done (or on failure).",
    )
    parser.add_argument(
        "--no-stage", action="store_true",
        help="write directly to --output as it's built instead of staging in "
        "--scratch-dir and moving the finished file at the end (old behaviour; "
        "use this if scratch isn't available in some environment).",
    )
    args = parser.parse_args()

    drop_patterns = []
    for item in args.drop:
        drop_patterns.extend(p.strip() for p in item.split(",") if p.strip())

    tree_whitelist = []
    for item in args.tree:
        tree_whitelist.extend(t.strip() for t in item.split(",") if t.strip())

    launch_dir = os.getcwd()
    final_output_path = os.path.abspath(args.output)
    print(f"Launch directory: {launch_dir}")
    print(f"Final output will be: {final_output_path}")

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"[error] no files matched pattern: {args.pattern}", file=sys.stderr)
        sys.exit(1)
    print(f"Matched {len(files)} input files.")

    if os.path.exists(final_output_path) and args.no_force:
        print(f"[error] {final_output_path} already exists and --no-force was given", file=sys.stderr)
        sys.exit(1)

    trees, tree_entries, bad_files = find_common_trees(files)
    if not trees:
        print("[error] no TTrees found in any input file", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(trees)} tree(s) in the input files: {list(trees.keys())}")
    if bad_files:
        print(f"[warn] {len(bad_files)} file(s) were unreadable and skipped (see above)", file=sys.stderr)

    if tree_whitelist:
        missing = [t for t in tree_whitelist if t not in trees]
        if missing:
            print(
                f"[error] --tree requested {missing}, but only these trees exist "
                f"in the matched files: {list(trees.keys())}. Check that your glob "
                f"pattern points at the ntuplizer output and not e.g. raw EDM skims.",
                file=sys.stderr,
            )
            sys.exit(1)
        trees = {name: trees[name] for name in tree_whitelist}
        print(f"Restricting to requested tree(s): {list(trees.keys())}")

    if drop_patterns:
        all_branches = set()
        for tree_name, file_list in trees.items():
            all_branches |= branch_names_for_tree(tree_name, file_list)
        matched_any = set()
        for pat in drop_patterns:
            hits = {b for b in all_branches if fnmatch.fnmatchcase(b, pat)}
            if not hits:
                print(f"[warn] --drop pattern '{pat}' matched no branch in any tree (typo?)", file=sys.stderr)
            matched_any |= hits
        if matched_any:
            print(f"Dropping {len(matched_any)} branch(es): {sorted(matched_any)}")

    compression = build_compression(args.compression, args.compression_level)

    step_size = args.step_size
    try:
        step_size = int(step_size)
    except ValueError:
        pass  # keep as a memory-target string, e.g. "50 MB"

    total_entries_selected = sum(tree_entries.get(t, 0) for t in trees.keys())
    print(f"Total entries to merge: {total_entries_selected}")

    if args.no_stage:
        scratch_run_dir = None
        build_output_path = final_output_path
        tmp_dir = os.path.dirname(final_output_path) or "."
    else:
        try:
            os.makedirs(args.scratch_dir, exist_ok=True)
            scratch_run_dir = tempfile.mkdtemp(
                prefix=f"hadd_uproot_{os.path.basename(final_output_path)}_",
                dir=args.scratch_dir,
            )
        except OSError as exc:
            print(f"[error] could not create scratch staging dir under {args.scratch_dir!r}: {exc}", file=sys.stderr)
            sys.exit(1)
        build_output_path = os.path.join(scratch_run_dir, os.path.basename(final_output_path))
        tmp_dir = scratch_run_dir
        print(f"Staging in {scratch_run_dir} (moving to {final_output_path} when done)")
    output_basename = os.path.basename(build_output_path)

    t_start = time.time()
    total_rows = 0
    progress = None if args.no_progress else Progress(total_entries_selected, build_output_path)
    try:
        with uproot.recreate(build_output_path, compression=compression) as out_file:
            for tree_name, file_list in trees.items():
                print(f"Merging tree '{tree_name}' from {len(file_list)} file(s)...")
                try:
                    if args.workers > 1 and len(file_list) > 1:
                        n_workers_used = min(args.workers, len(file_list))
                        print(f"  using {n_workers_used} worker process(es)...")
                        tmp_paths = merge_tree_multicore(
                            tree_name, file_list, args.workers, step_size,
                            downcast=not args.no_downcast,
                            drop_patterns=drop_patterns,
                            tmp_compression_name=args.tmp_compression,
                            tmp_compression_level=args.compression_level,
                            tmp_dir=tmp_dir, output_basename=output_basename,
                            progress=progress,
                        )
                        try:
                            # Fast final serial pass: consolidate the (already
                            # downcast + filtered) temp parts into the real
                            # output. on_chunk=None here -- this data was already
                            # counted once by the workers above; counting it
                            # again here would double the progress bar's total.
                            n = merge_tree(
                                out_file, tree_name, tmp_paths, step_size,
                                downcast=False, drop_filter=None, on_chunk=None,
                            )
                        finally:
                            for p in tmp_paths:
                                try:
                                    os.remove(p)
                                except OSError:
                                    pass
                    else:
                        n = merge_tree(
                            out_file, tree_name, file_list, step_size,
                            downcast=not args.no_downcast,
                            drop_filter=make_drop_filter(drop_patterns),
                            on_chunk=None if progress is None else progress.update,
                        )
                    total_rows += n
                    if progress is not None and progress.is_tty:
                        print(file=sys.stderr)  # move off the live bar line before the next print
                    print(f"  -> {n} entries written")
                except Exception as exc:
                    # -k semantics: don't let one broken tree kill the others
                    print(f"[warn] -k: failed to merge tree '{tree_name}': {exc}", file=sys.stderr)
        if progress is not None:
            progress.close()

        if scratch_run_dir is not None:
            os.makedirs(os.path.dirname(final_output_path) or ".", exist_ok=True)
            shutil.move(build_output_path, final_output_path)
            print(f"Moved staged output to {final_output_path}")
    finally:
        if scratch_run_dir is not None:
            shutil.rmtree(scratch_run_dir, ignore_errors=True)

    in_size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
    out_size = os.path.getsize(final_output_path)
    dt = time.time() - t_start

    print()
    print(f"Done in {dt:.1f}s. Wrote {total_rows} total entries across {len(trees)} tree(s).")
    print(f"Input total:  {human_size(in_size)}  ({len(files)} files)")
    print(f"Output total: {human_size(out_size)}  ({final_output_path})")
    if in_size > 0:
        print(f"Reduction:    {100 * (1 - out_size / in_size):.1f}%")


if __name__ == "__main__":
    main()