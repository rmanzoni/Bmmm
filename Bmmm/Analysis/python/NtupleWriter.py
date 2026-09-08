'''
Incremental (batched) ntuple output, shared by every channel.

Rows are buffered and flushed to the TTree every WRITE_EVERY, so the memory
footprint of a job is bounded by the buffer rather than growing with the output
file.

The schema is fixed and explicit: int64 for the event identifiers, float32 for
everything else. That distinction is the point of writing through uproot rather
than a TNtuple -- a TNtuple stores every column as float32, which silently
rounds any event number above 2^24, and 2018 event numbers are far past that.
Anything that de-duplicates or matches events needs run/lumi/event to survive
intact.

Deliberately numpy-only. This module used to build a pandas DataFrame per
flush, which pulled pandas -- and therefore numexpr and bottleneck -- into
every job. On a grid worker there is no ~/.local, and the pandas that CMSSW's
externals were compiled against need not be the one that gets imported; the
result was an ImportError between NumPy 1.x and 2.x before a single event was
read. Nothing here needs a DataFrame, so nothing here imports one.
'''

import numpy as np

WRITE_EVERY  = 50000
INT_BRANCHES = ('run', 'lumi', 'event')


def build_branch_types(branches):
    '''Fixed schema for the TTree: int64 for the event identifiers, float32 for
    everything else. Used by mktree so every extend() call matches it exactly.'''
    return {c: (np.int64 if c in INT_BRANCHES else np.float32) for c in branches}


def _as_float(value):
    '''Anything -> float, with anything uncoercible becoming NaN. Same contract
    as pandas.to_numeric(errors="coerce"), which this replaces.'''
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _as_int(value):
    '''Anything -> int64, with anything uncoercible (and NaN) becoming 0.

    int(value) is taken on the ORIGINAL object, not on a float of it, so an
    event number above 2^53 keeps every digit -- going through float first is
    the very rounding this module exists to avoid.'''
    if value is None:
        return 0
    try:
        if isinstance(value, float) and value != value:   # NaN
            return 0
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def rows_to_columns(rows, branches):
    '''list-of-dicts -> {branch: numpy array} with a FIXED dtype per branch, so
    every uproot extend() presents an identical schema. A branch absent from a
    row is filled the same way a missing value is: 0 for the identifiers, NaN
    for everything else.'''
    out = {}
    for col in branches:
        if col in INT_BRANCHES:
            out[col] = np.fromiter((_as_int(r.get(col)) for r in rows),
                                   dtype=np.int64, count=len(rows))
        else:
            out[col] = np.fromiter((_as_float(r.get(col)) for r in rows),
                                   dtype=np.float32, count=len(rows))
    return out


def flush(fout, row_list, branches):
    '''Append the buffered rows to the TTree and clear the buffer in place.'''
    if not row_list:
        return
    fout['tree'].extend(rows_to_columns(row_list, branches))
    row_list.clear()
