"""Functions shared by the result-processing kinds: formats, CSV, grouping and choice of the selected point."""
import csv
import pathlib
import re
import statistics
from dataclasses import dataclass, field

KV_SATURATION = 0.95


def format_number(x, decimals=1):
    """Format `x` with `decimals` decimals; an em dash if `x` is None or empty."""
    return "—" if x is None or x == "" else f"{float(x):.{decimals}f}"


def format_pct(x):
    """Format a 0-1 fraction as a whole percentage (0.5 -> "50%"); an em dash if None or empty."""
    return "—" if x is None or x == "" else f"{100 * float(x):.0f}%"


def strip_repetition(label):
    """prod-r2 -> prod"""
    return re.sub(r"-r\d+$", "", label)


def write_csv(path, rows):
    """Write `rows` (list of dicts) to the CSV file `path`; the keys of the first row give the header."""
    with open(pathlib.Path(path), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def median_of(rows, key):
    """Median of column `key` over `rows`; a None value counts as 0."""
    return statistics.median([row[key] or 0 for row in rows])


def group_rows(rows, config_of, level_of):
    """{(configuration, level): [rows]}"""
    groups = {}
    for row in rows:
        groups.setdefault((config_of(row), level_of(row)), []).append(row)
    return groups


def configurations(groups):
    """Sorted names of the configurations present in `groups` (keys are (configuration, level))."""
    return sorted({configuration for configuration, _ in groups})


@dataclass
class Choice:
    """Selected operating point of one configuration, as built by `choose`."""
    configuration: str
    levels: list                                    # all tested levels, in increasing order
    selected: object = None                            # highest acceptable level (None: none)
    throughput: float = 0.0                               # median output tokens/s per GPU at the selected level
    repetitions: int = 0
    low: float = 0.0
    high: float = 0.0
    rejected: list = field(default_factory=list)      # levels that were not accepted
    other_reason: set = field(default_factory=set)    # rejected for a reason other than interactivity
    without_metrics: bool = False                     # server metrics missing at some level
    rows: list = field(default_factory=list)       # rows of the selected level


def choose(groups, configuration, evaluate, prefer="level", missing=None):
    """Selected point of a configuration.

    evaluate(rows) -> (accepted, other_reason) for the repetitions of one level.
    prefer: "level" keeps the highest accepted level, "throughput" the accepted level with the highest throughput.
    missing(rows) -> bool flags levels without server metrics (accumulated over all levels)."""
    levels = sorted(level for (c, level) in groups if c == configuration)
    choice = Choice(configuration, levels)
    accepted_levels = []
    for level in levels:
        rows = groups[(configuration, level)]
        if missing is not None:
            choice.without_metrics |= missing(rows)
        is_accepted, other = evaluate(rows)
        if other:
            choice.other_reason.add(level)
        if is_accepted:
            throughputs = [row["output_throughput_per_gpu"] or 0 for row in rows]
            accepted_levels.append((statistics.median(throughputs), level, len(rows), min(throughputs), max(throughputs)))
        else:
            choice.rejected.append(level)
    if accepted_levels:
        key = (lambda a: (a[0], a[1])) if prefer == "throughput" else (lambda a: (a[1], a[0]))
        choice.throughput, choice.selected, choice.repetitions, choice.low, choice.high = max(accepted_levels, key=key)
        choice.rows = groups[(configuration, choice.selected)]
    return choice


def warn_rejected_lower_levels(choice, unit="level"):
    """Print a warning if levels below the selected one were rejected (non-monotonic
    curve); `unit` names the level in the message."""
    low = [str(c) for c in choice.rejected if c < choice.selected]
    if low:
        print(f"   WARNING: lower {unit}s ({', '.join(low)}) do not meet the target: "
              "non-monotonic curve, check these runs.")


def warn_missing_metrics(choice):
    """Print a warning if server metrics are missing at some level of `choice`."""
    if choice.without_metrics:
        print("   WARNING: server metrics missing for some levels: preemptions not checked.")


def warn_saturation(rows, unit="level"):
    """Print warnings if requests waited in the server queue, or if the KV cache reached KV_SATURATION,
    in any of `rows` (`unit` names the level in the messages)."""
    if any((row["max_waiting_requests"] or 0) > 0 for row in rows):
        print(f"   WARNING: requests waited in the server queue at this {unit}: "
              "TTFT includes waiting time.")
    if any((row["kv_cache_max_fraction"] or 0) >= KV_SATURATION for row in rows):
        print(f"   WARNING: KV cache is 95 % full or more at this {unit}: memory is almost saturated.")
