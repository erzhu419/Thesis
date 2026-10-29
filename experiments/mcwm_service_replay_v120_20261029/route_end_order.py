"""Adjudicate only unordered, contiguous same-time route_end log records."""
from itertools import groupby, zip_longest
import json
from pathlib import Path


def _blocks(path, side, counts):
    def records():
        with Path(path).open() as handle:
            for line, text in enumerate(handle, 1):
                counts[side] += 1
                yield line, json.loads(text)

    def key(item):
        line, row = item
        return ("route_end", row["time"]) if row.get("event") == "route_end" else ("exact", line)

    for identity, items in groupby(records(), key):
        entries = list(items)
        yield dict(kind=identity[0], time=identity[1] if identity[0] == "route_end" else None,
                   line=entries[0][0], rows=[row for _, row in entries])


def compare_route_end_blocks(actual, reference):
    """Retain exact row order everywhere except each same-time route_end block."""
    counts, changed_blocks, changed_rows, first_failure = [0, 0], 0, 0, None
    streams = [_blocks(actual, 0, counts), _blocks(reference, 1, counts)]
    try:
        for left, right in zip_longest(*streams):
            reason = None
            if left is None or right is None:
                reason = "unmatched_block"
            elif left["kind"] != right["kind"]:
                reason = "block_kind_differs"
            elif left["line"] != right["line"] or len(left["rows"]) != len(right["rows"]):
                reason = "block_boundary_differs"
            elif left["kind"] == "route_end":
                if left["time"] != right["time"]:
                    reason = "route_end_time_differs"
                else:
                    ordered = [sorted(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                                  allow_nan=False) for row in block["rows"])
                               for block in (left, right)]
                    if ordered[0] != ordered[1]:
                        reason = "route_end_records_differ"
                    elif left["rows"] != right["rows"]:
                        changed_blocks += 1
                        changed_rows += sum(a != b for a, b in zip(left["rows"], right["rows"]))
            elif left["rows"] != right["rows"]:
                reason = "ordered_record_differs"
            if reason is not None and first_failure is None:
                first_failure = dict(reason=reason,
                    actual_line=None if left is None else left["line"],
                    reference_line=None if right is None else right["line"])
    except (OSError, ValueError, KeyError) as error:
        if first_failure is None:
            first_failure = dict(reason="read_error", error=type(error).__name__+": "+str(error))
    finally:
        for stream in streams:
            stream.close()
    return dict(all_pass=first_failure is None, rows=counts,
        changed_blocks=changed_blocks, changed_rows=changed_rows, first_failure=first_failure,
        semantics="Rows are [actual,reference]. Only contiguous equal-time route_end blocks "
                  "are compared as sorted complete JSON records, retaining multiplicity. "
                  "Block boundaries and all other records remain ordered and exact.")
