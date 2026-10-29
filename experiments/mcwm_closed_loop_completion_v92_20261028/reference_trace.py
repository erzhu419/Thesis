"""Stream a factual parent and its reference-arm continuation without merging."""
from __future__ import annotations

import json
from pathlib import Path


class JoinedReference:
    """The readline/tell/close interface consumed by NativePrefixTrace.

    The continuation must be the unchanged reference arm. Shared rows are
    compared in full while the parent is emitted; only its unseen tail follows.
    ``stop_at`` is an exclusive issue-clock boundary, not a fabricated event.
    """
    def __init__(self, parent_path, reference_arm_path, *, stop_at=12900.):
        self.parent_path = Path(parent_path)
        self.reference_arm_path = Path(reference_arm_path)
        self.stop_at = float(stop_at)
        self._parent = self.parent_path.open(encoding='utf-8')
        self._arm = self.reference_arm_path.open(encoding='utf-8')
        self._phase = 'parent'
        self._last_output = None
        self._last_arm = None
        self._next_arm = None
        self._primed = False
        self._closed = self._exhausted = False
        self.parent_rows = self.overlap_rows = self.continuation_rows = 0

    def _read_arm(self):
        raw = self._arm.readline()
        if not raw:
            return None
        row = json.loads(raw)
        sequence = row['sequence']
        if self._last_arm is not None and sequence != self._last_arm+1:
            raise RuntimeError('Reference-arm trace skips or repeats an issue sequence')
        self._last_arm = sequence
        return raw, row

    def _emit(self, raw, row):
        expected = 0 if self._last_output is None else self._last_output+1
        if row['sequence'] != expected:
            raise RuntimeError('Joined factual trace skips or repeats an issue sequence')
        if row['started_at'] >= self.stop_at:
            self._exhausted = True
            return ''
        self._last_output = row['sequence']
        return raw

    def readline(self):
        if self._closed:
            raise ValueError('Read from a closed joined reference')
        if self._exhausted:
            return ''
        if not self._primed:
            self._next_arm = self._read_arm()
            self._primed = True
        if self._phase == 'parent':
            raw = self._parent.readline()
            if raw:
                row = json.loads(raw)
                if self._next_arm is None:
                    raise RuntimeError('Reference-arm trace ends inside the factual parent')
                _, arm_row = self._next_arm
                if arm_row['sequence'] < row['sequence']:
                    raise RuntimeError('Reference-arm overlap does not align with the factual parent')
                if arm_row['sequence'] == row['sequence']:
                    if arm_row != row:
                        raise RuntimeError('Factual parent and reference arm differ in their overlap')
                    self.overlap_rows += 1
                    self._next_arm = self._read_arm()
                result = self._emit(raw, row)
                if result:
                    self.parent_rows += 1
                return result
            if self.parent_rows == 0 or self.overlap_rows == 0:
                raise RuntimeError('Joining requires a nonempty verified reference-arm overlap')
            self._phase = 'continuation'
        if self._next_arm is None:
            raise RuntimeError('Reference-arm trace ends before the requested issue-clock boundary')
        raw, row = self._next_arm
        result = self._emit(raw, row)
        if result:
            self.continuation_rows += 1
            self._next_arm = self._read_arm()
        return result

    def tell(self):
        """A side-effect-free cookie for both source positions and join state."""
        if self._closed:
            raise ValueError('Tell on a closed joined reference')
        return (self._phase, self._parent.tell(), self._arm.tell(), self._last_output,
                self._last_arm, self._primed, self._exhausted,
                self.parent_rows, self.overlap_rows, self.continuation_rows)

    def close(self):
        # Fork children only close inherited readers. Closing must not advance
        # or seek a shared underlying file description owned by the parent.
        if not self._closed:
            self._parent.close()
            self._arm.close()
            self._closed = True

    def summary(self):
        return dict(parent_trace_path=str(self.parent_path),
            reference_arm_trace_path=str(self.reference_arm_path), stop_at=self.stop_at,
            parent_rows=self.parent_rows, overlap_rows=self.overlap_rows,
            continuation_rows=self.continuation_rows, last_sequence=self._last_output,
            boundary_reached=self._exhausted, merged_file_created=False)
