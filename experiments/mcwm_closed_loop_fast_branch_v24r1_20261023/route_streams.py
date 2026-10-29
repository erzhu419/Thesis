"""Give forked SUMO route parsers independent kernel file offsets.

Parser buffers stay in the copied process memory. Reopening each route file at
its saved kernel offset lets those buffers refill without advancing the parent.
"""
from __future__ import annotations

import fcntl
import os
import stat


def capture_route_streams():
    streams = []
    for entry in os.listdir('/proc/self/fd'):
        fd = int(entry)
        try:
            path = os.readlink(f'/proc/self/fd/{fd}')
        except FileNotFoundError:
            # listdir's own descriptor has already closed.
            continue
        if (not path.endswith('.rou.xml') or not stat.S_ISREG(os.fstat(fd).st_mode)
                or fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY):
            continue
        streams.append(dict(fd=fd, path=path, offset=os.lseek(fd, 0, os.SEEK_CUR),
                            inheritable=os.get_inheritable(fd)))
    return sorted(streams, key=lambda stream: stream['fd'])


def route_stream_offsets(streams):
    return {str(stream['fd']): os.lseek(stream['fd'], 0, os.SEEK_CUR)
            for stream in streams}


def detach_route_streams(streams):
    """Call in a child before any native operation, preserving parser buffers."""
    for stream in streams:
        replacement = os.open(stream['path'], os.O_RDONLY)
        try:
            os.lseek(replacement, stream['offset'], os.SEEK_SET)
            os.dup2(replacement, stream['fd'], inheritable=stream['inheritable'])
        finally:
            os.close(replacement)
