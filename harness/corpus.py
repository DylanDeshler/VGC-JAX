"""Trace corpus reading (07 §3): shards are concatenations of zstd frames, one per battle (oracle ShardWriter)."""
from __future__ import annotations

import pathlib
import struct

import orjson
import zstandard


def split_frames(buf: bytes):
    """Yield per-frame byte slices (RFC 8878 §3.1.1); mirrors oracle/lib/trace.js splitFrames."""
    pos = 0
    n = len(buf)
    while pos < n:
        start = pos
        magic = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        if (magic & 0xFFFFFFF0) == 0x184D2A50:
            pos += 4 + struct.unpack_from("<I", buf, pos)[0]
            continue
        if magic != 0xFD2FB528:
            raise ValueError(f"bad zstd magic at {start}")
        fhd = buf[pos]
        pos += 1
        fcs, single, checksum, did = fhd >> 6, (fhd >> 5) & 1, (fhd >> 2) & 1, fhd & 3
        if not single:
            pos += 1
        pos += (0, 1, 2, 4)[did]
        pos += ((1 if single else 0), 2, 4, 8)[fcs]
        while True:
            bh = buf[pos] | (buf[pos + 1] << 8) | (buf[pos + 2] << 16)
            pos += 3
            last, btype, size = bh & 1, (bh >> 1) & 3, bh >> 3
            pos += 1 if btype == 1 else size
            if last:
                break
        if checksum:
            pos += 4
        yield buf[start:pos]


def iter_battles(path):
    """Yield one battle (list of records) at a time from a shard."""
    d = zstandard.ZstdDecompressor()
    buf = pathlib.Path(path).read_bytes()
    for fr in split_frames(buf):
        text = d.decompress(fr, max_output_size=1 << 30)
        yield [orjson.loads(l) for l in text.splitlines() if l]


def shard_files(trace_dir):
    return sorted(pathlib.Path(trace_dir).glob("*.ndjson.zst"))
