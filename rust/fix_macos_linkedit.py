#!/usr/bin/env python3
"""Realign a macOS Mach-O ``__LINKEDIT`` built by Xcode 26's ``ld`` (ld-27106).

The linker can leave ``LC_SYMTAB.stroff`` at 4 mod 8 when the indirect
symbol table before it has an odd number of 4-byte entries; dyld then
refuses to load the extension with "mis-aligned LINKEDIT string pool".
Inserting four bytes before the string table fixes the pool but shifts the
code signature to 4 mod 16, so every linkedit blob after the string table
is realigned to its original alignment and the signature is regenerated.

Usage::

    python rust/fix_macos_linkedit.py path/to/normet_rust.abi3.so
    codesign --force --sign - path/to/normet_rust.abi3.so

The script exits without writing when nothing needs moving, so it is safe to
run on binaries from a healthy toolchain. This is a build workaround for the
local Apple toolchain, not part of the Python package: the extension is
opt-in and the default Python kernels are unaffected.
"""

from __future__ import annotations

import struct
import sys

LC_SEGMENT_64 = 0x19
LC_SYMTAB = 0x2
LC_DYLD_INFO = 0x22
LC_DYLD_INFO_ONLY = 0x80000022
LC_CODE_SIGNATURE = 0x1D
# function starts, data-in-code, exports trie, chained fixups
OTHER_LINKEDIT = (0x26, 0x29, 0x80000033, 0x80000034)


def pad_linkedit(path: str) -> int:
    """Pad the given Mach-O in place and return the number of bytes inserted."""
    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != 0xFEEDFACF:
        raise SystemExit(f"{path}: not a 64-bit little-endian Mach-O ({magic:#x})")
    ncmds = struct.unpack_from("<I", data, 16)[0]
    off = 32
    seg_linked = None
    symtab = None
    blobs: list[tuple[int, int, int, int]] = []  # (lc_off, field_off, dataoff, align)
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data, off)
        if cmd == LC_SEGMENT_64:
            if bytes(data[off + 8 : off + 24]).rstrip(b"\0") == b"__LINKEDIT":
                seg_linked = off
        elif cmd == LC_SYMTAB:
            symtab = (off,) + struct.unpack_from("<IIII", data, off + 8)
        elif cmd == LC_CODE_SIGNATURE:
            dataoff = struct.unpack_from("<I", data, off + 8)[0]
            blobs.append((off, 8, dataoff, 16))
        elif cmd in OTHER_LINKEDIT:
            dataoff = struct.unpack_from("<I", data, off + 8)[0]
            blobs.append((off, 8, dataoff, 8))
        elif cmd in (LC_DYLD_INFO, LC_DYLD_INFO_ONLY):
            vals = struct.unpack_from("<10I", data, off + 8)
            for k in range(0, 10, 2):
                if vals[k]:
                    blobs.append((off, 8 + 4 * k, vals[k], 8))
        off += cmdsize

    if seg_linked is None or symtab is None:
        raise SystemExit(f"{path}: missing __LINKEDIT or LC_SYMTAB")
    lc_sym = symtab[0]
    stroff = symtab[3]

    shift = 0
    pad = (-stroff) % 8
    if pad:
        data[stroff:stroff] = b"\0" * pad
        shift += pad
        struct.pack_into("<I", data, lc_sym + 16, stroff + shift)

    for lc_off, field_off, dataoff, align in sorted(blobs, key=lambda b: b[2]):
        if dataoff < stroff:
            continue
        cur = dataoff + shift
        pad = (-cur) % align
        if pad:
            data[cur:cur] = b"\0" * pad
            shift += pad
        struct.pack_into("<I", data, lc_off + field_off, dataoff + shift)

    if shift:
        filesize = struct.unpack_from("<Q", data, seg_linked + 48)[0]
        struct.pack_into("<Q", data, seg_linked + 48, filesize + shift)
        with open(path, "wb") as fh:
            fh.write(data)
    return shift


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: fix_macos_linkedit.py path/to/ext.so")
    moved = pad_linkedit(sys.argv[1])
    print(f"{sys.argv[1]}: inserted {moved} byte(s)")
