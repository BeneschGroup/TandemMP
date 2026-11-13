#!/usr/bin/env python3
"""Fail if a built wheel / sdist embeds a personal home-directory path.

Rust release builds bake absolute source paths (``C:\\Users\\<name>\\.cargo\\...``)
into the compiled extension for panic messages. Anything built on a developer
machine therefore leaks that developer's username. Wheels built on CI runners
only contain generic runner paths, which are allowed here.

Supported inputs: ``.whl`` / ``.zip``, ``.tar``, ``.tar.gz`` / ``.tgz`` /
``.tar.bz2`` / ``.tar.xz``, and plain files (scanned as raw bytes). Other
compressed formats are rejected, because scanning compressed bytes would pass
trivially.

Usage:
    python scripts/check_wheel_paths.py wheelhouse/*.whl dist/*.tar.gz
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

# Home-directory owners that are expected in CI / container builds, plus the
# placeholder the README tells people to remap their own home directory to.
ALLOWED_USERS = {"runner", "runneradmin", "root", "builder"}

# Characters that cannot appear in a path component (and NUL / control bytes).
_BAD = rb"\\/\x00-\x1f\"'<>|?*"

PATTERNS = [
    # Windows: C:\Users\<name>\...  The name may contain spaces or non-ASCII
    # bytes (UTF-8 in Rust binaries) and must be followed by a separator.
    re.compile(
        rb"[A-Za-z]:[\\/]+Users[\\/]+([^" + _BAD + rb"]+)(?=[\\/])", re.IGNORECASE
    ),
    # macOS / Linux: /Users/<name>, /home/<name>. No whitespace, so prose such
    # as "see /home/user for details" does not swallow following words.
    re.compile(rb"/Users/([^" + _BAD + rb"\s]+)"),
    re.compile(rb"/home/([^" + _BAD + rb"\s]+)"),
]

_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
_ZIP_SUFFIXES = (".whl", ".zip")
_UNSUPPORTED_SUFFIXES = (".gz", ".bz2", ".xz", ".zst", ".7z", ".rar", ".lz4")


class UnsupportedFormatError(ValueError):
    """The file is a compressed format this script cannot look inside."""


def _members(path: Path) -> Iterator[tuple[str, bytes]]:
    name = path.name.lower()
    if name.endswith(_ZIP_SUFFIXES):
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    yield info.filename, zf.read(info)
    elif name.endswith(_TAR_SUFFIXES):
        with tarfile.open(path, "r:*") as tf:
            for member in tf:
                if member.isfile():
                    fh = tf.extractfile(member)
                    if fh is not None:
                        yield member.name, fh.read()
    elif name.endswith(_UNSUPPORTED_SUFFIXES):
        raise UnsupportedFormatError(
            f"cannot scan {path.name}: unsupported compressed format "
            "(scanning it as raw bytes would pass trivially)"
        )
    else:
        yield path.name, path.read_bytes()


def scan(path: Path) -> list[tuple[str, str]]:
    """Return ``(member, username)`` for every disallowed home path found."""
    findings: set[tuple[str, str]] = set()
    for name, data in _members(path):
        for pattern in PATTERNS:
            for match in pattern.finditer(data):
                user = match.group(1).decode("utf-8", errors="replace")
                if user.lower() not in ALLOWED_USERS:
                    findings.add((name, user))
    return sorted(findings)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    status = 0
    for arg in argv:
        path = Path(arg)
        try:
            findings = scan(path)
        except UnsupportedFormatError as exc:
            print(f"ERROR {exc}")
            status = max(status, 2)
            continue
        if findings:
            status = max(status, 1)
            print(f"FAIL {path}")
            for name, user in findings:
                print(f"    {name}: embeds a path under user '{user}'")
        else:
            print(f"ok   {path}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
