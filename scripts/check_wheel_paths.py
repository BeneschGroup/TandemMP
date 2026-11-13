#!/usr/bin/env python3
"""Fail if a built wheel / sdist embeds a personal home-directory path.

Rust release builds bake absolute source paths (``C:\\Users\\<name>\\.cargo\\...``)
into the compiled extension for panic messages. Anything built on a developer
machine therefore leaks that developer's username. Wheels built on CI runners
only contain generic runner paths, which are allowed here.

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

# Home-directory owners that are expected in CI / container builds.
ALLOWED_USERS = {"runner", "runneradmin", "root", "user", "build", "vsts"}

_NAME = rb"([A-Za-z0-9._~-]+)"
PATTERNS = [
    re.compile(rb"[A-Za-z]:[\\/]+Users[\\/]+" + _NAME, re.IGNORECASE),
    re.compile(rb"/Users/" + _NAME),
    re.compile(rb"/home/" + _NAME),
]


def _members(path: Path) -> Iterator[tuple[str, bytes]]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    yield info.filename, zf.read(info)
    elif path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path) as tf:
            for member in tf:
                if member.isfile():
                    fh = tf.extractfile(member)
                    if fh is not None:
                        yield member.name, fh.read()
    else:
        yield path.name, path.read_bytes()


def scan(path: Path) -> list[tuple[str, str]]:
    findings: set[tuple[str, str]] = set()
    for name, data in _members(path):
        for pattern in PATTERNS:
            for match in pattern.finditer(data):
                user = match.group(1).decode("latin-1")
                if user.lower() not in ALLOWED_USERS:
                    findings.add((name, user))
    return sorted(findings)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    failed = False
    for arg in argv:
        path = Path(arg)
        findings = scan(path)
        if findings:
            failed = True
            print(f"FAIL {path}")
            for name, user in findings:
                print(f"    {name}: embeds a path under user '{user}'")
        else:
            print(f"ok   {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
