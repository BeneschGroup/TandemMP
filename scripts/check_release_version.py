#!/usr/bin/env python3
"""Check a release tag against the versions of the built artefacts.

Fails if the tag (``v0.8.0``) does not match the version of every wheel / sdist
passed in, so ``v1.2.3`` can never ship a ``0.8.0.dev0`` wheel. Versions are
compared as PEP 440 versions, so ``v0.8.0-dev`` matches ``0.8.0.dev0``.

When ``$GITHUB_OUTPUT`` is set, writes ``prerelease=true|false`` to it so the
release step can mark ``.devN`` / ``aN`` / ``bN`` / ``rcN`` tags as pre-releases.

Usage:
    python scripts/check_release_version.py <tag> dist/*.whl dist/*.tar.gz
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version


def tag_version(tag: str) -> Version:
    """Parse a release tag such as ``v0.8.0`` (leading ``v`` optional)."""
    return Version(tag[1:] if tag.startswith(("v", "V")) else tag)


def artefact_version(path: Path) -> Version:
    """Version encoded in a wheel or sdist filename."""
    name = path.name
    if name.endswith(".whl"):
        return parse_wheel_filename(name)[1]
    if name.endswith(".tar.gz"):
        return parse_sdist_filename(name)[1]
    msg = f"not a wheel or sdist: {name}"
    raise ValueError(msg)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    tag, files = argv[0], [Path(a) for a in argv[1:]]
    try:
        expected = tag_version(tag)
    except InvalidVersion:
        print(
            f"ERROR tag '{tag}' is not a valid PEP 440 version (expected e.g. v0.8.0)"
        )
        return 1

    bad = False
    for path in files:
        try:
            found = artefact_version(path)
        except (InvalidWheelFilename, InvalidSdistFilename, ValueError) as exc:
            print(f"ERROR {path.name}: {exc}")
            bad = True
            continue
        if found != expected:
            print(f"MISMATCH {path.name}: version {found} != tag {tag} ({expected})")
            bad = True
        else:
            print(f"ok       {path.name}: {found}")
    if bad:
        return 1

    prerelease = expected.is_prerelease or expected.is_devrelease
    print(f"version {expected}; prerelease={str(prerelease).lower()}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"prerelease={str(prerelease).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
