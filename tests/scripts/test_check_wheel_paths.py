"""Tests for ``scripts/check_wheel_paths.py`` (the personal-path leak guard)."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_wheel_paths.py"
_spec = importlib.util.spec_from_file_location("check_wheel_paths", _SCRIPT)
assert _spec is not None and _spec.loader is not None
cwp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cwp)

BS = "\\"


def make_wheel(
    tmp_path: Path, payload: bytes, name: str = "pkg-1.0-py3-none-any.whl"
) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("pkg/__init__.py", "x = 1\n")
        zf.writestr("pkg/ext.pyd", b"\x00\x01MZ" + payload + b"\x00rest")
    return path


def make_tar(tmp_path: Path, payload: bytes, name: str = "pkg-1.0.tar.gz") -> Path:
    path = tmp_path / name
    mode = "w:xz" if name.endswith(".xz") else "w:gz"
    with tarfile.open(path, mode) as tf:
        info = tarfile.TarInfo("pkg-1.0/ext.bin")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return path


# --- paths that must be flagged ---------------------------------------------


@pytest.mark.parametrize(
    ("payload", "user"),
    [
        (rf"C:{BS}Users{BS}alice{BS}.cargo{BS}registry{BS}x.rs".encode(), "alice"),
        (b"C:/Users/bob/.cargo/registry/x.rs", "bob"),
        (rf"c:{BS}users{BS}Carol{BS}x.rs".encode(), "Carol"),
        (rf"C:{BS}{BS}Users{BS}{BS}erin{BS}{BS}x.rs".encode(), "erin"),
        (b"/home/frank/work/x.rs", "frank"),
        (b"/Users/grace/src/x.rs", "grace"),
        # Non-ASCII (UTF-8) username.
        (f"C:{BS}Users{BS}Zo\u00eb{BS}.cargo{BS}x.rs".encode(), "Zo\u00eb"),
        # Windows usernames can contain spaces.
        (rf"C:{BS}Users{BS}John Smith{BS}.cargo{BS}x.rs".encode(), "John Smith"),
        # 8.3 short name.
        (rf"C:{BS}Users{BS}CHEM-B~1{BS}x.rs".encode(), "CHEM-B~1"),
    ],
)
def test_leaking_wheel_is_flagged(tmp_path: Path, payload: bytes, user: str) -> None:
    findings = cwp.scan(make_wheel(tmp_path, payload))
    assert [u for _, u in findings] == [user]
    assert (
        cwp.main([str(make_wheel(tmp_path, payload, "other-1.0-py3-none-any.whl"))])
        == 1
    )


@pytest.mark.parametrize("name", ["pkg-1.0.tar.gz", "pkg-1.0.tar.xz"])
def test_leaking_sdist_is_flagged(tmp_path: Path, name: str) -> None:
    sdist = make_tar(tmp_path, b"see /home/heidi/project/x.rs here", name)
    assert [u for _, u in cwp.scan(sdist)] == ["heidi"]


# --- paths that must be allowed ---------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        rf"C:{BS}Users{BS}runneradmin{BS}.cargo{BS}registry{BS}x.rs".encode(),
        rf"C:{BS}Users{BS}RunnerAdmin{BS}x.rs".encode(),  # case-insensitive
        b"/home/runner/work/repo/repo/x.rs",
        b"/root/.cargo/registry/x.rs",  # not a home-dir pattern at all
        b"/home/builder/x.rs",  # the placeholder the README tells people to use
        b"see /home/runner for details",  # prose: whitespace ends the name
        b"/rustc/ed61e7d7e242494fb7057f2657300d9e77bb4fcb/library/core/src/x.rs",
    ],
)
def test_clean_wheel_passes(tmp_path: Path, payload: bytes) -> None:
    wheel = make_wheel(tmp_path, payload)
    assert cwp.scan(wheel) == []
    assert cwp.main([str(wheel)]) == 0


def test_placeholders_removed_from_allowlist() -> None:
    # `user`, `build` and `vsts` used to be allowed; a real developer could have
    # those names, so they must now be flagged.
    assert not {"user", "build", "vsts"} & cwp.ALLOWED_USERS


@pytest.mark.parametrize("user", ["user", "build", "vsts"])
def test_formerly_allowed_names_are_flagged(tmp_path: Path, user: str) -> None:
    wheel = make_wheel(tmp_path, f"/home/{user}/x.rs".encode())
    assert [u for _, u in cwp.scan(wheel)] == [user]


# --- input handling -----------------------------------------------------------


def test_plain_binary_file_is_scanned_raw(tmp_path: Path) -> None:
    pyd = tmp_path / "ext.pyd"
    pyd.write_bytes(b"\x00" + f"C:{BS}Users{BS}ivan{BS}x.rs".encode() + b"\x00")
    assert [u for _, u in cwp.scan(pyd)] == ["ivan"]


@pytest.mark.parametrize("name", ["x.tar.zst", "x.7z", "x.rar", "x.bz2", "x.gz"])
def test_unsupported_compressed_formats_are_rejected(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"not really compressed")
    with pytest.raises(cwp.UnsupportedFormatError):
        cwp.scan(path)
    assert cwp.main([str(path)]) == 2


def test_main_returns_worst_status_over_all_files(tmp_path: Path) -> None:
    good = make_wheel(tmp_path, b"/home/runner/x", "a-1.0-py3-none-any.whl")
    bad = make_wheel(tmp_path, b"/home/judy/x", "b-1.0-py3-none-any.whl")
    assert cwp.main([str(good), str(bad)]) == 1
    assert cwp.main([]) == 2
