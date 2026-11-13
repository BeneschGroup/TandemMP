"""Tests for ``scripts/check_release_version.py`` (release tag vs built version)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_release_version.py"
_spec = importlib.util.spec_from_file_location("check_release_version", _SCRIPT)
assert _spec is not None and _spec.loader is not None
crv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crv)

WHEEL = "tandemmp-{v}-cp310-abi3-manylinux_2_28_x86_64.whl"
SDIST = "tandemmp-{v}.tar.gz"


def artefacts(version: str) -> list[str]:
    return [WHEEL.format(v=version), SDIST.format(v=version)]


def test_tag_version_strips_leading_v() -> None:
    assert str(crv.tag_version("v0.8.0")) == "0.8.0"
    assert str(crv.tag_version("0.8.0")) == "0.8.0"


def test_dev_tag_matches_normalised_dev_version() -> None:
    # pyproject says "0.8.0-dev"; wheels are named with the normalised form.
    assert crv.tag_version("v0.8.0-dev") == crv.tag_version("v0.8.0.dev0")
    assert crv.main(["v0.8.0-dev", *artefacts("0.8.0.dev0")]) == 0


def test_matching_release_passes() -> None:
    assert crv.main(["v0.8.0", *artefacts("0.8.0")]) == 0


def test_tag_that_does_not_match_the_build_is_rejected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The motivating case: tagging v1.2.3 must not ship a 0.8.0.dev0 wheel.
    assert crv.main(["v1.2.3", *artefacts("0.8.0.dev0")]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_final_tag_does_not_match_a_dev_build() -> None:
    assert crv.main(["v0.8.0", *artefacts("0.8.0.dev0")]) == 1


def test_one_bad_artefact_fails_the_run() -> None:
    files = [WHEEL.format(v="0.8.0"), SDIST.format(v="0.7.0")]
    assert crv.main(["v0.8.0", *files]) == 1


def test_invalid_tag_is_rejected() -> None:
    assert crv.main(["release-final", *artefacts("0.8.0")]) == 1


def test_non_artefact_is_rejected() -> None:
    assert crv.main(["v0.8.0", "tandemmp-0.8.0.zip"]) == 1


def test_usage_error() -> None:
    assert crv.main([]) == 2
    assert crv.main(["v0.8.0"]) == 2


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.8.0", "false"),
        ("1.0.0", "false"),
        ("0.8.0.dev0", "true"),
        ("1.0.0rc1", "true"),
        ("1.0.0a2", "true"),
        ("1.0.0b3", "true"),
    ],
)
def test_prerelease_flag_written_to_github_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str, expected: str
) -> None:
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert crv.main([f"v{version}", *artefacts(version)]) == 0
    assert out.read_text() == f"prerelease={expected}\n"


def test_nothing_written_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert crv.main(["v9.9.9", *artefacts("0.8.0")]) == 1
    assert not out.exists()
