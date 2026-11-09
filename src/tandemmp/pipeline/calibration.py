"""Contrast-to-mass calibration loading/saving.

Two on-disk formats are supported: Refeyn's own ``.mc`` JSON (produced by
ExploreMP, see :func:`load_mc`) and TandemMP's own plain-JSON format (see
:func:`save_calibration`), which also transparently reads the legacy
pre-refactor ``alex_area.calibration`` pickled dict. :func:`load_calibration`
dispatches between the two by file suffix. Apply a loaded calibration
process-wide via :func:`tandemmp.core.conversions.set_calibration`, or use
:func:`resolve_calibration` to accept either a :class:`Calibration` or a path
in one entry point.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tandemmp.core.conversions import Calibration

CalibrationLike = Calibration | str | Path
"""A calibration, or a filepath to load one from (see :func:`resolve_calibration`)."""


@dataclasses.dataclass(frozen=True)
class CalibrationResult:
    """A fitted :class:`Calibration` plus the data that produced it."""

    calibration: Calibration
    source_filepath: str
    calibrant_masses: list[float]
    fitted_contrast_peaks: list[float]
    residual_sum_of_squares: float
    #: ``.mc``-only metadata (``None`` for TandemMP's own JSON format).
    name: str | None = None
    r2: float | None = None


# .mc (Refeyn ExploreMP)


def load_mc(filepath: str | Path) -> CalibrationResult:
    """Load a :class:`CalibrationResult` from a Refeyn ``.mc`` calibration file.

    ``.mc``'s convention is ``contrast = gradient * mass + offset``, which
    matches :class:`Calibration`'s ``mass = (c - offset) / slope`` with
    ``slope = gradient``.
    """
    path = Path(filepath)
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        slope = float(payload["gradient"])
        offset = float(payload["offset"])
    except KeyError as exc:
        msg = f"{path} is missing required .mc field {exc}"
        raise ValueError(msg) from exc
    return CalibrationResult(
        calibration=Calibration(slope=slope, offset=offset),
        source_filepath=str(path),
        calibrant_masses=[float(m) for m in payload.get("calibrated_values", [])],
        fitted_contrast_peaks=[float(c) for c in payload.get("contrasts", [])],
        residual_sum_of_squares=0.0,
        name=payload.get("name"),
        r2=float(payload["r2"]) if "r2" in payload else None,
    )


# TandemMP's own JSON format
#
# Field names mirror the pre-refactor ``alex_area.calibration`` pickled-dict
# format (``c2kDa``, ``intercept``, ``raw_error``, ``calibrant_masses``,
# ``fitted_contrast_peaks``), plus ``source_filepath`` for provenance.
# load_calibration() also accepts that legacy pickled dict directly, since
# the field names line up.


def _legacy_calibration_to_mapping(result: CalibrationResult) -> dict[str, Any]:
    return {
        "c2kDa": result.calibration.slope,
        "intercept": result.calibration.offset,
        "raw_error": result.residual_sum_of_squares,
        "calibrant_masses": result.calibrant_masses,
        "fitted_contrast_peaks": result.fitted_contrast_peaks,
        "source_filepath": result.source_filepath,
    }


def _legacy_calibration_from_mapping(
    payload: Mapping[str, Any], *, default_source_filepath: str
) -> CalibrationResult:
    try:
        slope = float(payload["c2kDa"])
        offset = float(payload["intercept"])
    except KeyError as exc:
        msg = f"calibration data is missing required field {exc}"
        raise ValueError(msg) from exc
    return CalibrationResult(
        calibration=Calibration(slope=slope, offset=offset),
        source_filepath=str(payload.get("source_filepath", default_source_filepath)),
        calibrant_masses=[float(m) for m in payload.get("calibrant_masses", [])],
        fitted_contrast_peaks=[
            float(p) for p in payload.get("fitted_contrast_peaks", [])
        ],
        residual_sum_of_squares=float(payload.get("raw_error", 0.0)),
    )


def save_calibration(filepath: str | Path, result: CalibrationResult) -> Path:
    """Write *result* to *filepath* as human-editable JSON.

    Creates parent directories as needed. See the module docstring for the
    on-disk field names.
    """
    if not isinstance(result, CalibrationResult):
        msg = f"result must be a CalibrationResult, got {type(result).__name__}"
        raise TypeError(msg)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_legacy_calibration_to_mapping(result), fh, indent=2)
        fh.write("\n")
    return path


def _load_legacy_calibration(filepath: str | Path) -> CalibrationResult:
    """Load TandemMP's own JSON format, or a legacy pickled dict."""
    path = Path(filepath)
    raw = path.read_bytes()
    payload: Any
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = pickle.loads(raw)  # noqa: S301 — legacy pickled-dict format

    if not isinstance(payload, Mapping):
        msg = f"{path} does not contain a calibration"
        raise TypeError(msg)
    return _legacy_calibration_from_mapping(payload, default_source_filepath=str(path))


# Generic dispatch


def load_calibration(filepath: str | Path) -> CalibrationResult:
    """Load a :class:`CalibrationResult` from *filepath*.

    Dispatches on suffix: ``.mc`` goes through :func:`load_mc`; anything
    else goes through TandemMP's own JSON format (or legacy pickle) via
    :func:`_load_legacy_calibration`.
    """
    path = Path(filepath)
    if path.suffix.lower() == ".mc":
        return load_mc(path)
    return _load_legacy_calibration(path)


def resolve_calibration(calibration: CalibrationLike | None) -> Calibration | None:
    """Resolve *calibration* to a :class:`Calibration`, loading it if it's a path."""
    if calibration is None or isinstance(calibration, Calibration):
        return calibration
    return load_calibration(calibration).calibration
