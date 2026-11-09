"""Native (Rust/PyO3) extension modules for tandemmp.

Built automatically by ``pip install`` as a single extension
(``tandemmp._rust._native``) that registers each crate as a submodule, so
e.g. ``from tandemmp._rust import drift_correction`` keeps working. Drift
correction falls back to Python when the extension is unavailable; see
``tandemmp.drift.corrector``'s ``use_rust`` parameter.
"""

from __future__ import annotations

from tandemmp._rust._native import drift_correction

__all__ = ["drift_correction"]
